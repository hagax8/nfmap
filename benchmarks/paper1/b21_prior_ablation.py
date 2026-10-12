"""B21: does constraining the prior means to a fixed grid cost anything?

Replaces B5, which ran three prior conditions on DIGITS alone with three seeds
and produced differences (0.642 / 0.620 / 0.610) that sat inside the noise. That
is too thin for the claim it was being asked to carry: the fixed grid is the
main architectural difference from FloDR, which learns its mixture, and
"constraining the means costs nothing" is an acceptance of the null - which
needs more statistical power than a difference claim, not less.

Three conditions, all at K=100:

    single_gaussian - n_prior_components=1, i.e. the standard normalizing-flow
        prior. Included as a floor: it shows what the grid structure buys over
        no structure at all.
    learnable_gmm   - means learned, K fixed at 100.
    fixed_grid      - means pinned to the grid (NFMap's default).

The learned arm matches K rather than selecting it by held-out likelihood as
FloDR does. That is deliberate: matched K isolates the effect of fixing the
component POSITIONS, which is the thing being claimed, rather than confounding
it with model-size selection.

Operating points follow the rest of the paper rather than a single shared
setting. Chemistry is run at BOTH ends of the dial, because the two produce very
differently scaled maps and a fixed grid could plausibly cost something at one
end and not the other:
    chemistry  sigma_perp = 10 (local) and 0.003 (global), grid_bounds=12
    manifolds  the per-dataset sigma_perp maximising geodesic DistCor (B2F)

Reporting. A null result is stated as a bound, not an absence: for each
condition we report the paired per-seed difference from the fixed grid with a
95% confidence interval, so the claim is "any cost is smaller than X" rather
than "no significant difference found".

Usage:
    python b21_prior_ablation.py
    python b21_prior_ablation.py --datasets bace --seeds 42
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
# sklearn MUST be imported before torch on macOS: the reverse order segfaults
# silently (no traceback, no output) through a libomp symbol collision.
from common.data import load_dataset
from common.metrics import compute_dr_metrics, compute_grid_containment
from common.io import RESULTS_ROOT

# Fits run on CPU: an NFMap fit on the MPS backend is not reproducible across
# processes, and at these sizes CPU is also faster.
import torch as _torch
_torch.backends.mps.is_available = lambda: False

from nfmap import NFMap
from run_ablation import DEFAULTS

EXPERIMENT_ID = "B21"

CHEM = ["bace", "bbbp", "esol", "drd2", "egfr"]
MANIFOLD = ["swiss_roll", "s_curve", "severed_sphere"]
DATASETS = CHEM + MANIFOLD
SEEDS = [42, 0, 1, 2, 3]

CHEM_SIGMAS, GRID = [10.0, 0.003], 12.0
# Per-dataset (sigma_perp, grid_bounds) Table 1 reports for NFMap (seed-mean
# geodesic DistCor over B23), since ambient DistCor does not measure unrolling
# on these. Filled in main(). Was a hard-coded sigma from an MPS B2F run at
# grid 12; on CPU its swiss_roll pick (0.01) folds for every seed.
MANIFOLD_SIGMA = None
MANIFOLD_GRID = None


def _manifold_setting():
    from make_paper_tables import manifold_selection
    sel = {d: v["nfmap"] for d, v in manifold_selection().items() if "nfmap" in v}
    return ({d: p["sigma_perp"] for d, p in sel.items()},
            {d: p["grid_bounds"] for d, p in sel.items()})

CONDITIONS = {
    "single_gaussian": dict(n_prior_components=1, learnable_priors=False),
    "learnable_gmm":   dict(n_prior_components=100, learnable_priors=True),
    "fixed_grid":      dict(n_prior_components=100, learnable_priors=False),
}
METRICS = ["dist_corr", "t_c_f1", "acc"]


def paired_ci(diffs, alpha=0.05):
    """Mean paired difference and its 95% CI (t interval over seeds)."""
    from scipy import stats
    d = np.asarray([x for x in diffs if not np.isnan(x)])
    if len(d) < 2:
        return float("nan"), float("nan"), float("nan")
    m, se = d.mean(), stats.sem(d)
    h = se * stats.t.ppf(1 - alpha / 2, len(d) - 1)
    return float(m), float(m - h), float(m + h)


def main():
    global MANIFOLD_SIGMA, MANIFOLD_GRID
    MANIFOLD_SIGMA, MANIFOLD_GRID = _manifold_setting()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    # Lets dataset chunks run in parallel without overwriting each other.
    ap.add_argument("--out", default="prior_ablation.json")
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    out_dir = os.path.join(RESULTS_ROOT, EXPERIMENT_ID)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.out)
    rows = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            rows = json.load(f)
        print(f"[{EXPERIMENT_ID}] resuming with {len(rows)} existing runs", flush=True)
    # Manifold rows at a setting Table 1 no longer selects are dropped: kept,
    # they would be averaged and paired with the current ones.
    n0 = len(rows)
    rows = [r for r in rows if r["dataset"] not in MANIFOLD
            or (r["sigma_perp"], r.get("grid_bounds", GRID))
            == (MANIFOLD_SIGMA[r["dataset"]], MANIFOLD_GRID[r["dataset"]])]
    if len(rows) < n0:
        print(f"[{EXPERIMENT_ID}] dropped {n0 - len(rows)} manifold rows at "
              f"superseded settings", flush=True)
    # Grid is part of the key: a manifold's selected grid can change between
    # runs, and rows at the old grid must not count as done.
    key = lambda r: (r["dataset"], r["condition"], r["seed"], r["sigma_perp"],
                     r.get("grid_bounds", GRID))
    done = {key(r) for r in rows}

    n_fits = (sum(len(CHEM_SIGMAS) if d in CHEM else 1 for d in datasets)
              * len(CONDITIONS) * len(seeds))
    print(f"[{EXPERIMENT_ID}] {len(datasets)} datasets x {len(CONDITIONS)} priors "
          f"x {len(seeds)} seeds, chemistry at both ends of the dial "
          f"= {n_fits} fits", flush=True)

    for ds in datasets:
        d = load_dataset(ds, seed=42)
        X, y, is_cont = d["X"], d["y"], d["is_continuous"]
        sigmas = ([MANIFOLD_SIGMA[ds]] if ds in MANIFOLD else CHEM_SIGMAS)
        grid = MANIFOLD_GRID[ds] if ds in MANIFOLD else GRID
        print(f"\n[{EXPERIMENT_ID}] {ds}: {X.shape}  "
              f"sigma_perp={', '.join(f'{s:g}' for s in sigmas)}, "
              f"grid_bounds={grid:g}", flush=True)

        for sigma in sigmas:
            for cond, overrides in CONDITIONS.items():
                cell = []
                for seed in seeds:
                    k = (ds, cond, seed, sigma, grid)
                    if k in done:
                        cell.append(next(r for r in rows if key(r) == k))
                        continue
                    cfg = {**DEFAULTS, **overrides, "sigma_perp": sigma,
                           "grid_bounds": grid, "random_state": seed}
                    if ds in MANIFOLD:
                        cfg["prior_layout"] = "square"
                    try:
                        m = NFMap(**cfg)
                        Z = m.fit_transform(X)
                        met = compute_dr_metrics(X, Z, y, is_continuous=is_cont)
                        met.update(compute_grid_containment(
                            Z, getattr(m, "resolved_grid_bounds_", grid)))
                    except Exception as e:
                        print(f"    {cond} seed={seed}: FAILED ({e})", flush=True)
                        continue
                    rec = dict(dataset=ds, condition=cond, seed=seed,
                               sigma_perp=sigma, grid_bounds=grid,
                               **{k: float(v) for k, v in met.items()})
                    rows.append(rec); cell.append(rec)
                    with open(out_path, "w") as f:
                        json.dump(rows, f, indent=2)
                if cell:
                    print(f"    sig={sigma:<7g} {cond:<16} "
                          f"DistCor={np.mean([c['dist_corr'] for c in cell]):+.3f}"
                          f"±{np.std([c['dist_corr'] for c in cell]):.3f} "
                          f"T/C={np.mean([c['t_c_f1'] for c in cell]):.3f}", flush=True)

    # ── paired differences from the fixed grid, with CIs ────────────────────
    # Pairs are formed within (dataset, sigma_perp, seed): the two chemistry
    # operating points produce very differently scaled maps, so differencing
    # across them would compare unlike things.
    def report(label, subset):
        print(f"\n{label}")
        print(f"{'condition':<18}{'metric':<12}{'n':>5}{'mean diff':>11}{'95% CI':>22}")
        out = []
        for cond in ("single_gaussian", "learnable_gmm"):
            for met in METRICS:
                diffs = []
                for r in rows:
                    if r["condition"] != cond or (r["dataset"], r["sigma_perp"]) not in subset:
                        continue
                    base = next((q[met] for q in rows
                                 if q["condition"] == "fixed_grid"
                                 and (q["dataset"], q["sigma_perp"], q["seed"])
                                 == (r["dataset"], r["sigma_perp"], r["seed"])), None)
                    if base is not None and not np.isnan(r[met]) and not np.isnan(base):
                        diffs.append(r[met] - base)
                m, lo, hi = paired_ci(diffs)
                out.append(dict(group=label, condition=cond, metric=met,
                                n_pairs=len(diffs), mean_diff=m,
                                ci_low=lo, ci_high=hi))
                print(f"{cond:<18}{met:<12}{len(diffs):>5}{m:>+11.4f}"
                      f"   [{lo:+.4f}, {hi:+.4f}]")
        return out

    print(f"\n{'='*68}")
    print("Paired difference from fixed_grid (positive = other condition better)")
    summary = []
    summary += report("Chemistry, local end (sigma_perp=10)",
                      {(d, 10.0) for d in CHEM})
    summary += report("Chemistry, global end (sigma_perp=0.003)",
                      {(d, 0.003) for d in CHEM})
    summary += report("Manifolds (geodesic-selected sigma_perp)",
                      {(d, MANIFOLD_SIGMA[d]) for d in MANIFOLD})
    with open(os.path.join(out_dir, "paired_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  -> {out_path}")


if __name__ == "__main__":
    main()
