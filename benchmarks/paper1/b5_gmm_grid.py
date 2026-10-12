"""B5: GMM Grid — Single-Center vs. Learnable vs. Fixed Grid (Novelty #5).

nfmap_paper_experiments.md B5. Doesn't fit run_ablation.py's single-param
sweep pattern (each condition varies two params together: n_prior_components
AND learnable_priors), so it gets its own small script.

Three conditions, dataset fixed at digits:
    single_gaussian : n_prior_components=1                        (standard NF baseline)
    learnable_gmm   : n_prior_components=100, learnable_priors=True (may collapse/drift)
    fixed_grid      : n_prior_components=100, learnable_priors=False (NFMap default)

NOTE: the paper doc also asks for "prior mean trajectory during training for
learnable priors (do they collapse?)" — that needs new instrumentation in
nfmap.py's fit() loop (snapshot module_.prior_means every N epochs when
learnable_priors=True), which doesn't exist yet. Not implemented here;
flagged as a follow-up, same category as B1's Jacobian-norm logging gap.

Three seeds per condition: the per-run spread on digits is large enough that a
single seed cannot separate "learnable priors collapse" from "this seed's run
collapsed", and the paper reports mean +/- sd.

Usage:
    python b5_gmm_grid.py
    python b5_gmm_grid.py --seeds 42
"""
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from common.data import load_dataset
from common.metrics import compute_dr_metrics, compute_grid_containment
from common.io import save_run

from nfmap import NFMap

BASE_DEFAULTS = dict(
    n_components=2,
    use_canonical=False,
    complexity="medium",
    inject_weight=0.5,
    prior_weight=0.1,
    # sigma_perp 10 (local end of the dial) for every dataset; grid_bounds is
    # per dataset (GRID below). These are what the paper's prior figure (f11)
    # and the prior table's Digits rows report.
    sigma_perp=10.0,
    graph_weight=5.0,
    repulsion_weight=1.0,
    prior_layout="hexagonal",
)

# Digits at 6: at 12 its map filled only the middle of the canvas.
GRID = {"digits": 6.0}

CONDITIONS = {
    "single_gaussian": dict(n_prior_components=1, learnable_priors=False),
    "learnable_gmm": dict(n_prior_components=100, learnable_priors=True),
    "fixed_grid": dict(n_prior_components=100, learnable_priors=False),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="digits", choices=sorted(GRID))
    ap.add_argument("--seeds", default="42,0,1")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    # Data seed held at 42 while random_state varies, as in B2C/B4C/B2D: the
    # question is run-to-run variance of the fit, not of the sample.
    data = load_dataset(args.dataset, seed=42)
    X, y, is_continuous = data["X"], data["y"], data["is_continuous"]
    print(f"[B5] {args.dataset}: {X.shape[0]} x {X.shape[1]}  "
          f"{len(CONDITIONS)} conditions x {len(seeds)} seeds")

    for seed in seeds:
        for cond, overrides in CONDITIONS.items():
            label = f"{cond}_seed{seed}"
            config = dict(BASE_DEFAULTS, grid_bounds=GRID[args.dataset])
            config.update(overrides)
            config["random_state"] = seed

            t0 = time.time()
            try:
                model = NFMap(**config)
                Z = model.fit_transform(X)
                prior_means = model.module_.prior_means.detach().cpu().numpy()
                metrics = compute_dr_metrics(X, Z, y, is_continuous=is_continuous)
                resolved_grid_bounds = getattr(model, "resolved_grid_bounds_", config.get("grid_bounds"))
                metrics.update(compute_grid_containment(Z, resolved_grid_bounds))
                ok = True
            except Exception as e:
                print(f"    {label}: FAILED ({e})")
                # NaN-filled, not None: see the matching comment in run_ablation.py —
                # None round-trips through np.savez/np.load as an object-dtype
                # array, which breaks np.isnan downstream in make_figures.py.
                Z = np.full((len(y), 2), np.nan)
                prior_means = None
                metrics = compute_dr_metrics(None, None, y, is_continuous=is_continuous)
                metrics.update(compute_grid_containment(None, config.get("grid_bounds")))
                ok = False
            elapsed = time.time() - t0

            run_config = dict(config)
            run_config.update({"dataset": args.dataset, "condition": cond,
                               "seed": seed, "ok": ok, "elapsed_s": elapsed})
            save_run("B5", args.dataset, label, run_config, metrics, Z, y=y, prior_means=prior_means)

            print(f"    {label}: T/C_F1={metrics['t_c_f1']:.3f} Sil={metrics['sil']:.3f} "
                  f"Acc={metrics['acc']:.3f} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
