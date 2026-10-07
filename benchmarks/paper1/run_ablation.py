"""Generic single-hyperparameter NFMap ablation runner.

Covers B2 (sigma_perp), B3 (graph_weight), B4 (negative_sample_rate), and B7
(use_canonical) from nfmap_paper_experiments.md — all four are structurally
identical (fix a dataset, sweep one NFMap constructor argument, compute the
standard metric suite, persist each run) so they share one script instead of
four near-duplicates. See the paper1/Makefile for the exact invocation used
for each of those experiment IDs.

Usage:
    python run_ablation.py --experiment-id B3 --dataset digits \\
        --param graph_weight --values 0.0,1.0,5.0,10.0

Output: one results/<experiment-id>/<dataset>_<param>_<value>/ directory per
value, each with config.json + metrics.json + embeddings.npz.
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

# Shared Gaussian-prior defaults (Paper 1 scope only — no student_t, no hybrid
# kernel; those live in Paper 2). Diverges from benchmarks/nfmap_diagnostics.py's
# DEFAULTS in one place: grid_bounds=6.0 instead of 3.0, per B9's own finding
# that 3.0 leaves 16-60% of points outside the visible grid for ablations that
# aren't actually studying grid containment (B2/B3/B4) — a distracting,
# unrelated artifact in those figures. B5 and B9 deliberately keep/sweep 3.0
# themselves (grid containment IS what those two are about).
DEFAULTS = dict(
    n_components=2,
    use_canonical=False,
    complexity="medium",
    inject_weight=0.5,
    n_prior_components=100,
    prior_weight=0.1,
    grid_bounds=6.0,
    graph_weight=5.0,
    repulsion_weight=1.0,
    prior_layout="hexagonal",
)


def parse_value(raw):
    s = raw.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def slugify(value):
    return str(value).replace(" ", "").replace(".", "p").replace("-", "neg")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-id", required=True, help="e.g. B2, B3, B4, B7")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--param", required=True, help="NFMap constructor kwarg to sweep")
    ap.add_argument("--values", required=True, help="comma-separated values to sweep")
    ap.add_argument("--fixed", default="", help="comma-separated key=value overrides held "
                     "fixed alongside the sweep, e.g. use_canonical=true")
    ap.add_argument("--seeds", default="42,0,1",
                    help="comma-separated. Every reported number in the paper is a "
                         "mean over seeds: three separate findings in this document "
                         "(B2's high-end erosion, B4's monotonic trend, a dude "
                         "reversal in B4B) turned out to be single-seed noise.")
    args = ap.parse_args()

    values = [parse_value(v) for v in args.values.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    fixed_overrides = {}
    if args.fixed:
        for pair in args.fixed.split(","):
            k, v = pair.split("=")
            fixed_overrides[k] = parse_value(v)

    data = load_dataset(args.dataset, seed=seeds[0])
    X, y, is_continuous = data["X"], data["y"], data["is_continuous"]
    print(f"[{args.experiment_id}] {args.dataset}: {X.shape[0]} x {X.shape[1]}, "
          f"sweeping {args.param} over {values}" + (f", fixed={fixed_overrides}" if fixed_overrides else ""))

    for value in values:
        for seed in seeds:
            config = dict(DEFAULTS)
            config.update(fixed_overrides)
            config[args.param] = value
            config["random_state"] = seed
            # Seed belongs in the label: without it, extra seeds silently
            # overwrite each other and the sweep looks single-seed.
            method_label = f"{args.param}_{slugify(value)}_seed{seed}"

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
                print(f"    {method_label}: FAILED ({e})")
                # NaN-filled, not None: np.savez/np.load round-trips None as an
                # object-dtype array, which breaks np.isnan downstream in
                # make_figures.py.
                Z = np.full((len(y), 2), np.nan)
                prior_means = None
                metrics = compute_dr_metrics(None, None, y, is_continuous=is_continuous)
                metrics.update(compute_grid_containment(None, config.get("grid_bounds")))
                ok = False
            elapsed = time.time() - t0

            run_config = dict(config)
            run_config.update({"dataset": args.dataset, "param_swept": args.param,
                               "value": value, "seed": seed, "ok": ok, "elapsed_s": elapsed})
            save_run(args.experiment_id, args.dataset, method_label, run_config,
                     metrics, Z, y=y, prior_means=prior_means)

            print(f"    {method_label}: T/C_F1={metrics['t_c_f1']:.3f} "
                  f"DistCor={metrics['dist_corr']:.3f} DensCor={metrics['dens_corr']:.3f} "
                  f"Sil={metrics['sil']:.3f} Acc={metrics['acc']:.3f} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
