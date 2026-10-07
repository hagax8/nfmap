"""B2E: the B2D joint sweep applied to the three 3D manifold datasets.

B2D swept sigma_perp and grid_bounds jointly on the six chemistry datasets and
found a single shared setting (sigma_perp=0.003, grid_bounds=12) that beats GTM
on all six, overturning a conclusion that had been an artefact of holding
grid_bounds at 6. The manifold datasets never got that treatment, and two
problems in the current paper numbers trace back to it:

1. The sigma_perp=0.05 used for these three in A1 comes from a single-seed A/B
   against the default 10.0 (swiss_roll DistCor 0.706->0.832, s_curve
   0.710->0.758, severed_sphere 0.762->0.771), not from a sweep. Those gains
   shrink sharply under seeding - B2B's seeded sweep puts swiss_roll at ~0.75
   for sigma_perp=12 and ~0.80 at 0.067 (+0.05, not +0.13), with s_curve and
   severed_sphere showing no reliable gain at all.

2. More seriously, B2B found 17-30% of points outside grid_bounds at EVERY
   sigma_perp on all three, and read that as a property of sigma_perp. B2D
   showed the same signature on chemistry was a property of the fixed canvas.
   So every manifold number in the paper currently comes from a configuration
   with a fifth to a third of the map off-canvas.

The expected outcome differs from B2D's, and the mechanism says why. These are
3D inputs: d_scaled=3 with n_components=2 leaves d_perp=1. The chemistry
result works by squeezing variance out of z_perp so it reappears in z_par, but
with a single off-manifold dimension there is almost no reservoir to squeeze -
consistent with B2B's flat-within-noise sigma_perp response here. So the
likely finding is that sigma_perp is inert in the low-d_perp regime and
grid_bounds alone governs containment. That is a perfectly reportable result,
and it is the one the paper needs in order to stop quoting 0.05 as if it were
tuned.

Selection is as in B2B/B2D: among configurations keeping <=5% of points inside
grid_bounds, take the largest sigma_perp within one standard deviation of the
best DistCor, since the argmax of a noisy curve is upward biased.

Usage:
    python b2e_sigma_grid_manifold.py
    python b2e_sigma_grid_manifold.py --datasets swiss_roll --seeds 42
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from common.data import load_dataset
from common.metrics import compute_dr_metrics, compute_grid_containment
from common.io import RESULTS_ROOT

from nfmap import NFMap
from run_ablation import DEFAULTS

EXPERIMENT_ID = "B2E"

DATASETS = ["swiss_roll", "s_curve", "severed_sphere"]
# Same ladder as B2D so the two sweeps are directly comparable, and wide
# enough to cover both the value A1 currently uses (0.05, between the 0.03 and
# 0.1 rungs) and B2D's chemistry pick (0.003).
SIGMAS = [0.001, 0.003, 0.01, 0.03, 0.1, 1.0, 5.0, 10.0]
# Pushed past B2D's 24: containment failed at 17-30% for every sigma_perp at
# grid_bounds=6 here, a far worse starting point than chemistry's 6.5%, so the
# canvas may need to go wider than chemistry's before these fit on it.
GRIDS = [6.0, 12.0, 24.0, 48.0]
CONTAINMENT_MAX = 0.05

# prior_layout=square for all three, matching A1: a visual check found the
# hexagonal grid gave each unrolled fragment a noisier, less monotonic colour
# gradient than square did on the same data.
PRIOR_LAYOUT = "square"

# Reference values from the A1 table (single-seed; the seeded A1 re-run is
# queued). Recorded here so the sweep output can be read against the baselines
# without cross-referencing a results directory that is being regenerated.
BASELINES = {
    "swiss_roll":     {"gtm": 0.615, "tsne": 0.693, "umap": 0.290},
    "s_curve":        {"gtm": 0.627, "tsne": 0.643, "umap": 0.618},
    "severed_sphere": {"gtm": 0.781, "tsne": 0.687, "umap": 0.641},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--seeds", default="42,0,1")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]

    out_dir = os.path.join(RESULTS_ROOT, EXPERIMENT_ID)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sigma_grid_manifold.json")
    rows = []
    print(f"[{EXPERIMENT_ID}] {len(datasets)} datasets x {len(SIGMAS)} sigma x "
          f"{len(GRIDS)} grid x {len(seeds)} seeds = "
          f"{len(datasets)*len(SIGMAS)*len(GRIDS)*len(seeds)} fits", flush=True)

    for ds in datasets:
        data = load_dataset(ds, seed=42)
        X, y, is_cont = data["X"], data["y"], data["is_continuous"]
        base = BASELINES.get(ds, {})
        print(f"\n[{EXPERIMENT_ID}] {ds}: {X.shape}   d_perp="
              f"{min(X.shape[1], X.shape[0]) - DEFAULTS['n_components']}   "
              f"baselines GTM={base.get('gtm', float('nan')):.3f} "
              f"t-SNE={base.get('tsne', float('nan')):.3f} "
              f"UMAP={base.get('umap', float('nan')):.3f}", flush=True)
        for sigma in SIGMAS:
            for gb in GRIDS:
                for seed in seeds:
                    cfg = {**DEFAULTS, "prior_layout": PRIOR_LAYOUT,
                           "sigma_perp": sigma, "grid_bounds": gb,
                           "random_state": seed}
                    try:
                        m = NFMap(**cfg)
                        Z = m.fit_transform(X)
                        met = compute_dr_metrics(X, Z, y, is_continuous=is_cont)
                        met.update(compute_grid_containment(
                            Z, getattr(m, "resolved_grid_bounds_", gb)))
                    except Exception as e:
                        print(f"    sigma={sigma} grid={gb} seed={seed}: FAILED ({e})",
                              flush=True)
                        continue
                    rows.append(dict(dataset=ds, sigma_perp=sigma, grid_bounds=gb,
                                     seed=seed,
                                     **{k: float(v) for k, v in met.items()}))
                    with open(out_path, "w") as f:
                        json.dump(rows, f, indent=2)

                cell = [r for r in rows if r["dataset"] == ds
                        and r["sigma_perp"] == sigma and r["grid_bounds"] == gb]
                if not cell:
                    continue
                print(f"    sigma={sigma:<6} grid={gb:<5} "
                      f"DistCor={np.mean([r['dist_corr'] for r in cell]):+.3f}"
                      f"±{np.std([r['dist_corr'] for r in cell]):.3f} "
                      f"T/C={np.mean([r['t_c_f1'] for r in cell]):.3f} "
                      f"out={100*np.mean([r['frac_outside_grid_bounds'] for r in cell]):.1f}%",
                      flush=True)

    print(f"\n  -> {out_path}")


if __name__ == "__main__":
    main()
