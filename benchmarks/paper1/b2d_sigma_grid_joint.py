"""B2D: sigma_perp and grid_bounds swept jointly - they are not independent.

Every earlier sweep varied one of these at a time and reached a wrong conclusion
because of it. B2/B2B concluded sigma_perp could not usefully go below ~0.01,
since lower values pushed points off the canvas (28% outside grid_bounds at
0.001 on DUD-E, 100% at 0.0001). B2B further found no rho satisfying containment
at ANY value for the three manifold datasets, and read that as a property of
sigma_perp.

Both were artefacts of holding grid_bounds fixed at 6.0. The mechanism says they
must be coupled: the prior factorises as p(z) = p_GMM(z_par) *
N(z_perp; 0, sigma^2 I), so lowering sigma squeezes z_perp, and because the flow
is bijective with a volume penalty that variance has to reappear in z_par. It
then needs somewhere to go. Widen the canvas and you can squeeze harder.

A single-seed probe on two datasets bears this out - and overturns the headline
chemistry comparison:

                        bbbp DistCor   drd2 DistCor   (GTM: 0.628 / 0.394)
    sigma=0.01, grid=6      0.466          0.276
    sigma=0.003, grid=12    0.679          0.583
    sigma=0.001, grid=24    0.664          0.591

i.e. NFMap goes from clearly behind GTM to ahead of it, and containment on drd2
*improves* (6.5% -> 2.7%) rather than degrading, because the wider grid absorbs
the extra spread.

This sweeps the two jointly across all six chemistry datasets with three seeds,
to establish whether that holds up before it rewrites Section 6. Selection, as
in B2B, is constrained: among configurations keeping <=5% of points inside
grid_bounds, take the largest sigma_perp within one standard deviation of the
best DistCor, since the argmax of a noisy curve is upward biased.

Usage:
    python b2d_sigma_grid_joint.py
    python b2d_sigma_grid_joint.py --datasets bbbp --seeds 42
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from common.data import load_dataset
from common.metrics import compute_dr_metrics, compute_grid_containment
from common.io import RESULTS_ROOT

from nfmap import NFMap
from run_ablation import DEFAULTS

EXPERIMENT_ID = "B2D"

DATASETS = ["bace", "bbbp", "dude", "drd2", "egfr", "esol"]
# Spans the full dial, from the near-razor-thin regime up to and including
# NFMap's shipped default of 10.0 - the high end matters here because the
# sigma/grid interaction may look different when z_perp is roomy, and
# because Section 6 is reported at the default.
SIGMAS = [0.001, 0.003, 0.01, 0.03, 0.1, 1.0, 5.0, 10.0]
GRIDS = [6.0, 12.0, 24.0]
CONTAINMENT_MAX = 0.05

# For reference in the write-up; from A1/A2 with GTM matched to 100 nodes.
GTM_DISTCOR = {"bace": 0.428, "bbbp": 0.628, "dude": 0.428,
               "drd2": 0.394, "egfr": 0.446, "esol": 0.689}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--seeds", default="42,0,1")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]

    out_dir = os.path.join(RESULTS_ROOT, EXPERIMENT_ID)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sigma_grid_joint.json")
    rows = []
    print(f"[{EXPERIMENT_ID}] {len(datasets)} datasets x {len(SIGMAS)} sigma x "
          f"{len(GRIDS)} grid x {len(seeds)} seeds = "
          f"{len(datasets)*len(SIGMAS)*len(GRIDS)*len(seeds)} fits", flush=True)

    for ds in datasets:
        data = load_dataset(ds, seed=42)
        X, y = data["X"], data["y"]
        print(f"\n[{EXPERIMENT_ID}] {ds}: {X.shape}   GTM DistCor = "
              f"{GTM_DISTCOR.get(ds, float('nan')):.3f}", flush=True)
        for sigma in SIGMAS:
            for gb in GRIDS:
                for seed in seeds:
                    cfg = {**DEFAULTS, "sigma_perp": sigma, "grid_bounds": gb,
                           "random_state": seed}
                    t0 = time.time()
                    try:
                        m = NFMap(**cfg)
                        Z = m.fit_transform(X)
                        met = compute_dr_metrics(X, Z, y, is_continuous=False)
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
                print(f"    sigma={sigma:<6} grid={gb:<5} "
                      f"DistCor={np.mean([r['dist_corr'] for r in rows if r['dataset']==ds and r['sigma_perp']==sigma and r['grid_bounds']==gb]):+.3f} "
                      f"T/C={np.mean([r['t_c_f1'] for r in rows if r['dataset']==ds and r['sigma_perp']==sigma and r['grid_bounds']==gb]):.3f} "
                      f"out={100*np.mean([r['frac_outside_grid_bounds'] for r in rows if r['dataset']==ds and r['sigma_perp']==sigma and r['grid_bounds']==gb]):.1f}% "
                      f"({time.time()-t0:.0f}s/seed)", flush=True)

    print(f"\n  -> {out_path}")


if __name__ == "__main__":
    main()
