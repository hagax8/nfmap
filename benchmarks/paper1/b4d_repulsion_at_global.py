"""B4D: does repulsion_weight's contained optimum move at the new operating point?

B4 and B4C sweep repulsion_weight at NFMap's shipped defaults (sigma_perp=10,
grid_bounds=6). B2D then moved the chemistry operating point to sigma_perp=0.003,
grid_bounds=12. This checks whether repulsion's optimum survives that move.

The reason it might not is that all three parameters act on ONE quantity - the
spread of z_par relative to the canvas:

    sigma_perp down       squeezes z_perp, so variance reappears in z_par
    repulsion_weight up   pushes points apart in z_par directly
    grid_bounds up        enlarges the canvas that has to hold the result

and all three move frac_outside_grid_bounds. B4's seeded sweep on digits shows
repulsion's global-for-local trade arriving together with the containment cost,
exactly as sigma_perp's does:

    w     DistCor  T/C F1    Sil    outside
    0.0     0.606   0.966  0.499       0.0%
    1.0     0.613   0.975  0.389       2.7%
    2.0     0.632   0.973  0.348       5.4%
    10.0    0.638   0.914  0.135      27.8%

Under the <=5% containment constraint that leaves w=1.0, the shipped default -
so at the OLD corner the sweep confirms the default rather than challenging it.
Two things could change at the new one, in opposite directions:

  * grid_bounds=12 doubles the canvas, so a given w costs less containment.
    Repulsion might now afford to go higher than 1.0.
  * sigma_perp=0.003 has already spent much of the canvas on transferred
    z_perp variance, so there may be less headroom left, not more.

Which effect dominates is not predictable from the mechanism, hence this run.
It is deliberately a single slice (one sigma_perp, one grid_bounds, four
weights), not a 3D sweep: three parameters moving one quantity would trace a
degenerate ridge, and sweeping them jointly would cost thousands of fits to
re-discover a relationship the mechanism already predicts.

Selection is as in B2B/B2D/B2E: among configurations keeping <=5% of points
inside grid_bounds, take the value within one standard deviation of the best
DistCor, since the argmax of a noisy curve is upward biased.

Usage:
    python b4d_repulsion_at_global.py
    python b4d_repulsion_at_global.py --datasets bbbp --seeds 42
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

EXPERIMENT_ID = "B4D"

DATASETS = ["bace", "bbbp", "dude", "drd2", "egfr", "esol"]
WEIGHTS = [0.0, 1.0, 2.0, 5.0]

# B2D's shared chemistry setting: the best cell keeping <=5% of points inside
# grid_bounds, ahead of GTM on all six datasets (mean DistCor 0.653 vs GTM's
# mean 0.502). Held fixed here so repulsion_weight is the only variable.
SIGMA_PERP = 0.003
GRID_BOUNDS = 12.0
CONTAINMENT_MAX = 0.05

# For reference: B4's contained optimum at the OLD corner (sigma_perp=10,
# grid_bounds=6) was w=1.0, which is also NFMap's shipped default.
OLD_OPTIMUM = 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--seeds", default="42,0,1")
    ap.add_argument("--weights", default=",".join(str(w) for w in WEIGHTS))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]
    weights = [float(w) for w in args.weights.split(",")]

    out_dir = os.path.join(RESULTS_ROOT, EXPERIMENT_ID)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "repulsion_at_global.json")
    rows = []
    print(f"[{EXPERIMENT_ID}] sigma_perp={SIGMA_PERP}, grid_bounds={GRID_BOUNDS}  "
          f"{len(datasets)} datasets x {len(weights)} weights x {len(seeds)} seeds "
          f"= {len(datasets)*len(weights)*len(seeds)} fits", flush=True)

    for ds in datasets:
        data = load_dataset(ds, seed=42)
        X, y = data["X"], data["y"]
        print(f"\n[{EXPERIMENT_ID}] {ds}: {X.shape}", flush=True)
        for w in weights:
            for seed in seeds:
                cfg = {**DEFAULTS, "sigma_perp": SIGMA_PERP,
                       "grid_bounds": GRID_BOUNDS, "repulsion_weight": w,
                       "random_state": seed}
                try:
                    m = NFMap(**cfg)
                    Z = m.fit_transform(X)
                    met = compute_dr_metrics(X, Z, y, is_continuous=False)
                    met.update(compute_grid_containment(
                        Z, getattr(m, "resolved_grid_bounds_", GRID_BOUNDS)))
                except Exception as e:
                    print(f"    w={w} seed={seed}: FAILED ({e})", flush=True)
                    continue
                rows.append(dict(dataset=ds, repulsion_weight=w, seed=seed,
                                 sigma_perp=SIGMA_PERP, grid_bounds=GRID_BOUNDS,
                                 **{k: float(v) for k, v in met.items()}))
                with open(out_path, "w") as f:
                    json.dump(rows, f, indent=2)

            cell = [r for r in rows if r["dataset"] == ds and r["repulsion_weight"] == w]
            if not cell:
                continue
            out = np.mean([r["frac_outside_grid_bounds"] for r in cell])
            print(f"    w={w:<5} DistCor={np.mean([r['dist_corr'] for r in cell]):+.3f}"
                  f"±{np.std([r['dist_corr'] for r in cell]):.3f} "
                  f"T/C={np.mean([r['t_c_f1'] for r in cell]):.3f} "
                  f"Sil={np.mean([r['sil'] for r in cell]):.3f} "
                  f"out={100*out:.1f}%{'  (fails containment)' if out > CONTAINMENT_MAX else ''}",
                  flush=True)

    print(f"\n  -> {out_path}")


if __name__ == "__main__":
    main()
