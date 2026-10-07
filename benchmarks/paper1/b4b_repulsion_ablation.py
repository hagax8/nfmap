"""B4B: does repulsion help or hurt? Repulsion on vs. off, every dataset, 3 seeds.

Why this exists separately from B4. B4 sweeps `negative_sample_rate` over
{0,1,2,5,10} on digits alone. Two things made that insufficient:

1. `negative_sample_rate=0` crashed NFMap outright until 2026-10-01 (the repulsion
   block called torch.stack on an empty list), so B4 never actually measured the
   no-repulsion baseline - it only compared amounts of repulsion to each other,
   and those turned out to sit within run-to-run noise. The claim B4 exists to
   test ("more negative sampling sharpens cluster boundaries") was therefore
   never tested against its own control.
2. Once the crash was fixed, digits showed repulsion-off beating the default by
   +0.120 silhouette across 4 seeds (sd 0.004-0.008) - a large, reproducible
   effect in the *opposite* direction to the claim. But a single-seed check on
   dude pointed the other way, so the effect is plausibly dataset-dependent and
   cannot be settled on one dataset.

Design: binary contrast (off vs. the default rate=5) across breadth of datasets
rather than a deep sweep on one, because B4 already established that the amount
of repulsion barely matters while its presence/absence may matter a lot. Three
seeds per cell, because today's two biggest corrections in this paper (B2's
apparent high-end erosion, B4's apparent monotonicity) were both single-seed
noise read as trend.

Silhouette is the headline metric - it is what "sharpens cluster boundaries"
means - so the 9 labelled datasets carry the claim; the 3 continuous-target
manifolds (swiss_roll/s_curve/severed_sphere) have no silhouette and are
reported on T/C F1 / DistCor / DensCor only.

Usage:
    python b4b_repulsion_ablation.py                 # all 12 datasets, 3 seeds
    python b4b_repulsion_ablation.py --datasets digits,dude --seeds 42

Output: results/B4B/<dataset>_rate<r>_seed<s>/{config,metrics}.json + embeddings.npz
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
from run_ablation import DEFAULTS

EXPERIMENT_ID = "B4B"

# Same 12 datasets the shared data layer exposes. Ordered toy-then-chemistry so
# the figure's x-axis groups sensibly.
ALL = ["digits", "wine", "cancer", "iris", "moons", "circles",
       "swiss_roll", "s_curve", "severed_sphere",
       "bace", "bbbp", "dude"]

# The 3 manifold datasets are fit on a square grid everywhere else in this paper
# (A1 and B2 both), so match that here - otherwise this ablation would be
# measuring repulsion against a configuration the paper never actually uses.
SQUARE_LAYOUT = {"swiss_roll", "s_curve", "severed_sphere"}

RATES = [0, 5]  # off vs. NFMap's default


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default=",".join(ALL))
    ap.add_argument("--seeds", default="42,0,1")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    print(f"[{EXPERIMENT_ID}] {len(datasets)} datasets x {len(RATES)} rates x "
          f"{len(seeds)} seeds = {len(datasets) * len(RATES) * len(seeds)} fits")

    for ds in datasets:
        data = load_dataset(ds, seed=42)
        X, y, is_cont = data["X"], data["y"], data["is_continuous"]
        print(f"\n[{EXPERIMENT_ID}] {ds}: {X.shape[0]} x {X.shape[1]}"
              f"{' (continuous target - no silhouette)' if is_cont else ''}", flush=True)

        for seed in seeds:
            for rate in RATES:
                config = dict(DEFAULTS)
                if ds in SQUARE_LAYOUT:
                    config["prior_layout"] = "square"
                config["negative_sample_rate"] = rate
                config["random_state"] = seed
                method_label = f"rate{rate}_seed{seed}"

                t0 = time.time()
                try:
                    model = NFMap(**config)
                    Z = model.fit_transform(X)
                    metrics = compute_dr_metrics(X, Z, y, is_continuous=is_cont)
                    resolved = getattr(model, "resolved_grid_bounds_", config.get("grid_bounds"))
                    metrics.update(compute_grid_containment(Z, resolved))
                    ok = True
                except Exception as e:
                    print(f"    {method_label}: FAILED ({e})", flush=True)
                    Z = np.full((len(y), 2), np.nan)
                    metrics = compute_dr_metrics(None, None, y, is_continuous=is_cont)
                    metrics.update(compute_grid_containment(None, config.get("grid_bounds")))
                    ok = False
                elapsed = time.time() - t0

                run_config = dict(config)
                run_config.update({"dataset": ds, "negative_sample_rate": rate,
                                   "seed": seed, "ok": ok, "elapsed_s": elapsed})
                save_run(EXPERIMENT_ID, ds, method_label, run_config, metrics, Z, y=y)

                sil = metrics.get("sil")
                sil_s = "n/a" if sil is None or np.isnan(sil) else f"{sil:.3f}"
                print(f"    rate={rate} seed={seed}: Sil={sil_s} "
                      f"T/C_F1={metrics['t_c_f1']:.3f} DistCor={metrics['dist_corr']:.3f} "
                      f"DensCor={metrics['dens_corr']:.3f} Acc={metrics['acc']:.3f} "
                      f"({elapsed:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
