"""B4C: repulsion_weight dose-response, across datasets and seeds.

B4 sweeps repulsion_weight on digits alone and finds a clear, monotone effect:
silhouette falls 0.506 -> 0.057 as the weight rises 0 -> 10, T/C F1 and KNN
accuracy peak at the shipped default of 1.0, and the map falls apart at 10.
But B4B established that repulsion's effect is strongly dataset-dependent - the
silhouette response is concentrated on datasets with well-separated classes and
is absent on the weak-signal chemistry sets - so a single-dataset dose-response
curve is exactly the kind of result that has misled this paper before.

Three dataset groups, chosen to span the regimes B2 and B4B identified:
  * well-separated classes (digits, wine, cancer) - where compression has
    something to compress, and where silhouette reacted most in B4B
  * chemistry (bace, bbbp, dude) - weak cluster structure, where B4B found the
    off-vs-on contrast did essentially nothing
  * 3D manifolds (swiss_roll, s_curve, severed_sphere) - continuous targets, no
    silhouette at all, and the group where B4B found repulsion protecting
    DistCor most strongly

Run through a dedicated script rather than run_ablation.py because that script
has no seed in its output path, so multiple seeds would overwrite each other.
Deltas are reported per seed and then averaged, as in B4B.

Usage:
    python b4c_repulsion_sweep.py
    python b4c_repulsion_sweep.py --datasets digits --seeds 42
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
from run_ablation import DEFAULTS, slugify

EXPERIMENT_ID = "B4C"

DATASETS = ["digits", "wine", "cancer",
            "bace", "bbbp", "dude",
            "swiss_roll", "s_curve", "severed_sphere"]
SQUARE_LAYOUT = {"swiss_roll", "s_curve", "severed_sphere"}
WEIGHTS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--seeds", default="42,0,1")
    ap.add_argument("--weights", default=",".join(str(w) for w in WEIGHTS))
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    weights = [float(w) for w in args.weights.split(",")]
    print(f"[{EXPERIMENT_ID}] {len(datasets)} datasets x {len(weights)} weights x "
          f"{len(seeds)} seeds = {len(datasets)*len(weights)*len(seeds)} fits")

    for ds in datasets:
        data = load_dataset(ds, seed=42)
        X, y, is_cont = data["X"], data["y"], data["is_continuous"]
        print(f"\n[{EXPERIMENT_ID}] {ds}: {X.shape[0]} x {X.shape[1]}"
              f"{' (continuous - no silhouette)' if is_cont else ''}", flush=True)

        for seed in seeds:
            for w in weights:
                cfg = dict(DEFAULTS)
                if ds in SQUARE_LAYOUT:
                    cfg["prior_layout"] = "square"
                cfg["repulsion_weight"] = w
                cfg["random_state"] = seed
                label = f"w{slugify(w)}_seed{seed}"

                t0 = time.time()
                try:
                    model = NFMap(**cfg)
                    Z = model.fit_transform(X)
                    metrics = compute_dr_metrics(X, Z, y, is_continuous=is_cont)
                    resolved = getattr(model, "resolved_grid_bounds_", cfg.get("grid_bounds"))
                    metrics.update(compute_grid_containment(Z, resolved))
                    ok = True
                except Exception as e:
                    print(f"    {label}: FAILED ({e})", flush=True)
                    Z = np.full((len(y), 2), np.nan)
                    metrics = compute_dr_metrics(None, None, y, is_continuous=is_cont)
                    metrics.update(compute_grid_containment(None, cfg.get("grid_bounds")))
                    ok = False
                elapsed = time.time() - t0

                run_cfg = dict(cfg)
                run_cfg.update({"dataset": ds, "repulsion_weight": w, "seed": seed,
                                "ok": ok, "elapsed_s": elapsed})
                save_run(EXPERIMENT_ID, ds, label, run_cfg, metrics, Z, y=y)

                sil = metrics.get("sil")
                sil_s = "n/a" if sil is None or np.isnan(sil) else f"{sil:.3f}"
                print(f"    w={w:<5} seed={seed}: Sil={sil_s} "
                      f"T/C_F1={metrics['t_c_f1']:.3f} DistCor={metrics['dist_corr']:.3f} "
                      f"DensCor={metrics['dens_corr']:.3f} Acc={metrics['acc']:.3f} "
                      f"({elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
