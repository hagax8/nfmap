"""B2C: sigma_perp as a local/global dial - both ends, every dataset, 3 seeds.

B2 and B2B establish that sigma_perp trades neighbourhood fidelity against
global distance fidelity, and that the optimum is dataset-specific rather than
a function of latent geometry (B2B: rho* varies 17x at fixed d_perp, so the
dimensionless scaling law does not hold). Given that, reporting one tuned value
per dataset would be selection on the test set.

This runs both ends of the dial instead, so the paper can show the trade-off
rather than pick a winner:

    sigma_perp = 0.01  - global-optimising. Squeezes z_perp, forcing variance
                         into the 2D map. Better DistCor; worse T/C F1.
    sigma_perp = 10.0  - local-optimising, and NFMap's shipped default. Leaves
                         z_perp roomy. Better T/C F1 / accuracy / silhouette;
                         worse DistCor on chemistry.

Embeddings are saved for both so the figures can show the same dataset under
each setting side by side.

moons and circles are excluded: 2D inputs give d_scaled = n_components = 2, so
z_perp is empty and sigma_perp has nothing to act on.

Usage:
    python b2c_sigma_local_global.py
    python b2c_sigma_local_global.py --datasets dude --seeds 42
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

EXPERIMENT_ID = "B2C"

DATASETS = ["digits", "wine", "cancer", "iris",
            "bace", "bbbp", "dude", "drd2", "egfr", "esol",
            "swiss_roll", "s_curve", "severed_sphere"]
SQUARE_LAYOUT = {"swiss_roll", "s_curve", "severed_sphere"}
SETTINGS = {"global": 0.01, "local": 10.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--seeds", default="42,0,1")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]
    print(f"[{EXPERIMENT_ID}] {len(datasets)} datasets x 2 settings x {len(seeds)} seeds "
          f"= {len(datasets)*2*len(seeds)} fits")

    for ds in datasets:
        data = load_dataset(ds, seed=42)
        X, y, is_cont = data["X"], data["y"], data["is_continuous"]
        print(f"\n[{EXPERIMENT_ID}] {ds}: {X.shape}", flush=True)
        for seed in seeds:
            for name, sigma in SETTINGS.items():
                cfg = dict(DEFAULTS)
                if ds in SQUARE_LAYOUT:
                    cfg["prior_layout"] = "square"
                cfg["sigma_perp"] = sigma
                cfg["random_state"] = seed
                label = f"{name}_seed{seed}"

                t0 = time.time()
                try:
                    m = NFMap(**cfg)
                    Z = m.fit_transform(X)
                    met = compute_dr_metrics(X, Z, y, is_continuous=is_cont)
                    met.update(compute_grid_containment(
                        Z, getattr(m, "resolved_grid_bounds_", cfg["grid_bounds"])))
                    ok = True
                except Exception as e:
                    print(f"    {label}: FAILED ({e})", flush=True)
                    Z = np.full((len(y), 2), np.nan)
                    met = compute_dr_metrics(None, None, y, is_continuous=is_cont)
                    met.update(compute_grid_containment(None, cfg["grid_bounds"]))
                    ok = False
                elapsed = time.time() - t0

                run_cfg = dict(cfg)
                run_cfg.update({"dataset": ds, "setting": name, "sigma_perp": sigma,
                                "seed": seed, "ok": ok, "elapsed_s": elapsed})
                save_run(EXPERIMENT_ID, ds, label, run_cfg, met, Z, y=y)

                sil = met.get("sil")
                sil_s = "n/a" if sil is None or np.isnan(sil) else f"{sil:.3f}"
                print(f"    {name:<6} (sig={sigma:<5}) seed={seed}: "
                      f"DistCor={met['dist_corr']:+.3f} T/C_F1={met['t_c_f1']:.3f} "
                      f"DensCor={met['dens_corr']:+.3f} Sil={sil_s} Acc={met['acc']:.3f} "
                      f"out={100*met['frac_outside_grid_bounds']:.1f}% ({elapsed:.0f}s)",
                      flush=True)


if __name__ == "__main__":
    main()
