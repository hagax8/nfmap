"""B3B: graph_weight across datasets and seeds.

B3 sweeps graph_weight on digits alone (4 values x 3 seeds). That is far
thinner than the evidence behind the paper's other two headline parameters -
sigma_perp has 720 fits across 9 datasets (B2/B2D/B2E) and repulsion_weight
234 across 9 (B4/B4C/B4D) - and this paper has already been misled three
separate times by reading a single-dataset sweep as a general result (B2's
retracted high-end "erosion", B4B's "dude shows repulsion helping", B2B's
"no sigma_perp contains the manifold sets"). A parameter cannot sit in the
ablation section on one dataset's evidence.

Same dataset groups as B4C, so the two ablations are directly comparable:
  * well-separated classes (digits, wine, cancer)
  * chemistry (bace, bbbp, drd2)  - DUD-E dropped from the short paper
  * 3D manifolds (swiss_roll, s_curve, severed_sphere)

graph_weight scales the fuzzy-simplicial attraction term, so the expectation
is that it behaves like repulsion's mirror image: raising it should tighten
neighbourhoods (T/C F1 up) at the cost of global distance fidelity, and unlike
repulsion it should NOT push points off the canvas. Whether that holds at the
chemistry operating point is the open question - B4D found repulsion's effect
reversed sign there, so the attraction arm cannot be assumed to transfer either.

Usage:
    python b3b_graph_weight_breadth.py
    python b3b_graph_weight_breadth.py --datasets digits --seeds 42
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

EXPERIMENT_ID = "B3B"

DATASETS = ["digits", "wine", "cancer",
            "bace", "bbbp", "drd2",
            "swiss_roll", "s_curve", "severed_sphere"]
SQUARE_LAYOUT = {"swiss_roll", "s_curve", "severed_sphere"}
WEIGHTS = [0.0, 1.0, 5.0, 10.0]


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
          f"{len(seeds)} seeds = {len(datasets)*len(weights)*len(seeds)} fits",
          flush=True)

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
                cfg["graph_weight"] = w
                cfg["random_state"] = seed
                label = f"gw{slugify(w)}_seed{seed}"

                t0 = time.time()
                try:
                    model = NFMap(**cfg)
                    Z = model.fit_transform(X)
                    metrics = compute_dr_metrics(X, Z, y, is_continuous=is_cont)
                    resolved = getattr(model, "resolved_grid_bounds_",
                                       cfg.get("grid_bounds"))
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
                run_cfg.update({"dataset": ds, "graph_weight": w, "seed": seed,
                                "ok": ok, "elapsed_s": elapsed})
                save_run(EXPERIMENT_ID, ds, label, run_cfg, metrics, Z, y=y)

                sil = metrics.get("sil")
                sil_s = "n/a" if sil is None or np.isnan(sil) else f"{sil:.3f}"
                print(f"    gw={w:<5} seed={seed}: T/C_F1={metrics['t_c_f1']:.3f} "
                      f"DistCor={metrics['dist_corr']:.3f} Sil={sil_s} "
                      f"Acc={metrics['acc']:.3f} "
                      f"out={100*metrics['frac_outside_grid_bounds']:.1f}% "
                      f"({elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
