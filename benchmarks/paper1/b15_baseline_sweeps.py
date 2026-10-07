"""B15: sweep the baselines' own local/global knobs, so the comparison is fair.

Every baseline number in this project so far comes from one fixed setting -
t-SNE at perplexity=30, UMAP at n_neighbors=15/min_dist=0.1 - while NFMap's
sigma_perp was swept over four orders of magnitude and the best contained value
selected. That is not a fair comparison, and it is the same error this project
has already made twice (GTM at 256 grid nodes against NFMap's 100; UMAP and
t-SNE given raw 2048-dim input while NFMap reduced internally).

Both baselines have a documented local/global control:

  * t-SNE's perplexity sets the effective neighbourhood size of the
    conditional distributions; low values model very local structure.
  * UMAP's n_neighbors is described in its own documentation as controlling
    "how UMAP balances local versus global structure"; min_dist separately
    controls how tightly points may pack.

The paper's headline comparative claim is that NFMap reaches DistCor ~0.65 on
chemistry where t-SNE and UMAP sit near 0.01. That claim is only meaningful if
the baselines cannot get there with their own knobs turned. This finds out.

GTM's grid resolution (k) is already matched to NFMap's n_prior_components=100
and is not swept here - it is a resolution parameter, not a local/global dial.

Usage:
    python b15_baseline_sweeps.py
    python b15_baseline_sweeps.py --datasets drd2 --seeds 42
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from common.data import load_dataset
from common.metrics import compute_dr_metrics
from common.io import RESULTS_ROOT

EXPERIMENT_ID = "B15"

CHEM = ["bace", "bbbp", "esol", "drd2", "egfr"]
MANIFOLD = ["swiss_roll", "s_curve", "severed_sphere"]
DATASETS = CHEM + MANIFOLD

PERPLEXITIES = [300, 400, 500]   # extension: the first sweep peaked at its 200 boundary
N_NEIGHBORS = []
MIN_DISTS = []

# NFMap's best contained setting, for reference in the printout only.
NFMAP_REF = {"bace": 0.578, "bbbp": 0.689, "esol": 0.834, "drd2": 0.601,
             "egfr": 0.642, "swiss_roll": 0.743, "s_curve": 0.686,
             "severed_sphere": 0.742}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--seeds", default="42,0,1")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]

    from sklearn import manifold
    import umap as umap_lib

    out_dir = os.path.join(RESULTS_ROOT, EXPERIMENT_ID)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "baseline_sweeps_ext.json")
    rows = []

    for ds in datasets:
        d = load_dataset(ds, seed=42)
        X, y, is_cont = d["X"], d["y"], d["is_continuous"]
        n = X.shape[0]
        print(f"\n[{EXPERIMENT_ID}] {ds}: {X.shape}   "
              f"NFMap best contained DistCor = {NFMAP_REF.get(ds, float('nan')):.3f}",
              flush=True)

        def record(method, params, Z, seed):
            met = compute_dr_metrics(X, Z, y, is_continuous=is_cont)
            rows.append(dict(dataset=ds, method=method, seed=seed,
                             **params,
                             dist_corr=float(met["dist_corr"]),
                             t_c_f1=float(met["t_c_f1"]),
                             acc=float(met["acc"])))
            with open(out_path, "w") as f:
                json.dump(rows, f, indent=2)

        # ── t-SNE: perplexity ────────────────────────────────────────────────
        for p in PERPLEXITIES:
            # sklearn requires perplexity < n_samples; skip rather than clamp so
            # the swept grid stays honest about what was actually run.
            if p >= n:
                continue
            cell = []
            for seed in seeds:
                try:
                    Z = manifold.TSNE(n_components=2, init="pca", random_state=seed,
                                      perplexity=p).fit_transform(X)
                except Exception as e:
                    print(f"    tsne perplexity={p} seed={seed}: FAILED ({e})", flush=True)
                    continue
                record("tsne", {"perplexity": p}, Z, seed)
                cell.append(rows[-1])
            if cell:
                print(f"    tsne  perplexity={p:<5} "
                      f"DistCor={np.mean([c['dist_corr'] for c in cell]):+.3f}"
                      f"±{np.std([c['dist_corr'] for c in cell]):.3f}  "
                      f"T/C={np.mean([c['t_c_f1'] for c in cell]):.3f}", flush=True)

        # ── UMAP: n_neighbors ────────────────────────────────────────────────
        for nn in N_NEIGHBORS:
            if nn >= n:
                continue
            cell = []
            for seed in seeds:
                try:
                    Z = umap_lib.UMAP(n_neighbors=nn, min_dist=0.1,
                                      random_state=seed).fit_transform(X)
                except Exception as e:
                    print(f"    umap n_neighbors={nn} seed={seed}: FAILED ({e})", flush=True)
                    continue
                record("umap", {"n_neighbors": nn, "min_dist": 0.1}, Z, seed)
                cell.append(rows[-1])
            if cell:
                print(f"    umap  n_neighbors={nn:<5} "
                      f"DistCor={np.mean([c['dist_corr'] for c in cell]):+.3f}"
                      f"±{np.std([c['dist_corr'] for c in cell]):.3f}  "
                      f"T/C={np.mean([c['t_c_f1'] for c in cell]):.3f}", flush=True)

        # ── UMAP: min_dist, at the n_neighbors that did best above ───────────
        best_nn = 15
        umap_rows = [r for r in rows if r["dataset"] == ds and r["method"] == "umap"]
        if umap_rows:
            by_nn = {}
            for r in umap_rows:
                by_nn.setdefault(r["n_neighbors"], []).append(r["dist_corr"])
            best_nn = max(by_nn, key=lambda k: np.mean(by_nn[k]))
        for md in MIN_DISTS:
            if md == 0.1:
                continue                      # already covered by the n_neighbors arm
            cell = []
            for seed in seeds:
                try:
                    Z = umap_lib.UMAP(n_neighbors=best_nn, min_dist=md,
                                      random_state=seed).fit_transform(X)
                except Exception as e:
                    print(f"    umap min_dist={md} seed={seed}: FAILED ({e})", flush=True)
                    continue
                record("umap", {"n_neighbors": best_nn, "min_dist": md}, Z, seed)
                cell.append(rows[-1])
            if cell:
                print(f"    umap  min_dist={md:<5} (nn={best_nn}) "
                      f"DistCor={np.mean([c['dist_corr'] for c in cell]):+.3f}"
                      f"±{np.std([c['dist_corr'] for c in cell]):.3f}  "
                      f"T/C={np.mean([c['t_c_f1'] for c in cell]):.3f}", flush=True)

    print(f"\n  -> {out_path}")


if __name__ == "__main__":
    main()
