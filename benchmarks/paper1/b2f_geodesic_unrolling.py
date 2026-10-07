"""B2F: does a geodesic measure agree with visual unrolling on the manifolds?

The paper scores maps with DistCor = corr(pdist(X), pdist(Z)), i.e. against
AMBIENT distances. For a rolled manifold that is arguably the wrong comparison:
two points on adjacent layers of a swiss roll are close in R^3 but far along the
surface, so a map that preserves ambient distance is one that keeps the roll
ROLLED. The sigma_perp ladder shows exactly this - at the smallest sigma_perp the
swiss roll map is a closed ring, with the two ends of the manifold adjacent, and
it scores well.

This computes the geodesic alternative. Build a kNN graph on the input, take
all-pairs shortest paths (Isomap's construction), and correlate those distances
with the map's. An unrolled sheet has map distance proportional to geodesic
distance; a still-rolled or folded one does not.

Also reported, because the existing local metrics are structurally blind to
folds (a fold is locally consistent everywhere, so kNN R^2 and trustworthiness
cannot see it):

    fold_rate - of the pairs in the top decile of geodesic distance, the
                fraction landing in the bottom decile of map distance. A fold
                puts far-apart-on-the-manifold points close together on the map,
                so this rises when the sheet folds back on itself.

Purpose. We need either a stated criterion that justifies a sigma_perp for the
manifold datasets, or evidence that no single value is justifiable and the
sweep itself should be reported. Visual inspection says ~0.01 for swiss roll and
s_curve but 0.3-1 for the severed sphere; the constrained rule in the paper says
1.0 for all three. This decides between them on ground-truth geometry rather
than on either eyeballing or a metric we have reason to distrust.

Usage:
    python b2f_geodesic_unrolling.py
    python b2f_geodesic_unrolling.py --datasets s_curve --seeds 42
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

EXPERIMENT_ID = "B2F"

DATASETS = ["swiss_roll", "s_curve", "severed_sphere"]
SIGMAS = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
GRID_BOUNDS = 12.0
KNN_GRAPH = 10          # neighbours for the geodesic graph


def geodesic_distances(X, k=KNN_GRAPH):
    """All-pairs shortest-path distances on a kNN graph (Isomap's construction).

    Returns the condensed upper triangle, matching scipy's pdist layout, or None
    if the graph is disconnected (shortest paths would be infinite and the
    correlation undefined - better to report nothing than a silently dropped
    subset of pairs).
    """
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import shortest_path, connected_components

    G = kneighbors_graph(X, n_neighbors=k, mode="distance")
    G = G.maximum(G.T)                      # symmetrise
    n_comp, _ = connected_components(G, directed=False)
    if n_comp != 1:
        return None, n_comp
    D = shortest_path(G, method="D", directed=False)
    iu = np.triu_indices(len(X), k=1)
    return D[iu], 1


def fold_rate(d_geo, d_map, q=0.10):
    """Fraction of the most geodesically-distant pairs that land closest on the map."""
    far = d_geo >= np.quantile(d_geo, 1 - q)
    near_map = d_map <= np.quantile(d_map, q)
    return float(np.mean(near_map[far])) if far.any() else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--seeds", default="42,0,1")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]

    from scipy.spatial.distance import pdist
    from scipy.stats import pearsonr, spearmanr

    out_dir = os.path.join(RESULTS_ROOT, EXPERIMENT_ID)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "geodesic_unrolling.json")
    rows = []
    print(f"[{EXPERIMENT_ID}] {len(datasets)} datasets x {len(SIGMAS)} sigma x "
          f"{len(seeds)} seeds = {len(datasets)*len(SIGMAS)*len(seeds)} fits",
          flush=True)

    for ds in datasets:
        d = load_dataset(ds, seed=42)
        X, y = d["X"], d["y"]
        d_geo, n_comp = geodesic_distances(X)
        if d_geo is None:
            print(f"\n[{EXPERIMENT_ID}] {ds}: SKIPPED - kNN graph has {n_comp} "
                  f"components at k={KNN_GRAPH}, geodesics undefined", flush=True)
            continue
        d_amb = pdist(X)
        print(f"\n[{EXPERIMENT_ID}] {ds}: {X.shape}, geodesic graph connected. "
              f"corr(ambient, geodesic) = {pearsonr(d_amb, d_geo)[0]:+.3f}", flush=True)

        for sg in SIGMAS:
            cell = []
            for seed in seeds:
                cfg = {**DEFAULTS, "prior_layout": "square", "sigma_perp": sg,
                       "grid_bounds": GRID_BOUNDS, "random_state": seed}
                try:
                    m = NFMap(**cfg)
                    Z = m.fit_transform(X)
                    d_map = pdist(Z)
                    met = compute_dr_metrics(X, Z, y, is_continuous=True)
                    met.update(compute_grid_containment(
                        Z, getattr(m, "resolved_grid_bounds_", GRID_BOUNDS)))
                    rec = dict(
                        dataset=ds, sigma_perp=sg, seed=seed,
                        dist_corr_ambient=float(met["dist_corr"]),
                        dist_corr_geodesic=float(pearsonr(d_map, d_geo)[0]),
                        spearman_geodesic=float(spearmanr(d_map, d_geo)[0]),
                        fold_rate=fold_rate(d_geo, d_map),
                        knn_r2=float(met["acc"]),
                        t_c_f1=float(met["t_c_f1"]),
                        frac_outside=float(met["frac_outside_grid_bounds"]),
                    )
                except Exception as e:
                    print(f"    sigma={sg} seed={seed}: FAILED ({e})", flush=True)
                    continue
                rows.append(rec)
                cell.append(rec)
                with open(out_path, "w") as f:
                    json.dump(rows, f, indent=2)
            if cell:
                g = np.mean([c["dist_corr_geodesic"] for c in cell])
                a = np.mean([c["dist_corr_ambient"] for c in cell])
                fr = np.mean([c["fold_rate"] for c in cell])
                o = np.mean([c["frac_outside"] for c in cell])
                print(f"    sigma={sg:<7} geodesic={g:+.3f}  ambient={a:+.3f}  "
                      f"fold={100*fr:.1f}%  out={100*o:.0f}%", flush=True)

    print(f"\n  -> {out_path}")


if __name__ == "__main__":
    main()
