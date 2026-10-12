"""B23: Unified, reproducible (CPU-only) sweeps for Table S1 Comparison.

This script runs a comprehensive hyperparameter sweep for NFMap, UMAP, t-SNE, and GTM
across Chemistry, 3-D Manifold, and non-chemistry multi-dimensional datasets (digits, etc.).
It explicitly forces CPU execution for reproducibility (disabling MPS for PyTorch and
enforcing single-threading where possible). 

It calculates both ambient `dist_corr` and `dist_corr_geodesic` so that the Pareto
frontier for manifold datasets can correctly use geodesic distance as the global measure.

Usage:
    python b23_table1_reproducible_sweeps.py
    python b23_table1_reproducible_sweeps.py --datasets digits,drd2
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import torch as _torch
_torch.backends.mps.is_available = lambda: False
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.insert(0, ".")
from common.data import load_dataset
from common.metrics import compute_dr_metrics, compute_grid_containment, tear_rate
from common.io import RESULTS_ROOT
from common.metrics import geodesic_distances, fold_rate

from nfmap import NFMap
from run_ablation import DEFAULTS
from sklearn import manifold as skm
import umap as umap_lib
import ugtm

EXPERIMENT_ID = "B23"

CHEM = ["bace", "bbbp", "esol", "drd2", "egfr"]
MANIFOLD = ["swiss_roll", "s_curve", "severed_sphere"]
OTHER = ["digits", "iris", "cancer", "wine"]
DATASETS = CHEM + MANIFOLD + OTHER

# Sweeps
# Half-decade ladder to 1000, plus 5 (the S-curve's best cell in the
# earlier 8-rung sweep). Above 10 is where the manifolds unroll best (B24);
# 0.003 is the chemistry optimum and must stay. Two grids: grid_bounds is not
# just a containment knob - at the same sigma_perp, fully contained, ESOL
# scores 0.765 on grid 24 vs 0.583 on grid 12, and grid 12 alone moved every
# chemistry knee down by 0.04-0.18.
NF_SIGMAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 5.0, 10.0,
             30.0, 100.0, 300.0, 1000.0]
NF_GRIDS = [12.0, 24.0]
MANIFOLD_SET = set(MANIFOLD)
TSNE_PERPS = [5, 15, 30, 50, 100, 200, 300, 400, 500]
UMAP_NNS = [5, 15, 50, 100, 200, 400]
UMAP_MIN_DISTS = [0.0, 0.1, 0.5, 0.99]
GTM_K, GTM_M = 10, 6


def _compute_all_metrics(X, Z, y, is_cont, gb, m_nfmap=None, d_geo=None):
    met = compute_dr_metrics(X, Z, y, is_continuous=is_cont)
    
    # Calculate geodesic dist_corr
    if d_geo is not None:
        from scipy.spatial.distance import pdist
        from scipy.stats import pearsonr
        d_latent = pdist(Z, metric="euclidean")
        mask = np.isfinite(d_geo)
        if mask.sum() > 1:
            g_corr, _ = pearsonr(d_geo[mask], d_latent[mask])
            met["dist_corr_geodesic"] = float(g_corr)
            met["fold_rate"] = fold_rate(d_geo, d_latent)
        else:
            met["dist_corr_geodesic"] = float('nan')
    else:
        met["dist_corr_geodesic"] = float('nan')

    met["tear_rate"] = tear_rate(X, Z)

    # Grid containment
    if m_nfmap is not None:
        met.update(compute_grid_containment(Z, getattr(m_nfmap, "resolved_grid_bounds_", gb)))
    else:
        met["frac_outside_grid_bounds"] = 0.0
        
    return {k: float(v) for k, v in met.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--seeds", default="42,0,1")
    # New file: table1_sweeps.json (old 8-sigma x 3-grid ladder) no longer
    # reproduces under the current code and is kept for comparison only.
    ap.add_argument("--out", default="table1_sweeps_v2.json")
    # Subset runs, e.g. extending an existing sweep with new NFMap cells only.
    ap.add_argument("--nfmap-only", action="store_true")
    ap.add_argument("--gtm-only", action="store_true")
    ap.add_argument("--nf-grids", default=None, help="comma list; default NF_GRIDS")
    ap.add_argument("--nf-sigmas", default=None, help="comma list; default NF_SIGMAS")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]
    nf_grids = ([float(g) for g in args.nf_grids.split(",")] if args.nf_grids
                else NF_GRIDS)
    nf_sigmas = ([float(x) for x in args.nf_sigmas.split(",")] if args.nf_sigmas
                 else NF_SIGMAS)
    if args.gtm_only:
        nf_sigmas = []

    out_dir = os.path.join(RESULTS_ROOT, EXPERIMENT_ID)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.out)
    rows = []

    for ds in datasets:
        data = load_dataset(ds, seed=42)
        X, y, is_cont = data["X"], data["y"], data.get("is_continuous", False)
        n = X.shape[0]
        
        # Precompute geodesic distances for all datasets
        print(f"\n[{EXPERIMENT_ID}] {ds}: Precomputing geodesic distances...", flush=True)
        d_geo, n_comp = geodesic_distances(X)
        if d_geo is None:
            print(f"  Geodesic disconnected ({n_comp} parts), dist_corr_geodesic will be NaN", flush=True)
            
        print(f"\n[{EXPERIMENT_ID}] {ds}: Starting sweeps", flush=True)
        
        # ── 1. NFMap ──
        for sigma in nf_sigmas:
            for gb in nf_grids:
                for seed in seeds:
                    cfg = {**DEFAULTS, "sigma_perp": sigma, "grid_bounds": gb, "random_state": seed}
                    if ds in MANIFOLD_SET:
                        cfg["prior_layout"] = "square"
                    try:
                        m = NFMap(**cfg)
                        Z = m.fit_transform(X)
                        met = _compute_all_metrics(X, Z, y, is_cont, gb, m, d_geo)
                        rows.append(dict(dataset=ds, method="NFMap", seed=seed, sigma_perp=sigma, grid_bounds=gb, **met))
                    except Exception as e:
                        print(f"  NFMap sigma={sigma} gb={gb} seed={seed} FAILED: {e}")

        if args.nfmap_only:
            with open(out_path, "w") as f:
                json.dump(rows, f, indent=2)
            continue

        # ── 2. GTM ──
        for seed in seeds:
            try:
                # k=10 -> 10x10 = 100 nodes, matching NFMap's
                # n_prior_components=100 (see nfmap_diagnostics.py's GTM
                # branch). k is ugtm's sqrt node count; k=16 had silently
                # given GTM 256 nodes, the mismatch B15 already records.
                m = ugtm.eGTM(k=GTM_K, m=GTM_M, random_state=seed)
                Z = m.fit_transform(X)
                met = _compute_all_metrics(X, Z, y, is_cont, None, None, d_geo)
                rows.append(dict(dataset=ds, method="GTM", seed=seed, **met))
            except Exception as e:
                print(f"  GTM seed={seed} FAILED: {e}")

        if args.gtm_only:
            with open(out_path, "w") as f:
                json.dump(rows, f, indent=2)
            continue

        # ── 3. t-SNE ──
        for p in TSNE_PERPS:
            if p >= n: continue
            for seed in seeds:
                try:
                    Z = skm.TSNE(n_components=2, init="pca", random_state=seed, perplexity=p).fit_transform(X)
                    met = _compute_all_metrics(X, Z, y, is_cont, None, None, d_geo)
                    rows.append(dict(dataset=ds, method="tsne", seed=seed, perplexity=p, **met))
                except Exception as e:
                    print(f"  t-SNE p={p} seed={seed} FAILED: {e}")

        # ── 4. UMAP ──
        best_nn = 15
        umap_rows = []
        for nn in UMAP_NNS:
            if nn >= n: continue
            for seed in seeds:
                try:
                    Z = umap_lib.UMAP(n_neighbors=nn, min_dist=0.1, random_state=seed).fit_transform(X)
                    met = _compute_all_metrics(X, Z, y, is_cont, None, None, d_geo)
                    rec = dict(dataset=ds, method="umap", seed=seed, n_neighbors=nn, min_dist=0.1, **met)
                    rows.append(rec)
                    umap_rows.append(rec)
                except Exception as e:
                    print(f"  UMAP nn={nn} seed={seed} FAILED: {e}")
        
        # Best n_neighbors to sweep min_dist
        if umap_rows:
            by_nn = {}
            # Same global measure Table 1 selects on: geodesic for manifolds.
            gk = "dist_corr_geodesic" if ds in MANIFOLD_SET else "dist_corr"
            for r in umap_rows: by_nn.setdefault(r["n_neighbors"], []).append(r[gk])
            best_nn = max(by_nn, key=lambda k: np.mean(by_nn[k]))
            
        for md in UMAP_MIN_DISTS:
            if md == 0.1: continue
            for seed in seeds:
                try:
                    Z = umap_lib.UMAP(n_neighbors=best_nn, min_dist=md, random_state=seed).fit_transform(X)
                    met = _compute_all_metrics(X, Z, y, is_cont, None, None, d_geo)
                    rows.append(dict(dataset=ds, method="umap", seed=seed, n_neighbors=best_nn, min_dist=md, **met))
                except Exception as e:
                    print(f"  UMAP md={md} seed={seed} FAILED: {e}")

        # Write out dynamically so we don't lose data on crash
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2)

    print(f"\n  -> {out_path}")

if __name__ == "__main__":
    main()
