"""B22: fit and save the manifold maps used by f8/s2, as a standalone step.

Exists because fitting NFMap inline inside make_paper_figures.py segfaults:
matplotlib + torch + sklearn + nfmap loaded together in one process crash
during the actual fit (exit 139), even though the identical fit logic succeeds
in a script that never imports matplotlib. Rather than chase the exact
library conflict, this does what f3/DRD2 already does - fit in a plotting-free
process, save embeddings, and have the figure script only ever load them.

Fits every method on each manifold dataset at exactly the setting Table 1
reports for it (make_paper_tables.manifold_selection, from B23), so the map
figures cannot drift from the table. CPU only, via common/__init__.py: MPS
fits are not reproducible run-to-run.

Usage:
    python b22_fit_manifold_maps.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from common.data import load_dataset
from common.metrics import compute_dr_metrics, compute_grid_containment
from run_ablation import DEFAULTS

import torch as _torch
_torch.backends.mps.is_available = lambda: False

from nfmap import NFMap

from make_paper_tables import manifold_selection, selected_settings, MANIFOLD
# DRD2 at both ends of the dial, for fig_maps_chemistry (f3). Saved under
# drd2_global/drd2_local so they don't collide with any plain "drd2" entry.
CHEM_SIGMA = {"drd2_global": 0.003, "drd2_local": 10.0}
# Chemistry datasets whose baseline panels are shown in a figure (f3).
CHEM_BASELINES = ["drd2"]
GRID = 12.0
SEED = 42

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "B22")


def _fit_and_save(ds, sigma, is_cont, prior_layout=None, save_as=None, grid=GRID):
    d = load_dataset(ds, seed=42)
    X, y = d["X"], d["y"]
    cfg = {**DEFAULTS, "sigma_perp": sigma, "grid_bounds": grid, "random_state": SEED}
    if prior_layout:
        cfg["prior_layout"] = prior_layout
    name = save_as or ds
    print(f"[B22] {name}: fitting at sigma_perp={sigma:g}...", flush=True)
    m = NFMap(**cfg)
    Z = m.fit_transform(X)
    met = compute_dr_metrics(X, Z, y, is_continuous=is_cont)
    met.update(compute_grid_containment(
        Z, getattr(m, "resolved_grid_bounds_", grid)))
    np.savez(os.path.join(OUT_DIR, f"{name}.npz"), Z=Z, y=y)
    with open(os.path.join(OUT_DIR, f"{name}_metrics.json"), "w") as f:
        json.dump({**{k: float(v) for k, v in met.items()},
                   "setting": {"sigma_perp": sigma, "grid_bounds": grid}}, f, indent=2)
    print(f"      DistCor={met['dist_corr']:.3f} T/C={met['t_c_f1']:.3f} "
          f"out={100*met['frac_outside_grid_bounds']:.0f}%", flush=True)


def _fit_baseline_and_save(ds, method, params, seed=42):
    """t-SNE / UMAP / GTM at the Table 1 setting, saved like the NFMap maps
    above. Also fitted standalone: these segfault inside make_paper_figures.py's
    process (torch + sklearn + UMAP loaded together), which is what this whole
    script exists to avoid. Constructors match B23's exactly."""
    from common.metrics import compute_dr_metrics
    from sklearn import manifold as skm
    d = load_dataset(ds, seed=42)
    X, y, is_cont = d["X"], d["y"], d["is_continuous"]
    name = f"{ds}_{method}"
    print(f"[B22] {name}: fitting ({params})...", flush=True)
    if method == "tsne":
        Z = skm.TSNE(n_components=2, init="pca", random_state=seed,
                     perplexity=params["perplexity"]).fit_transform(X)
    elif method == "umap":
        import umap as umap_lib
        Z = umap_lib.UMAP(n_neighbors=params["n_neighbors"],
                          min_dist=params["min_dist"],
                          random_state=seed).fit_transform(X)
    else:
        import ugtm
        Z = ugtm.eGTM(k=10, m=6, random_state=seed).fit_transform(X)  # = B23
    met = compute_dr_metrics(X, Z, y, is_continuous=is_cont)
    np.savez(os.path.join(OUT_DIR, f"{name}.npz"), Z=Z, y=y)
    with open(os.path.join(OUT_DIR, f"{name}_metrics.json"), "w") as f:
        json.dump({**{k: float(v) for k, v in met.items()}, "setting": params},
                  f, indent=2)
    print(f"      DistCor={met['dist_corr']:.3f} T/C={met['t_c_f1']:.3f}", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sel = manifold_selection()
    missing = [ds for ds in MANIFOLD if "nfmap" not in sel.get(ds, {})]
    if missing:
        raise SystemExit(f"[B22] no B23 selection for {missing} - run "
                         "b23_table1_reproducible_sweeps.py first")
    for ds in MANIFOLD:
        nf = sel[ds]["nfmap"]
        _fit_and_save(ds, nf["sigma_perp"], is_cont=True, prior_layout="square",
                      grid=nf["grid_bounds"])
    for name, sigma in CHEM_SIGMA.items():
        _fit_and_save(name.split("_")[0], sigma, is_cont=False, save_as=name)

    # Geodesic distance matrix per manifold dataset, used by every panel in
    # the comparison figures. Precomputed here so make_paper_figures.py never
    # calls sklearn's kNN graph construction inline in its own process.
    from common.metrics import geodesic_distances
    for ds in MANIFOLD:
        X = load_dataset(ds, seed=42)["X"]
        d_geo, n_comp = geodesic_distances(X)
        if d_geo is not None:
            np.save(os.path.join(OUT_DIR, f"{ds}_geodesic.npy"), d_geo)
            print(f"[B22] {ds}: geodesic distances saved", flush=True)
        else:
            print(f"[B22] {ds}: kNN graph disconnected ({n_comp} parts), "
                  f"geodesic distances not saved", flush=True)

    # Baselines at their Table 1 settings: the manifolds for s2/f8, and DRD2
    # for f3 (whose baselines used to be A2's untuned defaults).
    sel.update(selected_settings(CHEM_BASELINES))
    for ds in MANIFOLD + CHEM_BASELINES:
        for method in ("tsne", "umap", "gtm"):
            if method in sel.get(ds, {}):
                _fit_baseline_and_save(ds, method, sel[ds][method])

    print(f"\n  -> {OUT_DIR}/<dataset>.npz")


if __name__ == "__main__":
    main()
