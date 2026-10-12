"""B20: the scale study, rerun reproducibly on the current architecture.

Figure f9 was originally built from maps generated during development and
archived to Navifold. Those runs predated several changes to the flow, and the
generating script lived outside this repository, so the figure was not
reproducible from the paper's own code. This script replaces it: self-contained,
resumable, and running the exact configurations the figure needs.

Two components are examined, both of which address crowding and are therefore
expected to matter only at full dataset scale:

    hierarchical kernel - the neighbourhood graph is built from a weighted
        Tanimoto similarity over three fingerprint tiers (whole molecule,
        Bemis-Murcko scaffold, generic skeleton) rather than Euclidean distance
        on PCA components, and passed to NFMap through its knn_indices /
        knn_dists contract.
    Student-t prior - heavier tails over the map coordinates, so points far
        from every grid component are penalised less. Swept over degrees of
        freedom, since the development runs found an optimum near nu=2.

Datasets are the complete ChEMBL activity sets (DRD2 869, hERG 9,400, EGFR
10,573), not the 600-molecule subsamples used elsewhere in paper1: at that size
neither component has a measurable effect (B16, B17).

Scoring is scaffold purity@5 and scaffold silhouette over exact Murcko
scaffolds, restricted to molecules whose scaffold has at least one other
member (see common/metrics.py).

Usage:
    python b20_scale_study.py                      # the full figure (12 fits)
    python b20_scale_study.py --datasets drd2      # cheapest dataset only
    python b20_scale_study.py --skip-dof           # kernel arm only
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))
from common.io import RESULTS_ROOT
from common.metrics import murcko_ids, score

EXPERIMENT_ID = "B20"

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATASETS = ["drd2", "herg", "egfr"]
DOFS = [2.0, 3.0, 5.0, 10.0, 20.0, 30.0]
DOF_DATASET = "egfr"            # the dof sweep is reported on EGFR only
SEED = 42

# Matches the production pipeline that generated the archived maps, minus the
# LMDT head and native predictor, which this paper does not use and which do
# not affect the map coordinates. grid_bounds is left at 'auto' because the
# Student-t path resolves it differently from the Gaussian one - pinning a
# single value would hand one prior family a canvas tuned for the other, which
# is the coupling error documented in Section 4.
NFMAP_KWARGS = dict(
    n_components=2, complexity="medium", n_prior_components=100,
    prior_weight=0.1, graph_weight=5.0, repulsion_weight=1.0,
    sigma_perp=10.0, negative_sample_rate=5, learnable_priors=False,
    vol_weight=1.0, max_epochs=300, patience=30, random_state=SEED,
    graph_kernel="fuzzy_simplicial", prior_layout="hexagonal",
    grid_bounds="auto", verbose=False,
)


def load_target(name):
    import pandas as pd
    path = os.path.join(DATA_DIR, f"{name}_clean.csv")
    if not os.path.exists(path):
        raise SystemExit(f"missing dataset: {path}")
    return pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--skip-dof", action="store_true")
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",")]

    from chem_kernel.hybrid_kernel import (compute_hierarchical_fingerprints,
                                           compute_hybrid_knn)
    from nfmap import NFMap

    out_dir = os.path.join(RESULTS_ROOT, EXPERIMENT_ID)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "scale_study.json")
    rows = []
    if os.path.exists(out_path):                 # resume rather than refit
        with open(out_path) as f:
            rows = json.load(f)
        print(f"[{EXPERIMENT_ID}] resuming with {len(rows)} existing runs",
              flush=True)
    done = {(r["dataset"], r["kernel"], r["prior_family"], r.get("prior_dof"))
            for r in rows}

    for ds in datasets:
        df = load_target(ds)
        smiles = df["smiles"].tolist()
        print(f"\n[{EXPERIMENT_ID}] {ds}: {len(smiles)} molecules, "
              f"computing fingerprint tiers...", flush=True)
        Xm, Xs, Xk, _, valid = compute_hierarchical_fingerprints(smiles)
        ids, n_scaf = murcko_ids([smiles[i] for i in valid])
        knn_i, knn_d = compute_hybrid_knn(Xm, Xs, Xk, k=15)
        print(f"      {Xm.shape[0]} parsed, {n_scaf} distinct scaffolds", flush=True)

        # Configurations: both kernels under the Gaussian prior, plus the dof
        # sweep under the hierarchical kernel on one dataset.
        configs = [("euclidean", "gaussian", None), ("hybrid", "gaussian", None)]
        if ds == DOF_DATASET and not args.skip_dof:
            configs += [("hybrid", "student_t", d) for d in DOFS]

        for kernel, family, dof in configs:
            key = (ds, kernel, family, dof)
            if key in done:
                print(f"    {kernel:<10} {family:<10} dof={str(dof):<5} "
                      f"(already done, skipped)", flush=True)
                continue
            cfg = {**NFMAP_KWARGS, "prior_family": family}
            if dof is not None:
                cfg["prior_dof"] = dof
            t0 = time.time()
            try:
                m = NFMap(**cfg)
                if kernel == "hybrid":
                    Z = m.fit(Xm, knn_indices=knn_i, knn_dists=knn_d).transform(Xm)
                else:
                    Z = m.fit_transform(Xm)
                purity, sil, chance, n_elig = score(np.asarray(Z), ids)
            except Exception as e:
                print(f"    {kernel} {family} dof={dof}: FAILED ({e})", flush=True)
                continue
            rec = dict(dataset=ds, n_molecules=int(Xm.shape[0]),
                       n_scaffolds=n_scaf, n_eligible=n_elig, kernel=kernel,
                       prior_family=family, prior_dof=dof,
                       scaffold_purity=purity, scaffold_silhouette=sil,
                       scaffold_chance=chance, seed=SEED,
                       fit_seconds=round(time.time() - t0, 1))
            rows.append(rec)
            with open(out_path, "w") as f:
                json.dump(rows, f, indent=2)
            print(f"    {kernel:<10} {family:<10} dof={str(dof):<5} "
                  f"purity={purity:.4f} sil={sil:+.4f} "
                  f"({rec['fit_seconds']:.0f}s)", flush=True)

    print(f"\n  -> {out_path}")


if __name__ == "__main__":
    main()
