"""B13: when we project a NEW compound, does where it lands mean anything?

This is the claim Navifold actually rests on, and the paper never tested it.
B12 (coordinate stability) showed only that a map is REPRODUCIBLE - that the same
compound lands in the same place twice. A map can be perfectly reproducible and
still place molecules arbitrarily. What matters to a chemist is whether a new
compound's map neighbours are its actual analogues.

Protocol. Split into train/test, fit on train only, project both. Then ask, over
every test-train pair, whether distance ON THE MAP tracks chemical distance.

Three design choices that make the result hard to dismiss:

1. Tanimoto is computed on 2048-bit Morgan r=2 fingerprints, while NFMap is
   trained on the 128-bit FP + 8 descriptors the rest of the paper uses. The
   yardstick is therefore a *different, higher-resolution* representation than
   anything the model saw or optimised against. (Separately: the 128-bit input
   is low enough resolution to collide heavily, which may be part of why every
   method looks weak on chemistry in Section 6 - see that section's "feature
   ceiling" discussion. Raising it everywhere is a bigger change than this
   script and is deliberately not done here.)

2. UMAP is the comparison, because it is the only baseline that can project
   out-of-sample at all. t-SNE has no transform(); it is included in the figure
   only, refit on train+test, and labelled - its absence from the table IS the
   point this paper makes about persistent coordinate systems.

3. Two metrics, because they answer different questions. The global Spearman
   over all test-train pairs is dominated by far-apart pairs, which is not what
   anyone uses a map for. Neighbourhood retrieval - of a test compound's k
   nearest neighbours on the map, how many are in its true top-k by Tanimoto -
   is the local, decision-relevant version.

Usage:
    python b13_projection_meaning.py
    python b13_projection_meaning.py --datasets drd2 --seeds 42
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

sys.path.insert(0, ".")
from common.data import load_dataset
from common.io import RESULTS_ROOT
from run_ablation import DEFAULTS

from nfmap import NFMap

EXPERIMENT_ID = "B13"

# Spelled out because "dude" and "egfr" are two different EGFR datasets and the
# distinction matters: DUD-E is actives vs. property-matched synthetic decoys,
# the ChEMBL set is real measured potency across diverse programs.
DATASET_LABELS = {
    "drd2": "DRD2 (ChEMBL)",
    "egfr": "EGFR (ChEMBL)",
    "dude": "DUD-E EGFR (actives/decoys)",
    "bace": "BACE",
    "bbbp": "BBBP",
    "esol": "ESOL",
}
TANIMOTO_NBITS = 2048
# Two yardsticks, and the second one is not optional. The production recipe's
# hybrid kernel builds its molecule tier as Morgan r=2 / 2048 bits - i.e. it is
# trained on a graph derived from EXACTLY the r=2 yardstick, while the paper's
# default NFMap and UMAP only ever see the 128-bit PCA'd features. Scoring the
# production config on r=2 alone would be circular and flatter it for free. r=3
# is a representation no configuration here is trained on, so it is the honest
# comparison; r=2 is kept because it is what the rest of B13 already reports.
TANIMOTO_RADII = (2, 3)
TOPK = 10

# Production recipe, from experiments/chembed_generation (project_target_onto_
# chembl_master.py and build_all_class_maps.py both describe it as the validated
# recipe): student-t prior, dof=2, auto grid bounds, hybrid 3-tier Tanimoto
# graph, hexagonal layout. Split into two variants deliberately - running the
# prior change alone separates which of the two factors is doing the work.
STUDENT_T_KWARGS = dict(prior_family="student_t", prior_dof=2.0,
                        grid_bounds="auto", prior_layout="hexagonal")
HYBRID_K, HYBRID_WEIGHTS = 15, dict(w_mol=0.4, w_scaf=0.4, w_skel=0.2)


def tanimoto_distance_matrix(mols_a, mols_b, radius):
    """1 - Tanimoto on 2048-bit Morgan at the given radius."""
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=TANIMOTO_NBITS)
    A = np.array([gen.GetFingerprintAsNumPy(m) for m in mols_a], dtype=np.float32)
    B = np.array([gen.GetFingerprintAsNumPy(m) for m in mols_b], dtype=np.float32)
    inter = A @ B.T
    union = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    return 1.0 - inter / np.maximum(union, 1e-9)


def retrieval_overlap(d_map, d_chem, k=TOPK):
    """Mean |top-k on the map AND top-k by Tanimoto| / k, per test compound.

    The question a chemist actually asks of a projected compound: are the things
    sitting next to it on the map the things that are genuinely most similar?
    """
    overlaps = []
    for i in range(d_map.shape[0]):
        a = set(np.argsort(d_map[i])[:k])
        b = set(np.argsort(d_chem[i])[:k])
        overlaps.append(len(a & b) / k)
    return float(np.mean(overlaps))


def score_projection(Z_tr, Z_te, d_chem):
    d_map = np.linalg.norm(Z_te[:, None, :] - Z_tr[None, :, :], axis=-1)
    rho, p = spearmanr(d_map.ravel(), d_chem.ravel())
    ret = retrieval_overlap(d_map, d_chem)
    # Chance baseline: k neighbours drawn uniformly from n_train training
    # compounds overlap the true top-k by k/n_train in expectation. Without it
    # a retrieval number is uninterpretable - 0.16 sounds poor but is ~8x chance.
    chance = TOPK / Z_tr.shape[0]
    return dict(spearman=float(rho), p_value=float(p),
                retrieval_at_k=ret, k=TOPK,
                retrieval_chance=float(chance),
                retrieval_enrichment=float(ret / chance))


def _hybrid_knn(mols, idx_tr):
    """Production 3-tier Tanimoto kNN graph over the TRAINING molecules only."""
    from rdkit import Chem
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from chem_kernel import compute_hierarchical_fingerprints, compute_hybrid_knn

    smiles = [Chem.MolToSmiles(mols[i]) for i in idx_tr]
    X_mol, X_scaf, X_skel, _, valid = compute_hierarchical_fingerprints(smiles)
    if len(valid) != len(smiles):
        raise RuntimeError(f"hybrid kernel dropped {len(smiles) - len(valid)} molecules; "
                           "index alignment with X would break")
    return compute_hybrid_knn(X_mol, X_scaf, X_skel, k=HYBRID_K, **HYBRID_WEIGHTS)


def _fit_nfmap(Xtr, seed, extra=None, knn=None):
    cfg = dict(DEFAULTS); cfg["random_state"] = seed
    if extra:
        cfg.update(extra)
    model = NFMap(**cfg)
    if knn is None:
        Z_fit = model.fit_transform(Xtr)
    else:
        Z_fit = model.fit_transform(Xtr, knn_indices=knn[0], knn_dists=knn[1])
    return model, Z_fit


def run_dataset(ds, seeds, out_dir):
    data = load_dataset(ds, return_mols=True)
    X, y, mols = data["X"], data["y"], data["mols"]
    rows, embeddings = [], {}
    for seed in seeds:
        idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed)
        Xtr, Xte = X[idx_tr], X[idx_te]
        mols_te = [mols[i] for i in idx_te]
        mols_tr = [mols[i] for i in idx_tr]
        d_chem = {r: tanimoto_distance_matrix(mols_te, mols_tr, r) for r in TANIMOTO_RADII}

        # Give the baselines the SAME representation NFMap works in. NFMap sees
        # the raw binary fingerprint and reduces it internally with TruncatedSVD
        # to 128 components (production's width); handing UMAP/t-SNE the raw
        # 2048-dim matrix instead compares methods across different input spaces
        # rather than different algorithms. Fit on train, apply to test, exactly
        # as NFMap's internal reduction does.
        svd = TruncatedSVD(n_components=128, random_state=seed).fit(Xtr)
        Xtr_svd, Xte_svd = svd.transform(Xtr), svd.transform(Xte)

        def record(method, Z_tr, Z_te, **extra_fields):
            out = dict(dataset=ds, seed=seed, method=method, **extra_fields)
            for r in TANIMOTO_RADII:
                sc = score_projection(Z_tr, Z_te, d_chem[r])
                out.update({f"{k}_r{r}": v for k, v in sc.items()})
            rows.append(out)
            bits = "  ".join(
                f"r={r}: rho={out[f'spearman_r{r}']:+.3f} "
                f"retr@{TOPK}={out[f'retrieval_at_k_r{r}']:.3f} "
                f"({out[f'retrieval_enrichment_r{r}']:.0f}x)"
                for r in TANIMOTO_RADII)
            print(f"  {ds:<6} seed={seed} {method:<16} {bits}", flush=True)
            return out

        variants = [
            ("nfmap", None, False),                    # paper default
            ("nfmap_student_t", STUDENT_T_KWARGS, False),   # prior change only
            ("nfmap_prod", STUDENT_T_KWARGS, True),    # prior + hybrid kernel
        ]
        knn = None
        Z_tr = Z_te = None
        for name, extra, use_hybrid in variants:
            try:
                if use_hybrid and knn is None:
                    knn = _hybrid_knn(mols, idx_tr)
                model, Z_fit = _fit_nfmap(Xtr, seed, extra, knn if use_hybrid else None)
                Ztr_, Zte_ = model.transform(Xtr), model.transform(Xte)
                drift = float(np.abs(Z_fit - Ztr_).max())
                record(name, Ztr_, Zte_, fit_vs_transform_max_abs_diff=drift)
                if name == "nfmap":
                    Z_tr, Z_te = Ztr_, Zte_
                if seed == seeds[0]:
                    embeddings[f"{name}_train"] = Ztr_
                    embeddings[f"{name}_test"] = Zte_
            except Exception as e:
                print(f"  {ds:<6} seed={seed} {name:<16} FAILED ({e})", flush=True)

        try:
            import umap
            um = umap.UMAP(random_state=seed).fit(Xtr_svd)
            Zu_tr, Zu_te = um.transform(Xtr_svd), um.transform(Xte_svd)
            record("umap", Zu_tr, Zu_te)
            if seed == seeds[0]:
                embeddings["umap_train"], embeddings["umap_test"] = Zu_tr, Zu_te
        except Exception as e:
            print(f"  {ds:<6} seed={seed} umap             FAILED ({e})", flush=True)

        # t-SNE cannot project out-of-sample, so it is refit on train+test. That
        # gives it information the others never get, and it still has no reusable
        # frame - shown for visual context, not scored.
        if seed == seeds[0]:
            try:
                from sklearn.manifold import TSNE
                Zt = TSNE(n_components=2, random_state=seed, init="pca").fit_transform(
                    np.vstack([Xtr_svd, Xte_svd]))
                embeddings["tsne_train"], embeddings["tsne_test"] = Zt[:len(Xtr)], Zt[len(Xtr):]
            except Exception as e:
                print(f"  {ds:<6} seed={seed} tsne             FAILED ({e})", flush=True)

            embeddings["y_train"], embeddings["y_test"] = y[idx_tr], y[idx_te]
            np.savez(os.path.join(out_dir, f"{ds}_embeddings.npz"),
                     **{k: v for k, v in embeddings.items() if v is not None})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="drd2,egfr,dude,bace,bbbp,esol")
    ap.add_argument("--seeds", default="42,0,1")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    out_dir = os.path.join(RESULTS_ROOT, EXPERIMENT_ID)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "projection_meaning.json")

    print(f"[{EXPERIMENT_ID}] does a projected compound land somewhere meaningful? "
          f"(Tanimoto on {TANIMOTO_NBITS}-bit Morgan, radii {TANIMOTO_RADII})")
    rows = []
    for ds in [d.strip() for d in args.datasets.split(",")]:
        rows += run_dataset(ds, seeds, out_dir)
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2)
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
