"""B10: Applicability Domain — LMDT vs. established baselines (Novelty #7).

NFMap's "thick manifold" storage vault motivates a companion claim used in
production (Navifold): an applicability-domain / out-of-domain detection
signal derived from the flow itself. Two candidates were explored internally
before this paper: the raw off-manifold residual norm (`manifold_distance =
sqrt(mean(z_perp**2))`, found not to reliably separate in-/out-of-domain
molecules for SVD/PCA-preprocessed pipelines - see nfmap_paper_novelty_notes.md)
and the Local Map Distance Test (LMDT, "inspired by AlphaFold's LDDT" per
experiments/lmdt_ood_test.py's docstring): does a point's k-nearest-neighbor
identity in the *original* input space survive the mapping into 2D, i.e. do
its true neighbors land near it on the map? Two implementations exist: the
exact k-NN computation (ref-LMDT, no training needed) and a small trained
MLP (`lmdt_head`, NFMap's `use_lmdt_head=True`) that approximates the same
quantity from a single forward pass, deployed in Navifold's production ONNX
pipeline because a live k-NN search isn't practical client-side.

This script tests all three LMDT-family variants against kNN-distance in the
*original* input space - one of the most established applicability-domain
baselines in the QSAR literature (see e.g. Jaworska et al. 2005; closely
related to the leverage/Hat-matrix approach used in Gaspar (2015)'s own
GTM-QSAR applicability-domain work, cf. nfmap.py's own cross-reference to
that paper) - via two experiments:

1. Worst-first cumulative-error plot: does the AD score actually predict where
   NFMap's own held-out predictions go wrong? Real pIC50 regression (5-NN on
   the fitted 2D map) on DRD2. Sort the test set from LEAST to MOST
   in-domain (worst AD score first) and, as points are cumulatively added in
   that order, plot RMSE over the growing subset - x-axis runs 0-100%
   (starting with just the worst few, ending with everyone), so a working AD
   score should show RMSE falling as more (better) points are folded in.
2. Permutation-OOD detection: per-feature-shuffled synthetic negatives (same
   construction nfmap.py's own lmdt_head training loss uses to define its
   OOD target) - ROC-AUC for telling real molecules apart from permuted
   ones, per AD score.
3. Real-error ROC-AUC: binarize the 174 real held-out predictions at their
   median |error| (above = "bad", below = "good") and compute a standard
   ROC curve/AUC per AD score for detecting the "bad" half directly - a
   single-number, threshold-free-ish complement to (1) that doesn't require
   picking a coverage level to read off.

Usage:
    python b10_applicability_domain.py --dataset drd2 --seed 42
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import argparse
import json
import sys

import numpy as np
import pandas as pd
import datamol as dm
import torch
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import spearmanr

sys.path.insert(0, ".")
from common.io import run_dir

sys.path.insert(0, "..")
sys.path.insert(0, "../..")
from nfmap import NFMap

K = 10  # neighbors for the reference LMDT computation and the kNN-distance baseline

# Cumulative-inclusion fractions, ascending: worst (highest AD score) points
# added first, so COVERAGES[0] is a small worst-only subset and COVERAGES[-1]
# is everyone. Named to match the "how much of the test set is included so
# far" reading, not the old best-first "coverage retained" one.
COVERAGES = np.round(np.arange(0.03, 1.01, 0.05), 2)


def get_features(mol):
    # Raw 2048-bit binary Morgan r=2, matching production (and common/data.py):
    # staying binary routes NFMap to its TruncatedSVD-to-128 path. Descriptors
    # are no longer concatenated - prod passes them separately, not as X.
    return dm.to_fp(mol, fpSize=2048, as_array=True).astype(np.float32)


def load_drd2_continuous():
    """DRD2 with continuous pIC50 target (not the pIC50>=7.0 binarization
    A2/B2 use) - same feature pipeline (raw 2048-bit binary Morgan FP) as the rest of Paper 1's DRD2 handling."""
    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(here, "..", "..", "experiments", "chembed_generation", "drd2_clean.csv")
    df = pd.read_csv(csv_path)
    X_list, y_list = [], []
    for _, row in df.iterrows():
        mol = dm.to_mol(row["smiles"])
        if mol is not None:
            X_list.append(get_features(mol))
            y_list.append(float(row["pIC50"]))
    X_full = np.array(X_list, dtype=np.float32)
    y_full = np.array(y_list, dtype=np.float32)
    # Not standardised: see get_features above.
    return X_full, y_full


def to_x_scaled(model, X_raw):
    """Replicate NFMap.transform()'s preprocessing chain up to (but not
    including) the flow itself - same pattern as b1_logdet_scaling.py's
    compute_jacobian_column_norms."""
    if model.is_binary_:
        X_pca = model.pca_.transform(X_raw)
    else:
        X_dense_scaled = model.scaler_pre_.transform(np.asarray(X_raw, dtype=np.float32))
        X_pca = model.pca_.transform(X_dense_scaled)
    return (X_pca / model.global_scale_).astype(np.float32)


def lmdt_head_predict(model, X_raw):
    X_scaled = to_x_scaled(model, X_raw)
    model.module_.eval()
    with torch.no_grad():
        x_t = torch.tensor(X_scaled, dtype=torch.float32, device=model.device_)
        pred = model.module_.lmdt_head(x_t).squeeze(-1)
    return pred.cpu().numpy()


def permute_features(X, rng):
    """Per-feature independent shuffle across the batch - identical
    construction to nfmap.py's own lmdt_head OOD training target."""
    X_ood = X.copy()
    for i in range(X_ood.shape[1]):
        X_ood[:, i] = X_ood[rng.permutation(len(X_ood)), i]
    return X_ood


def worst_first_curve(ad_scores, rmse_per_point, fractions):
    """Sort worst (highest AD score, least in-domain) first, then cumulate:
    at each fraction f, RMSE over the worst f*N points added so far. f=0's
    neighborhood is a small, worst-only subset; f=1.0 is everyone (equal to
    the overall RMSE, same endpoint the old best-first curve started from)."""
    order = np.argsort(ad_scores)[::-1]  # least in-domain (highest score) first
    sorted_err = rmse_per_point[order]
    out = []
    for frac in fractions:
        n_keep = max(int(round(frac * len(rmse_per_point))), 5)
        kept = sorted_err[:n_keep]
        out.append(float(np.sqrt(np.mean(kept ** 2))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="drd2", choices=["drd2"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[B10] Loading {args.dataset} (continuous target)...")
    X_std, y_full = load_drd2_continuous()
    print(f"[B10] {len(y_full)} molecules, target range [{y_full.min():.2f}, {y_full.max():.2f}]")

    X_train, X_test, y_train, y_test = train_test_split(
        X_std, y_full, test_size=0.2, random_state=args.seed
    )
    print(f"[B10] train={len(y_train)} test={len(y_test)}")

    print("[B10] Fitting NFMap (use_lmdt_head=True)...")
    model = NFMap(
        n_components=2, use_canonical=False, complexity="medium", inject_weight=0.5,
        n_prior_components=100, prior_weight=0.1, graph_weight=5.0, repulsion_weight=1.0,
        prior_layout="hexagonal", grid_bounds=6.0, random_state=args.seed,
        use_lmdt_head=True,
    )
    Z_train = model.fit_transform(X_train)
    Z_test = model.transform(X_test)

    reg = KNeighborsRegressor(n_neighbors=5).fit(Z_train, y_train)
    y_pred = reg.predict(Z_test)
    per_point_err = np.abs(y_pred - y_test)
    overall_rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
    print(f"[B10] Overall held-out RMSE: {overall_rmse:.4f} (n={len(y_test)})")

    print(f"[B10] Computing AD scores (k={K})...")
    knn_input = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(X_train)
    knn_dist, nbr_idx = knn_input.kneighbors(X_test)
    ref_lmdt_scores = np.array([
        np.mean(np.linalg.norm(Z_train[nbr_idx[i]] - Z_test[i], axis=1))
        for i in range(len(X_test))
    ])
    knn_baseline_scores = knn_dist.mean(axis=1)
    head_lmdt_scores = lmdt_head_predict(model, X_test)

    # Head-vs-ref agreement: LMDT's own design target is "does this point's
    # true neighborhood survive the mapping" (map distortion / position
    # trust), computed exactly by ref-LMDT. head-LMDT is trained to
    # approximate that same quantity in one forward pass. The faithful,
    # on-target validation of LMDT-as-designed is therefore "does the fast
    # deployed head actually recover what the expensive k-NN computation
    # says", NOT "does either one predict downstream regression error" -
    # that second question is a distinct, extended claim, tested separately
    # below. ref-LMDT's role here is ground truth, not a third competing AD
    # candidate.
    print("[B10] Head-vs-ref LMDT agreement (on-target validation)...")
    head_ref_pearson = float(np.corrcoef(head_lmdt_scores, ref_lmdt_scores)[0, 1])
    head_ref_spearman, head_ref_spearman_p = spearmanr(head_lmdt_scores, ref_lmdt_scores)
    print(f"[B10] head-LMDT vs ref-LMDT: Pearson r={head_ref_pearson:.4f} "
          f"Spearman rho={head_ref_spearman:.4f} (p={head_ref_spearman_p:.2e})")

    rmse_per_point = np.abs(y_pred - y_test)
    head_curve = worst_first_curve(head_lmdt_scores, rmse_per_point, COVERAGES)
    ref_curve = worst_first_curve(ref_lmdt_scores, rmse_per_point, COVERAGES)
    knn_curve = worst_first_curve(knn_baseline_scores, rmse_per_point, COVERAGES)

    head_rho, head_p = spearmanr(head_lmdt_scores, per_point_err)
    ref_rho, ref_p = spearmanr(ref_lmdt_scores, per_point_err)
    knn_rho, knn_p = spearmanr(knn_baseline_scores, per_point_err)
    print(f"[B10] Spearman(AD score, |error|): head-LMDT={head_rho:.4f} ref-LMDT={ref_rho:.4f} kNN={knn_rho:.4f}")

    # Real-error ROC-AUC: binarize the 174 real predictions at their median
    # |error| ("bad" = above median) and score each AD variant as a
    # classifier for that split directly - complements the cumulative curve
    # with a single, coverage-level-free number.
    print("[B10] Real-error ROC-AUC (median-split)...")
    bad_label = (per_point_err > np.median(per_point_err)).astype(int)
    head_fpr, head_tpr, _ = roc_curve(bad_label, head_lmdt_scores)
    ref_fpr, ref_tpr, _ = roc_curve(bad_label, ref_lmdt_scores)
    knn_fpr, knn_tpr, _ = roc_curve(bad_label, knn_baseline_scores)
    head_realerror_auc = float(roc_auc_score(bad_label, head_lmdt_scores))
    ref_realerror_auc = float(roc_auc_score(bad_label, ref_lmdt_scores))
    knn_realerror_auc = float(roc_auc_score(bad_label, knn_baseline_scores))
    print(f"[B10] Real-error ROC-AUC: head-LMDT={head_realerror_auc:.4f} "
          f"ref-LMDT={ref_realerror_auc:.4f} kNN={knn_realerror_auc:.4f}")

    print("[B10] Permutation-OOD detection...")
    rng = np.random.RandomState(args.seed)
    X_test_ood = permute_features(X_test, rng)
    head_scores_ood = lmdt_head_predict(model, X_test_ood)
    knn_dist_ood, nbr_idx_ood = knn_input.kneighbors(X_test_ood)
    Z_test_ood = model.transform(X_test_ood)
    ref_scores_ood = np.array([
        np.mean(np.linalg.norm(Z_train[nbr_idx_ood[i]] - Z_test_ood[i], axis=1))
        for i in range(len(X_test_ood))
    ])
    knn_scores_ood = knn_dist_ood.mean(axis=1)

    y_true_ood = np.concatenate([np.zeros(len(X_test)), np.ones(len(X_test_ood))])
    head_auc = float(roc_auc_score(y_true_ood, np.concatenate([head_lmdt_scores, head_scores_ood])))
    ref_auc = float(roc_auc_score(y_true_ood, np.concatenate([ref_lmdt_scores, ref_scores_ood])))
    knn_auc = float(roc_auc_score(y_true_ood, np.concatenate([knn_baseline_scores, knn_scores_ood])))
    print(f"[B10] Permutation-OOD ROC-AUC: head-LMDT={head_auc:.4f} ref-LMDT={ref_auc:.4f} kNN={knn_auc:.4f}")

    out_dir = run_dir("B10", args.dataset, "ad_study")
    config = dict(
        dataset=args.dataset, seed=args.seed, k=K, n_train=len(y_train), n_test=len(y_test),
        use_lmdt_head=True, lmdt_p95=float(getattr(model, "lmdt_p95_", float("nan"))),
    )
    metrics = dict(
        overall_rmse=overall_rmse,
        head_ref_pearson=head_ref_pearson,
        head_ref_spearman=float(head_ref_spearman), head_ref_spearman_p=float(head_ref_spearman_p),
        head_spearman=float(head_rho), head_spearman_p=float(head_p),
        ref_spearman=float(ref_rho), ref_spearman_p=float(ref_p),
        knn_spearman=float(knn_rho), knn_spearman_p=float(knn_p),
        head_perm_ood_auc=head_auc, ref_perm_ood_auc=ref_auc, knn_perm_ood_auc=knn_auc,
        head_realerror_auc=head_realerror_auc, ref_realerror_auc=ref_realerror_auc,
        knn_realerror_auc=knn_realerror_auc,
    )
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez(
        os.path.join(out_dir, "ad_results.npz"),
        coverages=COVERAGES, head_curve=head_curve, ref_curve=ref_curve, knn_curve=knn_curve,
        head_scores_id=head_lmdt_scores, ref_scores_id=ref_lmdt_scores, knn_scores_id=knn_baseline_scores,
        head_scores_ood=head_scores_ood, ref_scores_ood=ref_scores_ood, knn_scores_ood=knn_scores_ood,
        per_point_err=per_point_err,
        head_fpr=head_fpr, head_tpr=head_tpr, ref_fpr=ref_fpr, ref_tpr=ref_tpr,
        knn_fpr=knn_fpr, knn_tpr=knn_tpr,
    )
    print(f"[B10] Saved -> {out_dir}")


if __name__ == "__main__":
    main()
