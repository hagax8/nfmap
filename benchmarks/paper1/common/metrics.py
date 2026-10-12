"""Shared dimensionality-reduction metrics for Paper 1 experiments.

Extracted from benchmarks/nfmap_diagnostics.py so every experiment script (and
that exploratory diagnostics script itself) computes metrics the same way —
one source of truth, not N slightly-different reimplementations.
"""
import numpy as np
from scipy.spatial.distance import pdist, cdist
from scipy.stats import pearsonr
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


def _kth_neighbor_dist(pts, k=5):
    D = cdist(pts, pts)
    D.sort(axis=1)
    return D[:, k]


def compute_dr_metrics(X, Z, y, is_continuous=False):
    """Compute the standard NFMap-paper metric suite for one embedding.

    Parameters
    ----------
    X : (N, D) array, original feature space
    Z : (N, 2) array, 2D embedding (NaN entries treated as a failed run)
    y : (N,) array, class labels (discrete) or a continuous property
    is_continuous : if True, silhouette is skipped (undefined for a continuous
        target) and KNN accuracy becomes KNN regression R^2

    Returns
    -------
    dict with keys: trust, cont, t_c_f1, dist_corr, dens_corr, sil, acc, auc
    NaN-filled if Z contains NaNs (i.e. the underlying method failed/was unavailable).
    """
    if Z is None or np.any(np.isnan(Z)):
        keys = ["trust", "cont", "t_c_f1", "dist_corr", "dens_corr", "sil", "acc", "auc"]
        return {k: float("nan") for k in keys}

    trust = trustworthiness(X, Z)
    cont = trustworthiness(Z, X)
    t_c_f1 = 2 * (trust * cont) / (trust + cont) if (trust + cont) > 0 else 0.0

    dist_corr, _ = pearsonr(pdist(X), pdist(Z))
    dens_corr, _ = pearsonr(
        np.log(_kth_neighbor_dist(X, 5) + 1e-8),
        np.log(_kth_neighbor_dist(Z, 5) + 1e-8),
    )

    if not is_continuous:
        sil = silhouette_score(Z, y)
        knn = KNeighborsClassifier(n_neighbors=5, algorithm="brute").fit(Z, y)
        acc = knn.score(Z, y)
        # AUC via KNN's soft cluster-assignment scores on the embedding - see
        # the matching comment in nfmap_diagnostics.py for why this stands in
        # for "GMM cluster assignment score" (the GMM prior itself isn't
        # label-aware, so it can't directly discriminate true classes).
        try:
            proba = knn.predict_proba(Z)
            if proba.shape[1] == 2:
                auc = roc_auc_score(y, proba[:, 1])
            else:
                auc = roc_auc_score(y, proba, multi_class="ovr", average="macro")
        except ValueError:
            auc = float("nan")
    else:
        sil = float("nan")
        knn = KNeighborsRegressor(n_neighbors=5, algorithm="brute").fit(Z, y)
        acc = knn.score(Z, y)
        auc = float("nan")

    return {
        "trust": float(trust),
        "cont": float(cont),
        "t_c_f1": float(t_c_f1),
        "dist_corr": float(dist_corr),
        "dens_corr": float(dens_corr),
        "sil": float(sil),
        "acc": float(acc),
        "auc": float(auc),
    }


def compute_grid_containment(Z, grid_bounds):
    """How much of a trained embedding actually sits inside the visible GMM grid.

    NFMap's grid is a soft boundary, not a hard cage (unlike GTM's convex-hull
    guarantee) — some spillover is an intentional, documented feature (see
    nfmap_paper_experiments.md's "NFMap vs. GTM: Handling Outliers" note).
    This just measures how much spillover actually happens at a given config,
    since "some" and "60%" are very different claims and both start out
    looking the same on a raw scatter plot.

    Returns NaN if Z is invalid (failed run) or grid_bounds isn't a plain
    number (e.g. grid_bounds='auto' recorded before resolution).
    """
    if Z is None or getattr(Z, "dtype", None) == object or np.any(np.isnan(Z)):
        return {"frac_outside_grid_bounds": float("nan"), "max_abs_coord": float("nan")}
    try:
        gb = float(grid_bounds)
    except (TypeError, ValueError):
        return {"frac_outside_grid_bounds": float("nan"), "max_abs_coord": float("nan")}

    per_point_radius = np.max(np.abs(Z), axis=1)
    return {
        "frac_outside_grid_bounds": float(np.mean(per_point_radius > gb)),
        "max_abs_coord": float(np.max(np.abs(Z))),
    }


def fit_reference_baseline(kind, X, seed=42):
    """Fit a non-NFMap reference method (t-SNE / UMAP / GTM) for comparison.

    Returns Z (N,2), filled with NaN if the optional dependency isn't installed.
    """
    if kind == "tsne":
        from sklearn import manifold
        return manifold.TSNE(n_components=2, init="pca", random_state=seed, perplexity=30).fit_transform(X)

    if kind == "umap":
        try:
            import umap as umap_lib
        except ImportError:
            return np.full((len(X), 2), np.nan)
        return umap_lib.UMAP(n_neighbors=15, min_dist=0.1, random_state=seed).fit_transform(X)

    if kind == "gtm":
        try:
            from ugtm import eGTM
        except ImportError:
            return np.full((len(X), 2), np.nan)
        return eGTM(m=6).fit_transform(X)

    raise ValueError(f"Unknown reference baseline kind: {kind!r}")


K_PURITY = 5          # neighbourhood size for scaffold purity


# ── Scaffold locality ────────────────────────────────────────────────────────
# Generic neighbourhood measures saturate on chemical datasets (T/C F1 sits at
# 0.82-0.93 for every method), so they cannot discriminate. These ask the
# chemistry question instead: do molecules sharing a Bemis-Murcko scaffold stay
# together on the map?

def murcko_ids(smiles):
    """Integer id per molecule; equal iff identical Bemis-Murcko scaffold."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    out, lookup = [], {}
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out.append(-1)
            continue
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except Exception:
            out.append(-1)
            continue
        out.append(lookup.setdefault(scaf, len(lookup)))
    return np.asarray(out), len(lookup)


def score(Z, ids, k=K_PURITY):
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import silhouette_score
    valid = ids >= 0
    Z, ids = Z[valid], ids[valid]
    counts = {s: int((ids == s).sum()) for s in np.unique(ids)}
    eligible = np.array([counts[s] >= 2 for s in ids])
    if eligible.sum() < 2:
        return float("nan"), float("nan"), float("nan"), 0
    kk = min(k, len(Z) - 1)
    _, idx = NearestNeighbors(n_neighbors=kk + 1).fit(Z).kneighbors(Z)
    share = ids[idx[:, 1:]] == ids[:, None]
    purity = float(share[eligible].mean())
    n = len(ids)
    chance = float(np.mean([(counts[s] - 1) / (n - 1) for s in ids[eligible]]))
    # Silhouette needs >=2 distinct labels among the points it scores.
    Zs, ls = Z[eligible], ids[eligible]
    sil = (float(silhouette_score(Zs, ls)) if len(np.unique(ls)) >= 2
           else float("nan"))
    return purity, sil, chance, int(eligible.sum())


# ── Tears ────────────────────────────────────────────────────────────────────
# Mirror image of B2F's fold_rate. A fold puts far-apart points close on the
# map; a tear puts input-space neighbours far apart. Geodesic DistCor scores an
# intact-but-curled sheet (swiss roll at sigma_perp=1000, geo 0.45) about the
# same as one torn into reordered strips (sigma_perp=3, geo 0.47), because the
# all-pairs Pearson is dominated by long distances.

def tear_rate(X, Z, k=10, frac=0.10):
    """Fraction of each point's k input-space neighbours that are NOT among its
    frac*N nearest map neighbours, averaged over points. Rank-based, so local
    stretching does not count, only neighbours sent to another part of the map.
    A single global distance cutoff did not work: it could not separate the
    swiss roll's torn strips from its intact spiral (seed 42, grid 12). This
    does: torn 0.030-0.036, intact 0.010, unrolled s_curve 0.000. Folds also
    raise it (~0.02), so read it next to fold_rate."""
    from sklearn.neighbors import NearestNeighbors
    R = max(k + 1, int(round(frac * len(X))))
    _, ix = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    _, iz = NearestNeighbors(n_neighbors=R + 1).fit(Z).kneighbors(Z)
    hit = (ix[:, 1:, None] == iz[:, None, 1:]).any(axis=2)
    return float(1.0 - hit.mean())


# ── Geodesic structure (3-D manifolds) ───────────────────────────────────────
# Moved here from b2f_geodesic_unrolling.py (not part of the paper) so B22/B23
# do not import an experiment script.

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
