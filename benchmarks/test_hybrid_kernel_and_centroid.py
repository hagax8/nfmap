"""
test_hybrid_kernel_and_centroid.py
===================================
Correctness checks for the two mechanisms added to support chemistry-aware
(scaffold-clustered) maps:

1. Precomputed knn_indices/knn_dists passthrough on fit()/fit_transform() -
   lets an external, domain-specific kernel (e.g. chem_kernel's hybrid
   Tanimoto graph) define the neighborhood graph instead of nfmap's internal
   NNDescent/X_struct path.
2. EMA per-cluster centroid attraction (cluster_labels + centroid_weight) -
   pulls same-cluster points together in z_parallel via a running centroid,
   independent of whether same-cluster points co-occur in a minibatch.
3. warm_start_from - reuses a previously-fitted NFMap's PCA/scaler basis and
   module_ weights as the starting point for a new fit() call on a different
   (typically larger) X, for curriculum-style progressive training.
"""
import numpy as np
import torch
from scipy.spatial.distance import cdist
from nfmap import NFMap


def test_precomputed_knn_passthrough():
    rng = np.random.default_rng(0)
    N = 120
    X = rng.normal(size=(N, 16)).astype(np.float32)

    k = 5
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)
    knn_indices = np.argsort(D, axis=1)[:, :k]
    knn_dists = np.take_along_axis(D, knn_indices, axis=1)

    model = NFMap(n_components=2, max_epochs=5, verbose=False, random_state=0)
    Z = model.fit_transform(X, knn_indices=knn_indices, knn_dists=knn_dists)
    assert Z.shape == (N, 2)
    print("precomputed knn passthrough: OK", Z.shape)

    # Shape mismatch between knn_indices/knn_dists must raise, not silently misalign.
    try:
        NFMap(n_components=2, max_epochs=2, random_state=0).fit(
            X, knn_indices=knn_indices[:, :3], knn_dists=knn_dists
        )
        raise AssertionError("expected ValueError for mismatched knn_indices/knn_dists shapes")
    except ValueError as e:
        print("shape mismatch correctly raises ValueError:", str(e)[:80])


def test_ema_centroid_tightens_clusters():
    rng = np.random.default_rng(1)
    N = 150
    n_clusters = 5
    cluster_labels = rng.integers(0, n_clusters, size=N)
    centers = rng.normal(scale=4.0, size=(n_clusters, 16))
    X = (centers[cluster_labels] + rng.normal(scale=0.5, size=(N, 16))).astype(np.float32)

    model = NFMap(
        n_components=2, max_epochs=60, verbose=False, random_state=0,
        graph_weight=5.0, centroid_weight=2.0, centroid_momentum=0.9,
    )
    Z = model.fit_transform(X, cluster_labels=cluster_labels)
    assert Z.shape == (N, 2)
    assert hasattr(model, "cluster_centroids_")
    assert model.cluster_centroids_.shape == (n_clusters, 2)
    print("EMA centroid mechanism ran OK. classes_:", model.classes_)

    # Same data/seed with centroid_weight=0 should produce looser same-cluster spread.
    model_off = NFMap(
        n_components=2, max_epochs=60, verbose=False, random_state=0,
        graph_weight=5.0, centroid_weight=0.0,
    )
    Z_off = model_off.fit_transform(X, cluster_labels=cluster_labels)

    def mean_within_cluster_spread(Z, labels):
        spreads = []
        for c in np.unique(labels):
            pts = Z[labels == c]
            spreads.append(np.mean(np.sum((pts - pts.mean(0)) ** 2, axis=1)))
        return np.mean(spreads)

    spread_with = mean_within_cluster_spread(Z, cluster_labels)
    spread_without = mean_within_cluster_spread(Z_off, cluster_labels)
    print(f"within-cluster spread WITH centroid pull:    {spread_with:.4f}")
    print(f"within-cluster spread WITHOUT centroid pull: {spread_without:.4f}")
    assert spread_with < spread_without, "expected centroid attraction to tighten clusters"
    print("Tightening effect confirmed: OK")


def test_ema_centroid_handles_singleton_cluster():
    """A cluster with exactly one member must not crash the lazy-init path
    (centroid[c] is set directly on first sight, with no prior EMA blend)."""
    rng = np.random.default_rng(2)
    N = 40
    cluster_labels = np.zeros(N, dtype=int)
    cluster_labels[0] = 1  # a singleton cluster
    X = rng.normal(size=(N, 8)).astype(np.float32)

    model = NFMap(
        n_components=2, max_epochs=5, verbose=False, random_state=0,
        graph_weight=5.0, centroid_weight=1.0,
    )
    Z = model.fit_transform(X, cluster_labels=cluster_labels)
    assert not np.isnan(Z).any()
    print("singleton cluster handled without NaNs: OK")


def test_warm_start_from():
    rng = np.random.default_rng(3)
    X1 = (rng.random((60, 32)) > 0.7).astype(np.float32)
    X2 = (rng.random((150, 32)) > 0.7).astype(np.float32)

    m1 = NFMap(n_components=2, complexity="low", max_epochs=5, n_prior_components=10, random_state=0, verbose=False)
    m1.fit_transform(X1)

    m2 = NFMap(n_components=2, complexity="low", max_epochs=5, n_prior_components=10, random_state=0, verbose=False)
    Z2 = m2.fit_transform(X2, warm_start_from=m1)
    assert Z2.shape == (150, 2) and not np.isnan(Z2).any()
    assert m2.pca_ is m1.pca_, "pca_ should be reused (same object), not refit"
    assert m1.global_scale_ == m2.global_scale_
    print("warm_start_from: PCA/scaler reuse OK", Z2.shape)

    # max_epochs=0 -> module_ should be an exact, untouched copy of the source weights.
    m3 = NFMap(n_components=2, complexity="low", max_epochs=0, n_prior_components=10, random_state=0, verbose=False)
    m3.fit_transform(X2, warm_start_from=m1)
    sd1, sd3 = m1.module_.state_dict(), m3.module_.state_dict()
    assert all(torch.equal(sd3[k], sd1[k]) for k in sd1), "max_epochs=0 warm-start should exactly match source weights"
    print("warm_start_from: weights loaded exactly at max_epochs=0: OK")

    # 'auto' n_prior_components/grid_bounds must inherit the source's resolved value,
    # not re-derive a different one from the new (larger) N.
    m4 = NFMap(n_components=2, complexity="low", max_epochs=3, n_prior_components="auto", random_state=0, verbose=False)
    m4.fit_transform(X1)
    m5 = NFMap(n_components=2, complexity="low", max_epochs=3, n_prior_components="auto", random_state=0, verbose=False)
    m5.fit_transform(X2, warm_start_from=m4)
    assert m5.resolved_n_prior_components_ == m4.resolved_n_prior_components_
    print("warm_start_from: 'auto' n_prior_components inherited correctly: OK")

    # Explicit (non-'auto') architecture mismatch must raise clearly, not silently override.
    m6 = NFMap(n_components=2, complexity="low", max_epochs=3, n_prior_components=20, random_state=0, verbose=False)
    try:
        m6.fit_transform(X2, warm_start_from=m1)
        raise AssertionError("expected ValueError for mismatched n_prior_components")
    except ValueError as e:
        print("warm_start_from: explicit architecture mismatch correctly raises:", str(e)[:80])

    # warm_start_from an unfitted model must raise immediately.
    try:
        NFMap(n_components=2, max_epochs=3, random_state=0).fit_transform(
            X2, warm_start_from=NFMap(n_components=2)
        )
        raise AssertionError("expected ValueError for unfitted warm_start_from")
    except ValueError as e:
        print("warm_start_from: unfitted source correctly raises:", str(e)[:80])


if __name__ == "__main__":
    test_precomputed_knn_passthrough()
    test_ema_centroid_tightens_clusters()
    test_ema_centroid_handles_singleton_cluster()
    test_warm_start_from()
    print("\nAll checks passed.")
