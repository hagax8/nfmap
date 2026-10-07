"""
test_hybrid_kernel_fast.py
============================
Correctness checks for hybrid_kernel_fast's packed-bit Numba metric and
PyNNDescent-backed kNN graph builder, cross-checked against
hybrid_kernel.py's brute-force reference rather than just "does it run" -
per planning_docs/chembl_scale_kernel_optimization_plan.md item 2, bit
packing is an easy place to introduce silent bugs (padding, POPCOUNT
correctness, off-by-one byte offsets).

Includes the empty-Murcko-scaffold (acyclic molecule) edge case explicitly:
a first draft of hybrid_tanimoto_packed returned distance=0 for an
empty-vs-empty fingerprint pair, which this exact test caught as a mismatch
against the brute-force reference (which defines that case as similarity=0,
i.e. distance=1 - see hybrid_kernel._pairwise_tanimoto_distance). Decane in
the SMILES set below exists specifically to keep exercising this case.
"""
import numpy as np
from rdkit import Chem

from hybrid_kernel import (
    compute_hierarchical_fingerprints,
    _pairwise_tanimoto_distance,
)
from hybrid_kernel_fast import (
    pack_fingerprints,
    pack_hybrid_rows,
    hybrid_tanimoto_packed,
    compute_hybrid_knn_scale,
)

# Same hand-picked set as test_hybrid_kernel.py, plus decane again (already
# present) which is what exercises the empty-scaffold edge case here.
SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",       # aspirin
    "CC(=O)Oc1ccccc1C(=O)OC",      # aspirin methyl ester (same scaffold)
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "c1ccncc1",                     # pyridine
    "c1ccncn1",                     # pyrimidine (bioisostere of pyridine)
    "CCCCCCCCCC",                   # unrelated acyclic decane (empty Murcko scaffold)
]

W = dict(w_mol=0.4, w_scaf=0.4, w_skel=0.2)


def _reference_weighted_dist(X_mol, X_scaffold, X_skeleton, w_mol, w_scaf, w_skel):
    D = w_mol * _pairwise_tanimoto_distance(X_mol)
    D += w_scaf * _pairwise_tanimoto_distance(X_scaffold)
    if w_skel > 0:
        D += w_skel * _pairwise_tanimoto_distance(X_skeleton)
    return D


def test_packed_metric_matches_bruteforce_reference_exactly():
    """Every pairwise packed-bit distance, including the diagonal (which
    exercises the empty-scaffold zero-union edge case for decane), must
    match the brute-force float reference to float32 precision."""
    X_mol, X_scaffold, X_skeleton, scaffold_ids, _ = compute_hierarchical_fingerprints(SMILES)
    D_ref = _reference_weighted_dist(X_mol, X_scaffold, X_skeleton, **W)

    P = pack_hybrid_rows(X_mol, X_scaffold, X_skeleton)
    n_bytes = X_mol.shape[1] // 8
    N = len(SMILES)
    D_packed = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D_packed[i, j] = hybrid_tanimoto_packed(
                P[i], P[j],
                n_bytes_mol=n_bytes, n_bytes_scaf=n_bytes, n_bytes_skel=n_bytes,
                **W,
            )

    max_diff = np.abs(D_packed - D_ref).max()
    assert max_diff < 1e-5, f"packed metric diverges from brute-force reference by {max_diff}"
    # Explicitly assert the edge case, not just averaged away in max_diff:
    # decane (index 5) vs itself must be 0.6 (0.4*0 + 0.4*1 + 0.2*1), not 0.0.
    decane = SMILES.index("CCCCCCCCCC")
    assert abs(D_ref[decane, decane] - 0.6) < 1e-5, "reference sanity check on the edge case itself"
    assert abs(D_packed[decane, decane] - 0.6) < 1e-5, (
        "packed metric must reproduce the empty-scaffold zero-union convention "
        "(similarity=0, not 1, for two empty fingerprints), not just match on typical pairs"
    )
    print(f"packed metric matches brute-force reference exactly (max diff {max_diff:.2e}), "
          f"including the empty-scaffold edge case: OK")


def test_2tier_matches_when_w_skel_zero():
    """w_skel=0 must produce the same distances as omitting the skeleton
    tier entirely - checks the early-exit branch in hybrid_tanimoto_packed,
    not just the w_skel>0 path."""
    X_mol, X_scaffold, X_skeleton, _, _ = compute_hierarchical_fingerprints(SMILES)
    D_ref = _reference_weighted_dist(X_mol, X_scaffold, X_skeleton, w_mol=0.4, w_scaf=0.6, w_skel=0.0)

    P = pack_hybrid_rows(X_mol, X_scaffold, X_skeleton)
    n_bytes = X_mol.shape[1] // 8
    N = len(SMILES)
    D_packed = np.array([
        [
            hybrid_tanimoto_packed(
                P[i], P[j],
                n_bytes_mol=n_bytes, n_bytes_scaf=n_bytes, n_bytes_skel=n_bytes,
                w_mol=0.4, w_scaf=0.6, w_skel=0.0,
            )
            for j in range(N)
        ]
        for i in range(N)
    ])
    assert np.abs(D_packed - D_ref).max() < 1e-5
    print("w_skel=0 early-exit branch matches reference: OK")


def test_pack_fingerprints_rejects_non_byte_aligned_width():
    try:
        pack_fingerprints(np.zeros((5, 10), dtype=np.float32))
        raise AssertionError("expected ValueError for n_bits not a multiple of 8")
    except ValueError as e:
        print("non-byte-aligned width correctly rejected:", str(e)[:80])


def test_compute_hybrid_knn_scale_contract_and_scaffold_grouping():
    """Same behavioral contract as hybrid_kernel.compute_hybrid_knn's own
    test: no self-loops, aspirin's nearest neighbor is its methyl ester.
    At N=6, PyNNDescent's approximate search should recover the exact
    answer (dataset is far smaller than its own default n_neighbors search
    heuristics need to start losing recall)."""
    X_mol, X_scaffold, X_skeleton, scaffold_ids, valid_indices = compute_hierarchical_fingerprints(SMILES)
    assert valid_indices == list(range(len(SMILES)))

    k = 3
    knn_indices, knn_dists = compute_hybrid_knn_scale(
        X_mol, X_scaffold, X_skeleton, k=k, **W, random_state=0
    )
    assert knn_indices.shape == (len(SMILES), k)
    assert knn_dists.shape == (len(SMILES), k)
    for i in range(len(SMILES)):
        assert i not in knn_indices[i], f"row {i} contains a self-loop"
    assert knn_indices[0, 0] == 1, (
        f"expected aspirin's nearest neighbor to be its ester, got {knn_indices[0]}"
    )
    print("compute_hybrid_knn_scale contract and scaffold grouping: OK")


def test_compute_hybrid_knn_scale_matches_bruteforce_at_moderate_n():
    """At N=2000 (still small enough to brute-force as ground truth),
    PyNNDescent's approximate graph should recover almost all of the exact
    weighted-Tanimoto kNN edges - a recall check, not bit-for-bit identity,
    since NNDescent is approximate by construction."""
    from hybrid_kernel import compute_hybrid_knn as compute_hybrid_knn_bruteforce

    rng = np.random.default_rng(0)
    n_bits = 256  # smaller than the real 2048 purely to keep this test fast
    N = 2000
    # Synthetic but realistic: sparse-ish random binary fingerprints with a
    # handful of shared "scaffold" clusters, matching the real fingerprints'
    # sparsity (Morgan FPs are sparse, not ~50% density).
    n_clusters = 40
    cluster_centers = (rng.random((n_clusters, n_bits)) < 0.08).astype(np.float32)
    cluster_ids = rng.integers(0, n_clusters, size=N)
    flip_mask = rng.random((N, n_bits)) < 0.03
    X_mol = np.logical_xor(cluster_centers[cluster_ids], flip_mask).astype(np.float32)
    X_scaffold = np.logical_xor(cluster_centers[cluster_ids], rng.random((N, n_bits)) < 0.01).astype(np.float32)
    X_skeleton = X_scaffold.copy()

    k = 15
    knn_idx_ref, knn_dist_ref = compute_hybrid_knn_bruteforce(X_mol, X_scaffold, X_skeleton, k=k, **W)
    knn_idx_fast, knn_dist_fast = compute_hybrid_knn_scale(
        X_mol, X_scaffold, X_skeleton, k=k, **W, random_state=0
    )

    recall = np.mean([
        len(set(knn_idx_ref[i]) & set(knn_idx_fast[i])) / k
        for i in range(N)
    ])
    print(f"approximate-vs-brute-force kNN recall at N={N}: {recall:.4f}")
    assert recall > 0.9, f"expected >90% recall against brute-force ground truth, got {recall:.4f}"


if __name__ == "__main__":
    test_packed_metric_matches_bruteforce_reference_exactly()
    test_2tier_matches_when_w_skel_zero()
    test_pack_fingerprints_rejects_non_byte_aligned_width()
    test_compute_hybrid_knn_scale_contract_and_scaffold_grouping()
    test_compute_hybrid_knn_scale_matches_bruteforce_at_moderate_n()
    print("\nAll checks passed.")
