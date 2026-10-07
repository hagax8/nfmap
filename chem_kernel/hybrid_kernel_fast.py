"""
hybrid_kernel_fast.py
======================
Phase 2 (ChEMBL scale, ~2.5-3M molecules): packed-bit Numba weighted-Tanimoto
metric fed directly into PyNNDescent's approximate kNN graph construction -
the scale-out of hybrid_kernel.py's O(N^2) brute-force reference, which is
documented as fine only up to roughly 10-15k molecules.

Same 3-tier hierarchy (whole molecule + Murcko scaffold + generic skeleton
Morgan fingerprints) and the same weighted-Tanimoto formula as
hybrid_kernel.compute_hybrid_knn - compute_hierarchical_fingerprints() is
still the upstream step. This module only replaces the O(N^2)
distance-matrix + argsort with:

  1. Bit-packing each tier's dense {0,1} float32 fingerprint into uint8
     (np.packbits) - lets POPCOUNT do the Tanimoto intersection/union in
     one pass over bytes instead of one multiply-add per bit (8x fewer
     elements per row: 2048 bits -> 256 bytes per tier). NOTE the memory
     win is 8x, not the 32x a naive "float32 -> packed uint8" comparison
     suggests: PyNNDescent's own check_array forces ANY custom-callable
     metric's data array to float32 before storing it internally (its
     uint8 fast path is hardcoded to the literal metric names
     "bit_hamming"/"bit_jaccard", which a Python-level custom metric never
     matches - see hybrid_tanimoto_packed's docstring), so what's actually
     resident in memory is the packed byte VALUES stored as float32, not
     as uint8. Still an 8x reduction in row width (768 float32 values vs.
     6144 float32 values for the 3 tiers unpacked) and an 8x reduction in
     per-comparison work, just not the full 32x bit-packing implies on
     paper.
  2. A single @numba.njit popcount-table metric operating on the
     concatenated [mol|scaffold|skeleton] packed row, with tier weights
     passed through PyNNDescent's metric_kwds - verified (see
     planning_docs/chembl_scale_kernel_optimization_plan.md, item 1) that
     this actually reaches the compiled function, including its Python-level
     defaults when metric_kwds is omitted, rather than being silently
     ignored.
  3. pynndescent.NNDescent for the approximate kNN search itself -
     O(N log N)-ish instead of O(N^2).

Correctness: cross-checked against hybrid_kernel.py's brute-force reference
in test_hybrid_kernel_fast.py, including the zero-fingerprint (acyclic
molecule -> empty Murcko scaffold) edge case - see that test module's
docstring for why this edge case is not "practically unreachable" the way
hybrid_kernel.py's own comment assumes.
"""
from __future__ import annotations

from typing import Optional

import numba
import numpy as np
from pynndescent import NNDescent

# Population-count lookup table for a single byte (0-255 -> number of set
# bits). Referenced as a closed-over module global inside the njit
# functions below - numba compiles module-level numpy arrays in as a
# read-only constant, confirmed to work under nopython mode.
_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def pack_fingerprints(X: np.ndarray) -> np.ndarray:
    """Bit-pack a dense {0,1} fingerprint matrix (N, n_bits) into
    (N, n_bits // 8) uint8 - lossless for Tanimoto, since AND/OR/popcount
    only need the bits, not a float per bit."""
    if X.shape[1] % 8 != 0:
        raise ValueError(
            f"n_bits must be a multiple of 8 to bit-pack losslessly, got {X.shape[1]}"
        )
    return np.packbits(X.astype(np.uint8), axis=1)


def pack_hybrid_rows(
    X_mol: np.ndarray, X_scaffold: np.ndarray, X_skeleton: np.ndarray
) -> np.ndarray:
    """Bit-pack and concatenate the 3 fingerprint tiers into the single
    (N, n_bytes_mol + n_bytes_scaf + n_bytes_skel) uint8 array
    hybrid_tanimoto_packed expects as its per-row input."""
    return np.concatenate(
        [pack_fingerprints(X_mol), pack_fingerprints(X_scaffold), pack_fingerprints(X_skeleton)],
        axis=1,
    )


@numba.njit(fastmath=True, cache=True)
def _popcount_and_or(x, y, start, end):
    """(popcount(x&y), popcount(x|y)) over packed bytes x/y[start:end] -
    the two ingredients of a Tanimoto similarity, kept separate so the
    caller can apply the zero-union convention explicitly.

    x/y elements are cast to uint8 per-element before the bitwise ops
    rather than assumed to already be uint8. This isn't defensive
    programming for its own sake: PyNNDescent's own check_array forces
    ANY custom callable metric's input array to float32 (its uint8 fast
    path is hardcoded to the literal metric names "bit_hamming"/
    "bit_jaccard", which a Python-level custom metric never matches) -
    confirmed by hitting a "no implementation of and_(float32, float32)"
    TypingError with the naive uint8-only version. Packed byte values
    (0-255) survive that float32 round-trip exactly (float32 represents
    all integers below 2**24 exactly), so casting back per-element here
    recovers the exact bit pattern regardless of which dtype the caller
    (PyNNDescent internals, or a direct uint8 test call) hands in."""
    inter = 0
    union = 0
    for i in range(start, end):
        a = np.uint8(x[i])
        b = np.uint8(y[i])
        inter += _POPCOUNT_TABLE[a & b]
        union += _POPCOUNT_TABLE[a | b]
    return inter, union


@numba.njit(fastmath=True, cache=True)
def _tanimoto_dist_packed_range(x, y, start, end):
    """One tier's Tanimoto distance over packed bytes x/y[start:end].
    Matches hybrid_kernel._pairwise_tanimoto_distance's zero-union
    convention exactly: an empty-vs-empty fingerprint pair (e.g. two
    acyclic molecules' empty Murcko-scaffold fingerprints) gets
    similarity=0 -> distance=1, not similarity=1 -> distance=0. This isn't
    a rare corner case - roughly 0.7% of a random ChEMBL sample has an
    empty Murcko scaffold (any acyclic molecule), so it matters at
    ChEMBL scale even though it's cosmetically "practically unreachable"
    for the whole-molecule tier alone."""
    inter, union = _popcount_and_or(x, y, start, end)
    if union == 0:
        return 1.0
    return 1.0 - inter / union


@numba.njit(fastmath=True, cache=True)
def hybrid_tanimoto_packed(
    x, y,
    n_bytes_mol=256, n_bytes_scaf=256, n_bytes_skel=256,
    w_mol=0.4, w_scaf=0.4, w_skel=0.2,
):
    """PyNNDescent-compatible metric(x, y) -> float over a row that's the
    byte-concatenation [mol_bytes | scaffold_bytes | skeleton_bytes]
    produced by pack_hybrid_rows. Tier weights and byte-widths are passed
    through NNDescent's metric_kwds; confirmed (see planning doc) this
    reaches the compiled function correctly, defaults included."""
    d = 0.0
    if w_mol > 0.0:
        d += w_mol * _tanimoto_dist_packed_range(x, y, 0, n_bytes_mol)
    if w_scaf > 0.0:
        d += w_scaf * _tanimoto_dist_packed_range(
            x, y, n_bytes_mol, n_bytes_mol + n_bytes_scaf
        )
    if w_skel > 0.0:
        d += w_skel * _tanimoto_dist_packed_range(
            x, y, n_bytes_mol + n_bytes_scaf, n_bytes_mol + n_bytes_scaf + n_bytes_skel
        )
    return d


def compute_hybrid_knn_scale(
    X_mol: np.ndarray,
    X_scaffold: np.ndarray,
    X_skeleton: np.ndarray,
    k: int = 15,
    w_mol: float = 0.4,
    w_scaf: float = 0.4,
    w_skel: float = 0.2,
    n_jobs: int = -1,
    random_state: Optional[int] = 0,
    verbose: bool = False,
    **nndescent_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ChEMBL-scale (approximate) weighted-Tanimoto kNN graph across the 3
    fingerprint tiers - the Phase 2 replacement for
    hybrid_kernel.compute_hybrid_knn's O(N^2) brute force.

    Same weighted-Tanimoto formula and (knn_indices, knn_dists) contract as
    compute_hybrid_knn (both shape (N, k), self excluded, matching
    NFMap.fit()'s knn_indices/knn_dists passthrough) - a drop-in replacement
    past the ~10-15k molecule scale where brute force stops being
    practical. Approximate (PyNNDescent), not exact - see
    test_hybrid_kernel_fast.py for the recall bound this was validated
    against.

    Set w_skel=0.0 for a 2-tier (molecule + scaffold only) graph, matching
    compute_hybrid_knn's convention.
    """
    N = X_mol.shape[0]
    k = min(k, N - 1)
    n_bytes_mol = X_mol.shape[1] // 8
    n_bytes_scaf = X_scaffold.shape[1] // 8
    n_bytes_skel = X_skeleton.shape[1] // 8

    X_packed = pack_hybrid_rows(X_mol, X_scaffold, X_skeleton)

    index = NNDescent(
        X_packed,
        metric=hybrid_tanimoto_packed,
        metric_kwds=dict(
            n_bytes_mol=n_bytes_mol, n_bytes_scaf=n_bytes_scaf, n_bytes_skel=n_bytes_skel,
            w_mol=w_mol, w_scaf=w_scaf, w_skel=w_skel,
        ),
        n_neighbors=k + 1,  # +1: self is (almost always) its own nearest neighbor; stripped below
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
        **nndescent_kwargs,
    )
    knn_indices, knn_dists = index.neighbor_graph
    return _strip_self(knn_indices, knn_dists, k)


def _sparse_tier_to_packed(X_sparse, chunk_size: int = 20000) -> np.ndarray:
    """Bit-pack a scipy.sparse tier into (N, n_bits // 8) uint8, densifying
    only chunk_size rows at a time so peak transient memory is bounded by
    chunk_size regardless of N - see pack_hybrid_rows_sparse's docstring for
    why the tier arrives sparse rather than already dense."""
    N, n_bits = X_sparse.shape
    if n_bits % 8 != 0:
        raise ValueError(
            f"n_bits must be a multiple of 8 to bit-pack losslessly, got {n_bits}"
        )
    X_sparse = X_sparse.tocsr()
    out = np.empty((N, n_bits // 8), dtype=np.uint8)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        dense_chunk = np.asarray(X_sparse[start:end].toarray(), dtype=np.uint8)
        out[start:end] = np.packbits(dense_chunk, axis=1)
    return out


def pack_hybrid_rows_sparse(
    X_mol, X_scaffold, X_skeleton, chunk_size: int = 20000
) -> np.ndarray:
    """Sparse-input counterpart to pack_hybrid_rows - identical
    (N, n_bytes_mol + n_bytes_scaf + n_bytes_skel) uint8 output, but built by
    densifying each tier chunk_size rows at a time instead of requiring the
    caller to already hold a fully dense (N, n_bits) float32 array per tier.

    That upfront densification is the actual ChEMBL-scale memory ceiling -
    not the kNN search itself (see compute_hierarchical_fingerprints_sparse's
    docstring in hybrid_kernel.py) - so this is what lets
    compute_hybrid_knn_scale_sparse take compute_hierarchical_fingerprints_sparse's
    output directly."""
    return np.concatenate(
        [
            _sparse_tier_to_packed(X_mol, chunk_size),
            _sparse_tier_to_packed(X_scaffold, chunk_size),
            _sparse_tier_to_packed(X_skeleton, chunk_size),
        ],
        axis=1,
    )


def compute_hybrid_knn_scale_sparse(
    X_mol,
    X_scaffold,
    X_skeleton,
    k: int = 15,
    w_mol: float = 0.4,
    w_scaf: float = 0.4,
    w_skel: float = 0.2,
    n_jobs: int = -1,
    random_state: Optional[int] = 0,
    verbose: bool = False,
    chunk_size: int = 20000,
    **nndescent_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sparse-input counterpart to compute_hybrid_knn_scale - identical
    (knn_indices, knn_dists) contract and identical weighted-Tanimoto metric,
    but accepts X_mol/X_scaffold/X_skeleton as scipy.sparse matrices (e.g.
    from hybrid_kernel.compute_hierarchical_fingerprints_sparse) and
    bit-packs them in row-chunks (via pack_hybrid_rows_sparse) instead of
    requiring the caller to already hold a fully dense (N, n_bits) float32
    array per tier - use this one, not compute_hybrid_knn_scale, at
    ChEMBL-scale N where that upfront densification is the real memory
    ceiling (see planning_docs/chembl_scale_kernel_optimization_plan.md).
    """
    N = X_mol.shape[0]
    k = min(k, N - 1)
    n_bytes_mol = X_mol.shape[1] // 8
    n_bytes_scaf = X_scaffold.shape[1] // 8
    n_bytes_skel = X_skeleton.shape[1] // 8

    X_packed = pack_hybrid_rows_sparse(X_mol, X_scaffold, X_skeleton, chunk_size=chunk_size)

    index = NNDescent(
        X_packed,
        metric=hybrid_tanimoto_packed,
        metric_kwds=dict(
            n_bytes_mol=n_bytes_mol, n_bytes_scaf=n_bytes_scaf, n_bytes_skel=n_bytes_skel,
            w_mol=w_mol, w_scaf=w_scaf, w_skel=w_skel,
        ),
        n_neighbors=k + 1,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
        **nndescent_kwargs,
    )
    knn_indices, knn_dists = index.neighbor_graph
    return _strip_self(knn_indices, knn_dists, k)


def _strip_self(knn_indices: np.ndarray, knn_dists: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Drop each row's self-hit (matched by index equality, not by
    position - duplicate/near-duplicate fingerprints are common enough in
    ChEMBL that self isn't reliably column 0). If NNDescent's approximate
    search missed the self-hit for a row (rare), drops the farthest column
    instead so every row still comes out at exactly k neighbors. Fully
    vectorized (no per-row Python loop) to stay cheap at N in the millions.
    """
    N, k_plus_1 = knn_indices.shape
    row_ids = np.arange(N)
    self_mask = knn_indices == row_ids[:, None]
    has_self = self_mask.any(axis=1)
    self_col = np.argmax(self_mask, axis=1)
    drop_col = np.where(has_self, self_col, k_plus_1 - 1)
    cols = np.broadcast_to(np.arange(k_plus_1), knn_indices.shape)
    keep_mask = cols != drop_col[:, None]
    out_indices = knn_indices[keep_mask].reshape(N, k_plus_1 - 1)[:, :k]
    out_dists = knn_dists[keep_mask].reshape(N, k_plus_1 - 1)[:, :k]
    return out_indices.astype(np.int64), out_dists.astype(np.float32)
