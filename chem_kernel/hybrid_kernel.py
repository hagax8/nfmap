"""
hybrid_kernel.py
================
Chemistry-specific kNN graph construction for nfmap: a 3-tier hierarchy
(Whole Molecule + Murcko Scaffold + Generic Skeleton), each a Morgan
fingerprint, combined into one weighted-Tanimoto neighbor graph and fed into
NFMap.fit()'s `knn_indices`/`knn_dists` passthrough.

Deliberately lives OUTSIDE the `nfmap` package (top-level `chem_kernel/`, not
`nfmap/nfmap/`) so RDKit stays a chemistry-project concern, not a core-package
dependency - `nfmap` itself never imports this module.

Phase 1 (this file): plain NumPy/RDKit, brute-force O(N^2) pairwise Tanimoto -
correct and simple, fine at DRD2 scale (~1k compounds). Phase 2 (ChEMBL
scale, 2.5M compounds) is a separate, not-yet-built Numba/packbit-optimized
PyNNDescent metric, to be cross-checked against this file's brute-force
output on a subsample once the design here is validated - see the
implementation plan for sequencing rationale.
"""
from __future__ import annotations

from array import array

import numpy as np
import scipy.sparse as sp
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


def get_morgan_fp(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Dense {0,1} float32 Morgan fingerprint for an already-parsed RDKit Mol."""
    fp = np.zeros((n_bits,), dtype=np.float32)
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    Chem.DataStructs.ConvertToNumpyArray(bitvect, fp)
    return fp


def compute_hierarchical_fingerprints(
    smiles_list: list[str], radius: int = 2, n_bits: int = 2048
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """
    Compute the 3-tier hierarchy fingerprints for a list of SMILES.

    Returns
    -------
    X_mol, X_scaffold, X_skeleton : float32 arrays, shape (n_valid, n_bits)
        Morgan fingerprints of the whole molecule, its Bemis-Murcko scaffold,
        and the scaffold's generic graph framework (atoms->C, bonds->single -
        this is what lets bioisosteric ring swaps, e.g. pyridine <-> pyrimidine,
        land in the same skeleton bucket even though their exact Murcko
        scaffolds differ).
    scaffold_ids : int64 array, shape (n_valid,)
        Integer id per molecule from factorizing canonical Murcko scaffold
        SMILES - i.e. two molecules get the same id iff they share an exact
        Murcko scaffold. This is the `cluster_labels` input for NFMap's EMA
        centroid attraction.
    valid_indices : list[int]
        Indices into the original `smiles_list` that parsed successfully and
        are included (in order) in the arrays above - use this to align any
        parallel metadata (names, activities, ...) the caller is tracking.
    """
    mol_fps, scaffold_fps, skeleton_fps = [], [], []
    scaffold_smiles_list = []
    valid_indices = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
            # A molecule with no ring system (an acyclic scaffold) yields an
            # empty scaffold_mol - MakeScaffoldGeneric on an empty mol still
            # returns a valid (empty) Mol, so this doesn't need
            # special-casing beyond the MolFromSmiles None-check above.
            skeleton_mol = MurckoScaffold.MakeScaffoldGeneric(scaffold_mol)
        except ValueError:
            # RDKit's sanitization can reject the derived scaffold/skeleton
            # even when the original molecule parsed fine (e.g.
            # MakeScaffoldGeneric's atoms->C substitution can, in rare
            # cases, produce a structure with a valence RDKit considers
            # invalid - AtomValenceException et al. all derive from
            # ValueError). Not a hypothetical: this hit real ChEMBL data at
            # ~400k molecules during Phase-2 kNN-graph benchmarking - see
            # planning_docs/chembl_scale_kernel_optimization_plan.md.
            # Skip like an unparseable SMILES rather than crashing the
            # whole batch over one molecule.
            continue

        mol_fps.append(get_morgan_fp(mol, radius, n_bits))
        scaffold_fps.append(get_morgan_fp(scaffold_mol, radius, n_bits))
        skeleton_fps.append(get_morgan_fp(skeleton_mol, radius, n_bits))
        scaffold_smiles_list.append(Chem.MolToSmiles(scaffold_mol))
        valid_indices.append(i)

    X_mol = np.array(mol_fps, dtype=np.float32)
    X_scaffold = np.array(scaffold_fps, dtype=np.float32)
    X_skeleton = np.array(skeleton_fps, dtype=np.float32)
    _, scaffold_ids = np.unique(scaffold_smiles_list, return_inverse=True)
    scaffold_ids = scaffold_ids.astype(np.int64)

    return X_mol, X_scaffold, X_skeleton, scaffold_ids, valid_indices


def compute_hierarchical_fingerprints_sparse(
    smiles_list: list[str], radius: int = 2, n_bits: int = 2048
) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, np.ndarray, list[int]]:
    """
    Sparse-output counterpart to compute_hierarchical_fingerprints - identical
    3-tier hierarchy, identical (scaffold_ids, valid_indices) contract, but
    returns scipy.sparse.csr_matrix fingerprints instead of dense (n_valid,
    n_bits) float32 arrays.

    Why this exists: a Morgan fingerprint is only ~2-5% dense (a few dozen
    set bits out of n_bits=2048), so the dense path's (N, n_bits) float32
    array per tier is 20-50x larger than the actual information content -
    at ChEMBL scale (N~2.2M) that's ~17GB for X_mol ALONE, before
    X_scaffold/X_skeleton, before the model itself - a real OOM, not a
    hypothetical, hit while testing the master-map projection experiment
    (see chembl_scale_kernel_optimization_plan.md). Built from
    ExplicitBitVect.GetOnBits() directly, so no dense (n_valid, n_bits)
    array - not even a transient one - is ever materialized for the whole
    dataset, only per-molecule "which few dozen bits are on" lists.

    NFMap.fit()/transform()'s binary-fingerprint path (TruncatedSVD) accepts
    scipy.sparse input natively - see nfmap.py's is_binary_ branch - so the
    result of this function can be passed directly as `X` to
    NFMap.fit()/fit_transform() with no further conversion. Only the kNN-
    graph step (which needs a packed dense-byte layout for its Tanimoto
    metric) still needs densifying, done in row-chunks by
    hybrid_kernel_fast.compute_hybrid_knn_scale_sparse rather than all at
    once.
    """
    row_mol, col_mol = array("i"), array("i")
    row_scaf, col_scaf = array("i"), array("i")
    row_skel, col_skel = array("i"), array("i")
    scaffold_smiles_list = []
    valid_indices = []
    row = 0

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
            skeleton_mol = MurckoScaffold.MakeScaffoldGeneric(scaffold_mol)
        except ValueError:
            # See compute_hierarchical_fingerprints's matching except clause -
            # same real (not hypothetical) RDKit sanitization edge case.
            continue

        for bit in AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits).GetOnBits():
            row_mol.append(row)
            col_mol.append(bit)
        for bit in AllChem.GetMorganFingerprintAsBitVect(scaffold_mol, radius, nBits=n_bits).GetOnBits():
            row_scaf.append(row)
            col_scaf.append(bit)
        for bit in AllChem.GetMorganFingerprintAsBitVect(skeleton_mol, radius, nBits=n_bits).GetOnBits():
            row_skel.append(row)
            col_skel.append(bit)

        scaffold_smiles_list.append(Chem.MolToSmiles(scaffold_mol))
        valid_indices.append(i)
        row += 1

    n_valid = row

    def _to_csr(rows: array, cols: array) -> sp.csr_matrix:
        rows_np = np.frombuffer(rows, dtype=np.int32)
        cols_np = np.frombuffer(cols, dtype=np.int32)
        data = np.ones(len(rows_np), dtype=np.float32)
        return sp.csr_matrix((data, (rows_np, cols_np)), shape=(n_valid, n_bits), dtype=np.float32)

    X_mol = _to_csr(row_mol, col_mol)
    X_scaffold = _to_csr(row_scaf, col_scaf)
    X_skeleton = _to_csr(row_skel, col_skel)
    _, scaffold_ids = np.unique(scaffold_smiles_list, return_inverse=True)
    scaffold_ids = scaffold_ids.astype(np.int64)

    return X_mol, X_scaffold, X_skeleton, scaffold_ids, valid_indices


def _pairwise_tanimoto_distance(X: np.ndarray) -> np.ndarray:
    """
    Full N x N Tanimoto DISTANCE matrix for a dense binary fingerprint matrix,
    vectorized via matrix multiply (popcount-of-AND = X @ X.T for {0,1} rows) -
    no RDKit BulkTanimotoSimilarity Python loop needed. O(N^2) memory/compute -
    Phase 1 only, fine at DRD2 scale; not intended for ChEMBL-scale N.
    """
    intersection = X @ X.T
    row_sums = X.sum(axis=1)
    union = row_sums[:, None] + row_sums[None, :] - intersection
    # Guards the all-zero-fingerprint case (union=0) rather than dividing by
    # zero, treating it as similarity=0 (distance=1), not similarity=1. Not
    # actually rare at the scaffold/skeleton tier: any acyclic molecule has
    # an empty Murcko scaffold, and that's ~0.7% of a random ChEMBL sample -
    # see hybrid_kernel_fast.py's _tanimoto_dist_packed_range docstring,
    # which deliberately mirrors this same convention for exact parity.
    similarity = np.divide(
        intersection, union, out=np.zeros_like(intersection), where=union > 0
    )
    return 1.0 - similarity


def compute_hybrid_knn(
    X_mol: np.ndarray,
    X_scaffold: np.ndarray,
    X_skeleton: np.ndarray,
    k: int = 15,
    w_mol: float = 0.4,
    w_scaf: float = 0.4,
    w_skel: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Brute-force weighted-Tanimoto kNN graph across the 3 fingerprint tiers.

    Set w_skel=0.0 to get a 2-tier (molecule + scaffold only) graph from the
    same function - used to produce the 2-tier vs. 3-tier comparison maps.

    Returns (knn_indices, knn_dists), both shape (N, k), self excluded -
    matches NFMap.fit()'s knn_indices/knn_dists passthrough contract exactly
    (no self-column; fit() will not strip one).
    """
    N = X_mol.shape[0]
    k = min(k, N - 1)

    weighted_dist = w_mol * _pairwise_tanimoto_distance(X_mol)
    weighted_dist += w_scaf * _pairwise_tanimoto_distance(X_scaffold)
    if w_skel > 0:
        weighted_dist += w_skel * _pairwise_tanimoto_distance(X_skeleton)

    np.fill_diagonal(weighted_dist, np.inf)
    knn_indices = np.argsort(weighted_dist, axis=1)[:, :k]
    knn_dists = np.take_along_axis(weighted_dist, knn_indices, axis=1)
    return knn_indices.astype(np.int64), knn_dists.astype(np.float32)
