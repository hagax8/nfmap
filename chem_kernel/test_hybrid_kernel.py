"""
test_hybrid_kernel.py
======================
Correctness checks for chem_kernel's brute-force weighted-Tanimoto kNN graph.
Cross-checks against RDKit's own DataStructs.TanimotoSimilarity (an
independent implementation) rather than re-deriving the same matrix-multiply
formula, so this actually catches a wrong formula, not just a wrong wiring.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs

from hybrid_kernel import (
    compute_hierarchical_fingerprints,
    compute_hybrid_knn,
    get_morgan_fp,
    _pairwise_tanimoto_distance,
)

# A small, hand-picked set: two exact same-scaffold pairs (aspirin-like /
# ibuprofen-like variants), one pyridine/pyrimidine bioisostere pair sharing a
# ring skeleton but not an exact scaffold, and one unrelated molecule.
SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",       # aspirin
    "CC(=O)Oc1ccccc1C(=O)OC",      # aspirin methyl ester (same scaffold)
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "c1ccncc1",                     # pyridine
    "c1ccncn1",                     # pyrimidine (bioisostere of pyridine)
    "CCCCCCCCCC",                   # unrelated acyclic decane
]


def test_pairwise_tanimoto_distance_matches_rdkit():
    mols = [Chem.MolFromSmiles(s) for s in SMILES]
    fps_bitvect = [Chem.AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]
    X = np.array([get_morgan_fp(m) for m in mols], dtype=np.float32)

    D = _pairwise_tanimoto_distance(X)
    assert D.shape == (len(SMILES), len(SMILES))
    assert np.allclose(np.diag(D), 0.0, atol=1e-6), "self-distance should be 0"

    # Cross-check every pair against RDKit's own TanimotoSimilarity.
    for i in range(len(SMILES)):
        for j in range(len(SMILES)):
            expected_dist = 1.0 - DataStructs.TanimotoSimilarity(fps_bitvect[i], fps_bitvect[j])
            assert abs(D[i, j] - expected_dist) < 1e-5, (
                f"mismatch at ({i},{j}): got {D[i, j]:.6f}, expected {expected_dist:.6f}"
            )
    print("pairwise Tanimoto distance matches RDKit's own implementation: OK")


def test_hybrid_knn_contract_and_scaffold_grouping():
    X_mol, X_scaffold, X_skeleton, scaffold_ids, valid_indices = compute_hierarchical_fingerprints(SMILES)
    assert valid_indices == list(range(len(SMILES)))
    assert X_mol.shape == (len(SMILES), 2048)

    # aspirin (0) and its methyl ester (1) share an exact Murcko scaffold.
    assert scaffold_ids[0] == scaffold_ids[1]
    # pyridine (3) and pyrimidine (4) do NOT share an exact Murcko scaffold
    # (they ARE their own scaffolds, and differ by a ring nitrogen).
    assert scaffold_ids[3] != scaffold_ids[4]

    k = 3
    knn_indices, knn_dists = compute_hybrid_knn(X_mol, X_scaffold, X_skeleton, k=k, w_mol=0.4, w_scaf=0.4, w_skel=0.2)
    assert knn_indices.shape == (len(SMILES), k)
    assert knn_dists.shape == (len(SMILES), k)
    # Self-loop contract: no row may contain its own index.
    for i in range(len(SMILES)):
        assert i not in knn_indices[i], f"row {i} contains a self-loop"
    # Nearest neighbor of aspirin (0) should be its methyl ester (1) - same
    # scaffold AND highly similar whole-molecule structure.
    assert knn_indices[0, 0] == 1, f"expected aspirin's nearest neighbor to be its ester, got {knn_indices[0]}"
    print("hybrid kNN self-loop contract and scaffold grouping: OK")


def test_skeleton_tier_pulls_bioisosteres_closer():
    """The concrete claim the skeleton tier is supposed to buy: pyridine and
    pyrimidine (which share a generic ring skeleton but not an exact Murcko
    scaffold) should be relatively closer with w_skel > 0 than with w_skel = 0."""
    X_mol, X_scaffold, X_skeleton, scaffold_ids, _ = compute_hierarchical_fingerprints(SMILES)
    pyridine_idx, pyrimidine_idx = 3, 4

    _, dists_2tier = compute_hybrid_knn(X_mol, X_scaffold, X_skeleton, k=5, w_mol=0.4, w_scaf=0.6, w_skel=0.0)
    _, dists_3tier = compute_hybrid_knn(X_mol, X_scaffold, X_skeleton, k=5, w_mol=0.4, w_scaf=0.4, w_skel=0.2)

    # Recompute raw weighted distance between the pair directly (rather than
    # relying on both appearing within the top-5, which isn't guaranteed on
    # such a tiny/unrelated molecule set) to isolate the skeleton tier's effect.
    dist_2tier_pair = (
        0.4 * _pairwise_tanimoto_distance(X_mol)[pyridine_idx, pyrimidine_idx]
        + 0.6 * _pairwise_tanimoto_distance(X_scaffold)[pyridine_idx, pyrimidine_idx]
    )
    dist_3tier_pair = (
        0.4 * _pairwise_tanimoto_distance(X_mol)[pyridine_idx, pyrimidine_idx]
        + 0.4 * _pairwise_tanimoto_distance(X_scaffold)[pyridine_idx, pyrimidine_idx]
        + 0.2 * _pairwise_tanimoto_distance(X_skeleton)[pyridine_idx, pyrimidine_idx]
    )
    assert dist_3tier_pair < dist_2tier_pair, (
        f"expected skeleton tier to pull pyridine/pyrimidine closer: "
        f"2-tier={dist_2tier_pair:.4f}, 3-tier={dist_3tier_pair:.4f}"
    )
    print(f"skeleton tier pulls bioisosteres closer: 2-tier={dist_2tier_pair:.4f} -> 3-tier={dist_3tier_pair:.4f} OK")


if __name__ == "__main__":
    test_pairwise_tanimoto_distance_matches_rdkit()
    test_hybrid_knn_contract_and_scaffold_grouping()
    test_skeleton_tier_pulls_bioisosteres_closer()
    print("\nAll checks passed.")
