"""
NFMap Diagnostics — Digits Dataset
=====================================
Systematically tests hypotheses about what prevents NFMap from producing
UMAP-style clusters on the 64-D Digits dataset.

Each panel isolates one variable while holding everything else at the default.
t-SNE and UMAP are included as reference baselines.

Usage:
    python nfmap_diagnostics.py

Output:
    nfmap_diagnostics.png  — 2×4 panel comparison figure
    Silhouette scores printed to stdout + summary table at the end.
"""

import warnings
warnings.filterwarnings('ignore')

import os
# These pinned every BLAS/OpenMP pool to a single thread to work around macOS
# segfaults on scikit-learn/UMAP/Numba import. That workaround is no longer
# needed - the segfaults came from a conflicting TensorFlow install (removed
# 2026-10-03), and UMAP/t-SNE/GTM now import and run cleanly without it.
#
# Pinning them here was ALSO actively harmful, because it is set at import time
# and so applies to the NFMap fits too, while the ablation scripts (B2/B2D/B2E/
# B4C/B4D/B3B, ~1500 fits) run with the BLAS pools unpinned. The two code paths
# were therefore measuring in different numerical environments, and NFMap's
# optimisation turns out to be sensitive to the resulting change in float
# reduction order: on DRD2 at sigma_perp=0.003/grid_bounds=12, identical data,
# parameters and seed give DistCor 0.4206 single-threaded vs 0.5833 unpinned.
# That is a 0.16 swing from floating-point associativity alone, i.e. Table 1 and
# the ablations were not comparable.
#
# Set NFMAP_PIN_THREADS=1 to restore the old behaviour (e.g. if the macOS
# segfaults reappear on another machine); leave it unset for the paper's runs.
if os.environ.get("NFMAP_PIN_THREADS") == "1":
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[_v] = "1"
    os.environ["TBB_CXX_TYPE"] = "clang"

import time
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, manifold
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.manifold import trustworthiness
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr

try:
    import sys
    sys.modules['tensorflow'] = None
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("UMAP not installed — its panel will show a placeholder.")

from nfmap import NFMap

try:
    from ugtm import eGTM
    HAS_GTM = True
except ImportError:
    HAS_GTM = False
    print("ugtm not installed — GTM panel will show a placeholder.")

# ── Arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="NFMap Diagnostics")
parser.add_argument("--dataset", type=str, default="digits", choices=["digits", "wine", "cancer", "iris", "moons", "circles", "swiss_roll", "s_curve", "severed_sphere", "bbbp", "bace", "esol", "distinct_chems", "dude", "drd2", "egfr"], help="Dataset to evaluate on")
parser.add_argument("--seed", type=int, default=42,
                    help="Fit seed. Threaded into every method's random_state (NFMap, "
                         "t-SNE, UMAP, GTM) and into the saved run's label, so three "
                         "invocations with different --seed give three independent fits "
                         "of the SAME sample rather than overwriting each other. The "
                         "data subsetting itself stays at 42 - the question the paper "
                         "asks of these numbers is run-to-run variance of the fit, not "
                         "of the sample.")
parser.add_argument("--save_latent", action="store_true", help="Save the full D-dimensional latent representations for NFMap to .npy files")
parser.add_argument("--use_struct", action="store_true", help="Pass structural fingerprints to NFMap as X_struct")
parser.add_argument("--use_props", action="store_true", help="Pass physical properties to NFMap as y_props")
parser.add_argument("--save-results", action="store_true", help="Persist each config's config/metrics/embeddings "
                     "under paper1/results/A1/<dataset>_<kind>/, per nfmap_paper_experiments.md's Part A1 "
                     "reproducibility policy. Off by default — does not change any existing behavior.")
parser.add_argument("--experiment-id", default="A1", help="Experiment id to save results under (default A1; "
                     "A2 for chemistry-benchmark datasets, per nfmap_paper_experiments.md).")
parser.add_argument("--prior-layout", type=str, default=None, choices=["hexagonal", "square"],
                     help="Override DEFAULTS' prior_layout ('hexagonal') for this run only, e.g. 'square' "
                          "for manifold-unrolling datasets where a rectangular grid may better match the "
                          "unrolled sheet's natural topology (see nfmap_paper_experiments.md A1).")
parser.add_argument("--grid-bounds", type=float, default=None,
                     help="Override NFMap's grid_bounds for this run only. Needed because "
                          "grid_bounds and sigma_perp are NOT independent: squeezing z_perp "
                          "pushes variance into z_par, which then needs canvas to land on "
                          "(B2D). The shipped 'auto' resolves from N alone (3.0 at N=600), "
                          "which is indexed on the wrong quantity - B2D's chemistry optimum "
                          "is 12.0.")
parser.add_argument("--nfmap-variants", type=str, default=None,
                     help="Fit several NFMap configurations in one invocation, as a "
                          "comma-separated list of name=sigma_perp pairs, e.g. "
                          "'local=10.0,global=0.003'. Each becomes its own method with "
                          "kind 'nfmap_<name>', so the variants get separate result "
                          "directories instead of overwriting each other. Fitting them "
                          "together (rather than via repeated invocations) also means the "
                          "GTM/t-SNE/UMAP baselines are fit ONCE and shared, so every NFMap "
                          "variant is compared against numerically identical baselines. "
                          "Default (unset) keeps the single-NFMap behaviour.")
parser.add_argument("--sigma-perp", type=float, default=None,
                     help="Override NFMap's sigma_perp (default 10.0) for this run only. For genuinely "
                          "low-intrinsic-dimensionality inputs (e.g. 3D manifold datasets, where z_perp "
                          "ends up with only 1 dimension after PCA/scaling) a small value may help by "
                          "denying the flow an easy escape-hatch dimension to leak real 2D structure into "
                          "(see nfmap_paper_experiments.md A1's prior_layout note and B1's related finding).")
args = parser.parse_args()

X_struct_subset = None
y_props_subset = None

# ── Data Loading ──────────────────────────────────────────────────────────────
np.random.seed(42)

def get_features(mol):
    import datamol as dm
    import numpy as np
    from rdkit.Chem import Descriptors
    # 2048-bit Morgan r=2 + 8 descriptors. The descriptors are kept on purpose:
    # Euclidean distance on a fingerprint alone concentrates (pairwise distance
    # std/mean ~0.135 on BACE), which makes DistCor meaningless; adding them
    # roughly doubles that spread. See common/data.py's _get_features.
    fp = dm.to_fp(mol, fpSize=2048, as_array=True).astype(np.float32)
    desc = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.RingCount(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol)
    ]
    return np.concatenate([fp, desc]).astype(np.float32), fp, desc

if args.dataset == "digits":
    digits = datasets.load_digits(n_class=10)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=min(len(digits.target)-1, 600), random_state=42)
    train_idx, _ = next(sss.split(digits.data, digits.target))
    X_subset = digits.data[train_idx].astype(np.float32)
    y_subset = digits.target[train_idx]
    X_scaled = X_subset / 16.0
elif args.dataset == "bbbp":
    try:
        import pandas as pd
        import datamol as dm
    except ImportError:
        print("Please install pandas and datamol to run the chemistry dataset: pip install pandas datamol rdkit")
        exit(1)
        
    print("Downloading MoleculeNet BBBP (Blood-Brain Barrier Penetration) dataset...")
    df = pd.read_csv("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv")
    
    # Process smiles into molecules and compute Morgan fingerprints (2048 bits)
    X_list = []
    y_list = []
    fp_list = []
    desc_list = []
    for idx, row in df.iterrows():
        mol = dm.to_mol(row["smiles"])
        if mol is not None:
            feat, fp, desc = get_features(mol)
            X_list.append(feat)
            fp_list.append(fp)
            desc_list.append(desc)
            y_list.append(row["p_np"])
            
    X_full = np.array(X_list, dtype=np.float32)
    y_full = np.array(y_list, dtype=np.int32)
    fp_full = np.array(fp_list, dtype=np.float32)
    desc_full = np.array(desc_list, dtype=np.float32)
    
    # Subsample to keep the diagnostic script fast
    sss = StratifiedShuffleSplit(n_splits=1, train_size=min(len(y_full)-1, 600), random_state=42)
    train_idx, _ = next(sss.split(X_full, y_full))
    X_subset = X_full[train_idx]
    y_subset = y_full[train_idx]
    X_struct_subset = fp_full[train_idx]
    y_props_subset = desc_full[train_idx]
    y_props_subset = (y_props_subset - np.mean(y_props_subset, axis=0)) / (np.std(y_props_subset, axis=0) + 1e-8)
    
    # Standardize before PCA so MolWt doesn't dominate
    # Standardise -> 32 components -> standardise, so every method (NFMap, GTM,
    # t-SNE, UMAP) consumes the identical, well-scaled matrix. Reducing here
    # rather than inside NFMap is what keeps the comparison about algorithms
    # instead of input spaces.
    from sklearn.decomposition import PCA
    X_subset = (X_subset - X_subset.mean(0)) / (X_subset.std(0) + 1e-8)
    X_subset = PCA(n_components=32, random_state=42).fit_transform(X_subset)
    X_subset = ((X_subset - X_subset.mean(0)) / (X_subset.std(0) + 1e-8)).astype(np.float32)
    
    # Standard Scaling (Mean=0, Std=1) for Normalizing Flow
    X_scaled = (X_subset - np.mean(X_subset, axis=0)) / (np.std(X_subset, axis=0) + 1e-8)
elif args.dataset == "bace":
    try:
        import pandas as pd
        import datamol as dm
    except ImportError:
        print("Please install pandas and datamol to run the chemistry dataset: pip install pandas datamol rdkit")
        exit(1)
        
    print("Downloading MoleculeNet BACE (Classification) dataset...")
    df = pd.read_csv("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv")
    
    # Process smiles into molecules and compute Morgan fingerprints (2048 bits)
    X_list = []
    y_list = []
    fp_list = []
    desc_list = []
    for idx, row in df.iterrows():
        mol = dm.to_mol(row["mol"]) # In BACE, the SMILES column is named "mol"
        if mol is not None:
            feat, fp, desc = get_features(mol)
            X_list.append(feat)
            fp_list.append(fp)
            desc_list.append(desc)
            y_list.append(row["Class"]) # Target is binary Classification
            
    X_full = np.array(X_list, dtype=np.float32)
    y_full = np.array(y_list, dtype=np.int32)
    fp_full = np.array(fp_list, dtype=np.float32)
    desc_full = np.array(desc_list, dtype=np.float32)
    
    # Subsample to keep the diagnostic script fast
    sss = StratifiedShuffleSplit(n_splits=1, train_size=min(len(y_full)-1, 600), random_state=42)
    train_idx, _ = next(sss.split(X_full, y_full))
    X_subset = X_full[train_idx]
    y_subset = y_full[train_idx]
    X_struct_subset = fp_full[train_idx]
    y_props_subset = desc_full[train_idx]
    y_props_subset = (y_props_subset - np.mean(y_props_subset, axis=0)) / (np.std(y_props_subset, axis=0) + 1e-8)
    
    # Standardize before PCA so MolWt doesn't dominate
    # Standardise -> 32 components -> standardise, so every method (NFMap, GTM,
    # t-SNE, UMAP) consumes the identical, well-scaled matrix. Reducing here
    # rather than inside NFMap is what keeps the comparison about algorithms
    # instead of input spaces.
    from sklearn.decomposition import PCA
    X_subset = (X_subset - X_subset.mean(0)) / (X_subset.std(0) + 1e-8)
    X_subset = PCA(n_components=32, random_state=42).fit_transform(X_subset)
    X_subset = ((X_subset - X_subset.mean(0)) / (X_subset.std(0) + 1e-8)).astype(np.float32)
    
    # Standard Scaling (Mean=0, Std=1) for Normalizing Flow
    X_scaled = (X_subset - np.mean(X_subset, axis=0)) / (np.std(X_subset, axis=0) + 1e-8)
elif args.dataset == "esol":
    try:
        import pandas as pd
        import datamol as dm
    except ImportError:
        print("Please install pandas and datamol to run the chemistry dataset: pip install pandas datamol rdkit")
        exit(1)
        
    print("Downloading ESOL (Solubility) dataset...")
    df = pd.read_csv("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv")
    
    X_list = []
    y_list = []
    fp_list = []
    desc_list = []
    for idx, row in df.iterrows():
        mol = dm.to_mol(row["smiles"]) 
        if mol is not None:
            feat, fp, desc = get_features(mol)
            X_list.append(feat)
            fp_list.append(fp)
            desc_list.append(desc)
            sol = row['measured log solubility in mols per litre']
            # Convert regression into 3 distinct classes
            if sol < -4:
                y_list.append(0) # Insoluble
            elif sol < -2:
                y_list.append(1) # Moderate
            else:
                y_list.append(2) # Soluble
            
    X_full = np.array(X_list, dtype=np.float32)
    y_full = np.array(y_list, dtype=np.int32)
    fp_full = np.array(fp_list, dtype=np.float32)
    desc_full = np.array(desc_list, dtype=np.float32)
    
    # Subsample to keep the diagnostic script fast
    sss = StratifiedShuffleSplit(n_splits=1, train_size=min(len(y_full)-1, 600), random_state=42)
    train_idx, _ = next(sss.split(X_full, y_full))
    X_subset = X_full[train_idx]
    y_subset = y_full[train_idx]
    X_struct_subset = fp_full[train_idx]
    y_props_subset = desc_full[train_idx]
    y_props_subset = (y_props_subset - np.mean(y_props_subset, axis=0)) / (np.std(y_props_subset, axis=0) + 1e-8)
    
    # Standardize before PCA so MolWt doesn't dominate
    # Standardise -> 32 components -> standardise, so every method (NFMap, GTM,
    # t-SNE, UMAP) consumes the identical, well-scaled matrix. Reducing here
    # rather than inside NFMap is what keeps the comparison about algorithms
    # instead of input spaces.
    from sklearn.decomposition import PCA
    X_subset = (X_subset - X_subset.mean(0)) / (X_subset.std(0) + 1e-8)
    X_subset = PCA(n_components=32, random_state=42).fit_transform(X_subset)
    X_subset = ((X_subset - X_subset.mean(0)) / (X_subset.std(0) + 1e-8)).astype(np.float32)

    # Standard Scaling (Mean=0, Std=1) for Normalizing Flow
    X_scaled = (X_subset - np.mean(X_subset, axis=0)) / (np.std(X_subset, axis=0) + 1e-8)
elif args.dataset in ("drd2", "egfr"):
    # ChEMBL target-activity sets (distinct from the DUD-E EGFR actives/decoys
    # set above) - real potency data, not a MoleculeNet benchmark. Local CSVs
    # already fetched via experiments/chembed_generation/fetch_chembl_target_data.py
    # (see planning_docs/how_to_generate_and_evaluate_target_maps.md).
    #
    # Label: active/inactive via pIC50 >= 7.0 (IC50 < 100nM, a standard QSAR
    # threshold), NOT Murcko scaffold id. An earlier version used scaffold id
    # (hundreds of mostly-singleton classes) - dropped by design decision:
    # it broke macro-averaged AUC (trivially inflated by singleton classes)
    # and gave a much harder, less interpretable task than every other A2
    # dataset (BBBP/BACE/DUD-E are all binary active/inactive already).
    # pIC50>=7.0 gives a well-balanced split for both targets (44% active for
    # DRD2, 50% for EGFR).
    try:
        import pandas as pd
        import datamol as dm
    except ImportError:
        print("Please install pandas and datamol to run this dataset.")
        exit(1)

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                             "experiments", "chembed_generation", f"{args.dataset}_clean.csv")
    print(f"Loading {args.dataset.upper()} ChEMBL target set from {csv_path}...")
    df = pd.read_csv(csv_path)

    ACTIVE_THRESHOLD = 7.0
    X_list, y_list, fp_list, desc_list = [], [], [], []
    for _, row in df.iterrows():
        mol = dm.to_mol(row["smiles"])
        if mol is not None:
            feat, fp, desc = get_features(mol)
            X_list.append(feat)
            fp_list.append(fp)
            desc_list.append(desc)
            y_list.append(1 if row["pIC50"] >= ACTIVE_THRESHOLD else 0)

    X_full = np.array(X_list, dtype=np.float32)
    y_full = np.array(y_list, dtype=np.int32)
    fp_full = np.array(fp_list, dtype=np.float32)
    desc_full = np.array(desc_list, dtype=np.float32)
    print(f"{len(y_full)} molecules, {y_full.mean()*100:.1f}% active (pIC50 >= {ACTIVE_THRESHOLD})")

    sss = StratifiedShuffleSplit(n_splits=1, train_size=min(len(y_full) - 1, 600), random_state=42)
    train_idx, _ = next(sss.split(X_full, y_full))
    X_subset = X_full[train_idx]
    y_subset = y_full[train_idx]
    X_struct_subset = fp_full[train_idx]
    y_props_subset = desc_full[train_idx]
    y_props_subset = (y_props_subset - np.mean(y_props_subset, axis=0)) / (np.std(y_props_subset, axis=0) + 1e-8)

    # Standardise -> 32 components -> standardise, so every method (NFMap, GTM,
    # t-SNE, UMAP) consumes the identical, well-scaled matrix. Reducing here
    # rather than inside NFMap is what keeps the comparison about algorithms
    # instead of input spaces.
    from sklearn.decomposition import PCA
    X_subset = (X_subset - X_subset.mean(0)) / (X_subset.std(0) + 1e-8)
    X_subset = PCA(n_components=32, random_state=42).fit_transform(X_subset)
    X_subset = ((X_subset - X_subset.mean(0)) / (X_subset.std(0) + 1e-8)).astype(np.float32)

    X_scaled = (X_subset - np.mean(X_subset, axis=0)) / (np.std(X_subset, axis=0) + 1e-8)
elif args.dataset == "distinct_chems":
    try:
        import datamol as dm
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError:
        print("Please install datamol and rdkit.")
        exit(1)
        
    print("Generating Toy Dataset using Hybrid Continuous Physical Descriptors...")
    X_list = []
    y_list = []
    fp_list = []
    desc_list = []
    
    # Class 0: Alkanes (Straight Carbon Chains)
    for i in range(5, 105):
        mol = Chem.MolFromSmiles("C" * i)
        if mol:
            feat, fp, desc = get_features(mol)
            X_list.append(feat)
            fp_list.append(fp)
            desc_list.append(desc)
            y_list.append(0)
            
    # Class 1: Polyphenyls (Aromatic Rings)
    for i in range(1, 101):
        mol = Chem.MolFromSmiles("c1ccccc1" + ("-c1ccccc1" * i))
        if mol:
            feat, fp, desc = get_features(mol)
            X_list.append(feat)
            fp_list.append(fp)
            desc_list.append(desc)
            y_list.append(1)
            
    # Class 2: Polyethers / PEGs (Oxygen heavy)
    for i in range(1, 101):
        mol = Chem.MolFromSmiles("CO" + ("CCO" * i))
        if mol:
            feat, fp, desc = get_features(mol)
            X_list.append(feat)
            fp_list.append(fp)
            desc_list.append(desc)
            y_list.append(2)
            
    X_subset = np.array(X_list, dtype=np.float32)
    y_subset = np.array(y_list, dtype=np.int32)
    X_struct_subset = np.array(fp_list, dtype=np.float32)
    y_props_subset = np.array(desc_list, dtype=np.float32)
    y_props_subset = (y_props_subset - np.mean(y_props_subset, axis=0)) / (np.std(y_props_subset, axis=0) + 1e-8)
    
    # Standardize BEFORE PCA so large values like MolWt don't drown out the 0/1 fingerprints!
    X_subset = (X_subset - np.mean(X_subset, axis=0)) / (np.std(X_subset, axis=0) + 1e-8)
    
    from sklearn.decomposition import PCA
    print(f"Applying PCA on {X_subset.shape[0]} toy molecules...")
    pca = PCA(n_components=32, random_state=42)
    X_subset = pca.fit_transform(X_subset).astype(np.float32)
    
    # Standard Scaling (Mean=0, Std=1) perfectly centers the data for the Normalizing Flow
    # This prevents the flow from collapsing all points into the positive [0,1] quadrant
    X_scaled = (X_subset - np.mean(X_subset, axis=0)) / (np.std(X_subset, axis=0) + 1e-8)
elif args.dataset == "dude":
    try:
        import urllib.request
        import datamol as dm
        from sklearn.model_selection import StratifiedShuffleSplit
    except ImportError:
        print("Please install datamol and scikit-learn.")
        exit(1)
        
    print("Downloading DUD-E EGFR (Virtual Screening) dataset from UCSF...")
    
    act_req = urllib.request.urlopen("https://dude.docking.org/targets/egfr/actives_final.ism")
    actives = act_req.read().decode('utf-8').splitlines()
    
    dec_req = urllib.request.urlopen("https://dude.docking.org/targets/egfr/decoys_final.ism")
    decoys = dec_req.read().decode('utf-8').splitlines()
    
    # DUD-E has 35,000 decoys and 500 actives. Subsample decoys to balance the dataset.
    np.random.seed(42)
    decoys_subset = np.random.choice(decoys, size=len(actives), replace=False)
    
    X_list = []
    y_list = []
    fp_list = []
    desc_list = []
    
    print("Computing features for Actives...")
    for line in actives:
        smiles = line.split()[0]
        mol = dm.to_mol(smiles)
        if mol is not None:
            feat, fp, desc = get_features(mol)
            X_list.append(feat)
            fp_list.append(fp)
            desc_list.append(desc)
            y_list.append(1)
            
    print("Computing features for Property-Matched Decoys...")
    for line in decoys_subset:
        smiles = line.split()[0]
        mol = dm.to_mol(smiles)
        if mol is not None:
            feat, fp, desc = get_features(mol)
            X_list.append(feat)
            fp_list.append(fp)
            desc_list.append(desc)
            y_list.append(0)
            
    X_full = np.array(X_list, dtype=np.float32)
    y_full = np.array(y_list, dtype=np.int32)
    fp_full = np.array(fp_list, dtype=np.float32)
    desc_full = np.array(desc_list, dtype=np.float32)
    
    # Subsample to keep the diagnostic script fast
    sss = StratifiedShuffleSplit(n_splits=1, train_size=min(len(y_full)-1, 600), random_state=42)
    train_idx, _ = next(sss.split(X_full, y_full))
    X_subset = X_full[train_idx]
    y_subset = y_full[train_idx]
    X_struct_subset = fp_full[train_idx]
    y_props_subset = desc_full[train_idx]
    y_props_subset = (y_props_subset - np.mean(y_props_subset, axis=0)) / (np.std(y_props_subset, axis=0) + 1e-8)
    
    # See the note on the other chemistry branches.
    from sklearn.decomposition import PCA
    X_subset = (X_subset - X_subset.mean(0)) / (X_subset.std(0) + 1e-8)
    X_subset = PCA(n_components=32, random_state=42).fit_transform(X_subset)
    X_subset = ((X_subset - X_subset.mean(0)) / (X_subset.std(0) + 1e-8)).astype(np.float32)
    X_scaled = X_subset
elif args.dataset == "wine":
    X, y = datasets.load_wine(return_X_y=True)
    X_scaled = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    X_subset, y_subset = X, y
elif args.dataset == "cancer":
    X, y = datasets.load_breast_cancer(return_X_y=True)
    X_scaled = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    X_subset, y_subset = X, y
elif args.dataset == "iris":
    X, y = datasets.load_iris(return_X_y=True)
    X_scaled = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    X_subset, y_subset = X, y
elif args.dataset == "moons":
    X, y = datasets.make_moons(n_samples=600, noise=0.1, random_state=42)
    X_scaled = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    X_subset, y_subset = X, y
elif args.dataset == "circles":
    X, y = datasets.make_circles(n_samples=600, noise=0.05, factor=0.5, random_state=42)
    X_scaled = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    X_subset, y_subset = X, y
elif args.dataset == "swiss_roll":
    X, y = datasets.make_swiss_roll(n_samples=600, noise=0.0, random_state=42)
    X_scaled = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    X_subset, y_subset = X, y
elif args.dataset == "s_curve":
    X, y = datasets.make_s_curve(n_samples=600, noise=0.0, random_state=42)
    X_scaled = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    X_subset, y_subset = X, y
elif args.dataset == "severed_sphere":
    # Classic manifold-unrolling test case (cf. sklearn's "Manifold Learning
    # methods on a severed sphere" example, and ugtm's own visualization
    # examples gallery) - a sphere with both poles AND an azimuthal wedge
    # removed, leaving a surface that's topologically a disk with a boundary
    # (unlike a full closed sphere, which provably can't be flattened onto a
    # plane without tearing) - a genuinely different unrolling challenge from
    # swiss_roll/s_curve's already-open strips.
    rng = np.random.RandomState(42)
    n_init = 800
    p = rng.rand(n_init) * (2 * np.pi - 0.55)
    t = rng.rand(n_init) * np.pi
    indices = (t < (np.pi - (np.pi / 8))) & (t > (np.pi / 8))
    p, t = p[indices], t[indices]
    X = np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], axis=1)
    y = p  # continuous color = azimuthal angle, the "unrolled" coordinate
    X_scaled = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    X_subset, y_subset = X, y

is_continuous = args.dataset in ["swiss_roll", "s_curve", "severed_sphere"]

#X = X_scaled
X = X_scaled
y = y_subset
print(f"Dataset: {args.dataset.capitalize()}")
print(f"Subset: {X_subset.shape[0]} samples × {X_subset.shape[1]} features, {len(np.unique(y_subset))} classes\n")

# ── Shared NFMap defaults (only overrides differ between configs) ──────────────
# grid_bounds=6.0, not 3.0: matches benchmarks/paper1/run_ablation.py's DEFAULTS
# (bumped 2026-09-13 — at 3.0, 16-60% of points land outside the visible grid
# for ablations/benchmarks that aren't themselves about grid containment,
# which is a distracting, unrelated artifact in the resulting figures. See
# nfmap_paper_experiments.md's B9 for the finding this responds to.
DEFAULTS = dict(
    n_components=2,
    use_canonical=False,
    complexity='medium',
    inject_weight=0.5,
    n_prior_components=100,
    prior_weight=0.1,
    grid_bounds=6.0,
    graph_weight=5.0,
    repulsion_weight=1.0,
    prior_layout='hexagonal',
    random_state=args.seed,
)

# ── NFMap configurations ──────────────────────────────────────────────────────
# One entry per --nfmap-variants pair, or a single default entry when that flag
# is unset. Each variant gets kind "nfmap_<name>": save_run keys the output
# directory on the kind, so sharing "nfmap" across variants would silently
# leave only whichever ran last.
def _shared_overrides():
    return {"verbose": True,
            **({"prior_layout": args.prior_layout} if args.prior_layout else {}),
            **({"grid_bounds": args.grid_bounds} if args.grid_bounds is not None else {})}


def _nfmap_configs():
    if not args.nfmap_variants:
        return [{
            "label":      "NEW: NFMap",
            "hypothesis": "Default NFMap",
            "kind":       "nfmap",
            "overrides":  {**_shared_overrides(),
                           **({"sigma_perp": args.sigma_perp} if args.sigma_perp is not None else {})},
        }]
    out = []
    for pair in args.nfmap_variants.split(","):
        name, _, raw = pair.partition("=")
        name = name.strip()
        if not name or not raw.strip():
            raise SystemExit(f"--nfmap-variants: expected name=sigma_perp pairs, got {pair!r}")
        sigma = float(raw)
        out.append({
            "label":      f"NEW: NFMap ({name})",
            "hypothesis": f"NFMap, sigma_perp={sigma} ({name}-optimising end of the dial)",
            "kind":       f"nfmap_{name}",
            "overrides":  {**_shared_overrides(), "sigma_perp": sigma},
        })
    return out


def _is_nfmap(kind):
    return bool(kind) and kind.startswith("nfmap")


# ── Experiment configurations ─────────────────────────────────────────────────
# Each entry overrides only the variable under test; everything else is DEFAULTS.
CONFIGS = [

    # {
    #     "label":      "NEW: NFMap",
    #     "hypothesis": "Max epochs 1000, sigma_perp 10.0, graph_weight 5.0",
    #     "kind":       "nfmap",
    #     "overrides":  {"max_epochs": 1000, "patience": 100, "sigma_perp": 10.0, "graph_weight": 5.0},
    # },
    *_nfmap_configs(),
    # {
    #     "label":      "NEW: NFMap CURRENT BEST (Negative Sampling x5)",
    #     "hypothesis": "Sample 5 negative edges per positive to push clusters apart",
    #     "kind":       "nfmap",
    #     "overrides":  {"max_epochs": 1000, "patience": 100, "sigma_perp": 10.0, "graph_weight": 5.0, "negative_sample_rate": 5},
    # },
    # {
    #     "label":      "NEW: NFMap (Early Exaggeration)",
    #     "hypothesis": "Multiply graph attraction by 4 for the first 333 epochs",
    #     "kind":       "nfmap",
    #     "overrides":  {"max_epochs": 1000, "patience": 100, "sigma_perp": 10.0, "graph_weight": 5.0, "negative_sample_rate": 5, "early_exaggeration": True},
    # },
    # {
    #     "label":      "NEW: NFMap (Low Graph Weight)",
    #     "hypothesis": "Max epochs 1000, sigma_perp 10.0, graph_weight 1.0",
    #     "kind":       "nfmap",
    #     "overrides":  {"max_epochs": 1000, "patience": 100, "sigma_perp": 10.0, "graph_weight": 1.0},
    # },
    # {
    #     "label":      "NEW: NFMap (No Graph Weight)",
    #     "hypothesis": "Max epochs 1000, sigma_perp 10.0, graph_weight 0.0 (Pure Injective Flow)",
    #     "kind":       "nfmap",
    #     "overrides":  {"max_epochs": 1000, "patience": 100, "sigma_perp": 0.05, "graph_weight": 0.0},
    # },
    {
        "label":      "GTM\n(reference)" if HAS_GTM else "GTM\n(not installed)",
        "hypothesis": "Generative Topographic Mapping — NFMap's direct predecessor",
        "kind":       "gtm",
    },
    {
        "label":      "t-SNE\n(reference)",
        "hypothesis": "Gold-standard cluster baseline",
        "kind":       "tsne",
    },
    {
        "label":      "UMAP\n(reference)" if HAS_UMAP else "UMAP\n(not installed)",
        "hypothesis": "Gold-standard topology baseline",
        "kind":       "umap",
    },
]

# ── Run all configurations ────────────────────────────────────────────────────
results = []
for i, config in enumerate(CONFIGS):
    print(f"\n[{i+1}/{len(CONFIGS)}] {config['label']}")
    
    try:
        t0 = time.time()
        
        # Determine instantiation based on kind
        prior_means = None
        if _is_nfmap(config["kind"]):
            params = DEFAULTS.copy()
            params.update(config.get("overrides", {}))
            
            fit_kwargs = {}
            if args.use_struct and X_struct_subset is not None:
                fit_kwargs['X_struct'] = X_struct_subset
            if args.use_props and y_props_subset is not None:
                fit_kwargs['y_props'] = y_props_subset
                
            model = NFMap(**params)
            if args.save_latent:
                Z_full = model.fit_transform(X, return_full=True, **fit_kwargs)
                clean_label = config["label"].replace(" ", "_").replace(":", "")
                np.save(f"nfmap_latent_{args.dataset}_{clean_label}.npy", Z_full)
                Z = Z_full[:, :model.n_components]
            else:
                Z = model.fit_transform(X, **fit_kwargs)
            prior_means = model.module_.prior_means.detach().cpu().numpy()

        elif config["kind"] == "tsne":
            Z = manifold.TSNE(
                n_components=2, init='pca', random_state=args.seed, perplexity=30
            ).fit_transform(X)

        elif config["kind"] == "gtm":
            if HAS_GTM:
                # k=10 -> a 10x10 grid = 100 GTM nodes, matching NFMap's
                # n_prior_components=100. This had been eGTM(m=6), which leaves
                # k at its default 16 (a 16x16 grid = 256 nodes, 2.56x finer
                # than NFMap's) and instead tunes m, the RBF-centre count -
                # GTM's mapping flexibility, which has no NFMap analogue since
                # the flow replaces the RBF mapping. Grid resolution is the one
                # hyperparameter the two methods define identically, and
                # Novelty #5 is a claim about exactly that grid, so it has to
                # be matched for the comparison to mean anything.
                Z = eGTM(k=10, m=6, random_state=args.seed).fit_transform(X)
            else:
                Z = np.full((len(y), 2), np.nan)

        elif config["kind"] == "umap":
            if HAS_UMAP:
                Z = umap_lib.UMAP(
                    n_neighbors=15, min_dist=0.1, random_state=args.seed
                ).fit_transform(X)
            else:
                Z = np.full((len(y), 2), np.nan)

        t1 = time.time()
        elapsed = t1 - t0
        valid_Z = not np.any(np.isnan(Z))
        if valid_Z:
            trust = trustworthiness(X, Z)
            cont = trustworthiness(Z, X)
            t_c_f1 = 2 * (trust * cont) / (trust + cont) if (trust + cont) > 0 else 0
            
            dist_corr, _ = pearsonr(pdist(X), pdist(Z))
            
            def get_kth_dist(pts, k=5):
                from scipy.spatial.distance import cdist
                D = cdist(pts, pts)
                D.sort(axis=1)
                return D[:, k]
                
            dens_corr, _ = pearsonr(np.log(get_kth_dist(X, 5) + 1e-8), np.log(get_kth_dist(Z, 5) + 1e-8))

            if not is_continuous:
                sil = silhouette_score(Z, y)
                knn = KNeighborsClassifier(n_neighbors=5, algorithm='brute').fit(Z, y)
                acc = knn.score(Z, y)
                # AUC: how well the map's own local geometry discriminates
                # classes, using a KNN classifier's soft cluster-assignment
                # scores (predict_proba) as the discriminator - the "GMM
                # cluster assignment score" the doc's A1/A2 metric spec calls
                # for, implemented via KNN-on-the-embedding rather than the
                # GMM prior's own (unsupervised, not label-aware) posterior,
                # since the GMM components aren't fit per-class. Binary ->
                # standard ROC-AUC; multi-class -> macro-averaged one-vs-rest
                # (sklearn's roc_auc_score(multi_class='ovr')). NaN if it
                # can't be computed (e.g. a class with too few members for a
                # stable one-vs-rest score) rather than crashing the run.
                try:
                    proba = knn.predict_proba(Z)
                    if proba.shape[1] == 2:
                        auc = roc_auc_score(y, proba[:, 1])
                    else:
                        auc = roc_auc_score(y, proba, multi_class='ovr', average='macro')
                except ValueError:
                    auc = float('nan')
            else:
                sil = float('nan')
                knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute').fit(Z, y)
                acc = knn.score(Z, y)
                auc = float('nan')  # AUC is a classification metric; undefined for continuous targets
        else:
            trust, cont, t_c_f1, dist_corr, dens_corr, sil, acc, auc = float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

        print(f"    T/C_F1={t_c_f1:.3f} Trst={trust:.3f} Cont={cont:.3f} DistCor={dist_corr:.3f} DensCor={dens_corr:.3f} Sil={sil:.3f} AUC={auc:.3f} (time: {elapsed:.1f}s)")
        # Store the ACTUAL params used for this run (DEFAULTS + this config's
        # overrides, e.g. --prior-layout), not just DEFAULTS - otherwise a
        # saved config.json silently misreports any override (found 2026-09-16
        # via a --prior-layout square run whose config.json still said
        # "hexagonal": the fit itself used the override correctly, only the
        # saved metadata was wrong).
        results.append({"label": config["label"], "hypothesis": config["hypothesis"], "kind": config["kind"],
                         "Z": Z, "sil": sil, "trust": trust, "cont": cont, "t_c_f1": t_c_f1, "dist_corr": dist_corr, "dens_corr": dens_corr, "acc": acc, "auc": auc, "ok": True, "time": elapsed, "prior_means": prior_means,
                         "nfmap_params": params if _is_nfmap(config["kind"]) else None})
        if "NFMap" in config["label"]:
            results.append({"label": config["label"] + " (No Prior Overlay)", "hypothesis": "Plot without grid overlay for clarity", "kind": None,
                             "Z": Z.copy() if Z is not None else None, "sil": sil, "trust": trust, "cont": cont, "t_c_f1": t_c_f1, "dist_corr": dist_corr, "dens_corr": dens_corr, "acc": acc, "auc": auc, "ok": True, "time": 0.0, "prior_means": None})
    except Exception as e:
        t1 = time.time()
        elapsed = t1 - t0 if 't0' in locals() else 0.0
        print(f"    FAILED: {e} (time: {elapsed:.1f}s)")
        results.append({"label": config["label"], "hypothesis": config["hypothesis"], "kind": config["kind"],
                         "Z": None, "sil": float('nan'), "trust": float('nan'), "cont": float('nan'), "t_c_f1": float('nan'), "dist_corr": float('nan'), "dens_corr": float('nan'), "acc": float('nan'), "auc": float('nan'), "ok": False, "time": elapsed, "prior_means": None})

# ── Persist results (opt-in, --save-results) ───────────────────────────────────
# Reuses benchmarks/paper1/common/io.py's save_run so these runs land in the
# same results/<experiment_id>/<dataset>_<method>/{config,metrics,embeddings}
# layout as the rest of the paper1 ablation harness, per
# nfmap_paper_experiments.md's reproducibility policy. Off by default so this
# doesn't change nfmap_diagnostics.py's existing interactive-use behavior.
if args.save_results:
    _paper1_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper1")
    if _paper1_dir not in sys.path:
        sys.path.insert(0, _paper1_dir)
    from common.io import save_run

    n_saved = 0
    for res in results:
        if res["kind"] is None:
            continue  # skip the "(No Prior Overlay)" plot-only duplicate — not a separate run
        metrics_out = {k: res[k] for k in ("trust", "cont", "t_c_f1", "dist_corr", "dens_corr", "sil", "acc", "auc")}
        config_out = {"dataset": args.dataset, "kind": res["kind"], "label": res["label"],
                       "seed": args.seed, "ok": res["ok"], "elapsed_s": res["time"]}
        if _is_nfmap(res["kind"]):
            config_out.update(res.get("nfmap_params") or DEFAULTS)
        # Seed in the directory name, not just in config.json: save_run keys on
        # the method name, so without it a second seed would silently overwrite
        # the first and the "3-seed" table would be one run reported three times.
        save_run(args.experiment_id, args.dataset, f"{res['kind']}_seed{args.seed}",
                 config_out, metrics_out, res["Z"], y=y,
                 prior_means=res.get("prior_means"))
        n_saved += 1
    print(f"\nSaved {n_saved} runs under paper1/results/{args.experiment_id}/{args.dataset}_<kind>/")

# ── Plot ──────────────────────────────────────────────────────────────────────
# Calculate grid size dynamically
n_plots = len(results)
n_cols = 3
n_rows = math.ceil(n_plots / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
axes = axes.flatten()

scatter_plots = []


for ax, res in zip(axes, results):
    if res["ok"] and res["Z"] is not None and not np.any(np.isnan(res["Z"])):
        sc = ax.scatter(
            res["Z"][:, 0], res["Z"][:, 1],
            c=y, cmap='viridis' if is_continuous else 'tab10',
            vmin=None if is_continuous else 0, vmax=None if is_continuous else 9,
            s=15, alpha=0.85, linewidths=0,
        )
        if res.get("prior_means") is not None:
            ax.scatter(
                res["prior_means"][:, 0], res["prior_means"][:, 1],
                marker='X', s=100, c='black', edgecolors='white', linewidths=1.5, zorder=10
            )
        scatter_plots.append(sc)
        metrics_str = f"T/C_F1={res['t_c_f1']:.2f} Dens={res['dens_corr']:.2f} D-Cor={res['dist_corr']:.2f} Sil={res['sil']:.2f}"
        if not is_continuous:
            metrics_str += f" Acc={res['acc']:.2f} Sil={res['sil']:.2f}"
    else:
        ax.text(0.5, 0.5,
                "FAILED" if not res["ok"] else "N/A",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=14, color='#cc0000')
        metrics_str = ""

    ax.set_title(res["label"] + "\n" + metrics_str, fontsize=10, fontweight='bold')
    ax.set_xlabel(res["hypothesis"], fontsize=8, color='#555555', labelpad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('tight')

# Hide any unused axes
for i in range(len(results), len(axes)):
    axes[i].set_visible(False)

# Shared legend for classes
if scatter_plots and not is_continuous:
    handles, _ = scatter_plots[0].legend_elements()
    unique_classes = np.unique(y)
    fig.legend(handles, [f"Class {c}" for c in unique_classes], 
               loc='lower center', ncol=min(10, len(unique_classes)), 
               title='Classes', bbox_to_anchor=(0.5, 0.02))

fig.suptitle(
    f"NFMap Diagnostics — {args.dataset.capitalize()} Dataset\n"
    "Each panel isolates one hypothesis; t-SNE/UMAP shown as reference baselines.\n"
    "Hover over points to see their class labels.",
    fontsize=13, y=1.02,
)
plt.tight_layout(rect=[0, 0.08, 1, 1])  # Leave room for legend at bottom

# Interactive Hover Annotation
annot = axes[0].annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
                         zorder=100)
annot.set_visible(False)
current_ax = [None]  # Keep track of which ax the annot belongs to

def hover(event):
    if not event.inaxes:
        if annot.get_visible():
            annot.set_visible(False)
            fig.canvas.draw_idle()
        return

    # Check which scatter plot we are hovering over
    for sc in scatter_plots:
        if sc.axes == event.inaxes:
            cont, ind = sc.contains(event)
            if cont:
                # Move annotation to the active axes if needed
                if current_ax[0] != event.inaxes:
                    annot.remove()
                    event.inaxes.add_artist(annot)
                    current_ax[0] = event.inaxes

                # Get position and class of the hovered point
                pos = sc.get_offsets()[ind["ind"][0]]
                annot.xy = pos
                hovered_class = y[ind["ind"][0]]
                annot.set_text(f"Class: {hovered_class}")
                annot.set_visible(True)
                fig.canvas.draw_idle()
                return

    if annot.get_visible():
        annot.set_visible(False)
        fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

out_path = "nfmap_diagnostics.png"
plt.savefig(out_path, bbox_inches='tight', dpi=150)
print(f"\nSaved → {out_path}")

# ── Interactive HTML Output (Plotly) ──────────────────────────────────────────
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.colors as pcolors

    subplot_titles = []
    for res in results:
        sil_str = f"  sil={res['sil']:.3f}" if res["sil"] is not None and not np.isnan(res["sil"]) else " (FAILED)"
        subplot_titles.append(res["label"].replace('\n', ' ') + sil_str)

    fig_html = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles)
    
    unique_classes = np.unique(y)
    colors = pcolors.qualitative.Plotly
    
    for i, res in enumerate(results):
        row = i // n_cols + 1
        col = i % n_cols + 1
        if res["ok"] and res["Z"] is not None and not np.any(np.isnan(res["Z"])):
            for idx, c in enumerate(unique_classes):
                mask = (y == c)
                showlegend = True if i == 0 else False
                fig_html.add_trace(go.Scatter(
                    x=res["Z"][mask, 0], y=res["Z"][mask, 1],
                    mode='markers',
                    name=f"Class {c}",
                    marker=dict(color=colors[idx % len(colors)], size=5, opacity=0.8),
                    text=[f"Class: {c}"] * np.sum(mask),
                    hoverinfo='text',
                    showlegend=showlegend
                ), row=row, col=col)
                
    fig_html.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig_html.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    fig_html.update_layout(
        height=350 * n_rows, width=400 * n_cols,
        title_text=f"NFMap Diagnostics — {args.dataset.capitalize()} Dataset (Hover for details)",
        plot_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    
    html_path = "nfmap_diagnostics.html"
    fig_html.write_html(html_path)
    print(f"Saved → {html_path} (Open this file in your web browser for interactive hover!)")
except ImportError:
    pass

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 125)
if is_continuous:
    print(f"  {'Config':<38} {'T/C_F1':>8} {'Trust':>8} {'Cont':>8} {'DensCor':>8} {'DistCor':>8} {'KNN R2':>8}   {'Time (s)':>8}")
else:
    print(f"  {'Config':<38} {'T/C_F1':>8} {'Trust':>8} {'Cont':>8} {'DensCor':>8} {'DistCor':>8} {'KNN Acc':>8} {'Sil':>8}   {'Time (s)':>8}")
print("=" * 125)
for res in results:
    name = res["label"].replace('\n', ' ')
    if "(No Prior Overlay)" in name:
        continue
    if res["ok"]:
        t_c_f1_str = f"{res['t_c_f1']:.3f}"
        trust_str = f"{res['trust']:.3f}"
        cont_str = f"{res['cont']:.3f}"
        dens_str = f"{res['dens_corr']:.3f}"
        corr_str = f"{res['dist_corr']:.3f}"
        acc_str = f"{res['acc']:.3f}"
        sil_str = f"{res['sil']:.3f}"
        time_str = f"{res.get('time', 0.0):.1f}"
    else:
        t_c_f1_str, trust_str, cont_str, dens_str, corr_str, acc_str, sil_str = "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"
        time_str = f"{res.get('time', 0.0):.1f}"

    if is_continuous:
        print(f"  {name:<38} {t_c_f1_str:>8} {trust_str:>8} {cont_str:>8} {dens_str:>8} {corr_str:>8} {acc_str:>8}   {time_str:>8}")
    else:
        print(f"  {name:<38} {t_c_f1_str:>8} {trust_str:>8} {cont_str:>8} {dens_str:>8} {corr_str:>8} {acc_str:>8} {sil_str:>8}   {time_str:>8}")
print("=" * 125)

# Close matplotlib to prevent blocking script execution
plt.close('all')
