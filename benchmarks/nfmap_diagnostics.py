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
# Fix macOS OpenMP/Accelerate segfaults triggered by scikit-learn/UMAP/Numba imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["TBB_CXX_TYPE"] = "clang"

import time
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, manifold
from sklearn.metrics import silhouette_score
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
parser.add_argument("--dataset", type=str, default="digits", choices=["digits", "wine", "cancer", "iris", "moons", "circles", "swiss_roll", "s_curve", "bbbp", "bace", "esol", "distinct_chems", "dude"], help="Dataset to evaluate on")
parser.add_argument("--save_latent", action="store_true", help="Save the full D-dimensional latent representations for NFMap to .npy files")
parser.add_argument("--use_struct", action="store_true", help="Pass structural fingerprints to NFMap as X_struct")
parser.add_argument("--use_props", action="store_true", help="Pass physical properties to NFMap as y_props")
args = parser.parse_args()

X_struct_subset = None
y_props_subset = None

# ── Data Loading ──────────────────────────────────────────────────────────────
np.random.seed(42)

def get_features(mol):
    import datamol as dm
    import numpy as np
    from rdkit.Chem import Descriptors
    fp = dm.to_fp(mol, fpSize=128, as_array=True)
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
    return np.concatenate([fp, desc]), fp, desc

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
    X_subset = (X_subset - np.mean(X_subset, axis=0)) / (np.std(X_subset, axis=0) + 1e-8)
    
    from sklearn.decomposition import PCA
    print(f"Applying PCA to reduce hybrid features to 32 dimensions...")
    pca = PCA(n_components=32, random_state=42)
    X_subset = pca.fit_transform(X_subset).astype(np.float32)
    
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
    X_subset = (X_subset - np.mean(X_subset, axis=0)) / (np.std(X_subset, axis=0) + 1e-8)
    
    from sklearn.decomposition import PCA
    print(f"Applying PCA to reduce hybrid features to 32 dimensions...")
    pca = PCA(n_components=32, random_state=42)
    X_subset = pca.fit_transform(X_subset).astype(np.float32)
    
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
    X_subset = (X_subset - np.mean(X_subset, axis=0)) / (np.std(X_subset, axis=0) + 1e-8)
    
    from sklearn.decomposition import PCA
    print(f"Applying PCA to reduce hybrid features to 32 dimensions...")
    pca = PCA(n_components=32, random_state=42)
    X_subset = pca.fit_transform(X_subset).astype(np.float32)
    
    # Standard Scaling (Mean=0, Std=1) for Normalizing Flow
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
    
    # Standardize before PCA so MolWt doesn't dominate
    X_subset = (X_subset - np.mean(X_subset, axis=0)) / (np.std(X_subset, axis=0) + 1e-8)
    
    from sklearn.decomposition import PCA
    # print(f"Applying PCA to reduce hybrid features to 32 dimensions...")
    # pca = PCA(n_components=32, random_state=42)
    # X_subset = pca.fit_transform(X_subset).astype(np.float32)
    
    # Standard Scaling (Mean=0, Std=1) for Normalizing Flow
    X_scaled = (X_subset - np.mean(X_subset, axis=0)) / (np.std(X_subset, axis=0) + 1e-8)
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

is_continuous = args.dataset in ["swiss_roll", "s_curve"]

#X = X_scaled
X = X_scaled
y = y_subset
print(f"Dataset: {args.dataset.capitalize()}")
print(f"Subset: {X_subset.shape[0]} samples × {X_subset.shape[1]} features, {len(np.unique(y_subset))} classes\n")

# ── Shared NFMap defaults (only overrides differ between configs) ──────────────
DEFAULTS = dict(
    n_components=2,
    use_canonical=False,
    complexity='medium',
    n_prior_components=15,
    prior_weight=0.005,
    graph_weight=1.0,
    repulsion_weight=1.0,
    random_state=42,
)

# ── Experiment configurations ─────────────────────────────────────────────────
# Each entry overrides only the variable under test; everything else is DEFAULTS.
CONFIGS = [

    # {
    #     "label":      "NEW: NFMap",
    #     "hypothesis": "Max epochs 1000, sigma_perp 10.0, graph_weight 5.0",
    #     "kind":       "nfmap",
    #     "overrides":  {"max_epochs": 1000, "patience": 100, "sigma_perp": 10.0, "graph_weight": 5.0},
    # },
    {
        "label":      "NEW: NFMap",
        "hypothesis": "Default NFMap",
        "kind":       "nfmap",
        "overrides":  {"max_epochs": 1000, "patience": 100, "sigma_perp": 10, "graph_weight": 5.0, "negative_sample_rate": 5,
        "n_prior_components": 100, "learnable_priors": False, "prior_weight": 0.1, "verbose": True, "graph_kernel": "umap", "early_exaggeration": False},
    },
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
        if config["kind"] == "nfmap":
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
                n_components=2, init='pca', random_state=42, perplexity=30
            ).fit_transform(X)

        elif config["kind"] == "gtm":
            if HAS_GTM:
                Z = eGTM(m=6).fit_transform(X)
            else:
                Z = np.full((len(y), 2), np.nan)

        elif config["kind"] == "umap":
            if HAS_UMAP:
                Z = umap_lib.UMAP(
                    n_neighbors=15, min_dist=0.1, random_state=42
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
            else:
                sil = float('nan')
                knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute').fit(Z, y)
                acc = knn.score(Z, y)
        else:
            trust, cont, t_c_f1, dist_corr, dens_corr, sil, acc = float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

        print(f"    T/C_F1={t_c_f1:.3f} Trst={trust:.3f} Cont={cont:.3f} DistCor={dist_corr:.3f} DensCor={dens_corr:.3f} Sil={sil:.3f} (time: {elapsed:.1f}s)")
        results.append({"label": config["label"], "hypothesis": config["hypothesis"],
                         "Z": Z, "sil": sil, "trust": trust, "cont": cont, "t_c_f1": t_c_f1, "dist_corr": dist_corr, "dens_corr": dens_corr, "acc": acc, "ok": True, "time": elapsed, "prior_means": prior_means})
        if "NFMap" in config["label"]:
            results.append({"label": config["label"] + " (No Prior Overlay)", "hypothesis": "Plot without grid overlay for clarity",
                             "Z": Z.copy() if Z is not None else None, "sil": sil, "trust": trust, "cont": cont, "t_c_f1": t_c_f1, "dist_corr": dist_corr, "dens_corr": dens_corr, "acc": acc, "ok": True, "time": 0.0, "prior_means": None})
    except Exception as e:
        t1 = time.time()
        elapsed = t1 - t0 if 't0' in locals() else 0.0
        print(f"    FAILED: {e} (time: {elapsed:.1f}s)")
        results.append({"label": config["label"], "hypothesis": config["hypothesis"],
                         "Z": None, "sil": float('nan'), "trust": float('nan'), "cont": float('nan'), "t_c_f1": float('nan'), "dist_corr": float('nan'), "dens_corr": float('nan'), "acc": float('nan'), "ok": False, "time": elapsed, "prior_means": None})

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
