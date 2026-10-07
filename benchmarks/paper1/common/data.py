"""Dataset loaders for Paper 1 experiments.

Mirrors the exact preprocessing used in benchmarks/nfmap_diagnostics.py (same
scaling, same subsample size/seed, same feature construction for chemistry
datasets) so numbers from these scripts are directly comparable to anything
already produced with that exploratory script. Chemistry datasets are cached
to disk under data_cache/ on first fetch, both to avoid re-hitting the public
hosts on every rerun and so results stay reproducible if a host later changes
its data.

Each loader returns a dict:
    X            : (N, D) float32 array, ready to hand to NFMap
    y            : (N,) array — class labels (discrete) or a property (continuous)
    is_continuous: whether y is a continuous target (affects which metrics apply)
    name         : dataset name (for logging / output paths)
"""
import os
import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data_cache")

TOY_DATASETS = {"digits", "wine", "cancer", "iris", "moons", "circles", "swiss_roll", "s_curve", "severed_sphere"}
CHEMISTRY_DATASETS = {"bace", "dude", "bbbp", "esol", "drd2", "egfr"}
ALL_DATASETS = TOY_DATASETS | CHEMISTRY_DATASETS


def _get_features(mol):
    """2048-bit Morgan r=2 fingerprint + 8 physicochemical descriptors.

    The fingerprint width matches chem_kernel's hybrid-kernel fingerprints, so
    every representation in the paper is the same one.

    The descriptors are kept deliberately, and this is the one place the paper
    diverges from the production pipeline (which passes them to NFMap separately
    as y_props). Euclidean distance on a sparse binary fingerprint concentrates
    badly - measured on BACE, pairwise distances in the fingerprint-only space
    have a std/mean of just 0.135, meaning nearly every pair of molecules sits
    the same distance apart. DistCor then correlates against a near-constant and
    returns noise, including negative values. Adding the continuous descriptors
    raises that spread to 0.235 and restores a usable input metric. Production
    never notices because it does not compute DistCor; this paper's headline
    chemistry claim depends on it.
    """
    import datamol as dm
    from rdkit.Chem import Descriptors
    fp = dm.to_fp(mol, fpSize=2048, as_array=True)
    desc = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.RingCount(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
    ]
    return np.concatenate([fp, desc]).astype(np.float32)


# Production reduces the 2048-bit fingerprint to 128 components before the flow.
# Doing it HERE rather than inside NFMap is deliberate: NFMap would otherwise
# reduce internally while UMAP/t-SNE/GTM received the raw 2048-dim matrix, which
# compares methods across different input spaces instead of comparing the methods
# themselves. Every method in this paper now consumes exactly this matrix.
# 32, not production's 128. Production runs at ~50k molecules; these benchmarks
# use 600, so 128 components means more dimensions than a fifth of the sample
# count, and it fails in two compounding ways (measured on BACE):
#   1. the reference metric degrades - a plain PCA->2D projection, which should
#      score well by construction, drops from +0.410 to +0.367 DistCor, so the
#      achievable ceiling falls for every method;
#   2. z_perp inflates from 30 to 126 dimensions, the regime B2 shows NFMap is
#      sensitive to - NFMap's DistCor goes +0.150 -> -0.124, and UMAP turns
#      negative on all six chemistry sets too.
# Lowering sigma_perp does not rescue it (-0.089 at sigma_perp=0.5).
def _reduce_to_shared(X, seed, n_components=32):
    # Standardise before reducing, so the descriptors (MolWt ~300, TPSA ~80) do
    # not swamp the 0/1 fingerprint bits, and again after, so no single component
    # dominates the Euclidean metric every method is then judged on.
    from sklearn.decomposition import PCA
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    n = min(n_components, X.shape[1], X.shape[0])
    X = PCA(n_components=n, random_state=seed).fit_transform(X)
    return ((X - X.mean(0)) / (X.std(0) + 1e-8)).astype(np.float32)


def _load_toy(name, seed=42):
    from sklearn import datasets
    from sklearn.model_selection import StratifiedShuffleSplit

    if name == "digits":
        d = datasets.load_digits(n_class=10)
        sss = StratifiedShuffleSplit(n_splits=1, train_size=min(len(d.target) - 1, 600), random_state=seed)
        idx, _ = next(sss.split(d.data, d.target))
        X = d.data[idx].astype(np.float32) / 16.0
        y = d.target[idx]
        is_continuous = False
    elif name == "wine":
        X, y = datasets.load_wine(return_X_y=True)
        X = (X - X.min(0)) / (X.max(0) - X.min(0))
        is_continuous = False
    elif name == "cancer":
        X, y = datasets.load_breast_cancer(return_X_y=True)
        X = (X - X.min(0)) / (X.max(0) - X.min(0))
        is_continuous = False
    elif name == "iris":
        X, y = datasets.load_iris(return_X_y=True)
        X = (X - X.min(0)) / (X.max(0) - X.min(0))
        is_continuous = False
    elif name == "moons":
        X, y = datasets.make_moons(n_samples=600, noise=0.1, random_state=seed)
        X = (X - X.min(0)) / (X.max(0) - X.min(0))
        is_continuous = False
    elif name == "circles":
        X, y = datasets.make_circles(n_samples=600, noise=0.05, factor=0.5, random_state=seed)
        X = (X - X.min(0)) / (X.max(0) - X.min(0))
        is_continuous = False
    elif name == "swiss_roll":
        X, y = datasets.make_swiss_roll(n_samples=600, noise=0.0, random_state=seed)
        X = (X - X.min(0)) / (X.max(0) - X.min(0))
        is_continuous = True
    elif name == "s_curve":
        X, y = datasets.make_s_curve(n_samples=600, noise=0.0, random_state=seed)
        X = (X - X.min(0)) / (X.max(0) - X.min(0))
        is_continuous = True
    elif name == "severed_sphere":
        # Mirrors benchmarks/nfmap_diagnostics.py's severed_sphere loader
        # exactly (same rng, same n_init=800, same pole/wedge cut) so results
        # from this pipeline and that one are directly comparable.
        rng = np.random.RandomState(seed)
        n_init = 800
        p = rng.rand(n_init) * (2 * np.pi - 0.55)
        t = rng.rand(n_init) * np.pi
        indices = (t < (np.pi - (np.pi / 8))) & (t > (np.pi / 8))
        p, t = p[indices], t[indices]
        X = np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], axis=1)
        y = p
        X = (X - X.min(0)) / (X.max(0) - X.min(0))
        is_continuous = True
    else:
        raise ValueError(f"Unknown toy dataset: {name!r}")

    return X.astype(np.float32), y, is_continuous


def _fetch_cached(url, cache_name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, cache_name)
    if not os.path.exists(path):
        import urllib.request
        print(f"Fetching {url} -> {cache_name}")
        urllib.request.urlretrieve(url, path)
    return path


def _load_bace(seed=42):
    import pandas as pd
    import datamol as dm
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.decomposition import PCA

    path = _fetch_cached("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv", "bace.csv")
    df = pd.read_csv(path)

    X_list, y_list, mol_list = [], [], []
    for _, row in df.iterrows():
        mol = dm.to_mol(row["mol"])
        if mol is not None:
            mol_list.append(mol)
            X_list.append(_get_features(mol))
            y_list.append(row["Class"])

    X_full = np.array(X_list, dtype=np.float32)
    y_full = np.array(y_list, dtype=np.int32)

    sss = StratifiedShuffleSplit(n_splits=1, train_size=min(len(y_full) - 1, 600), random_state=seed)
    idx, _ = next(sss.split(X_full, y_full))
    X, y = X_full[idx], y_full[idx]
    mol_list = [mol_list[i] for i in idx]

    return _reduce_to_shared(X, seed), y, False, mol_list


def _load_bbbp(seed=42):
    import pandas as pd
    import datamol as dm
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.decomposition import PCA

    path = _fetch_cached("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv", "bbbp.csv")
    df = pd.read_csv(path)

    X_list, y_list, mol_list = [], [], []
    for _, row in df.iterrows():
        mol = dm.to_mol(row["smiles"])
        if mol is not None:
            mol_list.append(mol)
            X_list.append(_get_features(mol))
            y_list.append(row["p_np"])

    X_full = np.array(X_list, dtype=np.float32)
    y_full = np.array(y_list, dtype=np.int32)

    sss = StratifiedShuffleSplit(n_splits=1, train_size=min(len(y_full) - 1, 600), random_state=seed)
    idx, _ = next(sss.split(X_full, y_full))
    X, y = X_full[idx], y_full[idx]
    mol_list = [mol_list[i] for i in idx]

    return _reduce_to_shared(X, seed), y, False, mol_list


def _load_dude(seed=42, n_actives=None):
    """DUD-E EGFR, partial subsample (~600 total, matching nfmap_diagnostics.py).

    n_actives=None uses every active fetched (~500) and an equal number of
    decoys, then stratified-subsamples down to 600 total — same recipe as the
    exploratory script. NOT the full ~35,000-decoy set (see
    nfmap_paper_experiments.md's DUD-E scope open question — partial was
    chosen deliberately for this first pass).
    """
    import datamol as dm
    from sklearn.model_selection import StratifiedShuffleSplit

    act_path = _fetch_cached("https://dude.docking.org/targets/egfr/actives_final.ism", "dude_egfr_actives.ism")
    dec_path = _fetch_cached("https://dude.docking.org/targets/egfr/decoys_final.ism", "dude_egfr_decoys.ism")

    with open(act_path) as f:
        actives = f.read().splitlines()
    with open(dec_path) as f:
        decoys = f.read().splitlines()

    rng = np.random.RandomState(seed)
    decoys_subset = rng.choice(decoys, size=len(actives), replace=False)

    X_list, y_list, mol_list = [], [], []
    for line in actives:
        mol = dm.to_mol(line.split()[0])
        if mol is not None:
            mol_list.append(mol)
            X_list.append(_get_features(mol))
            y_list.append(1)
    for line in decoys_subset:
        mol = dm.to_mol(line.split()[0])
        if mol is not None:
            mol_list.append(mol)
            X_list.append(_get_features(mol))
            y_list.append(0)

    X_full = np.array(X_list, dtype=np.float32)
    y_full = np.array(y_list, dtype=np.int32)

    sss = StratifiedShuffleSplit(n_splits=1, train_size=min(len(y_full) - 1, 600), random_state=seed)
    idx, _ = next(sss.split(X_full, y_full))
    X, y = X_full[idx], y_full[idx]
    mol_list = [mol_list[i] for i in idx]

    return _reduce_to_shared(X, seed), y, False, mol_list


def _load_chembl_target(target, seed=42):
    """DRD2 / EGFR ChEMBL target-activity sets, active/inactive via pIC50>=7.0.

    Distinct from the DUD-E EGFR actives/decoys set: this is real potency data
    across chemically diverse programs, not actives-vs-synthetic-decoys. Mirrors
    nfmap_diagnostics.py's own branch so A2's numbers stay comparable.
    """
    import os
    import pandas as pd
    import datamol as dm
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.decomposition import PCA

    here = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(here, "..", "..", "..", "experiments",
                                  "chembed_generation", f"{target}_clean.csv"))
    X_list, y_list, mol_list = [], [], []
    for _, row in df.iterrows():
        mol = dm.to_mol(row["smiles"])
        if mol is not None:
            mol_list.append(mol)
            X_list.append(_get_features(mol))
            y_list.append(1 if row["pIC50"] >= 7.0 else 0)

    X_full = np.array(X_list, dtype=np.float32)
    y_full = np.array(y_list, dtype=np.int32)

    sss = StratifiedShuffleSplit(n_splits=1, train_size=min(len(y_full) - 1, 600), random_state=seed)
    idx, _ = next(sss.split(X_full, y_full))
    X, y = X_full[idx], y_full[idx]
    mol_list = [mol_list[i] for i in idx]

    return _reduce_to_shared(X, seed), y, False, mol_list


def _load_esol(seed=42):
    """ESOL aqueous solubility, binned to 3 classes (same thresholds as A2)."""
    import pandas as pd
    import datamol as dm
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.decomposition import PCA

    path = _fetch_cached(
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
        "esol.csv")
    df = pd.read_csv(path)
    X_list, y_list, mol_list = [], [], []
    for _, row in df.iterrows():
        mol = dm.to_mol(row["smiles"])
        if mol is not None:
            sol = row["measured log solubility in mols per litre"]
            mol_list.append(mol)
            X_list.append(_get_features(mol))
            y_list.append(0 if sol < -4 else (1 if sol < -2 else 2))

    X_full = np.array(X_list, dtype=np.float32)
    y_full = np.array(y_list, dtype=np.int32)

    sss = StratifiedShuffleSplit(n_splits=1, train_size=min(len(y_full) - 1, 600), random_state=seed)
    idx, _ = next(sss.split(X_full, y_full))
    X, y = X_full[idx], y_full[idx]
    mol_list = [mol_list[i] for i in idx]

    return _reduce_to_shared(X, seed), y, False, mol_list


def load_dataset(name, seed=42, return_mols=False):
    if name not in ALL_DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Choose from: {sorted(ALL_DATASETS)}")

    mols = None
    if name in TOY_DATASETS:
        X, y, is_continuous = _load_toy(name, seed=seed)
    elif name == "bace":
        X, y, is_continuous, mols = _load_bace(seed=seed)
    elif name == "bbbp":
        X, y, is_continuous, mols = _load_bbbp(seed=seed)
    elif name == "dude":
        X, y, is_continuous, mols = _load_dude(seed=seed)
    elif name == "esol":
        X, y, is_continuous, mols = _load_esol(seed=seed)
    elif name in ("drd2", "egfr"):
        X, y, is_continuous, mols = _load_chembl_target(name, seed=seed)

    out = {"X": X, "y": y, "is_continuous": is_continuous, "name": name}
    if return_mols:
        if mols is None:
            raise ValueError(f"{name!r} has no molecules (toy dataset)")
        out["mols"] = mols
    return out
