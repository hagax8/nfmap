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

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, manifold
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedShuffleSplit

from nfmap import NFMap

try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("UMAP not installed — its panel will show a placeholder.")

try:
    from ugtm import eGTM
    HAS_GTM = True
except ImportError:
    HAS_GTM = False
    print("ugtm not installed — GTM panel will show a placeholder.")

# ── Data: stratified 600-sample subset ───────────────────────────────────────
digits = datasets.load_digits(n_class=10)
sss = StratifiedShuffleSplit(n_splits=1, train_size=600, random_state=42)
train_idx, _ = next(sss.split(digits.data, digits.target))
X = digits.data[train_idx].astype(np.float32)
y = digits.target[train_idx]
print(f"Subset: {X.shape[0]} samples × {X.shape[1]} features, {len(np.unique(y))} classes\n")

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

    {
        "label":      "NEW: NFMap (Best)",
        "hypothesis": "Max epochs 1000, sigma_perp 10.0, graph_weight 5.0",
        "kind":       "nfmap",
        "overrides":  {"max_epochs": 1000, "patience": 100, "sigma_perp": 10.0, "graph_weight": 5.0},
    },
    {
        "label":      "NEW: NFMap (Low Graph Weight)",
        "hypothesis": "Max epochs 1000, sigma_perp 10.0, graph_weight 1.0",
        "kind":       "nfmap",
        "overrides":  {"max_epochs": 1000, "patience": 100, "sigma_perp": 10.0, "graph_weight": 1.0},
    },
    {
        "label":      "NEW: NFMap (No Graph Weight)",
        "hypothesis": "Max epochs 1000, sigma_perp 10.0, graph_weight 0.0 (Pure Injective Flow)",
        "kind":       "nfmap",
        "overrides":  {"max_epochs": 1000, "patience": 100, "sigma_perp": 0.05, "graph_weight": 0.0},
    },
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
for i, cfg in enumerate(CONFIGS):
    label_flat = cfg["label"].replace('\n', ' ')
    print(f"[{i+1}/{len(CONFIGS)}] {label_flat}")

    try:
        if cfg["kind"] == "nfmap":
            params = {**DEFAULTS, **cfg["overrides"]}
            Z = NFMap(**params).fit_transform(X)

        elif cfg["kind"] == "tsne":
            Z = manifold.TSNE(
                n_components=2, init='pca', random_state=42, perplexity=30
            ).fit_transform(X)

        elif cfg["kind"] == "gtm":
            if HAS_GTM:
                Z = eGTM().fit_transform(X)
            else:
                Z = np.full((len(y), 2), np.nan)

        elif cfg["kind"] == "umap":
            if HAS_UMAP:
                Z = umap_lib.UMAP(
                    n_neighbors=15, min_dist=0.1, random_state=42
                ).fit_transform(X)
            else:
                Z = np.full((len(y), 2), np.nan)

        sil = silhouette_score(Z, y) if not np.any(np.isnan(Z)) else float('nan')
        print(f"    silhouette = {sil:.4f}")
        results.append({"label": cfg["label"], "hypothesis": cfg["hypothesis"],
                         "Z": Z, "sil": sil, "ok": True})

    except Exception as e:
        print(f"    FAILED: {e}")
        results.append({"label": cfg["label"], "hypothesis": cfg["hypothesis"],
                         "Z": None, "sil": None, "ok": False})

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
cmap = plt.cm.get_cmap('tab10', 10)
last_sc = None  # for colorbar

for ax, res in zip(axes, results):
    if res["ok"] and res["Z"] is not None and not np.any(np.isnan(res["Z"])):
        sc = ax.scatter(
            res["Z"][:, 0], res["Z"][:, 1],
            c=y, cmap=cmap, vmin=0, vmax=9,
            s=15, alpha=0.85, linewidths=0,
        )
        last_sc = sc
        sil_str = f"  sil={res['sil']:.3f}"
    else:
        ax.text(0.5, 0.5,
                "FAILED" if not res["ok"] else "N/A",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=14, color='#cc0000')
        sil_str = ""

    ax.set_title(res["label"] + sil_str, fontsize=10, fontweight='bold')
    ax.set_xlabel(res["hypothesis"], fontsize=8, color='#555555', labelpad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('tight')

# Shared colorbar for digit classes
if last_sc is not None:
    cb = fig.colorbar(last_sc, ax=axes.tolist(), label='Digit class (0–9)',
                      shrink=0.45, pad=0.02, aspect=30)
    cb.set_ticks(np.arange(10))

fig.suptitle(
    "NFMap Diagnostics — Digits (600-sample stratified subset)\n"
    "Each panel isolates one hypothesis; t-SNE/UMAP shown as reference baselines.",
    fontsize=13, y=1.02,
)
plt.tight_layout()

out_path = "nfmap_diagnostics.png"
plt.savefig(out_path, bbox_inches='tight', dpi=150)
print(f"\nSaved → {out_path}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 58)
print(f"  {'Config':<38} {'Silhouette':>10}")
print("=" * 58)
for res in results:
    name = res["label"].replace('\n', ' ')
    sil_str = f"{res['sil']:.4f}" if res["sil"] is not None and not np.isnan(res["sil"]) else "FAILED / N/A"
    print(f"  {name:<38} {sil_str:>10}")
print("=" * 58)

plt.show()
