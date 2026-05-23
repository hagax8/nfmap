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

import argparse
import math
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

# ── Arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="NFMap Diagnostics")
parser.add_argument("--dataset", type=str, default="digits", choices=["digits", "wine", "cancer", "iris", "moons", "circles"], help="Dataset to evaluate on")
args = parser.parse_args()

# ── Data Loading ──────────────────────────────────────────────────────────────
np.random.seed(42)
if args.dataset == "digits":
    digits = datasets.load_digits(n_class=10)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=min(len(digits.target)-1, 600), random_state=42)
    train_idx, _ = next(sss.split(digits.data, digits.target))
    X_subset = digits.data[train_idx].astype(np.float32)
    y_subset = digits.target[train_idx]
    X_scaled = X_subset / 16.0
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

    {
        "label":      "NEW: NFMap (Best)",
        "hypothesis": "Max epochs 1000, sigma_perp 10.0, graph_weight 5.0",
        "kind":       "nfmap",
        "overrides":  {"max_epochs": 1000, "patience": 100, "sigma_perp": 10.0, "graph_weight": 5.0},
    },
    {
        "label":      "NEW: NFMap (Negative Sampling x5)",
        "hypothesis": "Sample 5 negative edges per positive to push clusters apart",
        "kind":       "nfmap",
        "overrides":  {"max_epochs": 1000, "patience": 100, "sigma_perp": 10.0, "graph_weight": 5.0, "negative_sample_rate": 5},
    },
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
            c=y, cmap='tab10', vmin=0, vmax=9,
            s=15, alpha=0.85, linewidths=0,
        )
        scatter_plots.append(sc)
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

# Hide any unused axes
for i in range(len(results), len(axes)):
    axes[i].set_visible(False)

# Shared legend for classes
if scatter_plots:
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
print("\n" + "=" * 58)
print(f"  {'Config':<38} {'Silhouette':>10}")
print("=" * 58)
for res in results:
    name = res["label"].replace('\n', ' ')
    sil_str = f"{res['sil']:.4f}" if res["sil"] is not None and not np.isnan(res["sil"]) else "FAILED / N/A"
    print(f"  {name:<38} {sil_str:>10}")
print("=" * 58)

# Close matplotlib to prevent blocking script execution
plt.close('all')
