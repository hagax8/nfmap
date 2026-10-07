"""Fig 1: NFMap architecture diagram (Flow -> 2D map -> GMM prior).

Not data-driven (no results/ dependency) - a schematic illustration of the
NFMap architecture itself, built the same way as every other Paper 1 figure
(a standalone script, reproducible with one command) rather than a hand-drawn
image dropped into the repo.

Usage:
    python fig1_architecture_diagram.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path
import numpy as np

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")

# Palette: neutral ink/paper ground, one accent per architectural branch -
# blue for the shared flow backbone, green for the map/GMM branch (what the
# user navigates), amber for the thick-manifold branch (hidden detail).
INK = "#1b2320"
PAPER = "#f7f8f6"
BLUE = "#2c5f8a"
BLUE_BG = "#e4edf4"
GREEN = "#1f6f4a"
GREEN_BG = "#e3f0e8"
AMBER = "#a3671a"
AMBER_BG = "#f6ecdb"
GREY = "#8a938e"


def box(ax, xy, w, h, text, face, edge, fontsize=10, fontweight="bold", text_color=None):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4, edgecolor=edge, facecolor=face, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, fontweight=fontweight, color=text_color or edge,
             zorder=4, wrap=True)
    return patch


def arrow(ax, start, end, color=INK, style="-|>", lw=1.6, connectionstyle="arc3,rad=0.0", ls="-"):
    a = FancyArrowPatch(start, end, arrowstyle=style, color=color, linewidth=lw,
                         connectionstyle=connectionstyle, zorder=2, linestyle=ls,
                         mutation_scale=14)
    ax.add_patch(a)


def draw_prior_grid(ax, cx, cy, w, h, n=5):
    """Small GMM-grid icon: crosses on a lattice, echoing the 'X' prior-mean
    markers used in every other Paper 1 figure - visual continuity with the
    rest of the paper's figures, not just a generic diagram."""
    xs = np.linspace(cx - w / 2, cx + w / 2, n)
    ys = np.linspace(cy - h / 2, cy + h / 2, n)
    for xi in xs:
        for yi in ys:
            ax.plot(xi, yi, marker="x", color=GREEN, markersize=5, markeredgewidth=1.3, zorder=4)


def draw_scatter_blob(ax, cx, cy, w, h, seed=0, n=40, color=AMBER):
    rng = np.random.RandomState(seed)
    pts = rng.normal(size=(n, 2)) * [w / 2, h / 2] + [cx, cy]
    ax.scatter(pts[:, 0], pts[:, 1], s=7, color=color, alpha=0.75, zorder=4, linewidths=0)


def main():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)

    # ── Stage 1: input data ──────────────────────────────────────────────
    box(ax, (0.3, 2.9), 2.0, 1.2, "Input data $X$\n(e.g. Morgan\nfingerprints, $D$-dim)",
        BLUE_BG, BLUE, fontsize=9)

    # ── Stage 2: linear preprocessing ────────────────────────────────────
    box(ax, (2.7, 2.9), 1.7, 1.2, "PCA / SVD\npreprocessing\n$D \\to d_{scaled}$",
        BLUE_BG, BLUE, fontsize=8.5, fontweight="normal")
    arrow(ax, (2.3, 3.5), (2.7, 3.5))

    # ── Stage 3: normalizing flow (the shared backbone) ──────────────────
    box(ax, (4.8, 2.5), 2.3, 2.0,
        "Normalizing Flow $f_\\theta$\n(invertible coupling\nblocks, bijective)",
        BLUE_BG, BLUE, fontsize=10)
    arrow(ax, (4.4, 3.5), (4.8, 3.5))

    # Flow output splits into z_parallel (top, the map) and z_perp (bottom,
    # thick manifold) - this split IS the "Thick Manifold" novelty.
    arrow(ax, (7.1, 4.1), (7.9, 5.3), connectionstyle="arc3,rad=0.25", color=GREEN)
    arrow(ax, (7.1, 2.9), (7.9, 1.7), connectionstyle="arc3,rad=-0.25", color=AMBER)

    # ── Top branch: z_parallel -> the navigable 2D map ───────────────────
    box(ax, (7.9, 4.9), 2.0, 1.0, "$z_{\\parallel}$ (2D)\n\"the map\"", GREEN_BG, GREEN, fontsize=9)
    arrow(ax, (9.9, 5.4), (10.5, 5.4))
    map_ax_x, map_ax_y, map_w, map_h = 10.6, 4.7, 2.1, 1.4
    box(ax, (map_ax_x, map_ax_y), map_w, map_h, "", GREEN_BG, GREEN)
    draw_prior_grid(ax, map_ax_x + map_w / 2, map_ax_y + map_h / 2, map_w * 0.75, map_h * 0.65, n=5)
    ax.text(map_ax_x + map_w / 2, map_ax_y - 0.28, "GMM prior grid\n(fixed/learnable)",
            ha="center", va="top", fontsize=8, color=GREEN)

    # ── Bottom branch: z_perp -> thick-manifold storage vault ────────────
    box(ax, (7.9, 0.9), 2.0, 1.0, "$z_{\\perp}$ ($d_{scaled}-2$ dims)\n\"storage vault\"",
        AMBER_BG, AMBER, fontsize=9)
    arrow(ax, (9.9, 1.4), (10.5, 1.4))
    blob_ax_x, blob_ax_y, blob_w, blob_h = 10.6, 0.7, 2.1, 1.4
    box(ax, (blob_ax_x, blob_ax_y), blob_w, blob_h, "", AMBER_BG, AMBER)
    draw_scatter_blob(ax, blob_ax_x + blob_w / 2, blob_ax_y + blob_h / 2, blob_w * 0.7, blob_h * 0.6)
    ax.text(blob_ax_x + blob_w / 2, blob_ax_y - 0.28, "$\\sigma_{perp}$-scaled Gaussian\n(fine-grained detail)",
            ha="center", va="top", fontsize=8, color=AMBER)

    # ── Loss annotations feeding back into the flow (dashed, grey) ───────
    ax.annotate("", xy=(5.95, 4.5), xytext=(8.9, 5.9),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.1, ls="--",
                                 connectionstyle="arc3,rad=0.3"))
    ax.text(7.9, 6.15, "graph loss (fuzzy simplicial kernel)\n+ GMM prior loss",
            ha="center", va="bottom", fontsize=7.8, color=GREY, style="italic")

    ax.annotate("", xy=(5.95, 2.5), xytext=(8.9, 1.0),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.1, ls="--",
                                 connectionstyle="arc3,rad=-0.3"))
    ax.text(7.9, 0.35, "$\\sigma_{perp}$ / volume-scaling loss",
            ha="center", va="top", fontsize=7.8, color=GREY, style="italic")

    fig.suptitle("NFMap architecture: bijective flow, split into a navigable 2D map and a\n"
                  "high-dimensional ‘thick manifold’ storage vault, both shaped by training losses",
                  fontsize=12, y=0.99)

    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, "fig1_architecture_diagram.png")
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
