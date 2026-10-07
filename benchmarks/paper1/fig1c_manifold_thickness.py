"""Fig 1c: what "thickness" actually means - a populated-band visualization.

Companion to fig1b_manifold_geometry.py (which shows a single point's tangent/
normal directions). This figure addresses a different, narrower question:
not "is there a direction off the manifold" but "what does it mean for the
manifold to have genuine thickness there, rather than being razor-thin."

Two panels, same patch of map, same set of underlying map locations: a thin
band (small sigma_perp, Brehmer & Cranmer's razor-thin regime) where sample
points hug the central sheet tightly vs. a thick band (large sigma_perp,
NFMap's default) where points spread further through the band - each point
still belongs to roughly the same 2D map location (similar x,y) but differs
in how far "into the thickness" it sits, i.e. in z_perp. Not data-driven - a
schematic, same policy as fig1_architecture_diagram.py and fig1b.

Usage:
    python fig1c_manifold_thickness.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")

INK = "#1b2320"
PAPER = "#f7f8f6"
GREEN = "#1f6f4a"
AMBER = "#a3671a"
GREY = "#8a938e"


def draw_panel(ax, sigma_perp, title, subtitle, seed=0):
    n = 22
    u = np.linspace(-1.3, 1.3, n)
    v = np.linspace(-1.3, 1.3, n)
    U, V = np.meshgrid(u, v)
    W = np.zeros_like(U)  # central sheet drawn flat for clarity - thickness
                           # is the point here, not curvature (that's fig1b)

    ax.plot_surface(U, V, W, color="#dbe7de", edgecolor="#9fb8a8", linewidth=0.15,
                     alpha=0.55, rstride=1, cstride=1, antialiased=True, zorder=1)

    # Sample "molecules" scattered across the map (random x,y) with a z
    # offset drawn from N(0, sigma_perp) - this z offset IS z_perp. Same
    # (x,y) cloud and same random seed for both panels, only sigma_perp
    # differs, so the two panels are a fair, direct comparison.
    rng = np.random.RandomState(seed)
    n_pts = 70
    xs = rng.uniform(-1.15, 1.15, n_pts)
    ys = rng.uniform(-1.15, 1.15, n_pts)
    zs = rng.normal(0, sigma_perp, n_pts)
    # clip for display only, so the thick panel doesn't blow past the axes
    zs = np.clip(zs, -1.1, 1.1)

    sc = ax.scatter(xs, ys, zs, c=zs, cmap="RdBu_r", vmin=-1.1, vmax=1.1,
                     s=28, alpha=0.9, edgecolors="white", linewidths=0.4, zorder=5)

    # A thin vertical guide + label at one highlighted point, to anchor "this
    # molecule's position within the thickness = its z_perp value"
    hi = np.argmax(np.abs(zs))
    ax.plot([xs[hi], xs[hi]], [ys[hi], ys[hi]], [0, zs[hi]],
            color=AMBER, linewidth=1.6, linestyle="--", zorder=6)

    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_zlim(-1.3, 1.3)
    ax.set_box_aspect((1, 1, 0.8))
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-55)
    ax.set_title(title, fontsize=12.5, fontweight="bold", color=INK, pad=-4)
    ax.text2D(0.5, 0.14, subtitle, transform=ax.transAxes, ha="center", va="top",
              fontsize=8.6, color=GREY, style="italic")
    return sc


def main():
    fig = plt.figure(figsize=(11.5, 5.6))
    fig.patch.set_facecolor(PAPER)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax1.set_facecolor(PAPER)
    ax2.set_facecolor(PAPER)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.83, bottom=0.2, wspace=0.08)

    draw_panel(ax1, sigma_perp=0.04, seed=7,
               title="Razor-thin (small σ⊥, Brehmer & Cranmer)",
               subtitle="Every molecule at this map location is forced to nearly\nthe same z⊥ ≈ 0 — little room to differ from its neighbors.")
    sc = draw_panel(ax2, sigma_perp=0.55, seed=7,
                     title="Thick (large σ⊥, NFMap default)",
                     subtitle="Molecules sharing a broad map location still spread\nthrough real z⊥ range — room for fine-grained differences.")

    cbar_ax = fig.add_axes([0.32, 0.07, 0.36, 0.035])
    cbar = fig.colorbar(sc, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("z⊥  (position within the thickness, for this map location)", fontsize=8.5, color=INK)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle("What \"thickness\" means: molecules sharing a 2D map location (x,y) can still\n"
                 "differ in z⊥ — their depth within the manifold's thickness at that location",
                 fontsize=12, fontweight="bold", color=INK, y=0.99)

    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, "fig1c_manifold_thickness.png")
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
