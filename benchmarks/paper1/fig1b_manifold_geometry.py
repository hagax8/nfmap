"""Fig 1b: geometric intuition for z_parallel / z_perp.

Not data-driven - an illustrative schematic (same policy as
fig1_architecture_diagram.py). Two panels:

Left  (latent space, z): the flow's target manifold {z_perp=0} is an exact
      flat plane in R^d by construction. z_parallel spans that plane;
      z_perp spans its exact Euclidean orthogonal complement - "parallel"
      and "perpendicular" are literally, trivially true here.
Right (data space, x): the *image* of that flat plane under the (nonlinear)
      flow f_theta is generally a *curved* manifold. Moving along z_perp and
      mapping back through f_theta gives the intended "off-manifold"
      direction, but it is only exactly orthogonal to the manifold's own
      tangent space in data space if the flow is locally isometric there
      (Jacobian orthogonal) - i.e. only if the optional canonical/isometric
      loss (Sec 3.8/7.8, off by default in this paper) were active. Without
      it, "perpendicular" in data space is the design target the geometric
      naming is aiming for, not a property guaranteed to hold exactly.

Usage:
    python fig1b_manifold_geometry.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")

# Same palette as fig1_architecture_diagram.py for visual continuity.
INK = "#1b2320"
PAPER = "#f7f8f6"
BLUE = "#2c5f8a"
GREEN = "#1f6f4a"
GREEN_BG = "#e3f0e8"
AMBER = "#a3671a"
AMBER_BG = "#f6ecdb"
GREY = "#8a938e"


def draw_panel(ax, curved, title, subtitle):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    n = 26
    u = np.linspace(-1.4, 1.4, n)
    v = np.linspace(-1.4, 1.4, n)
    U, V = np.meshgrid(u, v)
    if curved:
        # A gentle bend - z_scaled has no analytic form, this is purely
        # illustrative of "curved in general", not a fit to real data.
        W = 0.35 * np.sin(1.3 * U) * np.cos(1.1 * V) + 0.15 * (U ** 2 - V ** 2) * 0.3
    else:
        W = np.zeros_like(U)  # exactly flat - the latent-space manifold {z_perp=0}

    ax.plot_surface(U, V, W, color="#dbe7de", edgecolor="#9fb8a8", linewidth=0.2,
                     alpha=0.9, rstride=1, cstride=1, antialiased=True, zorder=1)

    # Marked point + local tangent-plane patch + normal vector, computed from
    # the true local gradient (curved panel) or trivially flat (latent panel).
    u0, v0 = 0.15, -0.1
    if curved:
        du, dv = 1e-3, 1e-3
        f = lambda uu, vv: 0.35 * np.sin(1.3 * uu) * np.cos(1.1 * vv) + 0.15 * (uu ** 2 - vv ** 2) * 0.3
        w0 = f(u0, v0)
        t_u = np.array([1, 0, (f(u0 + du, v0) - w0) / du])
        t_v = np.array([0, 1, (f(u0, v0 + dv) - w0) / dv])
    else:
        w0 = 0.0
        t_u = np.array([1, 0, 0])
        t_v = np.array([0, 1, 0])
    t_u = t_u / np.linalg.norm(t_u)
    t_v_orth = t_v - np.dot(t_v, t_u) * t_u
    t_v_orth = t_v_orth / np.linalg.norm(t_v_orth)
    normal = np.cross(t_u, t_v_orth)
    normal = normal / np.linalg.norm(normal)

    p0 = np.array([u0, v0, w0])

    # Tangent-plane patch (small square spanned by t_u, t_v_orth) = z_parallel -
    # a bright, high-contrast fill so it visibly pops off the underlying mesh.
    s = 0.6
    corners = [p0 + s * (a * t_u + b * t_v_orth) for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    corners = np.array(corners)
    tangent_patch = Poly3DCollection([corners], facecolor=GREEN, edgecolor="#0d3d24",
                                      linewidth=1.4, alpha=0.75, zorder=5)
    ax.add_collection3d(tangent_patch)

    # Normal "arrow" drawn as a thick line + cone marker (quiver alone
    # renders too thin/faint against the mesh in mplot3d).
    n_len = 1.15
    tip = p0 + normal * n_len
    ax.plot([p0[0], tip[0]], [p0[1], tip[1]], [p0[2], tip[2]],
            color=AMBER, linewidth=4.2, zorder=6, solid_capstyle="round")
    ax.scatter(*tip, color=AMBER, s=90, marker="^", zorder=7, edgecolor="#5c3a0e", linewidth=0.8)
    ax.scatter(*p0, color=INK, s=70, zorder=7, edgecolor="white", linewidth=1.2)

    label_pt = p0 + normal * (n_len + 0.32)
    ax.text(*label_pt, r"$z_{\perp}$", color=AMBER, fontsize=17, fontweight="bold", zorder=8, ha="center")
    tangent_label_pt = p0 + s * (t_u + t_v_orth) * 1.15
    ax.text(*tangent_label_pt, r"$z_{\parallel}$", color="#0d3d24", fontsize=17, fontweight="bold", zorder=8, ha="center")

    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_zlim(-1.3, 1.3)
    ax.set_box_aspect((1, 1, 0.75))
    ax.set_axis_off()
    ax.view_init(elev=20, azim=-50)
    ax.set_title(title, fontsize=13, fontweight="bold", color=INK, pad=-4)
    ax.text2D(0.5, 0.04, subtitle, transform=ax.transAxes, ha="center", va="top",
              fontsize=9, color=GREY, style="italic", wrap=True)


def main():
    fig = plt.figure(figsize=(11.5, 4.6))
    fig.patch.set_facecolor(PAPER)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax1.set_facecolor(PAPER)
    ax2.set_facecolor(PAPER)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.04, wspace=0.05)

    draw_panel(ax1, curved=False,
               title="Latent space  (z ∈ ℝᵈ)",
               subtitle="{z⊥ = 0} is an exact flat plane by construction —\nz∥, z⊥ are exact Euclidean orthogonal complements.")
    draw_panel(ax2, curved=True,
               title="Data space  (x ∈ ℝᴰ, after fθ)",
               subtitle="The image of that plane is generally curved —\n\"z⊥ ⊥ manifold\" holds exactly only if fθ is locally\nisometric there (§3.8's optional loss, off by default).")

    fig.suptitle("What \"parallel\" and \"perpendicular\" mean: exact in latent space, the design target in data space",
                  fontsize=12.5, fontweight="bold", color=INK, y=0.99)

    # Arrow between panels indicating the flow mapping
    fig.text(0.5, 0.46, r"$f_{\theta}$", ha="center", va="center", fontsize=15, color=BLUE, fontweight="bold")
    fig.patches.append(plt.matplotlib.patches.FancyArrowPatch(
        (0.465, 0.42), (0.535, 0.42), transform=fig.transFigure,
        arrowstyle="-|>", color=BLUE, linewidth=1.8, mutation_scale=16, zorder=10))

    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, "fig1b_manifold_geometry.png")
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
