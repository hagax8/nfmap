"""Publication figures for the short paper — a deliberately small, curated set.

Separate from make_figures.py, which is the exploratory harness: that script
renders every experiment we ran, at whatever layout was convenient for reading
on screen. This one renders only what the paper prints, under a house style:

  * at most TWO panels across the width, so every panel is large enough to read
    at single-column width in print;
  * 300 dpi, vector-friendly font sizes, no clipped or overlapping text;
  * constrained_layout everywhere, so long titles and axis labels never collide;
  * metrics quoted as mean +/- sd over 3 seeds, never a single fit.

Output goes to figures_paper/ so a regeneration of the exploratory figures can
never silently overwrite a figure that is in the manuscript.

Usage:
    python make_paper_figures.py              # all
    python make_paper_figures.py --which f2   # one
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.io import RESULTS_ROOT

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_paper")

# DUD-E is dropped from the short paper: it duplicates EGFR, and its
# actives/decoys split is synthetic (decoys are property-matched but
# topologically dissimilar by construction), so separation there is partly a
# statement about how the benchmark was built rather than about chemistry.
CHEM = ["bace", "bbbp", "esol", "drd2", "egfr"]
MANIFOLD = ["swiss_roll", "s_curve", "severed_sphere"]

GTM_REF = {"bace": 0.428, "bbbp": 0.628, "esol": 0.689, "drd2": 0.394,
           "egfr": 0.446, "swiss_roll": 0.615, "s_curve": 0.627,
           "severed_sphere": 0.781}

PRETTY = {"bace": "BACE", "bbbp": "BBBP", "esol": "ESOL", "drd2": "DRD2",
          "egfr": "EGFR", "swiss_roll": "Swiss roll", "s_curve": "S-curve",
          "severed_sphere": "Severed sphere", "digits": "Digits",
          "wine": "Wine", "cancer": "Cancer", "iris": "Iris",
          "moons": "Moons", "circles": "Circles"}


def house_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.6,
        "figure.constrained_layout.use": True,
    })


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(FIG_DIR, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {os.path.join(FIG_DIR, name)}.{{png,pdf}}")


def load_json(exp_id, fname):
    path = os.path.join(RESULTS_ROOT, exp_id, fname)
    if not os.path.exists(path):
        print(f"  (missing {path})")
        return None
    with open(path) as f:
        return json.load(f)


def cells(rows, key="dist_corr"):
    acc = {}
    for r in rows:
        acc.setdefault((r["dataset"], r["sigma_perp"], r["grid_bounds"]), []).append(r)
    return {k: (float(np.mean([x[key] for x in v])),
                float(np.std([x[key] for x in v])),
                float(np.mean([x["frac_outside_grid_bounds"] for x in v])))
            for k, v in acc.items()}


# ── F2: the thesis figure ────────────────────────────────────────────────────

def fig_regimes():
    """One panel. The two response curves cross, which is the paper's claim."""
    print("F2 (sigma_perp response reverses with d_perp)...")
    chem = load_json("B2D", "sigma_grid_joint.json")
    man = load_json("B2E", "sigma_grid_manifold.json")
    if not chem or not man:
        return
    chem = [r for r in chem if r["dataset"] in CHEM]

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for rows, label, color, marker in [
            (chem, r"chemistry  ($d_\perp \approx 30$)", "#1f77b4", "o"),
            (man, r"3D manifolds  ($d_\perp = 1$)", "#d95f02", "s")]:
        c = cells(rows)
        dss = sorted({k[0] for k in c})
        xs, mus, sds = [], [], []
        for sg in sorted({k[1] for k in c}):
            per = [c[(d, sg, 12.0)][0] for d in dss if (d, sg, 12.0) in c]
            if len(per) < 2:
                continue
            xs.append(sg); mus.append(np.mean(per)); sds.append(np.std(per))
        ax.errorbar(xs, mus, yerr=sds, fmt=f"-{marker}", color=color, capsize=2.5,
                    markersize=4.5, label=label, elinewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\sigma_\perp$")
    ax.set_ylabel("DistCor")
    ax.set_title("Squeezing the off-manifold subspace helps only\n"
                 "when there is variance in it to redistribute")
    ax.legend(frameon=False, loc="center left")
    ax.text(0.98, 0.04, "grid_bounds = 12;  mean $\\pm$ sd over datasets, 3 seeds",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#666")
    save(fig, "f2_sigma_regimes")


# ── F4: sigma_perp x grid_bounds, two across ─────────────────────────────────

def _grid_panels(rows, datasets, name, title, note):
    c = cells(rows)
    datasets = [d for d in datasets if any(k[0] == d for k in c)]
    if not datasets:
        return
    sigmas = sorted({k[1] for k in c})
    grids = sorted({k[2] for k in c})
    ncol = 2
    nrow = int(np.ceil(len(datasets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 2.65 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    vals = [v[0] for k, v in c.items() if k[0] in datasets]
    vmin, vmax = min(vals), max(vals)

    im = None
    for ax, ds in zip(axes, datasets):
        M = np.full((len(sigmas), len(grids)), np.nan)
        for i, sg in enumerate(sigmas):
            for j, gb in enumerate(grids):
                v = c.get((ds, sg, gb))
                if v:
                    M[i, j] = v[0]
        im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis",
                       vmin=vmin, vmax=vmax)
        ax.grid(False)
        for i, sg in enumerate(sigmas):
            for j, gb in enumerate(grids):
                v = c.get((ds, sg, gb))
                if not v:
                    continue
                if v[2] > 0.05:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                               hatch="///", edgecolor="#c3312f",
                                               linewidth=0.0, alpha=.95))
                ax.text(j, i, f"{v[0]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v[0] < (vmin + vmax) / 2 else "#111")
        ax.set_xticks(range(len(grids)))
        ax.set_xticklabels([f"{g:g}" for g in grids])
        ax.set_yticks(range(len(sigmas)))
        ax.set_yticklabels([f"{s:g}" for s in sigmas])
        ref = GTM_REF.get(ds)
        ax.set_title(f"{PRETTY.get(ds, ds)}" + (f"   (GTM {ref:.2f})" if ref else ""),
                     fontsize=9)
    for ax in axes[len(datasets):]:
        ax.set_visible(False)
    # The x label belongs on the lowest VISIBLE panel of each column, which is
    # not the same as the last row when the grid is partly empty (5 datasets in
    # a 2x3 grid leaves the second column one panel short).
    bottom = {}
    for i in range(len(datasets)):
        bottom[i % ncol] = i
    for i, ax in enumerate(axes[:len(datasets)]):
        if i in bottom.values():
            ax.set_xlabel("grid_bounds")
        if i % ncol == 0:
            ax.set_ylabel(r"$\sigma_\perp$")
    cb = fig.colorbar(im, ax=axes.tolist(), shrink=0.6, pad=0.02, aspect=28)
    cb.set_label("DistCor", fontsize=8)
    fig.suptitle(title, fontsize=10.5)
    fig.text(0.01, -0.012, note, fontsize=7, color="#555", ha="left")
    save(fig, name)


def fig_grid_chem():
    print("F4 (sigma_perp x grid_bounds, chemistry)...")
    rows = load_json("B2D", "sigma_grid_joint.json")
    if rows:
        _grid_panels([r for r in rows if r["dataset"] in CHEM], CHEM,
                     "f4_sigma_grid_chemistry",
                     r"$\sigma_\perp$ and grid_bounds cannot be chosen independently",
                     "Hatched = >5% of points outside grid_bounds. DistCor rises "
                     "monotonically as $\\sigma_\\perp$ falls, so the operating point is "
                     "set by the containment constraint, not by an interior optimum.")


def fig_grid_manifold():
    print("F5 (sigma_perp x grid_bounds, manifolds)...")
    rows = load_json("B2E", "sigma_grid_manifold.json")
    if rows:
        _grid_panels(rows, MANIFOLD, "f5_sigma_grid_manifold",
                     r"The same sweep at $d_\perp = 1$ inverts the response",
                     "Hatched = >5% of points outside grid_bounds. Low "
                     "$\\sigma_\\perp$ now HURTS: with one off-manifold dimension there "
                     "is no variance reservoir to redistribute into the map.")


# ── F6: repulsion ────────────────────────────────────────────────────────────

def fig_repulsion():
    print("F6 (repulsion_weight at the chemistry operating point)...")
    rows = load_json("B4D", "repulsion_at_global.json")
    if not rows:
        return
    rows = [r for r in rows if r["dataset"] in CHEM]
    acc = {}
    for r in rows:
        acc.setdefault(r["repulsion_weight"], []).append(r)
    ws = sorted(acc)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    for ax, key, lbl in [(axes[0], "dist_corr", "DistCor"),
                         (axes[1], "frac_outside_grid_bounds",
                          "points outside grid_bounds (%)")]:
        mus = np.array([np.mean([x[key] for x in acc[w]]) for w in ws])
        sds = np.array([np.std([x[key] for x in acc[w]]) for w in ws])
        if key.startswith("frac"):
            mus, sds = 100 * mus, 100 * sds
        ax.errorbar(ws, mus, yerr=sds, fmt="-o", color="#7570b3", capsize=2.5,
                    markersize=4.5, elinewidth=1)
        ax.set_xlabel("repulsion_weight")
        ax.set_ylabel(lbl)
    axes[1].axhline(5.0, ls="--", color="#c3312f", lw=1.1)
    axes[1].text(0.97, 5.6, "containment constraint", transform=axes[1].get_yaxis_transform(),
                 ha="right", va="bottom", fontsize=7, color="#c3312f")
    fig.suptitle("Repulsion does not buy global structure at the chemistry\n"
                 r"operating point ($\sigma_\perp$=0.003, grid_bounds=12)", fontsize=10.5)
    fig.text(0.01, -0.02, "5 chemistry datasets, 3 seeds. At the shipped defaults "
             r"($\sigma_\perp$=10, grid_bounds=6) DistCor RISES with repulsion; here it falls.",
             fontsize=7, color="#555", ha="left")
    save(fig, "f6_repulsion")


# ── F7: density as an applicability-domain signal ────────────────────────────

def fig_ood():
    print("F7 (log-likelihood as an applicability-domain signal)...")
    rows = load_json("B14", "density_ood.json")
    if not rows:
        return
    # ESOL's novel_classes cell is excluded: ESOL is a regression target binned
    # into 3 classes, so holding out the top bin is an extrapolation split, not
    # an OOD one - all three signals score below chance there, which is the tell.
    rows = [r for r in rows
            if r["dataset"] != "dude"
            and not (r["dataset"] == "esol" and r["ood"] == "novel_classes")]
    SIG = [("nfmap_loglik", "NFMap log-likelihood", "#1b9e77"),
           ("knn_distance", "kNN distance", "#7570b3"),
           ("pca_recon_error", "PCA reconstruction error", "#999999")]
    acc = {}
    for r in rows:
        acc.setdefault((r["dataset"], r["signal"]), []).append(r["auc"])
    chem = [d for d in CHEM if any(k[0] == d for k in acc)]
    other = sorted({k[0] for k in acc} - set(chem))

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.4), sharey=True)
    for ax, group, title in [(axes[0], chem, "Chemistry"),
                             (axes[1], other, "Non-chemistry")]:
        x = np.arange(len(group))
        w = 0.27
        for i, (sig, lbl, color) in enumerate(SIG):
            mus = [np.mean(acc.get((d, sig), [np.nan])) for d in group]
            sds = [np.std(acc.get((d, sig), [np.nan])) for d in group]
            ax.bar(x + (i - 1) * w, mus, w, yerr=sds, capsize=1.8, color=color,
                   label=lbl if ax is axes[0] else None, edgecolor="white",
                   linewidth=0.4)
        ax.axhline(0.5, ls="--", color="#555", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([PRETTY.get(d, d) for d in group], rotation=20, ha="right")
        ax.set_ylabel("OOD detection AUC")
        ax.set_title(title, fontsize=9.5)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, ncol=3, loc="lower center", fontsize=7.5)
    fig.suptitle("NFMap's exact log-likelihood as an applicability-domain signal",
                 fontsize=10.5)
    fig.text(0.01, -0.015, "Mean over OOD constructions and 3 seeds; dashed line is "
             "chance. Ahead of the kNN baseline on every chemistry set; behind on the "
             r"2D toys, where $z_\perp$ is nearly empty.", fontsize=7, color="#555")
    save(fig, "f7_applicability_domain")



# ── Map comparisons ──────────────────────────────────────────────────────────

METHOD_LABEL = {"nfmap_global": r"NFMap  ($\sigma_\perp$=0.003, global)",
                "nfmap_local":  r"NFMap  ($\sigma_\perp$=10, local)",
                "nfmap":        r"NFMap  ($\sigma_\perp$=1)",
                "gtm": "GTM", "tsne": "t-SNE", "umap": "UMAP"}

CLASS_NAMES = {
    "drd2": {0: "inactive (pIC50<7)", 1: r"active (pIC50$\geq$7)"},
    "bbbp": {0: "non-penetrant", 1: "penetrant"},
    "bace": {0: "inactive", 1: "active"},
    "egfr": {0: "inactive (pIC50<7)", 1: r"active (pIC50$\geq$7)"},
    "esol": {0: "low sol.", 1: "mid sol.", 2: "high sol."},
}


def _load_run(exp_id, ds, kind, seed=42):
    d = os.path.join(RESULTS_ROOT, exp_id, f"{ds}_{kind}_seed{seed}")
    if not os.path.isdir(d):
        return None
    with open(os.path.join(d, "metrics.json")) as f:
        m = json.load(f)
    npz = np.load(os.path.join(d, "embeddings.npz"), allow_pickle=True)
    return {"Z": npz["Z"], "y": npz["y"], "metrics": m}


def _seed_mean(exp_id, ds, kind, key, seeds=(42, 0, 1)):
    vals = []
    for s in seeds:
        r = _load_run(exp_id, ds, kind, s)
        if r and r["metrics"].get(key) is not None and not np.isnan(r["metrics"][key]):
            vals.append(r["metrics"][key])
    return (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)


def _map_panel(ax, run, title, ds, is_cont):
    Z, y = run["Z"], run["y"]
    if Z is None or np.any(np.isnan(Z)):
        ax.text(.5, .5, "N/A", ha="center", va="center", transform=ax.transAxes,
                color="#c3312f")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9)
        return
    if is_cont:
        ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="viridis", s=7, alpha=.85, linewidths=0)
    else:
        names = CLASS_NAMES.get(ds, {})
        for cls in np.unique(y):
            mask = y == cls
            ax.scatter(Z[mask, 0], Z[mask, 1], s=7, alpha=.8, linewidths=0,
                       label=names.get(int(cls), f"class {int(cls)}"))
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)


def _comparison(exp_id, ds, kinds, name, suptitle, is_cont=False, ncol=2):
    runs = [(k, _load_run(exp_id, ds, k)) for k in kinds]
    runs = [(k, r) for k, r in runs if r is not None]
    if not runs:
        print(f"  (no saved runs for {ds} in {exp_id} - re-run `make a1`/`make a2`)")
        return
    nrow = int(np.ceil(len(runs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.6, 3.3 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (kind, r) in zip(axes, runs):
        dc, n = _seed_mean(exp_id, ds, kind, "dist_corr")
        tc, _ = _seed_mean(exp_id, ds, kind, "t_c_f1")
        _map_panel(ax, r, METHOD_LABEL.get(kind, kind), ds, is_cont)
        ax.set_xlabel(f"DistCor {dc:.2f}   T/C F1 {tc:.2f}", fontsize=7.5, color="#444")
    for ax in axes[len(runs):]:
        ax.set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="lower center", ncol=min(3, len(l)), frameon=False,
                   fontsize=7.5, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(suptitle, fontsize=10.5)
    fig.text(0.01, -0.055 if h else -0.02,
             f"Maps are seed 42; metrics are means over 3 seeds.", fontsize=7, color="#666")
    save(fig, name)


def fig_maps_chemistry():
    print("F3 (chemistry maps: the dial vs the baselines)...")
    _comparison("A2", "drd2",
                ["nfmap_global", "nfmap_local", "gtm", "tsne", "umap"],
                "f3_maps_drd2",
                "DRD2: both ends of the dial, against the baselines")


def fig_maps_manifold():
    print("F8 (manifold maps)...")
    # S-curve rather than the swiss roll: less of a stock benchmark, and the
    # unrolling is legible at a glance (one continuous sheet or two broken
    # strips). No dial panel here - at d_perp=1 the two ends of sigma_perp are
    # visually near-identical (DistCor 0.69 vs 0.68), which is itself the point.
    #
    # NOTE for the caption: this figure carries the UNROLLING, not the margin.
    # S-curve's lead over GTM is +0.059, inside the ~0.15 per-fit spread, so the
    # numbers belong in Table 1 with their seeds, not in a claim under a map.
    _comparison("A1", "s_curve", ["nfmap", "gtm", "tsne", "umap"],
                "f8_maps_s_curve",
                r"S-curve ($d_\perp$=1): one setting, since the dial does little here",
                is_cont=True)


def fig_maps_supplementary():
    print("S1/S2 (complete map grids)...")
    for ds in CHEM:
        _comparison("A2", ds, ["nfmap_global", "nfmap_local", "gtm", "tsne", "umap"],
                    f"s1_maps_{ds}", f"{PRETTY.get(ds, ds)}: all methods")
    for ds in MANIFOLD:
        _comparison("A1", ds, ["nfmap", "gtm", "tsne", "umap"],
                    f"s2_maps_{ds}", f"{PRETTY.get(ds, ds)}: all methods", is_cont=True)


DISPATCH = {"f2": fig_regimes,
            "f3": fig_maps_chemistry, "f8": fig_maps_manifold,
            "supp": fig_maps_supplementary, "f4": fig_grid_chem, "f5": fig_grid_manifold,
            "f6": fig_repulsion, "f7": fig_ood}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", default="all",
                    help="comma-separated subset of: " + ",".join(DISPATCH))
    args = ap.parse_args()
    house_style()
    keys = list(DISPATCH) if args.which == "all" else args.which.split(",")
    for k in keys:
        if k not in DISPATCH:
            raise SystemExit(f"unknown figure {k!r}; choose from {', '.join(DISPATCH)}")
        DISPATCH[k]()


if __name__ == "__main__":
    main()
