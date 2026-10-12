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

# Fits in this module run on the CPU backend deliberately. Measured: an NFMap
# fit on MPS is not reproducible across processes (max|dZ| ~ 4.7 on a +/-12
# canvas at 300 epochs), because a float32-epsilon perturbation introduced
# somewhere in the MPS path amplifies through training; on CPU the same seed is
# bit-identical, and at these problem sizes CPU is also faster (9.2s vs 11.2s
# on 600x32, 100 epochs). Patched before nfmap is imported anywhere.
import torch as _torch
_torch.backends.mps.is_available = lambda: False

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

def _gtm_ref(datasets):
    """GTM's DistCor as reported in Table 1 (make_paper_tables), for the panel
    titles. Was a hard-coded dict from the retired A1 runs (256-node GTM),
    which no longer matched Table 1."""
    from make_paper_tables import table1_selection
    return {ds: k["GTM"][0] for ds, k in table1_selection(datasets).items() if "GTM" in k}


def _grid_panels(rows, datasets, name, title, note):
    gtm_ref = _gtm_ref(datasets)
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
        ref = gtm_ref.get(ds)
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
                     "Hatched = >5% of points outside grid_bounds. DistCor broadly "
                     "rises as $\\sigma_\\perp$ falls, so the operating point is "
                     "set by the containment constraint, not by an interior optimum.")


# ── F6: repulsion ────────────────────────────────────────────────────────────

# ── F7: density as an applicability-domain signal ────────────────────────────

def fig_ood():
    print("F7 (log-likelihood as an applicability-domain signal)...")
    rows = load_json("B14", "density_ood.json")
    if not rows:
        return
    # DUD-E excluded by choice: EGFR and DRD2 already cover target-based
    # activity sets. (NFMap leads there too, 0.908 vs 0.849.)
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
    fig.text(0.01, -0.035, "Mean over the applicable OOD constructions (permuted, noise; "
             "plus novel classes where there are $\\geq$3 classes) and 3 seeds;\n"
             "dashed line is chance. Ahead of the kNN baseline on every chemistry set; "
             r"behind on the 2D toys, where $z_\perp$ is empty.", fontsize=7, color="#555")
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


CHEM_SIGMA = {"global": 0.003, "local": 10.0}


def _fit_chemistry(ds, sigma, seed=42):
    """Load a pre-fitted chemistry map from b22_fit_manifold_maps.py.

    Previously fitted inline here, which crashes unpredictably (exit 139) in
    this module's full process - see _fit_manifold's docstring. b22 fits and
    saves drd2_global/drd2_local separately; this just loads the result.
    """
    name = {0.003: "drd2_global", 10.0: "drd2_local"}.get(sigma)
    if name is None:
        raise ValueError(f"no saved fit for sigma_perp={sigma}; add it to "
                         f"b22_fit_manifold_maps.py's CHEM_SIGMA")
    npz_path = os.path.join(RESULTS_ROOT, "B22", f"{name}.npz")
    met_path = os.path.join(RESULTS_ROOT, "B22", f"{name}_metrics.json")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"{npz_path} missing - run `python b22_fit_manifold_maps.py` first")
    data = np.load(npz_path)
    with open(met_path) as f:
        met = json.load(f)
    return {"Z": data["Z"], "y": data["y"], "metrics": met}


def fig_maps_chemistry():
    """DRD2 at both ends of the dial, against the baselines.

    NFMap panels are fitted by b22 at both ends of the dial (CPU; A2's were
    MPS and not reproducible). The baselines are b22's fits at exactly the
    settings Table 1 reports for DRD2 (make_paper_tables.table1_selection), so
    each panel matches its table row; they used to be A2's untuned defaults.
    """
    print("F3 (chemistry maps: the dial vs the baselines)...")
    ds = "drd2"
    runs = [(rf"NFMap ($\sigma_\perp$={CHEM_SIGMA['global']:g}, global)",
             _fit_chemistry(ds, CHEM_SIGMA["global"])),
            (rf"NFMap ($\sigma_\perp$={CHEM_SIGMA['local']:g}, local)",
             _fit_chemistry(ds, CHEM_SIGMA["local"]))]
    for kind, lbl in (("gtm", "GTM"), ("tsne", "t-SNE"), ("umap", "UMAP")):
        try:
            r = _fit_baseline(ds, kind, None)
        except FileNotFoundError:
            continue
        runs.append((lbl + _setting_tag(kind, r["metrics"].get("setting", {})), r))
    nrow = int(np.ceil(len(runs) / 2))
    fig, axes = plt.subplots(nrow, 2, figsize=(6.6, 3.3 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (label, r) in zip(axes, runs):
        _map_panel(ax, r, label, ds, False)
        m = r["metrics"]
        ax.set_xlabel(f"DistCor {m['dist_corr']:.2f}   T/C F1 {m['t_c_f1']:.2f}",
                      fontsize=7.5, color="#444")
    for ax in axes[len(runs):]:
        ax.set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="lower center", ncol=min(3, len(l)), frameon=False,
                   fontsize=7.5, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("DRD2: both ends of the dial, against the baselines",
                 fontsize=10.5)
    save(fig, "f3_maps_drd2")


# ── Manifold maps at the Table 1 settings ───────────────────────────────────

# Every panel is fitted by b22_fit_manifold_maps.py at exactly the setting
# Table 1 reports for that method (make_paper_tables.manifold_selection: seed-
# mean geodesic DistCor, NFMap restricted to contained cells). Geodesic, not
# ambient: ambient DistCor peaks where the swiss roll is still rolled, and
# these panels exist to show unrolling. Earlier versions hard-coded NFMap's
# sigma_perp from an MPS-fitted B2F and took baselines from B19 and GTM from
# A1, so a panel could show a different setting than the table row.


def _fit_manifold(ds, seed=42):
    """Load the pre-fitted manifold map from b22_fit_manifold_maps.py.

    Fitting NFMap inline inside this module used to segfault (exit 139):
    matplotlib + torch + sklearn + nfmap loaded together in one process crash
    during the actual fit, even though the identical fit succeeds in a script
    that never imports matplotlib. b22 fits and saves these separately; this
    just loads the result. Run `python b22_fit_manifold_maps.py` first if the
    .npz is missing or the B23 sweep has changed.
    """
    npz_path = os.path.join(RESULTS_ROOT, "B22", f"{ds}.npz")
    met_path = os.path.join(RESULTS_ROOT, "B22", f"{ds}_metrics.json")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"{npz_path} missing - run `python b22_fit_manifold_maps.py` first")
    data = np.load(npz_path)
    with open(met_path) as f:
        met = json.load(f)
    return {"Z": data["Z"], "y": data["y"], "metrics": met}


def _fit_baseline(ds, method, params, seed=42):
    """Load a pre-fitted t-SNE/UMAP baseline from b22_fit_manifold_maps.py.

    Previously fitted inline here, which crashes (exit 139) in this module's
    full process - torch + sklearn + UMAP loaded together. See _fit_manifold's
    docstring; same fix, same reason.
    """
    name = f"{ds}_{method}"
    npz_path = os.path.join(RESULTS_ROOT, "B22", f"{name}.npz")
    met_path = os.path.join(RESULTS_ROOT, "B22", f"{name}_metrics.json")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"{npz_path} missing - run `python b22_fit_manifold_maps.py` first")
    data = np.load(npz_path)
    with open(met_path) as f:
        met = json.load(f)
    return {"Z": data["Z"], "y": data["y"], "metrics": met}


def _setting_tag(method, p):
    """Panel-title suffix naming a baseline's Table 1 setting."""
    if method == "tsne":
        return rf" ($p$={p['perplexity']:g})"
    if method == "umap":
        return rf" ($k$={p['n_neighbors']:g}, $d$={p['min_dist']:g})"
    return ""


def _manifold_comparison(ds, name, suptitle):
    """Every method at the setting Table 1 reports for it (see b22)."""
    nf = _fit_manifold(ds)
    st = nf["metrics"]["setting"]
    runs = [(rf"NFMap ($\sigma_\perp$={st['sigma_perp']:g})", nf, None)]
    for m, lbl in (("gtm", "GTM"), ("tsne", "t-SNE"), ("umap", "UMAP")):
        try:
            r = _fit_baseline(ds, m, None)
        except FileNotFoundError:
            continue
        runs.append((lbl + _setting_tag(m, r["metrics"].get("setting", {})), r, None))

    # Geodesic DistCor is computed here for every panel from its own
    # embedding. Distance matrix is precomputed by b22_fit_manifold_maps.py: computing it
    # inline here (sklearn's kNN graph, in a process with torch already
    # loaded) is part of the same crash _fit_manifold's docstring describes.
    from scipy.spatial.distance import pdist as _pdist
    from scipy.stats import pearsonr as _pear
    _geo_path = os.path.join(RESULTS_ROOT, "B22", f"{ds}_geodesic.npy")
    _dgeo = np.load(_geo_path) if os.path.exists(_geo_path) else None

    nrow = int(np.ceil(len(runs) / 2))
    fig, axes = plt.subplots(nrow, 2, figsize=(6.6, 3.3 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (label, r, _unused) in zip(axes, runs):
        _map_panel(ax, r, label, ds, True)
        m = r["metrics"]
        Z = np.asarray(r["Z"])
        gtxt = ""
        if _dgeo is not None and not np.any(np.isnan(Z)):
            gtxt = f"   geodesic {_pear(_pdist(Z), _dgeo)[0]:.2f}"
        ax.set_xlabel(f"DistCor {m['dist_corr']:.2f}   T/C F1 {m['t_c_f1']:.2f}{gtxt}",
                      fontsize=7.5, color="#444")
    for ax in axes[len(runs):]:
        ax.set_visible(False)
    fig.suptitle(suptitle, fontsize=10.5)
    save(fig, name)


def fig_maps_supplementary():
    print("S2 (manifold map grids, Table 1 settings)...")
    for ds in MANIFOLD:
        _manifold_comparison(ds, f"s2_maps_{ds}",
                             f"{PRETTY.get(ds, ds)}: all methods")


# ── F9: components that matter only at full dataset scale ────────────────────

def fig_scale_dependence():
    """Hierarchical kernel and Student-t prior at full dataset scale.

    Built from B20, which is reproducible from this repository.

    Left panel is grouped bars per dataset rather than silhouette against n.
    The gains happen to be monotone in n here (+0.18, +0.21, +0.27), but with
    three datasets and single fits that is consistent with a scale relationship
    rather than evidence of one, and a trend line would assert the stronger
    reading.
    """
    print("F9 (scale-dependent components)...")
    rows = load_json("B20", "scale_study.json")
    if not rows:
        print("  (no B20 results - run b20_scale_study.py)")
        return
    get = lambda ds, k, fam, dof=None: next(
        (r for r in rows if r["dataset"] == ds and r["kernel"] == k
         and r["prior_family"] == fam and r.get("prior_dof") == dof), None)

    order = [("DRD2\n$n$=869", "drd2"), ("hERG\n$n$=9,400", "herg"),
             ("EGFR\n$n$=10,573", "egfr")]
    pairs = [(lbl, get(ds, "euclidean", "gaussian"), get(ds, "hybrid", "gaussian"))
             for lbl, ds in order]
    pairs = [(l, a, b) for l, a, b in pairs if a and b]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))

    ax = axes[0]
    labels = [p[0] for p in pairs]
    base = [p[1]["scaffold_silhouette"] for p in pairs]
    hyb = [p[2]["scaffold_silhouette"] for p in pairs]
    x = np.arange(len(pairs)); w = 0.36
    ax.bar(x - w/2, base, w, label="Euclidean graph", color="#9aaaa7",
           edgecolor="white", linewidth=0.5)
    ax.bar(x + w/2, hyb, w, label="hierarchical kernel", color="#1b6b61",
           edgecolor="white", linewidth=0.5)
    for xi, b, h in zip(x, base, hyb):
        ax.annotate("", xy=(xi + w/2, h), xytext=(xi - w/2, b),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=0.8))
        ax.text(xi, max(b, h, 0.0) + 0.03, f"{h-b:+.2f}", ha="center",
                fontsize=7.5, color="#444")
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(min(base + hyb) - 0.03, max(base + hyb + [0.0]) + 0.075)
    ax.set_ylabel("scaffold silhouette")
    ax.set_title("Hierarchical chemical kernel", fontsize=9.5, pad=14)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    g = get("egfr", "hybrid", "gaussian")
    dof_rows = sorted((r for r in rows if r["dataset"] == "egfr"
                       and r["prior_family"] == "student_t"),
                      key=lambda r: r["prior_dof"])
    dofs = [r["prior_dof"] for r in dof_rows]
    sils = [r["scaffold_silhouette"] for r in dof_rows]
    if g:
        ax.axhline(g["scaffold_silhouette"], ls="--", color="#9aaaa7", lw=1.2)
        ax.text(2, g["scaffold_silhouette"] + 0.004, "Gaussian prior",
                ha="left", va="bottom", fontsize=7.5, color="#777")
    ax.plot(dofs, sils, "-o", color="#1b6b61", markersize=5, lw=1.6)
    best = int(np.argmax(sils))
    # Label sits above-left of the optimum: the curve is flat enough that
    # anything placed to the right lands on the line.
    ax.annotate(rf"$\nu$={dofs[best]:g}", xy=(dofs[best], sils[best]),
                xytext=(dofs[best] + 0.6, sils[best] + 0.004), fontsize=8,
                color="#1b6b61", va="bottom", ha="left")
    ax.set_xlabel(r"Student-t degrees of freedom $\nu$")
    ax.set_ylabel("scaffold silhouette")
    ax.set_title("Prior tail weight (EGFR)", fontsize=9.5, pad=14)
    lo = min(sils + ([g["scaffold_silhouette"]] if g else []))
    ax.set_ylim(lo - 0.015, max(sils) + 0.022)

    fig.suptitle("Components that take effect only at full dataset scale",
                 fontsize=10.5)
    save(fig, "f9_scale_dependence")


# ── F10: graph_weight and repulsion_weight, both regimes ─────────────────────

def fig_loss_weights():
    """graph_weight and repulsion_weight swept on chemistry vs manifolds.

    Uses B3B (graph_weight) and B4C (repulsion_weight), filtered to the paper's
    own datasets. Coverage is partial on the chemistry side: B3B/B4C only
    include 3 of the paper's 5 chemistry datasets (bace, bbbp, drd2 - esol and
    egfr were not part of that breadth sweep), though all 3 manifolds are
    present. Stated in the caption rather than implied as full coverage.

    repulsion_weight shows the same regime reversal as sigma_perp (fig 2):
    DistCor falls with repulsion on chemistry and rises on manifolds.
    graph_weight does not reverse - both regimes peak near w=1 and decline
    thereafter in DistCor while T/C F1 rises monotonically in both - but the
    shipped default (w=5) sits past the empirical DistCor optimum in both
    regimes, which is itself worth reporting.
    """
    print("F10 (graph_weight and repulsion_weight, both regimes)...")
    PAPER_CHEM_SUBSET = {"bace", "bbbp", "drd2"}
    PAPER_MAN = set(MANIFOLD)

    def load_breadth(exp_id, wkey):
        d = os.path.join(RESULTS_ROOT, exp_id)
        if not os.path.isdir(d):
            return None
        acc = defaultdict(lambda: defaultdict(list))
        for run in os.listdir(d):
            cp, mp = os.path.join(d, run, "config.json"), os.path.join(d, run, "metrics.json")
            if not (os.path.exists(cp) and os.path.exists(mp)):
                continue
            c, m = json.load(open(cp)), json.load(open(mp))
            ds = c.get("dataset")
            if ds not in PAPER_CHEM_SUBSET | PAPER_MAN:
                continue
            group = "chemistry" if ds in PAPER_CHEM_SUBSET else "manifolds"
            acc[group][c[wkey]].append(m)
        return acc

    from collections import defaultdict
    panels = [("B3B", "graph_weight", "graph_weight"),
              ("B4C", "repulsion_weight", "repulsion_weight")]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    for row, (exp_id, wkey, wlabel) in enumerate(panels):
        acc = load_breadth(exp_id, wkey)
        if not acc:
            continue
        for col, (metric, mlabel) in enumerate([("dist_corr", "DistCor"),
                                                 ("t_c_f1", "T/C F1")]):
            ax = axes[row, col]
            for group, color in [("chemistry", "#1f77b4"), ("manifolds", "#d95f02")]:
                ws = sorted(acc[group])
                mus = [np.mean([x[metric] for x in acc[group][w]]) for w in ws]
                sds = [np.std([x[metric] for x in acc[group][w]]) for w in ws]
                ax.errorbar(ws, mus, yerr=sds, fmt="-o", color=color, capsize=2,
                           markersize=4, elinewidth=1,
                           label=group if (row == 0 and col == 0) else None)
            ax.set_xlabel(wlabel)
            ax.set_ylabel(mlabel)
            if col == 0:
                ax.set_title(f"{wlabel}", fontsize=9.5, loc="left")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Loss-term weights act differently in the two regimes",
                 fontsize=10.5)
    save(fig, "f10_loss_weights")


def fig_prior_digits():
    """Single Gaussian vs learned mixture vs fixed grid on digits (B5).

    One map per prior (seed 42; maps cannot be averaged),
    prior means drawn as crosses, metrics as means over B5's 3 seeds.
    """
    print("F11 (prior: single Gaussian / learned mixture / fixed grid)...")
    d = os.path.join(RESULTS_ROOT, "B5")
    if not os.path.isdir(d):
        return
    runs = {}
    for run in sorted(os.listdir(d)):
        cp = os.path.join(d, run, "config.json")
        if not os.path.exists(cp):
            continue
        with open(cp) as f:
            c = json.load(f)
        with open(os.path.join(d, run, "metrics.json")) as f:
            m = json.load(f)
        npz = np.load(os.path.join(d, run, "embeddings.npz"), allow_pickle=True)
        runs.setdefault((c["dataset"], c["condition"]), []).append(
            {"seed": c["seed"], "cfg": c, "m": m, "Z": npz["Z"], "y": npz["y"],
             "means": npz["prior_means"] if "prior_means" in npz.files else None})
    dsets = [ds for ds in ("digits",) if any(k[0] == ds for k in runs)]
    conds = [("single_gaussian", "Single Gaussian"),
             ("learnable_gmm", "Learned mixture"),
             ("fixed_grid", r"Fixed grid (NFMap)")]
    stat = lambda g, k: (np.mean([r["m"][k] for r in g]), np.std([r["m"][k] for r in g]))

    fig, axes = plt.subplots(len(dsets), len(conds), figsize=(10.5, 3.9 * len(dsets)),
                             squeeze=False)
    for i, ds in enumerate(dsets):
        names = CLASS_NAMES.get(ds, {})
        colors = plt.get_cmap("tab10").colors
        for j, (k, title) in enumerate(conds):
            ax = axes[i, j]
            g = runs.get((ds, k))
            if not g:
                ax.set_visible(False)
                continue
            r = next((x for x in g if x["seed"] == 42), g[0])
            Z, y = r["Z"], r["y"]
            for cls in np.unique(y):
                sel = y == cls
                ax.scatter(Z[sel, 0], Z[sel, 1], s=7, alpha=.8, linewidths=0,
                           color=colors[int(cls) % len(colors)],
                           label=names.get(int(cls), str(int(cls))))
            if r["means"] is not None:
                ax.scatter(r["means"][:, 0], r["means"][:, 1], marker="X", s=34,
                           c="black", edgecolors="white", linewidths=.6, zorder=10)
            ax.set_title(title if i == 0 else "", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            ax.set_aspect("equal", adjustable="datalim")
            (dc, _), (tc, _), (sil, _) = stat(g, "dist_corr"), stat(g, "t_c_f1"), stat(g, "sil")
            ax.set_xlabel(f"DistCor {dc:.2f}   T/C F1 {tc:.2f}   Sil {sil:.2f}",
                          fontsize=8, color="#444")
        h, l = axes[i, 0].get_legend_handles_labels()
        axes[i, -1].legend(h, l, loc="center left", bbox_to_anchor=(1.01, .5),
                           frameon=False, fontsize=7.5, markerscale=2,
                           ncol=2 if len(l) > 5 else 1, handletextpad=.1)
    fig.tight_layout()
    save(fig, "f11_prior_digits")


# One entry per figure in the paper.
DISPATCH = {"f2": fig_regimes, "f3": fig_maps_chemistry, "f4": fig_grid_chem,
            "f7": fig_ood, "f9": fig_scale_dependence, "f10": fig_loss_weights,
            "f11": fig_prior_digits, "s2": fig_maps_supplementary}


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

