"""Render Paper 1 figures/tables from saved results only — never retrains.

Reads results/<experiment_id>/*/{config.json,metrics.json,embeddings.npz}
(written by run_ablation.py / b5_gmm_grid.py) and produces the figures listed
in nfmap_paper_experiments.md's Summary Figure Plan. Re-run any generation
script and this will pick up the new numbers on the next call — no training
code lives in this file, matching the doc's "Save All Map Data for Post-Hoc
Plotting" policy.

Usage:
    python make_figures.py --which all
    python make_figures.py --which b2,b3
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

sys.path.insert(0, ".")
from common.io import iter_runs, RESULTS_ROOT
import fig1_architecture_diagram
import fig1b_manifold_geometry
import fig1c_manifold_thickness

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")


# ── loading ─────────────────────────────────────────────────────────────────

def _load_run_full(experiment_id, name):
    d = os.path.join(RESULTS_ROOT, experiment_id, name)
    with open(os.path.join(d, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(d, "metrics.json")) as f:
        metrics = json.load(f)
    npz = np.load(os.path.join(d, "embeddings.npz"), allow_pickle=True)
    Z = npz["Z"] if "Z" in npz.files else None
    y = npz["y"] if "y" in npz.files else None
    prior_means = npz["prior_means"] if "prior_means" in npz.files else None
    return config, metrics, Z, y, prior_means


def load_all(experiment_id):
    runs = []
    for name, _ in iter_runs(experiment_id):
        config, metrics, Z, y, prior_means = _load_run_full(experiment_id, name)
        runs.append({"name": name, "config": config, "metrics": metrics, "Z": Z, "y": y, "prior_means": prior_means})
    if not runs:
        print(f"  (no saved runs found for {experiment_id!r} — run its Makefile target first)")
    return runs


# ── shared plotting helpers ──────────────────────────────────────────────────

def _is_valid_embedding(Z):
    if Z is None or getattr(Z, "dtype", None) == object:
        return False
    return not np.any(np.isnan(Z))


# Colorblind-safe categorical palette (dataviz skill's validated 8-hue set,
# references/palette.md, light-mode hex - these are static PNG exports, no
# dark-mode variant to carry). Slots 1-3 (blue/orange/aqua) are the skill's
# own all-pairs-validated set for scatter plots specifically; slots 4-8 pass
# the weaker adjacent-pair check only. Slots 9-10 are an unvalidated
# extension for digits, the one dataset here with >8 classes - the skill's
# own guidance for >3-category scatters is "fold to Other or facet," which
# isn't practical for a "show all 10 digit classes at once" DR figure, so
# this is a considered, not random, choice (distinct hue/lightness from the
# other 8), compensated by a mandatory legend rather than silently claimed
# as fully validated.
CATEGORICAL_PALETTE = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
    "#8c5a2b",  # 9 brown (unvalidated extension, digits only)
    "#4a4a4a",  # 10 dark grey (unvalidated extension, digits only)
]

# Semantic class names for the datasets whose label encoding this harness
# controls directly - falls back to plain "class N" for everything else
# (digits' raw integers are already self-explanatory; wine/iris/moons/
# circles don't have a meaningfully-named class beyond that).
CLASS_NAMES = {
    "bbbp": ["BBB non-permeant", "BBB permeant"],
    "bace": ["inactive", "active"],
    "esol": ["insoluble", "moderate", "soluble"],
    "dude": ["decoy", "active"],
    "drd2": ["inactive (pIC50<7)", "active (pIC50≥7)"],
    "egfr": ["inactive (pIC50<7)", "active (pIC50≥7)"],
}


def plot_map_panel(ax, run, title, is_continuous=False, show_grid=False, dataset=None, show_auc=False):
    """Plot one NFMap/baseline 2D embedding.

    show_grid: only True for figures specifically about the GMM prior grid
    itself (B5, B9) - everywhere else the grid's "X" markers just occlude
    the actual data, so they're off by default.
    """
    Z, y, prior_means = run["Z"], run["y"], run["prior_means"]
    if not _is_valid_embedding(Z):
        ax.text(0.5, 0.5, "FAILED", ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="#cc0000")
    else:
        if is_continuous:
            sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="viridis", s=15, alpha=0.85, linewidths=0)
            cbar = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=6)
        else:
            classes = np.unique(y)
            cmap = ListedColormap(CATEGORICAL_PALETTE[: len(classes)])
            ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap=cmap, vmin=classes.min(), vmax=classes.max(),
                       s=15, alpha=0.85, linewidths=0)
            names = CLASS_NAMES.get(dataset)
            labels = names if names and len(names) == len(classes) else [str(int(c)) for c in classes]
            handles = [Line2D([0], [0], marker="o", linestyle="", markersize=6,
                              color=CATEGORICAL_PALETTE[i]) for i in range(len(classes))]
            ax.legend(handles, labels, fontsize=6, loc="best", framealpha=0.85,
                      ncol=2 if len(classes) > 5 else 1)

        if show_grid and prior_means is not None:
            ax.scatter(prior_means[:, 0], prior_means[:, 1], marker="X", s=80,
                       c="black", edgecolors="white", linewidths=1.2, zorder=10)
    m = run["metrics"]
    subtitle = f"T/C_F1={m['t_c_f1']:.2f} DistCor={m['dist_corr']:.2f} Sil={m.get('sil', float('nan')):.2f}"
    if show_auc:
        auc = m.get("auc", float("nan"))
        subtitle += f" AUC={auc:.2f}" if not np.isnan(auc) else " AUC=n/a"
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(subtitle, fontsize=8, color="#555555")
    ax.set_xticks([])
    ax.set_yticks([])


def save_fig(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")



# ── seed aggregation ─────────────────────────────────────────────────────────

CANONICAL_SEED = 42   # the seed whose map is drawn when panels cannot be averaged


def group_by_value(runs):
    """{value: [runs]} - sweeps now carry one run per (value, seed)."""
    out = {}
    for r in runs:
        out.setdefault(r["config"]["value"], []).append(r)
    return dict(sorted(out.items()))


def agg_metric(group, metric):
    """(values, means, sds) across seeds, for plotting with error bars."""
    vals, mus, sds = [], [], []
    for v, rs in group.items():
        xs = [r["metrics"][metric] for r in rs
              if r["metrics"].get(metric) is not None and not np.isnan(r["metrics"][metric])]
        if not xs:
            continue
        vals.append(v); mus.append(float(np.mean(xs))); sds.append(float(np.std(xs)))
    return vals, mus, sds


def one_per_value(runs, seed=CANONICAL_SEED):
    """One run per value for panel figures: a map cannot be averaged over seeds,
    so we draw a single seed's map and report seed-averaged metrics beside it."""
    out = []
    for v, rs in group_by_value(runs).items():
        pick = next((r for r in rs if r["config"].get("seed") == seed), rs[0])
        pick = dict(pick)
        pick["n_seeds"] = len(rs)
        pick["mean_metrics"] = {
            k: float(np.mean([r["metrics"][k] for r in rs
                              if r["metrics"].get(k) is not None
                              and not np.isnan(r["metrics"][k])] or [np.nan]))
            for k in ("t_c_f1", "dist_corr", "sil", "dens_corr")}
        out.append(pick)
    return out


# ── B2: sigma_perp sweep (Fig 5) ─────────────────────────────────────────────

# Per-dataset "sweet spot" for the qualitative low-vs-best comparison panel,
# chosen from each dataset's own sweep (not a single shared value - digits
# and dude both peak clearly around 1.0-10.0, bbbp's own metrics disagree
# with each other on where the peak is, see the write-up in
# nfmap_paper_experiments.md's B2 section). "Crushed" (0.0001) is shared
# across all three - it's the near-Brehmer-razor-thin regime none of them
# ever tested before this sweep.
B2_CRUSHED_VALUE = 0.0001
B2_SWEET_SPOT = {
    "digits": 1.0, "dude": 10.0, "bbbp": 1.0,
    # 3D-manifold datasets (z_perp collapses to 1 real dimension): DistCor
    # peaks at a much smaller value than the high-dimensional group above -
    # swiss_roll at 0.05, s_curve at 0.01, severed_sphere diffusely across
    # 0.5-20 (0.5 chosen as a representative interior point, not the sweep's
    # literal argmax at 20.0). See nfmap_paper_experiments.md's B2 section.
    "swiss_roll": 0.05, "s_curve": 0.01, "severed_sphere": 0.5,
}

def fig_b2():
    print("B2 (sigma_perp sweep, multi-dataset)...")
    all_runs = load_all("B2")
    if not all_runs:
        return
    datasets_present = sorted({r["config"].get("dataset", "") for r in all_runs} - {""})

    for ds in datasets_present:
        runs = [r for r in all_runs if r["config"].get("dataset") == ds]
        grouped = group_by_value(runs)
        n_seeds = max(len(v) for v in grouped.values())

        fig, ax = plt.subplots(figsize=(6, 4))
        metric_specs = [
            ("dist_corr", "-o", CATEGORICAL_PALETTE[0]),
            ("dens_corr", "-s", CATEGORICAL_PALETTE[1]),
            ("t_c_f1", "-^", CATEGORICAL_PALETTE[2]),
            ("sil", "-d", CATEGORICAL_PALETTE[3]),
        ]
        for metric, style, color in metric_specs:
            vals, mus, sds = agg_metric(grouped, metric)
            if not vals:
                continue
            ax.errorbar(vals, mus, yerr=sds, fmt=style, color=color, label=metric,
                        capsize=2, lw=1.4, markersize=5, elinewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("sigma_perp")
        ax.set_ylabel("score")
        ax.set_title(f"B2: sigma_perp sweep ({ds}) — mean ± sd over {n_seeds} seeds")
        ax.legend()
        save_fig(fig, f"fig5_sigma_perp_sweep_{ds}.png")

        crushed = min(runs, key=lambda r: abs(r["config"]["value"] - B2_CRUSHED_VALUE))
        sweet = min(runs, key=lambda r: abs(r["config"]["value"] - B2_SWEET_SPOT.get(ds, 5.0)))
        is_cont = ds in CONTINUOUS_DATASETS
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        plot_map_panel(axes[0], crushed, f"sigma_perp={crushed['config']['value']} (crushed)", dataset=ds, is_continuous=is_cont)
        plot_map_panel(axes[1], sweet, f"sigma_perp={sweet['config']['value']} (sweet spot)", dataset=ds, is_continuous=is_cont)
        fig.suptitle(f"B2: crushed vs. sweet-spot sigma_perp ({ds})")
        save_fig(fig, f"fig5b_sigma_perp_qualitative_{ds}.png")


# ── B3: graph_weight sweep (Fig 4) ───────────────────────────────────────────

def fig_b3():
    print("B3 (graph_weight sweep)...")
    runs = load_all("B3")
    if not runs:
        return
    runs = one_per_value(runs)

    fig, axes = plt.subplots(1, len(runs), figsize=(4.5 * len(runs), 5))
    if len(runs) == 1:
        axes = [axes]
    for ax, r in zip(axes, runs):
        plot_map_panel(ax, r, f"graph_weight={r['config']['value']}", dataset="digits")
        mm = r["mean_metrics"]
        ax.set_xlabel(f"T/C_F1={mm['t_c_f1']:.2f} DistCor={mm['dist_corr']:.2f} "
                      f"Sil={mm['sil']:.2f}  (mean of {r['n_seeds']} seeds)",
                      fontsize=8, color="#555")
    fig.suptitle("B3: graph_weight sweep (digits) — map quality progression")
    save_fig(fig, "fig4_graph_weight_sweep.png")


# ── B4: negative_sample_rate sweep ───────────────────────────────────────────

def fig_b4():
    print("B4 (repulsion_weight sweep)...")
    runs = load_all("B4")
    if not runs:
        return
    runs = one_per_value(runs)

    fig, axes = plt.subplots(1, len(runs), figsize=(4.5 * len(runs), 5))
    if len(runs) == 1:
        axes = [axes]
    for ax, r in zip(axes, runs):
        plot_map_panel(ax, r, f"repulsion_weight={r['config']['value']}", dataset="digits")
        mm = r["mean_metrics"]
        ax.set_xlabel(f"T/C_F1={mm['t_c_f1']:.2f} DistCor={mm['dist_corr']:.2f} "
                      f"Sil={mm['sil']:.2f}  (mean of {r['n_seeds']} seeds)",
                      fontsize=8, color="#555")
    fig.suptitle("B4: repulsion_weight sweep (digits) — repulsion strength")
    save_fig(fig, "fig_b4_repulsion_weight.png")


# ── B4B: repulsion on vs. off, all datasets (Fig 7b) ─────────────────────────

OFF_COLOR = CATEGORICAL_PALETTE[0]   # blue  — repulsion off (rate=0)
ON_COLOR = CATEGORICAL_PALETTE[1]    # orange — repulsion on (rate=5, default)


def _b4b_table(runs, metric):
    """{dataset: {rate: {seed: value}}} for one metric, NaNs dropped."""
    table = {}
    for r in runs:
        c = r["config"]
        v = r["metrics"].get(metric)
        if v is None or np.isnan(v):
            continue
        table.setdefault(c["dataset"], {}).setdefault(c["negative_sample_rate"], {})[c["seed"]] = v
    return table


def _b4b_paired(table):
    """Per dataset: (mean_off, sd_off, mean_on, sd_on, mean_delta, sd_delta).

    Deltas are computed *per seed* and then averaged, not as a difference of
    means: the two conditions share a seed, so pairing removes the shared
    seed-to-seed variation and is what makes a small effect readable at all.
    """
    out = {}
    for ds, by_rate in table.items():
        off, on = by_rate.get(0, {}), by_rate.get(5, {})
        shared = sorted(set(off) & set(on))
        if not shared:
            continue
        a = np.array([off[s] for s in shared])
        b = np.array([on[s] for s in shared])
        d = a - b
        out[ds] = (a.mean(), a.std(), b.mean(), b.std(), d.mean(), d.std())
    return out


def fig_b4b():
    print("B4B (repulsion on vs. off, all datasets)...")
    runs = load_all("B4B")
    if not runs:
        return

    # Three metrics, deliberately: they disagree, and the disagreement is the
    # finding. Silhouette rewards compact clusters regardless of whether the map
    # stayed faithful, so it favours repulsion-off (which collapses each class
    # into a tight clump); T/C F1 and DistCor measure neighbourhood/distance
    # fidelity and favour repulsion-on. Same trap as the circles-vs-UMAP result
    # in Section 5.
    for metric, label, fname in [
        ("sil", "silhouette", "fig7b_repulsion_silhouette.png"),
        ("t_c_f1", "T/C F1", "fig7c_repulsion_tcf1.png"),
        ("dist_corr", "DistCor", "fig7e_repulsion_distcor.png"),
    ]:
        paired = _b4b_paired(_b4b_table(runs, metric))
        if not paired:
            continue
        # Sort by effect size so the figure reads as a ranking, strongest first.
        order = sorted(paired, key=lambda d: paired[d][4], reverse=True)
        y = np.arange(len(order))

        fig, (ax_l, ax_r) = plt.subplots(
            1, 2, figsize=(11, 0.46 * len(order) + 2.6),
            gridspec_kw={"width_ratios": [1.15, 1]})

        # Left: absolute levels, so a delta can be judged against the scale it
        # sits on (+0.01 means something different on bbbp than on digits).
        h = 0.36
        ax_l.barh(y + h / 2, [paired[d][0] for d in order], height=h,
                  xerr=[paired[d][1] for d in order], color=OFF_COLOR,
                  label="repulsion off (rate=0)", error_kw={"lw": 1, "ecolor": "#555"})
        ax_l.barh(y - h / 2, [paired[d][2] for d in order], height=h,
                  xerr=[paired[d][3] for d in order], color=ON_COLOR,
                  label="repulsion on (rate=5, default)", error_kw={"lw": 1, "ecolor": "#555"})
        ax_l.set_yticks(y)
        ax_l.set_yticklabels(order)
        ax_l.invert_yaxis()
        ax_l.set_xlabel(label)
        ax_l.set_title(f"{label}: repulsion off vs. on", fontsize=11)
        ax_l.legend(fontsize=8, loc="lower right")
        ax_l.grid(axis="x", alpha=0.25)
        ax_l.set_axisbelow(True)

        # Right: the paired difference itself — the actual claim under test.
        deltas = np.array([paired[d][4] for d in order])
        errs = np.array([paired[d][5] for d in order])
        colors = [OFF_COLOR if v > 0 else ON_COLOR for v in deltas]
        ax_r.errorbar(deltas, y, xerr=errs, fmt="none", ecolor="#555", lw=1.2, capsize=3)
        ax_r.scatter(deltas, y, c=colors, s=55, zorder=5, edgecolors="white", linewidths=0.8)
        ax_r.axvline(0, color="#333", lw=1.1, ls="--")
        ax_r.set_yticks(y)
        ax_r.set_yticklabels([])
        ax_r.invert_yaxis()
        ax_r.set_xlabel(f"Δ {label}  (off − on, paired by seed)")
        ax_r.set_title("right of 0 = repulsion hurts", fontsize=11)
        ax_r.grid(axis="x", alpha=0.25)
        ax_r.set_axisbelow(True)

        n_seeds = max(len(v) for by_rate in _b4b_table(runs, metric).values()
                      for v in by_rate.values())
        fig.suptitle(
            f"B4B: does repulsion help? {label}, repulsion off vs. NFMap's default "
            f"(mean ± sd over {n_seeds} seeds)", fontsize=12.5)
        fig.tight_layout()
        save_fig(fig, fname)

    _fig_b4b_maps(runs)


B4B_MAP_SEED = 42  # canonical seed, matching every other figure in this paper


def _fig_b4b_maps(runs):
    """The qualitative companion: what repulsion off vs. on actually looks like.

    The claim under test ("repulsion sharpens cluster boundaries") is a visual
    one, so a silhouette delta alone doesn't settle it - a reader needs to see
    whether the maps differ the way the metric says they do.
    """
    by_ds = {}
    for r in runs:
        c = r["config"]
        if c.get("seed") != B4B_MAP_SEED:
            continue
        by_ds.setdefault(c["dataset"], {})[c["negative_sample_rate"]] = r

    # Order by silhouette effect size so the strongest case leads; datasets with
    # no silhouette (continuous targets) go last, in their own group.
    paired = _b4b_paired(_b4b_table(runs, "sil"))
    complete = [d for d, pair in by_ds.items() if 0 in pair and 5 in pair]
    labelled = sorted([d for d in complete if d in paired],
                      key=lambda d: paired[d][4], reverse=True)
    continuous = sorted(d for d in complete if d not in paired)
    order = labelled + continuous
    if not order:
        return

    for ds in order:
        is_cont = ds in CONTINUOUS_DATASETS
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        plot_map_panel(axes[0], by_ds[ds][0], "repulsion OFF (rate=0)",
                       dataset=ds, is_continuous=is_cont)
        plot_map_panel(axes[1], by_ds[ds][5], "repulsion ON (rate=5, default)",
                       dataset=ds, is_continuous=is_cont)
        delta = f"  (Δsil {paired[ds][4]:+.3f})" if ds in paired else ""
        fig.suptitle(f"B4B: {ds} — repulsion off vs. on (seed {B4B_MAP_SEED}){delta}",
                     fontsize=12)
        fig.tight_layout()
        save_fig(fig, f"fig7d_repulsion_maps_{ds}.png")

    # One combined overview so the cross-dataset pattern is readable in a single
    # figure, two datasets per row (off|on, off|on).
    n_rows = (len(order) + 1) // 2
    fig, axes = plt.subplots(n_rows, 4, figsize=(19, 4.3 * n_rows))
    axes = np.atleast_2d(axes)
    for i, ds in enumerate(order):
        row, col = divmod(i, 2)
        is_cont = ds in CONTINUOUS_DATASETS
        d = f" (Δsil {paired[ds][4]:+.3f})" if ds in paired else ""
        plot_map_panel(axes[row][col * 2], by_ds[ds][0], f"{ds} — OFF{d}",
                       dataset=ds, is_continuous=is_cont)
        plot_map_panel(axes[row][col * 2 + 1], by_ds[ds][5], f"{ds} — ON (default)",
                       dataset=ds, is_continuous=is_cont)
    for j in range(len(order), n_rows * 2):      # blank any unused pair slots
        row, col = divmod(j, 2)
        axes[row][col * 2].axis("off")
        axes[row][col * 2 + 1].axis("off")
    fig.suptitle("B4B: repulsion off vs. on across all datasets "
                 f"(seed {B4B_MAP_SEED}, ordered by silhouette effect size)",
                 fontsize=14)
    fig.tight_layout()
    save_fig(fig, "fig7d_repulsion_maps_all.png")


# ── B2C: sigma_perp as a local/global dial ───────────────────────────────────

B2C_DATASETS = ["digits", "bbbp", "dude", "drd2", "esol", "swiss_roll"]


def fig_b2c():
    """Same dataset under both ends of the sigma_perp dial, side by side.

    Reported as a pair rather than a tuned per-dataset value on purpose: B2B
    found the optimum is dataset-specific (rho* varies 17x at fixed d_perp, so
    the dimensionless scaling law does not hold), which makes any single tuned
    value selection on the data we then report.
    """
    print("B2C (sigma_perp local/global dial)...")
    res = os.path.join(RESULTS_ROOT, "B2C")
    if not os.path.isdir(res):
        print("  (no B2C results — run `make b2c` first)")
        return

    def load(ds, setting):
        runs = []
        for d in sorted(glob.glob(os.path.join(res, f"{ds}_{setting}_seed*"))):
            with open(os.path.join(d, "metrics.json")) as f:
                m = json.load(f)
            npz = np.load(os.path.join(d, "embeddings.npz"), allow_pickle=True)
            runs.append((m, npz))
        return runs

    datasets = [d for d in B2C_DATASETS if load(d, "global")]
    if not datasets:
        return
    fig, axes = plt.subplots(len(datasets), 2, figsize=(10.2, 4.6 * len(datasets) + 0.6))
    axes = np.atleast_2d(axes)

    for r, ds in enumerate(datasets):
        for c, (setting, sigma) in enumerate([("global", 0.01), ("local", 10.0)]):
            runs = load(ds, setting)
            if not runs:
                continue
            m, npz = runs[0]                      # first seed for the picture
            Z, y = npz["Z"], npz["y"]
            ax = axes[r][c]
            is_cont = ds in CONTINUOUS_DATASETS
            plot_map_panel(ax, {"Z": Z, "y": y, "metrics": m, "prior_means": None},
                           f"{ds} — sigma_perp={sigma} ({setting}-optimising)",
                           is_continuous=is_cont, dataset=ds)
            # metrics averaged over seeds, not the single plotted one
            dc = np.mean([mm["dist_corr"] for mm, _ in runs])
            tc = np.mean([mm["t_c_f1"] for mm, _ in runs])
            ax.set_xlabel(f"DistCor={dc:+.3f}   T/C F1={tc:.3f}   (mean of {len(runs)} seeds)",
                          fontsize=9, color="#333")

    fig.suptitle("B2C: sigma_perp trades global distance fidelity against local "
                 "neighbourhood fidelity\nLeft: squeezed z_perp forces geometry into "
                 "the map. Right: roomy z_perp (NFMap's default).", fontsize=12.5)
    # Reserve space for the two-line suptitle, or it lands on the first row's
    # panel titles.
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    save_fig(fig, "fig5c_sigma_perp_local_global.png")


# ── B5: GMM grid (Fig 7) ─────────────────────────────────────────────────────

def fig_b5():
    print("B5 (GMM grid: single/learnable/fixed)...")
    runs = load_all("B5")
    if not runs:
        return
    order = {"single_gaussian": 0, "learnable_gmm": 1, "fixed_grid": 2}
    # One map per condition (maps cannot be averaged), with the seed-averaged
    # metrics printed beneath so the panel is not read as a single-seed result.
    by_cond = {}
    for r in runs:
        by_cond.setdefault(r["config"].get("condition", r["name"]), []).append(r)
    conds = sorted(by_cond, key=lambda c: order.get(c, 99))

    fig, axes = plt.subplots(1, len(conds), figsize=(5 * len(conds), 5))
    if len(conds) == 1:
        axes = [axes]
    for ax, cond in zip(axes, conds):
        group = by_cond[cond]
        r = next((x for x in group if x["config"].get("seed") == CANONICAL_SEED), group[0])
        plot_map_panel(ax, r, cond, dataset="digits", show_grid=True)
        mean = lambda k: np.mean([x["metrics"][k] for x in group
                                  if x["metrics"].get(k) is not None
                                  and not np.isnan(x["metrics"][k])] or [np.nan])
        ax.set_xlabel(f"T/C_F1={mean('t_c_f1'):.3f}  Sil={mean('sil'):.3f}  "
                      f"Acc={mean('acc'):.3f}  (mean of {len(group)} seeds)",
                      fontsize=8, color="#555")
    fig.suptitle("B5: single Gaussian vs. learnable GMM vs. fixed grid (digits)")
    save_fig(fig, "fig7_gmm_grid.png")


# B7 (isometric regularization) was dropped from the paper — see the note in
# the Makefile and B7's "Status" writeup in nfmap_paper_experiments.md. The
# generation script (run_ablation.py --param canonical_weight) still works if
# anyone wants to reproduce it; this file just no longer renders it.


# ── B9: prior_weight vs. grid containment ────────────────────────────────────

def _plot_containment_lever(ax, runs, param_label):
    grouped = group_by_value(runs)

    vals, mus, sds = agg_metric(grouped, "frac_outside_grid_bounds")
    ax.errorbar(vals, [100 * m for m in mus], yerr=[100 * s for s in sds],
                fmt="-o", color="tab:red", capsize=2, elinewidth=1,
                label="% outside grid_bounds")
    ax.set_xlabel(param_label)
    ax.set_ylabel("% points outside grid_bounds", color="tab:red")
    ax.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax.twinx()
    for metric, style, color, lbl in [("sil", "-s", "tab:blue", "silhouette"),
                                      ("t_c_f1", "-^", "tab:green", "T/C_F1")]:
        vals, mus, sds = agg_metric(grouped, metric)
        if vals:
            ax2.errorbar(vals, mus, yerr=sds, fmt=style, color=color, capsize=2,
                         elinewidth=1, label=lbl)
    ax2.set_ylabel("quality metric")
    return ax, ax2


def fig_b9():
    print("B9 (grid containment: prior_weight vs. grid_bounds levers)...")
    runs = load_all("B9")
    gb_runs = load_all("B9_gridbounds")
    if not runs and not gb_runs:
        return

    # Two independent levers, side by side: does raising prior_weight (keeping
    # grid_bounds fixed) or widening grid_bounds (keeping prior_weight fixed)
    # fix containment, and at what cost to quality?
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    if runs:
        ax1, ax2 = _plot_containment_lever(axes[0], runs, "prior_weight (grid_bounds=3.0 fixed)")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="center right")
        ax1.set_title("Lever 1: prior_weight")
    if gb_runs:
        ax1, ax2 = _plot_containment_lever(axes[1], gb_runs, "grid_bounds (prior_weight=0.1 fixed)")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="center right")
        ax1.set_title("Lever 2: grid_bounds")
    fig.suptitle("B9: two independent levers for grid containment (digits)")
    save_fig(fig, "fig10_prior_weight_containment.png")

    # Qualitative, lever 1: default (0.1) vs. a value that fully contains the grid (2.0).
    lo = next((r for r in runs if r["config"]["value"] == 0.1), runs[0])
    hi = next((r for r in runs if r["config"]["value"] == 2.0), runs[-1])
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plot_map_panel(axes[0], lo, f"prior_weight={lo['config']['value']} (default)", dataset="digits", show_grid=True)
    plot_map_panel(axes[1], hi, f"prior_weight={hi['config']['value']}", dataset="digits", show_grid=True)
    fig.suptitle("B9: qualitative comparison — default vs. fully-contained grid (prior_weight lever)")
    save_fig(fig, "fig10b_prior_weight_qualitative.png")

    # Qualitative, lever 2: grid_bounds=3.0 (the paper's old, pre-B9 default,
    # 25.7% overflow) vs. grid_bounds=6.0 (the value B9's own finding got
    # adopted as the new shared default across every other Paper 1
    # experiment, 0.3% overflow) - this is the lever actually recommended
    # (widening the canvas costs nothing in quality and even helps slightly,
    # unlike prior_weight which is quality-neutral rather than quality-
    # positive), so it gets the same qualitative treatment as lever 1.
    if gb_runs:
        gb_lo = next((r for r in gb_runs if r["config"]["value"] == 3.0), gb_runs[0])
        gb_hi = next((r for r in gb_runs if r["config"]["value"] == 6.0), gb_runs[-1])
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        plot_map_panel(axes[0], gb_lo, f"grid_bounds={gb_lo['config']['value']} (old default)", dataset="digits", show_grid=True)
        plot_map_panel(axes[1], gb_hi, f"grid_bounds={gb_hi['config']['value']} (recommended)", dataset="digits", show_grid=True)
        fig.suptitle("B9: qualitative comparison — old vs. recommended grid_bounds (grid_bounds lever)")
        save_fig(fig, "fig10c_gridbounds_qualitative.png")


# ── B1: log-det dimension scaling (Fig 6) ────────────────────────────────────

def _load_b1_runs():
    runs = {}
    b1_dir = os.path.join(RESULTS_ROOT, "B1")
    if not os.path.isdir(b1_dir):
        return runs
    for name in sorted(os.listdir(b1_dir)):
        d = os.path.join(b1_dir, name)
        diag_path = os.path.join(d, "diagnostics.npz")
        if not os.path.exists(diag_path):
            continue
        with open(os.path.join(d, "config.json")) as f:
            config = json.load(f)
        with open(os.path.join(d, "metrics.json")) as f:
            metrics = json.load(f)
        runs[name] = {"config": config, "metrics": metrics, "npz": np.load(diag_path)}
    return runs


def fig_b1():
    print("B1 (log-det dimension scaling: vol_weight)...")
    runs = _load_b1_runs()
    if not runs:
        print("  (no saved runs found for B1 — run b1_logdet_scaling.py first)")
        return

    order = sorted(runs.items(), key=lambda kv: kv[1]["config"]["vol_weight"])

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, r in order:
        ax.plot(r["npz"]["loss_history"], label=f"vol_weight={r['config']['vol_weight']}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.set_title("B1: loss curves, vol_weight OFF vs. ON (DUD-E EGFR)")
    ax.legend()
    save_fig(fig, "fig6a_logdet_loss_curves.png")

    # Density-normalized, not raw counts: parallel has only n_components (2)
    # dimensions worth of samples, perpendicular has d_scaled-2 (~134) - at
    # raw counts the parallel bars are ~67x smaller and effectively invisible
    # next to perpendicular's, regardless of how different their shapes are.
    # Density puts both on a comparable scale (area under each = 1) so shape,
    # not sample count, is what's being compared. Colorblind-safe blue/orange
    # (Okabe-Ito) instead of green/orange, which is a poor pair under
    # deuteranopia/protanopia; dashed mean lines make the two groups'
    # location unambiguous even where a bar is thin.
    PARALLEL_COLOR = "#0072B2"
    PERP_COLOR = "#E69F00"
    fig, axes = plt.subplots(1, len(order), figsize=(6.5 * len(order), 4.8))
    if len(order) == 1:
        axes = [axes]
    for ax, (name, r) in zip(axes, order):
        par = r["npz"]["parallel_norms"]
        perp = r["npz"]["perp_norms"]
        bins = np.linspace(0, max(par.max(), perp.max()), 30)
        ax.hist(par, bins=bins, density=True, alpha=0.55, color=PARALLEL_COLOR,
                label=f"parallel, n={len(par)} (mean={par.mean():.2f})")
        ax.hist(perp, bins=bins, density=True, alpha=0.55, color=PERP_COLOR,
                label=f"perpendicular, n={len(perp)} (mean={perp.mean():.2f})")
        ax.axvline(par.mean(), color=PARALLEL_COLOR, linestyle="--", linewidth=1.5)
        ax.axvline(perp.mean(), color=PERP_COLOR, linestyle="--", linewidth=1.5)
        ax.set_title(f"vol_weight={r['config']['vol_weight']}", pad=12)
        ax.set_xlabel("Jacobian column norm")
        ax.set_ylabel("density (dashed = group mean)")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("B1: per-dimension Jacobian column norms — parallel vs. perpendicular (DUD-E EGFR)", y=1.04)
    plt.tight_layout()
    save_fig(fig, "fig6b_logdet_jacobian_norms.png")


# ── A1/A2: NFMap vs. GTM/t-SNE/UMAP (Fig 2, Fig 3, Table 1) ──────────────────

def _load_by_dataset(experiment_id):
    """A1/A2 save one run per (dataset, kind, seed), flat under
    results/<experiment_id>/. Group back into {dataset: {kind: [runs]}} - a
    list, not a single run, because every method now gets three seeds and
    Table 1 reports mean +/- sd rather than whichever fit happened to run."""
    by_dataset = {}
    for r in load_all(experiment_id):
        ds = r["config"].get("dataset")
        kind = r["config"].get("kind")
        by_dataset.setdefault(ds, {}).setdefault(kind, []).append(r)
    return by_dataset


def _pick_seed(group):
    """The run whose map gets drawn. Maps cannot be averaged across seeds, so
    panels show one canonical fit and report seed-averaged numbers beside it."""
    return next((r for r in group if r["config"].get("seed") == CANONICAL_SEED), group[0])


BASELINE_ORDER = ["gtm", "tsne", "umap"]
METHOD_LABELS = {"nfmap": "NFMap", "gtm": "GTM", "tsne": "t-SNE", "umap": "UMAP"}
# A2 fits NFMap at both ends of the sigma_perp dial (kinds "nfmap_local" and
# "nfmap_global"), so the panel width is not fixed at 4 any more. Variants sort
# global-then-local so the panels read left-to-right as the dial turns.
VARIANT_ORDER = {"global": 0, "local": 1}
CONTINUOUS_DATASETS = {"swiss_roll", "s_curve", "severed_sphere"}


def method_order(methods):
    """NFMap variants first (global -> local), then the baselines."""
    nf = sorted((k for k in methods if k and k.startswith("nfmap")),
                key=lambda k: VARIANT_ORDER.get(k.partition("_")[2], 99))
    return nf + [k for k in BASELINE_ORDER if k in methods]


def method_label(kind):
    if kind in METHOD_LABELS:
        return METHOD_LABELS[kind]
    if kind.startswith("nfmap_"):
        variant = kind.partition("_")[2]
        sigma = {"global": "0.003", "local": "10"}.get(variant)
        return f"NFMap ({variant}" + (f", $\\sigma_\\perp$={sigma})" if sigma else ")")
    return kind


def _comparison_panel(ds, methods, fig_prefix, title_prefix, show_auc=False):
    is_continuous = ds in CONTINUOUS_DATASETS
    order = method_order(methods)
    fig, axes = plt.subplots(1, len(order), figsize=(5 * len(order), 5))
    if len(order) == 1:
        axes = [axes]
    for ax, kind in zip(axes, order):
        group = methods.get(kind)
        if not group:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        r = _pick_seed(group)
        label = method_label(kind)
        if len(group) > 1:
            label += f" (seed {r['config'].get('seed')} of {len(group)})"
        plot_map_panel(ax, r, label, is_continuous=is_continuous, dataset=ds, show_auc=show_auc)
    fig.suptitle(f"{title_prefix}: {ds} — NFMap vs. GTM / t-SNE / UMAP")
    save_fig(fig, f"{fig_prefix}_{ds}_comparison.png")


def _write_benchmark_table(rows, out_name):
    if not rows:
        return
    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, out_name)
    keys = list(rows[0].keys())
    with open(out_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
    print(f"  -> {out_path}")


def fig_a1():
    print("A1 (standard benchmarks: NFMap vs. GTM/t-SNE/UMAP)...")
    by_dataset = _load_by_dataset("A1")
    if not by_dataset:
        return

    # Fig 2: one comparison figure per representative dataset, all 4 methods
    # side by side (not NFMap alone) — per the Summary Figure Plan's
    # "digits/moons/swiss_roll", plus s_curve (the doc's other manifold
    # dataset, previously omitted from this panel by mistake).
    for ds in [d for d in ["digits", "wine", "cancer", "iris", "moons", "circles", "swiss_roll", "s_curve", "severed_sphere"] if d in by_dataset]:
        _comparison_panel(ds, by_dataset[ds], "fig2", "A1")

    _write_combined_table1()


def fig_a2():
    print("A2 (chemistry benchmarks: NFMap vs. GTM/t-SNE/UMAP)...")
    by_dataset = _load_by_dataset("A2")
    if not by_dataset:
        return

    # Fig 3: one comparison panel per chemistry dataset - all 6 (bbbp, bace,
    # esol, dude, drd2, egfr), not a curated subset, since there's no single
    # obvious "representative 3" the way A1 had digits/moons/swiss_roll.
    for ds in sorted(by_dataset):
        _comparison_panel(ds, by_dataset[ds], "fig3", "A2", show_auc=True)


def _write_combined_table1():
    # Table 1 per the doc is ONE combined table ("all datasets x all methods,
    # incl. DRD2/EGFR baseline") - merge A1 + A2 rather than writing two
    # separate CSVs. Safe to call from either fig_a1() or fig_a2() alone
    # (whichever ran will simply be missing from the table until both have).
    METRICS = ["t_c_f1", "trust", "cont", "dist_corr", "dens_corr", "sil", "acc", "auc"]
    rows = []
    for exp_id in ["A1", "A2"]:
        by_dataset = _load_by_dataset(exp_id)
        for ds, methods in sorted(by_dataset.items()):
            for kind, group in sorted(methods.items()):
                row = {"dataset": ds, "method": kind, "n_seeds": len(group)}
                for k in METRICS:
                    xs = [r["metrics"][k] for r in group
                          if r["metrics"].get(k) is not None
                          and not np.isnan(r["metrics"][k])]
                    row[k] = round(float(np.mean(xs)), 4) if xs else ""
                    row[k + "_sd"] = round(float(np.std(xs)), 4) if len(xs) > 1 else ""
                rows.append(row)
    _write_benchmark_table(rows, "table1_standard_benchmarks.csv")


# ── Table 2: ablation summary ────────────────────────────────────────────────

def table_ablation_summary():
    print("Table 2 (ablation summary)...")
    # One row per (experiment, dataset, condition), aggregated over seeds: the
    # sweeps now run 3 seeds each, and a per-run table would report single-seed
    # noise as if it were an effect - which is how two retracted claims got in.
    METRICS = ["t_c_f1", "dist_corr", "dens_corr", "sil", "acc",
               "frac_outside_grid_bounds"]
    groups = {}
    for exp_id in ["B2", "B3", "B4", "B5", "B9", "B9_gridbounds"]:
        for r in load_all(exp_id):
            c, m = r["config"], r["metrics"]
            cond = c.get("condition", f"{c.get('param_swept', '')}={c.get('value', '')}")
            groups.setdefault((exp_id, c.get("dataset", ""), cond), []).append(m)

    rows = []
    for (exp_id, ds, cond), ms in groups.items():
        row = {"experiment": exp_id, "dataset": ds, "condition": cond,
               "n_seeds": len(ms)}
        for k in METRICS:
            xs = [m[k] for m in ms
                  if m.get(k) is not None and not np.isnan(m[k])]
            row[k] = round(float(np.mean(xs)), 4) if xs else ""
            row[k + "_sd"] = round(float(np.std(xs)), 4) if len(xs) > 1 else ""
        rows.append(row)
    if not rows:
        print("  (no runs found)")
        return
    rows.sort(key=lambda r: (r["experiment"], r["dataset"], str(r["condition"])))

    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, "table2_ablation_summary.csv")
    keys = list(rows[0].keys())
    with open(out_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
    print(f"  -> {out_path}")




# ── B2D / B2E: sigma_perp x grid_bounds, jointly ─────────────────────────────

GTM_REF = {"bace": 0.428, "bbbp": 0.628, "dude": 0.428, "drd2": 0.394,
           "egfr": 0.446, "esol": 0.689,
           "swiss_roll": 0.615, "s_curve": 0.627, "severed_sphere": 0.781}


def _load_joint(exp_id, fname):
    path = os.path.join(RESULTS_ROOT, exp_id, fname)
    if not os.path.exists(path):
        print(f"  (no {exp_id} results at {path})")
        return None
    with open(path) as f:
        return json.load(f)


def _joint_cells(rows, key="dist_corr"):
    """{(dataset, sigma, grid): (mean, sd, mean_outside)} over seeds."""
    acc = {}
    for r in rows:
        acc.setdefault((r["dataset"], r["sigma_perp"], r["grid_bounds"]), []).append(r)
    return {k: (float(np.mean([x[key] for x in v])),
                float(np.std([x[key] for x in v])),
                float(np.mean([x["frac_outside_grid_bounds"] for x in v])))
            for k, v in acc.items()}


def _joint_heatmap(rows, datasets, title, out_name, subtitle):
    """One panel per dataset: DistCor over the sigma x grid grid, with cells
    that fail the <=5% containment constraint hatched out. The constraint is
    what makes the surface readable - the raw argmax always sits in the
    uncontained corner, which is the degeneracy B2B identified (a map that
    "wins" DistCor by leaving its own canvas)."""
    cells = _joint_cells(rows)
    sigmas = sorted({k[1] for k in cells})
    grids = sorted({k[2] for k in cells})
    datasets = [d for d in datasets if any(k[0] == d for k in cells)]
    if not datasets:
        return
    n = len(datasets)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.3 * nrow))
    axes = np.atleast_1d(axes).ravel()

    vals = [v[0] for v in cells.values()]
    vmin, vmax = min(vals), max(vals)
    im = None
    for ax, ds in zip(axes, datasets):
        M = np.full((len(sigmas), len(grids)), np.nan)
        for i, sg in enumerate(sigmas):
            for j, gb in enumerate(grids):
                c = cells.get((ds, sg, gb))
                if c:
                    M[i, j] = c[0]
        im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis",
                       vmin=vmin, vmax=vmax)
        for i, sg in enumerate(sigmas):
            for j, gb in enumerate(grids):
                c = cells.get((ds, sg, gb))
                if not c:
                    continue
                failed = c[2] > 0.05
                if failed:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                               hatch="xxx", edgecolor="#d62728",
                                               linewidth=0))
                ax.text(j, i, f"{c[0]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if c[0] < (vmin + vmax) / 2 else "black")
        ax.set_xticks(range(len(grids)))
        ax.set_xticklabels([f"{g:g}" for g in grids], fontsize=8)
        ax.set_yticks(range(len(sigmas)))
        ax.set_yticklabels([f"{sg:g}" for sg in sigmas], fontsize=8)
        ax.set_xlabel("grid_bounds", fontsize=9)
        ax.set_ylabel(r"$\sigma_\perp$", fontsize=10)
        ref = GTM_REF.get(ds)
        ax.set_title(f"{ds}" + (f"   (GTM {ref:.3f})" if ref else ""), fontsize=10)
    for ax in axes[len(datasets):]:
        ax.axis("off")
    if im is not None:
        cb = fig.colorbar(im, ax=axes.tolist(), shrink=0.75, pad=0.02)
        cb.set_label("DistCor (mean over 3 seeds)", fontsize=9)
    fig.suptitle(title + "\n" + subtitle, fontsize=12)
    save_fig(fig, out_name)


def fig_b2d():
    print("B2D (sigma_perp x grid_bounds, chemistry)...")
    rows = _load_joint("B2D", "sigma_grid_joint.json")
    if not rows:
        return
    _joint_heatmap(
        rows, ["bace", "bbbp", "dude", "drd2", "egfr", "esol"],
        r"B2D: $\sigma_\perp$ and grid_bounds are not independent (chemistry)",
        "fig5d_sigma_grid_chemistry.png",
        "Red hatching = >5% of points outside grid_bounds (constraint violated). "
        r"DistCor rises monotonically as $\sigma_\perp$ falls, so the setting is "
        "chosen by containment, not by an interior optimum.")


def fig_b2e():
    print("B2E (sigma_perp x grid_bounds, manifolds)...")
    rows = _load_joint("B2E", "sigma_grid_manifold.json")
    if not rows:
        return
    _joint_heatmap(
        rows, ["swiss_roll", "s_curve", "severed_sphere"],
        r"B2E: the same sweep at $d_\perp=1$ inverts the $\sigma_\perp$ response",
        "fig5e_sigma_grid_manifold.png",
        "Red hatching = >5% of points outside grid_bounds. Unlike chemistry, low "
        r"$\sigma_\perp$ HURTS: with one off-manifold dimension there is no "
        "variance reservoir to transfer into the map.")

    # The headline contrast: sigma_perp's response curve has a different SHAPE
    # in the two regimes. A heatmap per dataset shows the surface; this shows
    # the one-dimensional cut that makes the regime difference legible.
    chem = _load_joint("B2D", "sigma_grid_joint.json")
    if not chem:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for rws, label, color, gb in [
            (chem, "chemistry ($d_\\perp\\approx30$)", "tab:blue", 12.0),
            (rows, "3D manifolds ($d_\\perp=1$)", "tab:orange", 12.0)]:
        cells = _joint_cells(rws)
        sigmas = sorted({k[1] for k in cells})
        dss = sorted({k[0] for k in cells})
        mus, sds, xs = [], [], []
        for sg in sigmas:
            per = [cells[(d, sg, gb)][0] for d in dss if (d, sg, gb) in cells]
            if len(per) < 2:
                continue
            xs.append(sg); mus.append(np.mean(per)); sds.append(np.std(per))
        ax.errorbar(xs, mus, yerr=sds, fmt="-o", color=color, capsize=3,
                    label=label, lw=1.8, markersize=5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\sigma_\perp$   (grid_bounds = 12 throughout)")
    ax.set_ylabel("DistCor (mean over datasets and seeds)")
    ax.set_title(r"$\sigma_\perp$'s effect reverses direction with $d_\perp$"
                 "\nchemistry: squeezing $z_\\perp$ helps, monotonically.  "
                 "manifolds: it hurts, then flattens.", fontsize=11)
    ax.axvline(0.003, ls=":", color="tab:blue", alpha=.7)
    ax.axvline(1.0, ls=":", color="tab:orange", alpha=.7)
    ax.annotate("chosen 0.003", xy=(0.003, ax.get_ylim()[0]), fontsize=8,
                color="tab:blue", rotation=90, va="bottom", ha="right")
    ax.annotate("chosen 1.0", xy=(1.0, ax.get_ylim()[0]), fontsize=8,
                color="tab:orange", rotation=90, va="bottom", ha="right")
    ax.grid(alpha=.25)
    ax.legend(fontsize=9)
    save_fig(fig, "fig5f_sigma_regimes.png")


# ── B4D: repulsion_weight at the new operating point ─────────────────────────

def fig_b4d():
    print("B4D (repulsion_weight at sigma_perp=0.003, grid_bounds=12)...")
    rows = _load_joint("B4D", "repulsion_at_global.json")
    if not rows:
        return
    acc = {}
    for r in rows:
        acc.setdefault(r["repulsion_weight"], []).append(r)
    ws = sorted(acc)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    for ax, (key, lbl) in zip(axes, [("dist_corr", "DistCor"),
                                     ("t_c_f1", "T/C F1"),
                                     ("frac_outside_grid_bounds", "% outside grid_bounds")]):
        mus = [np.mean([x[key] for x in acc[w]]) for w in ws]
        sds = [np.std([x[key] for x in acc[w]]) for w in ws]
        if key.startswith("frac"):
            mus = [100 * m for m in mus]; sds = [100 * s for s in sds]
        ax.errorbar(ws, mus, yerr=sds, fmt="-o", color="tab:purple", capsize=3, lw=1.8)
        if key.startswith("frac"):
            ax.axhline(5.0, ls="--", color="tab:red", lw=1,
                       label="5% containment constraint")
            ax.legend(fontsize=8)
        ax.set_xlabel("repulsion_weight")
        ax.set_ylabel(lbl)
        ax.grid(alpha=.25)
    fig.suptitle("B4D: repulsion's effect on global structure REVERSES at the new "
                 "operating point\n"
                 r"at $\sigma_\perp$=10/gb=6 DistCor rose with repulsion (B4, digits); "
                 r"at $\sigma_\perp$=0.003/gb=12 it falls (6 chemistry datasets, 3 seeds)",
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    save_fig(fig, "fig6b_repulsion_at_global.png")


# ── B14: exact log-likelihood as an applicability-domain signal ──────────────

def fig_b14():
    print("B14 (density as an applicability-domain signal)...")
    rows = _load_joint("B14", "density_ood.json")
    if not rows:
        return
    # ESOL's novel_classes cell is excluded: ESOL is a regression target binned
    # into 3 classes, so "hold out the upper class" is an extrapolation split,
    # not an OOD one - all three signals score below chance there (0.37/0.34/
    # 0.29), which is the tell. The loader reports is_continuous=False, which is
    # why B14's >=3-classes branch constructed it at all.
    rows = [r for r in rows
            if not (r["dataset"] == "esol" and r["ood"] == "novel_classes")]
    SIGNALS = ["nfmap_loglik", "knn_distance", "pca_recon_error"]
    LBL = {"nfmap_loglik": "NFMap log-likelihood",
           "knn_distance": "kNN distance (baseline)",
           "pca_recon_error": "PCA reconstruction error"}
    CHEM = {"bace", "bbbp", "dude", "drd2", "egfr", "esol"}

    acc = {}
    for r in rows:
        acc.setdefault((r["dataset"], r["signal"]), []).append(r["auc"])
    dss = sorted({r["dataset"] for r in rows}, key=lambda d: (d not in CHEM, d))

    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    x = np.arange(len(dss))
    w = 0.26
    for i, sig in enumerate(SIGNALS):
        mus = [np.mean(acc.get((d, sig), [np.nan])) for d in dss]
        sds = [np.std(acc.get((d, sig), [np.nan])) for d in dss]
        ax.bar(x + (i - 1) * w, mus, w, yerr=sds, capsize=2, label=LBL[sig])
    ax.axhline(0.5, ls="--", color="#888", lw=1, label="chance")
    nchem = sum(d in CHEM for d in dss)
    ax.axvline(nchem - 0.5, color="#333", lw=1.2, ls=":")
    ax.text(nchem / 2 - 0.5, 1.02, "chemistry", ha="center", fontsize=9, color="#333")
    ax.text((nchem + len(dss)) / 2 - 0.5, 1.02, "non-chemistry", ha="center",
            fontsize=9, color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels(dss, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("OOD detection AUC")
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8.5, loc="lower left", ncol=2)
    ax.grid(axis="y", alpha=.25)
    ax.set_title("B14: NFMap's exact log-likelihood as an applicability-domain signal\n"
                 "mean over OOD constructions and 3 seeds. Ahead of the kNN baseline on "
                 "all 6 chemistry sets; behind on the 2D toys, where $z_\\perp$ is nearly "
                 "empty.", fontsize=11.5)
    save_fig(fig, "fig11_density_applicability_domain.png")


# ── dispatch ─────────────────────────────────────────────────────────────────

DISPATCH = {
    "fig1": lambda: (print("Fig 1 (architecture diagram)..."), fig1_architecture_diagram.main(),
                     print("Fig 1b (manifold geometry intuition)..."), fig1b_manifold_geometry.main(),
                     print("Fig 1c (manifold thickness)..."), fig1c_manifold_thickness.main()),
    "a1": fig_a1,
    "a2": fig_a2,
    "b1": fig_b1,
    "b2": fig_b2,
    "b2c": fig_b2c,
    "b2d": fig_b2d,
    "b2e": fig_b2e,
    "b3": fig_b3,
    "b4": fig_b4,
    "b4b": fig_b4b,
    "b4d": fig_b4d,
    "b5": fig_b5,
    "b9": fig_b9,
    "b14": fig_b14,
    "table2": table_ablation_summary,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", default="all",
                    help="comma-separated subset of: " + ",".join(DISPATCH))
    args = ap.parse_args()
    which = list(DISPATCH) if args.which == "all" else args.which.split(",")
    for key in which:
        if key not in DISPATCH:
            raise SystemExit(f"unknown figure key {key!r}; choose from {', '.join(DISPATCH)}")
        DISPATCH[key]()


if __name__ == "__main__":
    main()
