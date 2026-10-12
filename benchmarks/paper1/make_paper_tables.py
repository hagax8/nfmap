"""LaTeX tables for the short paper, generated from the result files.

Writes one .tex file per table into tables_paper/, each a standalone
\\begin{table} ... \\end{table} block using booktabs. Regenerate rather than
hand-edit, so the manuscript can never drift from the data.

Needs: \\usepackage{booktabs} and \\usepackage{siunitx} (for \\num) in the preamble.

Usage:
    python make_paper_tables.py
"""
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "tables_paper")

CHEM = ["bace", "bbbp", "esol", "drd2", "egfr"]
MANIFOLD = ["swiss_roll", "s_curve", "severed_sphere"]
PRETTY = {"bace": "BACE", "bbbp": "BBBP", "esol": "ESOL", "drd2": "DRD2",
          "egfr": "EGFR", "swiss_roll": "Swiss roll", "s_curve": "S-curve",
          "severed_sphere": "Severed sphere"}


def load(exp, fname):
    p = os.path.join(RES, exp, fname)
    return json.load(open(p)) if os.path.exists(p) else None


def write(name, tex):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        f.write(tex)
    print(f"  -> {p}")


B23_FILE = "table1_sweeps_v2.json"


OTHER = ["digits", "iris", "cancer", "wine"]
TABLE1_METHODS = ["NFMap", "GTM", "t-SNE", "UMAP"]
# Keys B22 uses for its saved fits.
METHOD_KEY = {"NFMap": "nfmap", "GTM": "gtm", "t-SNE": "tsne", "UMAP": "umap"}


def _table1_rows():
    """(nfmap, baselines, gtm) rows behind Table 1, all from B23."""
    all_rows = load("B23", B23_FILE)
    if all_rows is None:
        raise SystemExit(f"results/B23/{B23_FILE} missing - run "
                         "b23_table1_reproducible_sweeps.py first")
    nf = [r for r in all_rows if r["method"] == "NFMap"]
    bl = [r for r in all_rows if r["method"] in ("umap", "tsne")]
    gtm = [r for r in all_rows if r["method"] == "GTM"]
    return nf, bl, gtm


def _table1_points(ds, nf, bl, gtm):
    """Per method, one (global, T/C F1, label, params, off_grid) point per
    setting, seed-averaged. off_grid is NFMap's fraction of points outside
    grid_bounds (None for methods without a canvas). No containment filter:
    it constrained NFMap alone - t-SNE/UMAP have no canvas - and with
    grid_bounds swept it bound only the S-curve. Reported instead."""
    global_key = "dist_corr_geodesic" if ds in MANIFOLD else "dist_corr"
    out = defaultdict(list)

    c = defaultdict(list)
    for r in nf:
        if r["dataset"] == ds:
            c[(r["sigma_perp"], r["grid_bounds"])].append(r)
    for (sg, gb), v in c.items():
        g_vals = [x.get(global_key, x.get("dist_corr", 0)) for x in v]
        t_vals = [x["t_c_f1"] for x in v]
        off = float(np.mean([x.get("frac_outside_grid_bounds", 0) for x in v]))
        if not np.isnan(np.mean(g_vals)):
            out["NFMap"].append((np.mean(g_vals), np.mean(t_vals),
                                 rf"$\sigma_\perp$={sg:g}, grid {gb:g}",
                                 {"sigma_perp": sg, "grid_bounds": gb}, off))

    c = defaultdict(list)
    for r in bl:
        if r["dataset"] == ds:
            key = (r["method"], r.get("perplexity") or (r["n_neighbors"], r["min_dist"]))
            c[key].append(r)
    for (m, tag), v in c.items():
        if m == "tsne":
            label, params = rf"$p$={tag}", {"perplexity": tag}
        else:
            label, params = rf"$k$={tag[0]}, $d$={tag[1]:g}", {"n_neighbors": tag[0], "min_dist": tag[1]}
        g_vals = [x.get(global_key, x.get("dist_corr", 0)) for x in v]
        t_vals = [x["t_c_f1"] for x in v]
        if not np.isnan(np.mean(g_vals)):
            out["t-SNE" if m == "tsne" else "UMAP"].append(
                (np.mean(g_vals), np.mean(t_vals), label, params, None))

    acc = [r for r in gtm if r.get("dataset") == ds]
    if acc:
        g_vals = [x.get(global_key, x.get("dist_corr", 0)) for x in acc]
        t_vals = [x["t_c_f1"] for x in acc]
        if not np.isnan(np.mean(g_vals)):
            out["GTM"] = [(np.mean(g_vals), np.mean(t_vals), "default", {}, None)]
    return out


def _frontier(pts):
    return [p for p in pts if not any(q[0] >= p[0] and q[1] >= p[1] and (q[0] > p[0] or q[1] > p[1]) for q in pts)]


def table1_selection(datasets):
    """The point Table 1 reports for every method on each dataset:
    {ds: {method: (global, T/C F1, label, params, off_grid)}}.

    Each method's sweep is reduced to its Pareto frontier over (global, T/C).
    Manifolds: the frontier's max-global point. Others: the knee, i.e. the
    frontier point nearest (1, 1) after min-max scaling over ALL methods'
    points on that dataset - so one method's knee depends on the others.
    Single source for the table, B21, B22 and the map figures.
    """
    nf, bl, gtm = _table1_rows()
    sel = {}
    for ds in datasets:
        P = _table1_points(ds, nf, bl, gtm)
        allp = [p for v in P.values() for p in v]
        if not allp:
            continue
        dmin, dmax = min(p[0] for p in allp), max(p[0] for p in allp)
        tmin, tmax = min(p[1] for p in allp), max(p[1] for p in allp)
        nz = lambda a, lo, hi: (a - lo) / (hi - lo) if hi > lo else 0.5
        knees = {}
        for m in TABLE1_METHODS:
            if m not in P:
                continue
            fr = _frontier(P[m])
            if ds in MANIFOLD:
                knees[m] = max(fr, key=lambda q: (q[0], q[1]))
            else:
                knees[m] = min(fr, key=lambda q: (1 - nz(q[0], dmin, dmax))**2 + (1 - nz(q[1], tmin, tmax))**2)
        sel[ds] = knees
    return sel


def selected_settings(datasets):
    """{ds: {"nfmap"|"gtm"|"tsne"|"umap": params}} at the Table 1 points."""
    return {ds: {METHOD_KEY[m]: k[3] for m, k in knees.items()}
            for ds, knees in table1_selection(datasets).items()}


def manifold_selection():
    return selected_settings(MANIFOLD)


# ── Table 1: comparison at each method's best setting ────────────────────────

def table_comparison():
    """Table 1: every method at the point table1_selection picks, one row per
    dataset, global and local side by side. Settings go in a companion table
    (tabS1b_settings.tex) so this one fits on a page with its discussion.
    """
    groups = ((CHEM, "Chemistry"), (MANIFOLD, "3-D manifolds"), (OTHER, "Other"))
    sel = table1_selection([d for g, _ in groups for d in g])
    meths = [m for m in TABLE1_METHODS if any(m in v for v in sel.values())]
    name = {"NFMap": r"\nfmap{}"}

    # Off-grid is stated in the caption rather than as a column: NFMap's
    # reported points are almost all fully on the grid.
    off = {ds: k["NFMap"][4] for ds, k in sel.items() if "NFMap" in k}
    flagged = {ds: v for ds, v in off.items() if v is not None and v > 0.02}
    rest_max = max((v for ds, v in off.items() if ds not in flagged and v is not None), default=0)
    off_txt = (rf"\nfmap{{}}'s reported maps keep at least {100 - 100 * rest_max:.0f}\% "
               r"of points inside grid\_bounds"
               + ("" if not flagged else ", except " + ", ".join(
                   rf"{PRETTY.get(d, d.capitalize())} ({100 - 100 * v:.0f}\%)" for d, v in flagged.items()))
               + ".")

    cols = "l" + "cc" * len(meths)
    lines = [
        r"\begin{table}[t]", r"\centering", r"\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{Each method at its own best setting (means over 3 seeds, CPU). "
        r"G: global structure (DistCor; geodesic DistCor on the 3-D manifolds). "
        r"L: local structure (T/C F1). Each sweep is reduced to its Pareto front "
        r"over (G, L); the reported point is the front's knee, or its "
        r"maximum-G point on the manifolds. Best per dataset in bold. "
        + off_txt.replace("\\", "\\\\").replace("\\\\", "\\") +
        r" Settings in Table~\ref{tab:settings}.}",
        r"\label{tab:comparison}",
        rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
        " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{name.get(m, m)}}}" for m in meths) + r" \\",
        "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(meths))),
        " & " + " & ".join(["G & L"] * len(meths)) + r" \\",
        r"\midrule",
    ]
    for group, label in groups:
        present = [ds for ds in group if ds in sel]
        if not present:
            continue
        lines.append(rf"\multicolumn{{{1 + 2 * len(meths)}}}{{l}}{{\textit{{{label}}}}} \\")
        for ds in present:
            k = sel[ds]
            # Bold on the displayed (2 dp) values, so ties as printed are
            # all bold - deciding on unrounded values bolded one of two
            # identical-looking 0.98s.
            best_g = max(round(v[0], 2) for v in k.values())
            best_l = max(round(v[1], 2) for v in k.values())
            cell = lambda v, b: (rf"\textbf{{{v:.2f}}}" if round(v, 2) == b else f"{v:.2f}")
            vals = []
            for m in meths:
                vals += ([cell(k[m][0], best_g), cell(k[m][1], best_l)] if m in k else ["--", "--"])
            lines.append(rf"\quad {PRETTY.get(ds, ds.capitalize())} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("tabS1_comparison.tex", "\n".join(lines))

    # Companion: the setting behind every cell above.
    lines = [
        r"\begin{table}[t]", r"\centering", r"\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{Hyperparameters selected by the sweep for each entry of "
        r"Table~\ref{tab:comparison} (the reported Pareto point; not fixed in "
        r"advance). \nfmap{}: $\sigma_\perp$ / grid\_bounds, swept over 14 "
        r"$\sigma_\perp$ values from $10^{-3}$ to $10^{3}$ and grid\_bounds 12 and 24. "
        r"t-SNE: perplexity. UMAP: n\_neighbors / min\_dist. GTM: not swept, one "
        r"configuration ($10\times10 = 100$ nodes, matching \nfmap{}'s 100 "
        r"prior components; $6\times6$ RBF centres).}",
        r"\label{tab:settings}",
        r"\begin{tabular}{lccc}", r"\toprule",
        r" & \nfmap{} & t-SNE & UMAP \\", r"\midrule",
    ]
    for group, label in groups:
        present = [ds for ds in group if ds in sel]
        if not present:
            continue
        lines.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{label}}}}} \\")
        for ds in present:
            k = sel[ds]
            nf = k.get("NFMap", (0, 0, "", {}))[3]
            ts = k.get("t-SNE", (0, 0, "", {}))[3]
            um = k.get("UMAP", (0, 0, "", {}))[3]
            lines.append(
                rf"\quad {PRETTY.get(ds, ds.capitalize())} & "
                + (f"{nf['sigma_perp']:g} / {nf['grid_bounds']:g}" if nf else "--") + " & "
                + (f"{ts['perplexity']:g}" if ts else "--") + " & "
                + (f"{um['n_neighbors']:g} / {um['min_dist']:g}" if um else "--") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("tabS1b_settings.tex", "\n".join(lines))


# ── Table 2: capabilities ────────────────────────────────────────────────────

def table_capabilities():
    """Capability comparison.

    Two earlier versions of the last column were wrong and are recorded here so
    they are not reintroduced.

    (1) 'Fixed coordinate frame', meaning coordinates comparable between
        models. Measured and false: two NFMap models trained on disjoint
        subsets place the same probe molecules a mean 3.6 units apart on a
        +/-12 canvas (Procrustes disparity 0.71), and two seeds on identical
        data disagree by a mean 10.1 units. The grid pins the prior; the flow
        is learned.

    (2) 'Bounded domain'. True but not the point, and it undersells what the
        grid is for.

    What the fixed grid actually provides is canonical bins: K components at
    known positions, so any quantity can be aggregated per node and rendered as
    a continuous field at a resolution set by K. This is GTM's responsibility-
    landscape machinery. Note the honest limit - any method with an inverse can
    render a field by rastering the plane and inverting it; what it cannot do is
    aggregate observed data at canonical cells.
    """
    tex = r"""\begin{table}[t]
\centering
\small
\caption{Capabilities of each method, independent of embedding quality.
\emph{Parametric} means unseen points can be projected onto an existing map.
\emph{Landscape grid}: the prior places $K$ components at fixed, known
positions, so any quantity---measured activity, predicted property, local
density---can be aggregated per node and rendered as a continuous background
field at a resolution set by $K$. Methods without a fixed latent grid can still
render a field by inverting an arbitrary raster, but have no canonical cells at
which to aggregate observed data. \emph{Heavy-tailed}: the low-dimensional
model has heavier-than-Gaussian tails, the usual remedy for crowding---built
into the similarity kernel of t-SNE (Student-t) and UMAP, and available in
\nfmap{} as a Student-t mixture prior. None of these methods give coordinates
that are comparable between separately trained models.}
\label{tab:capabilities}
\begin{tabular}{lccccc}
\toprule
 & Parametric & Invertible & Exact density & Landscape grid & Heavy-tailed \\
\midrule
t-SNE      & \ding{55} & \ding{55} & \ding{55} & \ding{55} & \ding{51} \\
UMAP       & \ding{51}$^{*}$ & \ding{55} & \ding{55} & \ding{55} & \ding{51} \\
GTM        & \ding{51} & \ding{55} & \ding{51} & \ding{51} & \ding{55} \\
FloDR      & \ding{51} & \ding{51} & \ding{51} & \ding{55}$^{\dagger}$ & \ding{55} \\
\nfmap{}   & \ding{51} & \ding{51} & \ding{51} & \ding{51} & \ding{51}$^{\ddagger}$ \\
\bottomrule
\end{tabular}

\vspace{2pt}
\footnotesize $^{*}$via the parametric variant; the standard implementation is
non-parametric. $^{\dagger}$FloDR's mixture components are learned rather than
fixed, so their positions and effective resolution vary with the data.
$^{\ddagger}$optional (\texttt{prior\_family='student\_t'}); the default prior is
a Gaussian mixture.
\end{table}
"""
    write("tab2_capabilities.tex", tex)


# ── Table 4: prior ablation ──────────────────────────────────────────────────

def table_prior():
    d_b5 = os.path.join(RES, "B5")
    acc_digits = defaultdict(list)
    if os.path.isdir(d_b5):
        for run in os.listdir(d_b5):
            c = json.load(open(os.path.join(d_b5, run, "config.json")))
            m = json.load(open(os.path.join(d_b5, run, "metrics.json")))
            # B5 also holds DRD2 (for the prior figure); this block is Digits.
            if c.get("dataset", "digits") == "digits":
                acc_digits[c["condition"]].append(m)
            
    b21_file = os.path.join(RES, "B21", "prior_ablation.json")
    b21 = json.load(open(b21_file)) if os.path.exists(b21_file) else []
    
    acc_chem = defaultdict(list)
    for r in b21:
        if r["dataset"] in CHEM and abs(r["sigma_perp"] - 0.003) < 1e-5:
            acc_chem[r["condition"]].append(r)
            
    # Each manifold at its own Table 1 sigma_perp - the values B21 fitted. A
    # single hard-coded 0.01 here used to keep only the swiss roll's rows.
    mani = {d: (v["nfmap"]["sigma_perp"], v["nfmap"]["grid_bounds"])
            for d, v in manifold_selection().items() if "nfmap" in v}
    acc_mani = defaultdict(list)
    for r in b21:
        if (r["dataset"] in mani
                and (r["sigma_perp"], r.get("grid_bounds", 12.0)) == mani[r["dataset"]]):
            acc_mani[r["condition"]].append(r)

    names = {"single_gaussian": "Single Gaussian",
             "learnable_gmm": "Learned mixture",
             "fixed_grid": r"Fixed grid (\nfmap{})"}
    
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Prior ablation across datasets (mean $\pm$ sd). "
        r"Pinning the mixture means to a fixed grid matches learning them on "
        r"both measures, while keeping the map on the grid. Against a single "
        r"Gaussian the fixed grid gives equal or better T/C F1 and trades some "
        r"DistCor on Digits and the 3-D manifolds. The fixed grid is adopted for "
        r"the landscape grid it provides (Table~\ref{tab:capabilities}). "
        r"Off grid: share of points outside grid\_bounds; for the learned mixture "
        r"the means start on the grid and are free to move, so this measures "
        r"drift from it. A single Gaussian has no grid, so it is not reported.}",
        r"\label{tab:prior}",
        r"\begin{tabular}{lccc}", r"\toprule",
        r"Prior & DistCor & T/C F1 & Off grid \\", r"\midrule",
    ]
    
    groups = [
        ("Digits", acc_digits),
        ("Chemistry (5 datasets)", acc_chem),
        ("3-D manifolds (3 datasets)", acc_mani)
    ]
    
    for label, acc in groups:
        if not acc:
            continue
        lines.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{label}}}}} \\")
        for cond in ["single_gaussian", "learnable_gmm", "fixed_grid"]:
            v = acc.get(cond, [])
            if not v:
                continue
            dc_mean, dc_std = np.mean([x["dist_corr"] for x in v]), np.std([x["dist_corr"] for x in v])
            tc_mean, tc_std = np.mean([x["t_c_f1"] for x in v]), np.std([x["t_c_f1"] for x in v])
            out = np.mean([x.get("frac_outside_grid_bounds", np.nan) for x in v])
            # A single Gaussian sits at the origin with a grid-independent
            # width; grid_bounds never enters it, so "outside" would only
            # measure how large the map grew.
            off = "--" if cond == "single_gaussian" else rf"{100*out:.1f}\%"
            lines.append(rf"\quad {names[cond]} & ${dc_mean:.3f} \pm {dc_std:.3f}$ & "
                         rf"${tc_mean:.3f} \pm {tc_std:.3f}$ & {off} \\")
        lines.append(r"\addlinespace")
        
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("tab4_prior.tex", "\n".join(lines))


# ── Table 5: run-to-run stability ────────────────────────────────────────────

# ── Table 6: applicability domain ────────────────────────────────────────────

def table_ad():
    rows = load("B14", "density_ood.json")
    if not rows:
        print("  (no B14 results)")
        return
    # DUD-E excluded by choice: EGFR and DRD2 already cover target-based
    # activity sets. (NFMap leads there too, 0.908 vs 0.849.)
    rows = [r for r in rows
            if r["dataset"] != "dude"
            and not (r["dataset"] == "esol" and r["ood"] == "novel_classes")]
    acc = defaultdict(list)
    for r in rows:
        acc[(r["dataset"], r["signal"])].append(r["auc"])
    sigs = [("nfmap_loglik", r"\nfmap{} $\log p(x)$"),
            ("knn_distance", "$k$-NN distance"),
            ("pca_recon_error", "PCA recon.\\ error")]
    dss = sorted({r["dataset"] for r in rows}, key=lambda d: (d not in CHEM, d))
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Out-of-distribution detection AUC, averaged over the applicable "
        r"OOD constructions (feature-permuted inputs and matched-moment Gaussian "
        r"noise for every dataset; held-out novel classes in addition where there "
        r"are at least three classes) and 3 seeds. \nfmap{}'s exact likelihood "
        r"leads on every chemical dataset. On the other sets it trails on average, "
        r"most clearly on the two-dimensional toys (circles, moons), where "
        r"$d_\perp = 0$ and the residual is empty.}",
        r"\label{tab:ad}",
        r"\begin{tabular}{l" + "c" * len(sigs) + "}", r"\toprule",
        "Dataset & " + " & ".join(n for _, n in sigs) + r" \\", r"\midrule",
    ]
    for label, sel in (("Chemistry", [d for d in dss if d in CHEM]),
                       ("Other", [d for d in dss if d not in CHEM])):
        lines.append(rf"\multicolumn{{{1+len(sigs)}}}{{l}}{{\textit{{{label}}}}} \\")
        for ds in sel:
            vals = [np.mean(acc.get((ds, s), [np.nan])) for s, _ in sigs]
            best = int(np.nanargmax(vals))
            cells = [rf"\textbf{{{v:.3f}}}" if i == best else f"{v:.3f}"
                     for i, v in enumerate(vals)]
            lines.append(f"\\quad {PRETTY.get(ds, ds.capitalize())} & " +
                         " & ".join(cells) + r" \\")
        means = [np.nanmean([np.mean(acc.get((d, s), [np.nan])) for d in sel])
                 for s, _ in sigs]
        lines.append(r"\quad \textit{mean} & " +
                     " & ".join(rf"\textit{{{m:.3f}}}" for m in means) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("tab6_applicability_domain.tex", "\n".join(lines))


# ── Prior: non-inferiority tests behind the prior table ─────────────────────

# Practical margins: about the size of seed-to-seed variation. Chosen after
# the data were in, so the paper calls them practical margins, not
# pre-registered ones.
NONINF_MARGIN = {"dist_corr": 0.02, "t_c_f1": 0.01}


def _prior_pairs():
    """Fixed-grid minus alternative, paired by (dataset, seed): Digits from B5,
    chemistry (sigma_perp 0.003) and manifolds (Table 1 setting) from B21 -
    the same rows table_prior reports."""
    mani = {d: (v["nfmap"]["sigma_perp"], v["nfmap"]["grid_bounds"])
            for d, v in manifold_selection().items() if "nfmap" in v}
    rows = [r for r in (load("B21", "prior_ablation.json") or [])
            if (r["dataset"] in CHEM and abs(r["sigma_perp"] - 0.003) < 1e-9)
            or (r["dataset"] in mani
                and (r["sigma_perp"], r.get("grid_bounds", 12.0)) == mani[r["dataset"]])]
    d_b5 = os.path.join(RES, "B5")
    for run in (sorted(os.listdir(d_b5)) if os.path.isdir(d_b5) else []):
        cp = os.path.join(d_b5, run, "config.json")
        if not os.path.exists(cp):
            continue
        c = json.load(open(cp))
        if c.get("dataset", "digits") != "digits":
            continue
        m = json.load(open(os.path.join(d_b5, run, "metrics.json")))
        rows.append(dict(dataset="digits", condition=c["condition"], seed=c["seed"], **m))
    return {(r["dataset"], r["condition"], r["seed"]): r for r in rows}


def prior_noninferiority():
    """One-sided non-inferiority of the fixed grid against each alternative
    prior: H0 mean(fixed - other) <= -margin. One-sample t-test on paired
    differences, confirmed by a Wilcoxon signed-rank test and by a t-test on
    per-dataset mean differences (seeds are not independent replicates).
    Prints a summary and writes results/B21/noninferiority.json."""
    from scipy.stats import ttest_1samp, wilcoxon
    idx = _prior_pairs()
    groups = {"chemistry": CHEM, "manifolds": MANIFOLD,
              "all": ["digits"] + CHEM + MANIFOLD}
    out = []
    print("\nPrior non-inferiority (fixed grid minus alternative, paired by dataset and seed)")
    for other in ("learnable_gmm", "single_gaussian"):
        for met, mg in NONINF_MARGIN.items():
            for g, dss in groups.items():
                pairs = [(ds, s) for (ds, c, s) in idx
                         if c == "fixed_grid" and ds in dss and (ds, other, s) in idx]
                if len(pairs) < 3:
                    continue
                d = np.array([idx[(ds, "fixed_grid", s)][met] - idx[(ds, other, s)][met]
                              for ds, s in pairs])
                per_ds = [np.mean([idx[(ds, "fixed_grid", s)][met] - idx[(ds, other, s)][met]
                                   for x, s in pairs if x == ds])
                          for ds in sorted({x for x, _ in pairs})]
                two = ttest_1samp(d, 0.0)
                rec = dict(
                    alternative=other, metric=met, group=g, margin=mg,
                    n_pairs=len(d), n_datasets=len(per_ds),
                    mean_diff=float(d.mean()),
                    ci95=[float(v) for v in two.confidence_interval(0.95)],
                    p_two_sided=float(two.pvalue),
                    p_noninf_t=float(ttest_1samp(d, -mg, alternative="greater").pvalue),
                    p_noninf_wilcoxon=float(wilcoxon(d + mg, alternative="greater").pvalue),
                    p_noninf_dataset=(float(ttest_1samp(per_ds, -mg, alternative="greater").pvalue)
                                      if len(per_ds) > 2 else None))
                out.append(rec)
                print(f"  vs {other:<16}{met:<10}{g:<10}n={len(d):2d}  "
                      f"diff {rec['mean_diff']:+.4f}  p_noninf={rec['p_noninf_t']:.2g}  "
                      f"wilcoxon={rec['p_noninf_wilcoxon']:.2g}  dataset-level="
                      + (f"{rec['p_noninf_dataset']:.2g}" if rec["p_noninf_dataset"] is not None else "--")
                      + f"  two-sided p={rec['p_two_sided']:.2g}")
    path = os.path.join(RES, "B21", "noninferiority.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  -> {path}")


if __name__ == "__main__":
    print("Writing LaTeX tables...")
    table_comparison()
    table_capabilities()
    table_prior()
    table_ad()
    prior_noninferiority()
