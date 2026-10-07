"""B12: do NFMap's coordinates actually mean the same thing twice?

Novelty #5 (GTM via Flows) sells a *navigable, persistent* coordinate system -
the property t-SNE structurally cannot offer and the reason the fixed grid is
worth its small cost in cluster metrics. The paper asserts this in the
Introduction and never measures it. This script measures it.

Protocol. Split into two disjoint training halves A and B plus a shared probe set
C held out of both. Fit on A, fit on B, project C through each, and ask how far
C's points move between the two maps.

Why the comparison needs care, in two places:

1. t-SNE has no out-of-sample projection at all - which is precisely the gap this
   paper exists to fill, so excluding it would be dodging the comparison. We
   instead fit t-SNE on A+C and B+C and read C's coordinates out of each. That is
   a strictly MORE favourable protocol than NFMap/UMAP get (t-SNE sees the probe
   points during fitting), and it is reported as such.

2. The GMM grid is symmetric under rotation and reflection, so even a correctly
   "fixed" grid permits the whole map to be globally transformed between runs. We
   therefore report displacement both RAW and after optimal Procrustes alignment.
   The claim that distinguishes NFMap is stability in the raw, unaligned frame,
   where t-SNE has no frame at all. If NFMap only wins after alignment, the
   navigability claim is materially weaker and the paper should say so.

Scale is normalised per embedding (divide by RMS radius) before any displacement
is computed, so a method is not rewarded for simply drawing a smaller map.

Usage:
    python b12_coordinate_stability.py
    python b12_coordinate_stability.py --datasets digits --seeds 42
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

sys.path.insert(0, ".")
from common.data import load_dataset
from common.io import RESULTS_ROOT
from run_ablation import DEFAULTS

from nfmap import NFMap

EXPERIMENT_ID = "B12"

DATASETS = ["digits", "wine", "cancer", "bace", "bbbp", "dude"]
SQUARE_LAYOUT = {"swiss_roll", "s_curve", "severed_sphere"}


def _normalise(Z):
    """Centre and scale to unit RMS radius, so displacement is comparable across
    methods that happen to draw maps at different scales."""
    Z = Z - Z.mean(0)
    rms = np.sqrt((Z ** 2).sum(1).mean())
    return Z / (rms + 1e-12)


def displacement(Z1, Z2):
    """(raw, procrustes-aligned) mean per-point displacement."""
    A, B = _normalise(Z1), _normalise(Z2)
    raw = float(np.linalg.norm(A - B, axis=1).mean())
    R, _ = orthogonal_procrustes(A, B)          # rotation + reflection only
    aligned = float(np.linalg.norm(A @ R - B, axis=1).mean())
    return raw, aligned


def nfmap_fit_project(X_train, X_probe, seed, dataset):
    cfg = dict(DEFAULTS)
    if dataset in SQUARE_LAYOUT:
        cfg["prior_layout"] = "square"
    cfg["random_state"] = seed
    model = NFMap(**cfg)
    model.fit(X_train)
    return model, model.transform(X_probe)


def _cells(Z, mu):
    return np.argmin(((Z[:, None, :] - mu[None]) ** 2).sum(-1), axis=1)


def grid_metrics(model_a, Z_a, model_b, Z_b):
    """Grid-relative stability, separating structure from global placement.

    A first pass measured only same-cell agreement and scale-normalised
    displacement, and the two disagreed wildly (0.00 agreement against a small
    0.187 displacement). Neither was wrong: the normalisation step centres the
    data, which silently removes a global translation, while cell assignment
    uses absolute coordinates and so sees every point shifted. On wine the two
    maps correlate at r=0.96-0.99 per axis yet sit ~1 grid cell apart, i.e.
    almost all the drift is one shared translation rather than per-point noise.

    So we report the pieces separately: displacement in units of grid spacing
    (interpretable against the structure that is supposed to be fixed), the
    centroid offset on its own, and cell agreement both as-is and after removing
    that offset. The gap between the last two is exactly the part of the
    persistence claim that a global anchor would fix.
    """
    mu = model_a.module_.prior_means.detach().cpu().numpy()
    mu_b = model_b.module_.prior_means.detach().cpu().numpy()
    spacing = None
    d = np.linalg.norm(mu[:, None, :] - mu[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    spacing = float(d.min(1).mean())

    offset = Z_b.mean(0) - Z_a.mean(0)
    drift = float(np.linalg.norm(Z_a - Z_b, axis=1).mean())
    return dict(
        grid_identical=bool(np.allclose(mu, mu_b)),
        grid_spacing=spacing,
        drift_in_cells=drift / spacing,
        centroid_offset_in_cells=float(np.linalg.norm(offset)) / spacing,
        same_cell=float((_cells(Z_a, mu) == _cells(Z_b, mu_b)).mean()),
        same_cell_translation_corrected=float(
            (_cells(Z_a, mu) == _cells(Z_b - offset, mu_b)).mean()),
        corr_x=float(np.corrcoef(Z_a[:, 0], Z_b[:, 0])[0, 1]),
        corr_y=float(np.corrcoef(Z_a[:, 1], Z_b[:, 1])[0, 1]),
    )


def run_dataset(ds, seeds, skip_baselines=False, out_dir=None):
    data = load_dataset(ds, seed=42)
    X = data["X"]
    rows = []
    for seed in seeds:
        rest, idx_c = train_test_split(np.arange(len(X)), test_size=0.2, random_state=seed)
        idx_a, idx_b = train_test_split(rest, test_size=0.5, random_state=seed)
        Xa, Xb, Xc = X[idx_a], X[idx_b], X[idx_c]

        # Baselines must see the SAME representation NFMap works in. NFMap takes
        # the raw input and reduces it internally (TruncatedSVD to 128 for binary
        # fingerprints); handing UMAP/t-SNE the raw matrix compares input spaces
        # rather than algorithms. Fit the reduction on each training half, exactly
        # as NFMap's own does, and apply it to the shared probe set.
        def _reduce(Xtrain, Xprobe):
            if Xtrain.shape[1] <= 128:
                return Xtrain, Xprobe
            r = TruncatedSVD(n_components=128, random_state=seed).fit(Xtrain)
            return r.transform(Xtrain), r.transform(Xprobe)

        Xa_r, Xc_a = _reduce(Xa, Xc)
        Xb_r, Xc_b = _reduce(Xb, Xc)

        model_a, Za = nfmap_fit_project(Xa, Xc, seed, ds)
        model_b, Zb = nfmap_fit_project(Xb, Xc, seed, ds)
        probe = {"nfmap_a": Za, "nfmap_b": Zb}
        save_probe = out_dir is not None and seed == seeds[0]
        raw, aligned = displacement(Za, Zb)
        gm = grid_metrics(model_a, Za, model_b, Zb)
        rows.append(dict(dataset=ds, seed=seed, method="nfmap", raw=raw,
                         aligned=aligned, **gm))
        print(f"  {ds:<8} seed={seed} nfmap  shape_r=({gm['corr_x']:.3f},{gm['corr_y']:.3f}) "
              f"drift={gm['drift_in_cells']:.2f} cells (of which translation "
              f"{gm['centroid_offset_in_cells']:.2f}) same_cell={gm['same_cell']:.3f} "
              f"-> {gm['same_cell_translation_corrected']:.3f} after de-translation", flush=True)

        # Same-data control. The A-vs-B protocol above confounds two things:
        # algorithmic instability, and the two halves genuinely containing
        # different structure. Refitting on the SAME half with a different seed
        # isolates the first. This matters most on bace/bbbp, whose A2 silhouette
        # is 0.03-0.10 - if there is barely any structure to find, two halves can
        # legitimately disagree without that being NFMap's fault, and the
        # difference between these two rows is what says which it is.
        model_b2, Zb2 = nfmap_fit_project(Xa, Xc, seed + 1000, ds)
        raw_s, aligned_s = displacement(Za, Zb2)
        gm_s = grid_metrics(model_a, Za, model_b2, Zb2)
        rows.append(dict(dataset=ds, seed=seed, method="nfmap_same_data",
                         raw=raw_s, aligned=aligned_s, **gm_s))
        print(f"  {ds:<8} seed={seed} nfmap_same_data  raw={raw_s:.3f} "
              f"procrustes={aligned_s:.3f} |r|=({abs(gm_s['corr_x']):.3f},"
              f"{abs(gm_s['corr_y']):.3f})", flush=True)

        if skip_baselines:
            continue
        try:
            import umap
            ua = umap.UMAP(random_state=seed).fit(Xa_r)
            ub = umap.UMAP(random_state=seed).fit(Xb_r)
            Zua, Zub = ua.transform(Xc_a), ub.transform(Xc_b)
            raw, aligned = displacement(Zua, Zub)
            probe["umap_a"], probe["umap_b"] = Zua, Zub
            rows.append(dict(dataset=ds, seed=seed, method="umap", raw=raw,
                             aligned=aligned, grid_cell_agreement=None))
            print(f"  {ds:<8} seed={seed} umap   raw={raw:.3f} aligned={aligned:.3f}", flush=True)
        except Exception as e:
            print(f"  {ds:<8} seed={seed} umap   FAILED ({e})", flush=True)

        # t-SNE: no .transform(), so it must re-fit with the probe points
        # included. Deliberately generous to t-SNE; noted in the write-up.
        try:
            Zta = TSNE(n_components=2, random_state=seed, init="pca").fit_transform(
                np.vstack([Xa_r, Xc_a]))[len(Xa_r):]
            Ztb = TSNE(n_components=2, random_state=seed, init="pca").fit_transform(
                np.vstack([Xb_r, Xc_b]))[len(Xb_r):]
            raw, aligned = displacement(Zta, Ztb)
            probe["tsne_a"], probe["tsne_b"] = Zta, Ztb
            rows.append(dict(dataset=ds, seed=seed, method="tsne_refit", raw=raw,
                             aligned=aligned, grid_cell_agreement=None))
            print(f"  {ds:<8} seed={seed} tsne   raw={raw:.3f} aligned={aligned:.3f} "
                  f"(refit protocol, favourable to t-SNE)", flush=True)
        except Exception as e:
            print(f"  {ds:<8} seed={seed} tsne   FAILED ({e})", flush=True)

        if save_probe:
            np.savez(os.path.join(out_dir, f"{ds}_probe.npz"), **probe)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--seeds", default="42,0,1")
    ap.add_argument("--skip-baselines", action="store_true",
                    help="NFMap only; UMAP/t-SNE need a quiet machine (numba JIT)")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    print(f"[{EXPERIMENT_ID}] coordinate stability: fit on disjoint halves, "
          f"project a shared probe set, measure drift")
    out_dir = os.path.join(RESULTS_ROOT, EXPERIMENT_ID)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "stability.json")

    # Written after every dataset, not once at the end: an earlier run lost a
    # completed dataset when importing a baseline library hung and the process
    # was killed, which a single final write cannot survive.
    rows = []
    for ds in [d.strip() for d in args.datasets.split(",")]:
        rows += run_dataset(ds, seeds, skip_baselines=args.skip_baselines, out_dir=out_dir)
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2)
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
