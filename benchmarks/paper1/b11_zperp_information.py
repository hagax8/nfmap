"""B11: is the "thick manifold" vault actually holding anything?

Novelty #2 claims z_perp stores fine-grained detail that will not fit on the 2D
map. B2 only establishes the negative half of that - crushing sigma_perp degrades
the map - which is the absence of harm, not the presence of stored information.
Nothing in the paper yet shows the vault has contents.

THE BIJECTION TRAP, and how this script avoids it. NFMap is a bijection, so
z = (z_par, z_perp) contains all of x by construction at ANY sigma_perp. Hence
"z_par+z_perp predicts better than z_par" is true before we run anything and
proves nothing; and crushing sigma_perp does not destroy information either, it
compresses it into a small numerical range. A naive version of this experiment
yields a confident, meaningless positive.

What IS non-trivial is the trade-off the paper actually claims:

    crushing sigma_perp should push structure OUT of z_perp and INTO z_par,
    making z_par alone a BETTER predictor while making the map a WORSE map.

B2 already owns the second half. This script measures the first. Guards:
  * every block is standardised before the probe, so we test structure, not scale
    (a crushed z_perp with intact rank structure will look identical to a default
    one - which would itself be the finding: thickness buys map cleanliness, not
    extra information);
  * a PCA-matched baseline (first 2 components vs. the rest) says whether NFMap's
    split is better than a trivial linear one;
  * two probes - a linear one (the standard representation-learning choice, and
    robust to z_perp having 60x more dimensions than z_par) and kNN. Where they
    disagree, the signal is present but not linearly accessible, which is worth
    knowing rather than averaging away.

E1b, the map-collision test, is the sharper claim and lives in the same script
because it reuses the latent decomposition: among pairs of points that land in
nearly the SAME place on the visible map, does their z_perp distance track their
property difference? That is the activity-cliff question stated exactly, and the
direct test of "points sharing a map location are distinguished by z_perp".

Usage:
    python b11_zperp_information.py                  # E1a, all datasets
    python b11_zperp_information.py --collision-only # E1b only (chemistry)
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

sys.path.insert(0, ".")
from common.data import load_dataset
from common.io import RESULTS_ROOT
from run_ablation import DEFAULTS

from nfmap import NFMap

EXPERIMENT_ID = "B11"

CRUSHED, DEFAULT_SIGMA = 0.0001, 10.0

# moons/circles are excluded by construction: 2D inputs give d_scaled=2 and
# n_components=2, so z_perp is empty - there is no vault to inspect.
E1A_DATASETS = ["digits", "wine", "cancer", "iris", "bace", "bbbp", "dude",
                "swiss_roll", "s_curve", "severed_sphere"]
SQUARE_LAYOUT = {"swiss_roll", "s_curve", "severed_sphere"}


def fit_and_split_latent(X, sigma_perp, seed, dataset):
    cfg = dict(DEFAULTS)
    if dataset in SQUARE_LAYOUT:
        cfg["prior_layout"] = "square"
    cfg["sigma_perp"] = sigma_perp
    cfg["random_state"] = seed
    model = NFMap(**cfg)
    model.fit(X)
    Z_full = model.transform(X, return_full=True)
    n = model.n_components
    return model, Z_full[:, :n], Z_full[:, n:]


def probe(block_tr, block_te, y_tr, y_te, is_continuous, seed):
    """Standardised linear + kNN probe. Returns (linear_score, knn_score)."""
    if block_tr.shape[1] == 0:
        return float("nan"), float("nan")
    sc = StandardScaler().fit(block_tr)
    a, b = sc.transform(block_tr), sc.transform(block_te)
    if is_continuous:
        lin = Ridge(alpha=1.0, random_state=seed).fit(a, y_tr).score(b, y_te)
        knn = KNeighborsRegressor(n_neighbors=5).fit(a, y_tr).score(b, y_te)
    else:
        lin = LogisticRegression(max_iter=2000, random_state=seed).fit(a, y_tr).score(b, y_te)
        knn = KNeighborsClassifier(n_neighbors=5).fit(a, y_tr).score(b, y_te)
    return float(lin), float(knn)


def run_e1a(datasets, seeds):
    rows = []
    for ds in datasets:
        data = load_dataset(ds, seed=42)
        X, y, is_cont = data["X"], data["y"], data["is_continuous"]
        for seed in seeds:
            idx_tr, idx_te = train_test_split(
                np.arange(len(y)), test_size=0.2, random_state=seed,
                stratify=None if is_cont else y)
            for sigma in (CRUSHED, DEFAULT_SIGMA):
                _, z_par, z_perp = fit_and_split_latent(X, sigma, seed, ds)
                blocks = {
                    "z_par": z_par,
                    "z_perp": z_perp,
                    "both": np.hstack([z_par, z_perp]),
                }
                for name, B in blocks.items():
                    lin, knn = probe(B[idx_tr], B[idx_te], y[idx_tr], y[idx_te], is_cont, seed)
                    rows.append(dict(dataset=ds, seed=seed, sigma_perp=sigma, block=name,
                                     n_dims=int(B.shape[1]), linear=lin, knn=knn))
                    print(f"  {ds:<15} seed={seed} sigma={sigma:<8} {name:<7} "
                          f"(d={B.shape[1]:>3}) linear={lin:.3f} knn={knn:.3f}", flush=True)

            # PCA-matched control: is NFMap's 2-vs-rest split better than a
            # trivial linear one at the same dimensions?
            p = PCA(n_components=min(X.shape), random_state=seed).fit(X[idx_tr])
            P = p.transform(X)
            for name, B in {"pca_first2": P[:, :2], "pca_rest": P[:, 2:]}.items():
                lin, knn = probe(B[idx_tr], B[idx_te], y[idx_tr], y[idx_te], is_cont, seed)
                rows.append(dict(dataset=ds, seed=seed, sigma_perp=None, block=name,
                                 n_dims=int(B.shape[1]), linear=lin, knn=knn))
                print(f"  {ds:<15} seed={seed} {'PCA control':<13} {name:<7} "
                      f"(d={B.shape[1]:>3}) linear={lin:.3f} knn={knn:.3f}", flush=True)
    return rows


def run_e1b(seeds, k=10):
    """Map-collision test: among same-map-location pairs, does z_perp track |dy|?"""
    # drd2 now comes from the shared loader (b10 was removed with the rest of
    # the LMDT work). Note this uses the binarised pIC50>=7.0 label A2 uses,
    # not the continuous pIC50 the old b10 loader returned, so |dy| here is a
    # class difference rather than a potency difference.
    data = load_dataset("drd2", return_mols=True)
    X, y = data["X"], data["y"].astype(float)
    rows = []
    for seed in seeds:
        for sigma in (CRUSHED, DEFAULT_SIGMA):
            _, z_par, z_perp = fit_and_split_latent(X, sigma, seed, "drd2")

            # k nearest neighbours in the VISIBLE map = "same place on the map"
            d_par = np.linalg.norm(z_par[:, None, :] - z_par[None, :, :], axis=-1)
            np.fill_diagonal(d_par, np.inf)
            nbr = np.argsort(d_par, axis=1)[:, :k]
            i_idx = np.repeat(np.arange(len(y)), k)
            j_idx = nbr.ravel()
            keep = i_idx < j_idx                      # dedupe unordered pairs
            i_idx, j_idx = i_idx[keep], j_idx[keep]

            d_perp = np.linalg.norm(z_perp[i_idx] - z_perp[j_idx], axis=1)
            d_par_pairs = d_par[i_idx, j_idx]
            dy = np.abs(y[i_idx] - y[j_idx])

            rho_perp, p_perp = spearmanr(d_perp, dy)
            # Partial Spearman: the collision pairs are close in z_par but not
            # identical, and the residual z_par distance explains some of |dy|
            # on its own (see rho_par below). Residualising both rank vectors
            # against it isolates what z_perp adds beyond the visible map,
            # which is the actual claim - without this, z_perp gets credit for
            # structure the map was already showing.
            def _rank(v):
                return np.argsort(np.argsort(v)).astype(float)
            rp, rd, rc = _rank(d_perp), _rank(dy), _rank(d_par_pairs)
            rc_c = rc - rc.mean()
            resid_p = rp - rp.mean() - rc_c * ((rp - rp.mean()) @ rc_c) / (rc_c @ rc_c)
            resid_d = rd - rd.mean() - rc_c * ((rd - rd.mean()) @ rc_c) / (rc_c @ rc_c)
            rho_partial = float(np.corrcoef(resid_p, resid_d)[0, 1])
            # Control 1: among these same collision pairs, the residual z_par
            # distance should explain little - they were selected to be close.
            rho_par, _ = spearmanr(d_par_pairs, dy)
            # Control 2: random pairs, to show the collision selection matters.
            rng = np.random.RandomState(seed)
            ri, rj = rng.randint(0, len(y), len(i_idx)), rng.randint(0, len(y), len(i_idx))
            m = ri != rj
            rho_rand, _ = spearmanr(np.linalg.norm(z_perp[ri[m]] - z_perp[rj[m]], axis=1),
                                    np.abs(y[ri[m]] - y[rj[m]]))

            rows.append(dict(seed=seed, sigma_perp=sigma, n_pairs=int(len(i_idx)),
                             rho_zperp_vs_dy=float(rho_perp), p_value=float(p_perp),
                             rho_zperp_partial=rho_partial,
                             rho_zpar_vs_dy=float(rho_par),
                             rho_random_pairs=float(rho_rand),
                             dy_std_among_collisions=float(dy.std()),
                             dy_std_overall=float(np.abs(y[ri[m]] - y[rj[m]]).std())))
            print(f"  drd2 seed={seed} sigma={sigma:<8} n_pairs={len(i_idx):>5} "
                  f"rho(|dz_perp|,|dy|)={rho_perp:+.3f} (p={p_perp:.1e}) partial={rho_partial:+.3f}  "
                  f"[control rho(|dz_par|,|dy|)={rho_par:+.3f}, random={rho_rand:+.3f}]  "
                  f"sd|dy| collisions={dy.std():.3f} vs overall={np.abs(y[ri[m]] - y[rj[m]]).std():.3f}",
                  flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default=",".join(E1A_DATASETS))
    ap.add_argument("--seeds", default="42,0,1")
    ap.add_argument("--collision-only", action="store_true")
    ap.add_argument("--skip-collision", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    out_dir = os.path.join(RESULTS_ROOT, EXPERIMENT_ID)
    os.makedirs(out_dir, exist_ok=True)

    if not args.collision_only:
        print(f"[{EXPERIMENT_ID}] E1a: predictive decomposition")
        rows = run_e1a([d.strip() for d in args.datasets.split(",")], seeds)
        with open(os.path.join(out_dir, "e1a_decomposition.json"), "w") as f:
            json.dump(rows, f, indent=2)
        print(f"  -> {out_dir}/e1a_decomposition.json")

    if not args.skip_collision:
        print(f"\n[{EXPERIMENT_ID}] E1b: map-collision test (drd2)")
        rows = run_e1b(seeds)
        with open(os.path.join(out_dir, "e1b_collisions.json"), "w") as f:
            json.dump(rows, f, indent=2)
        print(f"  -> {out_dir}/e1b_collisions.json")


if __name__ == "__main__":
    main()
