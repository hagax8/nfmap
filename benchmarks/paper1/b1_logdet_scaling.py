"""B1: Log-Det Dimension Scaling (Novelty #1).

Claim: without the (n_components / d_scaled) volume-loss scaling factor, the
flow "cheats" by stretching the perpendicular (z_perp, "garbage"/thick-
manifold) dimensions to inflate the log-det reward, at the expense of the
real 2D topological mapping. Tests vol_weight=0.0 (scaling off) vs. 1.0
(scaling on, NFMap's default) on DUD-E EGFR (136D, the doc's "hard case").

No changes to nfmap.py were needed for this: `model.loss_history_` already
tracks per-epoch loss, and the per-dimension Jacobian column norms are
computed here POST-HOC on the already-fitted model using its existing public
attributes (module_.forward_flow/inverse_flow, pca_, global_scale_) — the
same JVP trick exact_density() already uses internally, just run from outside
against the full z (parallel + perp) instead of the parallel-only,
z_perp-zeroed version exact_density() uses for its own pullback-density
calculation.

Usage:
    python b1_logdet_scaling.py --dataset dude --seed 42
"""
import os
# Fix macOS OpenMP/Accelerate segfaults - same workaround as
# benchmarks/nfmap_diagnostics.py. Empirically needed here specifically
# (unlike run_ablation.py/b5_gmm_grid.py, which don't hit this): the nested
# double-backward torch.autograd.grad calls in compute_jacobian_column_norms
# below reliably segfaulted (exit 139) without this, even though a plain
# fit_transform() never does.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["TBB_CXX_TYPE"] = "clang"

import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from common.data import load_dataset
from common.metrics import compute_dr_metrics
from common.io import run_dir

from nfmap import NFMap

DEFAULTS = dict(
    n_components=2,
    use_canonical=False,
    complexity="medium",
    inject_weight=0.5,
    n_prior_components=100,
    prior_weight=0.1,
    grid_bounds=6.0,
    graph_weight=5.0,
    repulsion_weight=1.0,
    prior_layout="hexagonal",
)


def compute_jacobian_column_norms(model, X_raw, n_samples=100, seed=42):
    """Per-dimension Jacobian column norms of the full flow bijection, at the
    model's ACTUAL trained z (parallel + perp) for a subsample of the input -
    deliberately not the z_perp=0 conditioning exact_density() uses, since the
    point here is whether z_perp columns are inflated in practice, not the
    manifold-conditioned pullback density.

    Returns (parallel_norms, perp_norms): 1D arrays, one value per
    (sample, dimension) pair in each group.
    """
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X_raw), size=min(n_samples, len(X_raw)), replace=False)
    X_sub = X_raw[idx]

    # Replicate NFMap.transform()'s preprocessing chain (pca_ [+ scaler_pre_]
    # -> /global_scale_) up to, but not including, its no_grad inverse_flow
    # call - we need gradients past this point.
    if model.is_binary_:
        X_pca = model.pca_.transform(X_sub)
    else:
        X_dense_scaled = model.scaler_pre_.transform(np.asarray(X_sub, dtype=np.float32))
        X_pca = model.pca_.transform(X_dense_scaled)
    X_scaled = X_pca / model.global_scale_

    model.module_.eval()
    x_t = torch.tensor(X_scaled, dtype=torch.float32, device=model.device_)
    z, _ = model.module_.inverse_flow(x_t)
    z_req = z.detach().requires_grad_(True)
    x_gen, _ = model.module_.forward_flow(z_req)

    u = torch.ones_like(x_gen, requires_grad=True)
    g = torch.autograd.grad(torch.sum(x_gen * u), z_req, create_graph=True)[0]

    D_scaled = z.shape[1]
    J_cols = []
    for i in range(D_scaled):
        retain = i < D_scaled - 1
        J_col = torch.autograd.grad(torch.sum(g[:, i]), u, create_graph=False, retain_graph=retain)[0]
        J_cols.append(J_col.detach())
    J = torch.stack(J_cols, dim=2)          # [B, ambient_D, D_scaled]
    col_norms = J.norm(dim=1).cpu().numpy()  # [B, D_scaled]

    n_components = model.n_components
    return col_norms[:, :n_components].flatten(), col_norms[:, n_components:].flatten()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="dude")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = load_dataset(args.dataset, seed=args.seed)
    X, y, is_continuous = data["X"], data["y"], data["is_continuous"]
    print(f"[B1] {args.dataset}: {X.shape[0]} x {X.shape[1]}")

    for vol_weight in [0.0, 1.0]:
        config = dict(DEFAULTS)
        config["vol_weight"] = vol_weight
        config["random_state"] = args.seed
        label = f"vol_weight_{str(vol_weight).replace('.', 'p')}"

        print(f"\nFitting vol_weight={vol_weight}...")
        model = NFMap(**config)
        Z = model.fit_transform(X)
        metrics = compute_dr_metrics(X, Z, y, is_continuous=is_continuous)
        print(f"  T/C_F1={metrics['t_c_f1']:.3f} DistCor={metrics['dist_corr']:.3f} "
              f"DensCor={metrics['dens_corr']:.3f}")

        parallel_norms, perp_norms = compute_jacobian_column_norms(model, X, seed=args.seed)
        print(f"  Jacobian col norms: parallel mean={parallel_norms.mean():.3f} "
              f"(std={parallel_norms.std():.3f}), perp mean={perp_norms.mean():.3f} "
              f"(std={perp_norms.std():.3f})")

        out_dir = run_dir("B1", args.dataset, label)
        run_config = dict(config, dataset=args.dataset, vol_weight=vol_weight, seed=args.seed)
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(run_config, f, indent=2, default=str)
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        np.savez(
            os.path.join(out_dir, "diagnostics.npz"),
            Z=Z, y=y,
            loss_history=np.array(model.loss_history_, dtype=np.float32),
            parallel_norms=parallel_norms,
            perp_norms=perp_norms,
        )
        print(f"  Saved -> {out_dir}")


if __name__ == "__main__":
    main()
