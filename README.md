# NFMap: Bounded Normalizing Flow Maps

A PyTorch and scikit-learn compatible library for 2D dimensionality reduction
with a **bijective normalizing flow** over a **fixed Gaussian-mixture grid
prior**.

NFMap maps data to a latent space of the same dimension, split into a
2-dimensional navigable map (`z_parallel`) and an off-manifold residual
(`z_perp`). Because the flow is bijective, the map is **fully invertible**: any
location on the 2D canvas can be carried back to data space. Because the prior
over the map coordinates is a *fixed* grid rather than a learned mixture, map
positions are defined by construction and comparable across models — the
property NFMap inherits from Generative Topographic Mapping (GTM).

## What is distinctive

- **Fixed GMM grid prior.** Component means sit on a hexagonal or square grid of
  radius `grid_bounds` and are not learned. This is GTM's construction, carried
  into a flow.
- **A single scalar for the local/global trade-off.** The prior factorises as
  `p(z) = p_GMM(z_par) · N(z_perp; 0, sigma_perp² I)`. Because the flow is
  bijective and volume-penalised, variance squeezed out of `z_perp` must
  reappear in `z_par`, so `sigma_perp` controls how much of the data's geometry
  is expressed in the two visible dimensions.
- **Exact likelihood**, usable as an applicability-domain / out-of-distribution
  signal.
- **Out-of-sample projection and exact inversion**, neither of which t-SNE or
  UMAP provides.

A caveat worth stating up front: the *direction* of the `sigma_perp` trade-off
depends on the data. On high-dimensional descriptor data, shrinking it improves
global distance fidelity; on low-dimensional manifolds that fold back on
themselves in the ambient space, it degrades them. See `benchmarks/paper1/`.

## Related work

NFMap was developed independently from, and concurrently with,
[FloDR](https://arxiv.org/abs/2607.26278) (Baraka & Probst, arXiv:2607.26278,
July 2026), which also trains a bijection from affine coupling layers, reads the
leading coordinates as a 2D embedding and retains the remainder as a residual.
The approaches differ in two respects: NFMap places a **fixed grid prior** on
the map coordinates where FloDR learns a Gaussian mixture, and NFMap models the
residual **isotropically with a single scale** `sigma_perp` where FloDR
conditions the residual on the embedding and trades local against global
structure through an ordinal loss weight.

## Installation

```bash
git clone https://github.com/hagax8/nfmap.git
cd nfmap
pip install -e .
```

Requires Python >= 3.8, PyTorch, NumPy, scikit-learn.

## Usage

```python
from nfmap import NFMap

model = NFMap(n_components=2, sigma_perp=10.0, grid_bounds=12.0)
Z = model.fit_transform(X)          # (n_samples, 2) map coordinates
Z_new = model.transform(X_new)      # project unseen points
X_back = model.inverse_transform(Z) # carry map locations back to data space
logp = model.score_samples(X_new)   # exact log-likelihood
```

`sigma_perp` is the main dial. Small values push geometry into the 2D map
(better global distance fidelity, worse neighbourhood fidelity); large values
leave it in the residual. `grid_bounds` must be chosen together with it — the
two are coupled, and sweeping either alone gives misleading conclusions.

## Repository structure

```text
├── nfmap/                # Core package (RealNVP blocks, pullback geometry)
├── benchmarks/
│   ├── nfmap_diagnostics.py   # NFMap vs GTM / t-SNE / UMAP
│   └── paper1/                # Full benchmark suite and results
│       ├── common/            # Shared data, metrics and IO layer
│       ├── b*.py              # Individual experiments
│       └── results/           # Per-run configs and metrics
└── docs/                 # Mathematical write-ups
```

The benchmark suite in `benchmarks/paper1/` reports every method on an
identical input matrix, with three seeds per cell, and includes hyperparameter
sweeps of the baselines as well as of NFMap. A manuscript describing these
results is in preparation.

## Status

Research code, under active development. A single fit has non-trivial
run-to-run variation, so results should be averaged over seeds.

## Licence

MIT — see [LICENSE](LICENSE).
