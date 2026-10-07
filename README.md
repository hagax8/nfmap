# NFMap: Bounded Normalizing Flow Maps

A PyTorch and scikit-learn compatible library for 2D dimensionality reduction
with a **bijective normalizing flow** over a **fixed Gaussian-mixture grid
prior**.

NFMap maps data to a latent space of the same dimension, split into a
2-dimensional navigable map (`z_parallel`) and an off-manifold residual
(`z_perp`). The flow is **exactly bijective**: carrying both parts back through
it reconstructs the input to floating-point precision. Mapping back from a 2D
location *alone* is instead a generative operation — the residual is imputed, so
one map position corresponds to a family of data points. Because the prior
over the map coordinates is a *fixed* grid rather than a learned mixture, the
map has a bounded domain of known extent with reference nodes at known
positions — the property NFMap inherits from Generative Topographic Mapping
(GTM). Note this does *not* mean coordinates are comparable between separately
trained models: the grid pins the prior, but the flow is learned, so two models
place the same molecules differently.

## What is distinctive

- **Fixed GMM grid prior.** Component means sit on a hexagonal or square grid of
  radius `grid_bounds` and are not learned, so the embedding has a bounded
  domain on a defined scale rather than an arbitrary run-dependent one. This is
  GTM's construction, carried into a flow.
- **A single scalar for the local/global trade-off.** The prior factorises as
  `p(z) = p_GMM(z_par) · N(z_perp; 0, sigma_perp² I)`. Because the flow is
  bijective and volume-penalised, variance squeezed out of `z_perp` must
  reappear in `z_par`, so `sigma_perp` controls how much of the data's geometry
  is expressed in the two visible dimensions.
- **Exact likelihood**, usable as an applicability-domain / out-of-distribution
  signal.
- **Out-of-sample projection and an exact inverse**, neither of which t-SNE or
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

Z = model.fit_transform(X)            # (n_samples, 2) map coordinates
Z_new = model.transform(X_new)        # project unseen points
logp = model.score_samples(X_new)     # exact log-likelihood

# Generate: a 2D map location back to data space. The residual is imputed,
# so this samples from the family of points sitting at that position.
X_gen = model.inverse_transform(Z)

# Exact round trip: keep the residual and the reconstruction is lossless.
Z_full = model.transform(X, return_full=True)   # (n_samples, d)
X_rec = model.inverse_transform(Z_full)         # == X, to float precision
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

Research code, under active development. Two caveats worth knowing:

- A single fit has non-trivial run-to-run variation (DistCor spread of ~0.15
  across seeds at a fixed setting), so results should be averaged over seeds.
- Map coordinates are **not** comparable between separately trained models, or
  even between seeds on the same data. Within one fitted model the mapping is
  deterministic and unseen points project onto the same frame.

## Licence

MIT — see [LICENSE](LICENSE).
