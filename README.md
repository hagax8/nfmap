# NFMap: Bounded Normalizing Flow Maps

A PyTorch and scikit-learn compatible library for 2D dimensionality reduction
with a **bijective normalizing flow** over a **fixed Gaussian-mixture grid
prior**.

NFMap maps data to a latent space of the same dimension, split into a
2-dimensional navigable map (`z_parallel`) and an off-manifold residual
(`z_perp`). The flow is **exactly bijective**: carrying both parts back through
it reconstructs the input to floating-point precision. Because the prior
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
- **A residual scale as the main local/global control.** The prior factorises
  as `p(z) = p_GMM(z_par) · N(z_perp; 0, sigma_perp² I)`. Because the flow is
  bijective and its log-determinant is part of the objective, variance squeezed
  out of `z_perp` reappears in `z_par`, so `sigma_perp` controls how much of the
  data's geometry is expressed in the two visible dimensions. Its effect is
  coupled to `grid_bounds` and the graph weights (see below).
- **Exact likelihood**, usable as an applicability-domain / out-of-distribution
  signal.
- **Out-of-sample projection and an exact inverse**, neither of which t-SNE or
  UMAP provides.
- **Options against crowding on large datasets:** a heavy-tailed Student-t
  mixture prior over the map, and, for molecules, a hierarchical chemical
  kernel that builds the neighbourhood graph from molecule, Bemis-Murcko
  scaffold and generic-skeleton fingerprints.

A caveat worth stating up front: the *direction* of the `sigma_perp` trade-off
depends on the data. On high-dimensional descriptor data, shrinking it improves
global distance fidelity; on low-dimensional manifolds that fold back on
themselves in the ambient space, it degrades them. See `benchmarks/paper1/`.

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

# Exact round trip: keep the residual and the reconstruction is lossless.
Z_full = model.transform(X, return_full=True)   # (n_samples, d)
X_rec = model.inverse_transform(Z_full)         # == X, to float precision
```

`sigma_perp` is the main dial. Small values push geometry into the 2D map
(better global distance fidelity, worse neighbourhood fidelity); large values
leave it in the residual. `grid_bounds` must be chosen together with it — the
two are coupled, and sweeping either alone gives misleading conclusions.

### Large datasets: Student-t prior and chemical kernel

```python
# Student-t mixture prior: heavier tails over the map coordinates. With
# grid_bounds='auto', the canvas is widened to match the heavier tails.
model = NFMap(prior_family="student_t", prior_dof=2.0, grid_bounds="auto")

# Hierarchical chemical kernel (needs RDKit): the neighbourhood graph comes
# from weighted Tanimoto similarity over molecule, scaffold and generic-skeleton
# fingerprints, and is passed to NFMap in place of its own kNN graph.
from chem_kernel.hybrid_kernel import (compute_hierarchical_fingerprints,
                                       compute_hybrid_knn)

X_mol, X_scaf, X_skel, _, valid = compute_hierarchical_fingerprints(smiles)
knn_idx, knn_dist = compute_hybrid_knn(X_mol, X_scaf, X_skel, k=15)
Z = model.fit(X_mol, knn_indices=knn_idx, knn_dists=knn_dist).transform(X_mol)
```

Both matter mainly at scale (thousands of molecules); on a few hundred points
there is little crowding to relieve. `valid` indexes the SMILES that RDKit
could parse.

## Repository structure

```text
├── nfmap/                 # Core package
├── chem_kernel/           # Hierarchical chemical kernel (used by the scale study)
├── benchmarks/paper1/     # Everything needed to reproduce the paper
│   ├── common/            # Shared data loading, metrics and IO
│   ├── data/              # ChEMBL activity sets (DRD2, EGFR, hERG)
│   ├── b*.py              # One script per experiment
│   ├── results/           # Results reported in the paper
│   ├── make_paper_tables.py
│   ├── make_paper_figures.py
│   └── Makefile
└── docs/                  # Mathematical write-ups
```

## Reproducing the paper

```bash
pip install -e .                                   # NFMap itself
pip install -r benchmarks/paper1/requirements.txt  # baselines, chemistry, plotting
cd benchmarks/paper1
make paper        # every table and figure, from the committed results
make experiments  # re-run every fit behind them (hours; use make -j)
```

All fits run on CPU with a fixed thread count, so they are reproducible run to
run; `requirements.txt` pins the versions that produced the committed results.
Every method sees an identical input matrix, with three seeds per setting, and
the baselines (t-SNE, UMAP) are swept over their own hyperparameters as NFMap
is.

## Status

Research code, under active development.

## Data

`benchmarks/paper1/data/` contains activity sets derived from ChEMBL,
redistributed under CC BY-SA 3.0 (see the README in that folder).

## Licence

MIT — see [LICENSE](LICENSE).
