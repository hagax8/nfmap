# NFMap: Bounded Normalizing Flow Maps

A PyTorch and scikit-learn compatible library for **Bounded Normalizing Flow Maps**, designed specifically for continuous dimensionality reduction, metric-preserving visualization, and interpolatable manifold projection.

`NFMap` is particularly optimized for mapping complex continuous spaces (like molecular Morgan Fingerprints or dense graph latent embeddings) into a stable, bounded 2D coordinate system without the boundary extrapolation issues ("explosions") typical of standard normalizing flows or sequence-based autoencoders.

---

## Repository Structure

The repository has been cleaned up and structured as a professional, installable Python package:

```text
├── nfmap/               # Core Python package
│   ├── __init__.py      # Package entry point (imports NFMap)
│   └── nfmap.py         # PyTorch RealNVP & Pullback geometry logic
├── benchmarks/          # Performance benchmarks & diagnostics
│   ├── nfmap_diagnostics.py
│   └── test_heldout_diagnostics.py
├── docs/                # Core mathematical write-ups & guides
├── planning_docs/       # Spec and plans for future phases (e.g., Imolgen Graph VAE)
├── experiments/         # CDDD, ChemBed, and Molecular VAE experiments
└── archive/             # Preserved raw historical research, scripts, and plots
```

---

## Installation

To install `NFMap` in editable mode:

```bash
git clone https://github.com/hagax8/nfmap.git
cd nfmap
pip install -e .
```

### Dependencies
- Python >= 3.8
- PyTorch
- Scikit-Learn
- SciPy
- NumPy
- Tqdm

---

## Quick Start

`NFMap` exposes a scikit-learn compatible API (`fit`, `transform`, `inverse_transform`):

```python
import numpy as np
from nfmap import NFMap

# Generate dummy high-dimensional data (e.g., 600 samples of 128D vectors)
X = np.random.randn(600, 128)

# Initialize the flow mapper
mapper = NFMap(
    n_components=2,          # Target dimension (2D for mapping)
    n_blocks=5,              # Number of flow coupling blocks
    hidden_dim=512,          # Hidden layers in coupling blocks
    prior_weight=0.005,      # Weight of log-likelihood prior
    verbose=True
)

# Fit the map and project points to 2D
X_2d = mapper.fit_transform(X)

# Map a new 2D coordinate back to the 128D space
Z_flow = np.array([[0.5, -0.2, 0.0, ..., 0.0]])  # Pad extra dims with 0
X_reconstructed = mapper.inverse_transform(Z_flow)
```

---

## Benchmarks & Diagnostics

To run the benchmarking suite comparing `NFMap` with UMAP, t-SNE, and GTM on standard ML manifolds (Moons, Swiss Roll, S-Curve, or Digits):

```bash
python benchmarks/nfmap_diagnostics.py --dataset digits
```
