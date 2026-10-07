from __future__ import annotations
from typing import Optional, Union
import os
import time
import warnings
import math
import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD, PCA, IncrementalPCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    from pynndescent import NNDescent
    HAS_PYNNDESCENT = True
except ImportError:
    HAS_PYNNDESCENT = False


class RealNVPBlock(nn.Module):
    """
    A single Affine Coupling Layer for the RealNVP normalizing flow.
    
    Reference:
    Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2017). 
    Density estimation using Real NVP. ICLR 2017. 
    (https://arxiv.org/abs/1605.08803)
    """
    def __init__(self, dim, hidden_dim, mask):
        super().__init__()
        self.register_buffer('mask', mask)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim * 2)
        )
        
        # Initialize last layer to zero so the flow starts as an identity mapping
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        """Forward flow mapping from z -> x."""
        x_masked = x * self.mask
        st = self.net(x_masked)
        s, t = st.chunk(2, dim=-1)
        
        # Apply changes only to the unmasked parts
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)
        
        # Constrain s for numerical stability
        s = torch.tanh(s)
        
        x_out = x * torch.exp(s) + t
        log_det = s.sum(dim=-1)
        return x_out, log_det

    def inverse(self, x_out):
        """Inverse flow mapping from x -> z."""
        x_masked = x_out * self.mask
        st = self.net(x_masked)
        s, t = st.chunk(2, dim=-1)
        
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)
        
        s = torch.tanh(s)
        
        x_in = (x_out - t) * torch.exp(-s)
        log_det = -s.sum(dim=-1)
        return x_in, log_det


class _ZPerpMDNHead(nn.Module):
    """
    Small Mixture Density Network (Bishop, 1994) predicting a distribution over
    z_perp conditioned on the 2D map coordinates (z_parallel).

    Used to fill z_perp during generate(z_perp_strategy='predicted') with values
    that reflect real, locally-typical structure instead of zeros/global noise.
    A mixture (rather than a single Gaussian) is needed because a single Gaussian
    fit to a multimodal location regresses to the average of the modes, which is
    typically the least-typical point in that neighborhood.
    """
    def __init__(self, n_components, d_perp, n_mixture=3, hidden_dim=128):
        super().__init__()
        self.n_mixture = n_mixture
        self.d_perp = d_perp
        self.net = nn.Sequential(
            nn.Linear(n_components, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.logit_head = nn.Linear(hidden_dim, n_mixture)
        self.mu_head = nn.Linear(hidden_dim, n_mixture * d_perp)
        self.log_sigma_head = nn.Linear(hidden_dim, n_mixture * d_perp)

        for m in (self.logit_head, self.mu_head, self.log_sigma_head):
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)

    def forward(self, coords):
        """coords: (B, n_components) -> log_pi (B,K), mu (B,K,D_perp), log_sigma (B,K,D_perp)."""
        h = self.net(coords)
        log_pi = torch.log_softmax(self.logit_head(h), dim=-1)
        mu = self.mu_head(h).view(-1, self.n_mixture, self.d_perp)
        log_sigma = self.log_sigma_head(h).view(-1, self.n_mixture, self.d_perp).clamp(-5.0, 2.0)
        return log_pi, mu, log_sigma


class _NFMapModule(nn.Module):
    """
    Core PyTorch Module handling the differential geometry:
    Rectangular Flows, Pullback Metric, and Canonical Loss.
    
    References to the original implementations:
    1. Brehmer & Cranmer (NeurIPS 2020) — Injective Flows & Density:
       https://github.com/johannbrehmer/manifold-flow
    2. Caterini et al. (NeurIPS 2021) — Rectangular Flows padding:
       https://github.com/layer6ai-labs/rectangular-flows
    3. Flouris & Konukoglu (NeurIPS 2023) — Canonical/Orthogonal Regularization:
       https://github.com/k-flouris/cmf
    """
    def _get_grid_means(self, n_comps, dim, layout='square', grid_bounds=3.0):
        if dim != 2:
            return torch.randn(n_comps, dim) * 2.0
            
        if layout == 'hexagonal':
            if n_comps <= 1:
                return torch.zeros(n_comps, dim)
            
            # Area of circle
            target_area = np.pi * grid_bounds**2
            area_per_point = target_area / n_comps
            # Hexagonal packing distance
            d = np.sqrt((2 * area_per_point) / np.sqrt(3))
            
            points = []
            n_steps = int(np.ceil(grid_bounds / d)) + 2
            for i in range(-n_steps, n_steps + 1):
                for j in range(-n_steps, n_steps + 1):
                    x = i * d
                    if j % 2 != 0:
                        x += d / 2.0
                    y = j * (np.sqrt(3) / 2.0) * d
                    points.append([x, y])
                    
            points = np.array(points)
            # Sort by distance from origin and take nearest n_comps
            distances = np.linalg.norm(points, axis=1)
            sorted_indices = np.argsort(distances)
            best_points = points[sorted_indices[:n_comps]]
            
            return torch.tensor(best_points, dtype=torch.float32)
            
        elif layout == 'concentric':
            if n_comps <= 1:
                return torch.zeros(n_comps, dim)
            
            R = int(np.round(np.sqrt(n_comps))) 
            if R < 1: R = 1
            
            c = (n_comps - 1) / (R * (R + 1) / 2)
            points_per_ring = [int(np.round(c * r)) for r in range(1, R + 1)]
            
            # Distribute the rounding remainder. Uses a fixed local RandomState
            # rather than the global np.random: the prior grid is otherwise a
            # purely deterministic construction, and drawing from the global
            # stream made the layout depend on whatever the caller had sampled
            # beforehand (and perturbed that stream in turn).
            diff = (n_comps - 1) - sum(points_per_ring)
            ring_rng = np.random.RandomState(0)
            for _ in range(abs(diff)):
                idx = int(ring_rng.randint(0, R))
                points_per_ring[idx] += 1 if diff > 0 else -1
                
            means = [[0.0, 0.0]]
            scale_factor = grid_bounds / R
            for r in range(1, R + 1):
                n_points = points_per_ring[r - 1]
                if n_points > 0:
                    radius = r * scale_factor
                    for i in range(n_points):
                        angle = (2.0 * np.pi * i) / n_points
                        means.append([radius * np.cos(angle), radius * np.sin(angle)])
            return torch.tensor(means, dtype=torch.float32)
            
        else: # square
            side = int(np.ceil(np.sqrt(n_comps)))
            grid_1d = torch.linspace(-grid_bounds, grid_bounds, side)
            xx, yy = torch.meshgrid(grid_1d, grid_1d, indexing='ij')
            grid = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
            if grid.shape[0] == n_comps:
                return grid
            # n_comps isn't a perfect square (side**2 != n_comps), so the
            # side x side grid has more candidates than we need. Keep the
            # n_comps nearest the origin instead of a raster-order prefix
            # (grid[:n_comps]) - the old behavior, which for e.g. n_comps=200
            # (side=15, 225 candidates) silently dropped the entire
            # x=+grid_bounds row plus part of the row before it: a real,
            # asymmetric coverage gap along one whole edge, not a considered
            # tradeoff (nothing else in this function - or its caller -
            # accounted for a missing edge). Symmetric radial truncation
            # matches the 'hexagonal' branch's own selection rule above
            # ("sort by distance from origin, take nearest n_comps"), so
            # both layouts degrade the same well-behaved way when n_comps
            # doesn't exactly fill the tiling.
            distances = torch.norm(grid, dim=1)
            nearest_indices = torch.argsort(distances)[:n_comps]
            return grid[nearest_indices]

    def __init__(self, d_scaled, n_components=2, n_blocks=5, hidden_dim=512, use_canonical=False, canonical_weight=0.05, sigma_perp=0.05, prior_weight=0.005, n_prior_components=15, learnable_priors=True, learnable_prior_weights=False, vol_weight=1.0, prop_weight=1.0, prior_layout='square', grid_bounds=3.0, prior_family='gaussian', prior_dof=2.0, prior_init_scale=None, **kwargs):
        super().__init__()
        self.d_scaled = d_scaled
        self.n_components = n_components
        self.use_canonical = use_canonical
        self.canonical_weight = canonical_weight
        self.sigma_perp = sigma_perp
        self.prior_weight = prior_weight
        self.vol_weight = vol_weight
        self.n_prior_components = n_prior_components
        self.prop_weight = prop_weight
        self.prior_layout = prior_layout
        self.prior_family = prior_family
        self.prior_dof = prior_dof
        self.use_native_predictor = kwargs.get('use_native_predictor', False)
        self.use_lmdt_head = kwargs.get('use_lmdt_head', True)
        self.use_zperp_mdn = kwargs.get('use_zperp_mdn', False)

        # Native Predictor Parameters
        if self.use_native_predictor:
            self.macro_weights = nn.Parameter(torch.zeros(n_prior_components))
            self.micro_head = nn.Linear(d_scaled - n_components, 1)
            nn.init.normal_(self.micro_head.weight, std=0.01)
            nn.init.zeros_(self.micro_head.bias)
            
        # LMDT Predictor Head
        if self.use_lmdt_head:
            self.lmdt_head = nn.Sequential(
                nn.Linear(d_scaled, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
            # Initialize with small weights
            for m in self.lmdt_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    nn.init.zeros_(m.bias)

        # z_perp Mixture Density Head (predicts z_perp distribution from 2D coords)
        if self.use_zperp_mdn:
            self.zperp_mdn = _ZPerpMDNHead(
                n_components,
                d_scaled - n_components,
                n_mixture=kwargs.get('n_mdn_components', 3),
                hidden_dim=kwargs.get('mdn_hidden_dim', 128),
            )

        self.grid_bounds = grid_bounds
        # GMM Prior Parameters
        grid_means = self._get_grid_means(n_prior_components, n_components, layout=self.prior_layout, grid_bounds=self.grid_bounds)
        # Initial per-component sigma. Historically a bare hardcoded -1.0
        # (sigma=exp(-1)=0.368) regardless of grid_bounds/n_prior_components -
        # fine as long as grid_bounds/n_prior_components stay at the values
        # that constant was implicitly tuned for, but it means widening
        # grid_bounds alone (holding n_prior_components fixed) spreads
        # components apart without giving them any more reach, silently
        # increasing the spacing-to-sigma ratio. prior_init_scale lets a
        # caller preserve that ratio explicitly when deliberately widening
        # the canvas - None keeps the exact original constant, so this is a
        # no-op for every existing caller.
        init_log_scale = -1.0 if prior_init_scale is None else float(np.log(prior_init_scale))
        if learnable_priors:
            self.prior_means = nn.Parameter(grid_means)
            self.prior_log_scales = nn.Parameter(torch.full((n_prior_components,), init_log_scale))
        else:
            self.register_buffer('prior_means', grid_means)
            self.register_buffer('prior_log_scales', torch.full((n_prior_components,), init_log_scale))
        
        # Mixing weights (pi_k = softmax(prior_logits)) over the fixed grid.
        # Classic GTM (Bishop, Svensen & Williams 1998) fixes this at uniform
        # 1/K - every latent grid node gets equal prior probability, so the
        # *only* way the model can explain the data is by warping how points
        # land across the grid, never by quietly reweighting which nodes
        # matter. This was bugged for a long time: prior_logits was always a
        # trainable nn.Parameter here regardless of learnable_priors (which
        # only ever gated prior_means/prior_log_scales above), so the
        # optimizer had a cheap escape hatch the GTM analogy doesn't allow -
        # confirmed on a real 50k-molecule ChEMBL subsample (same config as
        # production: K=200, student_t/dof=2, hexagonal) to collapse real
        # per-point responsibility mass to ~9% effective components (121/200
        # grid nodes receiving ~0 real data, 53% of points pulled toward a
        # distant heavyweight component instead of their true nearest one)
        # while a uniform-weight fit of the identical data reaches ~32%
        # effective components, 100% of the grid receiving real data, and
        # 100% of points matching their true nearest component - with no
        # measured embedding-quality cost (trustworthiness/continuity/
        # distance-correlation all within noise of the trainable variant).
        # learnable_prior_weights=True restores the old always-trainable
        # behavior for anyone who wants to experiment with it.
        if learnable_prior_weights:
            self.prior_logits = nn.Parameter(torch.zeros(n_prior_components))
        else:
            self.register_buffer('prior_logits', torch.zeros(n_prior_components))
        
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            # Alternate masking for coupling layers
            mask = torch.zeros(d_scaled)
            if i % 2 == 0:
                mask[:d_scaled // 2] = 1.0  # Block 1, 3, 5: mask first half
            else:
                mask[d_scaled // 2:] = 1.0  # Block 2, 4, 6: mask second  half
            self.blocks.append(RealNVPBlock(d_scaled, hidden_dim, mask))

    def forward_flow(self, z):
        """Map from latent z (padded) to data space x."""
        log_det_total = 0
        x = z
        for block in self.blocks:
            x, log_det = block(x)
            log_det_total += log_det
        return x, log_det_total

    def inverse_flow(self, x):
        """Retro-projection from data space x to latent z."""
        log_det_total = 0
        z = x
        for block in reversed(self.blocks):
            z, log_det = block.inverse(z)
            log_det_total += log_det
        return z, log_det_total

    def _prior_component_log_density(self, diff_sq, log_scales):
        """
        Per-component log density for the GMM prior on the 2D map, given the
        squared (scaled) distance to each component `diff_sq` and that
        component's `log_scales` - both already isotropic-Mahalanobis in the
        caller. Two interchangeable families, selected by `self.prior_family`:

        - 'gaussian' (default): standard isotropic Normal per component.
        - 'student_t': isotropic multivariate Student-t per component, with
          shared degrees-of-freedom `self.prior_dof`. Heavier tails than the
          Gaussian mean the restoring force pulling a point toward a component
          (∝ (ν+d)/(ν+diff_sq)) *decays* at large diff_sq instead of growing
          linearly - so points sitting ambiguously between two prior modes get
          a softer, more graded assignment rather than being snapped hard
          toward whichever mode's density edges ahead. This is the intended
          lever against the "same scaffold split across distant map regions"
          symptom of the crowding problem; nu -> inf recovers the Gaussian
          exactly. Off by default - purely additive, zero effect unless
          prior_family='student_t' is explicitly requested.
        """
        d = self.n_components
        if self.prior_family == 'student_t':
            nu = self.prior_dof
            # nu/d are fixed Python floats (a hyperparameter, not a learned
            # tensor), so math.lgamma folds them into one plain-float
            # constant here rather than torch.lgamma(torch.tensor(...)) -
            # equivalent, but avoids an ONNX-export TracerWarning ("torch.tensor
            # results are registered as constants in the trace") from
            # generate_navifold_drd2.py's export_projector.
            log_gamma_const = math.lgamma((nu + d) / 2.0) - math.lgamma(nu / 2.0) - 0.5 * d * np.log(nu * np.pi)
            return (log_gamma_const
                    - d * log_scales
                    - 0.5 * (nu + d) * torch.log1p(diff_sq / nu))
        elif self.prior_family == 'gaussian':
            return (-0.5 * diff_sq
                    - d * log_scales
                    - 0.5 * d * np.log(2 * np.pi))
        else:
            raise ValueError(f"Unknown prior_family {self.prior_family!r}; expected 'gaussian' or 'student_t'.")

    def compute_loss(self, x, sigma_perp=None, y_props=None):
        """
        Computes the total loss for training the NFMap network by combining up to 5 penalty terms.
        
        [From Literature: Manifold Normalizing Flows (Brehmer et al., 2020)]
        1. loss_prior: Forces the 2D map coordinates (z_parallel) to cluster tightly around the GMM bells.
        2. loss_manifold: Penalizes the off-manifold distance (z_perp). Forces molecules to lie flat on the 2D map.
        3. loss_vol: A log-determinant volume penalty (standard in flows), BUT modified here with a novel 
           dimension-scaling factor (n_components / d_scaled) to prevent the 126 perpendicular dimensions 
           from exploding to infinity to cheat the loss function.
        
        [Novel NFMap Contributions]
        4. loss_prop: (Optional) Trains the network to predict chemical properties natively from the 2D map.
        5. loss_ortho/diag: (Optional) Conformal regularization (J^T J) to preserve distances/angles (isometric mapping).
        """
        z, log_det_inv = self.inverse_flow(x)

        z_parallel = z[:, :self.n_components]
        z_perp     = z[:, self.n_components:]

        # ── GMM Prior ────────────────────────────────────────────────────────
        # We model the base distribution of the 2D map as a Gaussian Mixture Model (GMM).
        # This allows the map to have natural clusters rather than a single dense blob.
        z_expanded  = z_parallel.unsqueeze(1) # Shape: [Batch, 1, 2]
        means       = self.prior_means.unsqueeze(0) # Shape: [1, K_components, 2]
        log_scales  = self.prior_log_scales.unsqueeze(0)
        
        # 1. Calculate squared distance from each molecule to each of the K cluster centers
        diff_sq     = torch.sum(
            ((z_expanded - means) / torch.exp(log_scales).unsqueeze(-1)) ** 2,
            dim=-1)
            
        # 2. Calculate the per-component log density (Gaussian, or Student-t if
        #    prior_family='student_t' - see _prior_component_log_density)
        log_N       = self._prior_component_log_density(diff_sq, log_scales)

        # 3. Multiply by the "weight" (pi) of each cluster, and sum them up (logsumexp for stability)
        log_pi      = torch.log_softmax(self.prior_logits, dim=0).unsqueeze(0)
        log_p_gmm   = torch.logsumexp(log_pi + log_N, dim=1)
        
        # We want to MAXIMIZE probability, which means MINIMIZING negative probability
        loss_prior  = -self.prior_weight * log_p_gmm

        # ── Manifold alignment ───────────────────────────────────────────────
        # Penalizes off-manifold dimensions (z_perp). 
        # Unlike standard Brehmer flows which drive sigma_perp -> 0 to create a razor-thin manifold,
        # we often use a large sigma_perp (e.g. 0.5, 5, or 10) to create a "thick" manifold. 
        # (loss_manifold itself is computed further down, once z_perp_remaining
        # is known - any property dimensions consumed by the legacy 3rd-axis
        # mapping below are excluded from the penalty. An earlier copy of that
        # computation sat here and was unconditionally overwritten.)
        sigma_perp = sigma_perp if sigma_perp is not None else self.sigma_perp

        z_perp_flow = z[:, self.n_components:self.d_scaled]
        
        loss_prop = 0.0
        if y_props is not None:
            if getattr(self, 'use_native_predictor', False):
                # ── Native Macro/Micro Property Predictor ────────────────────────
                # This predicts chemical properties (like binding affinity) directly from the map.
                # 'y_macro' predicts the property based purely on which cluster the molecule is in (global property)
                p_k = torch.softmax(log_pi + log_N, dim=1)
                y_macro = torch.sum(p_k * self.macro_weights.unsqueeze(0), dim=1, keepdim=True)
                
                # 'y_micro' predicts fine-grained details using the perpendicular dimensions (local property)
                y_micro = self.micro_head(z_perp_flow)
                y_pred = y_macro + y_micro
                
                # Standard Mean Squared Error (MSE) to train the property predictor
                loss_prop = torch.nn.functional.mse_loss(y_pred, y_props) * self.prop_weight
                # L2 penalty on micro_head weights
                loss_prop += 0.1 * torch.sum(self.micro_head.weight ** 2)
                z_perp_remaining = z_perp_flow
            else:
                # ── Legacy 3rd-Axis Orthogonal Property Mapping ──────────────────
                # Instead of a neural network, this forces the very first dimension of z_perp
                # to literally BE the property value (e.g. z_perp[0] = LogP).
                if z_perp_flow.shape[1] > 0:
                    K = y_props.shape[1]
                    if K > z_perp_flow.shape[1]:
                        raise ValueError("Too many properties for available perpendicular dimensions.")
                    z_props = z_perp_flow[:, :K]
                    loss_prop = torch.nn.functional.mse_loss(z_props, y_props) * self.prop_weight
                    z_perp_remaining = z_perp_flow[:, K:]
                else:
                    z_perp_remaining = z_perp_flow
        else:
            z_perp_remaining = z_perp_flow
            
        if z_perp_remaining.shape[1] > 0:
            # L2 penalty pushing the perpendicular dimensions to zero.
            # This mathematically forces the intrinsic manifold to lie perfectly flat 
            # at z_perp = 0, with a variance of sigma_perp.
            loss_manifold = 0.5 / (sigma_perp ** 2) * torch.mean(z_perp_remaining ** 2, dim=-1)
        else:
            loss_manifold = 0.0

        # ── Volume term: full-flow log-det scaled to manifold dims ───────────
        # In maximum likelihood training, the flow maximizes the log-determinant of the Jacobian 
        # (i.e., it gets a "reward" for stretching the space).
        # In standard flows, this stretching is balanced by the prior penalizing points far from the origin.
        # However, because we only care about a 2D intrinsic manifold (n_components=2), the remaining 
        # perpendicular dimensions (e.g., 126 dims) are loosely constrained by `sigma_perp` (our Thick Manifold).
        # If left unscaled, the network will "cheat" by aggressively stretching those 126 perpendicular 
        # dimensions to infinity to artificially inflate the log-det reward, causing gradients to explode.
        # We scale the log-det down by (2 / D) to prevent these "garbage" dimensions from overpowering the loss.
        loss_vol = -log_det_inv * (self.n_components / self.d_scaled) * self.vol_weight

        # Combine all standard losses into a single scalar that the optimizer will minimize
        loss = (loss_prior + loss_manifold + loss_vol + loss_prop).mean()

        # ── Optional conformal regularization ────────────────────────────────
        # This enforces that the map is "isometric" (preserves distances and angles perfectly).
        # It calculates the pullback metric tensor G = J^T J and forces it to be the Identity matrix.
        # To use with toy data, not complex high-dim stuff
        if self.use_canonical:
            z_para_req  = z_parallel.detach().requires_grad_(True)
            z_perp_zero = torch.zeros_like(z_perp)
            z_padded    = torch.cat([z_para_req, z_perp_zero], dim=-1)
            x_gen, _    = self.forward_flow(z_padded)
            
            # 1. Jacobian-Vector Product (JVP) trick to efficiently get the Jacobian J
            u    = torch.ones_like(x_gen, requires_grad=True)
            g    = torch.autograd.grad(torch.sum(x_gen * u), z_para_req,
                                       create_graph=True)[0]
            J_cols = []
            for i in range(self.n_components):
                # Compute each column of the Jacobian matrix
                J_cols.append(torch.autograd.grad(
                    torch.sum(g[:, i]), u,
                    create_graph=True, retain_graph=True)[0])
            J = torch.stack(J_cols, dim=2)
            
            # 2. Compute the Pullback Metric Tensor: G = J^T J
            G = torch.bmm(J.transpose(1, 2), J)
            
            # 3. Force G to equal the Identity Matrix (Orthogonal and Unit scale)
            eye      = torch.eye(self.n_components, device=G.device).unsqueeze(0)
            off_diag = 1.0 - eye
            loss_ortho = (torch.abs(G) * off_diag).sum(dim=(1, 2)).mean() # Force off-diagonals to 0
            diag_G     = torch.diagonal(G, dim1=-2, dim2=-1)
            loss_diag  = torch.abs(diag_G - 1.0).mean()                   # Force diagonals to 1
            loss      += self.canonical_weight * loss_ortho + self.canonical_weight * loss_diag

        return loss, z_parallel

    def _prior_component_log_joint(self, z_parallel):
        """
        log(pi_k) + log N_k(z_parallel) for every mixture component k -> [B, K].

        The GMM prior on the 2D map, evaluated component-by-component. Factored
        out because exact_density / full_log_density / compute_responsibilities
        all need exactly this and had drifted into three copies of it.
        """
        z_expanded = z_parallel.unsqueeze(1)
        means = self.prior_means.unsqueeze(0)
        log_scales = self.prior_log_scales.unsqueeze(0)

        diff_sq = torch.sum(
            ((z_expanded - means) / torch.exp(log_scales).unsqueeze(-1)) ** 2,
            dim=-1)
        log_N = self._prior_component_log_density(diff_sq, log_scales)
        log_pi = torch.log_softmax(self.prior_logits, dim=0).unsqueeze(0)
        return log_pi + log_N

    def full_log_density(self, x, z_perp_log_scale=None):
        """
        Exact log density over the FULL ambient (scaled) space R^{d_scaled}.

        exact_density() below answers "how typical is this point's position *on*
        the 2D manifold", and is by construction blind to how far the point lies
        *off* that manifold - two points with the same map coordinate score
        identically no matter how off-manifold one of them is. That makes it the
        wrong signal for out-of-distribution detection, where "is this point
        anywhere near the data at all" is the whole question.

        This method scores the generative model the training loss actually
        implies, which has both halves:

            z_parallel ~ GMM prior            (where on the map)
            z_perp     ~ N(0, diag(s^2))      (how far off the manifold)
            x          = forward_flow(z)

        giving, by the standard change-of-variables formula for the bijection
        f = inverse_flow,

            log p(x) = log p_GMM(z_par) + log N(z_perp; 0, diag(s^2))
                       + log|det df/dx|

        which is a properly normalised density on R^{d_scaled} (it integrates to
        1), so it can be thresholded and compared across points.

        `z_perp_log_scale` is the per-dimension log std of z_perp, calibrated on
        the training data by NFMap.fit(). Falling back to self.sigma_perp (the
        training hyperparameter) when it is absent keeps older pickled models
        working, but that fallback is only a rough scale - a model fitted with
        the current code supplies the calibrated vector.
        """
        z, log_det_inv = self.inverse_flow(x)
        z_parallel = z[:, :self.n_components]
        z_perp = z[:, self.n_components:]

        # ── where on the map (2D GMM prior) ──────────────────────────────────
        log_p_par = torch.logsumexp(self._prior_component_log_joint(z_parallel), dim=1)

        # ── how far off the manifold (Gaussian on z_perp) ────────────────────
        if z_perp.shape[1] == 0:
            log_p_perp = torch.zeros_like(log_p_par)
        else:
            if z_perp_log_scale is None:
                log_sigma = torch.full(
                    (z_perp.shape[1],), float(np.log(self.sigma_perp)),
                    dtype=z_perp.dtype, device=z_perp.device)
            else:
                log_sigma = z_perp_log_scale.to(dtype=z_perp.dtype, device=z_perp.device)
            log_p_perp = (-0.5 * (z_perp / torch.exp(log_sigma)) ** 2
                          - log_sigma
                          - 0.5 * np.log(2 * np.pi)).sum(dim=-1)

        # ── volume change of the bijection ───────────────────────────────────
        return log_p_par + log_p_perp + log_det_inv

    def exact_density(self, x):
        """
        Calculate the exact probability density of data points on the map (log likelihood).
        
        This is the Normalizing Flow equivalent of the GTM log-likelihood 
        (cf. Bishop, Svensén, and Williams, 1998). While GTM calculates the data likelihood using 
        an RBF mapping with isotropic Gaussian noise, NFMap calculates it 
        using a deterministic bijection (the Flow) and the Change of Variables 
        formula from differential geometry.
        
        When you map a flat 2D plane into a complex d-dimensional space, the 2D plane gets
        stretched and distorted. If a 1x1 box on the map gets stretched into a 10x10
        box in the d-dimensional space, the "density" of molecules in that box drops (magnification).
        
        Formula (cf. Brehmer & Cranmer, "Flows for simultaneous manifold learning and density estimation", 2020):
        ln p_X(x) = ln p_Z(z) - ln(magnification_factor)
        ln p_X(x) = ln p_Z(z) - 1/2 * ln(det(G))
        
        Where:
        - p_Z(z) is the baseline density on the 2D map (modeled by our GMM).
        - G is the Pullback Metric Tensor (measuring the stretching/distortion).
        - sqrt(det(G)) is the exact volume magnification factor.
        """
        z, _ = self.inverse_flow(x)
        z_parallel = z[:, :self.n_components]
        
        # ── 1. Calculate the Magnification Factor (Volume Change) ────────────
        # To measure distortion, we enforce the mathematical manifold where z_perp = 0.
        z_para_req = z_parallel.detach().requires_grad_(True)
        z_perp_zero = torch.zeros((x.shape[0], self.d_scaled - self.n_components), device=x.device)
        z_padded = torch.cat([z_para_req, z_perp_zero], dim=-1)
        x_gen, _ = self.forward_flow(z_padded)
        
        # Efficiently compute the Jacobian (J) using the double-autograd JVP trick
        u = torch.ones_like(x_gen, requires_grad=True)
        dot1 = torch.sum(x_gen * u)
        g = torch.autograd.grad(dot1, z_para_req, create_graph=True)[0]
        
        J_cols = []
        for i in range(self.n_components):
            retain = True if i < self.n_components - 1 else False
            J_col = torch.autograd.grad(
                torch.sum(g[:, i]), 
                u, 
                create_graph=False, 
                retain_graph=retain
            )[0]
            J_cols.append(J_col)
            
        # J is the Jacobian matrix (maps small changes in 2D to changes in d-dim space)
        J = torch.stack(J_cols, dim=2) # [B, D, n_components]
        
        # The magnification factor (change in volume) is sqrt(det(G)), G = J^T J.
        # Computed through the singular values of J rather than det(J^T J):
        # the two are identical in exact arithmetic (det(G) = prod(sigma_i^2), so
        # 0.5*log det G = sum(log sigma_i)), but forming G first squares the
        # condition number and, for n_components=2, reduces to the difference
        # |J1|^2|J2|^2 - (J1.J2)^2, which cancels catastrophically in float32
        # exactly where the map degenerates (nearly collinear Jacobian columns).
        # Measured on a synthetic near-degenerate J: the det route silently
        # overestimated log p by up to 3.9 nats, and a uniformly contracted map
        # (sigma < 1e-4) hit the old min=1e-12 clamp and was capped without
        # warning. The singular-value route was exact to float32 precision in
        # every case, and needs no clamp.
        # MPS ships no SVD kernel (aten::_linalg_svd), and J here is a stack of
        # tiny D x n_components matrices, so the host round-trip is negligible.
        J_sv = J.cpu() if J.device.type == 'mps' else J
        log_sqrt_det_G = torch.log(
            torch.clamp(torch.linalg.svdvals(J_sv), min=1e-20)
        ).sum(dim=-1).to(J.device)
        
        # ── 2. Calculate the Baseline 2D Density (GMM Prior) ─────────────────
        log_p_z = torch.logsumexp(self._prior_component_log_joint(z_parallel), dim=1)
        
        # ── 3. Final Exact Density Calculation ────────────────────────────────
        # log p = log p_Z - log(magnification_factor)
        # Note: log(sqrt(det(G))) is mathematically identical to 0.5 * log(det(G))
        # because log(det(G)^(1/2)) = 0.5 * log(det(G)).
        # Eq. 14, Brehmer & Cranmer (2020): log p_M(x) = log p(u) - ½ log det(JᵀJ)
        log_p = log_p_z - log_sqrt_det_G
        return log_p

    def compute_responsibilities(self, x):
        """
        Calculate the exact GMM responsibilities (cluster probabilities) for each point.
        Returns a tensor of shape [Batch, n_prior_components] where each row sums to 1.
        """
        z, _ = self.inverse_flow(x)
        z_parallel = z[:, :self.n_components]

        log_unnormalized_probs = self._prior_component_log_joint(z_parallel)
        log_p_z = torch.logsumexp(log_unnormalized_probs, dim=1, keepdim=True)
        
        log_responsibilities = log_unnormalized_probs - log_p_z
        return torch.exp(log_responsibilities)


class NFMap(BaseEstimator, TransformerMixin):
    """
    NFMap: Normalizing Flow Manifold Mapper.

    A scikit-learn compatible dimensionality reduction model that combines
    Normalizing Flows (RealNVP) with topological graph regularization (the
    Fuzzy Simplicial Kernel) to learn a probabilistically grounded low-dimensional
    map of complex data manifolds.

    Conceptually related to GTM (Bishop, Svensén & Williams, 1998), but replaces
    the RBF network with a deep bijective flow and the EM update rule with gradient
    descent. `score_samples` gives the data log-likelihood, either over the full
    input space (the default, usable as an out-of-distribution score) or on the
    learned manifold only, via the pullback metric (Brehmer & Cranmer, 2020).

    Parameters
    ----------
    n_components : int, default=2
        Dimensionality of the learned map. Typically 2 for 2D visualization.
        Higher values may improve density estimation at the cost of interpretability.

    complexity : {'low', 'medium', 'high'}, default='medium'
        Controls the depth and width of the underlying RealNVP flow network.
        - 'low':    4 coupling blocks, smaller hidden layers, 60 max epochs.
                    Fast training, suitable for simple manifolds or quick prototyping.
        - 'medium': 6 coupling blocks, medium hidden layers, 80 max epochs.
                    Balanced. Good default for most datasets.
        - 'high':   10 coupling blocks, larger hidden layers, 100 max epochs.
                    Slower, but better for complex, high-dimensional datasets.
        Note: max_epochs set by complexity are overridden by the `max_epochs` parameter.
        Validated in `fit()`; any other value raises ValueError rather than
        being silently treated as 'medium'.

    use_canonical : bool, default=False
        If True, enforces an orthogonality constraint on the Jacobian of the flow
        so that the latent axes are decorrelated. Adds training overhead but can
        improve interpretability of the map axes. Experimental.

    canonical_weight : float, default=0.05
        Weight of the conformal/isometric regularization loss (the G=J^T J -> I
        constraint from `use_canonical`). Ignored unless `use_canonical=True`.
        0.05 was the original hardcoded value before this was exposed as a
        tunable parameter - kept as the default so existing behavior is
        unchanged. At that weight the effect on distance preservation is small
        relative to `graph_weight`'s default of 5.0; raise this (e.g. 0.5-5.0)
        to see a more pronounced isometric effect, at the cost of fighting the
        graph attraction/repulsion terms more.

    prior_weight : float, default=0.1
        Weight of the GMM prior loss term (negative log-likelihood on the 2D map).
        Controls how strongly points are attracted to the GMM cluster centers.
        - Higher values: forces points onto the GMM grid, preventing the flow from
          collapsing or flying off to infinity. Acts as a regularizer.
        - Lower values: looser coupling to the map prior, more freedom for the flow.
        Typical range: 0.005 (loose) to 0.5 (tight).

    graph_weight : float, default=5.0
        Weight of the graph attraction loss term (Fuzzy Simplicial Kernel).
        Controls how strongly neighboring points are attracted together in the
        2D map, producing UMAP-style cluster compactness.
        - Higher values: tighter clusters, stronger local structure preservation.
        - Lower values: looser clusters, potentially better global structure.
        Typical range: 1.0 (gentle) to 10.0 (aggressive).

    repulsion_weight : float, default=1.0
        Weight of the negative-sampling repulsion loss term.
        Controls how strongly non-neighboring points are pushed apart.
        Works together with `negative_sample_rate` to create inter-cluster whitespace.
        Typical range: 0.5 to 2.0.

    n_prior_components : int, default=100
        Number of Gaussian components in the GMM prior on the 2D map.
        This is analogous to the number of grid points in GTM.
        - Higher values: finer-grained, higher-resolution density representation.
          Better for large, complex datasets with many sub-populations.
        - Lower values: coarser representation, faster training.
        Typical range: 15 (coarse) to 200 (fine).

    grid_bounds : float or 'auto', default='auto'
        Half-width of the square/disk the GMM prior grid is laid out in
        (e.g. 3.0 -> roughly [-3, 3]). 'auto' scales with dataset size
        (N=100 -> 3.0, N=10,000 -> ~3.3, N=100,000 -> ~3.6) - see the
        heuristic in `fit()`. When `prior_family='student_t'` AND this is
        left at 'auto', the resolved value is additionally doubled, since
        student_t's weaker far-field attraction needs a wider canvas to
        avoid the coverage gap described under `prior_family` - this 2x
        factor is the ratio validated on one tested scale (~10k-molecule
        EGFR/hERG maps), applied as the most honest available extrapolation
        rather than something re-validated at every N. Passing an EXPLICIT
        numeric value bypasses this entirely (used exactly as given,
        regardless of `prior_family`) - the doubling only happens as part
        of 'auto' resolution, so pair a manually chosen grid_bounds with a
        manually chosen `prior_init_scale` too (see that parameter) rather
        than relying on auto-widening to also kick in.

    random_state : int, default=42
        Random seed for reproducibility. Controls weight initialization,
        PCA preprocessing, and any stochastic operations.

    learnable_priors : bool, default=False
        If True, the GMM cluster centers (prior means) are trainable parameters
        that drift during gradient descent. Can lead to cluster collapse where all
        centers converge to the same point.
        If False (recommended), cluster centers are initialized on a fixed grid
        and held constant during training, providing a stable reference frame
        analogous to the fixed grid in GTM.

    learnable_prior_weights : bool, default=False
        If True, the GMM prior's mixing weights (pi_k, one per grid component)
        are trainable and free to concentrate however gradient descent likes.
        If False (recommended, and the GTM-faithful choice - classic GTM
        fixes this at uniform 1/K), the mixing weights are held uniform and
        never trained, so the fixed grid's components stay equally weighted
        throughout - only the flow itself can decide how densely points
        cluster near any given component. True reproduces this library's old
        (unintentional) behavior, where prior_logits was always trainable
        regardless of learnable_priors; on real ChEMBL data that collapses
        most of the grid to near-zero responsibility (see nfmap.py's
        _NFMapModule.__init__ for the measured numbers), which is the main
        reason False is now the default.

    vol_weight : float, default=1.0
        Weight of the volume-preserving (Jacobian scaling) loss term.
        This term penalizes the flow for aggressively stretching the "garbage"
        perpendicular dimensions to maximize log-det rewards, which would
        otherwise bypass the topological learning objective.
        Keeping this at 1.0 is strongly recommended. See paper notes on
        Jacobian explosion for technical details.

    sigma_perp : float, default=10.0
        Scale of the quadratic penalty on the off-manifold (perpendicular)
        dimensions, `0.5 / sigma_perp**2 * mean(z_perp**2)`. Controls the
        "thickness" of the learned manifold. Note this is a penalty scale, not
        additive noise - nothing is sampled or added to z_perp (the parameter
        that does add noise, to the inputs, is `noise_std` on `fit()`).
        - Lower values (e.g. 0.05): thinner manifold, the flow must also model
          fine-grained off-manifold structure. Can lead to unstable training.
        - Higher values: broader manifold, the flow spends its capacity on the
          2D layout instead.
        Map quality is NOT monotone in this parameter - it rises steeply, then
        peaks, then declines. Measured on digits (3 seeds, 120 epochs), mean
        trustworthiness by sigma_perp: 0.05 -> 0.837, 0.2 -> 0.896, 1 -> 0.922,
        2 -> 0.927, 5 -> 0.933, 10 -> 0.931, 20 -> 0.924, 50 -> 0.915. The
        default sits at the top of that plateau. Typical range: 1.0 to 20.0 for
        visualization tasks; the optimum on other data/architectures has not
        been swept and may sit elsewhere.
        As a rule of thumb sigma_perp is approximately the resulting
        per-dimension std of z_perp while the penalty still binds (sigma_perp
        <~ 2: measured ratios 1.3x, 1.3x, 1.3x, 1.2x at 0.05/0.2/1/2). Above
        that the volume and graph terms start to dominate and the realised
        thickness grows sub-linearly (ratio 0.8x at 5, 0.5x at 10, 0.3x at 20),
        so read the fitted `z_perp_log_scale_` rather than assuming sigma_perp.

    max_epochs : int, default=1000
        Maximum number of training epochs. If `patience` is set, training may
        stop early if no improvement is observed.
        - 1000 is a solid default for most datasets.
        - For quick prototyping, 200-300 is usually sufficient.

    patience : int or None, default=100
        Number of epochs to wait for loss improvement before triggering early stopping.
        The monitored quantity is the *training* loss - there is no validation
        split - so this is a convergence criterion, not an overfitting guard,
        and the restored `best_state` is the lowest-training-loss epoch. Note
        that with `early_exaggeration=True` the objective itself changes at
        epoch `max_epochs // 3` (the attraction term stops being multiplied by
        4), so losses either side of that boundary are not comparable and the
        pre-boundary best may stand unbeatable for a while afterwards.
        If None, trains for exactly `max_epochs` epochs with no early stopping.
        Setting this to ~10% of `max_epochs` is a good rule of thumb.

    verbose : bool, default=False
        If True, prints a loss/LR progress line every 100 epochs during training.
        Useful for monitoring convergence.

    negative_sample_rate : int, default=5
        Number of negative (non-neighbor) points sampled per batch point each
        training step. This sets how many pairs go INTO the repulsion term's
        mean, so it controls the PRECISION of that estimate, not its strength.
        UMAP sums over its negative samples, which makes the rate a force knob
        there; NFMap averages, so the expected repulsive force is invariant to
        this value - measured on a fixed batch, loss_repel sits at 0.117 for
        every rate from 1 to 50 while the gradient's standard deviation falls
        from 0.036 to 0.002. Use `repulsion_weight` to change how hard points
        are pushed apart.
        Raising this does still improve the map, by reducing gradient noise
        rather than by adding force, and it saturates instead of
        over-separating (digits, 2 seeds: trustworthiness 0.877 / 0.916 /
        0.931 / 0.931 at rates 1 / 5 / 20 / 50, map spread flat at 2.13
        beyond 20). Contrast `repulsion_weight`, which keeps going and does
        eventually fragment the layout.
        - 0: repulsion skipped entirely (identical to `repulsion_weight=0`,
          bit-for-bit - the whole block is guarded, so neither draws any
          random numbers).
        - 1: noisy repulsion estimate, visibly worse layout.
        - 5: the default.
        - 20: at the plateau - measurably better than 5 for a little more
          compute, with nothing gained past it.

    early_exaggeration : bool, default=False
        If True, multiplies the `graph_weight` by 4x during the first third of
        training (similar to the early exaggeration phase in t-SNE). This forces
        the initial layout into well-separated clusters before fine-tuning.
        In practice, using a sustained `graph_weight=5.0` with `negative_sample_rate=5`
        achieves similar results without the phase transition instability.

    prop_weight : float, default=1.0
        Weight of the property prediction loss term. Only active when `y_props`
        is passed to `fit()` or when `use_native_predictor=True`.
        Controls how strongly the map is influenced by the target property labels
        during training, balancing topological fidelity vs. property alignment.

    graph_kernel : {'fuzzy_simplicial', 'perplexity'}, default='fuzzy_simplicial'
        The graph construction method used to identify neighbors for the
        graph regularization loss.
        - 'fuzzy_simplicial' (recommended): UMAP-style fuzzy graph, with a
          per-point adaptive bandwidth - better for varying local densities.
          UMAP-*style*, not UMAP-exact: the bandwidth is the heuristic
          `mean(d - rho)` rather than UMAP's binary search for
          `sum_j w_ij = log2(k)`, and the resulting graph is left directed
          (no `w + w^T - w*w^T` symmetrisation). On digits this gives a
          per-point weight sum of ~6.2 against UMAP's target of 3.91.
        - 'perplexity': t-SNE-style Gaussian kernel with fixed perplexity.
          Deprecated; will be removed in a future release.

    perplexity : float, default=30
        Perplexity for the t-SNE-style graph kernel. Only used when
        `graph_kernel='perplexity'`. Ignored for the default 'fuzzy_simplicial' kernel.
        Typical range: 5 to 50.

    use_native_predictor : bool, default=False
        If True, activates a dual-head property predictor (Macro + Micro heads)
        trained jointly with the flow. Enables `predict_activity()` after fitting.
        Requires `y_props` to be passed to `fit()`.
        - Macro head: predicts from GMM cluster responsibilities (coarse, map-level).
        - Micro head: predicts from off-manifold dimensions (fine, molecular-level).

    inject_weight : float, default=0.0
        Weight of the local injectivity loss — a regularizer that prevents the 2D map
        from projecting structurally different molecules to the same region.

        **Motivation**: The flow is bijective over the full (z_par, z_perp) space, but NOT
        over z_par alone — two very different molecules can share the same 2D coordinates
        with different perpendicular dimensions. This loss enforces that molecules which
        land close together in the 2D map are also close in the original input space.
        Particularly useful when `generate()` will be used (Paper 2 / Imolgen workflow).

        **Method**: Gaussian kernel with automatic median-heuristic bandwidth. For each
        batch, the kernel width τ is set to the median pairwise 2D distance, so no manual
        bandwidth tuning is required. The loss is:

            L_inject = Σ_{(i,j)} exp(−‖z_i − z_j‖² / τ²) · ‖x_i − x_j‖

        - Disabled by default (`inject_weight=0.0`) — has zero effect on existing results.
        - Typical values if enabled: 0.1 – 1.0. Start at 0.1 and increase if needed.
        - Note: this loss is **complementary** to `graph_weight` (which attracts kNN
          neighbors together); `inject_weight` prevents dissimilar molecules from
          collapsing to the same point.

    max_ambient_dim : int or None, default=None
        Maximum dimensionality of the ambient space passed to the flow network
        after PCA preprocessing. If None, the heuristic default is used
        (128 for binary data, 256 for continuous data). Set this to override
        the heuristic, e.g. `max_ambient_dim=512` for very high-dimensional
        Transformer embeddings.

    prior_family : {'gaussian', 'student_t'}, default='gaussian'
        Distribution family for each component of the GMM prior on the 2D map.
        - 'gaussian' (default): standard behavior, unchanged from prior releases.
        - 'student_t': isotropic multivariate Student-t per component (shared
          `prior_dof` degrees of freedom). Heavier tails mean the attraction
          force toward a prior mode decays at large distance instead of growing
          without bound, so points sitting ambiguously between two modes get a
          softer, more graded pull rather than being snapped hard toward
          whichever mode's density edges ahead. Intended as an additional,
          opt-in lever against crowding/mode-splitting, on top of (not a
          replacement for) kNN-graph-level fixes like a hybrid similarity
          kernel. `prior_dof -> inf` recovers 'gaussian' exactly.

        Choosing 'student_t' also changes what `grid_bounds='auto'` and
        `prior_init_scale=None` resolve to (see those two parameters) - the
        heavier tail spreads trained points out further at a fixed sigma
        (~45% larger spatial extent measured on real target maps at
        `prior_dof=5`), so both auto-scale up together to preserve coverage
        of that wider spread rather than silently leaving components sized
        for 'gaussian's tighter one.

    prior_dof : float, default=2.0
        Degrees of freedom (nu) for the Student-t prior components. Only used
        when `prior_family='student_t'`; ignored otherwise. Lower values (e.g.
        2-3) give heavier tails / softer mode competition; higher values
        (e.g. 30+) approach Gaussian behavior. The default of 2.0, combined
        with `grid_bounds`/`prior_init_scale` left at 'auto'/None, was the
        best-performing combination found across a dof sweep (2, 3, 5, 10,
        20, 30) on two independent chemistry benchmarks (EGFR, hERG) -
        confirmed to both improve scaffold-purity metrics over 'gaussian' AND
        keep the resulting map's activity-landscape coverage intact (no
        widening of grid_bounds/sigma to match, dof=2 alone regressed
        coverage badly - beyond-fade fraction 42% vs 'gaussian's 0%). Not
        yet validated at very different dataset scales (e.g. millions of
        molecules) - see `n_prior_components`'s own scaling limits.

    prior_init_scale : float or None, default=None
        Initial sigma (before any `learnable_priors` adaptation) for every
        GMM prior component. None means: for `prior_family='gaussian'`, keep
        the historical fixed value (sigma=exp(-1)~=0.368, independent of
        grid_bounds/n_prior_components - unchanged from prior releases); for
        `prior_family='student_t'`, auto-derive it from the resolved
        `grid_bounds`/`n_prior_components` so widening the canvas (see
        `grid_bounds`) doesn't just spread the same small fixed-reach
        components further apart without giving them any more reach to
        match - the auto value preserves the exact sigma-to-component-
        spacing ratio 'gaussian's own historical default has, just applied
        to student_t's wider canvas. Set explicitly to override either
        family's default (e.g. `prior_init_scale=1.5` for a manually chosen
        wider or narrower value than either auto-behavior would pick).

    Attributes
    ----------
    resolved_prior_init_scale_ : float or None
        The actual sigma used, after resolving `prior_init_scale='auto'`-
        style defaulting (see that parameter). None only when
        `prior_family='gaussian'` and `prior_init_scale` was left at its
        default (the historical fixed sigma is applied inside the flow
        module directly rather than being resolved here).

    Attributes
    ----------
    module_ : torch.nn.Module
        The underlying trained RealNVP flow network.
    pca_ : sklearn.decomposition.TruncatedSVD or PCA
        The fitted PCA/SVD object used for dimensionality reduction pre-processing.
    scaler_pre_ : sklearn.preprocessing.StandardScaler
        The fitted StandardScaler used for pre-processing continuous data.
    is_binary_ : bool
        Whether the input data was detected as binary (e.g. Morgan fingerprints).
    global_scale_ : float
        The global scale factor applied after PCA to stabilize flow training.
    device_ : torch.device
        The device (CPU or GPU) on which the model is trained.
    z_perp_log_scale_ : ndarray of shape (D_scaled - n_components,)
        Per-dimension log std of the off-manifold coordinates z_perp,
        measured on the training data after fitting. This is the scale of
        the Gaussian `score_samples(mode='full')` puts on z_perp, and it is
        measured rather than taken from `sigma_perp` because `sigma_perp`
        is a training penalty weight, not a fitted scale.
    pca_residual_dim_ : int
    pca_residual_var_ : float
        How many directions the PCA/SVD step discarded, and their mean
        per-direction variance on the training data. Together they form the
        term in `score_samples(mode='full')` that notices structure the
        linear bottleneck could not represent at all. Both are 0 / unused
        when PCA kept every direction.
    loss_history_ : list of float
        Total training loss after each epoch (the same quantity the
        `verbose=True` progress line prints every `max_epochs // 10`
        epochs, kept for every epoch instead). Useful for checking whether
        a run was still improving when early stopping (`patience`) or
        `max_epochs` cut it off, rather than judging convergence from the
        final loss value alone. Preserved across a `checkpoint_path` resume.

    References
    ----------
    Bishop, C.M., Svensén, M., Williams, C.K.I. (1998).
        "GTM: The Generative Topographic Mapping."
        Neural Computation, 10(1), 215-234.

    Brehmer, J., Cranmer, K. (2020).
        "Flows for simultaneous manifold learning and density estimation."
        Advances in Neural Information Processing Systems (NeurIPS).

    Examples
    --------
    >>> from nfmap import NFMap
    >>> from sklearn.datasets import load_digits
    >>> X, y = load_digits(return_X_y=True)
    >>> model = NFMap(n_components=2, complexity='medium', verbose=True)
    >>> Z = model.fit_transform(X)
    >>> Z.shape
    (1797, 2)
    >>> # Log-likelihood of each point; -log_p is an out-of-distribution score
    >>> log_p = model.score_samples(X)
    >>> # ...or the on-manifold density only (map position, ignoring distance
    >>> # off the manifold), which is what this method used to return:
    >>> log_p_manifold = model.score_samples(X, mode='manifold')
    """
    def __init__(self, n_components=2, complexity='medium', use_canonical=False, canonical_weight=0.05, prior_weight=0.1, graph_weight=5.0, repulsion_weight=1.0, n_prior_components='auto', random_state=42, learnable_priors=False, learnable_prior_weights=False, vol_weight=1.0, sigma_perp=10.0, max_epochs=1000, patience=100, verbose=False, negative_sample_rate=5, early_exaggeration=False, prop_weight=1.0, graph_kernel='fuzzy_simplicial', perplexity=30, use_native_predictor=False, use_lmdt_head=False, max_ambient_dim=None, inject_weight=0.0, prior_layout='square', grid_bounds='auto', prior_family='gaussian', prior_dof=2.0, prior_init_scale=None, use_zperp_mdn=False, n_mdn_components=3, mdn_hidden_dim=128, mdn_epochs=200, zperp_lmdt_weight=20.0, checkpoint_path=None, checkpoint_every=10, centroid_weight=0.0, centroid_momentum=0.99, learning_rate=1e-3):
        self.n_components = n_components
        self.complexity = complexity
        self.use_canonical = use_canonical
        self.canonical_weight = canonical_weight
        self.prior_weight = prior_weight
        self.graph_weight = graph_weight
        self.repulsion_weight = repulsion_weight
        self.n_prior_components = n_prior_components
        self.random_state = random_state
        self.learnable_priors = learnable_priors
        self.learnable_prior_weights = learnable_prior_weights
        self.vol_weight = vol_weight
        self.sigma_perp = sigma_perp
        self.max_epochs = max_epochs
        self.patience = patience
        self.verbose = verbose
        self.negative_sample_rate = negative_sample_rate
        self.early_exaggeration = early_exaggeration
        self.prop_weight = prop_weight
        self.graph_kernel = graph_kernel
        self.perplexity = perplexity
        self.use_native_predictor = use_native_predictor
        self.use_lmdt_head = use_lmdt_head
        self.max_ambient_dim = max_ambient_dim
        self.inject_weight = inject_weight
        self.prior_layout = prior_layout
        self.grid_bounds = grid_bounds
        self.prior_family = prior_family
        self.prior_dof = prior_dof
        self.prior_init_scale = prior_init_scale
        self.use_zperp_mdn = use_zperp_mdn
        self.n_mdn_components = n_mdn_components
        self.mdn_hidden_dim = mdn_hidden_dim
        self.mdn_epochs = mdn_epochs
        self.zperp_lmdt_weight = zperp_lmdt_weight
        # checkpoint_path: if set, fit() saves training progress (model/optimizer/
        # scheduler state + early-stopping bookkeeping) to this path every
        # checkpoint_every epochs, and automatically resumes from it if it already
        # exists when fit() is called again - for long runs (large-N training sets)
        # where losing progress to a crash/sleep/interrupt would be costly. No
        # effect on behavior or results when left as None (the default).
        self.checkpoint_path = checkpoint_path
        self.checkpoint_every = checkpoint_every
        # EMA per-cluster centroid attraction ("gravity"): pulls each point's
        # z_parallel toward a running average position for its own cluster_labels
        # id (see fit()'s `cluster_labels` argument). Attraction-only by design -
        # no matching repulsion term - because the existing repulsion_weight
        # negative sampling and the GMM volume/prior loss already provide a
        # generic, label-agnostic spacing force; a second, label-aware repulsion
        # term would be redundant and fight that existing tuning. No effect
        # unless both centroid_weight > 0 and cluster_labels is passed to fit().
        self.centroid_weight = centroid_weight
        self.centroid_momentum = centroid_momentum
        self.learning_rate = learning_rate

        if self.graph_kernel == 'perplexity':
            warnings.warn("The 'perplexity' (t-SNE) graph kernel is deprecated and will be removed in a future release. Please use the default 'fuzzy_simplicial' kernel instead.", DeprecationWarning, stacklevel=2)

        if self.prior_family not in ('gaussian', 'student_t'):
            raise ValueError(f"prior_family must be 'gaussian' or 'student_t', got {self.prior_family!r}.")
        
    def _compute_perplexity_weights(self, distances, perplexity=30.0, tol=1e-5, max_iter=50):
        N, K = distances.shape
        weights = np.zeros((N, K))
        target_entropy = np.log(perplexity)
        
        for i in range(N):
            dist_sq = distances[i] ** 2
            beta_min, beta_max, beta = -np.inf, np.inf, 1.0
            best_P = np.zeros(K)
            
            for _ in range(max_iter):
                shifted_dist_sq = dist_sq - np.min(dist_sq)
                exp_d = np.exp(-beta * shifted_dist_sq)
                sum_exp_d = np.sum(exp_d)
                P = exp_d / sum_exp_d
                best_P = P

                # Shannon entropy of P. The max-subtraction above is for
                # numerical stability only, so it has to be used CONSISTENTLY:
                # with P_j = exp(-beta*dt_j)/Z and Z = sum_j exp(-beta*dt_j),
                #   H = -sum_j P_j log P_j = beta * sum_j P_j*dt_j + log Z
                # where dt is the SHIFTED distance. Using the raw dist_sq here
                # (while Z came from the shifted ones) inflated H by exactly
                # beta*min(dist_sq), which is strictly positive once the self-
                # column is dropped. The bisection then saw "entropy too high"
                # at every beta, doubled beta all max_iter times, and returned a
                # one-hot row: every requested perplexity (5, 30, 50) came back
                # with an achieved perplexity of 1.00, i.e. each point attached
                # to its single nearest neighbour regardless of this setting.
                entropy = beta * np.sum(P * shifted_dist_sq) + np.log(sum_exp_d)
                diff = entropy - target_entropy
                if np.abs(diff) < tol:
                    break
                    
                if diff > 0:
                    beta_min = beta
                    beta = beta * 2.0 if beta_max == np.inf else (beta + beta_max) / 2.0
                else:
                    beta_max = beta
                    beta = beta / 2.0 if beta_min == -np.inf else (beta + beta_min) / 2.0
                        
            weights[i] = best_P
            
        return weights
        
    def _check_is_binary(self, X):
        """Check if X contains only 0 and 1 (sparse or dense)."""
        if sp.issparse(X):
            return np.all((X.data == 0) | (X.data == 1))
        return np.all((X == 0) | (X == 1))


    def fit_transform(
        self,
        X: ArrayLike,
        cluster_labels: Optional[ArrayLike] = None,
        y_props: Optional[ArrayLike] = None,
        X_struct: Optional[ArrayLike] = None,
        noise_std: float = 0.0,
        knn_indices: Optional[NDArray] = None,
        knn_dists: Optional[NDArray] = None,
        warm_start_from: Optional["NFMap"] = None,
    ) -> NDArray[np.float32]:
        """
        Fit the model with X and apply dimensionality reduction.
        """
        self.fit(
            X, cluster_labels, y_props=y_props, X_struct=X_struct, noise_std=noise_std,
            knn_indices=knn_indices, knn_dists=knn_dists, warm_start_from=warm_start_from,
        )
        return self.transform(X)

    def fit(
        self,
        X: ArrayLike,
        cluster_labels: Optional[ArrayLike] = None,
        y_props: Optional[ArrayLike] = None,
        X_struct: Optional[ArrayLike] = None,
        noise_std: float = 0.0,
        knn_indices: Optional[NDArray] = None,
        knn_dists: Optional[NDArray] = None,
        warm_start_from: Optional["NFMap"] = None,
    ) -> "NFMap":
        """
        Fit the model with X.

        warm_start_from : Optional[NFMap], default=None
            Another already-fitted NFMap instance to initialize from. Reuses
            its fitted PCA/scaler basis (transform, not re-fit) and its
            module_ weights as the starting point, instead of fitting PCA
            from scratch and initializing the flow randomly - for
            curriculum-style progressive training (train on an easy subset,
            then continue training on a larger subset without re-deriving a
            different ambient coordinate system out from under the
            already-learned flow; a fresh PCA fit on a different X would give
            a different basis that the warm-started weights were never
            trained against).

            Caller is responsible for keeping architecture-affecting
            hyperparameters (n_components, complexity, max_ambient_dim,
            n_prior_components, prior_layout, grid_bounds,
            use_native_predictor, use_lmdt_head, use_zperp_mdn) identical to
            warm_start_from's - a mismatch surfaces as a clear
            load_state_dict shape error, not a silently wrong result.
            n_prior_components='auto' / grid_bounds='auto' are handled
            automatically (forced to warm_start_from's already-resolved
            values rather than re-derived from the new X's size, which would
            otherwise silently produce a mismatched architecture across
            phases of different N).
        """
        if warm_start_from is not None and not hasattr(warm_start_from, "module_"):
            raise ValueError("warm_start_from must already be a fitted NFMap (call .fit() on it first).")

        cluster_labels_idx = None
        if cluster_labels is not None:
            cluster_labels = np.asarray(cluster_labels)
            # classes_ / cluster_labels_idx: map arbitrary label values (strings,
            # non-contiguous ints, ...) to contiguous 0..C-1 ids for indexing the
            # EMA centroid table below.
            self.classes_, cluster_labels_idx = np.unique(cluster_labels, return_inverse=True)

        # PCA + Scaling
        # is_binary_ must check actual value membership {0, 1}, not just bounds
        # (X.max()<=1 and X.min()>=0) - that bounds check misclassifies any
        # continuous data min-max-scaled to [0,1] (e.g. moons/circles/wine in
        # benchmarks/nfmap_diagnostics.py's own loaders) as binary. That's not
        # cosmetic: the binary branch below caps n_components_pca at
        # X.shape[1]-1, which collapses the whole embedding to 1D for any
        # low-dimensional (D=2-3) continuous dataset scaled this way - found
        # via a genuinely 1D `Z` on the moons dataset (D=2) while building
        # Paper 1's A1 standard-benchmark sweep.
        if warm_start_from is not None:
            self.is_binary_ = warm_start_from.is_binary_
        elif X.dtype == bool:
            self.is_binary_ = True
        elif sp.issparse(X):
            self.is_binary_ = bool(np.all(np.isin(X.data, [0, 1])))
        else:
            self.is_binary_ = bool(np.all((X == 0) | (X == 1)))

        # Determine ambient dimension limits (use smart defaults if not explicitly set)
        cap_binary = 128 if self.max_ambient_dim is None else self.max_ambient_dim
        cap_cont = X.shape[1] if self.max_ambient_dim is None else self.max_ambient_dim

        if self.is_binary_:
            if warm_start_from is not None:
                # Reuse the fitted basis exactly - transform only, never re-fit,
                # so every phase's z_parallel lives in the same coordinate system.
                self.pca_ = warm_start_from.pca_
                X_pca = self.pca_.transform(X)
            else:
                # Binary data: Simple TruncatedSVD for binary features
                n_components_pca = min(cap_binary, X.shape[1] - 1)
                self.pca_ = TruncatedSVD(n_components=n_components_pca, random_state=self.random_state)
                X_pca = self.pca_.fit_transform(X)
        else:
            if sp.issparse(X):
                X_dense = X.toarray()
            else:
                X_dense = np.asarray(X, dtype=np.float32)

            if warm_start_from is not None:
                self.scaler_pre_ = warm_start_from.scaler_pre_
                X_dense_scaled = self.scaler_pre_.transform(X_dense)
                self.pca_ = warm_start_from.pca_
                X_pca = self.pca_.transform(X_dense_scaled)
            else:
                self.scaler_pre_ = StandardScaler()
                X_dense_scaled = self.scaler_pre_.fit_transform(X_dense)

                # n_samples is a hard ceiling on how many components PCA can
                # produce, independent of how wide the data is. Without it any
                # continuous dataset with more features than samples - wide data
                # is entirely normal (fingerprints, spectra, omics) - raised
                # "n_components=... must be between 0 and min(n_samples,
                # n_features)" from sklearn. The binary branch never hit this
                # because it caps at 128. This only changes behaviour in the case
                # that previously crashed: whenever n_features <= n_samples the
                # min() is unaffected, so no existing result moves.
                n_components_pca = min(cap_cont, X_dense_scaled.shape[1], X_dense_scaled.shape[0])
                if y_props is not None:
                    n_components_pca = max(n_components_pca, min(y_props.shape[1] + self.n_components + 10, X_dense_scaled.shape[1]))

                # If user explicitly set max_ambient_dim, enforce it as a hard cap
                if self.max_ambient_dim is not None:
                    n_components_pca = min(n_components_pca, self.max_ambient_dim)

                # Use IncrementalPCA for massive dense datasets to prevent OOM
                if X_dense_scaled.shape[0] > 100000:
                    self.pca_ = IncrementalPCA(n_components=n_components_pca, batch_size=min(5000, X_dense_scaled.shape[0]))
                else:
                    # Use randomized solver for high efficiency on large datasets if size/dimensions are high
                    solver = 'randomized' if (X_dense_scaled.shape[0] > 10000 and n_components_pca > 10) else 'auto'
                    self.pca_ = PCA(n_components=n_components_pca, svd_solver=solver, random_state=self.random_state)

                self.pca_.fit(X_dense_scaled)
                X_pca = self.pca_.transform(X_dense_scaled)

        # Step 2: Global Scaling (preserve relative PCA variances)
        if warm_start_from is not None:
            self.global_scale_ = warm_start_from.global_scale_
        else:
            self.global_scale_ = np.std(X_pca[:, 0]) if np.std(X_pca[:, 0]) > 1e-6 else 1.0
        X_scaled = X_pca / self.global_scale_

        # Calibrate the PCA/SVD truncation residual so score_samples() can
        # account for the part of the input the flow never sees. PCA keeps
        # D_scaled directions and throws the rest away; an out-of-distribution
        # point can be perfectly ordinary *inside* the retained subspace and
        # wildly abnormal outside it, and without this term the score is blind
        # to that (the "PCA reconstruction error is the gatekeeper" observation
        # in the applicability-domain notes). Modelling the discarded part as
        # isotropic Gaussian noise of variance tau^2 turns it into a normalised
        # log-density term that composes additively with the flow's own.
        self._calibrate_pca_residual(
            X if self.is_binary_ else X_dense_scaled, X_pca)

        N, D_scaled = X_scaled.shape
        self.D_scaled_ = D_scaled

        # Build kNN graph on scaled data for neighborhood-preservation loss.
        # Also built when use_lmdt_head is on but graph_weight is 0: the LMDT
        # head regresses onto the 2D distance to a sampled kNN neighbour, so it
        # needs this graph too. Previously the whole block - graph AND head -
        # was gated on graph_weight > 0, so `use_lmdt_head=True, graph_weight=0`
        # silently left the head at its random initialisation while fit() went
        # on to compute lmdt_p95_ from it, handing downstream code a confidently
        # calibrated but meaningless signal.
        needs_knn_graph = self.graph_weight > 0 or getattr(self, 'use_lmdt_head', False)
        if needs_knn_graph:
            kernel_name = getattr(self, 'graph_kernel', 'fuzzy_simplicial')
            if kernel_name == 'fuzzy_simplicial':
                k_neighbors = min(15, N - 1)
            elif kernel_name == 'perplexity':
                k_neighbors = min(int(3 * getattr(self, 'perplexity', 30)), N - 1)
            else:
                raise ValueError(f"Unknown graph kernel: {kernel_name}")
                
            if knn_indices is not None and knn_dists is not None:
                # Precomputed neighbor graph from an external, domain-specific
                # kernel (e.g. chem_kernel's hybrid Tanimoto graph) - bypasses
                # NNDescent/X_struct entirely. Contract: shape (N, k), self
                # NOT included (unlike the internally-computed paths below,
                # this is NOT run through the self-column-stripping step -
                # doing so would silently discard a real neighbor).
                knn_indices = np.asarray(knn_indices)
                knn_dists = np.asarray(knn_dists)
                if knn_indices.shape != knn_dists.shape or knn_indices.shape[0] != N:
                    raise ValueError(
                        f"knn_indices/knn_dists must both have shape (N={N}, k); got "
                        f"knn_indices={knn_indices.shape}, knn_dists={knn_dists.shape}. "
                        f"They must NOT include a self-column - fit() will not strip one."
                    )
                k_neighbors = knn_indices.shape[1]
            else:
                if X_struct is not None:
                    # Determine if X_struct is binary or continuous - value-membership
                    # check, not a bounds check (see the matching is_binary_ fix/comment
                    # above: X.max()<=1 and X.min()>=0 misclassifies continuous data
                    # that happens to be scaled into [0,1]).
                    is_struct_binary = (X_struct.dtype == bool or bool(np.all((X_struct == 0) | (X_struct == 1))))
                    struct_metric = 'jaccard' if is_struct_binary else 'euclidean'

                    # Use structural features to define the exact topological neighborhood!
                    if HAS_PYNNDESCENT:
                        # random_state pinned: NNDescent is itself a randomized algorithm
                        # (random projection trees) and silently defaults to random_state=
                        # None, i.e. an unseeded, non-reproducible graph on every call -
                        # this was the actual root cause of NFMap.fit() being
                        # non-reproducible even across two back-to-back calls with the
                        # same random_state in the same process, since the kNN graph the
                        # rest of training is built on differed every time.
                        index = NNDescent(X_struct, metric=struct_metric, n_neighbors=k_neighbors + 1, n_jobs=-1, random_state=self.random_state)
                        knn_indices, knn_dists = index.neighbor_graph
                    else:
                        nn_model = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='auto', metric=struct_metric, n_jobs=-1)
                        nn_model.fit(X_struct)
                        knn_dists, knn_indices = nn_model.kneighbors(X_struct)
                else:
                    if HAS_PYNNDESCENT:
                        # See the random_state comment on the X_struct branch above -
                        # same fix, same reason.
                        index = NNDescent(X_scaled, metric='euclidean', n_neighbors=k_neighbors + 1, n_jobs=-1, random_state=self.random_state)
                        knn_indices, knn_dists = index.neighbor_graph
                    else:
                        nn_model = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='auto', n_jobs=-1)  # +1 because kneighbors includes self
                        nn_model.fit(X_scaled)
                        knn_dists, knn_indices = nn_model.kneighbors(X_scaled)

                # Drop self-column (only present for internally-computed graphs)
                knn_dists = knn_dists[:, 1:]
                knn_indices = knn_indices[:, 1:]

            if kernel_name == 'fuzzy_simplicial':
                rho = knn_dists[:, 0:1]
                sigmas = np.mean(knn_dists - rho, axis=1, keepdims=True) + 1e-8
                knn_weights = np.exp(-(knn_dists - rho) / sigmas)
            elif kernel_name == 'perplexity':
                knn_weights = self._compute_perplexity_weights(knn_dists, perplexity=getattr(self, 'perplexity', 30))
                knn_weights = knn_weights * 3.0
            
            knn_indices_tensor = torch.tensor(knn_indices, dtype=torch.long)
            knn_weights_tensor = torch.tensor(knn_weights, dtype=torch.float32)
            X_scaled_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Adaptive Hyperparameter Engine based on complexity.
        # Validated rather than defaulted: the old `else: # 'medium'` silently
        # accepted anything, so a typo ('small', 'med', 'hi') or a stale config
        # value trained a medium model while the caller believed otherwise -
        # invisible in the results, since a medium model is perfectly plausible.
        if self.complexity == 'low':
            n_blocks = 4
            hidden_dim = max(64, min(2 * D_scaled, 512))
            max_epochs = 60
        elif self.complexity == 'high':
            n_blocks = 10
            hidden_dim = max(128, min(4 * D_scaled, 512))
            max_epochs = 100
        elif self.complexity == 'medium':
            n_blocks = 6
            hidden_dim = max(64, min(3 * D_scaled, 512))
            max_epochs = 80
        else:
            raise ValueError(
                f"complexity must be 'low', 'medium' or 'high'; got {self.complexity!r}.")
            
        if self.max_epochs is not None:
            max_epochs = self.max_epochs
            
        base_lr = self.learning_rate
        weight_decay = 1e-4
        batch_size = min(256, N)
        patience = 15 if self.patience is None else self.patience

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        self.resolved_n_prior_components_ = self.n_prior_components
        if self.resolved_n_prior_components_ == 'auto':
            if warm_start_from is not None:
                # Force 'auto' to match warm_start_from's already-resolved value rather
                # than re-deriving it from this phase's (likely different) N - this sizes
                # the GMM prior (prior_means shape), so re-deriving it here would
                # silently build a differently-shaped module_ that warm_start_from's
                # weights can't be loaded into. An EXPLICIT (non-'auto') mismatch is
                # left alone here on purpose - it should surface as the clear
                # load_state_dict ValueError above, not be silently overridden.
                self.resolved_n_prior_components_ = warm_start_from.resolved_n_prior_components_
            else:
                # Heuristic: roughly scale with sqrt(N), bounded [10, 400]
                self.resolved_n_prior_components_ = int(np.clip(np.sqrt(N) * 4, 10, 400))

        self.resolved_grid_bounds_ = self.grid_bounds
        if self.resolved_grid_bounds_ == 'auto':
            if warm_start_from is not None:
                self.resolved_grid_bounds_ = warm_start_from.resolved_grid_bounds_
            else:
                # Heuristic: Larger datasets naturally contain more clusters and need a wider canvas.
                # Minimum is 3.0 to ensure enough Gaussian coverage.
                # N=100 -> 3.0, N=1,000 -> 3.0, N=10,000 -> 3.3, N=100,000 -> 3.6
                self.resolved_grid_bounds_ = float(np.clip(2.0 + np.log10(N) * 0.33, 3.0, 4.5))
                if self.prior_family == 'student_t':
                    # Student-t's weaker far-field attraction spreads trained
                    # points out further than Gaussian at the SAME grid_bounds/
                    # sigma (confirmed empirically: ~45% larger spatial extent
                    # at dof=5 on real EGFR/hERG maps) - without widening the
                    # canvas to match, an increasing fraction of points end up
                    # many sigma from every prior component (a real density-
                    # model mismatch, not just a display issue - see nfmap
                    # project notes on the activity-landscape coverage gap this
                    # caused before this fix). 2x is the exact ratio validated
                    # at one tested scale (grid_bounds 3.0 -> 6.0 on ~10k-
                    # molecule EGFR/hERG maps, n_prior_components=100) - applied
                    # here across the whole N-dependent curve as the most
                    # honest available extrapolation, NOT independently
                    # validated at every N. Revisit with a real sweep before
                    # trusting this far from that tested scale (e.g. the
                    # planned ChEMBL-scale, millions-of-molecules work).
                    self.resolved_grid_bounds_ *= 2.0

        self.resolved_prior_init_scale_ = self.prior_init_scale
        if self.resolved_prior_init_scale_ is None and self.prior_family == 'student_t':
            if warm_start_from is not None:
                self.resolved_prior_init_scale_ = warm_start_from.resolved_prior_init_scale_
            else:
                # Auto-derive sigma from the resolved grid_bounds/
                # n_prior_components so widening the canvas (immediately above)
                # doesn't just spread the SAME small fixed components further
                # apart without giving them any more reach - see this
                # attribute's use in _NFMapModule and the "spacing-to-sigma
                # ratio" discussion in nfmap project notes. Formula matches
                # _get_grid_means's own hexagonal-packing spacing exactly
                # (nearest-neighbor distance for n_prior_components points
                # spread over a disk of radius grid_bounds); 0.6438 is the
                # sigma/spacing ratio measured from this codebase's own
                # (unrelated, Gaussian-era) default sigma=exp(-1)~=0.368 at
                # grid_bounds=3.0, n_prior_components=100 - i.e. this
                # reproduces the DEFAULT Gaussian ratio for student_t too,
                # not a specially-tuned student_t constant. Approximate for
                # prior_layout values other than 'hexagonal' (the only layout
                # this was validated with) - close enough in practice since
                # 'square'/'concentric' component spacing is the same order
                # of magnitude for the same grid_bounds/n_prior_components,
                # but not derived for those layouts specifically.
                area_per_point = np.pi * (self.resolved_grid_bounds_ ** 2) / max(self.resolved_n_prior_components_, 1)
                spacing = np.sqrt(2 * area_per_point / np.sqrt(3))
                self.resolved_prior_init_scale_ = float(0.6438 * spacing)

        self.module_ = _NFMapModule(
            d_scaled=D_scaled,
            n_components=self.n_components,
            n_blocks=n_blocks,
            hidden_dim=hidden_dim,
            use_canonical=self.use_canonical,
            canonical_weight=self.canonical_weight,
            sigma_perp=self.sigma_perp,
            prior_weight=self.prior_weight,
            n_prior_components=self.resolved_n_prior_components_,
            learnable_priors=self.learnable_priors,
            learnable_prior_weights=self.learnable_prior_weights,
            vol_weight=self.vol_weight,
            prop_weight=self.prop_weight,
            prior_layout=self.prior_layout,
            grid_bounds=self.resolved_grid_bounds_,
            prior_family=self.prior_family,
            prior_dof=self.prior_dof,
            prior_init_scale=self.resolved_prior_init_scale_,
            use_native_predictor=self.use_native_predictor,
            use_lmdt_head=self.use_lmdt_head,
            use_zperp_mdn=self.use_zperp_mdn,
            n_mdn_components=self.n_mdn_components,
            mdn_hidden_dim=self.mdn_hidden_dim
        )

        # NFMAP_DEVICE overrides the auto-detected device (cuda > mps > cpu)
        # when set, e.g. NFMAP_DEVICE=cpu - unset behaves exactly as before.
        # Added after repeated silent, traceback-free process deaths during
        # long unattended MPS training runs on Apple Silicon that coincided
        # with the Mac sleeping (lid close survives idle-sleep prevention
        # tools like caffeinate, but not lid-close sleep) - MPS's Metal
        # command queue/GPU context doesn't reliably survive a sleep/wake
        # cycle, unlike CPU tensors. Forcing CPU trades some per-epoch speed
        # for actually surviving an unattended multi-hour run.
        forced_device = os.environ.get('NFMAP_DEVICE')
        if forced_device:
            self.device_ = torch.device(forced_device)
        elif torch.cuda.is_available():
            self.device_ = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device_ = torch.device('mps')
        else:
            self.device_ = torch.device('cpu')
        self.module_.to(self.device_)

        if warm_start_from is not None:
            try:
                self.module_.load_state_dict(warm_start_from.module_.state_dict())
            except RuntimeError as e:
                raise ValueError(
                    "warm_start_from's module_ weights don't match this fit() call's module_ "
                    "architecture - check that n_components/complexity/max_ambient_dim/"
                    "n_prior_components/prior_layout/use_native_predictor/use_lmdt_head/"
                    "use_zperp_mdn are identical to warm_start_from's NFMap instance."
                ) from e

        optimizer = AdamW(self.module_.parameters(), lr=base_lr, weight_decay=weight_decay)

        # Scheduler: Cosine Annealing with Warmup (warmup for 10% of total epochs)
        warmup_epochs = max(1, int(0.1 * max_epochs))
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # (epoch + 1), not epoch: LambdaLR evaluates this at epoch=0
                # before the first step, so the original form set lr to exactly
                # 0.0 for the whole first epoch - AdamW's update and its
                # decoupled weight decay are both scaled by lr, so that epoch
                # was a complete no-op. Negligible at max_epochs=1000, one
                # wasted epoch in 60 at complexity='low'.
                return float(epoch + 1) / float(warmup_epochs)
            else:
                progress = float(epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
                return max(1e-5 / base_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = LambdaLR(optimizer, lr_lambda)

        # Dataset includes point indices so we can look up kNN neighbors per batch.
        # Optional tensors (y_props, cluster_labels_idx) are appended and their
        # batch-tuple positions tracked explicitly, rather than assumed, since
        # either one may be absent independently of the other.
        tensors_to_batch = [torch.tensor(X_scaled, dtype=torch.float32), torch.arange(N)]
        y_props_batch_idx = None
        if y_props is not None:
            y_props_batch_idx = len(tensors_to_batch)
            tensors_to_batch.append(torch.tensor(y_props, dtype=torch.float32))

        use_centroid = cluster_labels_idx is not None and self.centroid_weight > 0
        cluster_batch_idx = None
        if use_centroid:
            cluster_batch_idx = len(tensors_to_batch)
            tensors_to_batch.append(torch.tensor(cluster_labels_idx, dtype=torch.long))

        dataset = TensorDataset(*tensors_to_batch)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        # EMA per-cluster centroid table ("gravity" - see centroid_weight docstring).
        # Lazily initialized per-cluster on first sight (rather than zero-init,
        # which would otherwise pull every new cluster's first-seen point toward
        # the origin) - _cluster_centroid_initialized_ tracks which rows are live.
        if use_centroid:
            num_clusters = len(self.classes_)
            self.cluster_centroids_ = torch.zeros(num_clusters, self.n_components, device=self.device_)
            self._cluster_centroid_initialized_ = torch.zeros(num_clusters, dtype=torch.bool, device=self.device_)

        best_loss = float('inf')
        epochs_without_improvement = 0
        best_state = None
        start_epoch = 0
        # Per-epoch total loss (the same `epoch_loss` value the verbose print
        # already shows every max_epochs//10 epochs, just kept for every
        # epoch instead of a sampled few) - lets a caller inspect the actual
        # convergence curve after fit() returns, e.g. to check whether a run
        # was cut short by patience while still improving, rather than
        # guessing from the final loss value alone. Always recorded (not
        # gated behind a flag) - a list of floats is negligible memory even
        # for a several-thousand-epoch run, and there's no real reason
        # anyone would want to opt out.
        self.loss_history_ = []

        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            if self.verbose:
                print(f"Resuming from checkpoint: {self.checkpoint_path}")
            ckpt = torch.load(self.checkpoint_path, map_location=self.device_, weights_only=False)
            self.module_.load_state_dict(ckpt['module_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            scheduler.load_state_dict(ckpt['scheduler_state'])
            start_epoch = ckpt['epoch'] + 1
            best_loss = ckpt['best_loss']
            epochs_without_improvement = ckpt['epochs_without_improvement']
            best_state = ckpt['best_state']
            self.loss_history_ = ckpt.get('loss_history', [])
            if self.verbose:
                print(f"  Resumed at epoch {start_epoch}/{max_epochs}, best_loss={best_loss:.4f}")

        # Training loop
        self.module_.train()
        for epoch in range(start_epoch, max_epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                x_batch = batch[0].to(self.device_)
                if noise_std > 0.0:
                    x_batch = x_batch + torch.randn_like(x_batch) * noise_std
                idx_batch = batch[1]
                y_batch = batch[y_props_batch_idx].to(self.device_) if y_props_batch_idx is not None else None
                cluster_batch = batch[cluster_batch_idx].to(self.device_) if cluster_batch_idx is not None else None
                optimizer.zero_grad()
                # ── Part 1/2: Per-Sample Density Loss (inside _NFMapModule) ────────────
                # compute_loss() handles all per-sample terms:
                #   - GMM prior log-likelihood (ln p_Z(z))
                #   - Volume / Jacobian scaling penalty
                #   - Native property prediction loss (if use_native_predictor=True)
                # It only needs the current batch — no global dataset context required.
                loss, z_batch_par = self.module_.compute_loss(x_batch, y_props=y_batch)

                # ── Part 2/2: Topological (Graph) Loss — Fuzzy Simplicial Kernel ────
                # This loss is computed here (in fit()) rather than in compute_loss()
                # because it requires relational lookups into the full kNN graph:
                #   - neigh_idx looks up neighbors for the current batch from the
                #     FULL knn_indices_tensor (built from all N training points).
                #   - x_neigh can be any point in the dataset, not just the batch.
                # The module's compute_loss() only receives the current batch and has
                # no access to this dataset-level graph structure.
                #
                # Loss split:
                #   compute_loss() → per-sample density (GMM prior, volume, property)
                #   fit()          → relational topology (kNN attraction + repulsion)
                if needs_knn_graph:
                    # Sample one random neighbor per batch point from the kNN graph
                    rand_k = torch.randint(k_neighbors, (len(idx_batch),))
                    neigh_idx = knn_indices_tensor[idx_batch, rand_k]
                    neigh_weights = knn_weights_tensor[idx_batch, rand_k].to(self.device_)
                    
                    # Map the neighbor into latent space and extract its 2D parallel coords
                    x_neigh = X_scaled_tensor[neigh_idx].to(self.device_)
                    z_neigh, _ = self.module_.inverse_flow(x_neigh)
                    z_neigh_par = z_neigh[:, :self.module_.n_components]

                    # Shared by the graph loss and the LMDT head's target.
                    dist_sq = torch.sum((z_batch_par - z_neigh_par) ** 2, dim=-1)

                if self.graph_weight > 0:
                    # Attraction: w_ij * log(1 + ||z_i - z_j||^2)
                    # Weighted by fuzzy simplicial edge strength — stronger neighbors
                    # are pulled tighter. log1p softens the gradient for very close pairs.
                    loss_attract = torch.mean(neigh_weights * torch.log1p(dist_sq))
                    if self.early_exaggeration and epoch < (max_epochs // 3):
                        loss_attract = loss_attract * 4.0  # t-SNE-style early phase
                    loss_graph = loss_attract
                    
                    if self.repulsion_weight > 0 and self.negative_sample_rate > 0:
                        # Repulsion: log(1 + 1 / (||z_i - z_neg||^2 + eps))
                        # Non-neighbours are sampled from within the batch (a
                        # cheap approximation - they are not checked against the
                        # kNN graph). Pushes unrelated points apart to create
                        # whitespace between clusters.
                        # Each negative is drawn uniformly from the OTHER batch
                        # members. torch.randperm left its fixed points in - on
                        # average one pair per permutation had the point paired
                        # with itself, contributing a constant log1p(1/1e-4) =
                        # 9.2 with identically zero gradient (d||z-z||^2/dz = 0),
                        # so it only inflated the reported loss that early
                        # stopping reads. The +(>= arange) trick maps a draw from
                        # [0, B-1) onto [0, B) \ {i}, which is exactly uniform.
                        B_neg = z_batch_par.shape[0]
                        if B_neg > 1:
                            ar = torch.arange(B_neg, device=self.device_).unsqueeze(1)
                            neg_idx = torch.randint(
                                B_neg - 1, (B_neg, self.negative_sample_rate),
                                device=self.device_)
                            neg_idx = neg_idx + (neg_idx >= ar).long()
                            z_neg_par = z_batch_par[neg_idx]
                        else:
                            z_neg_par = z_batch_par.unsqueeze(1)
                        dist_sq_neg = torch.sum((z_batch_par.unsqueeze(1) - z_neg_par) ** 2, dim=-1)
                        loss_repel = torch.mean(torch.log1p(1.0 / (dist_sq_neg + 1e-4)))
                        loss_graph = loss_graph + self.repulsion_weight * loss_repel
                        
                    loss = loss + self.graph_weight * loss_graph

                if getattr(self, 'use_lmdt_head', False):
                    # 1. Predict distance for real (ID) data
                    # squeeze(-1), not squeeze(): a trailing batch of size 1
                    # (drop_last=False) collapsed the bare squeeze() to a
                    # 0-d scalar, which then broadcast against a shape-(1,)
                    # target instead of matching it.
                    pred_id = self.module_.lmdt_head(x_batch).squeeze(-1)
                    # Target is actual Euclidean distance in 2D
                    target_id = torch.sqrt(dist_sq.detach() + 1e-8)

                    # 2. Predict distance for fake (OOD) data
                    # Generate OOD data by independently permuting each feature column
                    # across the batch to destroy covariance but preserve marginal
                    # distributions. Vectorized (single argsort + gather over all D
                    # columns at once) - was previously a Python-level loop issuing one
                    # torch.randperm() + one indexed assignment PER COLUMN, every single
                    # training step. Measured (NFMAP_DEBUG_TIMING) at ~50% of total
                    # per-epoch training time on MPS (30ms/call x 20 calls/epoch on a
                    # small 5000x2048 test), from D_scaled sequential small kernel
                    # dispatches - also the leading suspect for repeated silent,
                    # traceback-free process deaths during long (300-epoch) use_lmdt_head
                    # runs on MPS (never observed during short 60-epoch runs). This
                    # produces the same per-column-independent-permutation property
                    # (verified: both give a valid permutation of each original column's
                    # own values), just without the per-column Python loop.
                    B_ood, D_ood = x_batch.shape
                    perm_idx = torch.argsort(torch.rand(D_ood, B_ood, device=x_batch.device), dim=1)
                    x_ood = torch.gather(x_batch.t(), 1, perm_idx).t()
                    pred_ood = self.module_.lmdt_head(x_ood).squeeze(-1)
                    target_ood = torch.full_like(pred_ood, 10.0)
                    
                    # 3. MSE Loss
                    loss_lmdt = torch.nn.functional.mse_loss(pred_id, target_id) + \
                                torch.nn.functional.mse_loss(pred_ood, target_ood)
                    # We can just add it directly because the head has completely independent weights
                    loss = loss + loss_lmdt

                # ── Part 3/3: Local Injectivity Loss (optional, inject_weight=0.0 by default) ──
                # Prevents the 2D map from collapsing structurally different molecules to the
                # same region of latent space (many-to-one folding). This is complementary to
                # the graph attraction loss:
                #   graph_weight:  similar in X  → close in z_par  (attraction)
                #   inject_weight: close in z_par → similar in X   (injectivity)
                # Uses a Gaussian kernel with median-heuristic bandwidth (no manual tau needed).
                # Only active when inject_weight > 0; completely inactive by default.
                if getattr(self, 'inject_weight', 0.0) > 0:
                    # Pairwise 2D distances within the batch
                    dists_z = torch.cdist(z_batch_par, z_batch_par)          # (B, B)
                    # Fixed bandwidth: prevents the runaway expansion loop because the 
                    # penalty naturally dies out once points are further than ~2.0 apart.
                    # Since our GMM grid is always anchored in [-3, 3], a fixed tau works perfectly.
                    tau = 1.0
                    w_inject = torch.exp(-dists_z ** 2 / (tau ** 2))         # (B, B)
                    # Pairwise input-space distances within the batch
                    dists_x = torch.cdist(x_batch, x_batch)                  # (B, B)
                    # Penalise: large input distance when 2D coords are close
                    loss_inject = (w_inject * dists_x).mean()
                    loss = loss + self.inject_weight * loss_inject

                # ── EMA Cluster-Centroid Attraction ("gravity", centroid_weight=0.0 by
                # default) ───────────────────────────────────────────────────────────
                # Attraction-only pull of each point's z_parallel toward a running
                # average position for its own cluster_labels id - e.g. a Murcko
                # scaffold id, tightening exact-match clusters into crisp dots on top
                # of the graph loss's softer, similarity-based macro placement. No
                # repulsion counterpart: the graph loss's repulsion_weight negative
                # sampling above and the GMM prior/volume terms in compute_loss()
                # already provide generic, label-agnostic spacing.
                #
                # Deliberately NOT a pairwise same-batch pull: that only produces a
                # gradient when two same-cluster points land in the same minibatch,
                # which becomes vanishingly rare for rare clusters in a large dataset
                # (long-tail scaffold distributions at ChEMBL scale). The centroid is
                # a lookup table updated from whichever cluster members pass through
                # each batch, so every point gets a well-defined pull target every
                # step regardless of whether a same-cluster partner co-occurs with it.
                if use_centroid and cluster_batch is not None:
                    with torch.no_grad():
                        for c in torch.unique(cluster_batch):
                            member_mask = cluster_batch == c
                            batch_mean = z_batch_par[member_mask].mean(dim=0).detach()
                            if not self._cluster_centroid_initialized_[c]:
                                self.cluster_centroids_[c] = batch_mean
                                self._cluster_centroid_initialized_[c] = True
                            else:
                                self.cluster_centroids_[c] = (
                                    self.centroid_momentum * self.cluster_centroids_[c]
                                    + (1.0 - self.centroid_momentum) * batch_mean
                                )
                    centroid_target = self.cluster_centroids_[cluster_batch].detach()
                    loss_centroid = torch.nn.functional.mse_loss(z_batch_par, centroid_target)
                    loss = loss + self.centroid_weight * loss_centroid

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.module_.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * x_batch.shape[0]
                
            scheduler.step()
            epoch_loss /= N
            self.loss_history_.append(epoch_loss)

            if self.verbose and (epoch + 1) % max(1, max_epochs // 10) == 0:
                print(f"Epoch {epoch+1:4d}/{max_epochs} | Loss: {epoch_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

            # TEMPORARY DEBUG TIMING - see the OOD-loop instrumentation above.
            if os.environ.get('NFMAP_DEBUG_TIMING') == '1':
                _ood_t = getattr(self, '_dbg_ood_loop_time', 0.0)
                _ood_n = getattr(self, '_dbg_ood_loop_calls', 0)
                print(f"[debug_timing] epoch {epoch+1} done at {time.time():.2f} | "
                      f"cumulative OOD-loop time: {_ood_t:.2f}s over {_ood_n} calls", flush=True)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_without_improvement = 0
                best_state = {k: v.cpu() for k, v in self.module_.state_dict().items()}
            else:
                epochs_without_improvement += 1
                
            if (
                self.checkpoint_path
                and self.checkpoint_every > 0
                and ((epoch + 1) % self.checkpoint_every == 0 or epochs_without_improvement >= patience)
            ):
                torch.save({
                    'epoch': epoch,
                    'module_state': self.module_.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'best_loss': best_loss,
                    'epochs_without_improvement': epochs_without_improvement,
                    'best_state': best_state,
                    'loss_history': self.loss_history_,
                }, self.checkpoint_path)

            if epochs_without_improvement >= patience:
                break

        if best_state is not None:
            self.module_.load_state_dict(best_state)

        # Calibrate the thickness of the manifold from the fitted model, so
        # score_samples() has a properly normalised density over z_perp rather
        # than relying on the sigma_perp hyperparameter (which is a penalty
        # weight during training, not a fitted scale, and is typically off by
        # an order of magnitude from the z_perp the flow actually produces).
        self._calibrate_z_perp_scale(X_scaled)

        # Calibrate LMDT score if enabled
        if getattr(self, 'use_lmdt_head', False):
            self.module_.eval()
            with torch.no_grad():
                raw_preds = []
                for i in range(0, len(X_scaled), 256):
                    xb = torch.tensor(X_scaled[i:i+256], dtype=torch.float32, device=self.device_)
                    # squeeze(-1), not squeeze(): a trailing chunk of one row
                    # otherwise yields a 0-d array and np.concatenate below
                    # raises on the dimension mismatch (hit at N=513).
                    raw_preds.append(self.module_.lmdt_head(xb).squeeze(-1).cpu().numpy())
                raw_preds = np.concatenate(raw_preds)
                self.lmdt_p95_ = float(np.percentile(raw_preds, 95))
            self.module_.train()

        # Train z_perp Mixture Density Head (predicts z_perp distribution from 2D coords)
        if getattr(self, 'use_zperp_mdn', False):
            self.module_.eval()

            # Cache (z_parallel, z_perp) for every training point via one inverse_flow pass.
            z_par_list, z_perp_list = [], []
            with torch.no_grad():
                for i in range(0, len(X_scaled), 256):
                    xb = torch.tensor(X_scaled[i:i+256], dtype=torch.float32, device=self.device_)
                    z_full, _ = self.module_.inverse_flow(xb)
                    z_par_list.append(z_full[:, :self.n_components])
                    z_perp_list.append(z_full[:, self.n_components:])
            z_par_cache = torch.cat(z_par_list, dim=0)
            z_perp_cache = torch.cat(z_perp_list, dim=0)

            self._fit_zperp_mdn(z_par_cache, z_perp_cache, batch_size)

        return self

    def _calibrate_pca_residual(self, X_source, X_pca):
        """
        Fit the isotropic noise variance of whatever PCA/SVD threw away.

        X_source is the input to the PCA step (standardised dense data, or the
        raw binary matrix for the TruncatedSVD path); X_pca is its projection.
        Stores `pca_residual_dim_` (how many directions were discarded) and
        `pca_residual_var_` (their mean per-direction variance). When nothing
        was discarded both terms vanish and the residual contributes 0.
        """
        self.pca_residual_dim_ = 0
        self.pca_residual_var_ = 1.0

        n_features = X_source.shape[1]
        n_kept = X_pca.shape[1]
        n_discarded = int(n_features - n_kept)
        if n_discarded <= 0:
            return

        # Subsample: this is a single scalar variance, it does not need every row.
        rng = np.random.RandomState(self.random_state)
        n_rows = X_source.shape[0]
        idx = (rng.choice(n_rows, 20000, replace=False)
               if n_rows > 20000 else np.arange(n_rows))

        total_sq = 0.0
        for i in range(0, len(idx), 2048):
            chunk = idx[i:i + 2048]
            src = X_source[chunk]
            if sp.issparse(src):
                src = src.toarray()
            src = np.asarray(src, dtype=np.float64)
            recon = self.pca_.inverse_transform(X_pca[chunk]).astype(np.float64)
            total_sq += float(np.sum((src - recon) ** 2))

        self.pca_residual_dim_ = n_discarded
        self.pca_residual_var_ = max(total_sq / (len(idx) * n_discarded), 1e-12)

    def _calibrate_z_perp_scale(self, X_scaled):
        """
        Fit the per-dimension std of z_perp on the training data.

        This is the scale of the Gaussian that `full_log_density` places on the
        off-manifold coordinates. It must be measured rather than assumed: with
        the default sigma_perp=10 the training penalty is deliberately weak
        ("thick manifold"), so the z_perp the fitted flow actually produces has
        a quite different spread - using sigma_perp here would mis-weight the
        off-manifold term by whatever that gap happens to be.
        """
        d_perp = self.D_scaled_ - self.n_components
        if d_perp <= 0:
            self.z_perp_log_scale_ = np.zeros(0, dtype=np.float64)
            return

        self.module_.eval()
        total = np.zeros(d_perp, dtype=np.float64)
        count = 0
        with torch.no_grad():
            for i in range(0, len(X_scaled), 512):
                xb = torch.tensor(X_scaled[i:i + 512], dtype=torch.float32, device=self.device_)
                z, _ = self.module_.inverse_flow(xb)
                zp = z[:, self.n_components:].cpu().double().numpy()
                total += np.sum(zp ** 2, axis=0)
                count += zp.shape[0]
        self.module_.train()

        # Second moment about zero, not about the mean: the model asserts the
        # manifold sits at z_perp = 0, so a systematic offset is genuine
        # off-manifold mass and should cost likelihood, not be absorbed.
        var = np.maximum(total / max(count, 1), 1e-12)
        self.z_perp_log_scale_ = 0.5 * np.log(var)

    def _log_jacobian_const(self):
        """
        Constant log|det| of the deterministic preprocessing chain, so that
        score_samples() is a density in the *original* input units rather than
        in internal scaled-PCA units.

        StandardScaler divides feature j by scale_j; the global scaling divides
        every retained PCA coordinate by global_scale_. PCA/TruncatedSVD itself
        is an orthonormal projection and contributes 0. All of this is a single
        constant - it never changes the ranking of points, only the offset - but
        including it is what makes the returned number an honest log-density.
        """
        const = -float(self.D_scaled_) * float(np.log(self.global_scale_))
        if not self.is_binary_ and hasattr(self, 'scaler_pre_'):
            scale = np.asarray(self.scaler_pre_.scale_, dtype=np.float64)
            const -= float(np.sum(np.log(np.maximum(scale, 1e-12))))
        return const

    def _fit_zperp_mdn(self, z_par_cache, z_perp_cache, batch_size):
        """
        Train self.module_.zperp_mdn to predict a mixture-of-Gaussians distribution
        over z_perp conditioned on z_parallel, given cached (z_parallel, z_perp) pairs
        from real training data. Factored out of fit() so it can also be driven
        directly with hand-constructed pairs (see experiments/prove_zperp_mdn.py).
        """
        # Freeze everything except the new head. Gradients from the (optional)
        # typicality term still flow through forward_flow/lmdt_head into the head's
        # outputs — freezing a module's params doesn't cut the autograd graph, it
        # just stops gradient accumulating on those params.
        for p in self.module_.parameters():
            p.requires_grad_(False)
        for p in self.module_.zperp_mdn.parameters():
            p.requires_grad_(True)

        mdn_optimizer = torch.optim.Adam(self.module_.zperp_mdn.parameters(), lr=1e-3)
        mdn_loader = DataLoader(TensorDataset(z_par_cache, z_perp_cache), batch_size=batch_size, shuffle=True)
        use_typicality = getattr(self, 'use_lmdt_head', False) and self.zperp_lmdt_weight > 0
        d_perp = self.module_.zperp_mdn.d_perp

        for _ in range(self.mdn_epochs):
            for z_par_b, z_perp_b in mdn_loader:
                mdn_optimizer.zero_grad()
                log_pi, mu, log_sigma = self.module_.zperp_mdn(z_par_b)

                sigma = torch.exp(log_sigma)
                diff = (z_perp_b.unsqueeze(1) - mu) / sigma
                log_prob_k = (-0.5 * diff.pow(2).sum(-1)
                              - log_sigma.sum(-1)
                              - 0.5 * d_perp * math.log(2 * math.pi))
                loss = -torch.logsumexp(log_pi + log_prob_k, dim=1).mean()

                if use_typicality:
                    # Score each component's own mean (not a pi-weighted blend of
                    # means — blending in latent space is exactly the "in-between,
                    # off-manifold" failure this term is meant to avoid), weighted
                    # by that component's responsibility.
                    pi = torch.exp(log_pi)
                    B, K, _ = mu.shape
                    z_query = torch.cat(
                        [z_par_b.unsqueeze(1).expand(-1, K, -1), mu], dim=-1
                    ).reshape(B * K, -1)
                    x_hat, _ = self.module_.forward_flow(z_query)
                    lmdt_pred = self.module_.lmdt_head(x_hat).squeeze(-1).view(B, K)
                    loss = loss + self.zperp_lmdt_weight * (pi * lmdt_pred).sum(dim=1).mean()

                loss.backward()
                mdn_optimizer.step()

        # Restore gradients on the rest of the module for any future re-fit.
        for p in self.module_.parameters():
            p.requires_grad_(True)
        self.module_.train()

    def transform(self, X: ArrayLike, return_full: bool = False) -> NDArray[np.float32]:
        """Apply dimensionality reduction to X."""
        if self.is_binary_:
            X_pca = self.pca_.transform(X)
        else:
            if sp.issparse(X):
                X_dense = X.toarray()
            else:
                X_dense = np.asarray(X, dtype=np.float32)
            X_dense_scaled = self.scaler_pre_.transform(X_dense)
            X_pca = self.pca_.transform(X_dense_scaled)
                
        X_scaled = X_pca / self.global_scale_
        
        self.module_.eval()
        dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
        # Explicit generator: DataLoader draws a base seed from the global
        # torch RNG each time it is iterated, so these inference-only loaders
        # (shuffle=False - the generator is otherwise unused) were silently
        # advancing the caller's torch random stream on every transform/
        # score/generate call.
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False,
                                generator=torch.Generator())
        
        z_parallel_list = []
        with torch.no_grad():
            for batch in dataloader:
                x_batch = batch[0].to(self.device_)
                z, _ = self.module_.inverse_flow(x_batch)
                if return_full:
                    z_parallel_list.append(z.cpu().numpy())
                else:
                    z_parallel_list.append(z[:, :self.n_components].cpu().numpy())
                
        return np.concatenate(z_parallel_list, axis=0)

    def generate(self, coords, z_perp_strategy='zeros', z_perp_noise_std=1.0, temperature=1.0, random_state=None):
        """
        Generate high-dimensional data samples (X) from specified 2D coordinates.

        Parameters
        ----------
        coords : array-like of shape (n_samples, 2)
            The target 2D coordinates on the map.
        z_perp_strategy : str, default='zeros'
            How to fill the orthogonal (non-manifold) dimensions.
            - 'zeros': Sets all orthogonal dimensions to exactly 0.
            - 'noise': Samples orthogonal dimensions from a normal distribution.
            - 'predicted': Samples from the coordinate-conditioned mixture density
              head (requires the model to have been fit with use_zperp_mdn=True).
        z_perp_noise_std : float, default=1.0
            The standard deviation of the noise if z_perp_strategy='noise'.
        temperature : float, default=1.0
            Only used when z_perp_strategy='predicted'. Scales the sampled component's
            std when drawing z_perp. temperature=0 returns the deterministic mean of
            the most likely mixture component instead of a stochastic draw.
        random_state : int or None
            Seed for random number generator if using noise or predicted.

        Returns
        -------
        X_gen : np.ndarray of shape (n_samples, n_features)
            The generated data points in the original ambient space.
        """
        coords = np.asarray(coords)
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)

        n_samples = coords.shape[0]

        # Local RNG, not np.random.seed(): seeding the global generator here
        # made generate(random_state=...) silently reset the caller's own numpy
        # stream, so an unrelated sampling loop around a generate() call lost
        # its reproducibility. `rng` is used for every draw below.
        rng = np.random.RandomState(random_state) if random_state is not None else np.random

        z = np.zeros((n_samples, self.D_scaled_))
        z[:, :self.n_components] = coords

        if z_perp_strategy == 'noise':
            z[:, self.n_components:] = rng.normal(
                0, z_perp_noise_std, size=(n_samples, self.D_scaled_ - self.n_components)
            )
        elif z_perp_strategy == 'predicted':
            if not hasattr(self.module_, 'zperp_mdn'):
                raise ValueError(
                    "z_perp_strategy='predicted' requires the model to have been fit "
                    "with use_zperp_mdn=True."
                )
            # Local torch Generator for the same reason as `rng` above, and
            # pinned to CPU so the draw is identical on cpu/cuda/mps.
            gen = torch.Generator()
            if random_state is not None:
                gen.manual_seed(int(random_state))
            self.module_.eval()
            with torch.no_grad():
                coords_tensor = torch.tensor(coords, dtype=torch.float32, device=self.device_)
                log_pi, mu, log_sigma = self.module_.zperp_mdn(coords_tensor)
                if temperature <= 0:
                    component_idx = torch.argmax(log_pi, dim=-1)
                else:
                    component_idx = torch.multinomial(
                        torch.exp(log_pi).cpu(), 1, generator=gen
                    ).squeeze(-1).to(log_pi.device)
                idx = component_idx.view(-1, 1, 1).expand(-1, 1, mu.shape[-1])
                mu_sel = torch.gather(mu, 1, idx).squeeze(1)
                if temperature > 0:
                    log_sigma_sel = torch.gather(log_sigma, 1, idx).squeeze(1)
                    eps = torch.randn(mu_sel.shape, generator=gen, dtype=torch.float32).to(mu_sel.device)
                    z_perp_pred = mu_sel + temperature * torch.exp(log_sigma_sel) * eps
                else:
                    z_perp_pred = mu_sel
            z[:, self.n_components:] = z_perp_pred.cpu().numpy()

        return self.inverse_transform(z)

    def inverse_transform(self, z):
        """
        Takes z in R^{N x 2} or R^{N x D}, pads with zeros if needed, runs inverse flow + inverse scaling,
        and returns generated X_gen in R^{N x D}.
        """
        # Put the PyTorch flow network into evaluation mode (turns off BatchNorm/Dropout)
        self.module_.eval()
        
        # Convert input to PyTorch tensor and prepare DataLoader for batch processing
        # This prevents Out-Of-Memory (OOM) errors when generating millions of molecules
        z_tensor = torch.tensor(z, dtype=torch.float32)
        dataset = TensorDataset(z_tensor)
        # Explicit generator: DataLoader draws a base seed from the global
        # torch RNG each time it is iterated, so these inference-only loaders
        # (shuffle=False - the generator is otherwise unused) were silently
        # advancing the caller's torch random stream on every transform/
        # score/generate call.
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False,
                                generator=torch.Generator())
        
        x_gen_scaled_list = []
        
        with torch.no_grad():
            for batch in dataloader:
                z_batch = batch[0].to(self.device_)
                N_batch = z_batch.shape[0]
                
                # If pure 2D coordinates are passed, pad the remaining 
                # perpendicular dimensions with zeros so the flow can invert properly.
                if z_batch.shape[1] == self.n_components:
                    z_perp_zero = torch.zeros((N_batch, self.D_scaled_ - self.n_components), device=self.device_)
                    z_padded = torch.cat([z_batch, z_perp_zero], dim=-1)
                else:
                    z_padded = z_batch
                
                # Pass padded coordinates backwards through the RealNVP flow
                # (forward_flow maps from latent space Z back to data space X)
                x_gen_batch, _ = self.module_.forward_flow(z_padded)
                
                # Move generated high-dimensional embeddings back to CPU
                x_gen_scaled_list.append(x_gen_batch.cpu().numpy())
                
        # Recombine batches into a single array
        x_gen_scaled = np.concatenate(x_gen_scaled_list, axis=0)
        
        # Reverse the global scaling applied during data preparation
        x_gen_pca = x_gen_scaled * self.global_scale_
        
        # Pass backwards through the PCA layer to undo dimensionality reduction
        x_gen_pre = self.pca_.inverse_transform(x_gen_pca)
        
        # Restore original mean and standard deviation via inverse StandardScaler
        if self.is_binary_:
            x_gen = x_gen_pre
        else:
            x_gen = self.scaler_pre_.inverse_transform(x_gen_pre)
            
        return x_gen

    def score_samples(self, X: ArrayLike, mode: str = 'full') -> NDArray[np.float64]:
        """
        Log-likelihood ln p(x) of each row of X under the fitted model.

        Higher = more typical of the training data. Lower = more anomalous, so
        `-score_samples(X)` is a ready-made out-of-distribution / applicability-
        domain score (the sklearn convention, matching IsolationForest and
        LocalOutlierFactor).

        Parameters
        ----------
        mode : {'full', 'manifold'}, default='full'
            'full'
                Density over the whole input space. Adds up the three ways a
                point can be unusual, each as a properly normalised log-density
                term, so they combine on one scale with no tuning:
                  1. an unusual position *on* the 2D map (the GMM prior),
                  2. an unusual distance *off* the learned manifold (z_perp),
                  3. an unusual amount of structure PCA could not represent
                     (the truncation residual).
                This is the mode to use for OOD detection.
            'manifold'
                The Brehmer & Cranmer (2020) on-manifold density,
                ln p_M = ln p_Z(z_par) - 1/2 ln det(J^T J), via the pullback
                metric. This was the previous (only) behaviour of this method.
                It is a density with respect to area *on the 2D manifold*, so it
                deliberately ignores terms 2 and 3 above: it answers "given that
                this point lies on the manifold, how typical is where it sits",
                and two points with the same map coordinate score identically
                however far off-manifold one of them is. Correct for asking
                about map position; blind as an OOD signal, which is why it is
                no longer the default.

        Notes
        -----
        'full' needs no Jacobian and is roughly an order of magnitude cheaper
        than 'manifold', which backpropagates per batch to build J.
        """
        if mode not in ('full', 'manifold'):
            raise ValueError(f"mode must be 'full' or 'manifold', got {mode!r}")

        if self.is_binary_:
            X_source = X
            X_pca = self.pca_.transform(X)
        else:
            if sp.issparse(X):
                X_dense = X.toarray()
            else:
                X_dense = np.asarray(X, dtype=np.float32)
            X_source = self.scaler_pre_.transform(X_dense)
            X_pca = self.pca_.transform(X_source)
                
        X_scaled = X_pca / self.global_scale_
        
        self.module_.eval()
        dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
        # Explicit generator: DataLoader draws a base seed from the global
        # torch RNG each time it is iterated, so these inference-only loaders
        # (shuffle=False - the generator is otherwise unused) were silently
        # advancing the caller's torch random stream on every transform/
        # score/generate call.
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False,
                                generator=torch.Generator())

        if mode == 'manifold':
            log_p_list = []
            # The pullback metric needs gradients, so no torch.no_grad() here.
            for batch in dataloader:
                x_batch = batch[0].to(self.device_)
                log_p_batch = self.module_.exact_density(x_batch)
                log_p_list.append(log_p_batch.cpu().detach().numpy())
            return np.concatenate(log_p_list, axis=0).astype(np.float64)

        # ── mode == 'full' ───────────────────────────────────────────────────
        # Per-dimension z_perp scale fitted at the end of fit(). Models pickled
        # before that existed fall back to the sigma_perp hyperparameter inside
        # full_log_density.
        z_perp_log_scale = getattr(self, 'z_perp_log_scale_', None)
        if z_perp_log_scale is not None and len(z_perp_log_scale) > 0:
            z_perp_log_scale = torch.tensor(
                z_perp_log_scale, dtype=torch.float32, device=self.device_)
        else:
            z_perp_log_scale = None

        log_p_list = []
        with torch.no_grad():
            for batch in dataloader:
                x_batch = batch[0].to(self.device_)
                log_p_batch = self.module_.full_log_density(x_batch, z_perp_log_scale)
                log_p_list.append(log_p_batch.cpu().double().numpy())
        log_p = np.concatenate(log_p_list, axis=0)

        # Term 3: the structure PCA discarded, as an isotropic Gaussian.
        log_p += self._pca_residual_log_density(X_source, X_pca)

        # Constant change-of-variables offset, so the result is a log-density in
        # the original input units. Does not affect ranking.
        return log_p + self._log_jacobian_const()

    def _pca_residual_log_density(self, X_source, X_pca):
        """
        log N(residual; 0, tau^2 I) for the directions PCA/SVD discarded, with
        tau^2 calibrated on the training data by `_calibrate_pca_residual`.

        Returns 0.0 when nothing was discarded, or for models pickled before
        this term existed (in which case the score is simply the flow's own
        density, exactly as those models were scored at the time).
        """
        n_discarded = getattr(self, 'pca_residual_dim_', 0)
        if not n_discarded:
            return 0.0
        tau_sq = float(getattr(self, 'pca_residual_var_', 1.0))

        out = np.empty(X_pca.shape[0], dtype=np.float64)
        for i in range(0, X_pca.shape[0], 2048):
            src = X_source[i:i + 2048]
            if sp.issparse(src):
                src = src.toarray()
            src = np.asarray(src, dtype=np.float64)
            recon = self.pca_.inverse_transform(X_pca[i:i + 2048]).astype(np.float64)
            out[i:i + 2048] = np.sum((src - recon) ** 2, axis=1)

        return (-0.5 * out / tau_sq
                - 0.5 * n_discarded * np.log(2 * np.pi * tau_sq))

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float32]:
        """
        Predict the GMM cluster responsibilities (probabilities) for each data point.
        Returns a matrix of shape [N_samples, n_prior_components] where each row sums to 1.
        Useful for coloring maps by cluster or extracting soft clustering labels.
        Responsibilities are posterior probabilities p(k|z): 
        "Given that we are looking at $z$, what is the probability that it belongs to cluster k?"
        """
        if self.is_binary_:
            X_pca = self.pca_.transform(X)
        else:
            if sp.issparse(X):
                X_dense = X.toarray()
            else:
                X_dense = np.asarray(X, dtype=np.float32)
            X_dense_scaled = self.scaler_pre_.transform(X_dense)
            X_pca = self.pca_.transform(X_dense_scaled)
                
        X_scaled = X_pca / self.global_scale_
        
        self.module_.eval()
        dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
        # Explicit generator: DataLoader draws a base seed from the global
        # torch RNG each time it is iterated, so these inference-only loaders
        # (shuffle=False - the generator is otherwise unused) were silently
        # advancing the caller's torch random stream on every transform/
        # score/generate call.
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False,
                                generator=torch.Generator())
        
        resp_list = []
        with torch.no_grad():
            for batch in dataloader:
                x_batch = batch[0].to(self.device_)
                resp_batch = self.module_.compute_responsibilities(x_batch)
                resp_list.append(resp_batch.cpu().numpy())
                
        return np.concatenate(resp_list, axis=0)

    def predict_activity(self, X: ArrayLike) -> NDArray[np.float32]:
        """Predict activity natively using the built-in Macro/Micro property head."""
        if not getattr(self, 'use_native_predictor', False):
            raise ValueError("predict_activity requires use_native_predictor=True")
            
        self.module_.eval()
        if self.is_binary_:
            X_pca = self.pca_.transform(X)
        else:
            if sp.issparse(X):
                X_dense = X.toarray()
            else:
                X_dense = np.asarray(X, dtype=np.float32)
            X_dense_scaled = self.scaler_pre_.transform(X_dense)
            X_pca = self.pca_.transform(X_dense_scaled)
                
        X_scaled = X_pca / self.global_scale_
        
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device_)
        
        dataset = TensorDataset(X_tensor)
        # Explicit generator: DataLoader draws a base seed from the global
        # torch RNG each time it is iterated, so these inference-only loaders
        # (shuffle=False - the generator is otherwise unused) were silently
        # advancing the caller's torch random stream on every transform/
        # score/generate call.
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False,
                                generator=torch.Generator())
        
        preds_list = []
        with torch.no_grad():
            for batch in dataloader:
                x_batch = batch[0]
                z, _ = self.module_.inverse_flow(x_batch)
                z_parallel = z[:, :self.n_components]
                z_perp = z[:, self.n_components:]
                
                # ── 1. Macro Property Prediction (The 2D Map) ───────────
                # This calculates the property purely based on where the molecule lands on the 2D map.
                # First, we calculate the Gaussian responsibilities (p_k): "Which cluster does this molecule belong to?"
                # It's very similar to my implementation in ugtm and previous papers (cf. H. Gaspar's "GTM-Based QSAR Models and Their Applicability Domains" (2015))
                z_expanded = z_parallel.unsqueeze(1)
                means = self.module_.prior_means.unsqueeze(0)
                log_scales = self.module_.prior_log_scales.unsqueeze(0)
                
                # Calculate the per-component log-likelihood (Gaussian, or Student-t if
                # prior_family='student_t') for each of the K components
                diff_sq = torch.sum(((z_expanded - means) / torch.exp(log_scales).unsqueeze(-1)) ** 2, dim=-1)

                # Conditional probability
                # in GMM terminology, it's p(z|k): (if we're in cluster k, what's the probability of generating z?)
                log_N = self.module_._prior_component_log_density(diff_sq, log_scales)
                
                # Compute priors
                # in GMM terminology, it's actually p(k)
                log_pi = torch.log_softmax(self.module_.prior_logits, dim=0).unsqueeze(0)
                
                # Add the prior component weights (pi) to get the full posterior log-probability
                # And convert log-probabilities to normalized responsibilities (p_k) using softmax
                # in GMM terminology, it's p(k|z)
                # (nb: marginal probability is computed implicitely in the softmax)
                p_k = torch.softmax(log_pi + log_N, dim=1)
                
                # The "Macro" property is simply the weighted average of the learnable cluster properties
                y_macro = torch.sum(p_k * self.module_.macro_weights.unsqueeze(0), dim=1, keepdim=True)
                
                # ── 2. Micro Property Prediction (The Thick Manifold) ───
                # The 2D map can't capture every chemical nuance. The remaining perpendicular
                # dimensions (z_perp) contain all the subtle structural details that didn't fit on the map.
                # We feed these off-manifold dimensions through a small Feed-Forward network (micro_head)
                # to extract a highly specific, localized property correction.
                y_micro = self.module_.micro_head(z_perp)
                
                # ── 3. Final Hybrid Prediction ──────────────────────────
                # The final prediction is the Macro baseline (general neighborhood on the map) 
                # plus the Micro correction (specific molecular details).
                y_pred = y_macro + y_micro
                preds_list.append(y_pred.cpu().numpy())

        return np.concatenate(preds_list, axis=0)

    def save(self, path: str) -> None:
        """
        Persist the fitted estimator (torch weights, fitted PCA/scaler, and
        any ad-hoc attributes set after fit, e.g. scaler_y_mean_/scaler_y_std_)
        to a single file via joblib. Nothing on the instance is unpicklable —
        NNDescent indices used during fit() are local to that call and never
        stored on self — so a plain dump/load round-trips the whole model.
        """
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "NFMap":
        """Load a model previously written by `save()`."""
        import joblib
        return joblib.load(path)
