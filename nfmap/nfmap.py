import warnings
import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD, PCA
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
    def _get_grid_means(self, n_comps, dim):
        if dim != 2:
            return torch.randn(n_comps, dim) * 2.0
        side = int(np.ceil(np.sqrt(n_comps)))
        grid_1d = torch.linspace(-3.0, 3.0, side)
        xx, yy = torch.meshgrid(grid_1d, grid_1d, indexing='ij')
        grid = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
        return grid[:n_comps, :]

    def __init__(self, d_scaled, n_components=2, n_blocks=5, hidden_dim=512, use_canonical=False, sigma_perp=0.05, prior_weight=0.005, n_prior_components=15, learnable_priors=True, vol_weight=1.0, prop_weight=1.0, **kwargs):
        super().__init__()
        self.d_scaled = d_scaled
        self.n_components = n_components
        self.use_canonical = use_canonical
        self.sigma_perp = sigma_perp
        self.prior_weight = prior_weight
        self.vol_weight = vol_weight
        self.n_prior_components = n_prior_components
        self.prop_weight = prop_weight
        self.use_native_predictor = kwargs.get('use_native_predictor', False)
        
        # Native Predictor Parameters
        if self.use_native_predictor:
            self.macro_weights = nn.Parameter(torch.zeros(n_prior_components))
            self.micro_head = nn.Linear(d_scaled - n_components, 1)
            nn.init.normal_(self.micro_head.weight, std=0.01)
            nn.init.zeros_(self.micro_head.bias)
        
        # GMM Prior Parameters
        grid_means = self._get_grid_means(n_prior_components, n_components)
        if learnable_priors:
            self.prior_means = nn.Parameter(grid_means)
            self.prior_log_scales = nn.Parameter(torch.full((n_prior_components,), -1.0))
        else:
            self.register_buffer('prior_means', grid_means)
            self.register_buffer('prior_log_scales', torch.full((n_prior_components,), -1.0))
        
        self.prior_logits = nn.Parameter(torch.zeros(n_prior_components))
        
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
            
        # 2. Calculate the Gaussian probability for each cluster
        log_N       = (-0.5 * diff_sq
                       - self.n_components * log_scales
                       - 0.5 * self.n_components * np.log(2 * np.pi))
                       
        # 3. Multiply by the "weight" (pi) of each cluster, and sum them up (logsumexp for stability)
        log_pi      = torch.log_softmax(self.prior_logits, dim=0).unsqueeze(0)
        log_p_gmm   = torch.logsumexp(log_pi + log_N, dim=1)
        
        # We want to MAXIMIZE probability, which means MINIMIZING negative probability
        loss_prior  = -self.prior_weight * log_p_gmm

        # ── Manifold alignment ───────────────────────────────────────────────
        # Penalizes off-manifold dimensions (z_perp). 
        # Unlike standard Brehmer flows which drive sigma_perp -> 0 to create a razor-thin manifold,
        # we often use a large sigma_perp (e.g. 0.5, 5, or 10) to create a "thick" manifold. 
        sigma_perp = sigma_perp if sigma_perp is not None else self.sigma_perp
        distance_to_manifold = torch.mean(z_perp ** 2, dim=-1)
        loss_manifold = (0.5 / (sigma_perp ** 2)) * distance_to_manifold
        
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
            loss      += 0.05 * loss_ortho + 0.05 * loss_diag

        return loss, z_parallel

    def exact_density(self, x):
        """
        Calculate the exact probability density of data points on the map.
        
        This relies on the change of variables formula from differential geometry.
        When you map a flat 2D plane into a complex d-dimensional space, the 2D plane gets
        stretched and distorted. If a 1x1 box on the map gets stretched into a 10x10
        box in the d-dimensional space, the "density" of molecules in that box drops (magnification).
        
        Formula:
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
        
        # G is the Riemannian Pullback Metric Tensor: G = J^T J
        G = torch.bmm(J.transpose(1, 2), J) # [B, n_components, n_components]
        
        # The true magnification factor (change in volume) is sqrt(det(G))
        det_G = torch.linalg.det(G)
        
        # ── 2. Calculate the Baseline 2D Density (GMM Prior) ─────────────────
        z_expanded = z_parallel.unsqueeze(1)
        means = self.prior_means.unsqueeze(0)
        log_scales = self.prior_log_scales.unsqueeze(0)
        
        diff_sq = torch.sum(((z_expanded - means) / torch.exp(log_scales).unsqueeze(-1))**2, dim=-1)
        log_N_components = -0.5 * diff_sq - self.n_components * log_scales - 0.5 * self.n_components * np.log(2 * np.pi)
        log_pi = torch.log_softmax(self.prior_logits, dim=0).unsqueeze(0)
        
        # log p_Z(z)
        log_p_z = torch.logsumexp(log_pi + log_N_components, dim=1)
        
        # ── 3. Final Exact Density Calculation ────────────────────────────────
        # log p = log p_Z - log(magnification_factor)
        # Note: log(sqrt(det(G))) is identical to 0.5 * log(det(G))
        log_p = log_p_z - 0.5 * torch.log(torch.clamp(det_G, min=1e-12))
        return log_p

    def compute_responsibilities(self, x):
        """
        Calculate the exact GMM responsibilities (cluster probabilities) for each point.
        Returns a tensor of shape [Batch, n_prior_components] where each row sums to 1.
        """
        z, _ = self.inverse_flow(x)
        z_parallel = z[:, :self.n_components]
        
        z_expanded = z_parallel.unsqueeze(1)
        means = self.prior_means.unsqueeze(0)
        log_scales = self.prior_log_scales.unsqueeze(0)
        
        diff_sq = torch.sum(((z_expanded - means) / torch.exp(log_scales).unsqueeze(-1))**2, dim=-1)
        log_N_components = -0.5 * diff_sq - self.n_components * log_scales - 0.5 * self.n_components * np.log(2 * np.pi)
        log_pi = torch.log_softmax(self.prior_logits, dim=0).unsqueeze(0)
        
        log_unnormalized_probs = log_pi + log_N_components
        log_p_z = torch.logsumexp(log_unnormalized_probs, dim=1, keepdim=True)
        
        log_responsibilities = log_unnormalized_probs - log_p_z
        return torch.exp(log_responsibilities)


class NFMap(BaseEstimator, TransformerMixin):
    """
    scikit-learn compatible wrapper class for NFMap.
    Handles data preprocessing, PyTorch training loop, and adaptive hyperparameters.
    """
    def __init__(self, n_components=2, complexity='medium', use_canonical=False, prior_weight=0.005, graph_weight=1.0, repulsion_weight=1.0, n_prior_components=15, random_state=42, learnable_priors=True, vol_weight=1.0, sigma_perp=0.05, max_epochs=None, patience=None, verbose=False, negative_sample_rate=1, early_exaggeration=True, prop_weight=1.0, graph_kernel='fuzzy_simplicial', perplexity=30, use_native_predictor=False, max_ambient_dim=None):
        self.n_components = n_components
        self.complexity = complexity
        self.use_canonical = use_canonical
        self.prior_weight = prior_weight
        self.graph_weight = graph_weight
        self.repulsion_weight = repulsion_weight
        self.n_prior_components = n_prior_components
        self.random_state = random_state
        self.learnable_priors = learnable_priors
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
        self.max_ambient_dim = max_ambient_dim
        
        if self.graph_kernel == 'perplexity':
            warnings.warn("The 'perplexity' (t-SNE) graph kernel is deprecated and will be removed in a future release. Please use the default 'fuzzy_simplicial' kernel instead.", DeprecationWarning, stacklevel=2)
        
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
                
                entropy = beta * np.sum(P * dist_sq) + np.log(sum_exp_d)
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


    def fit_transform(self, X, y=None, y_props=None, X_struct=None, noise_std=0.0):
        """
        Fit the model with X and apply dimensionality reduction.
        """
        self.fit(X, y, y_props=y_props, X_struct=X_struct, noise_std=noise_std)
        return self.transform(X)

    def fit(self, X, y=None, y_props=None, X_struct=None, noise_std=0.0):
        """
        Fit the model with X.
        """
        if y is not None:
            self.classes_ = np.unique(y)
        
        # PCA + Scaling
        self.is_binary_ = (X.dtype == bool or (X.max() <= 1 and X.min() >= 0))
        
        # Determine ambient dimension limits (use smart defaults if not explicitly set)
        cap_binary = 128 if self.max_ambient_dim is None else self.max_ambient_dim
        cap_cont = 256 if self.max_ambient_dim is None else self.max_ambient_dim
        
        if self.is_binary_:
            # Binary data: Simple TruncatedSVD for binary features
            n_components_pca = min(cap_binary, X.shape[1] - 1)
            self.pca_ = TruncatedSVD(n_components=n_components_pca, random_state=self.random_state)
            X_pca = self.pca_.fit_transform(X)
        else:
            if sp.issparse(X):
                X_dense = X.toarray()
            else:
                X_dense = np.asarray(X, dtype=np.float32)
            self.scaler_pre_ = StandardScaler()
            X_dense_scaled = self.scaler_pre_.fit_transform(X_dense)
            
            n_components_pca = min(cap_cont, X_dense_scaled.shape[1])
            if y_props is not None:
                n_components_pca = max(n_components_pca, min(y_props.shape[1] + self.n_components + 10, X_dense_scaled.shape[1]))
            
            # If user explicitly set max_ambient_dim, enforce it as a hard cap
            if self.max_ambient_dim is not None:
                n_components_pca = min(n_components_pca, self.max_ambient_dim)
                
            # Use randomized solver for high efficiency on large datasets if size/dimensions are high
            solver = 'randomized' if (X_dense_scaled.shape[0] > 10000 and n_components_pca > 10) else 'auto'
            self.pca_ = PCA(n_components=n_components_pca, svd_solver=solver, random_state=self.random_state)
            X_pca = self.pca_.fit_transform(X_dense_scaled)
                
        # Step 2: Global Scaling (preserve relative PCA variances)
        self.global_scale_ = np.std(X_pca[:, 0]) if np.std(X_pca[:, 0]) > 1e-6 else 1.0
        X_scaled = X_pca / self.global_scale_
        
        N, D_scaled = X_scaled.shape
        self.D_scaled_ = D_scaled

        # Build kNN graph on scaled data for neighborhood-preservation loss
        if self.graph_weight > 0:
            kernel_name = getattr(self, 'graph_kernel', 'fuzzy_simplicial')
            if kernel_name == 'fuzzy_simplicial':
                k_neighbors = min(15, N - 1)
            elif kernel_name == 'perplexity':
                k_neighbors = min(int(3 * getattr(self, 'perplexity', 30)), N - 1)
            else:
                raise ValueError(f"Unknown graph kernel: {kernel_name}")
                
            if X_struct is not None:
                # Use structural fingerprints to define the exact topological neighborhood!
                if HAS_PYNNDESCENT:
                    index = NNDescent(X_struct, metric='jaccard', n_neighbors=k_neighbors + 1, n_jobs=-1)
                    knn_indices, knn_dists = index.neighbor_graph
                else:
                    nn_model = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='auto', metric='jaccard', n_jobs=-1)
                    nn_model.fit(X_struct)
                    knn_dists, knn_indices = nn_model.kneighbors(X_struct)
            else:
                if HAS_PYNNDESCENT:
                    index = NNDescent(X_scaled, metric='euclidean', n_neighbors=k_neighbors + 1, n_jobs=-1)
                    knn_indices, knn_dists = index.neighbor_graph
                else:
                    nn_model = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='auto', n_jobs=-1)  # +1 because kneighbors includes self
                    nn_model.fit(X_scaled)
                    knn_dists, knn_indices = nn_model.kneighbors(X_scaled)
            
            # Drop self-column
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

        # Adaptive Hyperparameter Engine based on complexity
        if self.complexity == 'low':
            n_blocks = 4
            hidden_dim = max(64, min(2 * D_scaled, 128))
            max_epochs = 60
        elif self.complexity == 'high':
            n_blocks = 10
            hidden_dim = max(128, min(4 * D_scaled, 512))
            max_epochs = 100
        else: # 'medium'
            n_blocks = 6
            hidden_dim = max(64, min(3 * D_scaled, 256))
            max_epochs = 80
            
        if self.max_epochs is not None:
            max_epochs = self.max_epochs
            
        base_lr = 1e-3
        weight_decay = 1e-4
        batch_size = min(256, N)
        patience = 15 if self.patience is None else self.patience

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        self.module_ = _NFMapModule(
            d_scaled=D_scaled,
            n_components=self.n_components,
            n_blocks=n_blocks,
            hidden_dim=hidden_dim,
            use_canonical=self.use_canonical,
            sigma_perp=self.sigma_perp,
            prior_weight=self.prior_weight,
            n_prior_components=self.n_prior_components,
            learnable_priors=self.learnable_priors,
            vol_weight=self.vol_weight,
            prop_weight=self.prop_weight,
            use_native_predictor=self.use_native_predictor
        )

        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.module_.to(self.device_)

        optimizer = AdamW(self.module_.parameters(), lr=base_lr, weight_decay=weight_decay)

        # Scheduler: Cosine Annealing with Warmup (warmup for 10% of total epochs)
        warmup_epochs = max(1, int(0.1 * max_epochs))
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch) / float(warmup_epochs)
            else:
                progress = float(epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
                return max(1e-5 / base_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = LambdaLR(optimizer, lr_lambda)

        # Dataset includes point indices so we can look up kNN neighbors per batch
        tensors_to_batch = [torch.tensor(X_scaled, dtype=torch.float32), torch.arange(N)]
        if y_props is not None:
            tensors_to_batch.append(torch.tensor(y_props, dtype=torch.float32))
        if X_struct is not None:
            tensors_to_batch.append(torch.tensor(np.asarray(X_struct, dtype=np.float32), dtype=torch.float32))
            
        dataset = TensorDataset(*tensors_to_batch)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        best_loss = float('inf')
        epochs_without_improvement = 0
        best_state = None

        # Training loop
        self.module_.train()
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                x_batch = batch[0].to(self.device_)
                if noise_std > 0.0:
                    x_batch = x_batch + torch.randn_like(x_batch) * noise_std
                idx_batch = batch[1]
                y_batch = batch[2].to(self.device_) if y_props is not None else None
                optimizer.zero_grad()
                loss, z_batch_par = self.module_.compute_loss(x_batch, y_props=y_batch)

                if self.graph_weight > 0:
                    rand_k = torch.randint(k_neighbors, (len(idx_batch),))
                    neigh_idx = knn_indices_tensor[idx_batch, rand_k]
                    neigh_weights = knn_weights_tensor[idx_batch, rand_k].to(self.device_)
                    
                    x_neigh = X_scaled_tensor[neigh_idx].to(self.device_)
                    z_neigh, _ = self.module_.inverse_flow(x_neigh)
                    z_neigh_par = z_neigh[:, :self.module_.n_components]
                    
                    dist_sq = torch.sum((z_batch_par - z_neigh_par) ** 2, dim=-1)
                    loss_attract = torch.mean(neigh_weights * torch.log1p(dist_sq))
                    if self.early_exaggeration and epoch < (max_epochs // 3):
                        loss_attract = loss_attract * 4.0
                    loss_graph = loss_attract
                    
                    if self.repulsion_weight > 0:
                        z_neg_par_list = []
                        for _ in range(self.negative_sample_rate):
                            shuffle_idx = torch.randperm(z_batch_par.shape[0], device=self.device_)
                            z_neg_par_list.append(z_batch_par[shuffle_idx])
                        z_neg_par = torch.stack(z_neg_par_list, dim=1)
                        dist_sq_neg = torch.sum((z_batch_par.unsqueeze(1) - z_neg_par) ** 2, dim=-1)
                        loss_repel = torch.mean(torch.log1p(1.0 / (dist_sq_neg + 1e-4)))
                        loss_graph = loss_graph + self.repulsion_weight * loss_repel

                    loss = loss + self.graph_weight * loss_graph

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.module_.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * x_batch.shape[0]
                
            scheduler.step()
            epoch_loss /= N
            
            if self.verbose and (epoch + 1) % max(1, max_epochs // 10) == 0:
                print(f"Epoch {epoch+1:4d}/{max_epochs} | Loss: {epoch_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_without_improvement = 0
                best_state = {k: v.cpu() for k, v in self.module_.state_dict().items()}
            else:
                epochs_without_improvement += 1
                
            if epochs_without_improvement >= patience:
                break
                
        if best_state is not None:
            self.module_.load_state_dict(best_state)
            
        return self

    def transform(self, X, return_full=False):
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
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
        
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
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
        
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

    def score_samples(self, X):
        """
        Returns exact probability density ln p_X(x) for the given data points 
        by computing the pullback metric.
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
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
        
        log_p_list = []
        # Evaluating score samples requires gradients for Jacobian calculations
        for batch in dataloader:
            x_batch = batch[0].to(self.device_)
            log_p_batch = self.module_.exact_density(x_batch)
            log_p_list.append(log_p_batch.cpu().detach().numpy())
            
        return np.concatenate(log_p_list, axis=0)

    def predict_proba(self, X):
        """
        Predict the GMM cluster responsibilities (probabilities) for each data point.
        Returns a matrix of shape [N_samples, n_prior_components] where each row sums to 1.
        Useful for coloring maps by cluster or extracting soft clustering labels.
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
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
        
        resp_list = []
        with torch.no_grad():
            for batch in dataloader:
                x_batch = batch[0].to(self.device_)
                resp_batch = self.module_.compute_responsibilities(x_batch)
                resp_list.append(resp_batch.cpu().numpy())
                
        return np.concatenate(resp_list, axis=0)

    def predict_activity(self, X):
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
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
        
        preds_list = []
        with torch.no_grad():
            for batch in dataloader:
                x_batch = batch[0]
                z, _ = self.module_.inverse_flow(x_batch)
                z_parallel = z[:, :self.n_components]
                z_perp = z[:, self.n_components:]
                
                # Compute macro
                z_expanded = z_parallel.unsqueeze(1)
                means = self.module_.prior_means.unsqueeze(0)
                log_scales = self.module_.prior_log_scales.unsqueeze(0)
                diff_sq = torch.sum(((z_expanded - means) / torch.exp(log_scales).unsqueeze(-1)) ** 2, dim=-1)
                log_N = (-0.5 * diff_sq - self.n_components * log_scales - 0.5 * self.n_components * np.log(2 * np.pi))
                log_pi = torch.log_softmax(self.module_.prior_logits, dim=0).unsqueeze(0)
                
                p_k = torch.softmax(log_pi + log_N, dim=1)
                y_macro = torch.sum(p_k * self.module_.macro_weights.unsqueeze(0), dim=1, keepdim=True)
                
                y_micro = self.module_.micro_head(z_perp)
                y_pred = y_macro + y_micro
                preds_list.append(y_pred.cpu().numpy())
                
        return np.concatenate(preds_list, axis=0)
