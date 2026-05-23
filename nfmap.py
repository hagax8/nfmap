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


class RealNVPBlock(nn.Module):
    """
    A single Affine Coupling Layer for the RealNVP normalizing flow.
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

    def __init__(self, d_scaled, n_components=2, n_blocks=5, hidden_dim=512, use_canonical=False, sigma_perp=0.05, prior_weight=0.005, n_prior_components=15, learnable_priors=True, vol_weight=1.0):
        super().__init__()
        self.d_scaled = d_scaled
        self.n_components = n_components
        self.use_canonical = use_canonical
        self.sigma_perp = sigma_perp
        self.prior_weight = prior_weight
        self.vol_weight = vol_weight
        self.n_prior_components = n_prior_components
        
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
                mask[:d_scaled // 2] = 1.0
            else:
                mask[d_scaled // 2:] = 1.0
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

    def compute_loss(self, x, sigma_perp=None):
        z, log_det_inv = self.inverse_flow(x)

        z_parallel = z[:, :self.n_components]
        z_perp     = z[:, self.n_components:]

        # ── GMM Prior ────────────────────────────────────────────────────────
        z_expanded  = z_parallel.unsqueeze(1)
        means       = self.prior_means.unsqueeze(0)
        log_scales  = self.prior_log_scales.unsqueeze(0)
        diff_sq     = torch.sum(
            ((z_expanded - means) / torch.exp(log_scales).unsqueeze(-1)) ** 2,
            dim=-1)
        log_N       = (-0.5 * diff_sq
                       - self.n_components * log_scales
                       - 0.5 * self.n_components * np.log(2 * np.pi))
        log_pi      = torch.log_softmax(self.prior_logits, dim=0).unsqueeze(0)
        log_p_gmm   = torch.logsumexp(log_pi + log_N, dim=1)
        loss_prior  = -self.prior_weight * log_p_gmm

        # ── Manifold alignment ───────────────────────────────────────────────
        sigma_perp = sigma_perp if sigma_perp is not None else self.sigma_perp
        loss_manifold = 0.5 / (sigma_perp ** 2) * torch.mean(z_perp ** 2, dim=-1)

        # ── Volume term: full-flow log-det scaled to manifold dims ───────────
        loss_vol = -log_det_inv * (self.n_components / self.d_scaled) * self.vol_weight

        loss = (loss_prior + loss_manifold + loss_vol).mean()

        # ── Optional conformal regularization ────────────────────────────────
        if self.use_canonical:
            z_para_req  = z_parallel.detach().requires_grad_(True)
            z_perp_zero = torch.zeros_like(z_perp)
            z_padded    = torch.cat([z_para_req, z_perp_zero], dim=-1)
            x_gen, _    = self.forward_flow(z_padded)
            u    = torch.ones_like(x_gen, requires_grad=True)
            g    = torch.autograd.grad(torch.sum(x_gen * u), z_para_req,
                                       create_graph=True)[0]
            J_cols = []
            for i in range(self.n_components):
                J_cols.append(torch.autograd.grad(
                    torch.sum(g[:, i]), u,
                    create_graph=True, retain_graph=True)[0])
            J = torch.stack(J_cols, dim=2)
            G = torch.bmm(J.transpose(1, 2), J)
            eye      = torch.eye(self.n_components, device=G.device).unsqueeze(0)
            off_diag = 1.0 - eye
            loss_ortho = (torch.abs(G) * off_diag).sum(dim=(1, 2)).mean()
            diag_G     = torch.diagonal(G, dim1=-2, dim2=-1)
            loss_diag  = torch.abs(diag_G - 1.0).mean()
            loss      += 0.05 * loss_ortho + 0.05 * loss_diag

        return loss, z_parallel

    def exact_density(self, x):
        """
        Calculate exact map density:
        ln p_X(x) = ln N(z_parallel | 0, I) - 1/2 ln det(G(z_parallel))
        """
        z, _ = self.inverse_flow(x)
        z_parallel = z[:, :self.n_components]
        
        # Compute G(z_parallel)
        z_para_req = z_parallel.detach().requires_grad_(True)
        z_perp_zero = torch.zeros((x.shape[0], self.d_scaled - self.n_components), device=x.device)
        z_padded = torch.cat([z_para_req, z_perp_zero], dim=-1)
        x_gen, _ = self.forward_flow(z_padded)
        
        # Efficiently compute the Jacobian using the double-autograd JVP trick
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
            
        J = torch.stack(J_cols, dim=2) # [B, D, n_components]
        G = torch.bmm(J.transpose(1, 2), J) # [B, n_components, n_components]
        
        det_G = torch.linalg.det(G)
        
        # log p(z) from GMM
        z_expanded = z_parallel.unsqueeze(1)
        means = self.prior_means.unsqueeze(0)
        log_scales = self.prior_log_scales.unsqueeze(0)
        
        diff_sq = torch.sum(((z_expanded - means) / torch.exp(log_scales).unsqueeze(-1))**2, dim=-1)
        log_N_components = -0.5 * diff_sq - self.n_components * log_scales - 0.5 * self.n_components * np.log(2 * np.pi)
        log_pi = torch.log_softmax(self.prior_logits, dim=0).unsqueeze(0)
        log_p_z = torch.logsumexp(log_pi + log_N_components, dim=1)
        
        log_p = log_p_z - 0.5 * torch.log(torch.clamp(det_G, min=1e-12))
        return log_p


class NFMap(BaseEstimator, TransformerMixin):
    """
    scikit-learn compatible wrapper class for NFMap.
    Handles data preprocessing, PyTorch training loop, and adaptive hyperparameters.
    """
    def __init__(self, n_components=2, complexity='medium', use_canonical=False, prior_weight=0.005, graph_weight=1.0, repulsion_weight=1.0, n_prior_components=15, random_state=42, learnable_priors=True, vol_weight=1.0, sigma_perp=0.05, max_epochs=None, patience=None, verbose=False, negative_sample_rate=1):
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
        
    def _check_is_binary(self, X):
        """Check if X contains only 0 and 1 (sparse or dense)."""
        if sp.issparse(X):
            return np.all((X.data == 0) | (X.data == 1))
        return np.all((X == 0) | (X == 1))

    def fit_transform(self, X, y=None):
        """
        Ingests data, routes/scales it, builds the network, trains via early stopping, 
        and returns the manifold coordinates.
        """
        # Step 1: PCA-alignment (GTM-inspired improvement for NFMap)
        # We align the input features with their principal components so that
        # the identity-initialized flow starts directly from the PCA projection.
        self.is_binary_ = self._check_is_binary(X)
        if self.is_binary_:
            n_components_pca = min(128, X.shape[1] - 1)
            self.pca_ = TruncatedSVD(n_components=n_components_pca, random_state=self.random_state)
            X_pca = self.pca_.fit_transform(X)
        else:
            if sp.issparse(X):
                X_dense = X.toarray()
            else:
                X_dense = np.asarray(X, dtype=np.float32)
            
            # Step 1: Pre-PCA Standardization (only for continuous data)
            self.scaler_pre_ = StandardScaler()
            X_dense_scaled = self.scaler_pre_.fit_transform(X_dense)
            
            n_components_pca = min(128, X_dense_scaled.shape[1])
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
            k_neighbors = min(15, N - 1)
            nn_model = NearestNeighbors(n_neighbors=k_neighbors + 1)  # +1 because kneighbors includes self
            nn_model.fit(X_scaled)
            knn_dists, knn_indices = nn_model.kneighbors(X_scaled)
            
            # Drop self-column
            knn_dists = knn_dists[:, 1:]
            knn_indices = knn_indices[:, 1:]
            
            # Compute exponential weights for distance: exp(-dist)
            # This gives higher weights to closer neighbors
            knn_weights = np.exp(-knn_dists)
            
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
            vol_weight=self.vol_weight
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
        dataset = TensorDataset(
            torch.tensor(X_scaled, dtype=torch.float32),
            torch.arange(N)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        best_loss = float('inf')
        epochs_without_improvement = 0
        best_state = None

        # Training loop
        self.module_.train()
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            for x_batch, idx_batch in dataloader:
                x_batch = x_batch.to(self.device_)
                optimizer.zero_grad()
                loss, z_batch_par = self.module_.compute_loss(x_batch)

                if self.graph_weight > 0:
                    # Graph loss: for each point, pick one random kNN neighbor and attract
                    rand_k = torch.randint(k_neighbors, (len(idx_batch),))
                    neigh_idx = knn_indices_tensor[idx_batch, rand_k]
                    neigh_weights = knn_weights_tensor[idx_batch, rand_k].to(self.device_)
                    
                    x_neigh = X_scaled_tensor[neigh_idx].to(self.device_)
                    
                    # Only run the flow on the neighbors (z_batch_par is already computed)
                    z_neigh, _ = self.module_.inverse_flow(x_neigh)
                    z_neigh_par = z_neigh[:, :self.module_.n_components]
                    
                    # Attractive: Student-t log-loss pulls kNN neighbors together
                    dist_sq = torch.sum((z_batch_par - z_neigh_par) ** 2, dim=-1)
                    loss_attract = torch.mean(neigh_weights * torch.log1p(dist_sq))

                    loss_graph = loss_attract
                    # Repulsive: push randomly sampled non-neighbors apart (UMAP negative sampling).
                    # Without this, different digit classes have no force separating them —
                    # attraction alone collapses everything into one blob.
                    if self.repulsion_weight > 0:
                        neg_idx = torch.randint(N, (len(idx_batch) * self.negative_sample_rate,))
                        x_neg = X_scaled_tensor[neg_idx].to(self.device_)
                        z_neg, _ = self.module_.inverse_flow(x_neg)
                        z_neg_par = z_neg[:, :self.module_.n_components]
                        
                        z_neg_par = z_neg_par.view(len(idx_batch), self.negative_sample_rate, self.n_components)
                        z_batch_par_exp = z_batch_par.unsqueeze(1)
                        dist_sq_neg = torch.sum((z_batch_par_exp - z_neg_par) ** 2, dim=-1)
                        
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
            
            # Early Stopping (monitoring the total training loss)
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
            
        # Get parallel components
        return self.transform(X)

    def fit(self, X, y=None):
        """Fit the model to X without returning representations."""
        self.fit_transform(X, y)
        return self

    def transform(self, X):
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
                z_parallel_list.append(z[:, :self.n_components].cpu().numpy())
                
        return np.concatenate(z_parallel_list, axis=0)

    def inverse_transform(self, z):
        """
        Takes z in R^{N x 2}, pads with zeros, runs inverse flow + inverse scaling,
        and returns generated X_gen in R^{N x D}.
        """
        self.module_.eval()
        z_tensor = torch.tensor(z, dtype=torch.float32)
        dataset = TensorDataset(z_tensor)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
        
        x_gen_scaled_list = []
        with torch.no_grad():
            for batch in dataloader:
                z_batch = batch[0].to(self.device_)
                N_batch = z_batch.shape[0]
                z_perp_zero = torch.zeros((N_batch, self.D_scaled_ - self.n_components), device=self.device_)
                z_padded = torch.cat([z_batch, z_perp_zero], dim=-1)
                
                x_gen_batch, _ = self.module_.forward_flow(z_padded)
                x_gen_scaled_list.append(x_gen_batch.cpu().numpy())
                
        x_gen_scaled = np.concatenate(x_gen_scaled_list, axis=0)
        x_gen_pca = x_gen_scaled * self.global_scale_
        x_gen_pre = self.pca_.inverse_transform(x_gen_pca)
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
