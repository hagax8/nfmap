import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import matplotlib.pyplot as plt

# Import the original RealNVPBlock from nfmap
from nfmap import RealNVPBlock

class TestNFMapModule(nn.Module):
    def __init__(self, d_scaled, n_components=2, n_blocks=10, hidden_dim=256, prior_weight=0.05, sigma_perp=0.05, reg_mode="canonical"):
        super().__init__()
        self.d_scaled = d_scaled
        self.n_components = n_components
        self.prior_weight = prior_weight
        self.sigma_perp = sigma_perp
        self.reg_mode = reg_mode
        
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            mask = torch.zeros(d_scaled)
            if i % 2 == 0:
                mask[:d_scaled // 2] = 1.0
            else:
                mask[d_scaled // 2:] = 1.0
            self.blocks.append(RealNVPBlock(d_scaled, hidden_dim, mask))

    def forward_flow(self, z):
        log_det_total = 0
        x = z
        for block in self.blocks:
            x, log_det = block(x)
            log_det_total += log_det
        return x, log_det_total

    def inverse_flow(self, x):
        log_det_total = 0
        z = x
        for block in reversed(self.blocks):
            z, log_det = block.inverse(z)
            log_det_total += log_det
        return z, log_det_total

    def compute_loss(self, x):
        z, log_det_inv = self.inverse_flow(x)
        z_parallel = z[:, :self.n_components]
        z_perp = z[:, self.n_components:]
        
        loss_prior = self.prior_weight * 0.5 * torch.sum(z_parallel**2, dim=-1)
        loss_manifold = 0.5 / (self.sigma_perp**2) * torch.sum(z_perp**2, dim=-1)
        loss_vol = -log_det_inv
        
        loss = (loss_prior + loss_manifold + loss_vol).mean()
        
        # Pullback Metric calculation
        z_para_req = z_parallel.detach().requires_grad_(True)
        z_perp_zero = torch.zeros_like(z_perp)
        z_padded = torch.cat([z_para_req, z_perp_zero], dim=-1)
        x_gen, _ = self.forward_flow(z_padded)
        
        J = []
        for i in range(self.d_scaled):
            grad_outputs = torch.zeros_like(x_gen)
            grad_outputs[:, i] = 1.0
            grads = torch.autograd.grad(
                outputs=x_gen, 
                inputs=z_para_req, 
                grad_outputs=grad_outputs, 
                create_graph=True, 
                retain_graph=True
            )[0]
            J.append(grads)
        J = torch.stack(J, dim=1) # [B, D, n_components]
        G = torch.bmm(J.transpose(1, 2), J) # [B, n_components, n_components]
        
        # Orthogonality loss
        eye = torch.eye(self.n_components, device=G.device).unsqueeze(0)
        off_diag_mask = 1.0 - eye
        loss_ortho = (torch.abs(G) * off_diag_mask).sum(dim=(1, 2)).mean()
        
        if self.reg_mode == "conformal":
            # Scale difference at the same point (from current nfmap.py)
            diag_G = torch.diagonal(G, dim1=-2, dim2=-1)
            mean_diag = diag_G.mean(dim=-1, keepdim=True)
            loss_diag = torch.abs(diag_G - mean_diag).sum(dim=-1).mean()
        elif self.reg_mode == "isometric_global":
            # Scale difference from global batch mean
            diag_G = torch.diagonal(G, dim1=-2, dim2=-1)
            global_mean = diag_G.mean()
            loss_diag = torch.abs(diag_G - global_mean).mean()
        elif self.reg_mode == "canonical":
            # Canonical: G should be close to Identity (scale = 1 everywhere)
            diag_G = torch.diagonal(G, dim1=-2, dim2=-1)
            loss_diag = torch.abs(diag_G - 1.0).mean()
            
        loss_cmf = 10.0 * loss_ortho + 10.0 * loss_diag
        loss = loss + loss_cmf
        return loss

class TestNFMap(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, prior_weight=0.05, reg_mode="canonical", random_state=42):
        self.n_components = n_components
        self.prior_weight = prior_weight
        self.reg_mode = reg_mode
        self.random_state = random_state

    def fit_transform(self, X, y=None):
        if sp.issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = np.asarray(X, dtype=np.float32)
        
        # PCA alignment
        self.pca_ = PCA(n_components=min(128, X_dense.shape[1]), random_state=self.random_state)
        X_pca = self.pca_.fit_transform(X_dense)
        
        # Scaling
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_pca)
        
        N, D_scaled = X_scaled.shape
        self.D_scaled_ = D_scaled
        
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            
        self.module_ = TestNFMapModule(
            d_scaled=D_scaled,
            n_components=self.n_components,
            prior_weight=self.prior_weight,
            reg_mode=self.reg_mode
        )
        
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.module_.to(self.device_)
        
        optimizer = AdamW(self.module_.parameters(), lr=1e-3, weight_decay=1e-4)
        
        dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=min(256, N), shuffle=True)
        
        self.module_.train()
        for epoch in range(100):
            for batch in dataloader:
                x_batch = batch[0].to(self.device_)
                optimizer.zero_grad()
                loss = self.module_.compute_loss(x_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.module_.parameters(), max_norm=1.0)
                optimizer.step()
                
        return self.transform(X)

    def transform(self, X):
        X_dense = np.asarray(X, dtype=np.float32)
        X_pca = self.pca_.transform(X_dense)
        X_scaled = self.scaler_.transform(X_pca)
        
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

X, color = datasets.make_s_curve(n_samples=1000, random_state=42)

# Experiment 1: Conformal scaling (current code)
print("Running conformal scaling...")
model_conf = TestNFMap(reg_mode="conformal", prior_weight=0.05)
Z_conf = model_conf.fit_transform(X)

# Experiment 2: Canonical (target G = I)
print("Running canonical (isometric to identity)...")
model_can = TestNFMap(reg_mode="canonical", prior_weight=0.05)
Z_can = model_can.fit_transform(X)

# Experiment 3: Canonical with lower prior weight (prior_weight=0.01)
print("Running canonical with prior_weight=0.01...")
model_can_low = TestNFMap(reg_mode="canonical", prior_weight=0.01)
Z_can_low = model_can_low.fit_transform(X)

# Experiment 4: Canonical with prior_weight=0.001
print("Running canonical with prior_weight=0.001...")
model_can_vlow = TestNFMap(reg_mode="canonical", prior_weight=0.001)
Z_can_vlow = model_can_vlow.fit_transform(X)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].scatter(Z_conf[:, 0], Z_conf[:, 1], c=color, cmap='viridis')
axes[0, 0].set_title("Conformal (prior=0.05)")
axes[0, 0].axis('tight')

axes[0, 1].scatter(Z_can[:, 0], Z_can[:, 1], c=color, cmap='viridis')
axes[0, 1].set_title("Canonical (prior=0.05)")
axes[0, 1].axis('tight')

axes[1, 0].scatter(Z_can_low[:, 0], Z_can_low[:, 1], c=color, cmap='viridis')
axes[1, 0].set_title("Canonical (prior=0.01)")
axes[1, 0].axis('tight')

axes[1, 1].scatter(Z_can_vlow[:, 0], Z_can_vlow[:, 1], c=color, cmap='viridis')
axes[1, 1].set_title("Canonical (prior=0.001)")
axes[1, 1].axis('tight')

plt.tight_layout()
plt.savefig('test_isometric_results.png')
print("Saved test_isometric_results.png")
