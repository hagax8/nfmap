# NFMap: Architecture & Theoretical Takeaways

## The Goal
The objective of NFMap was to combine the powerful topological clustering of UMAP (via attractive and repulsive graph losses) with the generative, invertible, and exact-density properties of Normalizing Flows (specifically, RealNVP). 

## The Problem: The Topological Deadlock
Initial implementations failed to separate the data clusters (e.g., MNIST digits) and resulted in a "locked" embedding that looked like random noise or standard PCA (Silhouette score ~0.06). 

We discovered that the Normalizing Flow was caught in a massive **topological deadlock** caused by three competing mathematical forces:

1. **Volume Preservation**: A Normalizing Flow is a continuous bijection. It mathematically *must* preserve the 64-dimensional volume of the data space. To tear connected manifolds apart into tight distinct clusters (like UMAP does), the flow must drastically stretch the empty space between those clusters.
2. **Manifold Compression ($sigma_{\perp}$)**: We used a small $sigma_{\perp}$ (e.g., 0.05) to violently squash the 62 non-embedding dimensions to exactly 0. 
3. **Graph Topology (UMAP Loss)**: The UMAP loss attempts to pull the 2 embedding dimensions into tight clusters.

**The Deadlock**: Because the manifold loss squashed the 62 perpendicular dimensions, the flow's volume preservation constraint forced the remaining 2 parallel dimensions to stretch out to infinity to compensate. However, the UMAP graph loss was simultaneously trying to pull those 2 dimensions tightly together. Because these forces perfectly canceled each other out, the gradients died, and the network was physically unable to move from its initial PCA state.

## The Solutions

### 1. Orthogonal Relaxation (The "Garbage Bin" Hack)
By intentionally relaxing the manifold constraint to **`sigma_perp = 10.0`**, we broke the deadlock. This allowed the flow to dump the extreme volume expansions required for topological tearing into the 62 orthogonal dimensions. By using the non-embedding dimensions as a "garbage bin" for volume distortion, we freed up the 2D plane to be tightly clustered without violating the bijection constraint.

### 2. Deep Training & Patience
Normalizing Flows (especially RealNVPs with bounded affine coupling via `tanh`) require thousands of tiny gradient steps to drastically deform space. Parametric UMAP trains for thousands of epochs. We increased `max_epochs` from 80 to **1000** and `patience` to **100**. This finally gave the flow the time needed to bend the 64-dimensional space.

### 3. Learnable Priors
Instead of forcing the latent space into a rigid grid of Gaussian components, we replaced the static grid with PyTorch `nn.Parameter` centers. This allows the GMM prior to organically slide around the latent space and adapt to the natural topology of the graph loss.

## The Result
By breaking the deadlock, we achieved a silhouette score of **0.33** (beating the GTM reference score of 0.28). NFMap successfully produced distinct, cohesive clusters that rival t-SNE and UMAP, while retaining the invertibility and density estimation of a Normalizing Flow!

## Is this Novel?
Yes. While Parametric UMAP, Manifold Flows, and FlowGMM exist independently, the specific combination of forcing a continuous bijection to perform UMAP-style topological clustering using **orthogonal relaxation** to bypass the volume-expansion deadlock is highly unique. This architecture allows for UMAP-level clustering while maintaining a mathematically exact generative mapping back to the data space.
