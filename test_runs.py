import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from nfmap import NFMap
import warnings
warnings.filterwarnings('ignore')

X, color = datasets.make_s_curve(n_samples=1000, random_state=42)

print("--- Running without canonical regularization ---")
model_no = NFMap(n_components=2, use_canonical=False, random_state=42)
Z_no = model_no.fit_transform(X)
var_no = np.var(Z_no, axis=0)
print("Variance (no canonical):", var_no)

print("\n--- Running with canonical regularization ---")
model_yes = NFMap(n_components=2, use_canonical=True, random_state=42)
Z_yes = model_yes.fit_transform(X)
var_yes = np.var(Z_yes, axis=0)
print("Variance (with canonical):", var_yes)

# Save plots to see the differences
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(Z_no[:, 0], Z_no[:, 1], c=color, cmap='viridis')
axes[0].set_title(f"NFMap (use_canonical=False)\nVar: {var_no}")
axes[0].axis('tight')

axes[1].scatter(Z_yes[:, 0], Z_yes[:, 1], c=color, cmap='viridis')
axes[1].set_title(f"NFMap (use_canonical=True)\nVar: {var_yes}")
axes[1].axis('tight')

plt.tight_layout()
plt.savefig('comparison_plot.png')
print("\nSaved comparison_plot.png")
