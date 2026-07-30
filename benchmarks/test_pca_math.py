from sklearn.decomposition import PCA
import numpy as np

np.random.seed(42)
X = np.random.randint(0, 2, size=(100, 2048)).astype(np.float32)
pca = PCA(n_components=128)
pca.fit(X)

X_new = X[0:1]
z_sklearn = pca.transform(X_new)

# JS implementation
mean = pca.mean_
comps = pca.components_
z_manual = (X_new - mean).dot(comps.T)

print("Max diff:", np.abs(z_sklearn - z_manual).max())
