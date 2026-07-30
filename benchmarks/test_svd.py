from sklearn.decomposition import TruncatedSVD
import numpy as np

X = np.random.randint(0, 2, size=(10, 20)).astype(np.float32)
svd = TruncatedSVD(n_components=2)
svd.fit(X)

print("is_binary:", (X.max() <= 1 and X.min() >= 0))
try:
    print("mean_:", svd.mean_)
except Exception as e:
    print("error:", type(e).__name__)
