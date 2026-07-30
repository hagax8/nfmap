import numpy as np
import pandas as pd
import json
from nfmap import NFMap

# Just load a few points to test if transform matches fit_transform
X = np.random.randint(0, 2, size=(100, 2048)).astype(np.float32)
y = np.random.randn(100, 1).astype(np.float32)

model = NFMap(n_components=2, max_epochs=10, random_state=42)
Z = model.fit_transform(X, y_props=y)
Z_pred = model.transform(X)

print("Max difference:", np.abs(Z - Z_pred).max())
