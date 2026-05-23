import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from nfmap import NFMap
import warnings
warnings.filterwarnings('ignore')

X_scurve, color = datasets.make_s_curve(n_samples=1000, random_state=42)

model = NFMap(n_components=2, use_canonical=True, random_state=42)
# Temporarily modify the class to print loss
orig_fit_transform = model.fit_transform

print("Starting debug training run...")
Z = model.fit_transform(X_scurve)
print("Finished!")

# Let's plot and save it to a file we can read the properties of, or just look at the variance
variance_z = np.var(Z, axis=0)
print(f"Variance of Z: {variance_z}")

