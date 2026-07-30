import pandas as pd
import numpy as np

df = pd.read_csv("/Users/helenagaspar/Documents/nfmap/experiments/chembed_generation/drd2_clean.csv")
idx = df[df["chembl_id"] == "CHEMBL61682"].index
if len(idx) > 0:
    i = idx[0]
    pts = np.fromfile("/Users/helenagaspar/Documents/navifold/navifold/public/data/drd2_morgan/map/points.bin", dtype=np.float32)
    x = pts[i*2]
    y = pts[i*2+1]
    print(f"Target coord for {i}: x={x:.4f}, y={y:.4f}")
else:
    print("Not found in drd2_clean.csv")
