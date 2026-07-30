import json
import numpy as np
import onnxruntime as ort
import sys

# 1. Load the map.json to get the ground truth Z for the first molecule
with open("/Users/helenagaspar/Documents/navifold/navifold/public/data/drd2_morgan/map.json", "r") as f:
    map_data = json.load(f)
mol_data = map_data["molecules"][0]
smiles = mol_data["smiles"]
Z_true = np.array([mol_data["x"], mol_data["y"]], dtype=np.float32)
print("Molecule:", smiles)
print("Z_true:", Z_true)

# 2. Re-project it using the ONNX model
from rdkit import Chem
from rdkit.Chem import AllChem

mol = Chem.MolFromSmiles(smiles)
fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
fp_arr = np.array(fp, dtype=np.float32).reshape(1, -1)

with open("/Users/helenagaspar/Documents/navifold/navifold/public/data/drd2_morgan/model/svd_components.json", "r") as f:
    svd_data = json.load(f)
comps = np.array(svd_data["data"], dtype=np.float32).reshape(128, 2048)
mean = np.array(svd_data["mean"], dtype=np.float32)

fp_centered = fp_arr - mean
x_pca = fp_centered.dot(comps.T).astype(np.float32)

with open("/Users/helenagaspar/Documents/navifold/navifold/public/data/drd2_morgan/model/global_scale.json", "r") as f:
    global_scale = json.load(f)["global_scale"]

x_scaled = x_pca / global_scale

session = ort.InferenceSession("/Users/helenagaspar/Documents/navifold/navifold/public/data/drd2_morgan/model/flow.onnx", providers=['CPUExecutionProvider'])
Z_pred = session.run(None, {"x_scaled": x_scaled.astype(np.float32)})[0].flatten()

print("Z_pred:", Z_pred)
print("Error:", np.abs(Z_true - Z_pred))
