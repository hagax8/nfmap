import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

df = pd.read_csv("/Users/helenagaspar/Documents/nfmap/experiments/chembed_generation/drd2_clean.csv")

fps_list = []
for smi in df["smiles"]:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fps_list.append(np.array(fp, dtype=np.float32))

X_fps = np.array(fps_list, dtype=np.float32)
print("X_fps.dtype:", X_fps.dtype)
print("X_fps.max:", X_fps.max())
print("X_fps.min:", X_fps.min())
print("is_binary:", (X_fps.dtype == bool or (X_fps.max() <= 1 and X_fps.min() >= 0)))
