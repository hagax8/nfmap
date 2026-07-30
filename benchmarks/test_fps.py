from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
mol = Chem.MolFromSmiles("Fc1ccc(-c2cncc(CN3CCN(c4ccccc4F)CC3)c2)cc1")
fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
fp_str = "".join(str(x) for x in fp)
print("Py   fp sum:", np.sum(fp))
print("Py   fp head:", fp_str[:100])
