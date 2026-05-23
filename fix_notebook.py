import json

with open('dimensionality_reduction_comparison.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        if 'import seaborn as sns' in source and 'import ugtm' not in source:
            source = source.replace("import seaborn as sns", "import seaborn as sns\nimport ugtm")
            
        if "algorithms = {" in source and "'GTM'" not in source:
            old_str = "'LLE': manifold.LocallyLinearEmbedding(n_neighbors=10, n_components=2, method='standard', random_state=42),"
            new_str = old_str + "\n    'GTM': ugtm.eGTM(m=2, model='means'),"
            source = source.replace(old_str, new_str)
            
        cell['source'] = source.splitlines(True)

with open('dimensionality_reduction_comparison.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
