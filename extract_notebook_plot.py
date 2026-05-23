import json
import base64

with open('dimensionality_reduction_comparison.ipynb', 'r') as f:
    nb = json.load(f)

# Find the first code cell output with an image (the S-curve cell is index 2, cell_type: code, execution_count: 3)
count = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        count += 1
        if count == 3: # S-curve cell
            for output in cell.get('outputs', []):
                if 'data' in output and 'image/png' in output['data']:
                    img_data = output['data']['image/png']
                    with open(f'notebook_plot_scurve.png', 'wb') as img_file:
                        img_file.write(base64.b64decode(img_data))
                    print("Successfully saved notebook_plot_scurve.png")
                    break
