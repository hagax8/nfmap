import nbformat
import sys

def add_swiss_roll_to_notebook(notebook_path):
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Check if Swiss Roll is already added
    for cell in nb.cells:
        if cell.cell_type == 'markdown' and 'Swiss Roll Dataset' in cell.source:
            print("Swiss Roll dataset already exists in the notebook.")
            return

    # Find the index of the S-curve cell
    s_curve_idx = -1
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and 'make_s_curve' in cell.source:
            s_curve_idx = i
            break

    if s_curve_idx == -1:
        print("Could not find S-curve cell in the notebook.")
        return

    # Create new cells for Swiss Roll
    markdown_cell = nbformat.v4.new_markdown_cell(source="## 1b. Swiss Roll Dataset")
    code_cell = nbformat.v4.new_code_cell(source="X_swiss, color_swiss = datasets.make_swiss_roll(n_samples=1000, random_state=42)\nplot_comparison(X_swiss, color_swiss, 'Swiss Roll Dataset')")

    # Insert cells after the S-curve code cell
    nb.cells.insert(s_curve_idx + 1, markdown_cell)
    nb.cells.insert(s_curve_idx + 2, code_cell)

    # Save the updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print("Successfully added Swiss Roll dataset to the notebook.")

if __name__ == '__main__':
    add_swiss_roll_to_notebook('/Users/helenagaspar/Documents/nfmap/dimensionality_reduction_comparison.ipynb')
