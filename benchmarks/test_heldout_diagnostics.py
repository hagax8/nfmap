import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from nfmap import NFMap

def run_heldout_experiment(X, y, dataset_name, is_continuous=False):
    print(f"\n--- Running Held-out Experiment on {dataset_name} ---")
    
    # Split into 80% train, 20% held-out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Train size: {len(X_train)} | Held-out size: {len(X_test)}")
    
    # Train NFMap
    nfmap = NFMap(
        n_components=2,
        use_canonical=False,
        complexity='low',  # low is enough for these simple datasets
        random_state=42,
        max_epochs=200,
        verbose=False
    )
    
    print("Training NFMap...")
    if is_continuous:
        Z_train = nfmap.fit_transform(X_train, y_props=y_train.reshape(-1, 1))
    else:
        Z_train = nfmap.fit_transform(X_train, y=y_train)
        
    print("Projecting held-out data...")
    Z_test = nfmap.transform(X_test)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    
    if is_continuous:
        # Plot train
        scatter_train = plt.scatter(Z_train[:, 0], Z_train[:, 1], c=y_train, cmap='viridis', 
                                  alpha=0.3, label='Train', s=30, edgecolors='none')
        # Plot test
        scatter_test = plt.scatter(Z_test[:, 0], Z_test[:, 1], c=y_test, cmap='viridis', 
                                 alpha=1.0, label='Held-out (Projected)', s=80, marker='*', edgecolors='black')
        plt.colorbar(scatter_train, label='Continuous Value')
    else:
        # Plot train
        for cls in np.unique(y_train):
            mask_train = (y_train == cls)
            plt.scatter(Z_train[mask_train, 0], Z_train[mask_train, 1], 
                      alpha=0.2, label=f'Train Class {cls}', s=30, edgecolors='none')
            
        # Plot test (using distinct marker and black edge for visibility)
        plt.gca().set_prop_cycle(None) # reset color cycle to match test colors to train colors
        for cls in np.unique(y_test):
            mask_test = (y_test == cls)
            plt.scatter(Z_test[mask_test, 0], Z_test[mask_test, 1], 
                      alpha=1.0, label=f'Held-out Class {cls}', s=100, marker='*', edgecolors='black')
                      
    plt.title(f'NFMap Projection of Held-Out Data: {dataset_name}')
    plt.xlabel('Z1')
    plt.ylabel('Z2')
    
    # Put legend outside
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    artifact_dir = "/Users/helenagaspar/.gemini/antigravity-ide/brain/5a5c2b02-05fb-4912-88fe-1fbcff933d0e"
    img_filename = f"heldout_{dataset_name.replace(' ', '_').lower()}.png"
    img_path = os.path.join(artifact_dir, img_filename)
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return img_path

def main():
    report_lines = ["# NFMap Held-Out Projection Diagnostics\n"]
    report_lines.append("This report demonstrates how `NFMap` projects unseen (held-out) data onto an existing manifold for various standard datasets. The training data is plotted as faded circles, and the held-out projected data is plotted as solid stars with black edges.\n")
    
    # 1. Moons Dataset
    X_moons, y_moons = datasets.make_moons(n_samples=1000, noise=0.1, random_state=42)
    img_path = run_heldout_experiment(X_moons, y_moons, "Moons Dataset", is_continuous=False)
    report_lines.append("## 1. Moons Dataset (2D Non-linear classes)")
    report_lines.append(f"![Moons]({img_path})\n")
    
    # 2. Swiss Roll Dataset
    X_swiss, y_swiss = datasets.make_swiss_roll(n_samples=1500, noise=0.1, random_state=42)
    img_path = run_heldout_experiment(X_swiss, y_swiss, "Swiss Roll", is_continuous=True)
    report_lines.append("## 2. Swiss Roll (3D continuous manifold)")
    report_lines.append(f"![Swiss Roll]({img_path})\n")
    
    # 3. Digits Dataset
    digits = datasets.load_digits()
    X_digits, y_digits = digits.data, digits.target
    img_path = run_heldout_experiment(X_digits, y_digits, "Digits Dataset", is_continuous=False)
    report_lines.append("## 3. Digits Dataset (64D distinct classes)")
    report_lines.append(f"![Digits]({img_path})\n")
    
    # 4. S-Curve Dataset
    X_scurve, y_scurve = datasets.make_s_curve(n_samples=1500, noise=0.1, random_state=42)
    img_path = run_heldout_experiment(X_scurve, y_scurve, "S-Curve", is_continuous=True)
    report_lines.append("## 4. S-Curve (3D continuous manifold)")
    report_lines.append(f"![S-Curve]({img_path})\n")
    
    artifact_dir = "/Users/helenagaspar/.gemini/antigravity-ide/brain/5a5c2b02-05fb-4912-88fe-1fbcff933d0e"
    report_path = os.path.join(artifact_dir, "heldout_diagnostics_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
        
    print(f"\nDone! Report written to {report_path}")

if __name__ == "__main__":
    main()
