"""Result persistence for Paper 1 experiments.

Implements the layout mandated by nfmap_paper_experiments.md's "Save All Map
Data for Post-Hoc Plotting" policy:

    results/<experiment_id>/<dataset>_<method>_<params>/
        embeddings.npz    # Z, y, prior_means (if any)
        metrics.json       # all computed metric scores
        config.json         # exact hyperparameters used

config.json/metrics.json are small and meant to be committed (they ARE the
paper's numeric results). embeddings.npz is not committed (repo-wide
*.npz gitignore, same convention already used for chem_kernel's fingerprint
caches) — it's a local convenience so make_figures.py can re-plot without
re-running training, not the long-term record.
"""
import json
import os
import numpy as np

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "results")


def _slugify(value):
    return str(value).replace(" ", "").replace(".", "p").replace("-", "neg")


def run_dir(experiment_id, dataset, method_label, root=RESULTS_ROOT):
    d = os.path.join(root, experiment_id, f"{dataset}_{method_label}")
    os.makedirs(d, exist_ok=True)
    return d


def save_run(experiment_id, dataset, method_label, config, metrics, Z, y=None, prior_means=None, root=RESULTS_ROOT):
    d = run_dir(experiment_id, dataset, method_label, root=root)

    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    npz_kwargs = {"Z": Z}
    if y is not None:
        npz_kwargs["y"] = y
    if prior_means is not None:
        npz_kwargs["prior_means"] = prior_means
    np.savez(os.path.join(d, "embeddings.npz"), **npz_kwargs)

    return d


def load_run(experiment_id, dataset, method_label, root=RESULTS_ROOT):
    d = run_dir(experiment_id, dataset, method_label, root=root)
    with open(os.path.join(d, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(d, "metrics.json")) as f:
        metrics = json.load(f)
    embeddings = np.load(os.path.join(d, "embeddings.npz"), allow_pickle=True)
    return config, metrics, embeddings


def iter_runs(experiment_id, root=RESULTS_ROOT):
    exp_dir = os.path.join(root, experiment_id)
    if not os.path.isdir(exp_dir):
        return
    for name in sorted(os.listdir(exp_dir)):
        d = os.path.join(exp_dir, name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "metrics.json")):
            yield name, d
