"""B14: is the flow's exact log-likelihood a usable out-of-distribution signal?

Why this is worth re-running. The paper previously tested the exact density once
(old C1) and dropped it at AUC=0.53. But that experiment scored density on DUD-E
actives vs. property-matched decoys - a *virtual screening* task. DUD-E decoys
are constructed to match actives on molecular weight, logP and so on, which puts
them in the same region of chemical space on purpose. A density model should
assign them similar likelihood; 0.53 was arguably the correct answer to a
question density was never meant to answer.

Out-of-distribution detection is the question density is actually for: is this
point from the training distribution at all? That was never tested.

Two bugs fixed since also bear directly on the density:
  * prior_logits (the GMM mixture weights) were trainable regardless of the
    fixed-grid setting, and at production scale collapsed to ~9% of components
    carrying real weight. A mixture whose weights have collapsed is a badly
    miscalibrated density, and log p(x) is read straight off it.
  * the internal PCA could request more components than samples.

Protocol - digits, three OOD constructions that fail in different ways:
  1. novel classes: train on 0-4, score held-out 0-4 (ID) against 5-9 (OOD).
     The standard novelty-detection setup; OOD points are real images with
     genuinely unseen structure.
  2. feature permutation: shuffle each pixel column across the batch. Marginals
     preserved, joint structure destroyed - a point that is individually
     plausible in every coordinate but jointly impossible.
  3. gaussian noise: matched mean/std, no structure at all. The easy case; if a
     signal cannot do this one it cannot do anything.

The off-manifold residual ||z_perp|| was also tested and is not reported: it
works (mean AUC 0.920 across these cells) but is consistently beaten by the
exact density (0.955), and it is a heuristic where the density is the
principled quantity. One native signal, not two.

Scored against three signals, so a win is a win over real alternatives:
  * NFMap log-likelihood        - the thing under test (score_samples)
  * kNN distance in input space - the standard non-parametric OOD baseline
  * PCA reconstruction error    - the linear "gatekeeper" the novelty notes
                                  describe as the SVD-pipeline OOD signal
LMDT is deliberately NOT included. It measures map distortion - "can this
point's position be trusted" - which is a different question from distribution
membership, so scoring it here would repeat the category error the old
applicability-domain section made. (It also cannot be compared fairly on the
permuted set: that corruption is lmdt_head's own training target.)

Usage:
    python b14_density_ood.py
    python b14_density_ood.py --seeds 42
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, ".")
from common.data import load_dataset
from common.io import RESULTS_ROOT
from run_ablation import DEFAULTS

from nfmap import NFMap

EXPERIMENT_ID = "B14"
KNN_K = 5


def make_ood(kind, X_id, rng):
    """An OOD set matched in shape to X_id, built to fail in a specific way."""
    if kind == "permuted":
        X = X_id.copy()
        for j in range(X.shape[1]):
            X[:, j] = X[rng.permutation(len(X)), j]
        return X
    if kind == "noise":
        return rng.normal(X_id.mean(), X_id.std(), size=X_id.shape).astype(np.float32)
    raise ValueError(kind)


def auc(score_id, score_ood):
    """ROC-AUC for flagging OOD. Scores are oriented so higher = more OOD."""
    y = np.r_[np.zeros(len(score_id)), np.ones(len(score_ood))]
    return float(roc_auc_score(y, np.r_[score_id, score_ood]))


def signals(model, pca, nn, X):
    """Every OOD signal for one batch, each oriented so higher = more OOD."""

    out = {
        # negated: low likelihood = out of distribution
        "nfmap_loglik": -model.score_samples(X),
        "knn_distance": nn.kneighbors(X)[0].mean(1),
        "pca_recon_error": np.linalg.norm(X - pca.inverse_transform(pca.transform(X)), axis=1),
    }

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="42,0,1")
    ap.add_argument("--datasets",
                    default="digits,wine,cancer,iris,moons,circles,"
                            "swiss_roll,s_curve,severed_sphere,"
                            "bace,bbbp,dude,drd2,egfr,esol")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    out_dir = os.path.join(RESULTS_ROOT, EXPERIMENT_ID)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for ds in [d.strip() for d in args.datasets.split(",")]:
      data = load_dataset(ds, seed=42)
      X_all, y_all = data["X"], data["y"]
      classes = np.unique(y_all)
      # Novel-class split: hold out the upper half of the classes where there
      # are enough of them, otherwise just the top class. Skipped for binary
      # and continuous-target data - with two classes "the other class" is not
      # out-of-distribution, it is the same chemical space with a different
      # label, which is the category error the old DUD-E density test made.
      if data["is_continuous"] or len(classes) < 3:
          cutoff = None
      elif len(classes) >= 6:
          cutoff = classes[len(classes) // 2]
      else:
          cutoff = classes[-1]
      print(f"\n[{EXPERIMENT_ID}] {ds}: {X_all.shape}, {len(classes)} classes"
            f"{'' if cutoff is not None else '  (novel-class split skipped)'}")
      for seed in seeds:
        rng = np.random.RandomState(seed)

        # --- 1. novel classes: hold out the upper half of the classes --------
        if cutoff is None:
            s_id = s_ood = None
        else:
         seen = y_all < cutoff
         X_seen, y_seen = X_all[seen], y_all[seen]
         X_tr, X_te = train_test_split(X_seen, test_size=0.3, random_state=seed)
         X_novel = X_all[~seen]

         model = NFMap(**{**DEFAULTS, "random_state": seed})
         model.fit(X_tr)
         pca = PCA(n_components=2, random_state=seed).fit(X_tr)
         nn = NearestNeighbors(n_neighbors=KNN_K).fit(X_tr)

         s_id = signals(model, pca, nn, X_te)
         s_ood = signals(model, pca, nn, X_novel)
         for name in s_id:
             a = auc(s_id[name], s_ood[name])
             rows.append(dict(dataset=ds, seed=seed, ood="novel_classes", signal=name, auc=a))
             print(f"  {ds:<7} seed={seed} novel_classes   {name:<16} AUC={a:.3f}", flush=True)

        # --- 2 & 3. corruption OOD, model trained on everything -------------
        X_tr2, X_te2 = train_test_split(X_all, test_size=0.3, random_state=seed)
        model2 = NFMap(**{**DEFAULTS, "random_state": seed})
        model2.fit(X_tr2)
        pca2 = PCA(n_components=2, random_state=seed).fit(X_tr2)
        nn2 = NearestNeighbors(n_neighbors=KNN_K).fit(X_tr2)
        s_id2 = signals(model2, pca2, nn2, X_te2)

        for kind in ("permuted", "noise"):
            s_ood2 = signals(model2, pca2, nn2, make_ood(kind, X_te2, rng))
            for name in s_id2:
                a = auc(s_id2[name], s_ood2[name])
                rows.append(dict(dataset=ds, seed=seed, ood=kind, signal=name, auc=a))
                print(f"  {ds:<7} seed={seed} {kind:<15} {name:<16} AUC={a:.3f}", flush=True)

        with open(os.path.join(out_dir, "density_ood.json"), "w") as f:
            json.dump(rows, f, indent=2)

    print(f"\n  -> {out_dir}/density_ood.json")


if __name__ == "__main__":
    main()
