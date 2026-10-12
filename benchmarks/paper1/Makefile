# NFMap Paper 1: reproduce every table and figure in the paper.
#
# Run from this directory (benchmarks/paper1/). All fits run on CPU with one
# thread (set in common/__init__.py), so results are reproducible run to run.
#
#   make paper        tables + figures from the committed results (minutes)
#   make experiments  re-run every fit behind them (hours; parallelise with -j)
#   make all          both
#
# Committed results are what the paper reports; `make experiments` regenerates
# them with identical numbers on the same software versions. B20 and B21 resume
# from their result files - delete results/B20/scale_study.json or
# results/B21/prior_ablation.json to refit them from scratch.

PYTHON ?= python

.PHONY: all paper tables figures experiments \
        b2d b2e b3b b4c b5 b14 b20 b21 b22 b23

all: experiments paper

paper: tables figures

tables:
	$(PYTHON) make_paper_tables.py

figures:
	$(PYTHON) make_paper_figures.py

experiments: b2d b2e b3b b4c b5 b14 b20 b21 b22 b23

# Fig. 2 and Fig. 4: sigma_perp x grid_bounds on chemistry (B2D) and the 3-D
# manifolds (B2E).
b2d:
	$(PYTHON) b2d_sigma_grid_joint.py
b2e:
	$(PYTHON) b2e_sigma_grid_manifold.py

# Fig. 10: graph attraction (B3B) and repulsion (B4C) weights.
b3b:
	$(PYTHON) b3b_graph_weight_breadth.py
b4c:
	$(PYTHON) b4c_repulsion_sweep.py

# Fig. 11 and the Digits rows of the prior table: single Gaussian vs learned
# mixture vs fixed grid.
b5:
	$(PYTHON) b5_gmm_grid.py

# Fig. 7 and the applicability-domain table.
b14:
	$(PYTHON) b14_density_ood.py

# Fig. 9: scale study on the full ChEMBL sets (single seed, slow). Allowed
# four threads: thread count does not change these fits' results.
b20:
	OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $(PYTHON) b20_scale_study.py

# Table 1 and the settings table: every method swept, three seeds.
b23:
	$(PYTHON) b23_table1_reproducible_sweeps.py

# Prior table (chemistry and manifold rows) and its non-inferiority test. Fits
# the manifolds at the setting Table 1 selects, so it needs B23.
b21: b23
	$(PYTHON) b21_prior_ablation.py

# Maps for Fig. 3 and the S-curve / severed-sphere / swiss-roll figures, at the
# settings Table 1 reports, so it needs B23.
b22: b23
	$(PYTHON) b22_fit_manifold_maps.py
