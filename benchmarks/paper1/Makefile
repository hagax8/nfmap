# NFMap Paper 1 — reproduction commands.
#
# Per nfmap_paper_experiments.md's reproducibility requirement: every number/
# figure in the paper must be reproducible from a single documented command.
# Run from this directory (benchmarks/paper1/).
#
# Uses the base conda env, which has torch/umap-learn/ugtm/datamol all
# together (the imolgen env used elsewhere in this repo for chem_kernel work
# has torch+rdkit but not umap/ugtm — see planning_docs/nfmap_paper2_chemistry_scale_plan.md
# for that env's own scripts).
PYTHON := /Users/helenagaspar/anaconda3/bin/python

.PHONY: ablations fig1 a1 a2 b1 b2 b3 b4 b4b b4c b5 b9 b11 b14

ablations: b1 b2 b3 b4 b4b b4c b5 b9 b11 b14

# Fig 1: architecture diagram (Flow -> 2D map -> GMM prior). Not data-driven -
# no results/ dependency, just an illustrative schematic, built as a script
# like everything else here rather than a hand-drawn image dropped into the repo.
# Fig 1b: companion geometric-intuition diagram for z_parallel/z_perp - exact
# orthogonal complements in latent space, the design target (not guaranteed
# without the optional isometric loss, B7/Sec 7.8) in data space.
fig1:
	$(PYTHON) fig1_architecture_diagram.py
	$(PYTHON) fig1b_manifold_geometry.py
	$(PYTHON) fig1c_manifold_thickness.py

# B1: Log-Det Dimension Scaling, Novelty #1. vol_weight=0.0 (scaling off) vs.
# 1.0 (on, default) on DUD-E EGFR (136D). No nfmap.py changes needed - the
# Jacobian-column-norm histogram (originally thought to be blocked on new
# training-loop instrumentation) is computed post-hoc on the fitted model
# using its existing public attributes. See b1_logdet_scaling.py's docstring.
b1:
	$(PYTHON) b1_logdet_scaling.py --dataset dude --seed 42

# A1: Standard Dimensionality Reduction Benchmarks — NFMap vs. GTM/t-SNE/UMAP
# across 9 standard datasets. Reuses benchmarks/nfmap_diagnostics.py (which
# already implements exactly this comparison + full metric suite) via its
# --save-results flag, rather than duplicating dataset-loading/baseline-fitting
# logic in a new script. use_canonical is left at its default (False) for all
# 9 datasets here, including the 5 manifold ones the doc originally scoped for
# use_canonical=True — B7's ablation found the effect too marginal to be worth
# forking this comparison over, so this benchmark stays at NFMap's plain
# defaults throughout for a cleaner, single-config table.
# --prior-layout square (swiss_roll/s_curve/severed_sphere only): a visual
# check (2026-09-16) found the default hexagonal grid gave each unrolled
# fragment a visibly noisier, less monotonic color gradient than a square
# grid did on the same data - square reads as more cleanly "unrolled" by eye,
# which is what these 3 datasets exist to demonstrate.
# digits/wine/cancer/iris/moons/circles stay on hexagonal, the better choice
# for cluster-style (non-manifold) data.
# --sigma-perp 1.0 --grid-bounds 12.0 (swiss_roll/s_curve/severed_sphere only).
# REPLACES an earlier --sigma-perp 0.05, whose justification has been retracted:
# it rested on a single-seed A/B against the default 10.0 (swiss_roll DistCor
# 0.706->0.832, s_curve 0.710->0.758, severed_sphere 0.762->0.771), and those
# gains largely evaporate under seeding. Worse, that configuration left 21.7-
# 27.1% of points outside the visible grid, because grid_bounds was never swept
# jointly with sigma_perp for these three (B2E).
#
# These are 3D inputs, so d_scaled=3 with n_components=2 leaves d_perp=1. That
# inverts the chemistry behaviour: squeezing z_perp only enriches z_par when
# there is a reservoir to transfer, and with ONE off-manifold dimension there
# is none - crushing sigma_perp instead destroys the dimension legitimately
# holding the manifold's thickness, which reappears in the map as distortion.
# So where chemistry wants sigma_perp LOW (B2D: 0.003), these want it moderate:
# B2E's sweep at grid_bounds=12 peaks at 0.1 (mean DistCor 0.732) and falls off
# both sides (0.644 at 0.001, 0.714 at 10.0).
#
# 1.0 rather than the argmax 0.1, for two independent reasons:
#   (a) a priori - the input is standardised, so residual variance per dimension
#       is ~1, and sigma_perp=1.0 models z_perp at the data's own scale, neither
#       squeezed nor inflated. A principled default, not a fitted one.
#   (b) it is what this paper's stated selection rule picks: largest sigma_perp
#       within 1 sd of the best DistCor subject to <=5% containment. 1.0 scores
#       0.723 vs the peak's 0.732 (inside the +/-0.016-0.025 seed sd) with
#       strictly better containment (2.1% mean / 5.1% worst, vs 4.7%/6.1%).
#       The argmax of a noisy curve is upward biased - the error that produced
#       B2's retracted high-end "erosion".
# grid_bounds=12.0 matches B2D's chemistry canvas, so the whole paper reports
# one canvas. sigma_perp=1.0 is what makes 12 viable here: containment at
# grid_bounds=12 runs 14.1% at sigma_perp=0.001 down to 2.1% at 1.0.
# Three seeds per cell for the headline comparison. NOTE: an 8-seed stability
# check at one operating point found a per-fit DistCor spread of ~0.15 (sd
# ~0.05), so 3 seeds characterises the mean but understates the spread by
# roughly 2x; close margins must not be claimed on 3 seeds alone.
SEEDS := 42 0 1

a1:
	for s in $(SEEDS); do \
		for d in digits wine cancer iris moons circles; do \
			$(PYTHON) ../nfmap_diagnostics.py --dataset $$d --save-results --seed $$s || exit 1; \
		done; \
		for d in swiss_roll s_curve severed_sphere; do \
			$(PYTHON) ../nfmap_diagnostics.py --dataset $$d --save-results --seed $$s \
				--prior-layout square --sigma-perp 1.0 --grid-bounds 12.0 || exit 1; \
		done; \
	done

# A2: Chemistry-Specific Benchmarks — NFMap vs. GTM/t-SNE/UMAP on real
# chemistry datasets: MoleculeNet BBBP/BACE/ESOL, DUD-E EGFR (partial, ~600
# actives+decoys - see the doc's open DUD-E-scope question), and the two
# ChEMBL target-activity sets (DRD2/EGFR) added as plain-baseline datasets per
# Part A2's note - active/inactive via pIC50>=7.0 as the label there (not
# Murcko scaffold id - that broke macro-AUC, see the doc). General
# classification AUC (KNN-on-embedding) is implemented and populated for all
# 6 datasets here; DUD-E's own *virtual screening* AUC (log-likelihood-ranked)
# is a different metric, scoped to C1 below, not this target.
# TWO NFMap configurations per dataset, both at grid_bounds=12, per B2D's joint
# sigma_perp x grid_bounds sweep:
#   global (sigma_perp=0.003) - the best shared setting keeping <=5% of points
#       inside grid_bounds. Mean DistCor 0.653 across the six, ahead of GTM on
#       all six (mean gain +0.150).
#   local  (sigma_perp=10.0)  - NFMap's shipped default. Mean DistCor 0.215 but
#       mean T/C F1 0.930 vs the global end's 0.854.
# Reporting both makes Table 1 show the trade-off rather than quietly pick one end,
# which is the honest presentation given neither dominates. grid_bounds is held
# at 12 for BOTH so sigma_perp is the only variable between them - varying the
# canvas too would reintroduce exactly the confound B2D was built to untangle,
# and at sigma_perp=10 the canvas barely matters anyway (DistCor +0.014, T/C
# -0.009 moving 6 -> 12, both inside seed noise, while containment improves
# from 2.4-6.8% to 0.0% on all six).
a2:
	for s in $(SEEDS); do \
		for d in bbbp bace esol drd2 egfr; do \
			$(PYTHON) ../nfmap_diagnostics.py --dataset $$d --save-results \
				--experiment-id A2 --seed $$s --grid-bounds 12.0 \
				--nfmap-variants "global=0.003,local=10.0" || exit 1; \
		done; \
	done

# C1 (Density as a QSAR Signal / Virtual Screening AUC) — DROPPED from the
# paper (decided 2026-09-15): run once, AUC=0.53 (barely above chance) for
# held-out DUD-E EGFR actives vs. decoys ranked by exact manifold density.
# Doesn't support the "density as QSAR signal" claim and doesn't touch any of
# Paper 1's core architecture claims, so dropping it costs nothing. Script
# kept for reproducibility (e.g. if revisited on a less adversarial decoy
# set later - DUD-E decoys are deliberately property-matched to be hard):
# `python c1_density_qsar_signal.py --seed 42 --n-decoys 2000` (no longer a
# Makefile target).

# B2: Thick Manifold (sigma_perp ablation), Novelty #2. Datasets: digits,
# dude, bbbp (2026-09-16: expanded from digits-only after a scope concern -
# one dataset's plateau chart doesn't carry a claim this central, and the
# original sweep never tested anything close to Brehmer et al.'s actual
# razor-thin regime, ~1e-4). digits = general baseline (shares a dataset with
# B3/B4/B5/B9); dude = the one chemistry dataset with real class separation
# (silhouette 0.27, see A2); bbbp = a second chemistry dataset with weak
# separation, to check the pattern isn't dude-specific. Sweep now spans
# 0.0001-20.0 (was 0.01-20.0) so the low end actually reaches toward
# Brehmer's regime instead of stopping well short of it.
# swiss_roll/s_curve/severed_sphere added 2026-09-17: same sweep, fixed at
# prior_layout=square (matching A1's manifold-dataset config), to replace
# the single before/after data point that A1's own sigma_perp=0.05 choice
# was originally based on with a real sweep, and to directly compare the
# "z_perp has ~1 real dimension" regime against the "z_perp has 30-60+ real
# dimensions" regime above - the two behave genuinely differently, not just
# at a different scale. See nfmap_paper_experiments.md's B2 section for the
# two-regime table.
b2:
	$(PYTHON) run_ablation.py --experiment-id B2 --dataset digits \
		--param sigma_perp --values 0.0001,0.001,0.01,0.05,0.5,1.0,5.0,10.0,20.0,50.0,100.0,500.0,1000.0,10000.0
	$(PYTHON) run_ablation.py --experiment-id B2 --dataset dude \
		--param sigma_perp --values 0.0001,0.001,0.01,0.05,0.5,1.0,5.0,10.0,20.0,50.0,100.0,500.0,1000.0,10000.0
	$(PYTHON) run_ablation.py --experiment-id B2 --dataset bbbp \
		--param sigma_perp --values 0.0001,0.001,0.01,0.05,0.5,1.0,5.0,10.0,20.0,50.0,100.0,500.0,1000.0,10000.0
	$(PYTHON) run_ablation.py --experiment-id B2 --dataset swiss_roll --fixed prior_layout=square \
		--param sigma_perp --values 0.0001,0.001,0.01,0.05,0.5,1.0,5.0,10.0,20.0,50.0,100.0,500.0,1000.0,10000.0
	$(PYTHON) run_ablation.py --experiment-id B2 --dataset s_curve --fixed prior_layout=square \
		--param sigma_perp --values 0.0001,0.001,0.01,0.05,0.5,1.0,5.0,10.0,20.0,50.0,100.0,500.0,1000.0,10000.0
	$(PYTHON) run_ablation.py --experiment-id B2 --dataset severed_sphere --fixed prior_layout=square \
		--param sigma_perp --values 0.0001,0.001,0.01,0.05,0.5,1.0,5.0,10.0,20.0,50.0,100.0,500.0,1000.0,10000.0

# B3: Topological Graph Regularization (graph_weight ablation), Novelty #4.
b3:
	$(PYTHON) run_ablation.py --experiment-id B3 --dataset digits \
		--param graph_weight --values 0.0,1.0,5.0,10.0

# B4: Repulsion strength (repulsion_weight), Novelty #4 repulsion arm.
# Sweeps repulsion_weight, NOT negative_sample_rate. The earlier sweep over
# negative_sample_rate found no effect, and that was guaranteed by the code
# rather than an empirical result: the repulsion term is
#   loss_repel = torch.mean(log1p(1/(d^2+eps)))   ... * repulsion_weight
# so the rate only sets how many Monte Carlo samples the mean is taken over -
# a lower-variance estimate of the same expectation, not a stronger force.
# repulsion_weight is the actual strength knob. (UMAP differs: it applies each
# negative as a separate gradient step, so there the count does scale the force.)
b4:
	$(PYTHON) run_ablation.py --experiment-id B4 --dataset digits \
		--param repulsion_weight --values 0.0,0.25,0.5,1.0,2.0,5.0,10.0

# B4B: repulsion ON vs. OFF across every dataset, 3 seeds each. B4 above only
# varies the *amount* of repulsion on one dataset, which turned out to sit
# within run-to-run noise; the live question is whether repulsion helps at all,
# and whether the answer is dataset-dependent. See b4b_repulsion_ablation.py's
# docstring for why this is breadth-with-seeds rather than a deeper sweep.
b4b:
	$(PYTHON) b4b_repulsion_ablation.py

# B5: GMM Grid — single-center vs. learnable vs. fixed, Novelty #5.
b5:
	$(PYTHON) b5_gmm_grid.py --dataset digits

# B7 (Conformal / Isometric Regularization, Novelty #6) — DROPPED from the
# paper (decided 2026-09-13): the effect, even after exposing canonical_weight
# as a real tunable parameter, is real but too marginal (~2.5-3% relative
# DistCor gain) to be worth a figure. The canonical_weight code change itself
# stays in nfmap.py (turns a hardcoded magic number into a real parameter,
# default-preserving) — it's just not featured here. See
# nfmap_paper_experiments.md's B7 section for the numbers that led to this call.

# B9: GMM Prior Weight vs. Grid Containment, Novelty #5 extension. Discovered
# while visually inspecting B3/B5's maps — a large fraction of points land
# outside the visible grid_bounds at the default prior_weight=0.1 (16-60%
# across those runs). NFMap's grid is a documented *soft* boundary (see the
# "NFMap vs. GTM" theoretical note), so some spillover is expected/intentional
# — this quantifies how much, and whether it's tunable without a quality cost.
# grid_bounds pinned explicitly to 3.0 here (not DEFAULTS' new 6.0 — see
# run_ablation.py) so this experiment's own numbers keep meaning "at the old
# default," independent of B2/B3/B4's now-higher default.
b9:
	$(PYTHON) run_ablation.py --experiment-id B9 --dataset digits \
		--param prior_weight --values 0.1,0.3,0.5,1.0,2.0 --fixed grid_bounds=3.0
	$(PYTHON) run_ablation.py --experiment-id B9_gridbounds --dataset digits \
		--param grid_bounds --values 3.0,4.0,5.0,6.0,8.0



# B11: does the thick manifold's "storage vault" hold anything? (Novelty #2)
# B2 only shows that crushing sigma_perp HURTS the map - the absence of harm,
# not the presence of stored information. E1a decomposes prediction into
# z_par / z_perp / both at crushed vs. default sigma_perp (standardised blocks
# + a PCA-matched control, because NFMap is a bijection and a naive version of
# this test is true by construction). E1b is the sharper one: among molecules
# landing in the SAME place on the visible map, does z_perp distance track
# their property difference - the activity-cliff question stated exactly.
b11:
	$(PYTHON) b11_zperp_information.py



# B4C: repulsion_weight dose-response across 9 datasets x 3 seeds. B4 runs the
# same sweep on digits alone; B4B showed the effect is strongly dataset-
# dependent, so the single-dataset curve is not safe to generalise from.
b4c:
	$(PYTHON) b4c_repulsion_sweep.py

# B14: is the exact log-likelihood a usable out-of-distribution signal?
# Replaces the removed LMDT work as the paper's applicability-domain section.
b14:
	$(PYTHON) b14_density_ood.py
