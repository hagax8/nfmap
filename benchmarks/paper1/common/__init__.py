import os

# Every Paper 1 fit runs on CPU. NFMap otherwise picks MPS on a Mac, and MPS
# fits are not reproducible run-to-run even at a fixed seed (Metal reductions
# have no fixed summation order). That is not cosmetic here: swiss_roll at
# sigma_perp=0.01, grid_bounds=12, seed 42 unrolls on one MPS run (kNN R^2
# 0.998) and folds on CPU (0.487), so a sweep fitted on MPS and a figure
# re-fitted on CPU showed different maps for the same config. CPU is
# deterministic and, at these sizes (~600 points), also ~3x faster.
# setdefault so NFMAP_DEVICE can still be overridden for one-off runs.
os.environ.setdefault("NFMAP_DEVICE", "cpu")

# CPU fits are only bit-reproducible at a fixed thread count: the number of
# threads sets how reductions are split, hence their summation order. Pinned
# to 1, matching B23 (Table 1). Must be set before torch is imported, so every
# script imports `common` before `nfmap`/`torch`.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
