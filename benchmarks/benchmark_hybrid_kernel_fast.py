"""
benchmark_hybrid_kernel_fast.py
=================================
Item 3 of planning_docs/chembl_scale_kernel_optimization_plan.md's suggested
order of operations: benchmark kNN graph construction time/memory at
increasing scale (10k -> 100k -> ~1M) on real ChEMBL SMILES, before
committing to the full ~2.5-3M run.

Fingerprint computation (RDKit, single-threaded, ~0.75ms/molecule - see
run notes) is run ONCE at the largest N and sliced down for smaller
checkpoints, so the benchmark isolates compute_hybrid_knn_scale's own cost
(bit-packing + PyNNDescent index construction) rather than re-paying
fingerprinting at every scale point.

Usage:
    python benchmark_hybrid_kernel_fast.py [--max-n 400000] [--out results.json]

Data source: experiments/chembed_generation/output/fit_subsample_700k.parquet
(700k real ChEMBL SMILES, already deduplicated/cleaned for a prior fit run -
see planning_docs/chembl_scale_kernel_optimization_plan.md for context).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "chem_kernel"))
from hybrid_kernel import compute_hierarchical_fingerprints
from hybrid_kernel_fast import compute_hybrid_knn_scale, pack_hybrid_rows

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "experiments", "chembed_generation",
    "output", "fit_subsample_700k.parquet",
)


def _rss_gb() -> float:
    """Peak RSS so far, in GB. ru_maxrss is bytes on macOS/Darwin, KB on
    Linux - this repo's dev environment is macOS, so no platform branch."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-n", type=int, default=400_000)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=None)
    parser.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "hybrid_kernel_fast_benchmark.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fp-cache", type=str,
        default=os.path.join(os.path.dirname(__file__), "hybrid_kernel_fast_fp_cache.npz"),
        help="Cache fingerprints here, keyed by (max_n, seed) - avoids re-paying the "
             "~0.8ms/mol RDKit fingerprinting cost (minutes at N=400k+) on every benchmark "
             "run when only sweeping checkpoints/kwargs. Pass --no-fp-cache to disable.",
    )
    parser.add_argument("--no-fp-cache", action="store_true")
    args = parser.parse_args()

    checkpoints = args.checkpoints or [10_000, 50_000, 100_000, 200_000, args.max_n]
    checkpoints = sorted(set(c for c in checkpoints if c <= args.max_n))

    cache_hit = False
    if not args.no_fp_cache and os.path.exists(args.fp_cache):
        cached = np.load(args.fp_cache, allow_pickle=False)
        if int(cached["max_n"]) == args.max_n and int(cached["seed"]) == args.seed:
            print(f"Loading cached fingerprints from {args.fp_cache} (max_n={args.max_n}, seed={args.seed}) ...")
            X_mol, X_scaffold, X_skeleton = cached["X_mol"], cached["X_scaffold"], cached["X_skeleton"]
            fp_time, n_valid = float(cached["fp_time"]), X_mol.shape[0]
            cache_hit = True
            print(f"  {n_valid} valid molecules loaded from cache (fingerprinting skipped)")

    if not cache_hit:
        print(f"Loading up to {args.max_n} SMILES from {DATA_PATH} ...")
        df = pd.read_parquet(DATA_PATH, columns=["smiles"])
        df = df.sample(n=args.max_n, random_state=args.seed).reset_index(drop=True)
        smiles = df["smiles"].tolist()

        print(f"Computing hierarchical fingerprints for {len(smiles)} molecules (one-time cost, ~0.75ms/mol expected)...")
        t0 = time.time()
        X_mol, X_scaffold, X_skeleton, scaffold_ids, valid_indices = compute_hierarchical_fingerprints(smiles)
        fp_time = time.time() - t0
        n_valid = X_mol.shape[0]
        print(f"  {n_valid}/{len(smiles)} valid, {fp_time:.1f}s total ({1000 * fp_time / len(smiles):.3f} ms/mol)")

        if not args.no_fp_cache:
            np.savez(
                args.fp_cache, X_mol=X_mol, X_scaffold=X_scaffold, X_skeleton=X_skeleton,
                fp_time=fp_time, max_n=args.max_n, seed=args.seed,
            )
            print(f"  cached fingerprints to {args.fp_cache} for reuse by future runs with the same --max-n/--seed")

    # Numba JIT-compiles hybrid_tanimoto_packed (and PyNNDescent's own
    # internals) on first call - warm that up here, outside the timed loop,
    # so the smallest checkpoint isn't dominated by one-time compile cost
    # (which would make the empirical N-scaling exponent look artificially
    # sub-linear and understate the real cost at ChEMBL scale).
    print("Warming up JIT compilation (untimed)...")
    _ = compute_hybrid_knn_scale(
        X_mol[:200], X_scaffold[:200], X_skeleton[:200], k=15, random_state=args.seed, verbose=False,
    )

    # Clamp the largest requested checkpoint down to n_valid instead of
    # skipping it outright - a handful of molecules failing to fingerprint
    # (invalid SMILES, or the scaffold-sanitization edge case above) is
    # normal at real ChEMBL scale and shouldn't cost us the top data point.
    checkpoints = sorted(set(min(c, n_valid) for c in checkpoints))

    results = {
        "fingerprint_time_s": fp_time,
        "fingerprint_ms_per_mol": 1000 * fp_time / n_valid,
        "n_valid": n_valid,
        "checkpoints": [],
    }

    for N in checkpoints:
        if N > n_valid:
            print(f"skipping checkpoint N={N}: only {n_valid} valid molecules available")
            continue
        print(f"\n=== N={N} ===")
        Xm, Xs, Xk = X_mol[:N], X_scaffold[:N], X_skeleton[:N]

        rss_before = _rss_gb()
        t0 = time.time()
        X_packed = pack_hybrid_rows(Xm, Xs, Xk)
        pack_time = time.time() - t0
        packed_gb = X_packed.nbytes / 1e9
        print(f"  bit-packing: {pack_time:.2f}s, packed array {packed_gb:.3f} GB "
              f"(unpacked float32 equivalent would be {Xm.nbytes * 3 / 1e9:.3f} GB)")

        t0 = time.time()
        knn_indices, knn_dists = compute_hybrid_knn_scale(
            Xm, Xs, Xk, k=15, random_state=args.seed, verbose=False, n_jobs=-1,
        )
        build_time = time.time() - t0
        rss_after = _rss_gb()

        print(f"  NNDescent build: {build_time:.2f}s ({1000 * build_time / N:.4f} ms/molecule)")
        print(f"  RSS: {rss_before:.2f} GB -> {rss_after:.2f} GB (delta {rss_after - rss_before:.2f} GB)")
        assert knn_indices.shape == (N, 15)
        assert not (knn_indices == np.arange(N)[:, None]).any(), "self-loop leaked through"

        results["checkpoints"].append({
            "N": N,
            "pack_time_s": pack_time,
            "packed_array_gb": packed_gb,
            "build_time_s": build_time,
            "build_ms_per_mol": 1000 * build_time / N,
            "rss_before_gb": rss_before,
            "rss_after_gb": rss_after,
        })

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.out}")

    print("\n--- Extrapolation to ChEMBL scale (2.5M / 3M) ---")
    cps = results["checkpoints"]
    if len(cps) >= 2:
        # Log-log linear regression (build_time ~ N^p) over ALL checkpoints,
        # not just the last two - a 2-point exponent is noisy, and here it's
        # actively misleading: NNDescent/thread-pool setup is a fixed cost
        # that's still non-negligible relative to build time at N=5k-20k, so
        # a 2-point slope from small N looks artificially sub-linear and
        # would UNDERESTIMATE the real cost at 2.5-3M. Regressing over the
        # full checkpoint range (plus the JIT warm-up already done above)
        # averages that fixed-cost distortion out rather than anchoring on it.
        log_n = np.log([c["N"] for c in cps])
        log_t = np.log([c["build_time_s"] for c in cps])
        p, log_c = np.polyfit(log_n, log_t, 1)
        print(f"  Log-log regression over all {len(cps)} checkpoints: build_time ~ N^{p:.2f}")
        if p < 0.9:
            print(f"  NOTE: exponent < 1 across these N is almost certainly fixed per-call overhead "
                  f"(thread pool / RP-forest setup), not true sub-linear scaling - treat the "
                  f"extrapolation below as optimistic, and re-run with larger checkpoints "
                  f"(e.g. --checkpoints 200000 400000 700000) for a trustworthy exponent.")
        last = cps[-1]
        for target_n in (2_500_000, 3_000_000):
            est_s = np.exp(log_c) * target_n ** p
            est_gb = last["packed_array_gb"] * (target_n / last["N"])
            print(f"  N={target_n:,}: build ~{est_s/60:.1f} min "
                  f"(regression fit), packed array ~{est_gb:.2f} GB uint8 "
                  f"(~{4 * est_gb:.2f} GB once PyNNDescent upcasts it to float32 internally)")
    else:
        print("  Need >=2 checkpoints to extrapolate a scaling exponent.")


if __name__ == "__main__":
    main()
