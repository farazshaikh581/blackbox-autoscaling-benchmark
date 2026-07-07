"""Preparatory task: wall-clock timing -> maximum evaluation budget.

Times the averaged oracle (one candidate, K replications) and extrapolates how
many evaluations fit in a given experimental window. This fixes the achievable
budget (evals x runs x algorithms) for the comparison protocol.

    python -m experiments.timing_benchmark --k 10 --tiers 3 --samples 100 \
        --window-hours 6 --runs 10 --algos 3
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from blackbox import default_topology, synthetic_diurnal, simulate_config, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--tiers", type=int, default=3)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--window-hours", type=float, default=6.0)
    ap.add_argument("--runs", type=int, default=10, help="independent runs planned")
    ap.add_argument("--algos", type=int, default=3, help="algorithms compared")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    topo = default_topology(args.tiers)
    wl = synthetic_diurnal()
    rng = np.random.default_rng(args.seed)

    # Warm up (numpy / import costs) then time `samples` averaged oracle calls.
    evaluate(topo, {"replicas": np.array([2] * args.tiers),
                    "cpu": np.array([1.0] * args.tiers),
                    "mem": np.array([1.0] * args.tiers)}, wl, args.k)

    t0 = time.perf_counter()
    for _ in range(args.samples):
        cfg = {
            "replicas": np.array([rng.integers(t.replica_min, t.replica_max + 1)
                                  for t in topo.tiers]),
            "cpu": np.array([rng.uniform(t.cpu_min, t.cpu_max) for t in topo.tiers]),
            "mem": np.array([rng.uniform(t.mem_min, t.mem_max) for t in topo.tiers]),
        }
        evaluate(topo, cfg, wl, args.k)
    dt = time.perf_counter() - t0

    per_eval_ms = dt / args.samples * 1000.0
    window_s = args.window_hours * 3600.0
    total_calls = window_s / (per_eval_ms / 1000.0)
    per_config = total_calls / (args.runs * args.algos)

    print(f"tiers={args.tiers}  K={args.k}  minutes/day=1440")
    print(f"per averaged-oracle call : {per_eval_ms:8.2f} ms")
    print(f"window                   : {args.window_hours:.1f} h "
          f"({args.runs} runs x {args.algos} algos)")
    print(f"total oracle calls in window : {total_calls:,.0f}")
    print(f"=> evaluation budget per (algo,run) : {per_config:,.0f}")


if __name__ == "__main__":
    main()
