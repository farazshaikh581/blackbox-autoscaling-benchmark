"""Preparatory task: choose K by monitoring the coefficient of variation.

For a sample of random configurations, measure how the per-objective coefficient
of variation (CoV = std/mean of the objective across workload replications) falls
as the replication count K increases. K* is the smallest K for which the worst
objective CoV drops below a target threshold, giving a deterministic averaged
oracle. This produces the noise metadata the ROAR-NET spec requires.

    python -m experiments.cov_replications --configs 20 --kmax 40 --target 0.05
"""
from __future__ import annotations

import argparse

import numpy as np

from blackbox import default_topology, synthetic_diurnal, simulate_config

OBJ = ["latency_ms", "cost", "energy_W"]


def random_config(topo, rng):
    n = np.array([rng.integers(t.replica_min, t.replica_max + 1) for t in topo.tiers])
    c = np.array([rng.uniform(t.cpu_min, t.cpu_max) for t in topo.tiers])
    m = np.array([rng.uniform(t.mem_min, t.mem_max) for t in topo.tiers])
    return {"replicas": n, "cpu": c, "mem": m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=int, default=20)
    ap.add_argument("--kmax", type=int, default=40)
    ap.add_argument("--target", type=float, default=0.05, help="target worst CoV")
    ap.add_argument("--tiers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    topo = default_topology(args.tiers)
    wl = synthetic_diurnal()
    rng = np.random.default_rng(args.seed)
    configs = [random_config(topo, rng) for _ in range(args.configs)]

    # For each config, one long run of kmax realizations; running CoV vs K.
    cov_by_k = np.zeros((args.kmax, 3))
    for cfg in configs:
        samples = np.array([
            simulate_config(topo, cfg, wl.realize(seed=k)).as_array()
            for k in range(args.kmax)
        ])
        for k in range(2, args.kmax + 1):
            sub = samples[:k]
            mean = sub.mean(0)
            std = sub.std(0, ddof=1)
            cov = np.divide(std, np.abs(mean), out=np.zeros(3), where=mean != 0)
            cov_by_k[k - 1] += cov
    cov_by_k /= len(configs)  # mean worst-case CoV across configs

    print(f"{'K':>4} | " + " | ".join(f"{o:>10}" for o in OBJ) + " |  worst")
    k_star = None
    for k in range(2, args.kmax + 1):
        row = cov_by_k[k - 1]
        worst = row.max()
        flag = ""
        if k_star is None and worst < args.target:
            k_star = k
            flag = "  <- K*"
        print(f"{k:>4} | " + " | ".join(f"{v:10.4f}" for v in row) +
              f" | {worst:6.4f}{flag}")

    if k_star:
        print(f"\nRecommended K* = {k_star} (worst objective CoV < {args.target})")
    else:
        print(f"\nNo K <= {args.kmax} reached CoV < {args.target}; raise --kmax.")


if __name__ == "__main__":
    main()
