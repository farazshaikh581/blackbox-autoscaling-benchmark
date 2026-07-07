"""Run MOEA/D on the black-box autoscaling oracle under a fixed budget.

MOEA/D decomposes the objective space with reference directions; those direction
vectors are the mathematical counterpart of the RL scalarization weights
(perf/cost/energy/balanced) used by the MORL baseline. Analyzing whether the RL
weight vectors cover the same region as MOEA/D's decomposition is one of the
analysis tasks.

Usage:
    python -m experiments.run_moead --partitions 12 --evals 500 --seed 1 --tiers 3
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.termination.max_eval import MaximumFunctionCallTermination
from pymoo.util.ref_dirs import get_reference_directions

from blackbox.oracle import RealEncodedAutoscalingProblem
from blackbox import default_topology


def build_moead(n_partitions: int) -> MOEAD:
    # Real-relaxed encoding -> standard SBX/PM operators (pymoo MOEA/D lacks
    # mixed-variable operators). Reference directions double as the decomposition
    # counterpart of the RL scalarization weights.
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=n_partitions)
    return MOEAD(ref_dirs, n_neighbors=min(15, len(ref_dirs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", type=int, default=12,
                    help="das-dennis partitions -> population = C(p+2,2)")
    ap.add_argument("--evals", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tiers", type=int, default=3)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    problem = RealEncodedAutoscalingProblem(
        topology=default_topology(args.tiers), k_replications=args.k,
        base_seed=args.base_seed,
    )
    algorithm = build_moead(args.partitions)
    res = minimize(
        problem, algorithm, MaximumFunctionCallTermination(args.evals),
        seed=args.seed, save_history=True, verbose=True,
    )

    F = np.atleast_2d(res.F)
    print(f"\nMOEA/D done: {problem.n_eval_calls} oracle calls, {len(F)} points")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.savez(args.out, F=F, seed=args.seed, evals=args.evals)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
