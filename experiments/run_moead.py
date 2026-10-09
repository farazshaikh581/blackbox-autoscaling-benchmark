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

from blackbox.oracle import RealEncodedAutoscalingProblem, DynamicAutoscalingProblem
from blackbox import default_topology


def build_moead(n_partitions: int, n_neighbors: int = None) -> MOEAD:
    # Real-relaxed encoding -> standard SBX/PM operators (pymoo MOEA/D lacks
    # mixed-variable operators). Reference directions double as the decomposition
    # counterpart of the RL scalarization weights.
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=n_partitions)
    neighbors = min(15, len(ref_dirs)) if n_neighbors is None else min(n_neighbors, len(ref_dirs))
    return MOEAD(ref_dirs, n_neighbors=neighbors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", type=int, default=12,
                    help="das-dennis partitions -> population = C(p+2,2)")
    ap.add_argument("--neighbors", type=int, default=None,
                    help="MOEA/D neighborhood size T. Default: min(15, "
                         "population), matching the original 3-way comparison")
    ap.add_argument("--evals", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tiers", type=int, default=3)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--cap", type=int, default=20,
                    help="held-out test-front cardinality cap; match the other "
                         "algorithms' population/archive size for a fair comparison")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--dynamic", action="store_true",
                    help="dynamic variant: search a per-minute replica POLICY (small NN, "
                         "its weights the decision variable -- neuroevolution) "
                         "instead of a static config; only cpu/mem stay static "
                         "per-tier decision variables. Not combinable with "
                         "--edge yet")
    ap.add_argument("--plot", default=None,
                    help="path to save convergence/front figures (default: "
                         "docs/figures/moead_run.png)")
    ap.add_argument("--no-plot", action="store_true", help="skip the figures")
    from experiments._common import (
        add_workload_args, add_topology_args, build_train_test, build_topology,
    )
    add_workload_args(ap)
    add_topology_args(ap)
    args = ap.parse_args()

    topo = build_topology(args)
    if args.dynamic and args.edge:
        raise SystemExit("--dynamic does not support --edge yet "
                         "(simulate_policy is single-location-only)")
    train_wl, test_wl = build_train_test(args)
    problem_cls = DynamicAutoscalingProblem if args.dynamic else RealEncodedAutoscalingProblem
    problem = problem_cls(
        topology=topo, workload=train_wl,
        k_replications=args.k, base_seed=args.base_seed, sla_ms=args.sla_ms,
    )
    algorithm = build_moead(args.partitions, args.neighbors)
    res = minimize(
        problem, algorithm, MaximumFunctionCallTermination(args.evals),
        seed=args.seed, save_history=True, verbose=True,
    )

    F = np.atleast_2d(res.F)
    print(f"\nMOEA/D done: {problem.n_eval_calls} oracle calls, {len(F)} train points")

    if not args.no_plot:
        from experiments._common import plot_single_run
        plot_single_run(res, F, "moead", args.plot or "docs/figures/moead_run.png",
                        label="MOEA/D")

    if args.out:
        from experiments._common import save_run, holdout_front
        if test_wl is not None:
            # Same cardinality cap as NSGA-II (--pop) and MORL (--archive), so the
            # three test fronts are cardinality-comparable regardless of MOEA/D's
            # (much larger) das-dennis population.
            F_test = holdout_front(
                problem_cls(topology=topo, workload=test_wl,
                           k_replications=args.k,
                           base_seed=args.base_seed,
                           sla_ms=args.sla_ms),
                res.X, cap=args.cap,
            )
            print(f"held-out test front: {len(F_test)} points")
            save_run(args.out, res, algo="moead", seed=args.seed, F_test=F_test)
        else:
            save_run(args.out, res, algo="moead", seed=args.seed)


if __name__ == "__main__":
    main()
