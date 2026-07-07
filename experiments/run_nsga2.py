"""Run NSGA-II on the black-box autoscaling oracle under a fixed budget.

Usage:
    python -m experiments.run_nsga2 --pop 20 --evals 500 --seed 1 --tiers 3 \
        --k 5 --out results/nsga2_seed1.npz

Records the anytime history (cumulative evals -> hypervolume) and the final
non-dominated set, for the HV / IGD+ / GD+ comparison against MOEA/D and the RL
baseline.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import (
    MixedVariableMating,
    MixedVariableSampling,
    MixedVariableDuplicateElimination,
)
from pymoo.optimize import minimize
from pymoo.termination.max_eval import MaximumFunctionCallTermination

from blackbox import AutoscalingProblem, default_topology


def build_nsga2(pop_size: int) -> NSGA2:
    return NSGA2(
        pop_size=pop_size,
        sampling=MixedVariableSampling(),
        mating=MixedVariableMating(
            eliminate_duplicates=MixedVariableDuplicateElimination()
        ),
        eliminate_duplicates=MixedVariableDuplicateElimination(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--evals", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tiers", type=int, default=3)
    ap.add_argument("--k", type=int, default=5, help="workload replications per eval")
    ap.add_argument("--base-seed", type=int, default=0, help="oracle workload seed")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    problem = AutoscalingProblem(
        topology=default_topology(args.tiers),
        k_replications=args.k,
        base_seed=args.base_seed,
    )
    algorithm = build_nsga2(args.pop)
    termination = MaximumFunctionCallTermination(args.evals)

    res = minimize(
        problem, algorithm, termination,
        seed=args.seed, save_history=True, verbose=True,
    )

    F = res.F
    print(f"\nNSGA-II done: {problem.n_eval_calls} oracle calls, "
          f"{len(F)} non-dominated points")
    print("objective ranges (min, max):")
    for j, name in enumerate(["latency_ms", "cost", "energy_W"]):
        print(f"  {name:12s} {F[:, j].min():10.3f}  {F[:, j].max():10.3f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        n_evals = np.array([e.evaluator.n_eval for e in res.history])
        np.savez(
            args.out,
            F=F, X_obj=F, seed=args.seed, pop=args.pop, evals=args.evals,
            hist_n_evals=n_evals,
        )
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
