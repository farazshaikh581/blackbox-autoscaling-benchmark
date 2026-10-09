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

from blackbox import AutoscalingProblem, DynamicAutoscalingProblem, default_topology


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
    ap.add_argument("--dynamic", action="store_true",
                    help="dynamic variant: search a per-minute replica POLICY (small NN, "
                         "its weights the decision variable -- neuroevolution) "
                         "instead of a static config; only cpu/mem stay static "
                         "per-tier decision variables. Not combinable with "
                         "--edge yet")
    ap.add_argument("--plot", default=None,
                    help="path to save convergence/front figures (default: "
                         "docs/figures/nsga2_run.png)")
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
    if args.dynamic:
        problem = DynamicAutoscalingProblem(
            topology=topo, workload=train_wl, k_replications=args.k,
            base_seed=args.base_seed, sla_ms=args.sla_ms,
        )
        # Fully real-valued (no more integer replicas), so plain default
        # SBX/PM operators apply directly -- unlike AutoscalingProblem's
        # mixed replicas/cpu/mem vector, this needs no MixedVariable* ops.
        algorithm = NSGA2(pop_size=args.pop, eliminate_duplicates=True)
    else:
        problem = AutoscalingProblem(
            topology=topo,
            workload=train_wl,
            k_replications=args.k,
            base_seed=args.base_seed,
            sla_ms=args.sla_ms,
        )
        algorithm = build_nsga2(args.pop)
    termination = MaximumFunctionCallTermination(args.evals)

    res = minimize(
        problem, algorithm, termination,
        seed=args.seed, save_history=True, verbose=True,
    )

    F = res.F
    print(f"\nNSGA-II done: {problem.n_eval_calls} oracle calls, "
          f"{len(F)} non-dominated points on train")
    print("train objective ranges (min, max):")
    for j, name in enumerate(["latency_ms", "cost", "energy_W"]):
        print(f"  {name:12s} {F[:, j].min():10.3f}  {F[:, j].max():10.3f}")

    if not args.no_plot:
        from experiments._common import plot_single_run
        plot_single_run(res, F, "nsga2", args.plot or "docs/figures/nsga2_run.png",
                        label="NSGA-II")

    if args.out:
        from experiments._common import save_run, holdout_front
        if test_wl is not None:
            test_problem_cls = DynamicAutoscalingProblem if args.dynamic else AutoscalingProblem
            F_test = holdout_front(
                test_problem_cls(topology=topo, workload=test_wl,
                                 k_replications=args.k, base_seed=args.base_seed,
                                 sla_ms=args.sla_ms),
                res.X, cap=args.pop,
            )
            print(f"held-out test front: {len(F_test)} points")
            save_run(args.out, res, algo="nsga2", seed=args.seed, F_test=F_test)
        else:
            save_run(args.out, res, algo="nsga2", seed=args.seed)


if __name__ == "__main__":
    main()
