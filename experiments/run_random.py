"""Random search baseline: uniform samples at the same budget, non-dominated archive.

Usage:
    python -m experiments.run_random --pop 20 --evals 500 --seed 1 --k 5 \
        [--dynamic] --out results/random_seed1.npz

Samples in batches of `--pop` so the anytime history has the same granularity
as the EAs' generations. The reported front follows the shared fairness rule:
the non-dominated archive, re-scored on the held-out day when one is given, and
capped to `--pop` points by crowding distance.
"""
from __future__ import annotations

import argparse

import numpy as np
from pymoo.core.evaluator import Evaluator
from pymoo.core.mixed import MixedVariableSampling
from pymoo.core.population import Population
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from blackbox import AutoscalingProblem, DynamicAutoscalingProblem


def _nd(pop):
    F = pop.get("F")
    return pop[NonDominatedSorting().do(F, only_non_dominated_front=True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=20, help="batch size and front cap")
    ap.add_argument("--evals", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tiers", type=int, default=3)
    ap.add_argument("--k", type=int, default=5, help="workload replications per eval")
    ap.add_argument("--base-seed", type=int, default=0, help="oracle workload seed")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--dynamic", action="store_true",
                    help="sample dynamic-variant policy weight vectors instead of static configs")
    from experiments._common import (
        add_workload_args, add_topology_args, build_train_test, build_topology,
        bounded_front, holdout_front, save_history,
    )
    add_workload_args(ap)
    add_topology_args(ap)
    args = ap.parse_args()

    topo = build_topology(args)
    if args.dynamic and args.edge:
        raise SystemExit("--dynamic does not support --edge yet")
    train_wl, test_wl = build_train_test(args)
    cls = DynamicAutoscalingProblem if args.dynamic else AutoscalingProblem
    kw = dict(topology=topo, k_replications=args.k, base_seed=args.base_seed,
              sla_ms=args.sla_ms)
    problem = cls(workload=train_wl, **kw)
    sampling = FloatRandomSampling() if args.dynamic else MixedVariableSampling()
    rng = np.random.default_rng(args.seed)
    evaluator = Evaluator()

    archive = Population()
    hist_n, hist_F = [], []
    n_done = 0
    while n_done < args.evals:
        n = min(args.pop, args.evals - n_done)
        batch = sampling(problem, n, random_state=rng)
        evaluator.eval(problem, batch)
        n_done += n
        archive = _nd(Population.merge(archive, batch))
        hist_n.append(n_done)
        hist_F.append(archive.get("F"))

    F_train, X_nd = archive.get("F"), archive.get("X")
    print(f"random search done: {problem.n_eval_calls} oracle calls, "
          f"{len(F_train)} non-dominated points on train")

    if args.out:
        if test_wl is not None:
            F = holdout_front(cls(workload=test_wl, **kw), X_nd, cap=args.pop)
            print(f"held-out test front: {len(F)} points")
        else:
            F = bounded_front(F_train, args.pop)
        save_history(args.out, F, hist_n, hist_F, "random", args.seed, F_train=F_train)


if __name__ == "__main__":
    main()
