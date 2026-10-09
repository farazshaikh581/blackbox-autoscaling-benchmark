"""Tuned Kubernetes HPA baseline for the dynamic problem.

Usage:
    python -m experiments.run_hpa --pop 20 --evals 500 --seed 1 --k 5 \
        --out results_dynamic/hpa_seed1.npz

The per-minute replica decision is the upstream HPA threshold rule
(`blackbox/hpa.py`), not a learned policy. Its knobs (per-tier cpu, mem and CPU
target) are tuned by NSGA-II at the same budget the learned policies get, so
the comparison asks whether a learned policy beats the best HPA setting the
same search effort can find.
"""
from __future__ import annotations

import argparse

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination.max_eval import MaximumFunctionCallTermination

from blackbox.hpa import HPAPolicyProblem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--evals", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tiers", type=int, default=3)
    ap.add_argument("--k", type=int, default=5, help="workload replications per eval")
    ap.add_argument("--base-seed", type=int, default=0, help="oracle workload seed")
    ap.add_argument("--out", type=str, default=None)
    from experiments._common import (
        add_workload_args, add_topology_args, build_train_test, build_topology,
        holdout_front, save_run,
    )
    add_workload_args(ap)
    add_topology_args(ap)
    args = ap.parse_args()
    if args.edge:
        raise SystemExit("run_hpa supports the single-location topology only")

    topo = build_topology(args)
    train_wl, test_wl = build_train_test(args)
    kw = dict(topology=topo, k_replications=args.k, base_seed=args.base_seed,
              sla_ms=args.sla_ms)
    problem = HPAPolicyProblem(workload=train_wl, **kw)
    res = minimize(problem, NSGA2(pop_size=args.pop, eliminate_duplicates=True),
                   MaximumFunctionCallTermination(args.evals),
                   seed=args.seed, save_history=True, verbose=False)
    print(f"HPA tuning done: {problem.n_eval_calls} oracle calls, "
          f"{len(res.F)} non-dominated points on train")

    if args.out:
        F_test = None
        if test_wl is not None:
            F_test = holdout_front(HPAPolicyProblem(workload=test_wl, **kw),
                                   res.X, cap=args.pop)
            print(f"held-out test front: {len(F_test)} points")
        save_run(args.out, res, algo="hpa", seed=args.seed, F_test=F_test)


if __name__ == "__main__":
    main()
