"""Mechanism check: why NSGA-II beats MOEA/D on the dynamic policy problem.

Decision-space coherence check from the paper. Runs directly against the
oracle under the current calibrated energy constants.

Two checks, both against the same held-out test-day workload used
everywhere else in this project:

1. Local sensitivity: sample a random decision vector, nudge it by a small
   step in a random direction, and measure each objective's relative jump.
   Repeated for both the dynamic policy space (99 decision variables for the
   default 3-tier topology) and the static config space (9 variables).
   Tests whether the dynamic space is simply "rougher."

2. Decision-space coherence: sample random points in each space, and check
   whether two points with a similar objective score also sit close together
   in (normalized) decision space -- the Spearman correlation between
   pairwise decision-distance and pairwise objective-distance. This is the
   actual mechanism check: MOEA/D's neighborhood mating assumes neighboring
   decision vectors give neighboring scores, an assumption that should hold
   less in a space where this correlation is weak.

    python -m experiments.mechanism_check \
        --trace data/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from blackbox import policy, simulator
from blackbox.hpa import HPAPolicyProblem
from blackbox.oracle import DynamicAutoscalingProblem, RealEncodedAutoscalingProblem
from blackbox.topology import default_topology
from blackbox.workload import from_azure_trace


def _random_point(xl: np.ndarray, xu: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return xl + rng.random(len(xl)) * (xu - xl)


def _make_static_evaluator(topo, workload, k, base_seed):
    problem = RealEncodedAutoscalingProblem(topology=topo, workload=workload,
                                            k_replications=k, base_seed=base_seed)

    def eval_one(x: np.ndarray) -> np.ndarray:
        config = problem._row_to_config(np.asarray(x, dtype=float))
        res = simulator.evaluate(topo, config, workload, k_replications=k,
                                 base_seed=base_seed)
        return res["mean"]

    return problem.xl, problem.xu, eval_one


def _make_dynamic_evaluator(topo, workload, k, base_seed):
    problem = DynamicAutoscalingProblem(topology=topo, workload=workload,
                                        k_replications=k, base_seed=base_seed)

    def eval_one(x: np.ndarray) -> np.ndarray:
        cpu, mem, weights = problem._split(np.asarray(x, dtype=float))
        policies = policy.decode(weights, topo.n_tiers)
        res = simulator.evaluate_policy(topo, policies, cpu, mem, workload,
                                        k_replications=k, base_seed=base_seed)
        return res["mean"]

    return problem.xl, problem.xu, eval_one


def _make_hpa_evaluator(topo, workload, k, base_seed):
    problem = HPAPolicyProblem(topology=topo, workload=workload,
                               k_replications=k, base_seed=base_seed)

    def eval_one(x: np.ndarray) -> np.ndarray:
        out = {}
        problem._evaluate(np.asarray(x, dtype=float), out)
        return out["F"]

    return problem.xl, problem.xu, eval_one


def local_sensitivity(xl, xu, eval_one, rng, n_trials=15, step_frac=0.02):
    """`n_trials` random points, each nudged by `step_frac` of the space's
    range in a random unit direction. Returns an (n_trials, 3) array of each
    objective's relative jump |f1 - f0| / max(|f0|, eps)."""
    span = xu - xl
    jumps = np.empty((n_trials, 3))
    for t in range(n_trials):
        x0 = _random_point(xl, xu, rng)
        f0 = eval_one(x0)
        direction = rng.normal(size=len(xl))
        direction /= np.linalg.norm(direction)
        x1 = np.clip(x0 + step_frac * span * direction, xl, xu)
        f1 = eval_one(x1)
        jumps[t] = np.abs(f1 - f0) / np.maximum(np.abs(f0), 1e-9)
    return jumps


def coherence_check(xl, xu, eval_one, rng, n_points=150):
    """`n_points` random points: Spearman correlation between pairwise
    decision-distance (normalized to [0,1] per dimension) and pairwise
    objective-distance (normalized to [0,1] per objective)."""
    X = np.array([_random_point(xl, xu, rng) for _ in range(n_points)])
    F = np.array([eval_one(x) for x in X])
    Xn = (X - xl) / np.maximum(xu - xl, 1e-9)
    f_lo, f_hi = F.min(axis=0), F.max(axis=0)
    Fn = (F - f_lo) / np.maximum(f_hi - f_lo, 1e-9)
    d_dec = pdist(Xn)
    d_obj = pdist(Fn)
    rho, p = spearmanr(d_dec, d_obj)
    return float(rho), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--test-day", type=int, default=3)
    ap.add_argument("--tiers", type=int, default=3)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--n-trials", type=int, default=15)
    ap.add_argument("--n-points", type=int, default=150)
    ap.add_argument("--seeds", type=str, default="1,2",
                    help="comma-separated coherence-check RNG seeds (two, per "
                         "the original methodology's 'confirmed on two random "
                         "seeds')")
    ap.add_argument("--spaces", type=str, default="static,dynamic",
                    help="comma-separated subset of static, dynamic, hpa")
    ap.add_argument("--skip-local", action="store_true",
                    help="run only the coherence check")
    ap.add_argument("--out", type=str, default="mechanism_check_results.json")
    args = ap.parse_args()

    topo = default_topology(args.tiers)
    workload = from_azure_trace(args.trace, day_index=args.test_day)

    makers = {"static": _make_static_evaluator,
              "dynamic": _make_dynamic_evaluator,
              "hpa": _make_hpa_evaluator}
    spaces = {}
    for name in args.spaces.split(","):
        spaces[name] = makers[name](topo, workload, args.k, args.base_seed)
        print(f"{name} space: {len(spaces[name][0])} decision variables")

    if not args.skip_local:
        rng = np.random.default_rng(0)
        print("\n=== 1. Local sensitivity ===")
        obj_names = ["latency_ms", "cost", "energy_W"]
        for space_name, (xl, xu, eval_one) in spaces.items():
            jumps = local_sensitivity(xl, xu, eval_one, rng, n_trials=args.n_trials)
            mean_pct = 100.0 * jumps.mean(axis=0)
            max_pct = 100.0 * jumps.max(axis=0)
            print(f"  {space_name}: mean relative jump % per objective "
                  f"({', '.join(obj_names)}) = "
                  f"{', '.join(f'{v:.2f}' for v in mean_pct)}")
            print(f"    max relative jump %: "
                  f"{', '.join(f'{v:.2f}' for v in max_pct)}")

    print("\n=== 2. Decision-space coherence ===")
    seeds = [int(s) for s in args.seeds.split(",")]
    results = {name: [] for name in spaces}
    for seed in seeds:
        rng2 = np.random.default_rng(seed)
        for space_name, (xl, xu, eval_one) in spaces.items():
            rho, p = coherence_check(xl, xu, eval_one, rng2, n_points=args.n_points)
            results[space_name].append((seed, rho, p))
            print(f"  {space_name} space, seed={seed}: "
                  f"Spearman(decision-dist, objective-dist) = {rho:.4f} (p={p:.2e})")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
