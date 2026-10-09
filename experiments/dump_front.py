"""Dump an optimized NSGA-II front as concrete configs for cluster deployment.

The saved run files (`results/*.npz`) keep only the objective vectors `F`, not the
decision configs, so this re-runs one NSGA-II search and decodes the final
non-dominated population into per-tier `(replicas, cpu, mem)` configs, saved with
their simulator objectives to `results/front_configs.json`. That file is the input
to `calibration/measure_front_multinode.py`, which deploys it on the testbed
and checks that the measured order of latency and power matches the oracle.

    python -m experiments.dump_front --pop 20 --evals 500 --seed 1
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from pymoo.optimize import minimize
from pymoo.termination.max_eval import MaximumFunctionCallTermination

from blackbox import AutoscalingProblem, default_topology
from blackbox.oracle import x_to_config
from experiments.run_nsga2 import build_nsga2

OBJ_NAMES = ["latency_ms", "cost", "energy_W"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--evals", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tiers", type=int, default=3)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--out", default="results/front_configs.json")
    ap.add_argument("--plot", default=None,
                    help="path to save the front figure (default: "
                         "docs/figures/dump_front.png)")
    ap.add_argument("--no-plot", action="store_true", help="skip the figure")
    args = ap.parse_args()

    topo = default_topology(args.tiers)
    problem = AutoscalingProblem(topology=topo, k_replications=args.k,
                                 base_seed=args.base_seed)
    res = minimize(problem, build_nsga2(args.pop),
                   MaximumFunctionCallTermination(args.evals),
                   seed=args.seed, verbose=False)

    tiers = [t.name for t in topo.tiers]
    configs = []
    for x, f in zip(np.atleast_1d(res.X), np.atleast_2d(res.F)):
        cfg = x_to_config(x, topo.n_tiers)
        configs.append({
            "replicas": [int(v) for v in cfg["replicas"]],
            "cpu": [round(float(v), 4) for v in cfg["cpu"]],
            "mem": [round(float(v), 4) for v in cfg["mem"]],
            "sim_objectives": {n: round(float(f[j]), 4)
                               for j, n in enumerate(OBJ_NAMES)},
        })
    configs.sort(key=lambda c: c["sim_objectives"]["latency_ms"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"tiers": tiers, "seed": args.seed,
                   "objective_names": OBJ_NAMES, "front": configs}, fh, indent=2)
    print(f"{len(configs)} front configs -> {args.out}")
    print(f"tiers: {tiers}  (latency {configs[0]['sim_objectives']['latency_ms']} .. "
          f"{configs[-1]['sim_objectives']['latency_ms']} ms)")

    if not args.no_plot:
        from experiments._common import plot_run_from_history
        plot_run_from_history([], [], np.atleast_2d(res.F), "nsga2",
                              args.plot or "docs/figures/dump_front.png",
                              label="Optimized front")


if __name__ == "__main__":
    main()
