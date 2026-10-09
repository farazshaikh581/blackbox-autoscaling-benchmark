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

from blackbox import (
    default_topology, simulate_config, evaluate, evaluate_policy, decode_policy,
    weight_vector_size,
)
from viz import COLOR, INK, INK_2, SINGLE, new_fig, style_axes, savefig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--tiers", type=int, default=3)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--window-hours", type=float, default=6.0)
    ap.add_argument("--runs", type=int, default=10, help="independent runs planned")
    ap.add_argument("--algos", type=int, default=3, help="algorithms compared")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dynamic", action="store_true",
                    help="dynamic variant: time the policy oracle (evaluate_policy) "
                         "instead of the static oracle (evaluate)")
    ap.add_argument("--weight-bound", type=float, default=2.0,
                    help="--dynamic only: policy weight sampling range, "
                         "matches DynamicAutoscalingProblem's default")
    ap.add_argument("--plot", default=None,
                    help="path to save the per-call timing distribution "
                         "(default: docs/figures/timing_benchmark.png)")
    ap.add_argument("--no-plot", action="store_true", help="skip the figure")
    from experiments._common import add_workload_args, build_workload
    add_workload_args(ap)
    args = ap.parse_args()

    topo = default_topology(args.tiers)
    wl = build_workload(args)
    rng = np.random.default_rng(args.seed)

    if args.dynamic:
        n_weights = weight_vector_size() * args.tiers

        def random_cfg():
            return {
                "cpu": np.array([rng.uniform(t.cpu_min, t.cpu_max) for t in topo.tiers]),
                "mem": np.array([rng.uniform(t.mem_min, t.mem_max) for t in topo.tiers]),
                "weights": rng.uniform(-args.weight_bound, args.weight_bound, n_weights),
            }

        def run(cfg):
            policies = decode_policy(cfg["weights"], topo.n_tiers)
            evaluate_policy(topo, policies, cfg["cpu"], cfg["mem"], wl, args.k)

        # Warm up (numpy / import costs) then time `samples` averaged oracle calls.
        run({"cpu": np.array([1.0] * args.tiers), "mem": np.array([1.0] * args.tiers),
             "weights": np.zeros(n_weights)})
        call_ms = []
        for _ in range(args.samples):
            t0 = time.perf_counter()
            run(random_cfg())
            call_ms.append((time.perf_counter() - t0) * 1000.0)
        dt = sum(call_ms) / 1000.0
    else:
        # Warm up (numpy / import costs) then time `samples` averaged oracle calls.
        evaluate(topo, {"replicas": np.array([2] * args.tiers),
                        "cpu": np.array([1.0] * args.tiers),
                        "mem": np.array([1.0] * args.tiers)}, wl, args.k)

        call_ms = []
        for _ in range(args.samples):
            cfg = {
                "replicas": np.array([rng.integers(t.replica_min, t.replica_max + 1)
                                      for t in topo.tiers]),
                "cpu": np.array([rng.uniform(t.cpu_min, t.cpu_max) for t in topo.tiers]),
                "mem": np.array([rng.uniform(t.mem_min, t.mem_max) for t in topo.tiers]),
            }
            t0 = time.perf_counter()
            evaluate(topo, cfg, wl, args.k)
            call_ms.append((time.perf_counter() - t0) * 1000.0)
        dt = sum(call_ms) / 1000.0

    per_eval_ms = dt / args.samples * 1000.0
    window_s = args.window_hours * 3600.0
    total_calls = window_s / (per_eval_ms / 1000.0)
    per_config = total_calls / (args.runs * args.algos)

    mode = "dynamic (policy)" if args.dynamic else "static (config)"
    print(f"mode={mode}  tiers={args.tiers}  K={args.k}  minutes/day=1440")
    print(f"per averaged-oracle call : {per_eval_ms:8.2f} ms")
    print(f"window                   : {args.window_hours:.1f} h "
          f"({args.runs} runs x {args.algos} algos)")
    print(f"total oracle calls in window : {total_calls:,.0f}")
    print(f"=> evaluation budget per (algo,run) : {per_config:,.0f}")

    if not args.no_plot:
        plot_timing(call_ms, mode, args.plot or "docs/figures/timing_benchmark.png")


def plot_timing(call_ms, mode, path):
    call_ms = np.asarray(call_ms)
    fig, ax = new_fig(SINGLE, 2.2)
    ax.hist(call_ms, bins=30, color=COLOR["measured"], edgecolor="white", linewidth=0.5)
    mean = call_ms.mean()
    ax.axvline(mean, color=INK, lw=1, ls="--")
    ax.annotate(f"mean {mean:.1f} ms", (mean, 1), xycoords=("data", "axes fraction"),
                xytext=(-4, -2), textcoords="offset points", va="top", ha="right",
                fontsize=7, color=INK_2)
    ax.set_xlabel("time per averaged oracle call (ms)")
    ax.set_ylabel("calls")
    ax.set_title(f"Oracle call time, {mode}, n={len(call_ms)}")
    style_axes(ax)
    savefig(fig, path)


if __name__ == "__main__":
    main()
