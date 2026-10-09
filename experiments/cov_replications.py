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

from blackbox import (
    default_topology, simulate_config, simulate_policy, decode_policy,
    weight_vector_size,
)
from viz import INK_2, INK_3, PALETTE, SINGLE, label_end, new_fig, style_axes, savefig

OBJ = ["latency_ms", "cost", "energy_W"]


def random_config(topo, rng):
    n = np.array([rng.integers(t.replica_min, t.replica_max + 1) for t in topo.tiers])
    c = np.array([rng.uniform(t.cpu_min, t.cpu_max) for t in topo.tiers])
    m = np.array([rng.uniform(t.mem_min, t.mem_max) for t in topo.tiers])
    return {"replicas": n, "cpu": c, "mem": m}


def random_dynamic_config(topo, rng, weight_bound):
    """--dynamic: random per-tier cpu/mem plus a random policy weight vector.

    Mirrors `random_config`, but replicas are not a decision variable here
    (`DynamicAutoscalingProblem`'s vector: cpu/mem per tier, then flattened
    policy weights). Bounds match that problem's default `weight_bound=2.0`.
    """
    c = np.array([rng.uniform(t.cpu_min, t.cpu_max) for t in topo.tiers])
    m = np.array([rng.uniform(t.mem_min, t.mem_max) for t in topo.tiers])
    n_weights = weight_vector_size() * topo.n_tiers
    w = rng.uniform(-weight_bound, weight_bound, size=n_weights)
    return {"cpu": c, "mem": m, "weights": w}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=int, default=20)
    ap.add_argument("--kmax", type=int, default=40)
    ap.add_argument("--target", type=float, default=0.05, help="target worst CoV")
    ap.add_argument("--tiers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dynamic", action="store_true",
                    help="dynamic variant: measure noise on random per-minute replica "
                         "policies (simulate_policy) instead of random static "
                         "configs (simulate_config)")
    ap.add_argument("--weight-bound", type=float, default=2.0,
                    help="--dynamic only: policy weight sampling range, "
                         "matches DynamicAutoscalingProblem's default")
    ap.add_argument("--plot", default=None,
                    help="path to save the CoV-vs-K figure (default: "
                         "docs/figures/cov_replications.png)")
    ap.add_argument("--no-plot", action="store_true", help="skip the figure")
    from experiments._common import add_workload_args, build_workload
    add_workload_args(ap)
    args = ap.parse_args()

    topo = default_topology(args.tiers)
    wl = build_workload(args)
    rng = np.random.default_rng(args.seed)

    if args.dynamic:
        configs = [random_dynamic_config(topo, rng, args.weight_bound)
                   for _ in range(args.configs)]

        def run(cfg, seed):
            policies = decode_policy(cfg["weights"], topo.n_tiers)
            return simulate_policy(topo, policies, cfg["cpu"], cfg["mem"],
                                    wl.realize(seed=seed)).as_array()
    else:
        configs = [random_config(topo, rng) for _ in range(args.configs)]

        def run(cfg, seed):
            return simulate_config(topo, cfg, wl.realize(seed=seed)).as_array()

    # For each config, one long run of kmax realizations; running CoV vs K.
    cov_by_k = np.zeros((args.kmax, 3))
    for cfg in configs:
        samples = np.array([run(cfg, k) for k in range(args.kmax)])
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

    if not args.no_plot:
        plot_cov(cov_by_k, args.target, k_star,
                 args.plot or "docs/figures/cov_replications.png")


def plot_cov(cov_by_k, target, k_star, path):
    # A deterministic objective (e.g. cost, which doesn't depend on the
    # stochastic workload realization) has CoV at float noise (~1e-17), which
    # would blow the log axis out to 17 decades and bury the informative
    # (latency/energy) lines. Floor at a decade below the target instead.
    floor = target * 1e-3
    ks = np.arange(2, len(cov_by_k) + 1)
    fig, ax = new_fig(SINGLE, 2.5)
    ax.axhline(target, color=INK_3, lw=1, ls="--")
    ax.annotate(f"target CoV {target}", (ks[-1], target), xytext=(0, -3),
                textcoords="offset points", ha="right", va="top",
                fontsize=7, color=INK_2)
    labels = {"latency_ms": "latency", "cost": "cost (deterministic)",
              "energy_W": "energy"}
    for j, name in enumerate(OBJ):
        y = np.clip(cov_by_k[1:, j], floor, None)
        ax.plot(ks, y, color=PALETTE[j])
        label_end(ax, ks[-1], y[-1], labels[name])
    if k_star:
        ax.axvline(k_star, color=INK_3, lw=0.8, ls=":")
        ax.annotate(f"K*={k_star}", (k_star, floor * 0.7), xytext=(3, 0),
                    textcoords="offset points", fontsize=7, color=INK_2)
    ax.set_xlabel("replications K")
    ax.set_ylabel("coefficient of variation (log)")
    ax.set_yscale("log")
    ax.set_ylim(bottom=floor * 0.5)
    ax.set_xlim(right=ks[-1] * 1.3)
    ax.set_title("Evaluation noise vs replications")
    style_axes(ax)
    savefig(fig, path)


if __name__ == "__main__":
    main()
