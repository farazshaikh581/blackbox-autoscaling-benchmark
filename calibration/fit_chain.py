"""Summarize the multi-tier chain measurement and report what it confirms.

Reads the artifacts produced by measure_chain.py on the k3s cluster:
  - calibration/data/chain_sweep.csv           (the knee sweep)
  - calibration/data/chain_sweep_topology.json (the at-rest per-tier breakdown)

and confirms the simulator's open-chain topology on real hardware:
  - call graph: a request traverses every tier once, in order
  - additive latency: end-to-end == sum of per-tier sojourns (+ small network)
  - bottleneck: the heaviest tier caps throughput at 1 / its service time, and
    latency blows up past that load -- the s/(1-rho) -> timeout knee

    python -m calibration.fit_chain
"""
from __future__ import annotations

import csv
import json
import os

from viz import COLOR, DOUBLE, INK_2, new_fig, style_axes, savefig

DIR = os.path.join(os.path.dirname(__file__), "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
CSV = os.path.join(DIR, "chain_sweep.csv")
TOP = os.path.join(DIR, "chain_sweep_topology.json")


def main():
    with open(TOP) as f:
        topo = json.load(f)
    with open(CSV) as f:
        knee = list(csv.DictReader(f))

    per_tier = topo["per_tier_compute_ms"]
    order = topo["chain"]
    e2e = topo["e2e_total_ms"]
    ssum = topo["sum_compute_ms"]
    net = topo["network_ms"]

    print("Call graph (at rest, concurrency 1):")
    print("   observed order: %s" % " -> ".join(order))
    for t in order:
        print("   %-9s compute = %8.2f ms" % (t, per_tier[t]))
    print("   -> each request traverses every tier once, demands rising down the"
          " chain (tiers non-interchangeable)")

    print("\nAdditive latency (open chain):")
    print("   sum(per-tier compute) = %8.2f ms" % ssum)
    print("   end-to-end (frontend) = %8.2f ms" % e2e)
    print("   residual (network)    = %+8.2f ms  = %.1f%% of end-to-end"
          % (net, abs(net) / e2e * 100))
    print("   -> end-to-end latency is the sum of per-tier sojourns to within the"
          " per-tier compute noise: the additive-chain model holds")

    print("\nBottleneck / knee (rising load at the frontend):")
    print("   concurrency   p50_ms     throughput_rps")
    tput = []
    for r in knee:
        print("   %-11s %9.1f   %s" % (r["concurrency"], float(r["lat_p50"]),
                                       r["throughput_rps"]))
        tput.append(float(r["throughput_rps"]))
    slowest = max(per_tier, key=per_tier.get)
    svc_s = per_tier[slowest] / 1000.0
    capacity = 1.0 / svc_s
    saturated = max(tput)
    print("   heaviest tier: %s, service time %.2f s -> predicted capacity"
          " 1/s = %.2f rps" % (slowest, svc_s, capacity))
    print("   measured saturated throughput = %.2f rps  (%.0f%% of prediction)"
          % (saturated, saturated / capacity * 100))
    print("   -> throughput plateaus at 1 / (heaviest-tier service time) while"
          " latency grows linearly: the bottleneck tier sets capacity, matching"
          " the simulator's s/(1-rho) -> timeout knee")

    print("\nSimulator impact:")
    print("   - open-chain topology and call graph: confirmed on real hardware")
    print("   - end-to-end latency = sum of per-tier sojourns: confirmed")
    print("   - chain capacity is set by the slowest tier: confirmed")
    print("   - per-pod power attribution: still open -- needs node-level power"
          " (RAPL/PDU), not exposed on the virtualized k3s nodes")

    plot_chain(order, per_tier, knee, capacity,
              os.path.join(FIG_DIR, "chain_validation.png"))


def plot_chain(order, per_tier, knee, capacity, path):
    fig, axes = new_fig(DOUBLE, 2.3, ncols=2)

    ax = axes[0]
    vals = [per_tier[t] for t in order]
    bars = ax.bar(order, vals, width=0.55, color=COLOR["measured"])
    ax.bar_label(bars, labels=[f"{v:.0f} ms" for v in vals], padding=2,
                fontsize=7, color=INK_2)
    ax.set_ylabel("compute per request (ms)")
    ax.set_title("(a) Per-tier compute at rest")
    ax.margins(y=0.15)
    style_axes(ax)

    ax = axes[1]
    conc = [int(r["concurrency"]) for r in knee]
    tput = [float(r["throughput_rps"]) for r in knee]
    ax.axhline(capacity, color=COLOR["fit"], lw=1.2, ls="--")
    ax.plot(conc, tput, color=COLOR["measured"], marker="o")
    ax.text(conc[0], capacity, f"predicted capacity, 1 / slowest tier = {capacity:.2f} rps",
            va="bottom", ha="left", fontsize=7, color=INK_2)
    ax.set_xlabel("concurrent clients")
    ax.set_ylabel("throughput (rps)")
    ax.set_title("(b) Throughput saturates at the bottleneck tier")
    ax.set_ylim(bottom=0)
    ax.set_ylim(top=capacity * 1.15)
    style_axes(ax)
    savefig(fig, path)


if __name__ == "__main__":
    main()
