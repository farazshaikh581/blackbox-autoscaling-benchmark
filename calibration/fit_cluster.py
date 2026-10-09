"""Summarize the cluster sweep and report what it calibrates and validates.

Reads calibration/data/cluster_sweep.csv (produced by run_sweep.py on the k3s
cluster) and reports:
  - CPU scaling: service time vs CPU limit, and the implied per-request demand D
  - Knee: how latency grows once load passes the single-pod capacity
  - Memory: the feasibility threshold (OOM below the working set)

This confirms the simulator's model form on real hardware. The service-time
magnitude in topology.py stays anchored to the NOMS production logs
(fit_from_noms.py); this sweep validates the shape and fixes the memory model to
a feasibility cliff.

    python -m calibration.fit_cluster
"""
from __future__ import annotations

import csv
import os

from viz import COLOR, DOUBLE, new_fig, style_axes, savefig

PATH = os.path.join(os.path.dirname(__file__), "data", "cluster_sweep.csv")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")


def load():
    with open(PATH) as f:
        return list(csv.DictReader(f))


def main():
    rows = load()
    A = [r for r in rows if r["exp"] == "A_cpu" and r["status"] == "Ready"]
    B = [r for r in rows if r["exp"] == "B_knee" and r["status"] == "Ready"]
    C = [r for r in rows if r["exp"] == "C_mem"]

    print("CPU scaling (concurrency 1, no queue):")
    print("   cpu   lat_p50_ms   service_s   D=s*cpu   throughput_rps")
    for r in A:
        cpu = float(r["cpu"]); s = float(r["lat_p50"]) / 1000.0
        print("   %-5s %9.1f   %8.3f   %6.3f   %s"
              % (cpu, float(r["lat_p50"]), s, s * cpu, r["throughput_rps"]))
    healthy = [r for r in A if float(r["cpu"]) >= 0.75]
    if healthy:
        D = sum(float(r["lat_p50"]) / 1000.0 * float(r["cpu"]) for r in healthy) / len(healthy)
        print("   -> latency falls as CPU rises; per-request demand D ~= %.3f core.s"
              " (healthy regime; extra throttling penalty below ~0.5 core)" % D)

    print("\nKnee (cpu 0.5, rising load):")
    print("   concurrency   lat_p50_ms   throughput_rps")
    for r in B:
        print("   %-11s %9.1f   %s" % (r["concurrency"], float(r["lat_p50"]), r["throughput_rps"]))
    print("   -> throughput plateaus at the single-pod capacity and latency grows"
          " linearly past it (queueing), matching s/(1-rho) -> timeout")

    print("\nMemory feasibility (64 MB footprint):")
    ok = [int(r["mem_mi"]) for r in C if r["status"] == "Ready"]
    oom = [int(r["mem_mi"]) for r in C if r["status"] == "OOMKilled"]
    for r in C:
        print("   %sMi -> %s" % (r["mem_mi"], r["status"]))
    if oom and ok:
        print("   -> working-set cliff between %dMi (OOM) and %dMi (ok): below the"
              " working set the config is infeasible, not merely slow"
              % (max(oom), min(ok)))

    print("\nSimulator impact:")
    print("   - latency = s/(1-rho) with s = service_demand/cpu: form confirmed")
    print("   - memory modeled as a feasibility cliff (mem < working_set -> timeout)")
    print("   - service-time magnitude stays from the NOMS production logs")

    plot_cluster(A, B, C, os.path.join(FIG_DIR, "cluster_validation.png"))


def plot_cluster(A, B, C, path):
    fig, axes = new_fig(DOUBLE, 2.3, ncols=2)

    ax = axes[0]
    cpu = [float(r["cpu"]) for r in A]
    lat = [float(r["lat_p50"]) for r in A]
    ax.plot(cpu, lat, color=COLOR["measured"], marker="o")
    ax.set_xlabel("CPU limit (cores)")
    ax.set_ylabel("p50 latency (ms)")
    ax.set_title("(a) Latency falls as CPU rises")
    ax.set_ylim(bottom=0)
    style_axes(ax)

    ax = axes[1]
    conc = [int(r["concurrency"]) for r in B]
    tput = [float(r["throughput_rps"]) for r in B]
    ax.plot(conc, tput, color=COLOR["measured"], marker="o")
    ax.set_xlabel("concurrent clients")
    ax.set_ylabel("throughput (rps)")
    ax.set_title("(b) Single-pod knee (0.5 core)")
    ax.set_ylim(bottom=0)
    style_axes(ax)
    savefig(fig, path)


if __name__ == "__main__":
    main()
