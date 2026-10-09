"""Server power model with per-node bin-packing (consolidation-aware).

The deployment's pods are scheduled onto physical nodes by their CPU *and* memory
requests, and every powered-on node draws power

    P_node(u) = P_idle + (P_max - P_idle) * u^alpha          (paper Eq. 3)

where `u` is that node's CPU utilization (actual cores burned / node capacity).
The deployment's power is the sum over its powered nodes, so total energy is

    E = P_idle * N_nodes  +  (P_max - P_idle) * sum_nodes u_node^alpha,

i.e. it is driven by the *number of powered nodes* (the idle floor) plus a convex
dynamic term. This is the standard energy-aware-scheduling / server-consolidation
model: adding replicas or growing a pod's requests spills pods onto more nodes and
raises energy, while consolidation lowers it. It supersedes the earlier
single-node pod-attribution model, under which energy was cpu/mem-independent and
*fell* with replica count (one node's idle split across pods) -- physically
backwards.

`P_max`, `P_idle`, and `alpha` are jointly MEASURED on bare-metal Intel RAPL
main-cluster nodes r1 and r2 (`calibration/data/edge_node_measured_curve_r1.json`,
`_r2.json`): a full 0 to 100% utilization sweep (21 points/node, 3 passes each,
package `energy_uj`), free-fit to the form above. r1 and r2 agree closely
(P_idle 2.22/2.23 W, P_max 84.8/85.2 W, alpha 0.741/0.742, R^2 >= 0.997); the
constants below are their average. r0 (control-plane) was measured too but
excluded: its P_max runs ~11 W lower, likely etcd/API-server contention for
cache and memory bandwidth, making it unrepresentative for calibration.

`alpha < 1` means power rises steeply just above zero utilization rather than
gradually -- this is NOT the same failure mode as using a desktop's deep-idle
floor (~9.6 W with no workload at all): a node hosting even one pod still
draws far more than idle (this curve gives ~38 W at 2 of 6 cores busy, matching
the cluster's own earlier ~36 W anchor point), so lightly-loaded active nodes
are still charged correctly and the model still rewards consolidation over
spreading.

Absolute watts are these hosts' scale; the Pareto *ordering* is invariant to
absolute scale, but P_idle/P_max and alpha are SHAPE parameters that DO affect
ordering -- which is precisely why the in-service floor matters.
"""
from __future__ import annotations

import os

import numpy as np

# --- Server power model parameters (MEASURED on bare-metal Intel RAPL) ---
# Defaults are the measured constants. BB_P_IDLE / BB_P_MAX / BB_ALPHA env vars
# override them for sensitivity checks; unset -> measured. Average of the
# full 0-100% utilization sweep fits on r1 and r2 (see module docstring).
P_IDLE = float(os.environ.get("BB_P_IDLE", 2.23))  # in-service node floor: fixed cost of a
                #   node hosting workload, avg of r1/r2 full-sweep fit (idle frac 0.026;
                #   NOT the ~9.6 W desktop deep-idle -- see module docstring)
P_MAX = float(os.environ.get("BB_P_MAX", 85.0))    # peak draw at full CPU, avg of r1/r2
                #   full-sweep fit (84.83, 85.19)
ALPHA = float(os.environ.get("BB_ALPHA", 0.74))    # convexity, avg of r1/r2 full-sweep fit
                #   (0.741, 0.742; R^2 >= 0.997) -- sub-linear, steep rise near u=0


def estimate_node_power(node_cpu_util_fraction):
    """Paper Eq. (3): a node's power as a convex function of its CPU utilization.

    Accepts a scalar or an array of utilizations (clipped to [0, 1]). A powered
    node at zero load still draws the idle floor `P_IDLE`.
    """
    u = np.clip(np.asarray(node_cpu_util_fraction, dtype=float), 0.0, 1.0)
    return P_IDLE + (P_MAX - P_IDLE) * (u ** ALPHA)


def pack_pods(replicas, cpu, mem, node_cpu, node_mem, n_tiers):
    """Bin-pack the deployment's pods onto nodes; return per-node per-tier counts.

    Each tier `i` contributes `replicas[i]` identical pods, each requesting
    `cpu[i]` cores and `mem[i]` GiB. Pods are placed first-fit-decreasing by CPU
    request (the usual scheduler heuristic), a pod fitting a node only if *both*
    its CPU and memory requests fit the node's remaining capacity. A pod larger
    than a whole node still gets its own node (utilization is clipped later).

    Returns an (N_nodes x n_tiers) integer array: entry [k, i] is how many tier-i
    pods sit on node k. Packing depends only on the configuration, so the caller
    computes it once and reuses it across the per-minute load profile.
    """
    replicas = np.asarray(replicas, dtype=int)
    cpu = np.asarray(cpu, dtype=float)
    mem = np.asarray(mem, dtype=float)

    # Pod list as (tier, cpu, mem), heaviest CPU first (first-fit-decreasing).
    pods = [(i, float(cpu[i]), float(mem[i]))
            for i in range(n_tiers) for _ in range(int(replicas[i]))]
    pods.sort(key=lambda p: p[1], reverse=True)

    nodes: list = []       # each: [remaining_cpu, remaining_mem, counts(n_tiers)]
    for tier, c, m in pods:
        placed = False
        for nd in nodes:
            if nd[0] >= c and nd[1] >= m:
                nd[0] -= c
                nd[1] -= m
                nd[2][tier] += 1
                placed = True
                break
        if not placed:
            counts = np.zeros(n_tiers, dtype=int)
            counts[tier] = 1
            nodes.append([node_cpu - c, node_mem - m, counts])

    if not nodes:  # no pods (all replicas zero) -> no powered nodes
        return np.zeros((0, n_tiers), dtype=int)
    return np.vstack([nd[2] for nd in nodes])


def pack_pods_by_class(replicas, cpu, mem, place, classes, n_tiers):
    """Bin-pack pods per placement class (for the edge/cloud extension).

    `place[i]` is the node-class index of tier `i`. Pods of tiers sharing a class
    pack onto that class's own nodes (its cpu/mem capacity), independently of the
    other classes. Reuses `pack_pods` once per class. Returns a list of
    `(class_index, counts)`, one entry per powered node, where `counts` is the
    per-tier pod count on that node. With a single class and all tiers placed on
    it, this reduces to `pack_pods`.
    """
    replicas = np.asarray(replicas, dtype=int)
    place = np.asarray(place, dtype=int)
    out = []
    for k, nc in enumerate(classes):
        reps_k = np.where(place == k, replicas, 0)
        if reps_k.sum() == 0:
            continue
        node_counts = pack_pods(reps_k, cpu, mem,
                                nc.cpu_capacity_cores, nc.mem_capacity_gib, n_tiers)
        for row in node_counts:
            out.append((k, row))
    return out
