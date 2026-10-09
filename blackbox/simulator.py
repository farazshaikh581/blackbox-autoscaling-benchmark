"""Offline deployment simulator: (config, workload) -> (latency, cost, energy).

This is the benchmark oracle's physics. It is intentionally analytic and fast
(no live cluster) so that thousands of evaluations fit in the evaluation budget,
while staying faithful in *form* to the calibrated behavior of the real system:

- Each tier is an M/M/c-style station. A replica with c_i cores serves
  mu = c_i / service_demand requests/sec; aggregate capacity C_i = n_i * mu.
- Sojourn time per tier follows the light-traffic service floor 1/C_i and blows
  up as offered load approaches capacity (the CPU-throttle "knee"), capped at
  the request timeout -> the flat-then-cliff latency curve the NOMS calibration
  found on real hardware.
- Memory below a tier's working set degrades effective capacity (throttle/OOM),
  making the continuous mem_limit knob physically meaningful.
- Energy reuses the NOMS power model (Eq. 3 + pod attribution) verbatim.

All physical constants live in `topology.TierSpec` / `Topology` and are
CALIBRATION targets. `simulate_config` is deterministic given a workload
realization; stochasticity enters only through `Workload.realize(seed)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from . import energy_model
from .topology import Topology
from .workload import Workload


@dataclass
class Objectives:
    """One evaluation's objective values (all to be MINIMIZED)."""

    latency_ms: float    # mean end-to-end latency over the day
    cost: float          # provisioned-resource cost (config-fixed)
    energy_w: float      # mean deployment power over the day

    def as_array(self) -> np.ndarray:
        return np.array([self.latency_ms, self.cost, self.energy_w], dtype=np.float64)


def _tier_latency_ms(
    lam_rps: float, service_time_s: float, capacity_rps: float, timeout_ms: float
) -> float:
    """Sojourn time for a load-balanced tier of `n` replicas.

    Each replica is an M/M/1 station with mean service time `service_time_s` and
    an even share `lam / n` of the arrivals. Its sojourn is
    `service_time / (1 - rho)`, where `rho = lam / capacity` and
    `capacity = n / service_time`. At light load the sojourn is the base service
    time (adding replicas does not shrink it, it only cuts queueing); as `rho`
    approaches one the queue blows up to the timeout plateau.
    """
    if service_time_s <= 0.0 or capacity_rps <= 0.0:
        return timeout_ms
    rho = lam_rps / capacity_rps
    if rho >= 1.0:
        return timeout_ms
    sojourn_s = service_time_s / (1.0 - rho)
    return min(timeout_ms, sojourn_s * 1000.0)


def provisioned_cost(topo: Topology, config: Dict[str, np.ndarray]) -> float:
    """Config-fixed cost: sum over tiers of replicas * (cpu price + mem price)."""
    n = config["replicas"]
    c = config["cpu"]
    m = config["mem"]
    cost = 0.0
    for i in range(topo.n_tiers):
        cost += float(n[i]) * (
            c[i] * topo.price_cpu_per_core + m[i] * topo.price_mem_per_gib
        )
    return cost


def simulate_config(
    topo: Topology,
    config: Dict[str, np.ndarray],
    rps_profile: np.ndarray,
) -> Objectives:
    """Evaluate a static config against one (already-realized) rps profile.

    `config` holds arrays keyed 'replicas' (int), 'cpu' (cores), 'mem' (GiB),
    each of length `topo.n_tiers`. `rps_profile` is per-minute realized rps. When
    `topo.node_classes` is set the config may also carry 'place' (a per-tier class
    index) and the edge/cloud path in `_simulate_config_multiclass` is used.
    """
    if topo.node_classes is not None:
        return _simulate_config_multiclass(topo, config, rps_profile)

    n = np.asarray(config["replicas"], dtype=float)
    c = np.asarray(config["cpu"], dtype=float)
    m = np.asarray(config["mem"], dtype=float)

    # Base service time per tier: s = service_demand / cpu_limit (seconds/request).
    # More CPU per replica serves each request faster.
    base_service_s = np.array(
        [topo.tiers[i].service_demand_s / c[i] for i in range(topo.n_tiers)]
    )
    demand_s = np.array([topo.tiers[i].service_demand_s for i in range(topo.n_tiers)])

    # --- Latency: sum of per-tier sojourns per minute (open chain). --------------
    latencies = np.empty(len(rps_profile))
    for t, lam in enumerate(rps_profile):
        end_to_end_ms = 0.0
        for i in range(topo.n_tiers):
            tier = topo.tiers[i]
            # Memory below the working set: Kubernetes OOM-kills the pod, so the
            # tier cannot serve. Confirmed on the cluster (48Mi < 64MB footprint
            # -> OOMKilled). Model it as a feasibility cliff, not a slowdown.
            wss = tier.working_set_base_gib + tier.working_set_per_rps_gib * lam
            if m[i] < wss:
                end_to_end_ms += topo.timeout_ms
                continue
            service_s = base_service_s[i]
            capacity = n[i] / service_s  # requests/sec the tier can serve
            end_to_end_ms += _tier_latency_ms(lam, service_s, capacity, topo.timeout_ms)
        latencies[t] = end_to_end_ms

    # --- Energy: per-node consolidation model (see energy_model). --------------
    # Bin-pack the pods onto nodes once (config-only), then sum every powered
    # node's power over the day. A pod of tier i burns min(c_i, lam*D_i/n_i) cores
    # (its share of the offered work, capped by its CPU limit); a node's power is
    # convex in the cores its pods burn, plus the idle floor for being powered on.
    counts = energy_model.pack_pods(n, c, m, topo.node_cpu_capacity_cores,
                                    topo.node_mem_capacity_gib, topo.n_tiers)
    lam = np.asarray(rps_profile, dtype=float)
    n_safe = np.where(n > 0, n, 1.0)
    offered_per_pod = np.outer(lam, demand_s / n_safe)      # (minutes x tiers)
    pod_cores = np.minimum(c, offered_per_pod)              # capped at the limit
    node_cores = pod_cores @ counts.T                       # (minutes x nodes)
    node_util = node_cores / topo.node_cpu_capacity_cores
    powers = energy_model.estimate_node_power(node_util).sum(axis=1)  # per minute

    return Objectives(
        latency_ms=float(np.mean(latencies)),
        cost=provisioned_cost(topo, config),
        energy_w=float(np.mean(powers)),
    )


def simulate_policy(
    topo: Topology,
    policies,
    cpu: np.ndarray,
    mem: np.ndarray,
    rps_profile: np.ndarray,
    place: np.ndarray | None = None,
) -> Objectives:
    """Evaluate a per-minute replica policy against one realized rps profile.

    Unlike `simulate_config`, replica counts change every minute. Each tier's
    `policies[i]` (a `policy.PolicyNet` or the tuned HPA rule) sees causal
    state, the previous minute's load and its own previous replica count, and
    picks this minute's replica level. The third state input is reserved and
    always zero. `cpu`/`mem` stay fixed per tier. Cost is accumulated per
    minute and averaged over the horizon, like latency and energy.

    With `topo.node_classes` set, each tier runs on the class `place[i]`
    (default: class 0), and packing, power and price become per class.
    """
    c = np.asarray(cpu, dtype=float)
    m = np.asarray(mem, dtype=float)
    n_tiers = topo.n_tiers

    classes = topo.node_classes
    place_arr = (np.asarray(place, dtype=int) if place is not None
                 else np.zeros(n_tiers, dtype=int))
    wan_ms = 0.0
    if classes is not None:
        access_ms = classes[int(place_arr[0])].access_rtt_ms
        wan_ms = access_ms + sum(
            topo.wan_rtt(int(place_arr[i]), int(place_arr[i + 1]))
            for i in range(n_tiers - 1)
        )

    base_service_s = np.array(
        [topo.tiers[i].service_demand_s / c[i] for i in range(n_tiers)]
    )
    demand_s = np.array([topo.tiers[i].service_demand_s for i in range(n_tiers)])
    capacity_max = np.array(
        [topo.tiers[i].replica_max / base_service_s[i] for i in range(n_tiers)]
    )

    replicas_prev = np.array(
        [topo.tiers[i].replica_min for i in range(n_tiers)], dtype=int
    )
    lam_prev = np.zeros(n_tiers)

    latencies = np.empty(len(rps_profile))
    costs = np.empty(len(rps_profile))
    powers = np.empty(len(rps_profile))
    # Packing depends only on the replica counts (cpu/mem are fixed), and a
    # policy often holds the same counts for many minutes, so cache it.
    pack_cache: Dict[tuple, tuple] = {}

    for t, lam in enumerate(rps_profile):
        n_t = np.empty(n_tiers, dtype=int)
        for i in range(n_tiers):
            tier = topo.tiers[i]
            state = (
                min(1.0, lam_prev[i] / capacity_max[i]) if capacity_max[i] > 0 else 0.0,
                replicas_prev[i] / tier.replica_max if tier.replica_max > 0 else 0.0,
                0.0,
            )
            frac = policies[i].act(state)
            target = tier.replica_min + frac * (tier.replica_max - tier.replica_min)
            n_i = int(round(target))
            n_i = max(tier.replica_min, min(tier.replica_max, n_i))
            n_t[i] = n_i

        pack_key = tuple(n_t.tolist())
        cached = pack_cache.get(pack_key)
        if cached is None:
            if classes is not None:
                packed = energy_model.pack_pods_by_class(
                    n_t, c, m, place_arr, classes, n_tiers)
                if packed:
                    node_cls = np.array([k for k, _ in packed], dtype=int)
                    counts = np.vstack([row for _, row in packed])
                else:
                    node_cls = np.zeros(0, dtype=int)
                    counts = np.zeros((0, n_tiers), dtype=int)
                node_cap = np.array([classes[k].cpu_capacity_cores for k in node_cls])
                node_p_idle = np.array([classes[k].p_idle for k in node_cls])
                node_p_max = np.array([classes[k].p_max for k in node_cls])
                node_alpha = np.array([classes[k].alpha for k in node_cls])
            else:
                counts = energy_model.pack_pods(n_t, c, m, topo.node_cpu_capacity_cores,
                                                topo.node_mem_capacity_gib, n_tiers)
                node_cap = node_p_idle = node_p_max = node_alpha = None
            cached = (counts, node_cap, node_p_idle, node_p_max, node_alpha)
            pack_cache[pack_key] = cached
        counts, node_cap, node_p_idle, node_p_max, node_alpha = cached

        end_to_end_ms = 0.0
        cost_t = 0.0
        for i in range(n_tiers):
            tier = topo.tiers[i]
            wss = tier.working_set_base_gib + tier.working_set_per_rps_gib * lam
            if classes is not None:
                nc_i = classes[int(place_arr[i])]
                cost_t += float(n_t[i]) * (
                    c[i] * nc_i.price_cpu_per_core + m[i] * nc_i.price_mem_per_gib
                )
            else:
                cost_t += float(n_t[i]) * (
                    c[i] * topo.price_cpu_per_core + m[i] * topo.price_mem_per_gib
                )
            if m[i] < wss:
                end_to_end_ms += topo.timeout_ms
                continue
            service_s = base_service_s[i]
            if n_t[i] == 0:
                end_to_end_ms += topo.timeout_ms if lam > 0 else 0.0
                continue
            capacity = n_t[i] / service_s
            end_to_end_ms += _tier_latency_ms(lam, service_s, capacity, topo.timeout_ms)
        if classes is not None:
            for i in range(n_tiers - 1):
                if int(place_arr[i]) != int(place_arr[i + 1]):
                    cost_t += topo.egress_cost_per_hop
            end_to_end_ms += wan_ms
        latencies[t] = end_to_end_ms
        costs[t] = cost_t

        n_safe = np.where(n_t > 0, n_t, 1.0)
        offered_per_pod = lam * demand_s / n_safe
        pod_cores = np.minimum(c, offered_per_pod)
        if counts.shape[0] > 0:
            node_cores = pod_cores @ counts.T
            if classes is not None:
                u = np.clip(node_cores / node_cap, 0.0, 1.0)
                powers[t] = float((node_p_idle + (node_p_max - node_p_idle)
                                   * (u ** node_alpha)).sum())
            else:
                node_util = node_cores / topo.node_cpu_capacity_cores
                powers[t] = float(energy_model.estimate_node_power(node_util).sum())
        else:
            powers[t] = 0.0

        replicas_prev = n_t
        lam_prev[:] = lam

    return Objectives(
        latency_ms=float(np.mean(latencies)),
        cost=float(np.mean(costs)),
        energy_w=float(np.mean(powers)),
    )


def evaluate_policy(
    topo: Topology,
    policies,
    cpu: np.ndarray,
    mem: np.ndarray,
    workload: Workload,
    k_replications: int,
    base_seed: int = 0,
    sla_ms: float | None = None,
) -> Dict[str, np.ndarray]:
    """Averaged, noisy oracle for a policy: mean objectives over K realizations.

    Mirrors `evaluate` (below) but calls `simulate_policy` instead of
    `simulate_config`, so it drops into the same held-out-day protocol and
    SLA-hinge framing (`experiments/_common.py`).
    """
    rows = np.empty((k_replications, 3))
    for k in range(k_replications):
        seed = base_seed + k
        profile = workload.realize(seed=seed)
        rows[k] = simulate_policy(topo, policies, cpu, mem, profile).as_array()
    if sla_ms is not None:
        rows[:, 0] = np.maximum(0.0, rows[:, 0] - float(sla_ms))
    mean = rows.mean(axis=0)
    std = rows.std(axis=0, ddof=1) if k_replications > 1 else np.zeros(3)
    cv = np.divide(std, np.abs(mean), out=np.zeros_like(std), where=mean != 0)
    return {"samples": rows, "mean": mean, "cv": cv}


def _simulate_config_multiclass(
    topo: Topology,
    config: Dict[str, np.ndarray],
    rps_profile: np.ndarray,
) -> Objectives:
    """Edge/cloud path: per-tier placement onto heterogeneous node classes.

    Adds three things to the single-location model:
    each tier runs on the node class given by `config['place']` (default all class
    0); consecutive tiers on different classes add the WAN RTT to latency and a
    fixed egress cost to cost; energy packs and powers each class's nodes with that
    class's own capacity and power curve. With one class and all tiers on it, it
    matches `simulate_config`.
    """
    classes = topo.node_classes
    n_tiers = topo.n_tiers
    n = np.asarray(config["replicas"], dtype=float)
    c = np.asarray(config["cpu"], dtype=float)
    m = np.asarray(config["mem"], dtype=float)
    place = np.asarray(config.get("place", np.zeros(n_tiers, dtype=int)), dtype=int)

    base_service_s = np.array(
        [topo.tiers[i].service_demand_s / c[i] for i in range(n_tiers)]
    )
    demand_s = np.array([topo.tiers[i].service_demand_s for i in range(n_tiers)])

    # Fixed network latency, independent of load: the client access RTT to the
    # entry tier's class (edge is near the user, cloud is far), plus the WAN RTT of
    # every cross-class hop along the chain.
    access_ms = classes[int(place[0])].access_rtt_ms
    wan_ms = access_ms + sum(topo.wan_rtt(int(place[i]), int(place[i + 1]))
                             for i in range(n_tiers - 1))

    # --- Latency: per-tier sojourns (as single-location) plus the WAN hops. ------
    latencies = np.empty(len(rps_profile))
    for t, lam in enumerate(rps_profile):
        end_to_end_ms = 0.0
        for i in range(n_tiers):
            tier = topo.tiers[i]
            wss = tier.working_set_base_gib + tier.working_set_per_rps_gib * lam
            if m[i] < wss:
                end_to_end_ms += topo.timeout_ms
                continue
            service_s = base_service_s[i]
            capacity = n[i] / service_s
            end_to_end_ms += _tier_latency_ms(lam, service_s, capacity, topo.timeout_ms)
        latencies[t] = end_to_end_ms + wan_ms

    # --- Energy: pack per class, power each node with its class's curve. ---------
    packed = energy_model.pack_pods_by_class(n, c, m, place, classes, n_tiers)
    lam = np.asarray(rps_profile, dtype=float)
    n_safe = np.where(n > 0, n, 1.0)
    offered_per_pod = np.outer(lam, demand_s / n_safe)      # (minutes x tiers)
    pod_cores = np.minimum(c, offered_per_pod)              # capped at the limit
    powers = np.zeros(len(lam))
    for k, counts in packed:
        nc = classes[k]
        node_cores = pod_cores @ counts                    # (minutes,)
        u = np.clip(node_cores / nc.cpu_capacity_cores, 0.0, 1.0)
        powers += nc.p_idle + (nc.p_max - nc.p_idle) * (u ** nc.alpha)

    # --- Cost: per-class prices plus a fixed egress cost per cross-class hop. ----
    cost = 0.0
    for i in range(n_tiers):
        nc = classes[int(place[i])]
        cost += float(n[i]) * (c[i] * nc.price_cpu_per_core + m[i] * nc.price_mem_per_gib)
    for i in range(n_tiers - 1):
        if int(place[i]) != int(place[i + 1]):
            cost += topo.egress_cost_per_hop

    return Objectives(
        latency_ms=float(np.mean(latencies)),
        cost=float(cost),
        energy_w=float(np.mean(powers)),
    )


def evaluate(
    topo: Topology,
    config: Dict[str, np.ndarray],
    workload: Workload,
    k_replications: int,
    base_seed: int = 0,
    sla_ms: float | None = None,
) -> Dict[str, np.ndarray]:
    """Averaged, noisy oracle: mean objectives over K workload realizations.

    Returns the per-replication objective matrix (K x 3), its mean, and the
    per-objective coefficient of variation (used to pick K in the noise study).

    `sla_ms` is an optional objective-framing transform (not physics): when set,
    the latency objective becomes the SLA-hinge `max(0, latency_ms - sla_ms)`,
    applied per replication before averaging. Below the SLA all latencies read as
    0, so the search stops spending Pareto points on shaving already-compliant
    latency and instead trades cost/energy within the feasible region; above it
    the objective grows with the violation. `None` keeps the raw latency (the
    default framing). Applied here, the one call every algorithm funnels through,
    so NSGA-II / MOEA/D / MORL all see the identical transformed objective.
    """
    rows = np.empty((k_replications, 3))
    for k in range(k_replications):
        profile = workload.realize(seed=base_seed + k)
        rows[k] = simulate_config(topo, config, profile).as_array()
    if sla_ms is not None:
        rows[:, 0] = np.maximum(0.0, rows[:, 0] - float(sla_ms))
    mean = rows.mean(axis=0)
    std = rows.std(axis=0, ddof=1) if k_replications > 1 else np.zeros(3)
    cv = np.divide(std, np.abs(mean), out=np.zeros_like(std), where=mean != 0)
    return {"samples": rows, "mean": mean, "cv": cv}
