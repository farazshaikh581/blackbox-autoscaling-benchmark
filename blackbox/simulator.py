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
from typing import Dict, List

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


def _tier_latency_ms(lam_rps: float, capacity_rps: float, timeout_ms: float) -> float:
    """M/M/1-equivalent sojourn time for a station of aggregate rate `capacity`.

    Sojourn T = 1 / (C - lambda) seconds while lambda < C; at/above capacity the
    station is saturated and requests hit the timeout plateau. This yields the
    flat service floor (1/C) at light load and the sharp knee near rho -> 1.
    """
    if capacity_rps <= 0.0:
        return timeout_ms
    if lam_rps >= capacity_rps:
        return timeout_ms
    sojourn_s = 1.0 / (capacity_rps - lam_rps)
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
    each of length `topo.n_tiers`. `rps_profile` is per-minute realized rps.
    """
    n = np.asarray(config["replicas"], dtype=float)
    c = np.asarray(config["cpu"], dtype=float)
    m = np.asarray(config["mem"], dtype=float)

    # Per-tier effective aggregate capacity, with a memory-shortfall penalty.
    base_mu = np.array(
        [c[i] / topo.tiers[i].service_demand_s for i in range(topo.n_tiers)]
    )  # requests/sec per replica

    latencies = np.empty(len(rps_profile))
    powers = np.empty(len(rps_profile))

    for t, lam in enumerate(rps_profile):
        end_to_end_ms = 0.0
        tier_cores_used: List[float] = []
        for i in range(topo.n_tiers):
            tier = topo.tiers[i]
            # Memory shortfall degrades effective service rate (throttle/OOM).
            wss = tier.working_set_base_gib + tier.working_set_per_rps_gib * lam
            mem_factor = min(1.0, m[i] / wss) if wss > 0 else 1.0
            capacity = n[i] * base_mu[i] * mem_factor
            end_to_end_ms += _tier_latency_ms(lam, capacity, topo.timeout_ms)
            # CPU cores actually burned at this tier = total demand (n-independent).
            tier_cores_used.append(lam * tier.service_demand_s)
        latencies[t] = end_to_end_ms

        # Energy: node power from total cluster CPU util (Eq. 3), then attribute
        # per tier and sum. Idle floor is split across that tier's replicas.
        total_cores = float(sum(tier_cores_used))
        node_util = total_cores / topo.node_cpu_capacity_cores
        node_power = energy_model.estimate_node_power(node_util)
        deployment_power = 0.0
        for i in range(topo.n_tiers):
            deployment_power += energy_model.attribute_pod_power(
                pod_cpu_usage_cores=tier_cores_used[i],
                total_node_cpu_usage_cores=total_cores,
                node_power=node_power,
                num_pods=int(n[i]),
            )
        powers[t] = deployment_power

    return Objectives(
        latency_ms=float(np.mean(latencies)),
        cost=provisioned_cost(topo, config),
        energy_w=float(np.mean(powers)),
    )


def evaluate(
    topo: Topology,
    config: Dict[str, np.ndarray],
    workload: Workload,
    k_replications: int,
    base_seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Averaged, noisy oracle: mean objectives over K workload realizations.

    Returns the per-replication objective matrix (K x 3), its mean, and the
    per-objective coefficient of variation (used to pick K in the noise study).
    """
    rows = np.empty((k_replications, 3))
    for k in range(k_replications):
        profile = workload.realize(seed=base_seed + k)
        rows[k] = simulate_config(topo, config, profile).as_array()
    mean = rows.mean(axis=0)
    std = rows.std(axis=0, ddof=1) if k_replications > 1 else np.zeros(3)
    cv = np.divide(std, np.abs(mean), out=np.zeros_like(std), where=mean != 0)
    return {"samples": rows, "mean": mean, "cv": cv}
