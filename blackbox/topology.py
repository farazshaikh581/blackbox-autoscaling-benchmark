"""Multi-tier microservice topology and per-tier decision-variable bounds.

The benchmark decision space is a *static configuration* of an open microservice
chain: request enters tier 0, traverses every tier in sequence, and exits. Each
tier exposes three knobs:

    replicas   integer   n_i   in [replica_min, replica_max]
    cpu_limit  real      c_i   in [cpu_min,     cpu_max]      (cores per replica)
    mem_limit  real      m_i   in [mem_min,     mem_max]      (GiB per replica)

The single-tier reduction (T = 1) is intended to reproduce the factorizator /
NOMS-paper behavior; T >= 2 gives the mixed-integer, multi-tier instance the
benchmark targets.

Per-tier physical constants (service_demand, working_set_*) are CALIBRATION
targets: the values here are documented defaults, to be fitted to the host
cluster with `hey` (see `calibration/`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class TierSpec:
    """Physical parameters of one microservice tier (calibration targets)."""

    name: str
    # CPU-seconds of work per request at this tier. Sets the base service rate:
    # a replica with c_i cores serves ~ c_i / service_demand requests/sec.
    service_demand_s: float = 0.010          # CALIBRATE (s of CPU per request)
    # Memory working set (GiB) needed to serve the offered load without thrash.
    # Below this the replica degrades (throttle / OOM); modeled as a capacity hit.
    working_set_base_gib: float = 0.20       # CALIBRATE
    working_set_per_rps_gib: float = 0.0     # CALIBRATE (growth with load)

    # Decision-variable bounds for this tier.
    replica_min: int = 1
    replica_max: int = 20
    cpu_min: float = 0.10                     # cores
    cpu_max: float = 2.00                     # cores
    mem_min: float = 0.10                     # GiB
    mem_max: float = 2.00                     # GiB


@dataclass(frozen=True)
class Topology:
    """An open chain of tiers plus cluster-level constants."""

    tiers: List[TierSpec]
    node_cpu_capacity_cores: float = 16.0    # CALIBRATE (host node size)
    sla_latency_ms: float = 20.0             # NOMS paper SLA (20 ms)
    timeout_ms: float = 5000.0               # request timeout / overload plateau
    price_cpu_per_core: float = 1.0          # cost weight for provisioned CPU
    price_mem_per_gib: float = 0.25          # cost weight for provisioned memory

    @property
    def n_tiers(self) -> int:
        return len(self.tiers)

    @property
    def n_vars(self) -> int:
        return 3 * self.n_tiers  # (replicas, cpu, mem) per tier


def default_topology(n_tiers: int = 3) -> Topology:
    """A small, documented default chain (frontend -> logic -> backend).

    Demands increase down the chain so the tiers are not interchangeable, which
    keeps the mixed-integer search non-trivial. Values are placeholders pending
    `hey` calibration.
    """
    presets = [
        TierSpec("frontend", service_demand_s=0.004, working_set_base_gib=0.15),
        TierSpec("logic",    service_demand_s=0.010, working_set_base_gib=0.30,
                 working_set_per_rps_gib=0.001),
        TierSpec("backend",  service_demand_s=0.018, working_set_base_gib=0.50,
                 working_set_per_rps_gib=0.002),
    ]
    return Topology(tiers=presets[:max(1, n_tiers)])
