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
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class TierSpec:
    """Physical parameters of one microservice tier (calibration targets)."""

    name: str
    # CPU-seconds of work per request at this tier. The base service time is
    # s = service_demand / cpu_limit, so a replica with c_i cores serves each
    # request in service_demand / c_i seconds. Anchored to the factorizator:
    # a ~4 ms light-load latency at a 0.5-core limit gives service_demand ~= 0.002.
    service_demand_s: float = 0.002          # cpu-seconds per request
    # Memory working set (GiB) below which the replica throttles (service time
    # stretches). The factorizator ran on a 0.25 GiB limit, so ~0.2 GiB base.
    working_set_base_gib: float = 0.20
    working_set_per_rps_gib: float = 0.0     # growth with load (host sweep)

    # Decision-variable bounds for this tier. Defaults span the factorizator
    # operating point (0.5 core, 0.25 GiB) with headroom for the search.
    replica_min: int = 1
    replica_max: int = 30
    cpu_min: float = 0.10                     # cores
    cpu_max: float = 2.00                     # cores
    mem_min: float = 0.10                     # GiB
    mem_max: float = 2.00                     # GiB

    # Kubernetes HPA scale-down stabilization window (default 300 s), used by
    # the tuned HPA baseline (blackbox/hpa.py).
    scaledown_period_s: float = 300.0


@dataclass(frozen=True)
class NodeClass:
    """A class of physical node at one location (for the edge/cloud extension).

    Placement assigns each tier to a node class. Pods of tiers on the same class
    bin-pack onto that class's nodes, each powered node of the class draws
    `p_idle + (p_max - p_idle) * u**alpha`, and cost uses the class's prices. Edge
    classes are small, expensive per core, and lower power; cloud classes are
    large and cheap. The values here are documented defaults, to be calibrated to
    the testbed (`calibration/cluster/measure_edge_node.py`).
    """

    name: str = "default"
    cpu_capacity_cores: float = 8.0
    mem_capacity_gib: float = 16.0
    p_idle: float = 31.6
    p_max: float = 85.6
    alpha: float = 1.5
    price_cpu_per_core: float = 1.0
    price_mem_per_gib: float = 0.25
    # Client access RTT (ms): the round trip from the user to a node of this class,
    # added once at the entry tier. This is the point of edge computing: an edge
    # class is close to the user (low access RTT), a cloud class is far. It is what
    # lets the edge class earn a place on the front (a latency-vs-cost trade);
    # without it the cheaper, larger cloud class dominates every objective.
    access_rtt_ms: float = 0.0


@dataclass(frozen=True)
class Topology:
    """An open chain of tiers plus cluster-level constants."""

    tiers: List[TierSpec]
    # Node size: an assumed general-purpose VM size (8 cores, 16 GiB). The
    # power curve is applied as a function of utilization.
    node_cpu_capacity_cores: float = 8.0
    # Node memory, for the pod bin-packing that drives the energy model: pods are
    # scheduled onto nodes by CPU *and* memory request, and each powered node
    # draws idle + dynamic power. 8 cores / 16 GiB is a typical general-purpose VM
    # and makes both dimensions able to bind (pods span cpu:mem 0.05..20). CALIBRATE.
    node_mem_capacity_gib: float = 16.0
    sla_latency_ms: float = 20.0             # NOMS paper SLA (20 ms)
    timeout_ms: float = 5000.0               # request timeout / overload plateau
    price_cpu_per_core: float = 1.0          # cost weight for provisioned CPU
    price_mem_per_gib: float = 0.25          # cost weight for provisioned memory

    # --- Edge/cloud extension (opt-in). None keeps single-location behavior:
    # the simulator uses the scalar node_*/price_* fields above and the
    # energy_model module constants, exactly as before. When node_classes is set,
    # placement assigns each tier to a class, packing/power/cost go per class, and
    # wan_rtt_ms / egress_cost_per_hop apply across differently-placed tiers.
    node_classes: Optional[Tuple[NodeClass, ...]] = None
    wan_rtt_ms: Optional[Tuple[Tuple[float, ...], ...]] = None  # class x class RTT (ms)
    egress_cost_per_hop: float = 0.0         # fixed cost per cross-class consecutive hop


    @property
    def n_tiers(self) -> int:
        return len(self.tiers)

    @property
    def n_vars(self) -> int:
        return 3 * self.n_tiers  # (replicas, cpu, mem) per tier

    @property
    def n_classes(self) -> int:
        return 1 if self.node_classes is None else len(self.node_classes)

    def wan_rtt(self, a: int, b: int) -> float:
        """RTT (ms) between node classes `a` and `b`; 0 within a class or if unset."""
        if self.wan_rtt_ms is None or a == b:
            return 0.0
        return float(self.wan_rtt_ms[a][b])


def default_topology(n_tiers: int = 3) -> Topology:
    """A small open chain: frontend -> logic -> backend.

    A request enters the frontend and traverses every tier once. Per-tier demands
    increase down the chain so the tiers are not interchangeable, which keeps the
    mixed-integer search non-trivial. The demands sit in the factorizator's
    calibrated range (base service time a few ms per tier at a 0.5-core limit);
    their order matches a three-tier chain measured on the testbed
    (`calibration/measure_chain.py`). The single-tier reduction (n_tiers=1) reproduces the
    factorizator directly.
    """
    presets = [
        TierSpec("frontend", service_demand_s=0.001, working_set_base_gib=0.15),
        TierSpec("logic",    service_demand_s=0.002, working_set_base_gib=0.30,
                 working_set_per_rps_gib=0.001),
        TierSpec("backend",  service_demand_s=0.0035, working_set_base_gib=0.50,
                 working_set_per_rps_gib=0.002),
    ]
    if n_tiers == 1:
        # Single-tier factorizator: 0.5-core operating point, ~4 ms base latency.
        return Topology(tiers=[TierSpec("factorizator", service_demand_s=0.002,
                                        working_set_base_gib=0.20)])
    return Topology(tiers=presets[:max(1, n_tiers)])


def edge_cloud_topology(n_tiers: int = 3) -> Topology:
    """The default chain with two node classes: edge (index 0) and cloud (index 1).

    Placement (a per-tier decision variable) chooses
    a class for each tier. The edge class is small, expensive per core, and lower
    power (close to the user); the cloud class is large and cheap. A request that
    crosses between classes on consecutive tiers pays the WAN RTT (added to
    latency) and a fixed egress cost. The edge class's power curve is measured,
    not guessed. `calibration/cluster/measure_edge_node.py` emulates a 4-core
    class on real bare-metal RAPL. It pins busy workers to 4 of a physical
    node's 6 cores and sweeps u = 0..1 relative to that emulated capacity. See
    the fit (R2=0.999). The cloud class is an assumed default; no 16-core
    node exists on the testbed to measure it.
    """
    wan_rtt = 30.0          # one-way edge<->cloud delay per crossing (ms)
    price_premium = 2.0     # edge price per core and per GiB, relative to cloud
    edge_alpha = 0.7        # measured edge power-curve exponent

    edge = NodeClass(
        name="edge", cpu_capacity_cores=4.0, mem_capacity_gib=8.0,
        p_idle=2.2, p_max=60.8, alpha=edge_alpha,
        price_cpu_per_core=1.0 * price_premium, price_mem_per_gib=0.25 * price_premium,
        access_rtt_ms=5.0,      # close to the user
    )
    cloud = NodeClass(
        name="cloud", cpu_capacity_cores=16.0, mem_capacity_gib=32.0,
        p_idle=45.0, p_max=200.0, alpha=1.5,
        price_cpu_per_core=1.0, price_mem_per_gib=0.25,
        access_rtt_ms=40.0,     # a distant region
    )
    base = default_topology(n_tiers)
    return Topology(
        tiers=base.tiers,
        sla_latency_ms=base.sla_latency_ms,
        timeout_ms=base.timeout_ms,
        node_classes=(edge, cloud),
        wan_rtt_ms=((0.0, wan_rtt), (wan_rtt, 0.0)),
        egress_cost_per_hop=0.5,
    )
