"""Black-box multi-objective autoscaling benchmark.

A static-configuration reformulation of the NOMS 2026 autoscaling problem
as an expensive, noisy, mixed-integer multi-objective optimization instance:

    decide  per-tier (replicas, cpu_limit, mem_limit)
    minimize (latency_ms, cost, energy_W)

evaluated by a fast offline simulator (K-replication averaged oracle) that is
calibrated/validated against a real cluster with `hey`.
"""
from .topology import Topology, TierSpec, default_topology
from .workload import Workload, synthetic_diurnal, from_azure_trace
from .simulator import Objectives, simulate_config, evaluate
from .oracle import AutoscalingProblem, RealEncodedAutoscalingProblem, x_to_config

__all__ = [
    "Topology",
    "TierSpec",
    "default_topology",
    "Workload",
    "synthetic_diurnal",
    "from_azure_trace",
    "Objectives",
    "simulate_config",
    "evaluate",
    "AutoscalingProblem",
    "RealEncodedAutoscalingProblem",
    "x_to_config",
]
