"""Black-box multi-objective autoscaling benchmark.

A static-configuration reformulation of the NOMS 2026 autoscaling problem
as an expensive, noisy, mixed-integer multi-objective optimization instance:

    decide  per-tier (replicas, cpu_limit, mem_limit)
    minimize (latency_ms, cost, energy_W)

evaluated by a fast offline simulator (K-replication averaged oracle) that is
calibrated/validated against a real cluster with `hey`.
"""
from .topology import (
    Topology, TierSpec, NodeClass, default_topology, edge_cloud_topology,
)
from .workload import Workload, synthetic_diurnal, from_azure_trace
from .simulator import Objectives, simulate_config, evaluate, simulate_policy, evaluate_policy
from .oracle import (
    AutoscalingProblem, RealEncodedAutoscalingProblem, DynamicAutoscalingProblem,
    x_to_config,
)
from .policy import PolicyNet, decode as decode_policy, weight_vector_size

__all__ = [
    "Topology",
    "TierSpec",
    "NodeClass",
    "default_topology",
    "edge_cloud_topology",
    "Workload",
    "synthetic_diurnal",
    "from_azure_trace",
    "Objectives",
    "simulate_config",
    "evaluate",
    "simulate_policy",
    "evaluate_policy",
    "AutoscalingProblem",
    "RealEncodedAutoscalingProblem",
    "DynamicAutoscalingProblem",
    "x_to_config",
    "PolicyNet",
    "decode_policy",
    "weight_vector_size",
]
