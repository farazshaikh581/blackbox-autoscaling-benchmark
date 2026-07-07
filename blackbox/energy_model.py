"""Server power / pod-power attribution model.

Ported verbatim (in intent) from the NOMS 2026 reference implementation
(`pareto-optimal_autoscaling/energy_model.py`) so the black-box benchmark scores
energy on the *same* definitions as the MORL baseline it is compared against.

Paper Eq. (3):   P_node(u) = P_idle + (P_max - P_idle) * u^alpha
Pod attribution: dynamic node power split by CPU share, idle power split evenly
across the active pods.

These constants are hardware parameters. They are the same numbers used in the
NOMS paper and can be re-calibrated per host (see `calibration/`).
"""
from __future__ import annotations

import numpy as np

# --- Server power model parameters (NOMS 2026 reference values) ---
P_IDLE = 50.0   # Watts, idle draw
P_MAX = 250.0   # Watts, peak draw
ALPHA = 2.0     # convexity exponent


def estimate_node_power(node_cpu_util_fraction: float) -> float:
    """Paper Eq. (3): node power as a convex function of normalized CPU util."""
    u = float(np.clip(node_cpu_util_fraction, 0.0, 1.0))
    return P_IDLE + (P_MAX - P_IDLE) * (u ** ALPHA)


def attribute_pod_power(
    pod_cpu_usage_cores: float,
    total_node_cpu_usage_cores: float,
    node_power: float,
    num_pods: int,
) -> float:
    """Attribute a share of node power to a pod group.

    Dynamic node power (node_power - P_idle) is split proportionally to CPU
    share; the idle floor is split evenly across the `num_pods` active pods.
    """
    total_node_cpu_usage_cores = float(total_node_cpu_usage_cores)
    pod_cpu_usage_cores = float(pod_cpu_usage_cores)
    num_pods = int(num_pods)

    if total_node_cpu_usage_cores <= 0.0:
        return 0.0

    dynamic_power = max(0.0, float(node_power) - P_IDLE)
    pod_share = float(np.clip(pod_cpu_usage_cores / total_node_cpu_usage_cores, 0.0, 1.0))
    pod_dynamic_power = dynamic_power * pod_share

    pod_idle_power_share = P_IDLE / num_pods if num_pods > 0 else 0.0
    return pod_dynamic_power + pod_idle_power_share
