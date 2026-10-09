"""Kubernetes HPA threshold rule as a per-minute replica policy.

The rule follows the upstream HPA controller: desired = ceil(current *
utilization / target), skipped when the ratio is within the 0.1 tolerance,
scale-up applied at once, scale-down held to the highest recommendation of
the last stabilization window (default 300 s). Utilization is CPU busy
fraction per replica, lam * service_demand / (cpu * replicas), the same
quantity the simulator's queueing model uses.

`HPAPolicyProblem` exposes the rule's own knobs (per-tier cpu, mem, and CPU
target) as a small real-valued problem, so any search method can tune the
rule at the same evaluation budget as the learned policies.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from . import simulator
from .topology import Topology, default_topology
from .workload import Workload, synthetic_diurnal

HPA_TOLERANCE = 0.1
TARGET_MIN, TARGET_MAX = 0.1, 0.95


class HPAPolicy:
    """One tier's HPA rule, driven through `simulate_policy`'s policy API.

    `simulate_policy` passes normalized state `(min(1, lam_prev /
    capacity_max), replicas_prev / replica_max, 0)` and maps the returned
    fraction onto `[replica_min, replica_max]` with `round`, so the fraction
    below is chosen to land exactly on the HPA's integer replica count.
    Stateful (stabilization window), so build a fresh one per realization.
    """

    def __init__(self, tier, target: float, window_min: int):
        self.rmin, self.rmax = tier.replica_min, tier.replica_max
        self.target = float(target)
        self.recent = deque(maxlen=max(1, int(window_min)))

    def act(self, state) -> float:
        # lam_prev * service_s is the busy replica count; it equals
        # state[0] * replica_max by capacity_max's definition.
        busy = state[0] * self.rmax
        current = int(round(state[1] * self.rmax))
        want = busy / self.target
        if current > 0 and abs(want / current - 1.0) <= HPA_TOLERANCE:
            desired = current
        else:
            desired = math.ceil(want - 1e-9)
        desired = max(self.rmin, min(self.rmax, desired))
        self.recent.append(desired)
        final = max(self.recent)
        if self.rmax == self.rmin:
            return 0.0
        return (final - self.rmin) / (self.rmax - self.rmin)


def build_policies(topo: Topology, targets) -> list:
    return [HPAPolicy(t, targets[i], round(t.scaledown_period_s / 60.0))
            for i, t in enumerate(topo.tiers)]


class HPAPolicyProblem(ElementwiseProblem):
    """Decision vector `[c_i, m_i, target_i]` per tier; objectives as usual.

    Mirrors `simulator.evaluate_policy` (K realizations, `base_seed + k`, the
    optional SLA hinge), but builds fresh HPA policies per realization so the
    stabilization window never carries over between realizations.
    """

    def __init__(
        self,
        topology: Optional[Topology] = None,
        workload: Optional[Workload] = None,
        k_replications: int = 5,
        base_seed: int = 0,
        sla_ms: Optional[float] = None,
    ):
        self.topo = topology if topology is not None else default_topology()
        if self.topo.node_classes is not None:
            raise ValueError("HPAPolicyProblem supports the single-location topology only")
        self.workload = workload if workload is not None else synthetic_diurnal()
        self.k_replications = int(k_replications)
        self.base_seed = int(base_seed)
        self.sla_ms = sla_ms
        self.n_eval_calls = 0
        xl, xu = [], []
        for t in self.topo.tiers:
            xl += [t.cpu_min, t.mem_min, TARGET_MIN]
            xu += [t.cpu_max, t.mem_max, TARGET_MAX]
        super().__init__(n_var=3 * self.topo.n_tiers, n_obj=3, n_ieq_constr=0,
                         xl=np.array(xl, float), xu=np.array(xu, float))

    def _evaluate(self, x, out, *args, **kwargs):
        v = np.asarray(x, dtype=float).reshape(self.topo.n_tiers, 3)
        cpu, mem, targets = v[:, 0], v[:, 1], v[:, 2]
        rows = np.empty((self.k_replications, 3))
        for k in range(self.k_replications):
            seed = self.base_seed + k
            profile = self.workload.realize(seed=seed)
            rows[k] = simulator.simulate_policy(
                self.topo, build_policies(self.topo, targets), cpu, mem,
                profile).as_array()
        if self.sla_ms is not None:
            rows[:, 0] = np.maximum(0.0, rows[:, 0] - float(self.sla_ms))
        self.n_eval_calls += 1
        out["F"] = rows.mean(axis=0)
