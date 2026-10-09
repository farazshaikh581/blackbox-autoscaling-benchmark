"""pymoo MixedVariableProblem wrapper: the ROAR-NET black-box oracle.

Decision vector (per tier i in 0..T-1):
    n_i  Integer  replicas          in [replica_min, replica_max]
    c_i  Real     cpu_limit (cores) in [cpu_min,     cpu_max]
    m_i  Real     mem_limit (GiB)   in [mem_min,     mem_max]

Objectives (all minimized):  [ latency_ms, cost, energy_W ]

The oracle is the K-replication averaged simulator (`simulator.evaluate`): for a
candidate it realizes the workload K times with distinct seeds and returns the
mean objective vector. Given a fixed `base_seed` the oracle is deterministic and
reproducible; noise is characterized by varying `base_seed` / K.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.core.variable import Integer, Real

from . import policy, simulator
from .topology import Topology, default_topology
from .workload import Workload, synthetic_diurnal


def _var_names(n_tiers: int):
    return (
        [f"n_{i}" for i in range(n_tiers)],
        [f"c_{i}" for i in range(n_tiers)],
        [f"m_{i}" for i in range(n_tiers)],
    )


def _place_names(n_tiers: int):
    return [f"p_{i}" for i in range(n_tiers)]


def x_to_config(x: Dict[str, float], n_tiers: int, n_classes: int = 1) -> Dict[str, np.ndarray]:
    """Convert a pymoo mixed-variable sample dict to a simulator config dict.

    With `n_classes > 1` (the edge/cloud topology) the dict also carries a per-tier
    placement variable `p_i`, decoded into the `place` array the multi-class
    simulator path reads.
    """
    n_names, c_names, m_names = _var_names(n_tiers)
    config = {
        "replicas": np.array([int(x[k]) for k in n_names], dtype=int),
        "cpu": np.array([float(x[k]) for k in c_names], dtype=float),
        "mem": np.array([float(x[k]) for k in m_names], dtype=float),
    }
    if n_classes > 1:
        p_names = _place_names(n_tiers)
        config["place"] = np.array([int(x[k]) for k in p_names], dtype=int)
    return config


class AutoscalingProblem(ElementwiseProblem):
    """Black-box multi-objective autoscaling configuration problem."""

    def __init__(
        self,
        topology: Optional[Topology] = None,
        workload: Optional[Workload] = None,
        k_replications: int = 5,
        base_seed: int = 0,
        sla_ms: Optional[float] = None,
    ):
        self.topo = topology if topology is not None else default_topology()
        self.workload = workload if workload is not None else synthetic_diurnal()
        self.k_replications = int(k_replications)
        self.base_seed = int(base_seed)
        self.sla_ms = sla_ms
        self.n_eval_calls = 0

        n_names, c_names, m_names = _var_names(self.topo.n_tiers)
        variables: Dict[str, object] = {}
        for i, t in enumerate(self.topo.tiers):
            variables[n_names[i]] = Integer(bounds=(t.replica_min, t.replica_max))
            variables[c_names[i]] = Real(bounds=(t.cpu_min, t.cpu_max))
            variables[m_names[i]] = Real(bounds=(t.mem_min, t.mem_max))
        # Edge/cloud topology: a per-tier integer placement variable (which node
        # class the tier runs on). Absent for the single-location default.
        if self.topo.n_classes > 1:
            for i, p in enumerate(_place_names(self.topo.n_tiers)):
                variables[p] = Integer(bounds=(0, self.topo.n_classes - 1))

        super().__init__(vars=variables, n_obj=3, n_ieq_constr=0)

    def _evaluate(self, x, out, *args, **kwargs):
        config = x_to_config(x, self.topo.n_tiers, self.topo.n_classes)
        res = simulator.evaluate(
            self.topo,
            config,
            self.workload,
            k_replications=self.k_replications,
            base_seed=self.base_seed,
            sla_ms=self.sla_ms,
        )
        self.n_eval_calls += 1
        out["F"] = res["mean"]


class RealEncodedAutoscalingProblem(Problem):
    """Real-relaxed encoding: replicas as continuous, rounded at evaluation.

    Same oracle and objectives as `AutoscalingProblem`, but the decision vector
    is fully real (layout per tier: [n_i, c_i, m_i]) so decomposition-based
    algorithms that lack mixed-variable operators (e.g. pymoo's MOEA/D) can run.
    Using this single encoding for both NSGA-II and MOEA/D keeps their
    comparison on an identical search space.
    """

    def __init__(self, topology=None, workload=None, k_replications=5, base_seed=0,
                 sla_ms=None):
        self.topo = topology if topology is not None else default_topology()
        self.workload = workload if workload is not None else synthetic_diurnal()
        self.k_replications = int(k_replications)
        self.base_seed = int(base_seed)
        self.sla_ms = sla_ms
        self.n_eval_calls = 0
        self.n_classes = self.topo.n_classes
        # Per-tier layout is [n, c, m] (single-location) or [n, c, m, p] with a
        # real placement variable in [0, n_classes) floored to a class at eval.
        self._per_tier = 4 if self.n_classes > 1 else 3

        xl, xu = [], []
        for t in self.topo.tiers:
            xl += [t.replica_min, t.cpu_min, t.mem_min]
            xu += [t.replica_max, t.cpu_max, t.mem_max]
            if self.n_classes > 1:
                xl += [0.0]
                xu += [float(self.n_classes)]
        super().__init__(n_var=self._per_tier * self.topo.n_tiers, n_obj=3,
                         n_ieq_constr=0,
                         xl=np.array(xl, float), xu=np.array(xu, float))

    def _row_to_config(self, row: np.ndarray) -> Dict[str, np.ndarray]:
        r = row.reshape(self.topo.n_tiers, self._per_tier)
        config = {
            "replicas": np.rint(r[:, 0]).astype(int),
            "cpu": r[:, 1].astype(float),
            "mem": r[:, 2].astype(float),
        }
        if self.n_classes > 1:
            config["place"] = np.clip(np.floor(r[:, 3]), 0,
                                      self.n_classes - 1).astype(int)
        return config

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.atleast_2d(X)
        F = np.empty((X.shape[0], 3))
        for i, row in enumerate(X):
            res = simulator.evaluate(
                self.topo, self._row_to_config(row), self.workload,
                k_replications=self.k_replications, base_seed=self.base_seed,
                sla_ms=self.sla_ms,
            )
            F[i] = res["mean"]
        self.n_eval_calls += X.shape[0]
        out["F"] = F


class DynamicAutoscalingProblem(ElementwiseProblem):
    """Dynamic variant: per-minute replica policy problem.

    Decision vector: `[c_i, m_i]` per tier (same bounds as `AutoscalingProblem`)
    followed by the flattened policy weight vector for every tier
    (`policy.weight_vector_size() * n_tiers` reals, bounds `[-weight_bound,
    weight_bound]`). Replicas are no longer a decision variable at all -- the
    policy produces them per minute (`simulator.simulate_policy`) -- so, unlike
    the static problem, the whole vector is real-valued: this ONE class serves
    both NSGA-II and MOEA/D directly (no mixed-variable / real-encoded split
    needed, unlike `AutoscalingProblem`/`RealEncodedAutoscalingProblem`).

    Objectives: the same `[latency_ms, cost, energy_W]` triple, via
    `simulator.evaluate_policy` instead of `simulator.evaluate`. The paper uses
    it with the single-location topology.
    """

    def __init__(
        self,
        topology: Optional[Topology] = None,
        workload: Optional[Workload] = None,
        k_replications: int = 5,
        base_seed: int = 0,
        sla_ms: Optional[float] = None,
        weight_bound: float = 2.0,
    ):
        self.topo = topology if topology is not None else default_topology()
        self.workload = workload if workload is not None else synthetic_diurnal()
        self.k_replications = int(k_replications)
        self.base_seed = int(base_seed)
        self.sla_ms = sla_ms
        self.n_eval_calls = 0
        self.weight_bound = float(weight_bound)

        xl, xu = [], []
        for t in self.topo.tiers:
            xl += [t.cpu_min, t.mem_min]
            xu += [t.cpu_max, t.mem_max]
        n_weights = policy.weight_vector_size() * self.topo.n_tiers
        xl += [-self.weight_bound] * n_weights
        xu += [self.weight_bound] * n_weights

        super().__init__(n_var=2 * self.topo.n_tiers + n_weights, n_obj=3,
                         n_ieq_constr=0, xl=np.array(xl, float), xu=np.array(xu, float))

    def _split(self, x: np.ndarray):
        n_tiers = self.topo.n_tiers
        cm = x[:2 * n_tiers].reshape(n_tiers, 2)
        return cm[:, 0], cm[:, 1], x[2 * n_tiers:]

    def _evaluate(self, x, out, *args, **kwargs):
        cpu, mem, weights = self._split(np.asarray(x, dtype=float))
        policies = policy.decode(weights, self.topo.n_tiers)
        res = simulator.evaluate_policy(
            self.topo, policies, cpu, mem, self.workload,
            k_replications=self.k_replications, base_seed=self.base_seed,
            sla_ms=self.sla_ms,
        )
        self.n_eval_calls += 1
        out["F"] = res["mean"]
