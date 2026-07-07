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

from . import simulator
from .topology import Topology, default_topology
from .workload import Workload, synthetic_diurnal


def _var_names(n_tiers: int):
    return (
        [f"n_{i}" for i in range(n_tiers)],
        [f"c_{i}" for i in range(n_tiers)],
        [f"m_{i}" for i in range(n_tiers)],
    )


def x_to_config(x: Dict[str, float], n_tiers: int) -> Dict[str, np.ndarray]:
    """Convert a pymoo mixed-variable sample dict to a simulator config dict."""
    n_names, c_names, m_names = _var_names(n_tiers)
    return {
        "replicas": np.array([int(x[k]) for k in n_names], dtype=int),
        "cpu": np.array([float(x[k]) for k in c_names], dtype=float),
        "mem": np.array([float(x[k]) for k in m_names], dtype=float),
    }


class AutoscalingProblem(ElementwiseProblem):
    """Black-box multi-objective autoscaling configuration problem."""

    def __init__(
        self,
        topology: Optional[Topology] = None,
        workload: Optional[Workload] = None,
        k_replications: int = 5,
        base_seed: int = 0,
    ):
        self.topo = topology if topology is not None else default_topology()
        self.workload = workload if workload is not None else synthetic_diurnal()
        self.k_replications = int(k_replications)
        self.base_seed = int(base_seed)
        self.n_eval_calls = 0

        n_names, c_names, m_names = _var_names(self.topo.n_tiers)
        variables: Dict[str, object] = {}
        for i, t in enumerate(self.topo.tiers):
            variables[n_names[i]] = Integer(bounds=(t.replica_min, t.replica_max))
            variables[c_names[i]] = Real(bounds=(t.cpu_min, t.cpu_max))
            variables[m_names[i]] = Real(bounds=(t.mem_min, t.mem_max))

        super().__init__(vars=variables, n_obj=3, n_ieq_constr=0)

    def _evaluate(self, x, out, *args, **kwargs):
        config = x_to_config(x, self.topo.n_tiers)
        res = simulator.evaluate(
            self.topo,
            config,
            self.workload,
            k_replications=self.k_replications,
            base_seed=self.base_seed,
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

    def __init__(self, topology=None, workload=None, k_replications=5, base_seed=0):
        self.topo = topology if topology is not None else default_topology()
        self.workload = workload if workload is not None else synthetic_diurnal()
        self.k_replications = int(k_replications)
        self.base_seed = int(base_seed)
        self.n_eval_calls = 0

        xl, xu = [], []
        for t in self.topo.tiers:
            xl += [t.replica_min, t.cpu_min, t.mem_min]
            xu += [t.replica_max, t.cpu_max, t.mem_max]
        super().__init__(n_var=3 * self.topo.n_tiers, n_obj=3, n_ieq_constr=0,
                         xl=np.array(xl, float), xu=np.array(xu, float))

    def _row_to_config(self, row: np.ndarray) -> Dict[str, np.ndarray]:
        r = row.reshape(self.topo.n_tiers, 3)
        return {
            "replicas": np.rint(r[:, 0]).astype(int),
            "cpu": r[:, 1].astype(float),
            "mem": r[:, 2].astype(float),
        }

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.atleast_2d(X)
        F = np.empty((X.shape[0], 3))
        for i, row in enumerate(X):
            res = simulator.evaluate(
                self.topo, self._row_to_config(row), self.workload,
                k_replications=self.k_replications, base_seed=self.base_seed,
            )
            F[i] = res["mean"]
        self.n_eval_calls += X.shape[0]
        out["F"] = F
