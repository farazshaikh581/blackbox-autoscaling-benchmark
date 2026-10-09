"""Tests for the held-out-day fairness helpers: cardinality cap and re-scoring."""
import numpy as np

from blackbox import AutoscalingProblem, default_topology, synthetic_diurnal
from experiments._common import bounded_front, holdout_front


def test_bounded_front_keeps_nondominated_and_caps():
    # A clean 3-objective Pareto set of 10 mutually non-dominated points plus one
    # dominated point that must be dropped regardless of the cap.
    pf = np.array([[i, 10 - i, 5] for i in range(10)], dtype=float)
    dominated = np.array([[5.0, 6.0, 6.0]])           # dominated by [5,5,5]-ish rows
    F = np.vstack([pf, dominated])
    out = bounded_front(F, cap=4)
    assert len(out) == 4                               # capped
    # the dominated point must not survive, and every kept row is one of the pf rows
    assert not any(np.array_equal(dominated[0], r) for r in out)
    assert all(any(np.array_equal(r, q) for q in pf) for r in out)


def test_bounded_front_below_cap_is_identity_set():
    pf = np.array([[i, 5 - i, 2] for i in range(5)], dtype=float)
    out = bounded_front(pf, cap=20)
    assert len(out) == 5


def test_holdout_front_rescores_on_test_problem():
    from pymoo.core.mixed import MixedVariableSampling

    topo = default_topology(2)
    train = AutoscalingProblem(topology=topo, workload=synthetic_diurnal(),
                               k_replications=2, base_seed=42)
    # A few real decision dicts sampled from the problem's own variables, as a
    # stand-in for a run's final non-dominated set.
    X = MixedVariableSampling()(train, 6).get("X")

    # Re-score those exact configs on a *different* workload realization (a proxy
    # for the held-out day) via a test problem, and cap the resulting front.
    test = AutoscalingProblem(topology=topo, workload=synthetic_diurnal(),
                              k_replications=2, base_seed=99)
    F_test = holdout_front(test, X, cap=4)
    assert F_test.ndim == 2 and F_test.shape[1] == 3
    assert len(F_test) <= 4
    assert np.all(np.isfinite(F_test))
