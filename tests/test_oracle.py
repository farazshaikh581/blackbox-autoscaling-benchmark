"""Smoke tests for the black-box autoscaling oracle."""
import numpy as np
import pytest

from blackbox import AutoscalingProblem, default_topology, synthetic_diurnal, evaluate


def test_single_evaluation_shapes():
    topo = default_topology(n_tiers=3)
    wl = synthetic_diurnal()
    config = {
        "replicas": np.array([3, 3, 3]),
        "cpu": np.array([1.0, 1.0, 1.0]),
        "mem": np.array([1.0, 1.0, 1.0]),
    }
    res = evaluate(topo, config, wl, k_replications=5, base_seed=0)
    assert res["mean"].shape == (3,)
    assert res["cv"].shape == (3,)
    assert np.all(res["mean"] > 0)


def test_oracle_is_deterministic_given_seed():
    topo = default_topology(2)
    wl = synthetic_diurnal()
    config = {"replicas": np.array([2, 2]), "cpu": np.array([0.8, 0.8]),
              "mem": np.array([0.9, 0.9])}
    a = evaluate(topo, config, wl, k_replications=4, base_seed=7)["mean"]
    b = evaluate(topo, config, wl, k_replications=4, base_seed=7)["mean"]
    np.testing.assert_allclose(a, b)


def test_more_replicas_reduce_latency_under_load():
    # Under a heavy static load, adding replicas must not increase latency.
    topo = default_topology(1)
    wl = synthetic_diurnal(peak_rps=300.0)
    lo = evaluate(topo, {"replicas": np.array([1]), "cpu": np.array([0.5]),
                         "mem": np.array([2.0])}, wl, 3)["mean"][0]
    hi = evaluate(topo, {"replicas": np.array([10]), "cpu": np.array([0.5]),
                         "mem": np.array([2.0])}, wl, 3)["mean"][0]
    assert hi <= lo


def test_calibrated_single_tier_base_latency():
    # At light load the calibrated single tier (0.5 core, ~4 ms service time)
    # sits near its service floor, in the same regime as the measured ~4 ms.
    topo = default_topology(1)
    wl = synthetic_diurnal(peak_rps=5.0, base_rps=1.0)
    lat = evaluate(topo, {"replicas": np.array([2]), "cpu": np.array([0.5]),
                          "mem": np.array([0.25])}, wl, 3)["mean"][0]
    assert 2.0 < lat < 12.0


def test_more_cpu_lowers_latency_under_load():
    # More CPU per replica shortens the service time, so latency drops.
    topo = default_topology(1)
    wl = synthetic_diurnal(peak_rps=200.0)
    slow = evaluate(topo, {"replicas": np.array([2]), "cpu": np.array([0.3]),
                           "mem": np.array([1.0])}, wl, 3)["mean"][0]
    fast = evaluate(topo, {"replicas": np.array([2]), "cpu": np.array([1.5]),
                           "mem": np.array([1.0])}, wl, 3)["mean"][0]
    assert fast <= slow


def test_pymoo_problem_evaluates():
    prob = AutoscalingProblem(k_replications=3)
    assert prob.n_obj == 3
    assert prob.n_var == 3 * prob.topo.n_tiers


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
