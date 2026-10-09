"""Tests for the optional SLA-hinge latency framing (--sla-ms)."""
import numpy as np

from blackbox import AutoscalingProblem, default_topology, synthetic_diurnal
from blackbox import simulator
from blackbox.oracle import RealEncodedAutoscalingProblem
from experiments.aggregate import compliant_mask
from experiments.run_morl import run_morl


def _one_config(topo):
    return {
        "replicas": np.array([t.replica_min for t in topo.tiers], dtype=int),
        "cpu": np.array([t.cpu_min for t in topo.tiers], dtype=float),
        "mem": np.array([t.mem_max for t in topo.tiers], dtype=float),
    }


def test_hinge_transforms_latency_objective_only():
    topo = default_topology(2)
    wl = synthetic_diurnal()
    cfg = _one_config(topo)
    raw = simulator.evaluate(topo, cfg, wl, k_replications=3, base_seed=0)["mean"]
    sla = float(raw[0]) * 0.5  # an SLA the config comfortably violates
    hinged = simulator.evaluate(topo, cfg, wl, k_replications=3, base_seed=0,
                                sla_ms=sla)["mean"]
    # latency objective is shifted down by the SLA (per-replication hinge, so the
    # equality is up to float rounding); cost and energy are untouched.
    assert np.isclose(hinged[0], raw[0] - sla)
    assert hinged[1] == raw[1] and hinged[2] == raw[2]


def test_hinge_clamps_compliant_latency_to_zero():
    topo = default_topology(2)
    wl = synthetic_diurnal()
    cfg = _one_config(topo)
    raw = simulator.evaluate(topo, cfg, wl, k_replications=1, base_seed=0)["mean"]
    # An SLA well above the achieved latency: the hinge must read exactly 0.
    hinged = simulator.evaluate(topo, cfg, wl, k_replications=1, base_seed=0,
                                sla_ms=float(raw[0]) + 1000.0)["mean"]
    assert hinged[0] == 0.0


def test_none_sla_is_backward_compatible():
    topo = default_topology(2)
    wl = synthetic_diurnal()
    cfg = _one_config(topo)
    a = simulator.evaluate(topo, cfg, wl, k_replications=2, base_seed=1)["mean"]
    b = simulator.evaluate(topo, cfg, wl, k_replications=2, base_seed=1,
                           sla_ms=None)["mean"]
    assert np.array_equal(a, b)


def test_problems_expose_finite_hinged_fronts():
    topo = default_topology(2)
    wl = synthetic_diurnal()
    for problem in (
        AutoscalingProblem(topology=topo, workload=wl, k_replications=1,
                           base_seed=0, sla_ms=20.0),
        RealEncodedAutoscalingProblem(topology=topo, workload=wl, k_replications=1,
                                      base_seed=0, sla_ms=20.0),
    ):
        # A couple of random decision vectors evaluated through each encoding.
        if isinstance(problem, RealEncodedAutoscalingProblem):
            X = problem.xl + (problem.xu - problem.xl) * np.random.default_rng(0).random(
                (4, len(problem.xl)))
            F = problem.evaluate(X, return_values_of=["F"])
        else:
            from pymoo.core.mixed import MixedVariableSampling
            X = MixedVariableSampling()(problem, 4).get("X")
            F = problem.evaluate(X, return_values_of=["F"])
        F = np.atleast_2d(F)
        assert np.all(np.isfinite(F))
        assert np.all(F[:, 0] >= 0.0)  # hinged latency is never negative


def test_morl_runs_with_sla():
    topo = default_topology(2)
    out = run_morl(topo, evals=40, batch=10, warmup=10, k_replications=1,
                   base_seed=0, seed=1, workload=synthetic_diurnal(), sla_ms=20.0)
    F = out["F"]
    assert F.ndim == 2 and F.shape[1] == 3
    assert np.all(F[:, 0] >= 0.0)


def test_compliant_mask_hinged_vs_raw():
    # latency (col 0), cost, energy. Two compliant, one violator.
    F = np.array([[0.0, 5.0, 5.0],    # hinged-compliant / raw-compliant
                  [3.0, 4.0, 4.0],    # hinged-violator(3) but raw-compliant(<=20)
                  [50.0, 1.0, 1.0]])  # violator in both
    # Hinged fronts store max(0, lat-sla): only exact-zero latency is compliant.
    assert list(compliant_mask(F, 20.0, hinged=True)) == [True, False, False]
    # Raw-latency fronts: everything at or below the SLA counts.
    assert list(compliant_mask(F, 20.0, hinged=False)) == [True, True, False]


def test_aggregate_reports_compliant_column(tmp_path):
    import csv
    from experiments import aggregate

    # Two tiny hinged fronts per algo: some compliant (lat 0), some violators.
    rng = np.random.default_rng(0)
    for algo in ("nsga2", "moead"):
        for seed in (1, 2, 3):
            lat = np.array([0.0, 0.0, 4.0, 9.0])
            ce = 1.0 + rng.random((4, 2))
            F = np.column_stack([lat, ce])
            np.savez(tmp_path / f"{algo}_seed{seed}.npz",
                     F=F, algo=algo, seed=seed)

    import sys
    argv = ["aggregate", "--results", str(tmp_path), "--sla-ms", "20", "--hinged"]
    old = sys.argv
    try:
        sys.argv = argv
        aggregate.main()
    finally:
        sys.argv = old

    with open(tmp_path / "comparison.csv") as f:
        rows = list(csv.DictReader(f))
    assert rows and "comp_hv_mean" in rows[0]
    # Compliant-region HV is a non-negative 2-D hypervolume.
    assert all(float(r["comp_hv_mean"]) >= 0.0 for r in rows)
