import numpy as np

from blackbox import default_topology
from blackbox.hpa import HPAPolicy, HPAPolicyProblem
from blackbox.workload import synthetic_diurnal


def _replicas(pol, frac):
    return round(pol.rmin + frac * (pol.rmax - pol.rmin))


def _state(pol, busy, current):
    return (busy / pol.rmax, current / pol.rmax, 0.0)


def test_hpa_scales_up_at_once_and_holds_scale_down():
    tier = default_topology().tiers[0]
    pol = HPAPolicy(tier, target=0.5, window_min=5)
    assert _replicas(pol, pol.act(_state(pol, busy=4.0, current=1))) == 8
    for _ in range(4):
        assert _replicas(pol, pol.act(_state(pol, busy=1.0, current=8))) == 8
    assert _replicas(pol, pol.act(_state(pol, busy=1.0, current=8))) == 2


def test_hpa_tolerance_band_keeps_current():
    tier = default_topology().tiers[0]
    pol = HPAPolicy(tier, target=0.5, window_min=1)
    assert _replicas(pol, pol.act(_state(pol, busy=5.2, current=10))) == 10
    assert _replicas(pol, pol.act(_state(pol, busy=6.0, current=10))) == 12


def test_hpa_problem_is_deterministic_for_fixed_seed():
    topo = default_topology()
    prob = HPAPolicyProblem(topology=topo, workload=synthetic_diurnal(),
                            k_replications=2, base_seed=3)
    x = (prob.xl + prob.xu) / 2
    a = prob.evaluate(np.atleast_2d(x), return_values_of=["F"])
    b = prob.evaluate(np.atleast_2d(x), return_values_of=["F"])
    assert np.allclose(a, b) and np.all(np.isfinite(a))
