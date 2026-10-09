"""Tests for the dynamic (per-minute policy) simulator.

Pins three things: (1) a policy that always targets max replicas reproduces
the static `simulate_config` path exactly; (2) the same holds on the
edge-cloud topology for a given placement; (3) the policy state has three
inputs and the third, reserved one is always zero.
"""
import numpy as np
import pytest

from blackbox import default_topology, edge_cloud_topology, synthetic_diurnal
from blackbox.policy import decode, weight_vector_size
from blackbox.simulator import simulate_config, simulate_policy


class _ScriptedPolicy:
    """Test-only stand-in for PolicyNet: a fixed, deterministic output sequence."""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self._i = 0
        self.states = []

    def act(self, state):
        self.states.append(tuple(state))
        v = self._outputs[min(self._i, len(self._outputs) - 1)]
        self._i += 1
        return v


def test_matches_static_config_when_always_max():
    topo = default_topology(3)
    profile = synthetic_diurnal().realize(seed=0)
    cpu = [1.0, 1.0, 1.0]
    mem = [1.0, 1.0, 1.0]
    policies = [_ScriptedPolicy([1.0]) for _ in range(topo.n_tiers)]

    dyn = simulate_policy(topo, policies, cpu, mem, profile)

    max_cfg = {"replicas": np.array([t.replica_max for t in topo.tiers]),
               "cpu": np.array(cpu), "mem": np.array(mem)}
    static = simulate_config(topo, max_cfg, profile)

    assert dyn.latency_ms == pytest.approx(static.latency_ms, rel=1e-9)
    assert dyn.cost == pytest.approx(static.cost, rel=1e-9)
    assert dyn.energy_w == pytest.approx(static.energy_w, rel=1e-9)


def test_multiclass_matches_static_config_when_always_max():
    topo = edge_cloud_topology(3)
    profile = synthetic_diurnal().realize(seed=0)
    cpu = [1.0, 1.0, 1.0]
    mem = [1.0, 1.0, 1.0]
    place = np.array([0, 1, 1])
    policies = [_ScriptedPolicy([1.0]) for _ in range(topo.n_tiers)]

    dyn = simulate_policy(topo, policies, cpu, mem, profile, place=place)

    max_cfg = {"replicas": np.array([t.replica_max for t in topo.tiers]),
               "cpu": np.array(cpu), "mem": np.array(mem), "place": place}
    static = simulate_config(topo, max_cfg, profile)

    assert dyn.latency_ms == pytest.approx(static.latency_ms, rel=1e-9)
    assert dyn.cost == pytest.approx(static.cost, rel=1e-9)
    assert dyn.energy_w == pytest.approx(static.energy_w, rel=1e-9)


def test_reserved_state_input_is_zero():
    topo = default_topology(1)
    policy = _ScriptedPolicy([0.0, 1.0, 0.5])
    simulate_policy(topo, [policy], cpu=[1.0], mem=[1.0],
                    rps_profile=np.array([10.0, 20.0, 5.0]))
    assert all(len(s) == 3 and s[2] == 0.0 for s in policy.states)


def test_decode_round_trip_shapes():
    n_tiers = 3
    size = weight_vector_size()
    weights = np.random.RandomState(0).randn(size * n_tiers)
    nets = decode(weights, n_tiers)
    assert len(nets) == n_tiers
    state = np.array([0.5, 0.5, 0.0])
    for net in nets:
        out = net.act(state)
        assert 0.0 <= out <= 1.0


def test_decode_rejects_wrong_length():
    with pytest.raises(ValueError):
        decode(np.zeros(5), n_tiers=3)
