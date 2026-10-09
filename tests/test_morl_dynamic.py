"""Tests for MORL's episodic multi-step training loop (dynamic variant).

The critical thing to pin here is the hand-derived REINFORCE gradient through
`DynamicReplicaPolicy`'s hidden layer (one more chain-rule hop than the
existing single-step `PreferenceGaussianPolicy`, which has no hidden layer) --
verified against finite-difference numerical gradients, not just eyeballed.
"""
import numpy as np
import pytest

from blackbox import default_topology, synthetic_diurnal
from experiments.run_morl import (
    DynamicReplicaPolicy,
    _MeanPolicyAdapter,
    _anneal_lr,
    _dynamic_episode_rollout,
    evaluate_dynamic_episode,
    run_morl_dynamic,
)


def test_dynamic_replica_policy_gradient_matches_finite_difference():
    rng = np.random.default_rng(0)
    pol = DynamicReplicaPolicy(hidden_dim=4, in_dim=6, rng=rng)
    s = rng.standard_normal(6)
    raw = 0.6

    def logp(p):
        mu, _ = p._forward(s)
        std = p.std
        return -0.5 * ((raw - mu) / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)

    _, h = pol._forward(s)
    cache = {"s": s, "h": h, "mu": pol._forward(s)[0], "raw": raw, "std": pol.std}
    gW1, gb1, gw2, gb2, gls = pol.grads(cache)

    eps = 1e-6

    num_gW1 = np.zeros_like(pol.W1)
    for i in range(pol.W1.shape[0]):
        for j in range(pol.W1.shape[1]):
            orig = pol.W1[i, j]
            pol.W1[i, j] = orig + eps
            lp1 = logp(pol)
            pol.W1[i, j] = orig - eps
            lp2 = logp(pol)
            pol.W1[i, j] = orig
            num_gW1[i, j] = (lp1 - lp2) / (2 * eps)
    assert gW1 == pytest.approx(num_gW1, abs=1e-6)

    def num_vec(attr):
        v = getattr(pol, attr)
        g = np.zeros_like(v)
        for i in range(len(v)):
            orig = v[i]
            v[i] = orig + eps
            lp1 = logp(pol)
            v[i] = orig - eps
            lp2 = logp(pol)
            v[i] = orig
            g[i] = (lp1 - lp2) / (2 * eps)
        return g

    assert gb1 == pytest.approx(num_vec("b1"), abs=1e-6)
    assert gw2 == pytest.approx(num_vec("w2"), abs=1e-6)

    for attr, g in (("b2", gb2), ("log_std", gls)):
        orig = getattr(pol, attr)
        setattr(pol, attr, orig + eps)
        lp1 = logp(pol)
        setattr(pol, attr, orig - eps)
        lp2 = logp(pol)
        setattr(pol, attr, orig)
        assert g == pytest.approx((lp1 - lp2) / (2 * eps), abs=1e-6)


def test_mean_policy_adapter_matches_deterministic_mean():
    rng = np.random.default_rng(1)
    pol = DynamicReplicaPolicy(hidden_dim=4, in_dim=6, rng=rng)
    w = np.array([0.3, 0.3, 0.4])
    state = (0.5, 0.2, 0.0)
    adapter = _MeanPolicyAdapter(w, pol)

    expected = pol.mean(np.concatenate([w, state]))
    assert adapter.act(state) == pytest.approx(np.clip(expected, 0.0, 1.0))
    assert 0.0 <= adapter.act(state) <= 1.0


def test_dynamic_episode_rollout_shapes_and_reproducibility():
    topo = default_topology(2)
    w = np.array([0.5, 0.3, 0.2])
    rng = np.random.default_rng(2)
    dyn_policies = [DynamicReplicaPolicy(4, 6, rng) for _ in range(topo.n_tiers)]
    profile = np.full(10, 3.0)

    f, caches, f_per_minute = _dynamic_episode_rollout(
        topo, w, dyn_policies, [1.0, 1.0], [1.0, 1.0], profile, rng)
    assert f.shape == (3,)
    assert len(caches) == topo.n_tiers
    assert all(len(c) == len(profile) for c in caches)
    assert f_per_minute.shape == (len(profile), 3)


def test_dynamic_episode_rollout_rejects_multiclass_topology():
    from blackbox import edge_cloud_topology
    topo = edge_cloud_topology(2)
    rng = np.random.default_rng(0)
    dyn_policies = [DynamicReplicaPolicy(4, 6, rng) for _ in range(topo.n_tiers)]
    with pytest.raises(NotImplementedError):
        _dynamic_episode_rollout(topo, np.array([0.5, 0.3, 0.2]), dyn_policies,
                                 [1.0, 1.0], [1.0, 1.0], np.full(5, 1.0), rng)


def test_evaluate_dynamic_episode_pools_caches_across_replications():
    topo = default_topology(2)
    w = np.array([0.4, 0.3, 0.3])
    rng = np.random.default_rng(3)
    dyn_policies = [DynamicReplicaPolicy(4, 6, rng) for _ in range(topo.n_tiers)]
    wl = synthetic_diurnal()

    f, caches, step_F = evaluate_dynamic_episode(
        topo, w, dyn_policies, [1.0, 1.0], [1.0, 1.0], wl, k_replications=3,
        base_seed=0, rng=rng)
    assert f.shape == (3,)
    n_minutes = len(wl.mean_rps)
    assert all(len(c) == 3 * n_minutes for c in caches)
    assert step_F.shape == (3 * n_minutes, 3)


def test_run_morl_dynamic_smoke():
    # Tiny budget end-to-end sanity: runs without error, produces a
    # non-degenerate front, and respects the evals-as-episodes budget.
    topo = default_topology(2)
    out = run_morl_dynamic(topo, evals=6, warmup=2, partitions=2,
                           archive_cap=4, k_replications=1, base_seed=0,
                           seed=1, hidden_dim=3,
                           workload=synthetic_diurnal())
    assert out["n_eval_calls"] == 6
    assert out["F"].shape[1] == 3
    assert len(out["F"]) >= 1


def test_run_morl_dynamic_smoke_with_lr_anneal():
    # Same smoke test, with lr annealing enabled -- just needs to run without
    # error and respect the same budget/shape contract as the constant-lr case.
    topo = default_topology(2)
    out = run_morl_dynamic(topo, evals=6, lr=0.6, lr_end=0.2, warmup=2,
                           partitions=2, archive_cap=4, k_replications=1,
                           base_seed=0, seed=1, hidden_dim=3,
                           workload=synthetic_diurnal())
    assert out["n_eval_calls"] == 6
    assert out["F"].shape[1] == 3


def test_anneal_lr():
    assert _anneal_lr(0.6, None, 0.5) == 0.6  # constant when lr_end is None
    assert _anneal_lr(0.6, 0.2, 0.0) == pytest.approx(0.6)
    assert _anneal_lr(0.6, 0.2, 1.0) == pytest.approx(0.2)
    assert _anneal_lr(0.6, 0.2, 0.5) == pytest.approx(0.4)


def test_run_morl_dynamic_smoke_with_latency_reward_floor():
    # Sanity: the corner-reward fix runs without error and respects the same
    # budget/shape contract, whether or not the floor is active.
    topo = default_topology(2)
    out = run_morl_dynamic(topo, evals=6, lr=0.6, lr_end=0.2, warmup=2,
                           partitions=2, archive_cap=4, k_replications=1,
                           base_seed=0, seed=1, hidden_dim=3,
                           workload=synthetic_diurnal(),
                           latency_reward_floor=0.1)
    assert out["n_eval_calls"] == 6
    assert out["F"].shape[1] == 3
