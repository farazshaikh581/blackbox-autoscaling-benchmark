"""Run the multi-objective RL (MORL) baseline on the black-box oracle.

The autoscaling instance is a *static* configuration search, so an episode is a
single step: the agent reads a scalarization preference `w` (a weight on the
3-simplex over latency/cost/energy) and emits one full deployment configuration;
the oracle returns the objective vector; the reward is the negative Tchebycheff
scalarization of the normalized objectives under `w`. This is the single-step
(contextual-bandit) reduction of the paper's preference-conditioned MORL policy.

The policy is a preference-conditioned diagonal Gaussian over the normalized
decision vector, trained by REINFORCE. Different preferences steer the mean into
different regions of configuration space, so sweeping `w` over a reference set of
weights (the same das-dennis directions MOEA/D decomposes with -- perf / cost /
energy corners and a balanced centre) traces out the Pareto front. The decision
vector uses the real-relaxed encoding (replicas rounded at evaluation), so the
search space is identical to MOEA/D's, and the oracle budget is the same fixed
number of calls, keeping the RL-vs-EA comparison fair.

Everything is deterministic given `--seed` (policy sampling) and `--base-seed`
(oracle workload), so the anytime hypervolume / GD+ / IGD+ / Wilcoxon protocol is
reproducible. No deep-learning dependency: the linear policy and its gradients are
plain numpy.

Usage:
    python -m experiments.run_morl --evals 500 --batch 20 --seed 1 --tiers 3 \
        --k 5 --out results/morl_seed1.npz
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

from blackbox import default_topology, energy_model
from blackbox import simulator
from blackbox.simulator import _tier_latency_ms


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _softplus(z):
    # numerically stable softplus
    return np.logaddexp(0.0, z)


def nondominated(F: np.ndarray) -> np.ndarray:
    F = np.atleast_2d(F)
    idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
    return F[idx]


class BoundedArchive:
    """Non-dominated archive capped at `cap` points, pruned by crowding distance.

    This is the direct counterpart of an EA's bounded elite population: the RL
    agent evaluates far more configurations than it reports, so without a cap its
    reported front would carry many more points than NSGA-II's / MOEA/D's ~20 and
    inflate hypervolume by cardinality alone. Capping at the EA population size
    and pruning the most crowded points (NSGA-II crowding distance) keeps the
    reported front cardinality-comparable and the comparison fair.

    Optionally retains the decision object (config) behind each objective row, in
    lockstep with `F`. That is what lets MORL re-score its *best-found* configs on
    the held-out day, exactly as the EAs re-score `res.X` -- the fair, apples-to-
    apples counterpart to the greedy policy sweep. `add(f)` without a config still
    works (configs stay `None`); pruning drops the config and its row together.
    """

    def __init__(self, cap: int):
        self.cap = int(cap)
        self.F = np.empty((0, 3))
        self.X: list = []  # decision configs, aligned row-for-row with self.F

    def add(self, f: np.ndarray, x=None) -> None:
        F = np.vstack([self.F, np.asarray(f, float)[None, :]])
        X = self.X + [x]
        keep = NonDominatedSorting().do(F, only_non_dominated_front=True)
        F = F[keep]
        X = [X[i] for i in keep]
        while len(F) > self.cap:
            j = int(np.argmin(self._crowding(F)))
            F = np.delete(F, j, axis=0)
            X.pop(j)
        self.F = F
        self.X = X

    @staticmethod
    def _crowding(F: np.ndarray) -> np.ndarray:
        n, m = F.shape
        cd = np.zeros(n)
        for k in range(m):
            order = np.argsort(F[:, k])
            cd[order[0]] = cd[order[-1]] = np.inf
            span = F[order[-1], k] - F[order[0], k]
            if span == 0:
                continue
            cd[order[1:-1]] += (F[order[2:], k] - F[order[:-2], k]) / span
        return cd

    def front(self) -> np.ndarray:
        return self.F.copy()

    def front_configs(self) -> list:
        """The stored config behind each non-dominated row (best-found set)."""
        return list(self.X)


class PreferenceGaussianPolicy:
    """Diagonal Gaussian over the normalized decision vector, conditioned on w.

    mean(w) = sigmoid(A w + b) in (0, 1)^d ; std = softplus(log_std), shared
    across preferences. Trained by REINFORCE. `d = 3 * n_tiers` (replicas, cpu,
    mem per tier, each scaled to [0, 1]).
    """

    def __init__(self, d: int, n_obj: int, rng: np.random.Generator):
        self.d = d
        self.rng = rng
        # Small init: mean starts near the centre of the box for every preference.
        self.A = 0.01 * rng.standard_normal((d, n_obj))
        self.b = np.zeros(d)
        self.log_std = np.full(d, -1.1)  # softplus(-1.1) ~ 0.29 initial exploration

    @property
    def std(self) -> np.ndarray:
        return _softplus(self.log_std) + 1e-3

    def mean(self, w: np.ndarray) -> np.ndarray:
        """Greedy (noise-free) normalized action for preference `w`."""
        return _sigmoid(self.A @ w + self.b)

    def sample(self, w: np.ndarray):
        """Return (action in [0,1]^d, cached forward terms for the gradient)."""
        z = self.A @ w + self.b
        mu = _sigmoid(z)
        std = self.std
        raw = mu + std * self.rng.standard_normal(self.d)
        action = np.clip(raw, 0.0, 1.0)
        return action, {"w": w, "mu": mu, "raw": raw, "std": std}

    def grads(self, cache: dict, advantage: float):
        """REINFORCE gradient contributions (ascending expected reward).

        d logpi/d mu     = (raw - mu) / std^2
        d mu/d z         = mu (1 - mu)          (sigmoid)
        d logpi/d log_std= ((raw-mu)^2/std^2 - 1) * softplus'(log_std)
        """
        w, mu, raw, std = cache["w"], cache["mu"], cache["raw"], cache["std"]
        dlogp_dmu = (raw - mu) / (std ** 2)
        dlogp_dz = dlogp_dmu * mu * (1.0 - mu)
        gA = advantage * np.outer(dlogp_dz, w)
        gb = advantage * dlogp_dz
        dlogp_dstd = ((raw - mu) ** 2 / (std ** 2) - 1.0) / std
        gls = advantage * dlogp_dstd * _sigmoid(self.log_std)  # softplus' = sigmoid
        return gA, gb, gls

    def apply(self, gA, gb, gls, lr: float):
        self.A += lr * gA
        self.b += lr * gb
        self.log_std += lr * gls
        # Keep exploration in a sane band (~0.02 .. ~0.7 std).
        self.log_std = np.clip(self.log_std, -4.0, 0.0)


class DynamicReplicaPolicy:
    """One tier's per-minute replica policy, preference-conditioned (dynamic variant).

    Same small-MLP architecture as `blackbox.policy.PolicyNet` (tanh hidden,
    sigmoid output) -- deliberately, so the RL-vs-EA comparison
    is about search METHOD, not confounded by giving MORL a smaller-capacity
    policy than NSGA-II/MOEA-D search. Unlike PolicyNet, this one is (a)
    preference-conditioned (state includes `w`, so one policy covers the
    whole front) and (b) stochastic (Gaussian around the sigmoid mean, for
    REINFORCE exploration -- the same "linear mean + Gaussian, trained by
    REINFORCE" family as `PreferenceGaussianPolicy`, with one extra
    chain-rule hop for the hidden layer since this has one).

    Input `s` (in_dim = 3 + 3 = 6): concat(preference w, the same causal
    per-minute state `blackbox.simulator.simulate_policy` uses --
    [lam_prev_norm, replicas_prev_norm, reserved 0]). Output: target replica
    fraction in [0, 1] (mapped to [replica_min, replica_max] by the caller).

    Gradient math verified against finite-difference numerical gradients
    (max abs diff ~1e-10) before use here -- hand-derived backprop through a
    hidden layer is easy to get subtly wrong, so this was checked, not
    assumed correct by inspection.
    """

    def __init__(self, hidden_dim: int, in_dim: int, rng: np.random.Generator):
        self.W1 = 0.1 * rng.standard_normal((hidden_dim, in_dim))
        self.b1 = np.zeros(hidden_dim)
        self.w2 = 0.1 * rng.standard_normal(hidden_dim)
        self.b2 = 0.0
        self.log_std = -1.1
        self.rng = rng

    @property
    def std(self) -> float:
        return float(_softplus(self.log_std)) + 1e-3

    def _forward(self, s: np.ndarray):
        h = np.tanh(self.W1 @ s + self.b1)
        z = float(self.w2 @ h + self.b2)
        return _sigmoid(z), h

    def mean(self, s: np.ndarray) -> float:
        """Greedy (noise-free) target replica fraction for state `s`."""
        mu, _ = self._forward(s)
        return float(mu)

    def sample(self, s: np.ndarray):
        """Return (raw target fraction, cached forward terms for the gradient).

        Unlike `PreferenceGaussianPolicy.sample`, this does NOT clip to
        [0, 1] here -- the caller (the rollout) needs the unclipped `raw` for
        the replica-fraction-to-count mapping's own clipping, and the cached
        `raw` used by `grads` must match whatever was actually used
        downstream for the log-prob to be consistent.
        """
        mu, h = self._forward(s)
        std = self.std
        raw = mu + std * self.rng.standard_normal()
        return float(raw), {"s": s, "h": h, "mu": mu, "raw": raw, "std": std}

    def grads(self, cache: dict):
        """d logpi(raw|s) / d params -- NOT yet scaled by the advantage."""
        s, h, mu, raw, std = cache["s"], cache["h"], cache["mu"], cache["raw"], cache["std"]
        dlogp_dmu = (raw - mu) / (std ** 2)
        dlogp_dz = dlogp_dmu * mu * (1.0 - mu)
        dh = dlogp_dz * self.w2 * (1.0 - h ** 2)
        gW1 = np.outer(dh, s)
        gb1 = dh
        gw2 = dlogp_dz * h
        gb2 = dlogp_dz
        dlogp_dstd = ((raw - mu) ** 2 / (std ** 2) - 1.0) / std
        gls = dlogp_dstd * _sigmoid(self.log_std)
        return gW1, gb1, gw2, gb2, gls

    def apply(self, gW1, gb1, gw2, gb2, gls, lr: float):
        self.W1 += lr * gW1
        self.b1 += lr * gb1
        self.w2 += lr * gw2
        self.b2 += lr * gb2
        self.log_std += lr * gls
        self.log_std = np.clip(self.log_std, -4.0, 0.0)


class _MeanPolicyAdapter:
    """Wraps a trained (preference, DynamicReplicaPolicy) pair to look like a
    deterministic `blackbox.policy.PolicyNet` (`.act(state) -> float`), so the
    trained policy can be scored via the existing
    `simulator.simulate_policy`/`evaluate_policy` machinery -- the same
    deterministic path NSGA-II/MOEA-D's decoded policies use.
    """

    def __init__(self, w: np.ndarray, dyn_policy: DynamicReplicaPolicy):
        self.w = w
        self.dyn_policy = dyn_policy

    def act(self, state) -> float:
        s = np.concatenate([self.w, np.asarray(state, dtype=float)])
        return _sigmoid_clip01(self.dyn_policy.mean(s))


def _sigmoid_clip01(x: float) -> float:
    # mean() already returns a sigmoid output in (0, 1); this just guards
    # against float edge cases so callers can trust [0, 1] exactly.
    return float(np.clip(x, 0.0, 1.0))


def _anneal_lr(lr: float, lr_end, progress: float) -> float:
    """Linearly interpolate lr -> lr_end as progress goes 0 -> 1.

    `lr_end=None` keeps a constant lr (returns `lr` unchanged).
    """
    if lr_end is None:
        return lr
    return lr + (lr_end - lr) * progress


def _dynamic_episode_rollout(topo, w, dyn_policies, cpu, mem, rps_profile, rng):
    """One STOCHASTIC per-minute rollout under preference `w` (dynamic variant).

    Mirrors `blackbox.simulator.simulate_policy`'s physics exactly (same
    causal state, same per-node
    consolidation energy model, same pack_pods caching by replica-count
    tuple), but samples each minute's replica target from
    `dyn_policies[i].sample(...)` (stochastic, preference-conditioned)
    instead of a deterministic `PolicyNet.act(...)`, and returns the
    per-tier sample caches alongside the objective triple so the caller can
    compute REINFORCE gradients.

    Returns `(f: np.ndarray[3], caches: List[List[dict]], f_per_minute:
    np.ndarray[T, 3])`. `caches[i]` is the list of this tier's per-minute
    sample caches for this one rollout; `f_per_minute[t]` is that same
    minute's own [latency, cost, power] triple, row-aligned with every
    tier's `caches[i][t]`, letting the caller credit each minute's samples
    with that minute's own reward instead of only the episode mean `f`.
    """
    if topo.node_classes is not None:
        raise NotImplementedError(
            "MORL --dynamic does not support the edge/cloud multiclass "
            "topology yet, same limitation as simulate_policy."
        )
    c = np.asarray(cpu, dtype=float)
    m = np.asarray(mem, dtype=float)
    n_tiers = topo.n_tiers

    base_service_s = np.array(
        [topo.tiers[i].service_demand_s / c[i] for i in range(n_tiers)]
    )
    demand_s = np.array([topo.tiers[i].service_demand_s for i in range(n_tiers)])
    capacity_max = np.array(
        [topo.tiers[i].replica_max / base_service_s[i] for i in range(n_tiers)]
    )

    replicas_prev = np.array(
        [topo.tiers[i].replica_min for i in range(n_tiers)], dtype=int
    )
    lam_prev = np.zeros(n_tiers)

    latencies = np.empty(len(rps_profile))
    costs = np.empty(len(rps_profile))
    powers = np.empty(len(rps_profile))
    caches = [[] for _ in range(n_tiers)]
    pack_cache: dict = {}

    for t, lam in enumerate(rps_profile):
        n_t = np.empty(n_tiers, dtype=int)
        for i in range(n_tiers):
            tier = topo.tiers[i]
            state = np.concatenate([w, [
                min(1.0, lam_prev[i] / capacity_max[i]) if capacity_max[i] > 0 else 0.0,
                replicas_prev[i] / tier.replica_max if tier.replica_max > 0 else 0.0,
                0.0,
            ]])
            raw, cache = dyn_policies[i].sample(state)
            caches[i].append(cache)
            frac = float(np.clip(raw, 0.0, 1.0))
            target = tier.replica_min + frac * (tier.replica_max - tier.replica_min)
            n_i = int(round(target))
            n_i = max(tier.replica_min, min(tier.replica_max, n_i))
            n_t[i] = n_i

        end_to_end_ms = 0.0
        cost_t = 0.0
        for i in range(n_tiers):
            tier = topo.tiers[i]
            wss = tier.working_set_base_gib + tier.working_set_per_rps_gib * lam
            cost_t += float(n_t[i]) * (
                c[i] * topo.price_cpu_per_core + m[i] * topo.price_mem_per_gib
            )
            if m[i] < wss:
                end_to_end_ms += topo.timeout_ms
                continue
            service_s = base_service_s[i]
            if n_t[i] == 0:
                end_to_end_ms += topo.timeout_ms if lam > 0 else 0.0
                continue
            capacity = n_t[i] / service_s
            end_to_end_ms += _tier_latency_ms(lam, service_s, capacity, topo.timeout_ms)
        latencies[t] = end_to_end_ms
        costs[t] = cost_t

        pack_key = tuple(n_t.tolist())
        counts = pack_cache.get(pack_key)
        if counts is None:
            counts = energy_model.pack_pods(n_t, c, m, topo.node_cpu_capacity_cores,
                                            topo.node_mem_capacity_gib, n_tiers)
            pack_cache[pack_key] = counts
        n_safe = np.where(n_t > 0, n_t, 1.0)
        offered_per_pod = lam * demand_s / n_safe
        pod_cores = np.minimum(c, offered_per_pod)
        if counts.shape[0] > 0:
            node_cores = pod_cores @ counts.T
            node_util = node_cores / topo.node_cpu_capacity_cores
            powers[t] = float(energy_model.estimate_node_power(node_util).sum())
        else:
            powers[t] = 0.0

        replicas_prev = n_t
        lam_prev[:] = lam

    f = np.array([float(np.mean(latencies)), float(np.mean(costs)), float(np.mean(powers))])
    f_per_minute = np.stack([latencies, costs, powers], axis=1)
    return f, caches, f_per_minute


def evaluate_dynamic_episode(topo, w, dyn_policies, cpu, mem, workload,
                             k_replications, base_seed, rng, sla_ms=None):
    """K-replication episode: mean objective triple, every tier's pooled
    per-minute sample caches across all K rollouts (each replication is its
    own independent stochastic rollout -- the realized load differs per
    replication, so the per-minute samples genuinely differ too, unlike the
    static case's single action shared across K replications), and every
    rollout's per-minute [latency, cost, power] triples pooled the same way
    (`step_F`, shape `(K*T, 3)`), row-aligned with `caches[i]` so the caller
    can credit each minute with its own reward (per-minute reward shaping)
    instead of only the episode mean.
    """
    rows = np.empty((k_replications, 3))
    caches = [[] for _ in range(topo.n_tiers)]
    step_rows = []
    for k in range(k_replications):
        profile = workload.realize(seed=base_seed + k)
        f, ep_caches, f_per_minute = _dynamic_episode_rollout(
            topo, w, dyn_policies, cpu, mem, profile, rng)
        rows[k] = f
        step_rows.append(f_per_minute)
        for i in range(topo.n_tiers):
            caches[i].extend(ep_caches[i])
    step_F = np.concatenate(step_rows, axis=0)
    if sla_ms is not None:
        # Hinge each replication before averaging (matches simulator.evaluate:
        # max(0, mean(x)) != mean(max(0, x)) in general). Hinge the per-minute
        # latencies the same way, so step-level rewards use the same SLA
        # framing as the episode-level one.
        rows[:, 0] = np.maximum(0.0, rows[:, 0] - float(sla_ms))
        step_F[:, 0] = np.maximum(0.0, step_F[:, 0] - float(sla_ms))
    mean = rows.mean(axis=0)
    return mean, caches, step_F


def _action_to_config(a: np.ndarray, xl: np.ndarray, xu: np.ndarray, n_tiers: int,
                      n_classes: int = 1):
    """Map a normalized [0,1]^d action to a simulator config (replicas rounded).

    With `n_classes > 1` each tier has a fourth decision value, the placement,
    denormalized to [0, n_classes) and floored to a node-class index.
    """
    per_tier = 4 if n_classes > 1 else 3
    row = (xl + a * (xu - xl)).reshape(n_tiers, per_tier)
    config = {
        "replicas": np.rint(row[:, 0]).astype(int),
        "cpu": row[:, 1].astype(float),
        "mem": row[:, 2].astype(float),
    }
    if n_classes > 1:
        config["place"] = np.clip(np.floor(row[:, 3]), 0, n_classes - 1).astype(int)
    return config


def _scalar_cost(f: np.ndarray, w: np.ndarray, ideal: np.ndarray,
                 nadir: np.ndarray, rho: float) -> float:
    """Augmented Tchebycheff scalarization (shared by `run_morl`'s static
    single-step case and `run_morl_dynamic`'s episodic case): the max term
    finds non-convex front regions, the weighted-sum term breaks ties and
    keeps a gradient everywhere.
    """
    span = np.where(nadir - ideal > 0, nadir - ideal, 1.0)
    fn = np.clip((f - ideal) / span, 0.0, None)
    return float(np.max(w * fn) + rho * np.sum(w * fn))


def greedy_sweep_dynamic(static_policy, dyn_policies, weights, xl, xu, n_tiers,
                         topology, workload, k_replications=5, base_seed=0,
                         sla_ms=None):
    """Dynamic-variant counterpart of `greedy_sweep`: for each preference `w`, the
    trained static head's noise-free cpu/mem plus the trained dynamic heads'
    noise-free per-minute policy, scored through the same deterministic
    `simulator.evaluate_policy` already uses (via `_MeanPolicyAdapter`).
    Post-training analysis only, like `greedy_sweep` -- not charged to the
    search budget.
    """
    F = np.empty((len(weights), 3))
    for idx, w in enumerate(weights):
        mu_static = static_policy.mean(w)
        row = (xl + mu_static * (xu - xl)).reshape(n_tiers, 2)
        cpu, mem = row[:, 0], row[:, 1]
        policies = [_MeanPolicyAdapter(w, dyn_policies[i]) for i in range(n_tiers)]
        res = simulator.evaluate_policy(topology, policies, cpu, mem, workload,
                                        k_replications=k_replications,
                                        base_seed=base_seed, sla_ms=sla_ms)
        F[idx] = res["mean"]
    return np.asarray(weights), F


def greedy_sweep(policy, weights, xl, xu, n_tiers, topology, workload,
                 k_replications=5, base_seed=0, sla_ms=None):
    """Objective vectors of the trained policy's greedy config per preference.

    Post-training analysis only (these evaluations are not part of the search
    budget): for each preference `w` the policy emits its noise-free mean action,
    which is mapped to a config and scored. Returns (weights, F) aligned by row,
    so a preference can be compared against the region a matching MOEA/D reference
    direction targets.
    """
    n_classes = topology.n_classes
    F = np.empty((len(weights), 3))
    for i, w in enumerate(weights):
        cfg = _action_to_config(policy.mean(w), xl, xu, n_tiers, n_classes)
        F[i] = simulator.evaluate(topology, cfg, workload,
                                  k_replications=k_replications,
                                  base_seed=base_seed, sla_ms=sla_ms)["mean"]
    return np.asarray(weights), F


def run_morl(topology, evals=500, batch=20, lr=0.2, warmup=20,
             partitions=6, archive_cap=20, k_replications=5, base_seed=0, seed=1,
             rho=0.05, verbose=False, return_policy=False, workload=None,
             test_workload=None, sla_ms=None):
    """Train the preference-conditioned policy under a fixed oracle-call budget.

    Returns a dict with the final non-dominated front `F`, the anytime history
    (`hist_n`, `hist_F`) matching the aggregator's format, and `n_eval_calls`.
    With `return_policy=True` the dict also carries the trained `policy`, the
    preference `weights`, and the `(xl, xu, n_tiers, workload)` needed to score a
    greedy sweep (see `greedy_sweep`). `workload` defaults to the synthetic
    diurnal profile.

    When `test_workload` is given the held-out protocol is used: the reported
    front `F` is the trained policy's greedy preference sweep re-scored on the
    test workload (with `F_train` the same sweep on the train workload), so it is
    directly comparable to the EAs' final configs re-scored on the same test day.
    These post-training sweeps are not charged to the search budget.
    """
    from blackbox.workload import synthetic_diurnal

    rng = np.random.default_rng(seed)
    workload = synthetic_diurnal() if workload is None else workload
    n_tiers = topology.n_tiers
    n_classes = topology.n_classes
    # Per tier: [replicas, cpu, mem] plus, for the edge/cloud topology, a
    # placement value in [0, n_classes) floored to a node-class index.
    per_tier = 4 if n_classes > 1 else 3
    d = per_tier * n_tiers

    xl, xu = [], []
    for t in topology.tiers:
        xl += [t.replica_min, t.cpu_min, t.mem_min]
        xu += [t.replica_max, t.cpu_max, t.mem_max]
        if n_classes > 1:
            xl += [0.0]
            xu += [float(n_classes)]
    xl, xu = np.asarray(xl, float), np.asarray(xu, float)

    # Reference preference set: das-dennis over 3 objectives. Includes the pure
    # perf/cost/energy corners and (for even partitions) the balanced centre.
    weights = get_reference_directions("das-dennis", 3, n_partitions=partitions)

    policy = PreferenceGaussianPolicy(d, n_obj=3, rng=rng)

    # Online objective normalization (ideal/nadir estimates) and per-weight
    # reward baselines for REINFORCE variance reduction.
    ideal = np.full(3, np.inf)
    nadir = np.full(3, -np.inf)
    baseline = {}  # weight index -> EMA of reward

    archive = BoundedArchive(archive_cap)  # reported front (cardinality-capped)
    hist_n, hist_F = [], []
    n_eval = 0

    def evaluate_config(cfg):
        nonlocal n_eval, ideal, nadir
        res = simulator.evaluate(topology, cfg, workload,
                                 k_replications=k_replications, base_seed=base_seed,
                                 sla_ms=sla_ms)
        f = res["mean"]
        n_eval += 1
        # Retain the config so the best-found set can be re-scored on the held-out
        # day, the fair counterpart to the EAs' res.X (see BoundedArchive).
        archive.add(f, cfg)
        # ideal/nadir track *all* evaluations (not just the archive) so the reward
        # normalization sees the true objective ranges.
        ideal = np.minimum(ideal, f)
        nadir = np.maximum(nadir, f)
        return f

    def scalar_cost(f, w):
        return _scalar_cost(f, w, ideal, nadir, rho)

    # --- Warmup: random configs to seed normalization and baselines. ----------
    wu = min(warmup, evals)
    for j in range(wu):
        w = weights[j % len(weights)]
        a = rng.random(d)
        f = evaluate_config(_action_to_config(a, xl, xu, n_tiers, n_classes))
        r = -scalar_cost(f, w)
        wi = j % len(weights)
        baseline[wi] = r if wi not in baseline else 0.5 * baseline[wi] + 0.5 * r
    hist_n.append(n_eval)
    hist_F.append(archive.front())

    # --- REINFORCE training loop. ---------------------------------------------
    gen = 0
    while n_eval < evals:
        bsz = min(batch, evals - n_eval)
        caches, advs = [], []
        for j in range(bsz):
            wi = (gen * batch + j) % len(weights)
            w = weights[wi]
            a, cache = policy.sample(w)
            f = evaluate_config(_action_to_config(a, xl, xu, n_tiers, n_classes))
            r = -scalar_cost(f, w)
            base = baseline.get(wi, r)
            baseline[wi] = 0.9 * base + 0.1 * r
            caches.append(cache)
            advs.append(r - base)

        # Standardize advantages within the batch -> scale-free, robust to the
        # non-stationary online normalization.
        advs = np.asarray(advs)
        if advs.std() > 1e-8:
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        else:
            advs = advs - advs.mean()

        gA = np.zeros_like(policy.A)
        gb = np.zeros_like(policy.b)
        gls = np.zeros_like(policy.log_std)
        for cache, adv in zip(caches, advs):
            dA, db, dls = policy.grads(cache, float(adv))
            gA += dA; gb += db; gls += dls
        inv = 1.0 / len(caches)
        policy.apply(gA * inv, gb * inv, gls * inv, lr)

        gen += 1
        hist_n.append(n_eval)
        hist_F.append(archive.front())
        if verbose:
            front = hist_F[-1]
            print(f"gen {gen:3d}  evals {n_eval:4d}  |front| {len(front):3d}  "
                  f"mean std {policy.std.mean():.3f}")

    out = {"F": archive.front(), "hist_n": hist_n, "hist_F": hist_F,
           "n_eval_calls": n_eval}
    if test_workload is not None:
        from experiments._common import bounded_front
        # Front 1 (the deployable-policy view): the greedy preference sweep, scored
        # on both days so train/test are the same object type as the EAs' fronts.
        _, F_train = greedy_sweep(policy, weights, xl, xu, n_tiers, topology,
                                  workload, k_replications, base_seed, sla_ms)
        _, F_test = greedy_sweep(policy, weights, xl, xu, n_tiers, topology,
                                 test_workload, k_replications, base_seed, sla_ms)
        out["F"] = bounded_front(F_test, archive_cap)
        out["F_train"] = bounded_front(F_train, archive_cap)
        # Front 2 (the best-found view, the fair apples-to-apples with the EAs'
        # res.X): re-score the archived best-found configs on the held-out day.
        configs = [c for c in archive.front_configs() if c is not None]
        bf = np.array([
            simulator.evaluate(topology, cfg, test_workload,
                               k_replications=k_replications, base_seed=base_seed,
                               sla_ms=sla_ms)["mean"]
            for cfg in configs
        ]) if configs else np.empty((0, 3))
        out["F_bestfound"] = bounded_front(bf, archive_cap) if len(bf) else out["F"]
    if return_policy:
        out.update(policy=policy, weights=weights, xl=xl, xu=xu,
                   n_tiers=n_tiers, workload=workload)
    return out


def run_morl_dynamic(topology, evals=500, lr=0.2, lr_end=None, warmup=10,
                     partitions=6, archive_cap=20, k_replications=5, base_seed=0,
                     seed=1, rho=0.05, hidden_dim=None, verbose=False,
                     workload=None, test_workload=None, sla_ms=None, batch=20,
                     latency_reward_floor=0.0, return_policy=False):
    """Train a preference-conditioned per-minute policy by episodic REINFORCE
    (dynamic variant).

    cpu/mem per tier stay a single preference-conditioned static action (the
    existing `PreferenceGaussianPolicy`, narrowed to 2*n_tiers dims since
    replicas are no longer part of it), decided once per episode exactly like
    `run_morl`'s static case. Replicas become a per-minute
    `DynamicReplicaPolicy` per tier -- same small-MLP capacity as
    `blackbox.policy.PolicyNet` (the neuroevolution-searched policy), so
    the RL-vs-EA comparison is about search method, not
    confounded by policy capacity -- sampled fresh every simulated minute.

    One episode = one K-replication rollout under one sampled preference `w`
    (`evaluate_dynamic_episode`). `batch` episodes are collected before a
    single gradient update, mirroring `run_morl`'s batched multi-sample
    updates and giving the same batch-level advantage standardization
    (mean/std over the batch), instead of the noisier single-episode update
    a first version used.

    The static cpu/mem head is a single per-episode decision, so it keeps
    using one advantage per episode (the scalarized episode-mean reward
    minus a per-preference EMA baseline), same as `run_morl`'s static case.

    The per-minute replica heads do NOT reuse that episode-level advantage
    for every one of their samples -- crediting every minute in an episode
    with the same single episode-mean-based number means an isolated bad
    minute (a capacity miss that spikes latency to the SLA
    timeout) gets exactly the same blame as every good minute around it,
    and a policy can't learn "don't do that specific thing" without also
    being pushed on unrelated minutes. Instead each minute is scored by its
    OWN scalarized [latency, cost, power] triple (`step_F` from
    `evaluate_dynamic_episode`) against a running `step_ideal`/`step_nadir`
    (kept separate from the episode-level `ideal`/`nadir`, since a single
    minute's raw latency can be far more extreme than any episode mean) and
    a per-preference EMA baseline of per-step reward. All per-minute
    advantages in the batch are standardized together, then each cache's
    gradient is weighted by its own advantage and the per-tier total is
    divided by the number of minutes that tier contributed across the
    batch -- one average over actual samples, not the old average-within-
    episode-then-average-again-across-episodes.

    `evals` counts EPISODES (K-replication rollouts), the same budget unit
    NSGA-II/MOEA-D/`run_morl`'s oracle calls use, so runs stay
    budget-comparable across every method. No archive/best-found front is
    reported here (unlike `run_morl`'s morl_bf): "best single sample seen
    during training" doesn't have a clean meaning for a continuously-updating
    multi-step policy the way it does for a static per-episode config, so
    only the final trained policy's greedy sweep is reported.

    `lr_end`, if given, linearly anneals the learning rate from `lr` down to
    `lr_end` over the course of training (by fraction of `evals` consumed),
    applied to both the static head and every per-minute replica head at
    each batch update. `lr_end=None` (default) keeps the old constant-lr
    behavior. Motivation: a higher starting lr for faster early learning,
    decaying toward the known-safe 0.2 floor so late training doesn't retain
    lr=0.6's instability (which reintroduced near-timeout latency spikes).

    `latency_reward_floor` guards against a specific failure mode found with
    lr annealing + per-minute reward shaping: at an extreme preference corner
    (das-dennis includes pure w=(0,1,0)/(0,0,1), all weight on cost or
    energy), the per-step scalarized reward gives latency literally ZERO
    weight, so the replica policy has no training signal telling it that
    starving replicas into a timeout is bad, as long as it saves cost/energy.
    Most preferences train fine (nearby weights still have nonzero latency
    weight), but a couple of corner-adjacent points can still collapse to
    near-5000ms latency (seen on seed 2 of a 10-seed sweep, 2 of 11 front
    points, while 9 of 11 were a clean 13-16ms). The floor is applied only to
    the per-step reward's weight vector (`max(w[latency], floor)`, not
    renormalized), not the episode-level reward used for the static cpu/mem
    head or the reported objective values themselves -- it changes what the
    replica policy is trained to avoid, not the objective definition or the
    static head's incentives. `latency_reward_floor=0.0` (default) is a
    no-op, matching old behavior.
    """
    from blackbox.policy import HIDDEN_DIM as _default_hidden
    from blackbox.workload import synthetic_diurnal
    from experiments._common import bounded_front

    hidden_dim = _default_hidden if hidden_dim is None else hidden_dim
    rng = np.random.default_rng(seed)
    workload = synthetic_diurnal() if workload is None else workload
    n_tiers = topology.n_tiers

    xl, xu = [], []
    for t in topology.tiers:
        xl += [t.cpu_min, t.mem_min]
        xu += [t.cpu_max, t.mem_max]
    xl, xu = np.asarray(xl, float), np.asarray(xu, float)
    d_static = 2 * n_tiers

    static_policy = PreferenceGaussianPolicy(d_static, n_obj=3, rng=rng)
    dyn_policies = [DynamicReplicaPolicy(hidden_dim, in_dim=3 + 3, rng=rng)
                    for _ in range(n_tiers)]
    weights = get_reference_directions("das-dennis", 3, n_partitions=partitions)

    ideal = np.full(3, np.inf)
    nadir = np.full(3, -np.inf)
    baseline: dict = {}
    # Per-minute reward shaping: own online range and per-preference baseline
    # for step-level rewards, kept separate from the episode-level ideal/
    # nadir/baseline above -- a single bad minute's raw latency can be far
    # more extreme than any episode mean, so it needs its own normalization.
    step_ideal = np.full(3, np.inf)
    step_nadir = np.full(3, -np.inf)
    step_baseline: dict = {}
    hist_n, hist_F = [], []
    front_samples: list = []
    n_eval = 0

    def static_cpu_mem(a):
        row = (xl + a * (xu - xl)).reshape(n_tiers, 2)
        return row[:, 0], row[:, 1]

    # --- Warmup: greedy (mean) action, seeds ideal/nadir/baseline. -------------
    wu = min(warmup, evals)
    for j in range(wu):
        w = weights[j % len(weights)]
        cpu, mem = static_cpu_mem(np.full(d_static, 0.5))
        policies = [_MeanPolicyAdapter(w, dyn_policies[i]) for i in range(n_tiers)]
        res = simulator.evaluate_policy(topology, policies, cpu, mem, workload,
                                        k_replications=k_replications,
                                        base_seed=base_seed, sla_ms=sla_ms)
        f = res["mean"]
        n_eval += 1
        ideal = np.minimum(ideal, f)
        nadir = np.maximum(nadir, f)
        wi = j % len(weights)
        r = -_scalar_cost(f, w, ideal, nadir, rho)
        baseline[wi] = r if wi not in baseline else 0.5 * baseline[wi] + 0.5 * r
    hist_n.append(n_eval)
    hist_F.append(np.empty((0, 3)))

    # --- Episodic REINFORCE, batched updates. -----------------------------------
    gen = 0
    while n_eval < evals:
        bsz = min(batch, evals - n_eval)
        batch_static_caches, batch_dyn_caches, advs, rs = [], [], [], []
        batch_step_F, batch_wi = [], []
        for j in range(bsz):
            wi = (gen * batch + j) % len(weights)
            w = weights[wi]

            a_static, cache_static = static_policy.sample(w)
            a_static = np.clip(a_static, 0.0, 1.0)
            cpu, mem = static_cpu_mem(a_static)
            f, caches, step_F = evaluate_dynamic_episode(
                topology, w, dyn_policies, cpu, mem, workload,
                k_replications, base_seed, rng, sla_ms,
            )
            n_eval += 1
            ideal = np.minimum(ideal, f)
            nadir = np.maximum(nadir, f)

            r = -_scalar_cost(f, w, ideal, nadir, rho)
            base = baseline.get(wi, r)
            baseline[wi] = 0.9 * base + 0.1 * r

            batch_static_caches.append(cache_static)
            batch_dyn_caches.append(caches)
            batch_step_F.append(step_F)
            batch_wi.append(wi)
            advs.append(r - base)
            front_samples.append(f)
            rs.append(r)

        # Standardize advantages within the batch -> scale-free, robust to the
        # non-stationary online normalization (same as `run_morl`'s static case).
        advs = np.asarray(advs)
        if advs.std() > 1e-8:
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        else:
            advs = advs - advs.mean()

        # Per-minute reward shaping: score every minute in the batch by its
        # OWN scalarized [latency, cost, power] triple against a running
        # step-level ideal/nadir and per-preference baseline, instead of
        # reusing the single episode-level `advs` above for every one of
        # that episode's per-minute samples.
        all_step_F = np.vstack(batch_step_F)
        step_ideal = np.minimum(step_ideal, all_step_F.min(axis=0))
        step_nadir = np.maximum(step_nadir, all_step_F.max(axis=0))

        batch_step_advs = []
        for step_F, wi in zip(batch_step_F, batch_wi):
            w = weights[wi]
            if latency_reward_floor > 0.0 and w[0] < latency_reward_floor:
                w = w.copy()
                w[0] = latency_reward_floor
            step_r = np.array([
                -_scalar_cost(sf, w, step_ideal, step_nadir, rho) for sf in step_F
            ])
            sbase = step_baseline.get(wi, float(step_r.mean()))
            step_baseline[wi] = 0.9 * sbase + 0.1 * float(step_r.mean())
            batch_step_advs.append(step_r - sbase)

        all_step_advs = np.concatenate(batch_step_advs)
        step_mean = all_step_advs.mean()
        step_scale = all_step_advs.std() + 1e-8 if all_step_advs.std() > 1e-8 else 1.0
        batch_step_advs = [(sa - step_mean) / step_scale for sa in batch_step_advs]

        # Linear lr anneal (lr -> lr_end over training progress), constant lr
        # when lr_end is None. Progress uses n_eval (post-batch) / evals.
        current_lr = _anneal_lr(lr, lr_end, n_eval / evals)

        gA = np.zeros_like(static_policy.A)
        gb = np.zeros_like(static_policy.b)
        gls_s = np.zeros_like(static_policy.log_std)
        for cache_static, adv in zip(batch_static_caches, advs):
            dA, db, dls = static_policy.grads(cache_static, float(adv))
            gA += dA; gb += db; gls_s += dls
        inv = 1.0 / bsz
        static_policy.apply(gA * inv, gb * inv, gls_s * inv, current_lr)

        for i, pol in enumerate(dyn_policies):
            gW1 = np.zeros_like(pol.W1)
            gb1 = np.zeros_like(pol.b1)
            gw2 = np.zeros_like(pol.w2)
            gb2 = 0.0
            gls = 0.0
            n_contrib = 0
            for caches, step_adv in zip(batch_dyn_caches, batch_step_advs):
                if not caches[i]:
                    continue
                for c, sa in zip(caches[i], step_adv):
                    dW1, db1, dw2, db2, dls = pol.grads(c)
                    gW1 += sa * dW1; gb1 += sa * db1
                    gw2 += sa * dw2; gb2 += sa * db2
                    gls += sa * dls
                    n_contrib += 1
            if n_contrib == 0:
                continue
            inv2 = 1.0 / n_contrib
            pol.apply(gW1 * inv2, gb1 * inv2, gw2 * inv2, gb2 * inv2,
                     gls * inv2, current_lr)

        gen += 1
        hist_n.append(n_eval)
        hist_F.append(nondominated(np.array(front_samples)))
        if verbose:
            print(f"batch {gen:4d}  evals {n_eval:4d}  lr {current_lr:.3f}  "
                  f"mean r {np.mean(rs):8.3f}  "
                  f"std(cpu/mem) {static_policy.std.mean():.3f}"
                  f"  std(replica, tier0) {dyn_policies[0].std:.3f}")

    _, F_train = greedy_sweep_dynamic(static_policy, dyn_policies, weights, xl, xu,
                                      n_tiers, topology, workload, k_replications,
                                      base_seed, sla_ms)
    out = {"F": bounded_front(F_train, archive_cap), "hist_n": hist_n,
           "hist_F": hist_F, "n_eval_calls": n_eval}
    if test_workload is not None:
        _, F_test = greedy_sweep_dynamic(static_policy, dyn_policies, weights, xl,
                                         xu, n_tiers, topology, test_workload,
                                         k_replications, base_seed, sla_ms)
        out["F"] = bounded_front(F_test, archive_cap)
        out["F_train"] = bounded_front(F_train, archive_cap)
    if return_policy:
        # For analyze_fronts's dynamic preference-behavior check: everything
        # greedy_sweep_dynamic needs to re-sweep this trained policy later.
        out.update(static_policy=static_policy, dyn_policies=dyn_policies,
                    weights=weights, xl=xl, xu=xu, n_tiers=n_tiers,
                    workload=workload)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, default=500)
    ap.add_argument("--batch", type=int, default=20, help="rollouts per policy update")
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--lr-end", type=float, default=None,
                    help="--dynamic only: linearly anneal lr from --lr down "
                         "to this floor over training. Default (None): "
                         "constant lr")
    ap.add_argument("--latency-reward-floor", type=float, default=0.0,
                    help="--dynamic only: minimum latency weight used in the "
                         "per-step reward, even at preference corners where "
                         "the real preference weight on latency is 0. "
                         "Default 0.0 (no floor, old behavior)")
    ap.add_argument("--warmup", type=int, default=20, help="random configs before training")
    ap.add_argument("--partitions", type=int, default=6,
                    help="das-dennis partitions for the preference set")
    ap.add_argument("--archive", type=int, default=20,
                    help="reported-front cap (match the EA population size for fairness)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tiers", type=int, default=3)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--dynamic", action="store_true",
                    help="dynamic variant: train a preference-conditioned per-minute "
                         "replica POLICY by episodic REINFORCE instead of a "
                         "single-step static config; only cpu/mem stay a "
                         "static per-episode action. --evals counts episodes "
                         "(K-replication rollouts), not single actions. Not "
                         "combinable with --edge.")
    ap.add_argument("--hidden-dim", type=int, default=None,
                    help="--dynamic only: hidden width of each tier's replica "
                         "policy MLP (default: blackbox.policy.HIDDEN_DIM, "
                         "matching the neuroevolution-searched policy "
                         "capacity for a fair comparison)")
    ap.add_argument("--plot", default=None,
                    help="path to save convergence/front figures (default: "
                         "docs/figures/morl_run.png)")
    ap.add_argument("--no-plot", action="store_true", help="skip the figures")
    from experiments._common import (
        add_workload_args, add_topology_args, build_train_test, build_topology,
    )
    add_workload_args(ap)
    add_topology_args(ap)
    args = ap.parse_args()

    topo = build_topology(args)
    if args.dynamic and args.edge:
        raise SystemExit("--dynamic does not support --edge yet "
                         "(the per-minute rollout is single-location-only)")
    train_wl, test_wl = build_train_test(args)

    if args.dynamic:
        out = run_morl_dynamic(
            topo, evals=args.evals, lr=args.lr, lr_end=args.lr_end,
            warmup=args.warmup, partitions=args.partitions,
            archive_cap=args.archive, k_replications=args.k,
            base_seed=args.base_seed, seed=args.seed,
            hidden_dim=args.hidden_dim, verbose=True, batch=args.batch,
            workload=train_wl, test_workload=test_wl, sla_ms=args.sla_ms,
            latency_reward_floor=args.latency_reward_floor,
        )
    else:
        out = run_morl(
            topo, evals=args.evals, batch=args.batch, lr=args.lr, warmup=args.warmup,
            partitions=args.partitions, archive_cap=args.archive, k_replications=args.k,
            base_seed=args.base_seed, seed=args.seed, verbose=True,
            workload=train_wl, test_workload=test_wl, sla_ms=args.sla_ms,
        )

    F = out["F"]
    where = "held-out test" if test_wl is not None else "train"
    print(f"\nMORL done: {out['n_eval_calls']} oracle calls, "
          f"{len(F)} non-dominated points ({where})")
    print("objective ranges (min, max):")
    for j, name in enumerate(["latency_ms", "cost", "energy_W"]):
        print(f"  {name:12s} {F[:, j].min():10.3f}  {F[:, j].max():10.3f}")

    if not args.no_plot:
        from experiments._common import plot_run_from_history
        plot_run_from_history(out["hist_n"], out["hist_F"], F, "morl",
                              args.plot or "docs/figures/morl_run.png", label="MORL")

    if args.out:
        from experiments._common import save_history
        save_history(args.out, F, out["hist_n"], out["hist_F"], algo="morl",
                     seed=args.seed, F_train=out.get("F_train"))
        # Held-out protocol: also emit the best-found front (archived configs
        # re-scored on the test day) as a separate "morl_bf" method, the fair
        # counterpart to the EAs' res.X. Same trained policy, no extra search
        # budget. File name: morl_seedN.npz -> morl_bf_seedN.npz. Not produced
        # in --dynamic mode: "best single sample during training" doesn't
        # generalize cleanly to a continuously-updating multi-step policy.
        if "F_bestfound" in out:
            d, base = os.path.split(args.out)
            bf_out = os.path.join(d, base.replace("morl", "morl_bf", 1)
                                  if base.startswith("morl") else "morl_bf_" + base)
            save_history(bf_out, out["F_bestfound"], out["hist_n"], out["hist_F"],
                         algo="morl_bf", seed=args.seed, F_train=out.get("F_train"))


if __name__ == "__main__":
    main()
