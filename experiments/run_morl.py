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
weights (the same das-dennis directions MOEA/D decomposes with, perf, cost, and
energy corners plus a balanced centre) traces out the Pareto front. The decision
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

from blackbox import default_topology
from blackbox import simulator


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
    """

    def __init__(self, cap: int):
        self.cap = int(cap)
        self.F = np.empty((0, 3))

    def add(self, f: np.ndarray) -> None:
        F = np.vstack([self.F, np.asarray(f, float)[None, :]])
        F = F[NonDominatedSorting().do(F, only_non_dominated_front=True)]
        while len(F) > self.cap:
            F = np.delete(F, int(np.argmin(self._crowding(F))), axis=0)
        self.F = F

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


def _action_to_config(a: np.ndarray, xl: np.ndarray, xu: np.ndarray, n_tiers: int):
    """Map a normalized [0,1]^d action to a simulator config (replicas rounded)."""
    row = (xl + a * (xu - xl)).reshape(n_tiers, 3)
    return {
        "replicas": np.rint(row[:, 0]).astype(int),
        "cpu": row[:, 1].astype(float),
        "mem": row[:, 2].astype(float),
    }


def greedy_sweep(policy, weights, xl, xu, n_tiers, topology, workload,
                 k_replications=5, base_seed=0):
    """Objective vectors of the trained policy's greedy config per preference.

    Post-training analysis only (these evaluations are not part of the search
    budget): for each preference `w` the policy emits its noise-free mean action,
    which is mapped to a config and scored. Returns (weights, F) aligned by row,
    so a preference can be compared against the region a matching MOEA/D reference
    direction targets.
    """
    F = np.empty((len(weights), 3))
    for i, w in enumerate(weights):
        cfg = _action_to_config(policy.mean(w), xl, xu, n_tiers)
        F[i] = simulator.evaluate(topology, cfg, workload,
                                  k_replications=k_replications,
                                  base_seed=base_seed)["mean"]
    return np.asarray(weights), F


def run_morl(topology, evals=500, batch=20, lr=0.2, warmup=20,
             partitions=6, archive_cap=20, k_replications=5, base_seed=0, seed=1,
             rho=0.05, verbose=False, return_policy=False):
    """Train the preference-conditioned policy under a fixed oracle-call budget.

    Returns a dict with the final non-dominated front `F`, the anytime history
    (`hist_n`, `hist_F`) matching the aggregator's format, and `n_eval_calls`.
    With `return_policy=True` the dict also carries the trained `policy`, the
    preference `weights`, and the `(xl, xu, n_tiers, workload)` needed to score a
    greedy sweep (see `greedy_sweep`).
    """
    from blackbox.workload import synthetic_diurnal

    rng = np.random.default_rng(seed)
    workload = synthetic_diurnal()
    n_tiers = topology.n_tiers
    d = 3 * n_tiers

    xl, xu = [], []
    for t in topology.tiers:
        xl += [t.replica_min, t.cpu_min, t.mem_min]
        xu += [t.replica_max, t.cpu_max, t.mem_max]
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
                                 k_replications=k_replications, base_seed=base_seed)
        f = res["mean"]
        n_eval += 1
        archive.add(f)
        # ideal/nadir track *all* evaluations (not just the archive) so the reward
        # normalization sees the true objective ranges.
        ideal = np.minimum(ideal, f)
        nadir = np.maximum(nadir, f)
        return f

    def scalar_cost(f, w):
        span = np.where(nadir - ideal > 0, nadir - ideal, 1.0)
        fn = np.clip((f - ideal) / span, 0.0, None)
        # Augmented Tchebycheff: the max term finds non-convex front regions, the
        # weighted-sum term breaks ties and keeps a gradient everywhere.
        return float(np.max(w * fn) + rho * np.sum(w * fn))

    # --- Warmup: random configs to seed normalization and baselines. ----------
    wu = min(warmup, evals)
    for j in range(wu):
        w = weights[j % len(weights)]
        a = rng.random(d)
        f = evaluate_config(_action_to_config(a, xl, xu, n_tiers))
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
            f = evaluate_config(_action_to_config(a, xl, xu, n_tiers))
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
    if return_policy:
        out.update(policy=policy, weights=weights, xl=xl, xu=xu,
                   n_tiers=n_tiers, workload=workload)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, default=500)
    ap.add_argument("--batch", type=int, default=20, help="rollouts per policy update")
    ap.add_argument("--lr", type=float, default=0.2)
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
    args = ap.parse_args()

    topo = default_topology(args.tiers)
    out = run_morl(
        topo, evals=args.evals, batch=args.batch, lr=args.lr, warmup=args.warmup,
        partitions=args.partitions, archive_cap=args.archive, k_replications=args.k,
        base_seed=args.base_seed, seed=args.seed, verbose=True,
    )

    F = out["F"]
    print(f"\nMORL done: {out['n_eval_calls']} oracle calls, {len(F)} non-dominated points")
    print("objective ranges (min, max):")
    for j, name in enumerate(["latency_ms", "cost", "energy_W"]):
        print(f"  {name:12s} {F[:, j].min():10.3f}  {F[:, j].max():10.3f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.savez(
            args.out,
            F=np.atleast_2d(F),
            hist_n_evals=np.asarray(out["hist_n"], dtype=int),
            hist_F=np.array([np.atleast_2d(f) for f in out["hist_F"]], dtype=object),
            algo="morl", seed=args.seed, evals=args.evals,
        )
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
