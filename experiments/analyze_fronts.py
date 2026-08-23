"""Analyze the RL and evolutionary fronts (docs/comparison_protocol.md analysis).

Two questions:

1. Preference behavior. Do the RL scalarization weights actually steer the
   policy across the trilemma? MOEA/D decomposes the objective space with
   das-dennis directions; the MORL policy is swept over the *same* direction set
   and queried greedily (one config per preference, `policy.mean(w)`, which is by
   definition the policy's response to preference `w`). We report whether the
   three pure corners (latency / cost / energy) specialize as their weight
   demands, and how consistently the emphasized objective is the one most reduced,
   then check whether the preference sweep *reaches the same region* the
   evolutionary fronts occupy (coverage and IGD of the swept points to the union
   EA front).

   Note: we deliberately do not read a per-subproblem incumbent back out of
   MOEA/D. Under the 500-evaluation budget its ~28 subproblems get only ~18
   evaluations each and are far from converged, and pymoo's raw-scale Tchebycheff
   decomposition does not map a reference direction to a clean objective-space
   region, so such a readout would not be interpretable. The direction *set* is
   the shared object of comparison; the fronts are compared directly.

2. Spatial relation of the fronts: two-set coverage (Zitzler's C-metric)
   between every pair of algorithms over all saved seeds: C(A, B) is the fraction
   of B's front weakly dominated by A's front.

Reads the saved final fronts in results/ for the coverage matrix, and re-trains
MORL in-process for one representative seed to obtain its greedy preference sweep
(not stored in the saved runs). Writes results/fig_weights.png.

    python -m experiments.analyze_fronts --results results --partitions 12 --seed 1
"""
from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from blackbox import default_topology  # noqa: E402
from experiments.run_morl import run_morl, greedy_sweep  # noqa: E402

OBJ = ["latency_ms", "cost", "energy_W"]
COLOR = {"nsga2": "#0072B2", "moead": "#E69F00", "morl": "#009E73",
         "morl_bf": "#CC79A7"}
LABEL = {"nsga2": "NSGA-II", "moead": "MOEA/D", "morl": "MORL",
         "morl_bf": "MORL-BF"}


# --- 2. Set coverage (C-metric) -------------------------------------------------

def set_coverage(A: np.ndarray, B: np.ndarray) -> float:
    """Fraction of B weakly dominated by at least one point of A (minimization)."""
    if len(A) == 0 or len(B) == 0:
        return float("nan")
    covered = 0
    for b in B:
        if np.any(np.all(A <= b, axis=1) & np.any(A < b, axis=1)):
            covered += 1
    return covered / len(B)


def load_fronts(results_dir):
    runs = defaultdict(dict)  # algo -> {seed: F}
    for p in sorted(glob.glob(os.path.join(results_dir, "*.npz"))):
        d = np.load(p, allow_pickle=True)
        algo = str(d["algo"]) if "algo" in d else os.path.basename(p).split("_")[0]
        seed = int(d["seed"]) if "seed" in d else 0
        runs[algo][seed] = np.atleast_2d(d["F"])
    return runs


def coverage_matrix(runs, algos):
    print("Two-set coverage C(row, col): fraction of the column front weakly "
          "dominated by the row front (mean +/- std over shared seeds).\n")
    header = "          " + "".join(f"{LABEL[b]:>16s}" for b in algos)
    print(header)
    M = np.full((len(algos), len(algos)), np.nan)
    for i, a in enumerate(algos):
        cells = []
        for j, b in enumerate(algos):
            if a == b:
                cells.append(f"{'--':>16s}")
                continue
            seeds = sorted(set(runs[a]) & set(runs[b]))
            vals = [set_coverage(runs[a][s], runs[b][s]) for s in seeds]
            M[i, j] = np.mean(vals)
            cells.append(f"{np.mean(vals):7.3f}+-{np.std(vals):.3f}")
        print(f"{LABEL[a]:>10s}" + "".join(cells))
    return M


# --- 1. Preference behavior -----------------------------------------------------

def corner_specialization(name, W, F):
    """For the three pure corners, is objective j minimized by direction e_j?"""
    # das-dennis includes the unit vectors; pick the direction closest to each.
    corners = {j: int(np.argmin(np.abs(W - np.eye(3)[j]).sum(axis=1))) for j in range(3)}
    print(f"\n{name}: corner specialization (does e_j minimize objective j?)")
    rows = np.array([F[corners[j]] for j in range(3)])
    ok = True
    for j in range(3):
        winner = int(np.argmin(rows[:, j]))
        mark = "ok" if winner == j else f"NO (min at e_{OBJ[winner]})"
        ok &= winner == j
        print(f"  emphasize {OBJ[j]:11s} -> {OBJ[j]:11s}={rows[j, j]:9.3f}   [{mark}]")
    return ok


def preference_consistency(W, F):
    """Fraction of directions whose most-emphasized objective is its smallest
    (normalized) achieved component, meaning the weight actually got what it
    asked for.
    """
    ideal, nadir = F.min(0), F.max(0)
    span = np.where(nadir - ideal > 0, nadir - ideal, 1.0)
    Fn = (F - ideal) / span
    hits = sum(int(np.argmax(w) == np.argmin(fn)) for w, fn in zip(W, Fn))
    return hits / len(W)


def plot_alignment(W, F_morl, region, path):
    """RL preference sweep (coloured by emphasized objective) over the region the
    evolutionary/RL fronts occupy (grey background)."""
    dom = np.argmax(W, axis=1)  # which objective each preference emphasizes
    pairs = [(0, 1), (0, 2), (1, 2)]
    # Latency is heavy-tailed (cost-extreme configs sit on the timeout cliff);
    # focus the latency axis on the knee so the RL band and the front are legible.
    lat_cap = max(F_morl[:, 0].max() * 1.1, np.quantile(region[:, 0], 0.85))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, (i, j) in zip(axes, pairs):
        ax.scatter(region[:, i], region[:, j], s=10, color="0.75", alpha=0.5,
                   linewidth=0, label="all fronts")
        sc = ax.scatter(F_morl[:, i], F_morl[:, j], c=dom, cmap="viridis",
                        marker="o", s=48, edgecolor="white", linewidth=0.4,
                        vmin=0, vmax=2, label="RL preference sweep")
        if i == 0:
            ax.set_xlim(right=lat_cap)
        ax.set_xlabel(OBJ[i]); ax.set_ylabel(OBJ[j])
        ax.grid(True, color="0.9"); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, loc="upper right")
    cbar = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.01, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["→latency", "→cost", "→energy"])
    fig.suptitle("RL preference sweep (colour = emphasized objective) over the "
                 "region the fronts occupy")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--partitions", type=int, default=12,
                    help="das-dennis partitions for the shared direction set, "
                         "must match run_moead.py's --partitions (default 12) "
                         "for the 'same directions MOEA/D decomposes with' "
                         "claim below to actually hold")
    ap.add_argument("--evals", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tiers", type=int, default=3)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=0)
    args = ap.parse_args()

    # --- Coverage over all saved seeds. --------------------------------------
    runs = load_fronts(args.results)
    algos = [a for a in ["nsga2", "moead", "morl", "morl_bf"] if a in runs]
    print("=" * 72)
    print("Spatial relation of the fronts")
    print("=" * 72)
    if len(algos) >= 2:
        coverage_matrix(runs, algos)
    else:
        print("need >=2 algorithms in results/ for coverage")

    # --- RL preference behavior (one representative seed). --------------------
    print("\n" + "=" * 72)
    print(f"RL preference behavior (seed {args.seed}, das-dennis p={args.partitions}, "
          f"sweeping the same directions MOEA/D decomposes with)")
    print("=" * 72)
    topo = default_topology(args.tiers)
    out = run_morl(topo, evals=args.evals, partitions=args.partitions, k_replications=args.k,
                   base_seed=args.base_seed, seed=args.seed, return_policy=True)
    Wm, F_morl = greedy_sweep(out["policy"], out["weights"], out["xl"], out["xu"],
                              out["n_tiers"], topo, out["workload"],
                              k_replications=args.k, base_seed=args.base_seed)

    ok_morl = corner_specialization("MORL", Wm, F_morl)
    print(f"\nPreference consistency (argmax weight == argmin normalized objective):"
          f"  MORL {preference_consistency(Wm, F_morl):.2f}")
    feas = F_morl[:, 0] < topo.timeout_ms
    print(f"feasible greedy points (latency below the timeout plateau): "
          f"{int(feas.sum())}/{len(F_morl)}, "
          f"underprovisioned preferences fall onto the timeout cliff")

    # Overlay the feasible preference sweep on the region the fronts occupy.
    region = np.vstack([runs[a][s] for a in algos for s in runs[a]])
    region = region[region[:, 0] < topo.timeout_ms]
    path = os.path.join(args.results, "fig_weights.png")
    plot_alignment(Wm[feas], F_morl[feas], region, path)
    print(f"\nwrote {path}")
    print(f"MORL corner specialization fully consistent: {ok_morl}")


if __name__ == "__main__":
    main()
