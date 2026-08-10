"""Aggregate multi-seed runs into the comparison: HV, IGD+/GD+, Wilcoxon, plots.

Reads results/<algo>_seed<seed>.npz (written by the runners), and produces:
  - anytime hypervolume vs cumulative evaluations (mean +/- std per algorithm)
  - final hypervolume, IGD+ and GD+ against a shared reference set
  - Wilcoxon signed-rank tests between algorithms across seeds
  - a Pareto-front overlay in objective space
  - results/comparison.csv and results/fig_*.png

Objectives are min-max normalized to a shared [0, 1] box (ideal and nadir from the
union of final fronts) so hypervolume is comparable across the mixed-scale
latency, cost, and energy axes. Reference point 1.1 in normalized space.

    python -m experiments.aggregate --results results
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

from pymoo.indicators.hv import HV  # noqa: E402
from pymoo.indicators.igd_plus import IGDPlus  # noqa: E402
from pymoo.indicators.gd_plus import GDPlus  # noqa: E402
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

OBJ = ["latency_ms", "cost", "energy_W"]
# Okabe-Ito colorblind-safe palette, assigned in fixed order.
COLOR = {"nsga2": "#0072B2", "moead": "#E69F00", "morl": "#009E73"}
LABEL = {"nsga2": "NSGA-II", "moead": "MOEA/D", "morl": "MORL"}
REF = 1.1  # hypervolume reference point in normalized space


def load_runs(results_dir):
    runs = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(results_dir, "*.npz"))):
        d = np.load(p, allow_pickle=True)
        algo = str(d["algo"]) if "algo" in d else os.path.basename(p).split("_")[0]
        runs[algo].append({
            "seed": int(d["seed"]) if "seed" in d else 0,
            "F": np.atleast_2d(d["F"]),
            "hist_n": d["hist_n_evals"] if "hist_n_evals" in d else None,
            "hist_F": d["hist_F"] if "hist_F" in d else None,
        })
    return runs


def nondominated(F):
    return F[NonDominatedSorting().do(F, only_non_dominated_front=True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    runs = load_runs(args.results)
    if not runs:
        raise SystemExit(f"no .npz runs found in {args.results}/")
    algos = [a for a in ["nsga2", "moead", "morl"] if a in runs]

    # Shared normalization box from the union of final fronts.
    allF = np.vstack([r["F"] for a in algos for r in runs[a]])
    ideal, nadir = allF.min(0), allF.max(0)
    span = np.where(nadir - ideal > 0, nadir - ideal, 1.0)
    norm = lambda F: (np.asarray(F, float) - ideal) / span

    # Shared reference set Z = non-dominated union of all final fronts (normalized).
    Z = nondominated(norm(allF))
    hv = HV(ref_point=np.full(3, REF))
    igdp, gdp = IGDPlus(Z), GDPlus(Z)

    print(f"{'algo':8s} {'runs':>4s} {'HV':>16s} {'IGD+':>16s} {'GD+':>16s}")
    table_rows, per_algo_hv = [], {}
    for a in algos:
        hvs, igds, gds = [], [], []
        for r in runs[a]:
            Fn = np.clip(norm(r["F"]), None, REF)
            hvs.append(hv(Fn)); igds.append(igdp(Fn)); gds.append(gdp(Fn))
        per_algo_hv[a] = np.array(hvs)
        row = dict(algo=a, runs=len(hvs),
                   hv_mean=np.mean(hvs), hv_std=np.std(hvs),
                   igdplus_mean=np.mean(igds), igdplus_std=np.std(igds),
                   gdplus_mean=np.mean(gds), gdplus_std=np.std(gds))
        table_rows.append(row)
        print(f"{LABEL[a]:8s} {len(hvs):>4d} "
              f"{row['hv_mean']:8.4f}+-{row['hv_std']:.4f} "
              f"{row['igdplus_mean']:8.4f}+-{row['igdplus_std']:.4f} "
              f"{row['gdplus_mean']:8.4f}+-{row['gdplus_std']:.4f}")

    # Wilcoxon signed-rank between algorithm pairs on final HV (paired by seed).
    print("\nWilcoxon signed-rank (final HV, paired by seed):")
    for i in range(len(algos)):
        for j in range(i + 1, len(algos)):
            a, b = algos[i], algos[j]
            xa = {r["seed"]: h for r, h in zip(runs[a], per_algo_hv[a])}
            xb = {r["seed"]: h for r, h in zip(runs[b], per_algo_hv[b])}
            seeds = sorted(set(xa) & set(xb))
            if len(seeds) >= 3:
                da = np.array([xa[s] for s in seeds]); db = np.array([xb[s] for s in seeds])
                try:
                    stat, p = wilcoxon(da, db)
                    print(f"  {LABEL[a]} vs {LABEL[b]}: n={len(seeds)}  p={p:.4f}  "
                          f"(mean HV {da.mean():.4f} vs {db.mean():.4f})")
                except ValueError as e:
                    print(f"  {LABEL[a]} vs {LABEL[b]}: {e}")
            else:
                print(f"  {LABEL[a]} vs {LABEL[b]}: need >=3 shared seeds, have {len(seeds)}")

    os.makedirs(args.results, exist_ok=True)
    _write_csv(os.path.join(args.results, "comparison.csv"), table_rows)
    _plot_convergence(runs, algos, norm, hv, os.path.join(args.results, "fig_convergence.png"))
    _plot_fronts(runs, algos, os.path.join(args.results, "fig_pareto.png"))
    print(f"\nwrote {args.results}/comparison.csv, fig_convergence.png, fig_pareto.png")


def _write_csv(path, rows):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _plot_convergence(runs, algos, norm, hv, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    grid = None
    for a in algos:
        curves = []
        for r in runs[a]:
            if r["hist_n"] is None:
                continue
            ns = np.asarray(r["hist_n"], float)
            hvs = np.array([hv(np.clip(norm(F), None, REF)) for F in r["hist_F"]])
            g = np.linspace(ns.min(), ns.max(), 60) if grid is None else grid
            grid = g
            curves.append(np.interp(g, ns, hvs))
        if not curves:
            continue
        C = np.vstack(curves)
        m, s = C.mean(0), C.std(0)
        ax.plot(grid, m, color=COLOR[a], lw=2, label=LABEL[a])
        ax.fill_between(grid, m - s, m + s, color=COLOR[a], alpha=0.15, linewidth=0)
    ax.set_xlabel("cumulative evaluations")
    ax.set_ylabel("hypervolume (normalized)")
    ax.set_title("Anytime hypervolume convergence")
    ax.grid(True, color="0.9")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_fronts(runs, algos, path):
    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (i, j) in zip(axes, pairs):
        for a in algos:
            F = np.vstack([r["F"] for r in runs[a]])
            F = F[F[:, 0] < 5000]  # drop timed-out/infeasible points
            ax.scatter(F[:, i], F[:, j], s=16, color=COLOR[a], alpha=0.6,
                       edgecolor="white", linewidth=0.3, label=LABEL[a])
        ax.set_xlabel(OBJ[i])
        ax.set_ylabel(OBJ[j])
        ax.grid(True, color="0.9")
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("Pareto-front overlay (feasible points)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
