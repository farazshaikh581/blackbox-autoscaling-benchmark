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

With --sla-ms S an extra *compliant-region* column is reported: the 2-D (cost,
energy) hypervolume restricted to points that meet the SLA. Under the SLA-hinge
framing the full 3-D HV rewards a method for exploring configs that *violate* the
very SLA the hinge encodes (they populate the latency>SLA tail); the
compliant-region HV instead scores only the cost/energy trade-off an operator
would actually deploy. On hinged fronts (--hinged, latency stored as
max(0, lat-S)) compliant == latency 0; on raw-latency fronts compliant == lat<=S.

    python -m experiments.aggregate --results results
    python -m experiments.aggregate --results results_sla --sla-ms 20 --hinged
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

from viz import METHOD, plot_fronts_faceted, plot_hv_convergence  # noqa: E402

OBJ = ["latency_ms", "cost", "energy_W"]
# Method identities (color, marker, label) are shared repo-wide in viz.py.
COLOR = {a: m["color"] for a, m in METHOD.items()}
LABEL = {a: m["label"] for a, m in METHOD.items()}
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


def compliant_mask(F, sla_ms, hinged):
    """Rows meeting the SLA. On hinged fronts (max(0, lat-sla) stored) the
    compliant set is exactly latency 0; on raw-latency fronts it is lat<=sla."""
    lat = np.asarray(F, float)[:, 0]
    return lat <= 1e-9 if hinged else lat <= sla_ms


def wilcoxon_block(metric, runs, algos, values):
    """Pairwise Wilcoxon signed-rank over per-seed values[a] (aligned to runs[a])."""
    print(f"\nWilcoxon signed-rank ({metric}, paired by seed):")
    for i in range(len(algos)):
        for j in range(i + 1, len(algos)):
            a, b = algos[i], algos[j]
            xa = {r["seed"]: v for r, v in zip(runs[a], values[a])}
            xb = {r["seed"]: v for r, v in zip(runs[b], values[b])}
            seeds = sorted(set(xa) & set(xb))
            if len(seeds) >= 3:
                da = np.array([xa[s] for s in seeds]); db = np.array([xb[s] for s in seeds])
                try:
                    stat, p = wilcoxon(da, db)
                    print(f"  {LABEL[a]} vs {LABEL[b]}: n={len(seeds)}  p={p:.4f}  "
                          f"(mean {da.mean():.4f} vs {db.mean():.4f})")
                except ValueError as e:
                    print(f"  {LABEL[a]} vs {LABEL[b]}: {e}")
            else:
                print(f"  {LABEL[a]} vs {LABEL[b]}: need >=3 shared seeds, have {len(seeds)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--sla-ms", type=float, default=None,
                    help="report a compliant-region (cost,energy) HV column for "
                         "points meeting this SLA (ms)")
    ap.add_argument("--hinged", action="store_true",
                    help="fronts store hinged latency max(0, lat-sla); compliant "
                         "== latency 0 (set when aggregating an --sla-ms run)")
    args = ap.parse_args()

    runs = load_runs(args.results)
    if not runs:
        raise SystemExit(f"no .npz runs found in {args.results}/")
    algos = [a for a in ["nsga2", "moead", "morl", "morl_bf", "random", "hpa"] if a in runs]

    # Shared normalization box from the union of final fronts.
    allF = np.vstack([r["F"] for a in algos for r in runs[a]])
    ideal, nadir = allF.min(0), allF.max(0)
    span = np.where(nadir - ideal > 0, nadir - ideal, 1.0)
    norm = lambda F: (np.asarray(F, float) - ideal) / span

    # Shared reference set Z = non-dominated union of all final fronts (normalized).
    Z = nondominated(norm(allF))
    hv = HV(ref_point=np.full(3, REF))
    igdp, gdp = IGDPlus(Z), GDPlus(Z)

    has_comp = args.sla_ms is not None
    hv2 = HV(ref_point=np.full(2, REF))  # 2-D (cost, energy) compliant-region HV
    header = f"{'algo':8s} {'runs':>4s} {'HV':>16s} {'IGD+':>16s} {'GD+':>16s}"
    if has_comp:
        header += f" {'compHV(SLA)':>16s}"
    print(header)
    table_rows, per_algo_hv, per_algo_comp = [], {}, {}
    for a in algos:
        hvs, igds, gds, chvs = [], [], [], []
        for r in runs[a]:
            Fn = np.clip(norm(r["F"]), None, REF)
            hvs.append(hv(Fn)); igds.append(igdp(Fn)); gds.append(gdp(Fn))
            if has_comp:
                ce = Fn[compliant_mask(r["F"], args.sla_ms, args.hinged)][:, 1:3]
                chvs.append(hv2(ce) if len(ce) else 0.0)
        per_algo_hv[a] = np.array(hvs)
        row = dict(algo=a, runs=len(hvs),
                   hv_mean=np.mean(hvs), hv_std=np.std(hvs),
                   igdplus_mean=np.mean(igds), igdplus_std=np.std(igds),
                   gdplus_mean=np.mean(gds), gdplus_std=np.std(gds))
        line = (f"{LABEL[a]:8s} {len(hvs):>4d} "
                f"{row['hv_mean']:8.4f}+-{row['hv_std']:.4f} "
                f"{row['igdplus_mean']:8.4f}+-{row['igdplus_std']:.4f} "
                f"{row['gdplus_mean']:8.4f}+-{row['gdplus_std']:.4f}")
        if has_comp:
            per_algo_comp[a] = np.array(chvs)
            row["comp_hv_mean"] = float(np.mean(chvs)); row["comp_hv_std"] = float(np.std(chvs))
            line += f" {row['comp_hv_mean']:8.4f}+-{row['comp_hv_std']:.4f}"
        table_rows.append(row)
        print(line)

    # Wilcoxon signed-rank between algorithm pairs (paired by seed).
    wilcoxon_block("final HV", runs, algos, per_algo_hv)
    if has_comp:
        wilcoxon_block("compliant-region HV", runs, algos, per_algo_comp)

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
    curves = {}
    for a in algos:
        per_run = [(np.asarray(r["hist_n"], float),
                    np.array([hv(np.clip(norm(F), None, REF)) for F in r["hist_F"]]))
                   for r in runs[a] if r["hist_n"] is not None]
        if per_run:
            curves[a] = per_run
    plot_hv_convergence(curves, path)


def _plot_fronts(runs, algos, path):
    fronts = {}
    for a in algos:
        F = np.vstack([r["F"] for r in runs[a]])
        fronts[a] = F[F[:, 0] < 5000]  # drop timed-out/infeasible points
    plot_fronts_faceted(fronts, path)


if __name__ == "__main__":
    main()
