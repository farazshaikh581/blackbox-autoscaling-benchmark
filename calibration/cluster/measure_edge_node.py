"""Measure a REAL second node class for the edge-cloud variant.

`blackbox/topology.py`'s `edge_cloud_topology()` ships a small "edge" NodeClass
(p_idle=20.0, p_max=60.0, alpha=1.4) documented as a guessed default, "to be
calibrated to the testbed". The cluster only has one physical node size (r0-r5,
6-core bare-metal RAPL). A genuine second class has to be emulated on one of
them instead. This pins busy workers to only `EDGE_CORES` of its 6 physical
cores, so the node never exercises its full package power, and refits
`(P_idle, P_max, alpha)` from real RAPL package energy over that restricted
range. It runs remotely on the target node via ssh. With `--edge-cores 6`
it sweeps the whole node; that is how the main power curve (r1, r2) was
measured.

`u` here is the level relative to the emulated class's own capacity
(`k_busy / EDGE_CORES`), not relative to the physical node's 6 cores. That is
the utilization semantic `energy_model`/`NodeClass` actually consumes: a class
at u=1.0 means fully loaded for that class, regardless of the physical
silicon underneath it. With `--edge-cores 6` on a full node, this also works
as a general full-utilization RAPL sweep, not just an edge-class emulation.

Run with the target node otherwise idle (teardown the blackbox namespace
first; a k3s system pod or two is fine, same background load the r0/r1/r2
consolidation sweep already tolerated):

    python -m calibration.cluster.measure_edge_node                # sweep + fit
    python -m calibration.cluster.measure_edge_node --node r3       # a different node

Writes calibration/data/edge_node_sweep_<node>.csv and
calibration/data/edge_node_measured_curve_<node>.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess

import numpy as np
from scipy.optimize import curve_fit

from viz import COLOR, INK_3, SINGLE, new_fig, note, style_axes, savefig

SSH = ["ssh", "-o", "BatchMode=yes"]
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "figures")

# Remote, stdlib-only worker: pin `k` busy loops to cores 0..k-1, hold for
# `window` seconds, print the real RAPL package energy_uj delta / window. Runs standalone
# over ssh (no repo checkout assumed on the node).
REMOTE_SCRIPT = r"""
import os, sys, time, multiprocessing as mp
PKG = "/sys/class/powercap/intel-rapl:0"
k = int(sys.argv[1]); window = float(sys.argv[2])

def busy(core):
    try: os.sched_setaffinity(0, {core})
    except OSError: pass
    x = 0.0001
    while True:
        for _ in range(100000):
            x = (x * 1.0000001 + 1.23) % 1e6
            x = x * x % 7.13

def read_uj(): return int(open(PKG + "/energy_uj").read())
def read_max(): return int(open(PKG + "/max_energy_range_uj").read())

procs = [mp.Process(target=busy, args=(c,), daemon=True) for c in range(k)]
for p in procs: p.start()
time.sleep(2.0)  # warmup
pmax = read_max()
e0 = read_uj()
time.sleep(window)
e1 = read_uj()
d = e1 - e0
if d < 0: d += pmax
watts = d / 1e6 / window
for p in procs: p.terminate()
for p in procs: p.join()
print(f"{watts:.3f}")
"""


def sh(node, cmd, timeout=60):
    return subprocess.run(SSH + [node, cmd], capture_output=True, text=True,
                          timeout=timeout)


def remote_measure(node, k, window):
    r = sh(node, f"sudo -n python3 -c {shlex.quote(REMOTE_SCRIPT)} {k} {window}",
          timeout=window + 30)
    if r.returncode != 0:
        raise RuntimeError(f"remote measure k={k} failed: {r.stderr.strip()[-300:]}")
    return float(r.stdout.strip())


def model(u, p_idle, p_max, alpha):
    return p_idle + (p_max - p_idle) * np.power(np.clip(u, 1e-9, 1), alpha)


def fit(U, P):
    # Free 3-parameter fit, not pinned to a raw k=0 idle reading.
    # already hit this exact trap on r0-r2. A raw idle sample can catch the
    # package in a deep C-state (single-digit watts) that badly understates
    # the fixed cost of a node actually in service, which silently flips
    # the model's consolidation-vs-spread ranking. The fix there was to fit
    # P_idle by least squares against the full (u, watts) sweep instead of
    # anchoring it to u=0. Same approach here.
    popt, pcov = curve_fit(model, U, P, p0=[15, 60, 1.4],
                          bounds=([0, 20, 0.2], [40, 200, 6]), maxfev=40000)
    pi, pm, a = popt
    perr = np.sqrt(np.diag(pcov))
    pred = model(U, pi, pm, a)
    r2 = 1 - ((P - pred) ** 2).sum() / ((P - P.mean()) ** 2).sum()
    return float(pi), float(pm), float(a), float(r2), perr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", default="r5", help="ssh alias of the target node")
    ap.add_argument("--edge-cores", type=int, default=4,
                    help="cores pinned busy at u=1.0 (matches NodeClass.cpu_capacity_cores)")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--window", type=float, default=6.0)
    args = ap.parse_args()

    r = sh(args.node, "sudo -n test -r /sys/class/powercap/intel-rapl:0/energy_uj && echo ok")
    if "ok" not in r.stdout:
        raise SystemExit(f"{args.node}: RAPL energy_uj not sudo-readable")

    print(f"node={args.node}  emulated edge capacity={args.edge_cores} cores  "
          f"passes={args.passes}  window={args.window}s")

    rows = []
    for p in range(1, args.passes + 1):
        for k in range(args.edge_cores + 1):
            watts = remote_measure(args.node, k, args.window)
            u = k / args.edge_cores
            rows.append({"pass": p, "k_cores": k, "u": u, "P_total": watts})
            print(f"  pass {p} k={k}/{args.edge_cores} (u={u:.2f}): "
                  f"P_total={watts:.2f} W", flush=True)

    out_csv = os.path.join(DATA_DIR, f"edge_node_sweep_{args.node}.csv")
    out_json = os.path.join(DATA_DIR, f"edge_node_measured_curve_{args.node}.json")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pass", "k_cores", "u", "P_total"])
        w.writeheader()
        w.writerows(rows)

    U = np.array([r["u"] for r in rows])
    P = np.array([r["P_total"] for r in rows])
    raw_idle_w = float(np.median([r["P_total"] for r in rows if r["k_cores"] == 0]))
    p_idle, p_max, alpha, r2, perr = fit(U, P)

    print(f"\nraw k=0 median (informational only, not the P_idle anchor, "
          f"see fit() comment): {raw_idle_w:.2f} W")
    print(f"free 3-param fit: P_idle={p_idle:.2f} (+/-{perr[0]:.2f})  "
          f"P_max={p_max:.2f} (+/-{perr[1]:.2f})  alpha={alpha:.3f} (+/-{perr[2]:.3f})  "
          f"R2={r2:.4f}")

    curve = {
        "hardware": f"imec bare-metal {args.node}, 6-core, RAPL package",
        "method": (f"emulated {args.edge_cores}-core edge class: busy workers pinned "
                  f"to {args.edge_cores} of 6 physical cores, u = k/{args.edge_cores}, "
                  f"real RAPL package energy_uj over {args.passes} passes x {args.window}s/level"),
        "form": "P_total_W = P_idle + (P_max-P_idle)*u**alpha  (u relative to the emulated class's own capacity)",
        "measured_constants": {"P_IDLE": round(p_idle, 1), "P_MAX": round(p_max, 1),
                              "ALPHA": round(alpha, 1)},
        "free_fit": {"P_idle": round(p_idle, 2), "P_max": round(p_max, 2),
                    "alpha": round(alpha, 3),
                    "P_idle_stderr": round(float(perr[0]), 3),
                    "P_max_stderr": round(float(perr[1]), 3),
                    "alpha_stderr": round(float(perr[2]), 3),
                    "R2": round(r2, 4)},
        "raw_k0_median_W": round(raw_idle_w, 2),
        "prior_guessed_defaults": {"P_idle": 20.0, "P_max": 60.0, "alpha": 1.4},
        "n_samples": len(rows),
    }
    with open(out_json, "w") as f:
        json.dump(curve, f, indent=2)
    print(f"\nwrote {out_csv}\nwrote {out_json}")

    plot_curve(U, P, p_idle, p_max, alpha, r2, args.node, args.edge_cores,
              os.path.join(FIG_DIR, f"edge_node_power_curve_{args.node}.png"))


def plot_curve(U, P, p_idle, p_max, alpha, r2, node, edge_cores, path, prior=None):
    fig, ax = new_fig(SINGLE, 2.5)
    grid = np.linspace(1e-3, 1.0, 200)
    if prior:
        ax.plot(grid, model(grid, *prior), color=INK_3, lw=1.1, ls="--",
                label="earlier guessed curve")
    ax.scatter(U, P, s=12, color=COLOR["measured"], alpha=0.75,
               edgecolor="white", linewidth=0.3, label="measured (RAPL)")
    ax.plot(grid, model(grid, p_idle, p_max, alpha), color=COLOR["fit"],
            label="fitted model")
    note(ax, f"$P_{{idle}}$={p_idle:.1f} W, $P_{{max}}$={p_max:.1f} W\n"
             f"$\\alpha$={alpha:.2f}, $R^2$={r2:.3f}", loc="lower right")
    ax.set_xlabel(f"utilization u ({edge_cores}-core edge class)")
    ax.set_ylabel("package power (W)")
    ax.set_title(f"Edge node power curve ({node})")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    style_axes(ax, grid="both")
    savefig(fig, path)


if __name__ == "__main__":
    main()
