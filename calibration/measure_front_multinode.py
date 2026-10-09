"""Deploy the REAL optimized front on the dedicated 6-node bare-metal cluster
(r0-r5) and confirm both the latency AND energy predictions
against real hardware.

This is a dedicated, unshared 36-core cluster (6 cores x 6 real bare-metal nodes with Intel RAPL),
so the entire front fits. This deploys the REAL per-tier
(replicas, cpu, mem) configs from `experiments/dump_front.py`, not a
substitute. `deploy_chain.sh REQUEST_EQUALS_LIMIT=1` makes pod cpu/mem
*requests* equal their limits, so the scheduler spreads replicas across nodes
by their real footprint (matching what `energy_model.pack_pods` assumes when
scoring a config) instead of silently piling everything onto one node.

Because these are real bare-metal nodes (unlike shared KVM guests), we
can also sample real RAPL package power during each config's load window and
compare it to the simulator's predicted energy_W -- a shared VM cluster could never
check this axis.

    python -m calibration.measure_front_multinode                # dry-run: plan
    python -m calibration.measure_front_multinode --deploy        # deploy + measure

Writes calibration/data/front_multinode.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import time

import numpy as np
from scipy.stats import rankdata

from viz import DOUBLE, new_fig, rank_scatter, savefig

TIERS = ["frontend", "logic", "backend"]
ENV = {"frontend": "FE", "logic": "LOGIC", "backend": "BE"}
NS = "blackbox"
NODE_ALIASES = ["r0", "r1", "r2", "r3", "r4", "r5"]
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
DEPLOY = os.path.join(os.path.dirname(__file__), "cluster", "deploy_chain.sh")
OUT = os.path.join(os.path.dirname(__file__), "data", "front_multinode.csv")
CPU_BUDGET = 32.0      # of 36 real cores; leaves headroom for loadgen + k3s system
MEM_BUDGET = 100.0     # GiB; generous, not expected to bind


def kubectl(*a, timeout=60, check=True):
    r = subprocess.run(["kubectl", "-n", NS, *a], capture_output=True, text=True,
                       timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(a)}: {r.stderr.strip()}")
    return r.stdout.strip()


def feasible(cfg):
    cpu = sum(r * c for r, c in zip(cfg["replicas"], cfg["cpu"]))
    mem = sum(r * m for r, m in zip(cfg["replicas"], cfg["mem"]))
    return cpu <= CPU_BUDGET and mem <= MEM_BUDGET


def deploy(cfg, max_attempts=3, retry_delay=20):
    env = dict(os.environ)
    env["QUOTA_CPU"] = str(CPU_BUDGET)
    env["QUOTA_MEM"] = "150Gi"
    env["REQUEST_EQUALS_LIMIT"] = "1"
    for i, t in enumerate(TIERS):
        p = ENV[t]
        env[f"{p}_REPLICAS"] = str(cfg["replicas"][i])
        env[f"{p}_CPU"] = str(cfg["cpu"][i])
        env[f"{p}_MEM"] = f"{int(round(cfg['mem'][i] * 1024))}Mi"
    last_err = None
    for attempt in range(1, max_attempts + 1):
        r = subprocess.run(["bash", DEPLOY], env=env, capture_output=True, text=True,
                           timeout=240)
        if r.returncode == 0:
            return
        last_err = r.stderr.strip()
        if attempt < max_attempts:
            # A rollout failure here has been observed to be a transient
            # scheduling race (leftover Terminating pods from the prior
            # config still holding node capacity), not a hard per-node
            # capacity wall -- an identical redeploy on an otherwise-idle
            # cluster has succeeded on retry. Give it a settle window and
            # try again before concluding it truly doesn't fit anywhere.
            print(f"    deploy attempt {attempt}/{max_attempts} failed, "
                  f"retrying in {retry_delay}s: {last_err[-200:]}")
            time.sleep(retry_delay)
    raise RuntimeError(f"deploy failed after {max_attempts} attempts: {last_err[-400:]}")


def measure_latency(units_fe, concurrency, duration):
    pod = kubectl("get", "pod", "-l", "app=loadgen", "-o",
                  "jsonpath={.items[0].metadata.name}")
    url = f"http://frontend.{NS}.svc:8080/work"
    out = kubectl("exec", pod, "--", "python", "/load/loadgen.py", "--url", url,
                  "--units", str(units_fe), "--concurrency", str(concurrency),
                  "--duration", str(duration), timeout=int(duration) + 90)
    return json.loads(out.splitlines()[-1])


def read_power_async(alias, win):
    """Launch a non-blocking RAPL sample (energy_uj delta / win) on `alias`."""
    cmd = ('F=/sys/class/powercap/intel-rapl:0/energy_uj; R="cat"; [ -r "$F" ] || R="sudo -n cat"; '
           'a=$($R "$F"); sleep %d; b=$($R "$F"); '
           'awk "BEGIN{printf \\"%%.2f\\", ($b-$a)/1e6/%d}"') % (win, win)
    # The orchestrating node cannot ssh to itself, so read its counter locally.
    argv = (["bash", "-c", cmd] if socket.gethostname().startswith(alias + ".")
            else ["ssh", "-o", "BatchMode=yes", alias, cmd])
    return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def check_power_readable():
    """Abort early if any node's RAPL counter cannot be read."""
    procs = {a: read_power_async(a, 1) for a in NODE_ALIASES}
    bad = []
    for a, p in procs.items():
        out, err = p.communicate(timeout=30)
        try:
            float(out.strip())
        except ValueError:
            bad.append(f"{a}: {err.strip()[-120:]}")
    if bad:
        raise SystemExit("RAPL not readable on: " + "; ".join(bad))


def node_placement():
    """{tier: {node_alias: pod_count}}, from real scheduling (evidence of spread)."""
    out = kubectl("get", "pod", "-l", "chain=blackbox", "-o",
                  "jsonpath={range .items[*]}{.metadata.labels.app}{\" \"}"
                  "{.spec.nodeName}{\"\\n\"}{end}")
    placement = {t: {} for t in TIERS}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        tier, node = parts
        if tier not in placement:
            continue
        alias = next((a for a in NODE_ALIASES if node.startswith(a + ".")), node)
        placement[tier][alias] = placement[tier].get(alias, 0) + 1
    return placement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", action="store_true",
                    help="actually deploy + measure (default: dry-run plan only)")
    ap.add_argument("--front", default="results/front_configs.json",
                    help="experiments/dump_front.py output to deploy from")
    ap.add_argument("--units-fe", type=int, default=15)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--duration", type=float, default=20.0,
                    help="loadgen duration (s); RAPL samples over the same window")
    args = ap.parse_args()

    if not os.path.exists(args.front):
        raise SystemExit(f"{args.front} not found -- run "
                         f"`python -m experiments.dump_front --out {args.front}` first")
    with open(args.front) as f:
        data = json.load(f)
    front = data["front"]
    plan = [c for c in front if feasible(c)]
    skipped = len(front) - len(plan)

    print(f"front has {len(front)} configs; {len(plan)} fit the {CPU_BUDGET}-core "
          f"budget ({skipped} skipped, datacenter-scale low-latency tail)")
    for c in plan:
        cpu = sum(r * cc for r, cc in zip(c["replicas"], c["cpu"]))
        print(f"  replicas={c['replicas']}  cpu={c['cpu']}  ({cpu:.1f} cores)  "
              f"sim latency={c['sim_objectives']['latency_ms']:8.1f} ms  "
              f"sim energy={c['sim_objectives']['energy_W']:7.1f} W")

    if not args.deploy:
        print(f"\n[dry-run] pass --deploy to deploy each on {'/'.join(NODE_ALIASES)} and measure.")
        return

    check_power_readable()
    rows = []
    for i, cfg in enumerate(plan, 1):
        print(f"\n[{i}/{len(plan)}] deploy replicas={cfg['replicas']} "
              f"cpu={cfg['cpu']} ...", flush=True)
        try:
            deploy(cfg)
        except RuntimeError as e:
            # Fits the namespace ResourceQuota (feasible() above) but the
            # scheduler's default spread-not-pack scoring can still fragment
            # per-node headroom so no single node has room for the last
            # replica of a tight config, even though total free cpu across
            # nodes exceeds it. Record as a failed deploy (same -1.0 sentinel
            # as a zero-success load test) and move on rather than aborting
            # the rest of the front.
            print(f"    DEPLOY FAILED: {str(e)[-200:]}")
            rows.append({
                "replicas": "|".join(str(r) for r in cfg["replicas"]),
                "cpu": "|".join(str(c) for c in cfg["cpu"]),
                "sim_latency_ms": cfg["sim_objectives"]["latency_ms"],
                "meas_p50_ms": -1.0, "meas_p90_ms": -1.0,
                "throughput_rps": 0.0, "errors": "deploy_failed",
                "sim_energy_W": cfg["sim_objectives"]["energy_W"],
                "meas_energy_W": float("nan"),
                "nodes_used": "|".join(sorted({a for t in node_placement().values() for a in t})),
                **{f"watts_{a}": None for a in NODE_ALIASES},
            })
            continue
        time.sleep(3)  # let readiness settle past the rollout-status race
        power_procs = {a: read_power_async(a, int(args.duration)) for a in NODE_ALIASES}
        lat = measure_latency(args.units_fe, args.concurrency, args.duration)
        watts = {}
        for a, p in power_procs.items():
            out, err = p.communicate(timeout=30)
            try:
                watts[a] = float(out.strip())
            except ValueError:
                watts[a] = float("nan")
        total_w = sum(w for w in watts.values() if w == w)
        placement = node_placement()
        nodes_used = sorted({a for t in placement.values() for a in t})
        row = {
            "replicas": "|".join(str(r) for r in cfg["replicas"]),
            "cpu": "|".join(str(c) for c in cfg["cpu"]),
            "sim_latency_ms": cfg["sim_objectives"]["latency_ms"],
            "meas_p50_ms": round(lat["lat_ms_p50"], 1),
            "meas_p90_ms": round(lat["lat_ms_p90"], 1),
            "throughput_rps": round(lat["throughput_rps"], 2),
            "errors": lat["errors"],
            "sim_energy_W": cfg["sim_objectives"]["energy_W"],
            "meas_energy_W": round(total_w, 2),
            "nodes_used": "|".join(nodes_used),
            **{f"watts_{a}": watts.get(a) for a in NODE_ALIASES},
        }
        rows.append(row)
        print(f"    sim lat {row['sim_latency_ms']} ms | meas p50 {row['meas_p50_ms']} ms  "
              f"||  sim energy {row['sim_energy_W']} W | meas {row['meas_energy_W']} W  "
              f"||  nodes used: {nodes_used}")

    # meas_p50_ms == -1.0 is loadgen.py's "zero successful requests" sentinel
    # (the config timed out on every request), not a real latency sample --
    # including it in a rank correlation corrupts the result (it sorts as the
    # smallest value, so a handful of total failures look like the fastest
    # configs). Latency Spearman and the energy-vs-served comparison exclude
    # those rows; the all-configs energy Spearman does not, since RAPL still
    # gives a real power reading even when the app served nothing. A config
    # that never deployed (scheduler couldn't fit it -- see deploy()'s
    # except block) has no RAPL sample either, so meas_energy_W is NaN;
    # those rows are excluded from every Spearman, including "all configs".
    served = [r for r in rows if r["meas_p50_ms"] > 0]
    failed = [r for r in rows if r["meas_p50_ms"] < 0]
    energy_rows = [r for r in rows if r["meas_energy_W"] == r["meas_energy_W"]]  # drop NaN (deploy_failed)
    rho_lat = spearman([r["sim_latency_ms"] for r in served],
                       [r["meas_p50_ms"] for r in served]) if len(served) >= 3 else float("nan")
    rho_e_served = spearman([r["sim_energy_W"] for r in served],
                            [r["meas_energy_W"] for r in served]) if len(served) >= 3 else float("nan")
    rho_e_all = spearman([r["sim_energy_W"] for r in energy_rows],
                        [r["meas_energy_W"] for r in energy_rows]) if len(energy_rows) >= 3 else float("nan")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_multi = sum(1 for r in rows if "|" in r["nodes_used"])
    print(f"\n{len(served)}/{len(rows)} configs got genuine served traffic; "
          f"{len(failed)} hit zero successful requests (excluded from latency Spearman)")
    print(f"Spearman(sim latency, measured p50), served only  = {rho_lat:.3f} (n={len(served)})")
    print(f"Spearman(sim energy, measured watts), served only = {rho_e_served:.3f} (n={len(served)})")
    print(f"Spearman(sim energy, measured watts), all configs = {rho_e_all:.3f} (n={len(energy_rows)})")
    print(f"{n_multi}/{len(rows)} configs genuinely spread across >1 physical node")
    print(f"saved -> {OUT}")

    plot_multinode(served, energy_rows, rho_lat, rho_e_all,
                   os.path.join(FIG_DIR, "front_multinode_sim_vs_measured.png"))


def plot_multinode(served, energy_rows, rho_lat, rho_e, path):
    # Rank-based rendering: sim latency (sub-ms) and
    # measured latency (seconds, real queueing) sit on different magnitudes,
    # so raw-value scatter hides the rank agreement Spearman actually reports.
    fig, axes = new_fig(DOUBLE, 2.7, ncols=2)
    rank_scatter(axes[0], [r["sim_latency_ms"] for r in served],
                 [r["meas_p50_ms"] for r in served], rho_lat)
    axes[0].set_xlabel("rank, simulated latency")
    axes[0].set_ylabel("rank, measured p50 latency")
    axes[0].set_title("(a) Latency, configs that served traffic")
    rank_scatter(axes[1], [r["sim_energy_W"] for r in energy_rows],
                 [r["meas_energy_W"] for r in energy_rows], rho_e)
    axes[1].set_xlabel("rank, simulated energy")
    axes[1].set_ylabel("rank, measured RAPL power")
    axes[1].set_title("(b) Energy, all deployed configs")
    savefig(fig, path)


def spearman(a, b):
    # Average ranks for ties (several configs share a predicted energy).
    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return (ra * rb).sum() / denom if denom else float("nan")


if __name__ == "__main__":
    main()
