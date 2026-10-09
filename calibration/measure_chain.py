"""Validate the multi-tier open-chain topology on the real cluster.

Assumes the chain is deployed (calibration/cluster/deploy_chain.sh:
frontend -> logic -> backend, each running app.py with DOWNSTREAM_URL set).

Two experiments:
  T  topology : at rest (concurrency 1), probe each tier and the whole chain, and
                check that the request traverses every tier once (the `chain`
                field) and that the frontend's end-to-end time equals the sum of
                the per-tier compute sojourns plus a small fixed network cost --
                the additive open chain the simulator assumes.
  K  knee     : drive rising load at the frontend; the heaviest tier saturates
                first and its queueing dominates end-to-end latency, so the tiers
                are non-interchangeable and the bottleneck sets capacity.

Writes calibration/data/chain_sweep.csv and prints the additivity summary.

    python -m calibration.measure_chain
"""
from __future__ import annotations

import csv
import json
import os
import subprocess

NS = "blackbox"
TIERS = ["frontend", "logic", "backend"]
OUT = os.path.join(os.path.dirname(__file__), "data", "chain_sweep.csv")

# Runs inside the loadgen pod (in-cluster DNS): hit each tier n times and report
# the averaged per-tier compute / downstream / total breakdown as JSON. Kept to
# %-formatting and plain dicts so it is valid on the pod's Python 3.9.
PROBE_SRC = r"""
import urllib.request, json, time
def hit(url, n=7):
    urllib.request.urlopen(url, timeout=60).read()  # warm
    rows = []
    for _ in range(n):
        t0 = time.perf_counter()
        b = json.loads(urllib.request.urlopen(url, timeout=60).read())
        b["_e2e_ms"] = (time.perf_counter() - t0) * 1000.0
        rows.append(b)
    avg = lambda k: sum(r.get(k, 0) for r in rows) / len(rows)
    return {"compute_ms": avg("compute_ms"), "downstream_ms": avg("downstream_ms"),
            "total_ms": avg("total_ms"), "e2e_ms": avg("_e2e_ms"),
            "chain": rows[0].get("chain")}
base = "http://%s." + "NAMESPACE" + ".svc:8080/work"
out = {t: hit(base % t) for t in ["backend", "logic", "frontend"]}
print(json.dumps(out))
"""


def kubectl(*args, timeout=60, check=True):
    r = subprocess.run(["kubectl", "-n", NS, *args], capture_output=True,
                       text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def loadgen_pod() -> str:
    return kubectl("get", "pod", "-l", "app=loadgen", "-o",
                   "jsonpath={.items[0].metadata.name}")


def probe_chain() -> dict:
    src = PROBE_SRC.replace("NAMESPACE", NS)
    out = kubectl("exec", loadgen_pod(), "--", "python3", "-c", src, timeout=120)
    return json.loads(out.splitlines()[-1])


def run_load(tier: str, concurrency: int, duration: float) -> dict:
    url = f"http://{tier}.{NS}.svc:8080/work"
    out = kubectl("exec", loadgen_pod(), "--", "python", "/load/loadgen.py",
                  "--url", url, "--concurrency", str(concurrency),
                  "--duration", str(duration), timeout=int(duration) + 60)
    return json.loads(out.splitlines()[-1])


def top_by_tier() -> dict:
    """Per-tier CPU (millicores) from metrics-server -- the energy-model input."""
    out = kubectl("top", "pod", "-l", "chain=blackbox", "--no-headers",
                  timeout=20, check=False)
    cpu = {t: 0.0 for t in TIERS}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            for t in TIERS:
                if parts[0].startswith(t):
                    cpu[t] += float(parts[1].rstrip("m"))
    return cpu


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    # --- Experiment T: topology / call graph / additivity --------------------
    print("Experiment T: topology (concurrency 1, at rest)")
    p = probe_chain()
    per_tier = {t: p[t]["compute_ms"] for t in TIERS}
    chain = p["frontend"]["chain"]
    e2e_total = p["frontend"]["total_ms"]
    sum_compute = sum(per_tier.values())
    net = e2e_total - sum_compute
    for t in TIERS:
        print(f"  {t:9s} compute={per_tier[t]:8.2f} ms")
    print(f"  observed call graph: {chain}")
    print(f"  sum(per-tier compute) = {sum_compute:8.2f} ms")
    print(f"  frontend end-to-end   = {e2e_total:8.2f} ms")
    print(f"  network overhead      = {net:8.2f} ms "
          f"({net / max(len(TIERS) - 1, 1):.1f} ms/hop)  "
          f"-> additive to within {abs(net) / e2e_total * 100:.1f}%")

    # --- Experiment K: knee / tier interaction -------------------------------
    print("\nExperiment K: knee (rising load at the frontend)")
    rows = []
    for conc in [1, 2, 4, 8]:
        lg = run_load("frontend", conc, 15.0)
        cpu = top_by_tier()
        row = dict(exp="K_knee", concurrency=conc,
                   completed=lg["completed"], errors=lg["errors"],
                   throughput_rps=round(lg["throughput_rps"], 2),
                   lat_p50=round(lg["lat_ms_p50"], 1),
                   lat_p90=round(lg["lat_ms_p90"], 1),
                   lat_p99=round(lg["lat_ms_p99"], 1),
                   cpu_frontend_m=round(cpu["frontend"]),
                   cpu_logic_m=round(cpu["logic"]),
                   cpu_backend_m=round(cpu["backend"]))
        rows.append(row)
        print(f"  conc={conc}  p50={row['lat_p50']}ms  tput={row['throughput_rps']}rps  "
              f"cpu fe/logic/be = {row['cpu_frontend_m']}/{row['cpu_logic_m']}/"
              f"{row['cpu_backend_m']}m")

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Topology summary row is captured separately in the printout; persist the
    # additive check alongside the knee sweep for the fit step.
    with open(OUT.replace(".csv", "_topology.json"), "w") as f:
        json.dump({"per_tier_compute_ms": per_tier, "chain": chain,
                   "e2e_total_ms": e2e_total, "sum_compute_ms": sum_compute,
                   "network_ms": net}, f, indent=2)
    print(f"\nsaved -> {OUT} (+ _topology.json)")


if __name__ == "__main__":
    main()
