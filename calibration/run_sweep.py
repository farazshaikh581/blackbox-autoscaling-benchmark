"""Drive calibration sweeps against the `blackbox` namespace on the cluster.

Three experiments:
  A  cpu scaling  : concurrency 1, vary CPU limit -> base service time s(cpu)
  B  knee         : fixed config, vary load -> latency blow-up and capacity
  C  memory       : vary memory limit at a fixed footprint -> feasibility threshold

For each point it patches the target deployment, waits for the new pod, runs the
in-cluster loadgen, and records latency, throughput, and `kubectl top`. Results
go to calibration/data/cluster_sweep.csv.

    python -m calibration.run_sweep
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import time

NS = "blackbox"
URL = "http://target.blackbox.svc:8080/work"
OUT = os.path.join(os.path.dirname(__file__), "data", "cluster_sweep.csv")


def kubectl(*args, timeout=60, check=True):
    r = subprocess.run(["kubectl", "-n", NS, *args], capture_output=True,
                       text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def set_target(cpu: float, mem_mi: int, replicas: int):
    cpu_m = f"{int(cpu * 1000)}m"
    mem = f"{mem_mi}Mi"
    patch = {"spec": {"template": {"spec": {"containers": [{
        "name": "target",
        "resources": {"limits": {"cpu": cpu_m, "memory": mem},
                      "requests": {"cpu": cpu_m, "memory": mem}}}]}}}}
    kubectl("patch", "deploy/target", "-p", json.dumps(patch))
    kubectl("scale", "deploy/target", f"--replicas={replicas}")


def wait_ready(replicas: int, timeout=50) -> str:
    """Poll until the target has `replicas` ready pods, or a crash/OOM shows."""
    stop = time.time() + timeout
    while time.time() < stop:
        pods = json.loads(kubectl("get", "pods", "-l", "app=calib-target",
                                  "-o", "json"))["items"]
        current = [p for p in pods if p["metadata"].get("deletionTimestamp") is None]
        ready = sum(1 for p in current
                    if all(c.get("ready") for c in
                           p.get("status", {}).get("containerStatuses", [])))
        for p in current:
            for c in p.get("status", {}).get("containerStatuses", []):
                w = c.get("state", {}).get("waiting", {})
                last = c.get("lastState", {}).get("terminated", {})
                if last.get("reason") == "OOMKilled" or w.get("reason") in (
                        "CrashLoopBackOff",):
                    return "OOMKilled" if last.get("reason") == "OOMKilled" else "CrashLoop"
        if ready >= replicas:
            return "Ready"
        time.sleep(2)
    return "Timeout"


def loadgen_pod() -> str:
    return kubectl("get", "pod", "-l", "app=loadgen", "-o",
                   "jsonpath={.items[0].metadata.name}")


def run_load(units: int, concurrency: int, duration: float) -> dict:
    pod = loadgen_pod()
    out = kubectl("exec", pod, "--", "python", "/load/loadgen.py",
                  "--url", URL, "--units", str(units),
                  "--concurrency", str(concurrency), "--duration", str(duration),
                  timeout=int(duration) + 40)
    return json.loads(out.splitlines()[-1])


def read_top() -> tuple:
    try:
        out = kubectl("top", "pod", "-l", "app=calib-target", "--no-headers",
                      timeout=20, check=False)
        cpu_m = mem_mi = 0.0
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 3:
                cpu_m += float(parts[1].rstrip("m"))
                mem_mi += float(parts[2].rstrip("Mi"))
        return cpu_m, mem_mi
    except Exception:
        return -1.0, -1.0


def measure(exp, cpu, mem_mi, replicas, concurrency, units, duration, writer, state):
    if state.get("cfg") != (cpu, mem_mi, replicas):
        set_target(cpu, mem_mi, replicas)
        status = wait_ready(replicas)
        state["cfg"] = (cpu, mem_mi, replicas)
    else:
        status = "Ready"

    row = dict(exp=exp, cpu=cpu, mem_mi=mem_mi, replicas=replicas,
               concurrency=concurrency, units=units, status=status)
    if status == "Ready":
        time.sleep(2)
        lg = run_load(units, concurrency, duration)
        top_cpu, top_mem = read_top()
        row.update(completed=lg["completed"], errors=lg["errors"],
                   throughput_rps=round(lg["throughput_rps"], 2),
                   lat_p50=round(lg["lat_ms_p50"], 1), lat_p90=round(lg["lat_ms_p90"], 1),
                   lat_p99=round(lg["lat_ms_p99"], 1), lat_mean=round(lg["lat_ms_mean"], 1),
                   top_cpu_m=round(top_cpu), top_mem_mi=round(top_mem))
    else:
        row.update(completed=0, errors=-1, throughput_rps=0, lat_p50=-1,
                   lat_p90=-1, lat_p99=-1, lat_mean=-1, top_cpu_m=-1, top_mem_mi=-1)
    writer.writerow(row)
    print(f"  {exp}  cpu={cpu} mem={mem_mi}Mi n={replicas} conc={concurrency} "
          f"u={units} -> {status} p50={row['lat_p50']}ms tput={row['throughput_rps']}rps")
    return row


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fields = ["exp", "cpu", "mem_mi", "replicas", "concurrency", "units", "status",
              "completed", "errors", "throughput_rps", "lat_p50", "lat_p90",
              "lat_p99", "lat_mean", "top_cpu_m", "top_mem_mi"]
    state: dict = {}
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        # units=10 (~150 ms at 1 core) spans several CFS periods, so throttling
        # scales latency cleanly rather than quantizing it.
        print("Experiment A: cpu scaling (concurrency 1, no queue)")
        for cpu in [0.25, 0.5, 0.75, 1.0]:
            measure("A_cpu", cpu, 256, 1, 1, 10, 15, w, state)

        print("Experiment B: knee (cpu 0.5, vary load)")
        for conc in [1, 2, 4, 8]:
            measure("B_knee", 0.5, 256, 1, conc, 10, 15, w, state)

        print("Experiment C: memory feasibility (64 MB footprint)")
        for mem in [48, 96, 128, 256]:
            measure("C_mem", 0.5, mem, 1, 1, 10, 12, w, state)

    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
