"""Calibrate and validate the offline simulator against a real cluster.

This is the ONLY place `hey` / the live cluster is used: to fit the simulator's
physical constants (per-tier service_demand, node CPU capacity, energy params)
and to report sim-vs-real error. Once calibrated, all search runs use the fast
offline oracle. Later, only the final non-dominated configs are re-measured here
for confirmation (feasible: tens of configs, not tens of thousands).

Two modes:
  measure   drive a grid of configs on the cluster, log measured objectives -> CSV
  fit       fit sim constants to a measurement CSV and report MAE / MAPE

Requires the host cluster (MicroK8s + Prometheus) and `hey`. The measurement
plumbing mirrors the NOMS reference (`run_simulation.py`): kubectl to apply
resources, `hey` to load, `kubectl top` for CPU/RAM.
"""
from __future__ import annotations

import argparse
import subprocess

import numpy as np
import pandas as pd

# ------------------------------ measurement (live cluster) ------------------

KUBECTL = "microk8s kubectl"


def apply_config(namespace: str, deployment: str, replicas: int,
                 cpu_cores: float, mem_gib: float) -> None:
    """Apply a static per-tier config to the cluster (scale + resource limits)."""
    cpu_m = int(cpu_cores * 1000)
    mem_mi = int(mem_gib * 1024)
    patch = (
        '{"spec":{"template":{"spec":{"containers":[{"name":"%s",'
        '"resources":{"limits":{"cpu":"%dm","memory":"%dMi"},'
        '"requests":{"cpu":"%dm","memory":"%dMi"}}}]}}}}'
        % (deployment, cpu_m, mem_mi, cpu_m, mem_mi)
    )
    subprocess.run(f"{KUBECTL} -n {namespace} patch deployment {deployment} "
                   f"--patch '{patch}'", shell=True, check=False)
    subprocess.run(f"{KUBECTL} -n {namespace} scale deployment {deployment} "
                   f"--replicas={replicas}", shell=True, check=False)
    # TODO: wait for rollout / in-place resize to settle before load.


def measure_once(url: str, num_requests: int, duration_s: int = 60) -> dict:
    """Drive load with `hey` and parse measured latency + success ratio.

    Parsing mirrors the NOMS reference; extend to read Prometheus for CPU/power.
    """
    import re
    concurrency = min(num_requests, 50)
    cmd = f"hey -n {num_requests} -c {concurrency} -z {duration_s}s -m GET '{url}'"
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
    lat = None
    m = re.search(r"Average:\s+(\d+\.\d+)\s*(ms|secs)?", out)
    if m:
        v = float(m.group(1))
        lat = v if m.group(2) == "secs" else v / 1000.0
    succ = re.search(r"\[2..]\s+(\d+)\s+responses", out)
    success = int(succ.group(1)) / num_requests if succ else 0.0
    return {"latency_s": lat, "success_ratio": success}


# ------------------------------ fitting (offline) ---------------------------

def fit_service_demand(df: pd.DataFrame) -> dict:
    """Least-squares fit of per-tier service_demand from measured (config,latency).

    Model: measured tier latency ~ 1 / (n * c / D - lambda). Given measured
    (replicas n, cpu c, offered rps lambda, latency L) rows, solve for D per tier.
    Placeholder: implement the per-tier regression once measurement CSV exists.
    """
    raise NotImplementedError(
        "Fit against a measurement CSV collected with `--mode measure`. "
        "Columns expected: tier, replicas, cpu, mem, rps, latency_s."
    )


def report_mae(sim: np.ndarray, real: np.ndarray) -> None:
    mae = np.mean(np.abs(sim - real))
    mape = np.mean(np.abs(sim - real) / np.clip(np.abs(real), 1e-9, None)) * 100
    print(f"sim-vs-real  MAE={mae:.4f}  MAPE={mape:.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["measure", "fit"], required=True)
    ap.add_argument("--csv", default="calibration/data/measurements.csv")
    args = ap.parse_args()
    if args.mode == "measure":
        print("Live-cluster measurement requires host access (MicroK8s + hey).")
        print("Fill in the config grid and call apply_config()/measure_once().")
    else:
        df = pd.read_csv(args.csv)
        print(fit_service_demand(df))


if __name__ == "__main__":
    main()
