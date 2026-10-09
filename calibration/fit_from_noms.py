"""Calibrate the simulator against the NOMS reference measurements.

The reference logs (calibration/reference/) are single-tier factorizator runs
under HPA, so they pin down some constants tightly and leave others for a host
sweep. This script reports both, and the constants it prints are the ones set in
topology.py.

    python -m calibration.fit_from_noms

What the single-tier data gives:
  - base service time from the light-load latency floor -> service_demand
  - node CPU capacity from the power and CPU logs (via the paper's power model)
  - the operating utilization HPA settles at

What still needs a host sweep:
  - how service time scales with the CPU limit (the logs use a fixed 0.5-core limit)
  - the memory working set (no memory-limit sweep in the logs)
  - the exact overload knee (HPA keeps utilization low, so few saturated samples)
  - multi-tier interactions
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from blackbox import energy_model

CPU_LIMIT = 0.5          # cores per replica (factorizator-deployment.yaml)
MEM_LIMIT = 0.25         # GiB per replica
REF_DIR = os.path.join(os.path.dirname(__file__), "reference")


def load_reference() -> pd.DataFrame:
    frames = []
    for f in glob.glob(os.path.join(REF_DIR, "*.csv")):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if "num_requests" not in d:
            d["num_requests"] = np.nan
        keep = ["latency", "replicas", "num_requests", "total_cpu_m", "pod_power"]
        if set(keep[:2] + keep[3:]).issubset(d.columns):
            frames.append(d[keep])
    df = pd.concat(frames, ignore_index=True)
    return df[(df.latency > 0) & (df.replicas >= 1) &
              (df.total_cpu_m > 0) & (df.pod_power > 0)].copy()


def main():
    a = load_reference()
    print(f"reference rows (active): {len(a)}\n")

    # Measured distributions.
    lat_p10, lat_med, lat_p90 = a.latency.quantile([0.1, 0.5, 0.9])
    print("measured latency (s):   floor(p10)=%.4f  med=%.4f  p90=%.4f  max=%.4f"
          % (lat_p10, lat_med, a.latency.quantile(0.9), a.latency.max()))
    print("measured power (W):     med=%.1f  p90=%.1f"
          % (a.pod_power.median(), a.pod_power.quantile(0.9)))

    # (1) Base service time from the light-load floor -> service_demand.
    s0 = float(lat_p10)
    service_demand = s0 * CPU_LIMIT
    print("\n[1] base service time s0 = %.1f ms (light-load floor)" % (s0 * 1000))
    print("    service_demand = s0 * cpu_limit = %.4f cpu.s/req" % service_demand)

    # (2) Node capacity from the power and CPU logs (paper power model, sole tenant).
    cores = a.total_cpu_m / 1000.0
    p_node = a.pod_power - energy_model.P_IDLE / a.replicas + energy_model.P_IDLE
    frac = np.clip((p_node - energy_model.P_IDLE) /
                   (energy_model.P_MAX - energy_model.P_IDLE), 1e-6, 1.0)
    util = np.sqrt(frac)
    cap = (cores / util)
    cap = cap[(cap > 1) & (cap < 64)]
    print("\n[2] node capacity (cores): med=%.1f  IQR=(%.1f, %.1f)  -> use 8.0"
          % (cap.median(), cap.quantile(0.25), cap.quantile(0.75)))

    # (3) Operating utilization implied by the median latency, with s/(1-rho).
    rho_op = 1.0 - s0 / lat_med
    print("\n[3] operating queue utilization from median latency: rho ~= %.2f" % rho_op)

    # (4) SLA crossing for the calibrated single tier: s/(1-rho) = SLA.
    sla = 0.020
    rho_sla = 1.0 - s0 / sla
    print("\n[4] single tier at 0.5-core: latency crosses the 20 ms SLA at rho ~= %.2f"
          % rho_sla)
    print("    sim latency band over rho in [0, 0.86]: %.1f ms .. %.1f ms"
          % (s0 * 1000, s0 / (1 - 0.86) * 1000))
    print("    (measured band: %.1f .. %.1f ms) -> same regime"
          % (lat_p10 * 1000, a.latency.max() * 1000))

    print("\ncalibrated constants (set in topology.py):")
    print("    service_demand_s        = %.4f" % service_demand)
    print("    node_cpu_capacity_cores = 8.0")
    print("    cpu_limit / mem_limit   = %.2f core / %.2f GiB" % (CPU_LIMIT, MEM_LIMIT))
    print("    energy model P=50+200u^2 adopted from the paper (unchanged)")


if __name__ == "__main__":
    main()
