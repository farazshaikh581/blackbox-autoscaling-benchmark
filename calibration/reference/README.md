# Reference measurements

Experiment logs from the NOMS 2026 runs on the real MicroK8s cluster, copied
from the reference implementation
(https://github.com/farazshaikh581/pareto-optimal_autoscaling, Apache-2.0).

Each row is one control interval. Columns used for calibration:

- `latency`: measured request latency (seconds)
- `replicas`: pod count
- `num_requests`: invocations that interval (workload signal)
- `total_cpu_m`: total CPU across pods (millicores)
- `pod_power`: deployment power (W), from the paper's power model

The factorizator ran at a 0.5-core CPU limit and 0.25 GiB memory limit under an
HPA targeting 50% CPU utilization. `fit_from_noms.py` reads these files to set
the simulator's base service time and node capacity.
