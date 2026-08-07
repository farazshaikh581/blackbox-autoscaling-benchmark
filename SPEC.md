<!--
SPDX-FileCopyrightText: 2026 Faraz Shaikh

SPDX-License-Identifier: CC-BY-4.0
-->

# Black-Box Multi-Objective Kubernetes Autoscaling

Faraz Shaikh, University of Perugia

Copyright 2026 Faraz Shaikh.

This document is licensed under CC-BY-4.0.

## Introduction

Kubernetes autoscaling has to balance three goals that pull against each
other: how fast requests get served, how much the deployment costs, and
how much energy it draws. Turning up replicas or resource limits can
lower latency, but it raises both cost and energy. There is no single
best setting, only a trade-off between the three.

This problem is a black-box multi-objective optimization problem over a
mixed discrete and continuous decision space. A candidate solution sets
integer replica counts and continuous CPU and memory limits per
microservice tier. Evaluating a candidate means running it through a
simulation of the deployment against a real workload trace, which
returns latency, cost, and energy as noisy values, not exact ones.

The problem and its energy model come from a multi-objective
reinforcement learning framework for Pareto-optimal autoscaling,
published at IEEE/IFIP NOMS 2026. This document formalizes that same
problem as a reusable black-box benchmark instance, independent of any
one solving method.

## Task

Find replica, CPU, and memory settings for every tier that trade off
latency, cost, and energy well, meaning no other setting is at least as
good on all three at once.

## Detailed description

**Topology.** A request enters an open chain of tiers and passes
through each one once, so end-to-end latency is the sum of the
per-tier sojourn times. The tiers are not interchangeable: each has its
own fixed compute demand per request. The default chain has three
tiers (frontend, logic, backend), with demand rising down the chain.

Confirmed on a real k3s cluster: the open-chain call graph, additive
per-tier latency, and single-bottleneck capacity all hold on real
hardware. Every request traverses each tier once, in order, with
demand rising down the chain. End-to-end latency equals the sum of
per-tier sojourns to within about 7%. Throughput plateaus at the
heaviest tier's predicted rate (measured 0.73 rps vs 0.74 rps predicted
from `1 / heaviest-tier service time`), confirming the simulator's
bottleneck-tier model rather than an assumed one.

**Decision variables.** For a chain of `T` tiers, a solution sets three
values per tier:

- `n_i`, an integer replica count in `[replica_min, replica_max]`
- `c_i`, a continuous CPU limit in cores, in `[cpu_min, cpu_max]`
- `m_i`, a continuous memory limit in GiB, in `[mem_min, mem_max]`

A solution is the full vector across all tiers, `3T` values in total.
Two encodings are provided: a mixed-variable one (`AutoscalingProblem`,
used by NSGA-II) and a real-relaxed one with rounding
(`RealEncodedAutoscalingProblem`, used by MOEA/D and for
same-encoding comparisons).

**Feasibility.** The only constraints are the per-variable bounds
above. Every point inside those bounds is feasible. There are no
cross-tier or joint constraints.

**Objectives (all minimized).**

- `latency_ms`: mean end-to-end time over the workload. Each tier
  behaves like a queueing station whose sojourn time rises from a
  light-load floor to a timeout plateau as it nears capacity, the same
  CPU-throttle knee seen on real hardware.
- `cost`: `sum_i n_i (c_i price_cpu + m_i price_mem)`, fixed by the
  solution, no randomness.
- `energy_W`: mean deployment power. Node power follows
  `P = P_idle + (P_max - P_idle) u^alpha`, attributed to pods by CPU
  share with an even split of the idle floor. Currently placeholder
  hardware constants (`P_idle=50, P_max=250, alpha=2.0`), pending a
  real-cluster calibration pass (open item, tracked separately).

**Evaluation and noise.** A solution is scored by a fast analytic
oracle, not by a live deployment, since deploying every candidate a
search touches is neither feasible nor reproducible at the scale this
problem is searched.

Each oracle call realizes the workload with multiplicative Gamma noise
rather than returning a fixed number, so a single call is noisy. The
reported score for a solution is the mean over `K` independent
realizations with seeds `base_seed ... base_seed + K - 1`,
deterministic for a fixed `base_seed`. `K` is chosen by
`experiments/cov_replications.py`, the smallest `K` whose
worst-objective coefficient of variation drops below a target
threshold, rather than fixed arbitrarily. Resolved: on the real Azure
trace, worst-objective CoV is 0.0026 at K=2, well under the 0.05
target. `K* = 2`; the benchmark runs at K=5 throughout, a margin above
this minimum, not a requirement.

The maximum evaluation budget is set by `experiments/timing_benchmark.py`.
Resolved: the current code runs at about 24 ms per oracle call, so a
full 2000-evaluation sweep across all algorithms and seeds costs about
15 minutes on 6 cores.

## Instance data file

An instance is fully described by a chain topology and a workload
source. The two encodings used by this benchmark's own code
(`blackbox/topology.py`, `blackbox/workload.py`) are:

**Topology**, one record per tier plus a few chain-level constants:

```json
{
  "tiers": [
    {
      "name": "frontend",
      "service_demand_s": 0.004,
      "working_set_base_gib": 0.15,
      "working_set_per_rps_gib": 0.0,
      "replica_min": 1, "replica_max": 20,
      "cpu_min": 0.1, "cpu_max": 2.0,
      "mem_min": 0.1, "mem_max": 2.0
    },
    {
      "name": "logic",
      "service_demand_s": 0.01,
      "working_set_base_gib": 0.3,
      "working_set_per_rps_gib": 0.001,
      "replica_min": 1, "replica_max": 20,
      "cpu_min": 0.1, "cpu_max": 2.0,
      "mem_min": 0.1, "mem_max": 2.0
    },
    {
      "name": "backend",
      "service_demand_s": 0.018,
      "working_set_base_gib": 0.5,
      "working_set_per_rps_gib": 0.002,
      "replica_min": 1, "replica_max": 20,
      "cpu_min": 0.1, "cpu_max": 2.0,
      "mem_min": 0.1, "mem_max": 2.0
    }
  ],
  "node_cpu_capacity_cores": 16.0,
  "sla_latency_ms": 20.0,
  "timeout_ms": 5000.0,
  "price_cpu_per_core": 1.0,
  "price_mem_per_gib": 0.25
}
```

**Workload**, either a parametric synthetic profile:

```json
{ "kind": "synthetic_diurnal", "peak_rps": 200.0, "base_rps": 20.0 }
```

or a reference to a segment of the Azure Functions 2021 invocation
trace, binned to per-minute mean requests per second, with the day
index used:

```json
{ "kind": "azure_trace", "trace_path": "...", "day_index": 0 }
```

Today, an instance is set up by constructing these objects directly in
Python, or by passing the equivalent command-line flags to the search
scripts, not by loading a single serialized instance file. The JSON
shapes above describe the fields precisely enough to write a loader.
Building one is an open item for this benchmark's formalization, not
yet done.

## Solution file

A solution is the decision vector described above. A full search run
reports a Pareto front, a list of non-dominated solutions, each paired
with its evaluated objective triple:

```json
{
  "front": [
    {
      "replicas": [2, 4, 6],
      "cpu": [1.0, 1.0, 1.0],
      "mem": [0.3, 0.5, 0.8],
      "objectives": { "latency_ms": 10.98, "cost": 13.85, "energy_W": 55.88 }
    }
  ]
}
```

## Example

### Instance

The default three-tier topology (frontend, logic, backend), with the
synthetic diurnal workload (`peak_rps` 200, `base_rps` 20).

### Solution

Replicas `[2, 4, 6]`, CPU `[1.0, 1.0, 1.0]` cores, memory
`[0.3, 0.5, 0.8]` GiB, one value per tier in frontend, logic, backend
order.

### Explanation

Evaluated with 5 workload replications at a fixed seed, this solution
scores `latency_ms = 10.98`, `cost = 13.85`, `energy_W = 55.88`.
Backend is the heaviest tier (`service_demand_s = 0.018`), so it
carries the most replicas (6) to keep its per-replica load under
capacity; frontend is the lightest and needs only 2. Each tier's
memory limit sits above its working set (`0.15/0.3/0.5` GiB) with
headroom, so no tier is thrown into the timeout plateau, which is why
latency stays low (about 11 ms) instead of spiking toward the 5000 ms
timeout.

## Acknowledgements

This problem statement is based upon work from COST Action Randomised
Optimisation Algorithms Research Network (ROAR-NET), CA22137, is
supported by COST (European Cooperation in Science and Technology).

This work was carried out during a Short-Term Scientific Mission at
Ghent University - imec, under the supervision of Prof. Filip De
Turck, and builds on the applicant's own prior NOMS 2026 paper below.

## References

Shaikh, F. Pareto-Optimal Autoscaling: A Multi-Objective Reinforcement
Learning Framework for the Performance-Cost-Energy Trilemma. IEEE/IFIP
NOMS 2026.

Blank, J. and Deb, K. pymoo: Multi-Objective Optimization in Python.
IEEE Access, 2020.

---

The comparison protocol (algorithms, metrics, budget) and the
remaining open items live in `docs/comparison_protocol.md`, since the
ROAR-NET problem-statement template above covers the instance
definition only, not the benchmark study built on top of it.
