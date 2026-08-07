# Black-Box Multi-Objective Autoscaling Benchmark

[![CI](https://github.com/farazshaikh581/blackbox-autoscaling-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/farazshaikh581/blackbox-autoscaling-benchmark/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Optimizer: pymoo](https://img.shields.io/badge/optimizer-pymoo-brightgreen)](https://pymoo.org/)

A benchmark for multi-objective optimization of Kubernetes autoscaling
configurations. It casts autoscaling as a static configuration search over a
mixed integer and continuous decision space, and compares a reinforcement
learning policy against evolutionary algorithms (NSGA-II and MOEA/D) under a
fixed evaluation budget.

The problem and its energy model follow my NOMS 2026 paper, *Pareto-Optimal
Autoscaling: A Multi-Objective Reinforcement Learning Framework for the
Performance-Cost-Energy Trilemma*. Its reference implementation is at
[pareto-optimal_autoscaling](https://github.com/farazshaikh581/pareto-optimal_autoscaling).
The finished benchmark is intended for the ROAR-NET benchmark library.

## Problem

A candidate is a static configuration of an open microservice chain. Each tier
carries three decision variables:

| variable | type    | meaning           | bounds                       |
|----------|---------|-------------------|------------------------------|
| `n_i`    | integer | replicas          | `[replica_min, replica_max]` |
| `c_i`    | real    | CPU limit (cores) | `[cpu_min, cpu_max]`         |
| `m_i`    | real    | memory limit (GiB)| `[mem_min, mem_max]`         |

Three objectives are minimized, scored over the Azure Functions 2021 workload:

- `latency_ms`: mean end-to-end latency
- `cost`: provisioned resource cost, fixed by the configuration
- `energy_W`: mean deployment power, using the paper's node power model
  `P = 50 + 200 u^2` with per-pod attribution

The oracle is a fast offline simulator, averaged over `K` stochastic workload
replications. It is deterministic for a fixed seed, which keeps the anytime
hypervolume, GD+/IGD+, and Wilcoxon protocol reproducible. Real cluster runs
with `hey` calibrate and validate the simulator and confirm the final fronts.
Running the search itself on a live cluster is not practical at thousands of
evaluations, so the live testbed is left as a follow-up.

## Layout

```
blackbox/                package: physics and oracle
  topology.py            microservice chain and per-tier bounds (calibration targets)
  workload.py            Azure trace or synthetic diurnal, with stochastic replication
  simulator.py           (config, workload) -> (latency, cost, energy)
  energy_model.py        node power model and pod attribution
  oracle.py              pymoo problems: mixed-variable and real-relaxed encodings
experiments/
  run_nsga2.py           NSGA-II on the mixed-variable encoding
  run_moead.py           MOEA/D on the real-relaxed encoding
  cov_replications.py    pick K from the coefficient of variation
  timing_benchmark.py    wall-clock timing to a maximum evaluation budget
calibration/
  calibrate.py           hey and cluster hooks to fit constants and check sim vs real
tests/test_oracle.py
SPEC.md                  ROAR-NET problem-statement specification
docs/comparison_protocol.md  algorithm comparison protocol and open items
```

## Quickstart

```bash
pip install -r requirements.txt
pytest tests/ -q

# a small NSGA-II run on the default 3-tier chain
python -m experiments.run_nsga2 --pop 20 --evals 500 --seed 1 --k 5

# MOEA/D under the same budget
python -m experiments.run_moead --partitions 12 --evals 500 --seed 1 --k 5

# preparatory analyses
python -m experiments.cov_replications --configs 20 --kmax 40 --target 0.05
python -m experiments.timing_benchmark --k 10 --window-hours 6 --runs 10 --algos 3
```

## Status

The oracle, both algorithms, and the two preparatory analyses run end to end on
documented placeholder constants (marked `CALIBRATE` in `topology.py`). The
multi-tier open-chain topology is confirmed on a real k3s cluster (`SPEC.md`).
Open tasks are tracked in the issues: fit the constants to a real cluster, add
the RL baseline on the same oracle, and build the hypervolume, GD+/IGD+, and
Wilcoxon aggregation. See `SPEC.md` for the problem definition and
`docs/comparison_protocol.md` for the study protocol.
