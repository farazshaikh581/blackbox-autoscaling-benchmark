# Benchmark Specification (draft)

Black-box multi-objective Kubernetes autoscaling. This document tracks the
formalization needed to submit the instance to the ROAR-NET benchmark library.

## 1. Decision space (mixed discrete and continuous)

For a chain of `T` tiers, the decision vector is `x` in `(Z x R x R)^T`:

- `n_i` in `{replica_min, ..., replica_max}`: integer replicas
- `c_i` in `[cpu_min, cpu_max]`: continuous CPU limit (cores)
- `m_i` in `[mem_min, mem_max]`: continuous memory limit (GiB)

Two encodings are provided: a mixed-variable one (`AutoscalingProblem`, used by
NSGA-II) and a real-relaxed one with rounding (`RealEncodedAutoscalingProblem`,
used by MOEA/D and for same-encoding comparisons).

## 2. Objectives (minimize)

`f(x) = [ latency_ms, cost, energy_W ]`

- `latency_ms`: mean end-to-end sojourn over the workload. Each tier is an
  M/M/1-equivalent station with service floor `1/(n mu)` that rises to a timeout
  plateau near capacity, giving the CPU-throttle knee seen on real hardware.
- `cost`: `sum_i n_i (c_i price_cpu + m_i price_mem)`, fixed by the configuration.
- `energy_W`: mean deployment power. Node power `P = 50 + 200 u^2`, attributed to
  pods by CPU share with an even split of the idle floor.

## 3. Evaluation oracle and noise model

- Workload: Azure Functions 2021 invocation trace binned to per-minute mean rps
  over a 1440-minute day, with a synthetic diurnal fallback.
- Stochasticity: each replication realizes the mean rps with multiplicative Gamma
  noise, producing noisy objective values.
- Oracle: the mean objective vector over `K` replications with seeds
  `base_seed ... base_seed + K - 1`. Deterministic for a fixed `base_seed`.
- `K` is chosen by `experiments/cov_replications.py` (smallest K whose worst
  objective coefficient of variation drops below a target). The maximum
  evaluation budget is set by `experiments/timing_benchmark.py`.

## 4. Comparison protocol

- Budget: 500 evaluations at population 20, scaling toward 2000 at population 50,
  subject to the timing study.
- Algorithms: NSGA-II, MOEA/D, and the RL baseline, all scored on the same oracle
  for a fair objective-space comparison.
- Metrics: anytime hypervolume against cumulative evaluations, final GD+ and IGD+,
  and the Wilcoxon signed-rank test across 10 independent seeds.
- Analysis: the relation between the RL scalarization weights and the MOEA/D
  reference directions, and the spatial relation of the RL and evolutionary fronts.

## 5. Open items

- [ ] Calibrate `topology.py` constants (`service_demand_s`, working set,
      `node_cpu_capacity_cores`, energy parameters) to a real cluster.
- [ ] Confirm the multi-tier topology and call graph.
- [ ] Add the RL baseline on the offline oracle.
- [ ] Build the hypervolume, GD+/IGD+, and Wilcoxon aggregation with
      anytime-convergence plots.
- [ ] Confirm the final fronts on a real cluster.
