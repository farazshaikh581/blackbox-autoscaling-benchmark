# Comparison protocol

This document tracks the benchmark study built on top of the instance
defined in `SPEC.md`: which algorithms are compared, on what metrics,
under what budget. Moved here from the old `SPEC.md` sections 4 and 5
when `SPEC.md` was reformatted to the ROAR-NET problem-statement
template, which covers the instance definition only.

## Protocol

- Budget: 2000 evaluations at population 50, scaled up from the 500/pop
  20 floor once the timing study confirmed the target budget is
  affordable.
- Algorithms: NSGA-II, MOEA/D, and the RL baseline, all scored on the
  same oracle for a fair objective-space comparison.
- Metrics: anytime hypervolume against cumulative evaluations, final
  GD+ and IGD+, and the Wilcoxon signed-rank test across 10 independent
  seeds.
- Analysis: the relation between the RL scalarization weights and the
  MOEA/D reference directions, and the spatial relation of the RL and
  evolutionary fronts.

An early NSGA-II/MOEA-D check at both budgets, held-out protocol (train
days 0/1/2, test day 3, base_seed 42, K=5, n=10 seeds): at the 500-eval
floor the two are statistically tied (HV 1.3131 vs 1.3066, Wilcoxon
p=0.160). At the 2000-eval target they separate, NSGA-II ahead (HV
1.3235 vs 1.3045, p=0.0020). The RL baseline is not in this check yet;
that three-way comparison lands with the full study.

## Open items

- [ ] Calibrate `topology.py` constants (`service_demand_s`, working
      set, `node_cpu_capacity_cores`, energy parameters) to a real
      cluster.
- [x] Confirm the multi-tier topology and call graph (`SPEC.md`,
      Detailed description).
- [ ] Add the RL baseline on the offline oracle.
- [ ] Build the hypervolume, GD+/IGD+, and Wilcoxon aggregation with
      anytime-convergence plots.
- [ ] Confirm the final fronts on a real cluster.
