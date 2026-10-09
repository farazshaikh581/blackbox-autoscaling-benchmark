"""Distinctness-on-front test for the edge-cloud placement axis: does placement
admit Pareto-optimal points that no fixed, single-class deployment can reach,
or is it redundant with a well-chosen fixed class?

Method: search three topologies at the same protocol (NSGA-II, same tiers,
budget, seed, K-replications) --

  edge-only:  every tier pinned to the edge NodeClass's constants, no WAN/egress
  cloud-only: every tier pinned to the cloud NodeClass's constants, no WAN/egress
  mixed:      edge_cloud_topology(), placement free (the actual benchmark)

-- in the same (latency, cost, energy) objective space (placement is a decision
variable, not an objective, so all three fronts are directly comparable). The
headline metric is `set_coverage` (Zitzler's C-metric, reused from
`experiments.analyze_fronts`): the fraction of the mixed front weakly
dominated by the union of the two pure-class fronts. 1 minus that is the
fraction of Pareto-optimal points that exist ONLY because placement is free.

    python -m experiments.placement_distinctness [--seed 1] [--base-seed 42]
"""
from __future__ import annotations

import argparse

import numpy as np

from pymoo.optimize import minimize
from pymoo.termination.max_eval import MaximumFunctionCallTermination

from blackbox import AutoscalingProblem, Topology, edge_cloud_topology
from experiments.run_nsga2 import build_nsga2
from experiments.analyze_fronts import set_coverage

EVALS = 500
POP = 20
SEED = 1
BASE_SEED = 42
K = 5


def pure_class_topology(edge_cloud: Topology, class_idx: int) -> Topology:
    """Single-class topology using node class `class_idx`'s own constants
    (power curve, prices, access RTT), no WAN/egress since there is only one
    class -- the fixed-deployment baseline placement is being compared against.
    """
    cls = edge_cloud.node_classes[class_idx]
    return Topology(tiers=edge_cloud.tiers, sla_latency_ms=edge_cloud.sla_latency_ms,
                    timeout_ms=edge_cloud.timeout_ms, node_classes=(cls,))


def search_front(topo: Topology) -> np.ndarray:
    problem = AutoscalingProblem(topology=topo, k_replications=K, base_seed=BASE_SEED)
    res = minimize(problem, build_nsga2(POP), MaximumFunctionCallTermination(EVALS),
                   seed=SEED, verbose=False)
    return np.atleast_2d(res.F)


def main():
    global SEED, BASE_SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--base-seed", type=int, default=BASE_SEED)
    args = ap.parse_args()
    SEED, BASE_SEED = args.seed, args.base_seed

    mixed_topo = edge_cloud_topology(3)
    edge_topo = pure_class_topology(mixed_topo, 0)
    cloud_topo = pure_class_topology(mixed_topo, 1)

    print("searching edge-only front...")
    F_edge = search_front(edge_topo)
    print("searching cloud-only front...")
    F_cloud = search_front(cloud_topo)
    print("searching mixed (placement-free) front...")
    F_mixed = search_front(mixed_topo)

    F_pure_union = np.vstack([F_edge, F_cloud])
    coverage = set_coverage(F_pure_union, F_mixed)
    admitted_only_by_placement = 1.0 - coverage

    print(f"\nedge-only front:  {len(F_edge)} points")
    print(f"cloud-only front: {len(F_cloud)} points")
    print(f"mixed front:      {len(F_mixed)} points")
    print(f"\nfraction of mixed front dominated by the pure-class union: {coverage:.3f}")
    print(f"fraction admitted ONLY by placement (genuinely new trade-offs): "
          f"{admitted_only_by_placement:.3f}")

    # reverse direction: does the mixed search also cover everything the pure
    # searches found, or does fixing a class ever find something placement misses?
    reverse = set_coverage(F_mixed, F_pure_union)
    print(f"\n(sanity) fraction of the pure-class union dominated by the mixed "
          f"front: {reverse:.3f}  (1.0 would mean mixed strictly dominates fixed)")


if __name__ == "__main__":
    main()
