"""Tests for the edge/cloud extension (Tier 1: node classes, placement, WAN, egress).

These pin the new oracle-side model in `_simulate_config_multiclass`:
- a single node class matching the default constants reproduces the
  single-location path exactly (the extension is a strict generalization);
- placement onto heterogeneous classes moves energy and cost;
- crossing classes on consecutive tiers adds WAN latency and egress cost.
"""
import numpy as np
import pytest

from pymoo.optimize import minimize
from pymoo.termination.max_eval import MaximumFunctionCallTermination

from blackbox import (
    Topology, NodeClass, AutoscalingProblem, default_topology, edge_cloud_topology,
    synthetic_diurnal, evaluate,
)
from blackbox.oracle import RealEncodedAutoscalingProblem, x_to_config
from blackbox import energy_model as em


def _obj(topo, rep, cpu, mem, wl, place=None):
    cfg = {"replicas": np.array(rep), "cpu": np.array(cpu, float),
           "mem": np.array(mem, float)}
    if place is not None:
        cfg["place"] = np.array(place, int)
    return evaluate(topo, cfg, wl, k_replications=1, base_seed=0)["mean"]


def _one_class_like(base):
    """A single node class whose constants match the single-location defaults."""
    return NodeClass(
        name="node",
        cpu_capacity_cores=base.node_cpu_capacity_cores,
        mem_capacity_gib=base.node_mem_capacity_gib,
        p_idle=em.P_IDLE, p_max=em.P_MAX, alpha=em.ALPHA,
        price_cpu_per_core=base.price_cpu_per_core,
        price_mem_per_gib=base.price_mem_per_gib,
    )


def test_single_class_matches_single_location_path():
    # One class equal to the defaults, all tiers on it, no WAN/egress -> identical.
    base = default_topology(3)
    multi = Topology(tiers=base.tiers, node_classes=(_one_class_like(base),))
    wl = synthetic_diurnal()
    for rep, cpu, mem in [([1, 1, 1], [1, 1, 1], [1, 1, 1]),
                          ([6, 4, 2], [1.5, 0.5, 2.0], [1.0, 0.5, 2.0]),
                          ([3, 3, 3], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0])]:
        a = _obj(base, rep, cpu, mem, wl)
        b = _obj(multi, rep, cpu, mem, wl, place=[0, 0, 0])
        assert np.allclose(a, b, rtol=1e-9, atol=1e-9), (rep, a, b)


def test_placement_changes_energy_and_cost():
    # All-edge vs all-cloud placement gives different energy and cost.
    topo, wl = edge_cloud_topology(3), synthetic_diurnal()
    all_edge = _obj(topo, [2, 2, 2], [1, 1, 1], [1, 1, 1], wl, place=[0, 0, 0])
    all_cloud = _obj(topo, [2, 2, 2], [1, 1, 1], [1, 1, 1], wl, place=[1, 1, 1])
    assert all_edge[2] != all_cloud[2]          # energy differs by class power curve
    assert all_edge[1] != all_cloud[1]          # cost differs by class prices


def test_wan_latency_added_across_classes():
    # Hold the entry tier's class fixed (so client access RTT cancels); splitting
    # the middle tier onto the other class adds two WAN hops.
    topo, wl = edge_cloud_topology(3), synthetic_diurnal()
    same = _obj(topo, [3, 3, 3], [1, 1, 1], [1, 1, 1], wl, place=[1, 1, 1])
    split = _obj(topo, [3, 3, 3], [1, 1, 1], [1, 1, 1], wl, place=[1, 0, 1])
    # two cross-class hops (cloud->edge, edge->cloud) at 30 ms each = 60 ms.
    assert np.isclose(split[0] - same[0], 60.0, atol=1e-6)


def test_client_access_latency_at_entry_tier():
    # The entry tier's class sets the client access RTT; a far class adds it.
    from dataclasses import replace
    base = default_topology(3)
    near = _one_class_like(base)                       # access_rtt_ms 0
    far = replace(near, name="far", access_rtt_ms=40.0)
    topo = Topology(tiers=base.tiers, node_classes=(near, far))  # no WAN matrix
    wl = synthetic_diurnal()
    entry_near = _obj(topo, [3, 3, 3], [1, 1, 1], [1, 1, 1], wl, place=[0, 0, 0])
    entry_far = _obj(topo, [3, 3, 3], [1, 1, 1], [1, 1, 1], wl, place=[1, 0, 0])
    assert np.isclose(entry_far[0] - entry_near[0], 40.0, atol=1e-6)


def test_egress_cost_per_cross_hop():
    # Two identical classes isolate egress: moving a tier between them changes no
    # per-tier price or power, so cost rises by exactly egress_cost_per_hop per hop.
    base = default_topology(3)
    cls = _one_class_like(base)
    topo = Topology(tiers=base.tiers, node_classes=(cls, cls),
                    wan_rtt_ms=((0.0, 30.0), (30.0, 0.0)), egress_cost_per_hop=0.5)
    wl = synthetic_diurnal()
    no_hop = _obj(topo, [1, 1, 1], [1, 1, 1], [1, 1, 1], wl, place=[0, 0, 0])
    one_hop = _obj(topo, [1, 1, 1], [1, 1, 1], [1, 1, 1], wl, place=[0, 0, 1])
    assert np.isclose(one_hop[1] - no_hop[1], 0.5, atol=1e-9)   # cost: one egress hop
    assert np.isclose(one_hop[0] - no_hop[0], 30.0, atol=1e-6)  # latency: one WAN hop
    # energy rises: a tier on its own class cannot consolidate onto the other
    # class's node, so splitting forces an extra powered node.
    assert one_hop[2] > no_hop[2]


def test_edge_cloud_topology_defaults():
    topo = edge_cloud_topology(3)
    edge, cloud = topo.node_classes
    assert edge.alpha == 0.7
    assert edge.price_cpu_per_core == 2.0 and edge.price_mem_per_gib == 0.50
    assert topo.wan_rtt_ms == ((0.0, 30.0), (30.0, 0.0))


def test_place_defaults_to_class_zero():
    # Omitting 'place' places every tier on class 0 (no crash, no WAN/egress).
    topo, wl = edge_cloud_topology(3), synthetic_diurnal()
    default_place = _obj(topo, [2, 2, 2], [1, 1, 1], [1, 1, 1], wl)
    explicit = _obj(topo, [2, 2, 2], [1, 1, 1], [1, 1, 1], wl, place=[0, 0, 0])
    assert np.allclose(default_place, explicit)


def test_pack_pods_by_class_splits_by_placement():
    # Tiers on different classes never share a node.
    edge = NodeClass(name="edge", cpu_capacity_cores=4, mem_capacity_gib=8)
    cloud = NodeClass(name="cloud", cpu_capacity_cores=16, mem_capacity_gib=32)
    packed = em.pack_pods_by_class(
        np.array([2, 2, 2]), np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]),
        np.array([0, 1, 0]), (edge, cloud), 3)
    # every node belongs to one class and holds only that class's tiers
    for k, counts in packed:
        tiers_here = np.nonzero(counts)[0]
        assert all((t in (0, 2)) == (k == 0) for t in tiers_here)
    assert sum(int(counts.sum()) for _, counts in packed) == 6


# --- Increment 2: placement in the optimizer encodings ---------------------

def test_x_to_config_emits_place_only_when_multiclass():
    single = x_to_config({"n_0": 1, "c_0": 0.5, "m_0": 0.5}, 1, n_classes=1)
    assert "place" not in single
    multi = x_to_config({"n_0": 1, "c_0": 0.5, "m_0": 0.5, "p_0": 1}, 1, n_classes=2)
    assert list(multi["place"]) == [1]


def test_mixedvar_problem_adds_placement_vars_on_edge():
    topo = edge_cloud_topology(3)
    prob = AutoscalingProblem(topology=topo, workload=synthetic_diurnal())
    assert all(f"p_{i}" in prob.vars for i in range(topo.n_tiers))
    # single-location has no placement vars
    base = AutoscalingProblem(topology=default_topology(3), workload=synthetic_diurnal())
    assert not any(k.startswith("p_") for k in base.vars)


def test_realencoded_problem_adds_placement_column_on_edge():
    topo = edge_cloud_topology(3)
    prob = RealEncodedAutoscalingProblem(topology=topo, workload=synthetic_diurnal())
    assert prob.n_var == 4 * topo.n_tiers
    row = np.array([2, 1.0, 1.0, 1.9,   3, 0.5, 0.5, 0.4,   1, 2.0, 1.0, 1.0])
    cfg = prob._row_to_config(row)
    assert list(cfg["place"]) == [1, 0, 1]          # floored real -> class index
    assert cfg["place"].max() <= topo.n_classes - 1


def test_nsga2_searches_placement_on_edge():
    from experiments.run_nsga2 import build_nsga2
    topo = edge_cloud_topology(3)
    prob = AutoscalingProblem(topology=topo, workload=synthetic_diurnal(),
                              k_replications=1)
    res = minimize(prob, build_nsga2(8), MaximumFunctionCallTermination(32), seed=1)
    assert res.F.shape[1] == 3 and len(res.F) >= 1
    places = {tuple(x_to_config(x, topo.n_tiers, topo.n_classes)["place"])
              for x in np.atleast_1d(res.X)}
    for pl in places:                                # every placement is valid
        assert all(0 <= v <= topo.n_classes - 1 for v in pl)


def test_moead_runs_on_edge():
    from experiments.run_moead import build_moead
    topo = edge_cloud_topology(3)
    prob = RealEncodedAutoscalingProblem(topology=topo, workload=synthetic_diurnal(),
                                         k_replications=1)
    res = minimize(prob, build_moead(3), MaximumFunctionCallTermination(30), seed=1)
    F = np.atleast_2d(res.F)
    assert F.shape[1] == 3
    for row in np.atleast_2d(res.X):
        cfg = prob._row_to_config(row)
        assert 0 <= cfg["place"].min() and cfg["place"].max() <= topo.n_classes - 1


def test_morl_runs_on_edge():
    from experiments.run_morl import run_morl
    topo = edge_cloud_topology(3)
    out = run_morl(topo, evals=40, batch=10, warmup=10, partitions=3, archive_cap=10,
                   k_replications=1, seed=1, workload=synthetic_diurnal())
    F = out["F"]
    assert F.shape[1] == 3 and len(F) >= 1
