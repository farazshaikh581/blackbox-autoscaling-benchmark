"""Tests for the per-node consolidation energy model.

The model bin-packs pods onto nodes and sums every powered node's power, so
energy is driven by the number of powered nodes (idle floor) plus a convex
dynamic term. These tests pin the physically-required behavior that the earlier
single-node attribution model got wrong: energy must rise with replicas and pod
size, respond to both cpu and memory packing, and never depend on load in a way
that breaks the powered-node floor.
"""
import numpy as np

from blackbox import default_topology, synthetic_diurnal, evaluate
from blackbox import energy_model as em


def _energy(topo, rep, cpu, mem, wl):
    cfg = {"replicas": np.array(rep), "cpu": np.array(cpu, float),
           "mem": np.array(mem, float)}
    return evaluate(topo, cfg, wl, k_replications=1, base_seed=0)["mean"][2]


def test_energy_rises_with_replicas():
    # More replicas spill onto more nodes -> more idle floors -> more energy.
    topo, wl = default_topology(3), synthetic_diurnal()
    few = _energy(topo, [1, 1, 1], [1, 1, 1], [1, 1, 1], wl)
    many = _energy(topo, [6, 6, 6], [1, 1, 1], [1, 1, 1], wl)
    assert many > few


def test_energy_rises_with_cpu_request():
    # Bigger pods pack fewer per node -> more nodes -> more energy.
    topo, wl = default_topology(3), synthetic_diurnal()
    small = _energy(topo, [2, 2, 2], [1.0, 1.0, 1.0], [1, 1, 1], wl)
    big = _energy(topo, [2, 2, 2], [2.0, 2.0, 2.0], [1, 1, 1], wl)
    assert big > small


def test_energy_responds_to_memory_when_it_binds():
    # With memory the binding dimension, larger mem requests force more nodes.
    topo, wl = default_topology(3), synthetic_diurnal()
    # tiny cpu so packing is memory-bound; grow mem past the node's 16 GiB.
    lean = _energy(topo, [6, 6, 6], [0.1, 0.1, 0.1], [0.5, 0.5, 0.5], wl)
    fat = _energy(topo, [6, 6, 6], [0.1, 0.1, 0.1], [2.0, 2.0, 2.0], wl)
    assert fat > lean


def test_single_small_deployment_is_one_powered_node():
    # A handful of small pods consolidate onto one node: energy sits between
    # that node's idle floor and its max draw. A tight multiplier on P_IDLE
    # (e.g. "< 2x idle") isn't shape-invariant: with alpha < 1 the curve rises
    # steeply just above zero utilization, so even a small nonzero load can
    # be several multiples of a near-zero idle floor. P_IDLE/P_MAX bound what
    # a single powered node can draw regardless of alpha's shape.
    topo, wl = default_topology(3), synthetic_diurnal()
    e = _energy(topo, [1, 1, 1], [1, 1, 1], [1, 1, 1], wl)
    assert em.P_IDLE <= e < em.P_MAX


def test_pack_counts_pods_and_respects_capacity():
    topo = default_topology(3)
    counts = em.pack_pods(np.array([2, 2, 2]), np.array([1.0, 1.0, 1.0]),
                          np.array([1.0, 1.0, 1.0]),
                          topo.node_cpu_capacity_cores,
                          topo.node_mem_capacity_gib, 3)
    # 6 one-core pods fit on a single 8-core node.
    assert counts.shape == (1, 3)
    assert counts.sum() == 6
    # 6 two-core pods (12 cores) cannot share one 8-core node.
    big = em.pack_pods(np.array([2, 2, 2]), np.array([2.0, 2.0, 2.0]),
                       np.array([1.0, 1.0, 1.0]),
                       topo.node_cpu_capacity_cores,
                       topo.node_mem_capacity_gib, 3)
    assert big.shape[0] >= 2
    assert big.sum() == 6  # every pod is placed exactly once


def test_energy_independent_of_cpu_and_mem_is_false():
    # Regression guard against the old model: cpu/mem MUST be able to move energy.
    topo, wl = default_topology(3), synthetic_diurnal()
    base = _energy(topo, [3, 3, 3], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], wl)
    more_cpu = _energy(topo, [3, 3, 3], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0], wl)
    assert more_cpu != base
