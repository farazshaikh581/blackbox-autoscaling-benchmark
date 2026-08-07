"""Tests for the front-analysis primitives."""
import numpy as np
import pytest

from experiments.analyze_fronts import set_coverage, preference_consistency


def test_set_coverage_full_and_none():
    # A strictly dominates every point of B (minimization).
    A = np.array([[1.0, 1.0, 1.0]])
    B = np.array([[2.0, 2.0, 2.0], [3.0, 2.0, 4.0]])
    assert set_coverage(A, B) == 1.0
    # B does not dominate A.
    assert set_coverage(B, A) == 0.0


def test_set_coverage_partial():
    A = np.array([[1.0, 5.0, 1.0]])          # low latency/energy, high cost
    B = np.array([[2.0, 6.0, 2.0],           # dominated by A
                  [0.5, 0.5, 0.5]])          # dominates A, not covered
    assert set_coverage(A, B) == 0.5


def test_preference_consistency_perfect():
    # Each direction's emphasized objective is exactly the one minimized.
    W = np.eye(3)
    F = np.array([[1.0, 9.0, 9.0],   # e_latency -> min latency
                  [9.0, 1.0, 9.0],   # e_cost    -> min cost
                  [9.0, 9.0, 1.0]])  # e_energy  -> min energy
    assert preference_consistency(W, F) == 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
