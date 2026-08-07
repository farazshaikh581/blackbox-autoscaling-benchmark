"""Smoke tests for the MORL baseline runner."""
import numpy as np
import pytest

from blackbox import default_topology
from experiments.run_morl import run_morl, BoundedArchive


def test_run_respects_budget_and_format():
    topo = default_topology(2)
    out = run_morl(topo, evals=120, batch=20, warmup=20, k_replications=2, seed=1)
    # The oracle-call budget is honored exactly.
    assert out["n_eval_calls"] == 120
    # Final front and every history snapshot are 3-objective and non-empty.
    assert out["F"].shape[1] == 3 and len(out["F"]) > 0
    assert len(out["hist_n"]) == len(out["hist_F"])
    assert out["hist_n"][-1] == 120
    for f in out["hist_F"]:
        assert f.shape[1] == 3 and len(f) > 0


def test_front_is_cardinality_capped_for_fairness():
    topo = default_topology(3)
    cap = 20
    out = run_morl(topo, evals=200, batch=20, archive_cap=cap, k_replications=2, seed=3)
    # Reported front never exceeds the EA-comparable cap.
    assert len(out["F"]) <= cap
    assert all(len(f) <= cap for f in out["hist_F"])


def test_deterministic_given_seed():
    topo = default_topology(2)
    a = run_morl(topo, evals=100, batch=20, k_replications=2, seed=5)["F"]
    b = run_morl(topo, evals=100, batch=20, k_replications=2, seed=5)["F"]
    np.testing.assert_allclose(np.sort(a, axis=0), np.sort(b, axis=0))


def test_bounded_archive_keeps_nondominated_and_caps():
    arc = BoundedArchive(cap=3)
    # Feed a mix; a dominated point must never survive.
    for f in [[3, 3, 3], [1, 5, 5], [5, 1, 5], [5, 5, 1], [4, 4, 4], [2, 2, 2]]:
        arc.add(np.array(f, float))
    F = arc.front()
    assert len(F) <= 3
    # [4,4,4] is dominated by [2,2,2]; it must be gone.
    assert not any(np.allclose(row, [4, 4, 4]) for row in F)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
