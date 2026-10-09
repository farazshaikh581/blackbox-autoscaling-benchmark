"""Tests for workload construction: synthetic, Azure trace, and CLI selection."""
import argparse

import numpy as np
import pandas as pd
import pytest

from blackbox.workload import (
    from_azure_trace, synthetic_diurnal, MINUTES_PER_DAY,
    build_azure_cache, _azure_cache_path,
)
from experiments._common import build_workload, build_train_test


def _write_trace(path):
    # end_timestamp in seconds -> minute-of-day bins 0, 0, 1, 60 on day 0.
    df = pd.DataFrame({"app": list("abcd"),
                       "end_timestamp": [0, 30, 90, 3600]})
    df.to_csv(path, index=False)


def test_from_azure_trace_bins_per_minute(tmp_path):
    p = tmp_path / "trace.csv"
    _write_trace(p)
    wl = from_azure_trace(str(p), day_index=0)          # scale = 1/60 by default
    assert wl.mean_rps.shape == (MINUTES_PER_DAY,)
    assert wl.name.startswith("azure-2021")
    np.testing.assert_allclose(wl.mean_rps[0], 2 / 60.0)  # two invocations minute 0
    np.testing.assert_allclose(wl.mean_rps[1], 1 / 60.0)
    np.testing.assert_allclose(wl.mean_rps[60], 1 / 60.0)
    assert wl.mean_rps.sum() == pytest.approx(4 / 60.0)   # four invocations total


def test_from_azure_trace_tab_separated(tmp_path):
    # Real trace has ships as tab-separated in some extractions.
    p = tmp_path / "trace.tsv"
    p.write_text("app\tfunc\tend_timestamp\tduration\n"
                 "a\tf\t0\t0.1\na\tf\t30\t0.1\na\tf\t3600\t0.1\n")
    wl = from_azure_trace(str(p))
    np.testing.assert_allclose(wl.mean_rps[0], 2 / 60.0)
    np.testing.assert_allclose(wl.mean_rps[60], 1 / 60.0)


def test_from_azure_trace_no_header(tmp_path):
    # Positional columns app, func, end_timestamp, duration with no header.
    p = tmp_path / "trace_nohdr.csv"
    p.write_text("a,f,0,0.1\na,f,30,0.1\na,f,90,0.1\n")
    wl = from_azure_trace(str(p))
    np.testing.assert_allclose(wl.mean_rps[0], 2 / 60.0)
    np.testing.assert_allclose(wl.mean_rps[1], 1 / 60.0)


def test_from_azure_trace_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        from_azure_trace(str(tmp_path / "nope.csv"))


def test_build_workload_synthetic_default():
    args = argparse.Namespace(workload="synthetic", trace=None, day=0)
    wl = build_workload(args)
    assert wl.name == synthetic_diurnal().name


def test_build_workload_azure_requires_trace():
    args = argparse.Namespace(workload="azure", trace=None, day=0)
    with pytest.raises(SystemExit):
        build_workload(args)


def test_build_workload_azure_loads(tmp_path):
    p = tmp_path / "trace.csv"
    _write_trace(p)
    args = argparse.Namespace(workload="azure", trace=str(p), day=0)
    wl = build_workload(args)
    assert wl.name.startswith("azure-2021")


def _write_multiday_trace(path, days=4):
    # One invocation per minute of each day, so day d contributes end_timestamps
    # d*86400 + minute*60. Per-minute count is exactly 1 (rps = 1/60) everywhere.
    ts = [d * 86400 + m * 60 for d in range(days) for m in range(MINUTES_PER_DAY)]
    pd.DataFrame({"app": "a", "func": "f", "end_timestamp": ts,
                  "duration": 0.1}).to_csv(path, index=False)


def test_from_azure_trace_multiday_concatenates(tmp_path):
    p = tmp_path / "trace.csv"
    _write_multiday_trace(p, days=4)
    wl = from_azure_trace(str(p), days=[0, 1, 2])
    assert wl.mean_rps.shape == (3 * MINUTES_PER_DAY,)
    np.testing.assert_allclose(wl.mean_rps, 1 / 60.0)   # 1 invocation/min everywhere
    assert "days0-1-2" in wl.name


def test_build_train_test_holdout_split(tmp_path):
    p = tmp_path / "trace.csv"
    _write_multiday_trace(p, days=4)
    args = argparse.Namespace(workload="azure", trace=str(p), day=0,
                              train_days="0,1,2", test_day=3)
    train, test = build_train_test(args)
    assert train.mean_rps.shape == (3 * MINUTES_PER_DAY,)   # three train days
    assert test.mean_rps.shape == (MINUTES_PER_DAY,)        # one held-out day
    assert "day3" in test.name


def test_build_train_test_rejects_leaky_test_day(tmp_path):
    p = tmp_path / "trace.csv"
    _write_multiday_trace(p, days=4)
    args = argparse.Namespace(workload="azure", trace=str(p), day=0,
                              train_days="0,1,2", test_day=1)  # 1 is in train
    with pytest.raises(SystemExit):
        build_train_test(args)


def test_azure_cache_matches_uncached_and_writes_file(tmp_path):
    p = tmp_path / "trace.csv"
    _write_multiday_trace(p, days=4)
    uncached = from_azure_trace(str(p), days=[0, 1, 2], use_cache=False)
    assert not _azure_cache_path(p).exists()               # no cache written yet
    cached = from_azure_trace(str(p), days=[0, 1, 2], use_cache=True)
    assert _azure_cache_path(p).exists()                   # cache now on disk
    np.testing.assert_array_equal(uncached.mean_rps, cached.mean_rps)
    # A second load reads the cache and is still identical.
    again = from_azure_trace(str(p), days=[0, 1, 2], use_cache=True)
    np.testing.assert_array_equal(cached.mean_rps, again.mean_rps)


def test_azure_cache_reuses_across_scale(tmp_path):
    # The cache stores unscaled counts, so a different scale reuses it (no reparse).
    p = tmp_path / "trace.csv"
    _write_multiday_trace(p, days=2)
    a = from_azure_trace(str(p), day_index=0, requests_per_invocation_scale=1.0)
    b = from_azure_trace(str(p), day_index=0, requests_per_invocation_scale=2.0)
    np.testing.assert_allclose(b.mean_rps, 2.0 * a.mean_rps)


def test_azure_cache_invalidates_on_source_change(tmp_path):
    p = tmp_path / "trace.csv"
    _write_multiday_trace(p, days=4)                       # 1 invocation/min
    first = from_azure_trace(str(p), day_index=0)
    np.testing.assert_allclose(first.mean_rps, 1 / 60.0)
    # Overwrite with a denser trace (2 invocations/min) -> different file size, so
    # the size+mtime signature no longer matches and the cache is rebuilt.
    ts = [d * 86400 + m * 60 for d in range(4) for m in range(MINUTES_PER_DAY)
          for _ in range(2)]
    pd.DataFrame({"app": "a", "func": "f", "end_timestamp": ts,
                  "duration": 0.1}).to_csv(p, index=False)
    second = from_azure_trace(str(p), day_index=0)
    np.testing.assert_allclose(second.mean_rps, 2 / 60.0)  # reflects new data


def test_build_azure_cache_returns_n_days(tmp_path):
    p = tmp_path / "trace.csv"
    _write_multiday_trace(p, days=5)
    assert build_azure_cache(str(p)) == 5
    assert _azure_cache_path(p).exists()


def test_build_train_test_synthetic_has_no_holdout():
    args = argparse.Namespace(workload="synthetic", trace=None, day=0,
                              train_days=None, test_day=None)
    train, test = build_train_test(args)
    assert test is None
    assert train.name == synthetic_diurnal().name


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
