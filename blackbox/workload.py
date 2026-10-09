"""Workload model: per-minute request rate + stochastic replication.

The reference workload is the Microsoft Azure Functions 2021 invocation trace
(same source as the NOMS paper), aggregated to invocations-per-minute over a
1440-minute day. When the trace file is not present the module falls back to a
synthetic diurnal profile so the benchmark runs out of the box.

Stochasticity (the benchmark's noise source) is injected per replication: the
per-minute mean rate m(t) is realized as a noisy draw, so K replications with K
different seeds give K noisy objective evaluations of the *same* configuration.
This is what the noise study characterizes (variance vs. K).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

MINUTES_PER_DAY = 1440


@dataclass
class Workload:
    """A per-minute mean arrival-rate profile (requests/sec)."""

    mean_rps: np.ndarray            # shape (MINUTES_PER_DAY,), mean rps per minute
    cv: float = 0.10                # coefficient of variation of the noise draw
    name: str = "synthetic-diurnal"

    def realize(self, seed: int) -> np.ndarray:
        """One stochastic realization of the arrival process.

        Gamma-distributed multiplicative noise around the per-minute mean
        (mean 1, coefficient of variation `cv`), clipped to be non-negative.
        Deterministic given `seed` -> reproducible oracle.
        """
        rng = np.random.default_rng(seed)
        if self.cv <= 0:
            return self.mean_rps.copy()
        shape = 1.0 / (self.cv ** 2)
        scale = 1.0 / shape
        noise = rng.gamma(shape=shape, scale=scale, size=self.mean_rps.shape)
        return np.maximum(0.0, self.mean_rps * noise)


def synthetic_diurnal(peak_rps: float = 200.0, base_rps: float = 20.0) -> Workload:
    """A smooth day: low overnight, midday peak, evening shoulder."""
    t = np.arange(MINUTES_PER_DAY)
    # Two bumps (midday + evening) on a low baseline.
    midday = np.exp(-0.5 * ((t - 780) / 180) ** 2)      # ~13:00
    evening = 0.6 * np.exp(-0.5 * ((t - 1200) / 90) ** 2)  # ~20:00
    shape = midday + evening
    shape = shape / shape.max()
    mean_rps = base_rps + (peak_rps - base_rps) * shape
    return Workload(mean_rps=mean_rps.astype(np.float64), name="synthetic-diurnal")


def _parse_azure_day_counts(p: Path) -> np.ndarray:
    """Parse the raw trace into per-day per-minute invocation counts.

    Returns an array of shape (n_days, MINUTES_PER_DAY) of invocations per minute,
    unscaled. This is the expensive step (a full read of the ~300 MB file), so it
    is what the on-disk cache memoizes.
    """
    # Columns app, func, end_timestamp (seconds), duration (seconds). Sniff the
    # delimiter and tolerate a missing header row, since the extracted file has
    # shipped both comma- and tab-separated with/without a header.
    df = pd.read_csv(p, sep=None, engine="python")
    if "end_timestamp" not in df.columns:
        df = pd.read_csv(p, sep=None, engine="python", header=None,
                         names=["app", "func", "end_timestamp", "duration"])
    ts = pd.to_numeric(df["end_timestamp"], errors="coerce").dropna().to_numpy()
    minute = ((ts - ts.min()) // 60).astype(int)      # end_timestamp is in seconds
    day = minute // MINUTES_PER_DAY
    minute_of_day = minute % MINUTES_PER_DAY

    n_days = int(day.max()) + 1 if len(day) else 0
    counts = np.zeros((n_days, MINUTES_PER_DAY), dtype=np.float64)
    for d in range(n_days):
        idx, cnt = np.unique(minute_of_day[day == d], return_counts=True)
        counts[d, idx] = cnt
    return counts


def _azure_cache_path(p: Path) -> Path:
    return p.with_name(p.name + ".cache.npz")


def _source_signature(p: Path) -> np.ndarray:
    st = p.stat()
    return np.array([st.st_size, int(st.st_mtime)], dtype=np.int64)


def _load_or_build_azure_cache(p: Path, rebuild: bool = False) -> np.ndarray:
    """Per-day per-minute counts for the trace, memoized to <trace>.cache.npz.

    The cache is keyed on the source file's size and mtime, so editing or
    replacing the trace invalidates it automatically. The write is atomic (temp
    file + rename) so a concurrent reader never sees a half-written cache; still,
    prefer `build_azure_cache` once before a parallel run so workers only read it.
    """
    cache = _azure_cache_path(p)
    sig = _source_signature(p)
    if cache.exists() and not rebuild:
        try:
            with np.load(cache) as d:
                if "source_sig" in d.files and np.array_equal(d["source_sig"], sig):
                    return d["counts"]
        except Exception:
            pass  # unreadable / stale cache -> rebuild below
    counts = _parse_azure_day_counts(p)
    # Temp name must end in .npz, else np.savez appends it and os.replace misses.
    tmp = cache.with_name(cache.name + f".tmp{os.getpid()}.npz")
    np.savez(tmp, counts=counts, source_sig=sig)
    os.replace(tmp, cache)   # atomic on the same filesystem
    return counts


def build_azure_cache(path: str, rebuild: bool = False) -> int:
    """Warm (or rebuild) the trace cache; return the number of days available.

    Call this once before a parallel benchmark so the many worker processes read
    the cache instead of each re-parsing the ~300 MB trace.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Azure trace not found at {path}.")
    return int(_load_or_build_azure_cache(p, rebuild=rebuild).shape[0])


def from_azure_trace(
    path: str,
    day_index: int = 0,
    requests_per_invocation_scale: float = 1.0 / 60.0,
    days: Optional[Sequence[int]] = None,
    use_cache: bool = True,
    rebuild_cache: bool = False,
) -> Workload:
    """Load one or more days from the Azure Functions 2021 invocation trace.

    The raw trace is invocations with `end_timestamp`; we bin to invocations per
    minute and convert to a mean rps (`scale`, default: invocations/min -> /sec).

    By default a single `day_index` is loaded (mean_rps of length MINUTES_PER_DAY).
    Pass `days=[i, j, ...]` to concatenate several days into one profile in the
    given order (length MINUTES_PER_DAY * len(days)); the simulator averages its
    per-minute objectives over the whole profile, so this is how the train set of
    three days is assembled for the held-out-day protocol. `days` overrides
    `day_index`.

    `use_cache` (default) reads/writes a per-day-counts cache next to the trace
    (`<trace>.cache.npz`), so repeated and parallel loads skip the ~300 MB parse;
    `rebuild_cache` forces a fresh parse. The cache stores unscaled counts, so a
    different `requests_per_invocation_scale` reuses it. Requesting a day beyond
    the trace yields a zero profile, matching the raw-parse behavior.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Azure trace not found at {path}. Download "
            "AzureFunctionsInvocationTraceForTwoWeeksJan2021.rar (see README) "
            "and extract it, or use synthetic_diurnal()."
        )
    if use_cache:
        counts = _load_or_build_azure_cache(p, rebuild=rebuild_cache)
    else:
        counts = _parse_azure_day_counts(p)
    n_days = counts.shape[0]

    day_list = [int(day_index)] if days is None else [int(d) for d in days]

    def _one_day(d: int) -> np.ndarray:
        row = counts[d] if 0 <= d < n_days else np.zeros(MINUTES_PER_DAY)
        return row * requests_per_invocation_scale

    mean_rps = np.concatenate([_one_day(d) for d in day_list])
    tag = f"day{day_list[0]}" if len(day_list) == 1 else "days" + "-".join(map(str, day_list))
    return Workload(mean_rps=mean_rps, name=f"azure-2021-{tag}")
