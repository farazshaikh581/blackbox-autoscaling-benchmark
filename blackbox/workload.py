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

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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


def from_azure_trace(
    path: str,
    day_index: int = 0,
    requests_per_invocation_scale: float = 1.0 / 60.0,
) -> Workload:
    """Load one day from the Azure Functions 2021 invocation trace.

    The raw trace is invocations with `end_timestamp`; we bin to invocations per
    minute and convert to a mean rps (`scale`, default: invocations/min -> /sec).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Azure trace not found at {path}. Download "
            "AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt (see README) "
            "or use synthetic_diurnal()."
        )
    df = pd.read_csv(p)
    df["end_timestamp"] = pd.to_numeric(df["end_timestamp"], errors="coerce")
    df = df.dropna(subset=["end_timestamp"])
    ts = pd.to_datetime(df["end_timestamp"], unit="s")
    minute = ((ts - ts.min()).dt.total_seconds() // 60).astype(int)
    day = minute // MINUTES_PER_DAY
    sel = df[day == day_index]
    minute_of_day = (minute[day == day_index] % MINUTES_PER_DAY)
    counts = np.zeros(MINUTES_PER_DAY, dtype=np.float64)
    binned = minute_of_day.value_counts()
    counts[binned.index.to_numpy()] = binned.to_numpy()
    mean_rps = counts * requests_per_invocation_scale
    return Workload(mean_rps=mean_rps, name=f"azure-2021-day{day_index}")
