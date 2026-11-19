"""Time to cross: crossing detection, log-log slopes, and the isolated-route scalar model (RQ2).

Convention, as the pilot's `updates_to_reach`: a series starts with its step-0 value, and the time
to cross a line is the first index i with series[i] >= line. None means the series never crossed
(censored).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def crossing_step(series: np.ndarray, line: float) -> int | None:
    """First index at or above `line`, or None if the 1-D series never gets there."""
    above = np.nonzero(np.asarray(series, dtype=float) >= line)[0]
    return int(above[0]) if len(above) else None


def loglog_slopes(xs: Sequence[float], ys: Sequence[float | None]) -> tuple[float, np.ndarray]:
    """(least-squares slope of log10 y on log10 x, slopes between neighbouring points)."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("need at least two (x, y) pairs of equal length")
    if any(y is None for y in ys):
        raise ValueError("a censored point (None) has no time to cross; the slope is undefined")
    lx, ly = np.log10(np.asarray(xs, dtype=float)), np.asarray(ys, dtype=float)
    if np.any(ly <= 0):
        raise ValueError("times to cross must be positive")
    ly = np.log10(ly)
    return float(np.polyfit(lx, ly, 1)[0]), np.diff(ly) / np.diff(lx)
