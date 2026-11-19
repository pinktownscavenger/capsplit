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


# ------------------------------------------------------------- isolated-route scalar model
#
# On a world with one correct route whose length equals the horizon H (docs/proofs.md section 7),
# a base with one per-step bonus g on the route action and equal logits on the other A - 1
# actions keeps every route cell identical under exact softmax PG. The state is the logit gap g,
# with p = e^g / (e^g + A - 1) and q = J = p^H, and each update adds lr * A / (A - 1) * q (1 - p):
# the route action gains q (1 - p) and each other action loses q (1 - p) / (A - 1).


def _gap0(q0: float, horizon: int, n_actions: int) -> float:
    p0 = q0 ** (1.0 / horizon)
    return float(np.log((n_actions - 1) * p0 / (1.0 - p0)))


def scalar_isolated_trace(
    q0: float, horizon: int, lr: float, steps: int, n_actions: int = 4, stop_line: float | None = None
) -> np.ndarray:
    """q at updates 0..T of the scalar recursion; stops after the first value >= stop_line if given."""
    gap, others, gain = _gap0(q0, horizon, n_actions), n_actions - 1, lr * n_actions / (n_actions - 1)
    out = []
    for _ in range(steps + 1):
        p = 1.0 / (1.0 + others * np.exp(-gap))
        q = p**horizon
        out.append(q)
        if stop_line is not None and q >= stop_line:
            break
        gap += gain * q * (1.0 - p)
    return np.array(out)


def scalar_isolated_updates(q0: float, horizon: int, lr: float, line: float, cap: int, n_actions: int = 4) -> int | None:
    """Updates of the scalar recursion until q >= line, or None if not within `cap`."""
    return crossing_step(scalar_isolated_trace(q0, horizon, lr, cap, n_actions, stop_line=line), line)


_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(400)


def integral_time(q0: float, horizon: int, lr: float, line: float, n_actions: int = 4) -> float:
    """Continuous-time T = ((A-1)/A)/lr * integral of dp / (p^(H+1) (1-p)^2) from q0^(1/H) to line^(1/H).

    Integrated in u = log p with 400-point Gauss-Legendre (no scipy).
    """
    a, b = np.log(q0 ** (1.0 / horizon)), np.log(line ** (1.0 / horizon))
    u = 0.5 * (b - a) * _NODES + 0.5 * (b + a)
    p = np.exp(u)
    integrand = np.exp(-horizon * u) / (1.0 - p) ** 2  # f(p) dp with dp = p du
    return float(0.5 * (b - a) * np.dot(_WEIGHTS, integrand) * (n_actions - 1) / n_actions / lr)
