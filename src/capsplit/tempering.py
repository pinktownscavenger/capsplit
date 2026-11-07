"""Re-tempering the base policy without training: the two families and their frontier."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .engine import TrajectoryTable, table_success_prob, trajectory_log_probs, trajectory_probs

DEFAULT_BETAS = np.unique(np.r_[np.exp(np.linspace(np.log(0.25), np.log(8.0), 25)), 1.0])

_GOLDEN = (math.sqrt(5) - 1) / 2
_REFINE_STEPS = 25
_REFINE_POINTS = 2


def sequence_probs(table: TrajectoryTable, logits: np.ndarray, beta: float) -> np.ndarray:
    """P_beta(tau) proportional to P(tau)^beta, over all trajectories (before any solution map)."""
    log_p = beta * trajectory_log_probs(table, logits)
    probs = np.exp(log_p - log_p.max())
    return probs / probs.sum()


@dataclass(frozen=True)
class Frontier:
    best_j: float
    best_beta: float
    best_family: str  # "per_step" or "sequence"
    boundary_hit: bool  # best_beta is at either end of the range


def _golden_section(j_of_log_beta, low: float, high: float) -> tuple[float, float]:
    """Maximise over [low, high] in log beta. Returns (log beta, J)."""
    c, d = high - _GOLDEN * (high - low), low + _GOLDEN * (high - low)
    fc, fd = j_of_log_beta(c), j_of_log_beta(d)
    for _ in range(_REFINE_STEPS):
        if fc > fd:
            high, d, fd = d, c, fc
            c = high - _GOLDEN * (high - low)
            fc = j_of_log_beta(c)
        else:
            low, c, fc = c, d, fd
            d = low + _GOLDEN * (high - low)
            fd = j_of_log_beta(d)
    middle = (low + high) / 2
    return middle, j_of_log_beta(middle)


def frontier(table: TrajectoryTable, logits: np.ndarray, betas=DEFAULT_BETAS, refine: bool = True) -> Frontier:
    """Best J over both families. A finite grid is a lower bound on the sup, so refine it.

    pass@N is monotone in J, so maximising J gives the best beta at every budget N.
    It depends only on the base policy: compute it once per base and reuse it.
    """
    betas = np.unique(np.r_[np.asarray(betas, dtype=float), 1.0])
    families = {
        "per_step": lambda b: trajectory_probs(table, logits, b),
        "sequence": lambda b: sequence_probs(table, logits, b),
    }
    best_j, best_beta, best_family = -1.0, 1.0, "per_step"
    for family, probs_at in families.items():

        def j_at(beta, probs_at=probs_at):
            return table_success_prob(table, probs_at(beta))

        values = np.array([j_at(b) for b in betas])
        for i in np.argsort(values)[::-1][: _REFINE_POINTS if refine else 1]:
            beta, j = float(betas[i]), float(values[i])
            if refine:
                low, high = betas[max(i - 1, 0)], betas[min(i + 1, len(betas) - 1)]
                log_beta, refined = _golden_section(lambda x: j_at(math.exp(x)), math.log(low), math.log(high))
                if refined > j:
                    beta, j = math.exp(log_beta), refined
            if j > best_j:
                best_j, best_beta, best_family = j, beta, family
    boundary_hit = bool(best_beta <= betas[0] * 1.001 or best_beta >= betas[-1] * 0.999)
    return Frontier(best_j, best_beta, best_family, boundary_hit)
