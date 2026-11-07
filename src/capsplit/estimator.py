"""Sampled split and its sample sizes. Classification uses the exact q under pi0; only the means are estimated."""
import math
from dataclasses import dataclass

import numpy as np

from capsplit.engine import TrajectoryTable, solution_probs, trajectory_probs
from capsplit.evaluate import _check_distribution


def _check_delta(delta: float) -> None:
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")


def chernoff_size(mu: float, eta: float, delta: float) -> int:
    """Samples per policy for error at most eta * max(m, mu) on all four means (D11)."""
    if not 0.0 < eta <= 1.0:
        raise ValueError(f"eta must be in (0, 1], got {eta}")
    if mu <= 0.0:
        raise ValueError(f"mu must be positive, got {mu}")
    _check_delta(delta)
    return math.ceil(3.0 * math.log(8.0 / delta) / (eta**2 * mu))


def hoeffding_size(eps: float, delta: float) -> int:
    """Samples per policy for error at most eps on all four means."""
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    _check_delta(delta)
    return math.ceil(math.log(8.0 / delta) / (2.0 * eps**2))


@dataclass(frozen=True)
class SplitEstimate:
    delta_rel: float
    delta_cap: float
    means: np.ndarray  # [reach_pi0, unreach_pi0, reach_pi1, unreach_pi1]


def _reachable(table: TrajectoryTable, probs0: np.ndarray, n: int, c: float) -> np.ndarray:
    return solution_probs(table, probs0) >= c / n


def _class_means(table: TrajectoryTable, weights: np.ndarray, reachable: np.ndarray) -> np.ndarray:
    """[mass on reachable solutions, mass on unreachable solutions]."""
    q = solution_probs(table, weights)
    return np.array([q[reachable].sum(), q[~reachable].sum()])


def exact_means(table: TrajectoryTable, logits0: np.ndarray, probs1: np.ndarray, n: int, c: float = 1.0) -> np.ndarray:
    probs1 = np.asarray(probs1, dtype=float)
    _check_distribution(table, probs1)
    probs0 = trajectory_probs(table, logits0)
    reachable = _reachable(table, probs0, n, c)
    return np.concatenate((_class_means(table, probs0, reachable), _class_means(table, probs1, reachable)))


def _sample_means(table, probs: np.ndarray, reachable: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    counts = rng.multinomial(m, probs / probs.sum())
    return _class_means(table, counts / m, reachable)


def estimate_split(
    table: TrajectoryTable,
    logits0: np.ndarray,
    probs1: np.ndarray,
    n: int,
    m: int,
    rng: np.random.Generator,
    c: float = 1.0,
) -> SplitEstimate:
    """Draw m trajectories from each policy, independently."""
    if m < 1:
        raise ValueError(f"m must be at least 1, got {m}")
    probs1 = np.asarray(probs1, dtype=float)
    _check_distribution(table, probs1)
    probs0 = trajectory_probs(table, logits0)
    reachable = _reachable(table, probs0, n, c)
    means = np.concatenate(
        (_sample_means(table, probs0, reachable, m, rng), _sample_means(table, probs1, reachable, m, rng))
    )
    return SplitEstimate(delta_rel=means[2] - means[0], delta_cap=means[3] - means[1], means=means)


def within_bound(estimate: SplitEstimate, exact: np.ndarray, eta: float, mu: float) -> bool:
    """Every mean within eta * max(m, mu) of its exact value."""
    exact = np.asarray(exact, dtype=float)
    return bool((np.abs(estimate.means - exact) <= eta * np.maximum(exact, mu)).all())
