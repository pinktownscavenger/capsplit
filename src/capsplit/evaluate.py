"""Score a pair of policies: the split, the excess score and the 2x2 label."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import pass_at_n
from .engine import TrajectoryTable, solution_probs, table_success_prob, trajectory_probs
from .tempering import DEFAULT_BETAS, Frontier, frontier, sequence_probs

_EPS = 1e-9
_SATURATED_BELOW = 0.05


@dataclass(frozen=True)
class Report:
    delta_rel: float
    delta_cap: float
    delta_j: float
    excess: float  # E_N
    j_gap: float  # J(pi1) - frontier.best_j
    label: str  # pure_sharpening | compounded_sharpening | reweighting | capability
    improved: bool  # delta_j > 0
    best_beta: float
    best_family: str
    boundary_hit: bool
    saturated: bool  # 1 - best pass@N < 0.05
    e_n_valid: bool  # n_actions ** horizon >= 100 * n
    crossed: dict  # solution -> (q0, q1), for s below c/N under pi0 and at or above it under pi1
    n: int
    c: float


@dataclass(frozen=True)
class BenchmarkReport:
    delta_rel: float  # mean over prompts
    delta_cap: float  # mean over prompts
    delta_j: float  # mean over prompts
    excess: float  # mean pass@N(pi1) - best mean pass@N over one shared beta and family
    label: str  # beyond reach: delta_cap >= c/n. Beyond temperature: excess > 1e-9
    per_prompt_j0: tuple
    per_prompt_j1: tuple
    n: int
    c: float


def label_cell(delta_cap: float, j_gap: float, reach_threshold: float, eps: float = _EPS) -> str:
    """The 2x2 cell (blueprint D2, provisional).

    Beyond reach needs real gain on U_N of at least the threshold. Beyond temperature
    needs a gap above eps. Pass `excess` as `j_gap` on a benchmark.
    """
    beyond_reach = delta_cap > eps and delta_cap >= reach_threshold
    beyond_temperature = j_gap > eps
    if beyond_reach and beyond_temperature:
        return "capability"
    if beyond_reach:
        return "compounded_sharpening"
    if beyond_temperature:
        return "reweighting"
    return "pure_sharpening"


def _check_distribution(table: TrajectoryTable, probs: np.ndarray) -> None:
    if probs.shape != (table.index.shape[0],):
        raise ValueError(f"probs1 shape {probs.shape} != {(table.index.shape[0],)}")
    if (probs < 0).any() or abs(probs.sum() - 1.0) > 1e-6:
        raise ValueError("probs1 is not a probability distribution")


def _split(table: TrajectoryTable, logits0: np.ndarray, probs1: np.ndarray, n: int, c: float):
    q0 = solution_probs(table, trajectory_probs(table, logits0))
    q1 = solution_probs(table, probs1)
    reachable = q0 >= c / n
    change = q1 - q0
    return float(change[reachable].sum()), float(change[~reachable].sum()), q0, q1


def evaluate(
    table: TrajectoryTable,
    logits0: np.ndarray,
    probs1: np.ndarray,
    n: int,
    *,
    c: float = 1.0,
    frontier_result: Frontier | None = None,
    reach_threshold: float | None = None,
    betas=DEFAULT_BETAS,
) -> Report:
    """Compare a base policy (logits) with a trained or planted one (trajectory probabilities).

    The frontier depends only on the base, so pass `frontier_result` when logging during training.
    """
    probs1 = np.asarray(probs1, dtype=float)
    _check_distribution(table, probs1)
    best = frontier_result if frontier_result is not None else frontier(table, logits0, betas)
    delta_rel, delta_cap, q0, q1 = _split(table, logits0, probs1, n, c)
    j1 = table_success_prob(table, probs1)
    best_pass = pass_at_n(best.best_j, n)
    threshold = c / n if reach_threshold is None else reach_threshold
    j_gap = j1 - best.best_j
    delta_j = delta_rel + delta_cap
    line = c / n
    crossed = {
        s: (float(a), float(b))
        for s, a, b in zip(table.solutions, q0, q1)
        if a < line <= b
    }
    return Report(
        delta_rel=delta_rel,
        delta_cap=delta_cap,
        delta_j=delta_j,
        excess=pass_at_n(j1, n) - best_pass,
        j_gap=j_gap,
        label=label_cell(delta_cap, j_gap, threshold),
        improved=delta_j > 0,
        best_beta=best.best_beta,
        best_family=best.best_family,
        boundary_hit=best.boundary_hit,
        saturated=1.0 - best_pass < _SATURATED_BELOW,
        e_n_valid=table.n_actions**table.horizon >= 100 * n,
        crossed=crossed,
        n=n,
        c=c,
    )


def _best_mean_pass(tables, logits0s, n: int, betas) -> float:
    """Sup of mean pass@N over one beta shared by every prompt. Grid only, no refinement."""
    best = -1.0
    for beta in np.unique(np.r_[np.asarray(betas, dtype=float), 1.0]):
        for probs_at in (trajectory_probs, sequence_probs):
            js = [table_success_prob(t, probs_at(t, l, beta)) for t, l in zip(tables, logits0s)]
            best = max(best, float(np.mean([pass_at_n(j, n) for j in js])))
    return best


def evaluate_benchmark(tables, logits0s, probs1s, n: int, *, c: float = 1.0, betas=DEFAULT_BETAS) -> BenchmarkReport:
    """Benchmark version: pass@N is averaged over prompts before subtracting (blueprint D5)."""
    if not len(tables) == len(logits0s) == len(probs1s):
        raise ValueError("tables, logits0s and probs1s must have the same length")
    probs1s = [np.asarray(p, dtype=float) for p in probs1s]
    for table, probs in zip(tables, probs1s):
        _check_distribution(table, probs)
    splits = [_split(t, l, p, n, c) for t, l, p in zip(tables, logits0s, probs1s)]
    delta_rel = float(np.mean([s[0] for s in splits]))
    delta_cap = float(np.mean([s[1] for s in splits]))
    j0 = tuple(table_success_prob(t, trajectory_probs(t, l)) for t, l in zip(tables, logits0s))
    j1 = tuple(table_success_prob(t, p) for t, p in zip(tables, probs1s))
    excess = float(np.mean([pass_at_n(j, n) for j in j1])) - _best_mean_pass(tables, logits0s, n, betas)
    return BenchmarkReport(
        delta_rel=delta_rel,
        delta_cap=delta_cap,
        delta_j=delta_rel + delta_cap,
        excess=excess,
        label=label_cell(delta_cap, excess, c / n),
        per_prompt_j0=j0,
        per_prompt_j1=j1,
        n=n,
        c=c,
    )
