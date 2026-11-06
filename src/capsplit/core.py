"""Exact reliability/capability split of policy improvement.

A policy is a dict mapping a solution (any hashable) to its probability q(s).
Correct solutions are given as a set S*. In small worlds the dict comes from
enumerating all trajectories, so every quantity here is exact.
"""
from __future__ import annotations

from typing import Hashable, Iterable, Mapping

Policy = Mapping[Hashable, float]


def success_prob(q: Policy, correct: Iterable[Hashable]) -> float:
    """J(pi) = sum of q(s) over verified-correct solutions."""
    return sum(q.get(s, 0.0) for s in correct)


def pass_at_n(j: float, n: int) -> float:
    """pass@N for a single task: 1 - (1 - J)^N."""
    return 1.0 - (1.0 - j) ** n


def reachable_set(q0: Policy, correct: Iterable[Hashable], n: int, c: float = 1.0) -> set:
    """R_N = {s in S*: q0(s) >= c/N}. c=1 is the plan's Definition 1."""
    return {s for s in correct if q0.get(s, 0.0) >= c / n}


def split(q0: Policy, q1: Policy, correct: Iterable[Hashable], n: int, c: float = 1.0) -> dict:
    """Return delta_rel, delta_cap and their sum J(pi1) - J(pi0) (Proposition 1)."""
    correct = set(correct)
    reach = reachable_set(q0, correct, n, c)
    unreach = correct - reach
    d_rel = sum(q1.get(s, 0.0) - q0.get(s, 0.0) for s in reach)
    d_cap = sum(q1.get(s, 0.0) - q0.get(s, 0.0) for s in unreach)
    return {"delta_rel": d_rel, "delta_cap": d_cap, "delta_j": d_rel + d_cap}


def sequence_temper(q: Policy, beta: float) -> dict:
    """Sequence-level family: P_beta(tau) proportional to P(tau)^beta over the support."""
    w = {s: p**beta for s, p in q.items() if p > 0}
    z = sum(w.values())
    return {s: v / z for s, v in w.items()}


def excess_score(
    q0: Policy,
    q1: Policy,
    correct: Iterable[Hashable],
    n: int,
    betas: Iterable[float],
    tempering=sequence_temper,
) -> float:
    """E_N = pass@N(pi1) - sup over beta of pass@N(tempered pi0).

    `tempering` is a function (q, beta) -> q'. For several families (e.g. per-step
    as well), use `excess_score_families`.
    """
    correct = set(correct)
    best = max(pass_at_n(success_prob(tempering(q0, b), correct), n) for b in betas)
    return pass_at_n(success_prob(q1, correct), n) - best


def label_2x2(delta_cap: float, e_n: float, tol: float = 1e-12) -> str:
    """Cell of the 2x2 diagnostic."""
    beyond_reach = delta_cap > tol
    beyond_temp = e_n > tol
    if beyond_reach and beyond_temp:
        return "capability"
    if beyond_reach:
        return "compounded_sharpening"
    if beyond_temp:
        return "reweighting"
    return "pure_sharpening"


def excess_score_families(q1: Policy, correct: Iterable[Hashable], n: int, families, betas) -> float:
    """E_N with the sup taken over several tempering families.

    `families` is a list of callables beta -> policy dict (e.g. the sequence-level
    family `lambda b: sequence_temper(q0, b)` and the per-step family from worlds).
    """
    correct = set(correct)
    best = max(
        pass_at_n(success_prob(fam(b), correct), n) for fam in families for b in betas
    )
    return pass_at_n(success_prob(q1, correct), n) - best
