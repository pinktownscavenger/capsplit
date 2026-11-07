"""Exact engine: a task compiled to arrays, so every policy can be scored without re-enumeration.

A policy is a float array of logits shaped (n_states, n_actions). A trajectory
distribution is an array shaped (T,). Grid worlds and sequence tasks both compile
to a `TrajectoryTable`, so metrics never need to know which kind of task they see.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryTable:
    index: np.ndarray  # int32 (T, H): state * n_actions + action; padding == n_states * n_actions
    solution_ids: np.ndarray  # int32 (T,): index into `solutions`, or -1 if the verifier rejects
    solutions: tuple  # the correct solutions only; may be empty
    n_states: int
    n_actions: int
    horizon: int


def log_policy(logits: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Log of the per-step policy pi^beta, renormalised at each state."""
    z = beta * np.asarray(logits, dtype=float)
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def trajectory_log_probs(table: TrajectoryTable, logits: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Log-probability of each trajectory under per-step tempering at beta."""
    logits = np.asarray(logits)
    if logits.shape != (table.n_states, table.n_actions):
        raise ValueError(f"logits shape {logits.shape} != {(table.n_states, table.n_actions)}")
    # The appended zero is what the padding index reads, so padded steps add nothing.
    flat = np.append(log_policy(logits, beta).ravel(), 0.0)
    return flat[table.index].sum(axis=1)


def trajectory_probs(table: TrajectoryTable, logits: np.ndarray, beta: float = 1.0) -> np.ndarray:
    return np.exp(trajectory_log_probs(table, logits, beta))


def solution_probs(table: TrajectoryTable, probs: np.ndarray) -> np.ndarray:
    """q(s) for each correct solution."""
    correct = table.solution_ids >= 0
    return np.bincount(table.solution_ids[correct], weights=probs[correct], minlength=len(table.solutions))


def table_success_prob(table: TrajectoryTable, probs: np.ndarray) -> float:
    """J: total probability of trajectories the verifier accepts."""
    return float(probs[table.solution_ids >= 0].sum())
