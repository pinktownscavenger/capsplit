"""Shared machinery for fixed-length token-sequence tasks: the prefix tree and its exact table.

State numbering (used by every sequence policy's `state_logits`): level l holds the
V**l prefixes of length l, and prefix (t_0, ..., t_{l-1}) is node
`node_offsets(V, L)[l] + code`, where code is the prefix read as a base-V number.
"""
from __future__ import annotations

from typing import Hashable, Sequence

import numpy as np

from capsplit.engine import TrajectoryTable
from capsplit.seqtask.verifier import node_offsets

from .interfaces import Task

__all__ = ["node_offsets", "compile_prefix_tree", "n_nodes", "prefix_nodes", "SequenceTask"]


def n_nodes(vocab: int, length: int) -> int:
    return int(node_offsets(vocab, length)[-1] + vocab ** (length - 1))


def compile_prefix_tree(vocab: int, length: int, correct: Sequence[tuple]) -> TrajectoryTable:
    """Every length-L sequence over V tokens, with `correct` as the accepted solutions.

    Same layout as `capsplit.seqtask.verifier.compile_sequence`, for an arbitrary correct set.
    """
    offsets = node_offsets(vocab, length)
    codes = np.arange(vocab**length)
    powers = vocab ** np.arange(length - 1, -1, -1)
    prefixes = codes[:, None] // (powers * vocab)
    tokens = (codes[:, None] // powers) % vocab
    solutions = tuple(sorted(set(tuple(int(t) for t in s) for s in correct)))
    solution_ids = np.full(len(codes), -1, dtype=np.int32)
    if solutions:
        solution_ids[np.dot(np.array(solutions), powers)] = np.arange(len(solutions))
    return TrajectoryTable(
        index=((offsets + prefixes) * vocab + tokens).astype(np.int32),
        solution_ids=solution_ids,
        solutions=solutions,
        n_states=n_nodes(vocab, length),
        n_actions=vocab,
        horizon=length,
    )


def prefix_nodes(vocab: int, length: int, actions: np.ndarray) -> np.ndarray:
    """(B, L) node id of the state before each action, for a batch of full sequences."""
    actions = np.asarray(actions, dtype=np.int64)
    offsets = node_offsets(vocab, length)
    codes = np.zeros(actions.shape[0], dtype=np.int64)
    out = np.empty_like(actions)
    for level in range(length):
        out[:, level] = offsets[level] + codes
        codes = codes * vocab + actions[:, level]
    return out


class SequenceTask(Task):
    """A task whose episode is exactly `horizon` tokens and whose state is the prefix.

    Subclasses define `prompts()` and `correct_solutions(prompt_id)`; verification, dynamics,
    demonstrations and the exact table follow from those.
    """

    def correct_solutions(self, prompt_id: int) -> Sequence[tuple]:
        raise NotImplementedError

    def verify(self, prompt_id: int, actions: Sequence[int]) -> bool:
        return tuple(int(a) for a in actions) in self._correct_set(prompt_id)

    def initial_state(self, prompt_id: int) -> Hashable:
        return ()

    def step(self, prompt_id: int, state: Hashable, action: int) -> tuple[Hashable, bool]:
        nxt = tuple(state) + (int(action),)  # type: ignore[arg-type]
        return nxt, len(nxt) == self.horizon

    def demonstrations(self, prompt_id: int) -> Sequence[tuple]:
        return tuple(self.correct_solutions(prompt_id))

    def table(self, prompt_id: int) -> TrajectoryTable:
        cache = self.__dict__.setdefault("_table_cache", {})
        if prompt_id not in cache:
            cache[prompt_id] = compile_prefix_tree(self.n_actions, self.horizon, self.correct_solutions(prompt_id))
        return cache[prompt_id]

    def _correct_set(self, prompt_id: int) -> frozenset:
        cache = self.__dict__.setdefault("_correct_cache", {})
        if prompt_id not in cache:
            cache[prompt_id] = frozenset(tuple(int(t) for t in s) for s in self.correct_solutions(prompt_id))
        return cache[prompt_id]
