"""Fixed-length token sequences with a seeded accepted set, compiled to a trajectory table."""
from dataclasses import dataclass

import numpy as np

from capsplit.engine import TrajectoryTable


@dataclass(frozen=True)
class SeqTask:
    vocab: int
    length: int
    common: tuple  # tuple of token tuples
    rare: tuple
    distractors: tuple
    shared_prefix: int

    def is_accepted(self, seq: tuple) -> bool:
        return seq in self.common or seq in self.rare


def _digits(codes: np.ndarray, base: int, length: int) -> np.ndarray:
    """Base-`base` digits of each code, most significant first: (len(codes), length)."""
    powers = base ** np.arange(length - 1, -1, -1)
    return (np.asarray(codes)[:, None] // powers) % base


def _sample_sequences(rng: np.random.Generator, tokens: np.ndarray, length: int, count: int) -> list[tuple]:
    """`count` distinct sequences over `tokens`."""
    pool = len(tokens) ** length
    if count > pool:
        raise ValueError(f"asked for {count} sequences, but only {pool} exist")
    codes = rng.choice(pool, size=count, replace=False)
    return [tuple(row) for row in tokens[_digits(codes, len(tokens), length)].tolist()]


def _sample_rare(rng, upper, common, n_rare, length, k) -> list[tuple]:
    """Each rare sequence keeps a common prefix and ends in upper-half tokens, distinct per prefix."""
    by_prefix: dict = {}
    for i in range(n_rare):
        by_prefix.setdefault(common[i % len(common)][:k], []).append(i)
    rare: list = [None] * n_rare
    for prefix, members in by_prefix.items():
        for i, suffix in zip(members, _sample_sequences(rng, upper, length - k, len(members))):
            rare[i] = prefix + suffix
    return rare


def make_task(vocab, length, n_common, n_rare, shared_prefix, seed, n_distractors: int = 30) -> SeqTask:
    if vocab < 2 or length < 1:
        raise ValueError(f"need vocab >= 2 and length >= 1, got {vocab} and {length}")
    if not 0 <= shared_prefix <= length - 1:
        raise ValueError(f"shared_prefix must be in 0..{length - 1}, got {shared_prefix}")
    if min(n_common, n_rare, n_distractors) < 0:
        raise ValueError("sequence counts can't be negative")
    if n_rare > 0 and n_common == 0:
        raise ValueError("rare sequences need a common sequence to copy")

    rng = np.random.default_rng(seed)
    tokens = np.arange(vocab)
    lower, upper = tokens[: vocab // 2], tokens[vocab // 2 :]
    common = _sample_sequences(rng, lower, length, n_common)
    rare = _sample_rare(rng, upper, common, n_rare, length, shared_prefix)

    powers = vocab ** np.arange(length - 1, -1, -1)
    accepted = [int(np.dot(sequence, powers)) for sequence in common + rare]
    candidates = np.setdiff1d(np.arange(vocab**length), accepted)
    if n_distractors > len(candidates):
        raise ValueError(f"asked for {n_distractors} distractors, but only {len(candidates)} sequences remain")
    distractors = _digits(rng.choice(candidates, size=n_distractors, replace=False), vocab, length)

    return SeqTask(
        vocab=vocab,
        length=length,
        common=tuple(common),
        rare=tuple(rare),
        distractors=tuple(tuple(row) for row in distractors.tolist()),
        shared_prefix=shared_prefix,
    )


def make_benchmark(vocab, length, prompts, n_common, n_rare, shared_prefix, seed) -> list[SeqTask]:
    return [make_task(vocab, length, n_common, n_rare, shared_prefix, seed + i) for i in range(prompts)]


def node_offsets(vocab: int, length: int) -> np.ndarray:
    """First node id of each level, where level l holds the vocab**l prefixes of length l."""
    return np.concatenate(([0], np.cumsum(vocab ** np.arange(length - 1))))


def compile_sequence(task: SeqTask) -> TrajectoryTable:
    vocab, length = task.vocab, task.length
    offsets = node_offsets(vocab, length)
    codes = np.arange(vocab**length)
    powers = vocab ** np.arange(length - 1, -1, -1)
    prefixes = codes[:, None] // (powers * vocab)  # base-vocab value of the tokens before each position
    tokens = (codes[:, None] // powers) % vocab

    solutions = tuple(sorted(task.common + task.rare))
    solution_ids = np.full(len(codes), -1, dtype=np.int32)
    if solutions:
        solution_ids[np.dot(solutions, powers)] = np.arange(len(solutions))

    return TrajectoryTable(
        index=((offsets + prefixes) * vocab + tokens).astype(np.int32),
        solution_ids=solution_ids,
        solutions=solutions,
        n_states=int(offsets[-1] + vocab ** (length - 1)),
        n_actions=vocab,
        horizon=length,
    )
