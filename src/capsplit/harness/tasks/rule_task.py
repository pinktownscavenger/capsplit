"""The rule task: prompt-conditioned token sequences with an easy rule and a hard rule (H9).

A prompt is (type, a, b) with type 0 = easy, 1 = hard and a, b in 0..V-1, so there are
2 * V * V prompts. Every prompt has exactly two correct length-L sequences, identical on
positions 0..L-2 and ending in r and r + 1 (mod V), where r is the rule's last token.

* Easy rule:  e_i(a, b) = (a + i * b + i * (i + 1) / 2) mod V. Its step e_{i+1} - e_i = b + i + 1
  changes every position, so no easy-rule sequence is constant or periodic with period 1
  (the plain (a + i * b) rule gives constant sequences at b = 0, which the network fits
  measurably differently); (a, b) is recoverable from (e_0, e_1).
* Hard rule:  h_i(a, b) = (e_i(a, b) + d_i(a, b)) mod V with d_i = 1 + ((b + i * a + 1) mod (V - 2)).
  The offset d_i lies in 1..V-2, so h_i != e_i and h_i + 1 != e_i at every position: a hard
  solution can never coincide with the easy-rule sequence at a position >= k.
* Easy prompts follow the easy rule everywhere. Hard prompts follow the easy rule on positions
  < k and the hard rule on positions >= k, where k = `shared_prefix` is the shared-structure
  dial (0 <= k <= L - 1). So each hard solution shares exactly its first k tokens with the
  prompt's easy-rule sequence and differs from it at position k.

Traps (H10): the trap of a hard prompt is its easy-rule sequence (the easy skill misapplied);
the trap of an easy prompt is a random wrong sequence drawn from the task seed. Traps are
never correct. They are part of the task so that tests and the base recipe agree on them.

A seeded third of each type's prompts is held out (tags "split": "train" / "heldout").
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..interfaces import Prompt
from ..registry import TASKS
from ..sequence import SequenceTask

EASY, HARD = 0, 1
_TYPE_NAMES = ("easy", "hard")


@TASKS.register("rule_task")
class RuleTask(SequenceTask):
    def __init__(
        self,
        vocab: int = 6,
        length: int = 5,
        shared_prefix: int = 2,
        seed: int = 0,
        heldout_fraction: float = 1 / 3,
    ):
        if vocab < 3:
            raise ValueError(f"need vocab >= 3 so the hard offset range 1..V-2 is non-empty, got {vocab}")
        if length < 2:
            raise ValueError(f"need length >= 2 (two solutions differ in the last token), got {length}")
        if not 0 <= shared_prefix <= length - 1:
            raise ValueError(f"shared_prefix must be in 0..{length - 1}, got {shared_prefix}")
        if not 0.0 <= heldout_fraction < 1.0:
            raise ValueError(f"heldout_fraction must be in [0, 1), got {heldout_fraction}")
        self.n_actions, self.horizon = vocab, length
        self.vocab, self.length = vocab, length
        self.shared_prefix, self.seed = shared_prefix, seed
        # Cardinality of each prompt token, for prompt-conditioned policies.
        self.prompt_sizes = (2, vocab, vocab)

        rng = np.random.default_rng([seed, 0x5EED])
        per_type = vocab * vocab
        n_heldout = int(round(heldout_fraction * per_type))
        heldout = {kind: set(rng.permutation(per_type)[:n_heldout].tolist()) for kind in (EASY, HARD)}
        prompts: list[Prompt] = []
        for kind in (EASY, HARD):
            for a in range(vocab):
                for b in range(vocab):
                    split = "heldout" if a * vocab + b in heldout[kind] else "train"
                    tags = {"difficulty": _TYPE_NAMES[kind], "split": split}
                    prompts.append(Prompt(len(prompts), (kind, a, b), tags))
        self._prompts = tuple(prompts)

        # Easy traps: one random wrong sequence per easy prompt, from the task seed.
        self._easy_traps: dict[int, tuple] = {}
        for prompt in self._prompts:
            if prompt.tokens[0] == EASY:
                correct = set(self.correct_solutions(prompt.id))
                while True:
                    trap = tuple(int(t) for t in rng.integers(0, vocab, length))
                    if trap not in correct:
                        break
                self._easy_traps[prompt.id] = trap

    # ------------------------------------------------------------------ rules

    def easy_rule(self, a: int, b: int) -> tuple:
        return tuple((a + i * b + i * (i + 1) // 2) % self.vocab for i in range(self.length))

    def hard_rule(self, a: int, b: int) -> tuple:
        offsets = (1 + ((b + i * a + 1) % (self.vocab - 2)) for i in range(self.length))
        return tuple((e + d) % self.vocab for e, d in zip(self.easy_rule(a, b), offsets))

    def easy_rule_sequence(self, prompt_id: int) -> tuple:
        """The easy rule applied at every position of this prompt (for any prompt type)."""
        _, a, b = self._prompts[prompt_id].tokens
        return self.easy_rule(a, b)

    def is_hard(self, prompt_id: int) -> bool:
        return self._prompts[prompt_id].tokens[0] == HARD

    # ------------------------------------------------------------------ Task API

    def prompts(self) -> Sequence[Prompt]:
        return self._prompts

    def correct_solutions(self, prompt_id: int) -> Sequence[tuple]:
        kind, a, b = self._prompts[prompt_id].tokens
        easy = self.easy_rule(a, b)
        if kind == EASY:
            body = easy
        else:
            hard = self.hard_rule(a, b)
            k = self.shared_prefix
            body = easy[:k] + hard[k:]
        last = body[-1]
        return (body, body[:-1] + ((last + 1) % self.vocab,))

    def trap(self, prompt_id: int) -> tuple:
        """The incorrect sequence the base is trained to prefer (H10)."""
        if self.is_hard(prompt_id):
            return self.easy_rule_sequence(prompt_id)
        return self._easy_traps[prompt_id]
