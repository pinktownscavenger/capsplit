"""Control arm: no post-training. The policy is never updated, so every gain is exactly zero."""
from __future__ import annotations

from ..interfaces import Trainer
from ..registry import TRAINERS


@TRAINERS.register("none")
class NoTraining(Trainer):
    def train(self, policy, task, budget, seed, prompt_ids, checkpoints, on_checkpoint) -> None:
        for step in sorted(checkpoints):
            if 0 <= step <= budget:
                on_checkpoint(step, policy)
