"""Supervised pre-training, and the search for the rare weight that sets the rare mass under the base."""
import math

import numpy as np
import torch

from capsplit.engine import solution_probs, trajectory_probs

from .policy import SequencePolicy, node_logits
from .verifier import SeqTask, compile_sequence

_COMMON_TOTAL, _DISTRACTOR_TOTAL = 0.25, 0.75
_CALIBRATION_STEPS = 8
_LOG10_WEIGHT_RANGE = (-8.0, -1.0)


def sequence_weights(task: SeqTask, rare_weight: float) -> dict:
    weights = {sequence: _COMMON_TOTAL / len(task.common) for sequence in task.common}
    weights.update({sequence: _DISTRACTOR_TOTAL / len(task.distractors) for sequence in task.distractors})
    weights.update({sequence: rare_weight for sequence in task.rare})
    return weights


def pretrain(policy: SequencePolicy, task: SeqTask, rare_weight: float, steps: int = 300, lr: float = 3e-3) -> None:
    """Full-batch weighted maximum likelihood on the task's sequences."""
    weights = sequence_weights(task, rare_weight)
    targets = torch.tensor(list(weights))
    inputs = torch.cat((torch.full((len(targets), 1), task.vocab), targets[:, :-1]), dim=1)
    sequence_weight = torch.tensor(list(weights.values()), dtype=torch.float32)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(steps):
        log_probs = torch.log_softmax(policy(inputs), dim=-1).gather(2, targets.unsqueeze(2)).squeeze(2)
        loss = -(sequence_weight * log_probs.sum(dim=1)).sum() / sequence_weight.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def _median_rare_q(task: SeqTask, table, policy: SequencePolicy) -> float:
    q = solution_probs(table, trajectory_probs(table, node_logits(policy, task.vocab, task.length)))
    return float(np.median([q[table.solutions.index(sequence)] for sequence in task.rare]))


def calibrate(task: SeqTask, n: int, seed: int, target_ratio: float = 0.2) -> tuple[SequencePolicy, dict]:
    """Bisect log10 of the rare weight to put median rare q under the base at target_ratio / n.

    Returns the policy that landed closest, with `rare_weight` and `median_rare_q`.
    """
    table = compile_sequence(task)
    target = target_ratio / n
    low, high = _LOG10_WEIGHT_RANGE
    best = None
    for _ in range(_CALIBRATION_STEPS):
        middle = (low + high) / 2
        policy = SequencePolicy(task.vocab, seed=seed)
        pretrain(policy, task, rare_weight=10.0**middle)
        median = _median_rare_q(task, table, policy)
        distance = abs(math.log(median / target))
        if best is None or distance < best[0]:
            best = (distance, policy, {"rare_weight": 10.0**middle, "median_rare_q": median})
        if median < target:
            low = middle
        else:
            high = middle
    return best[1], best[2]
