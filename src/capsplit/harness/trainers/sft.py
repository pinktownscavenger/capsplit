"""Supervised fine-tuning (SFT) arm: behaviour cloning of teacher demonstrations.

RL framing. This arm does **not** use the reward. It never samples from the policy and never
calls `Task.verify`. Instead it is maximum-likelihood imitation of correct trajectories that a
teacher shows (`Task.demonstrations`), i.e. behaviour cloning, the "SFT" contrast to RL. The
update rule is gradient ascent on the log-likelihood of demonstrated trajectories:

    theta <- theta + lr * grad_theta  mean_{(x, tau) in batch} log pi_theta(tau | x)

implemented as one Adam step on the loss  L = -mean log pi_theta(tau | x), where
log pi_theta(tau | x) = sum_t log pi_theta(a_t | s_t) is the teacher-forced log-probability
(`Policy.log_prob`).

Which prompts. By default the arm trains on the **hard train prompts** only: the prompts in
`prompt_ids` (the train split it is given) whose tags match `only_tags`. It never trains on a
prompt outside `prompt_ids`.

Parameters (all optional):

* ``only_tags`` (dict, default ``{"difficulty": "hard"}``): keep prompts whose tags match every
  key=value. ``None`` or ``{}`` keeps every prompt in `prompt_ids`.
* ``batch`` (int, default all pairs): number of (prompt, demonstration) pairs per update. If it
  is smaller than the number of pairs, each update uses a minibatch drawn uniformly without
  replacement by a generator seeded from `seed`; otherwise every update is full-batch.
* ``lr`` (float, default 1e-3): Adam learning rate. Adam is created once per `train` call.
* ``demos_per_prompt`` (int, default all): use the first n of `task.demonstrations(pid)`.

Reusable helpers (imported by the SFT->RL arm):

* ``demonstration_pairs(task, prompt_ids, only_tags=DEFAULT_ONLY_TAGS, demos_per_prompt=None)
  -> list[tuple[int, tuple]]``
* ``sft_loss(policy, task, pairs) -> torch.Tensor`` (scalar, with gradients)
* ``sft_step(policy, task, optimizer, pairs) -> float`` (one optimizer step; returns the loss)
* ``minibatch(pairs, batch, generator) -> list[tuple[int, tuple]]``

`training_log()` returns an `rl.SampleLog` for the imitated prompts: SFT samples nothing, so
every count is zero and every first_success_update is null.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from ..interfaces import Policy, Task, Trainer, call_checkpoints
from ..registry import TRAINERS
from .rl import SampleLog

DEFAULT_ONLY_TAGS: Mapping[str, str] = {"difficulty": "hard"}

Pair = tuple[int, tuple]


def demonstration_pairs(
    task: Task,
    prompt_ids: Sequence[int],
    only_tags: Mapping[str, str] | None = DEFAULT_ONLY_TAGS,
    demos_per_prompt: int | None = None,
) -> list[Pair]:
    """(prompt id, demonstration) pairs for the prompts in `prompt_ids` whose tags match `only_tags`.

    Order follows `prompt_ids` (duplicates removed), then `task.demonstrations(pid)`. Only
    prompts in `prompt_ids` are ever used. Raises ValueError if no prompt passes the filter or
    the kept prompts have no demonstrations.
    """
    tags = dict(only_tags or {})
    prompts = task.prompts()
    seen: set[int] = set()
    kept: list[int] = []
    for pid in prompt_ids:
        pid = int(pid)
        if pid in seen:
            continue
        seen.add(pid)
        if all(prompts[pid].tags.get(k) == v for k, v in tags.items()):
            kept.append(pid)
    if not kept:
        raise ValueError(
            f"SFT: no prompt among the {len(seen)} training prompts matches only_tags={tags}; nothing to imitate"
        )
    if demos_per_prompt is not None and demos_per_prompt < 1:
        raise ValueError(f"SFT: demos_per_prompt must be >= 1, got {demos_per_prompt}")
    pairs: list[Pair] = []
    for pid in kept:
        demos = list(task.demonstrations(pid))
        if demos_per_prompt is not None:
            demos = demos[:demos_per_prompt]
        pairs.extend((pid, tuple(int(a) for a in demo)) for demo in demos)
    if not pairs:
        raise ValueError(f"SFT: the {len(kept)} selected prompts have no demonstrations")
    return pairs


def sft_loss(policy: Policy, task: Task, pairs: Sequence[Pair]) -> torch.Tensor:
    """Scalar loss -mean over `pairs` of the teacher-forced log pi(demonstration | prompt)."""
    if not pairs:
        raise ValueError("SFT: empty batch")
    pids = [pid for pid, _ in pairs]
    demos = [demo for _, demo in pairs]
    return -policy.log_prob(task, pids, demos).mean()


def sft_step(policy: Policy, task: Task, optimizer: torch.optim.Optimizer, pairs: Sequence[Pair]) -> float:
    """One behaviour-cloning update on `pairs`: zero grads, backprop `sft_loss`, optimizer step.

    Returns the loss before the step.
    """
    optimizer.zero_grad()
    loss = sft_loss(policy, task, pairs)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def minibatch(pairs: Sequence[Pair], batch: int | None, generator: torch.Generator) -> list[Pair]:
    """`batch` pairs drawn without replacement (all pairs, in order, if batch is None or >= len)."""
    if batch is None or batch >= len(pairs):
        return list(pairs)
    order = torch.randperm(len(pairs), generator=generator)[:batch].tolist()
    return [pairs[i] for i in order]


_KNOWN_PARAMS = {"only_tags", "batch", "lr", "demos_per_prompt"}


@TRAINERS.register("sft")
class SFT(Trainer):
    """Behaviour cloning on teacher demonstrations (no reward). See the module docstring."""

    name = "sft"

    def __init__(self, **params: Any):
        unknown = set(params) - _KNOWN_PARAMS
        if unknown:
            raise ValueError(f"SFT: unknown parameters {sorted(unknown)}; known: {sorted(_KNOWN_PARAMS)}")
        super().__init__(**params)
        self.only_tags: Mapping[str, str] | None = params.get("only_tags", DEFAULT_ONLY_TAGS)
        self.batch: int | None = params.get("batch")
        self.lr: float = float(params.get("lr", 1e-3))
        self.demos_per_prompt: int | None = params.get("demos_per_prompt")
        if self.batch is not None and int(self.batch) < 1:
            raise ValueError(f"SFT: batch must be >= 1, got {self.batch}")
        self.sample_log: SampleLog | None = None

    def training_log(self) -> dict | None:
        """The (all-zero) sample log of the last `train` call; None before any call."""
        return None if self.sample_log is None else self.sample_log.to_dict()

    def train(self, policy, task, budget, seed, prompt_ids, checkpoints, on_checkpoint) -> None:
        pairs = demonstration_pairs(task, prompt_ids, self.only_tags, self.demos_per_prompt)
        self.sample_log = SampleLog([pid for pid, _ in pairs], checkpoints)
        self.sample_log.updates = max(int(budget), 0)
        generator = torch.Generator().manual_seed(int(seed))
        optimizer = torch.optim.Adam(list(policy.parameters()), lr=self.lr)
        call_checkpoints(0, checkpoints, policy, on_checkpoint)
        for step in range(1, int(budget) + 1):
            sft_step(policy, task, optimizer, minibatch(pairs, self.batch, generator))
            call_checkpoints(step, checkpoints, policy, on_checkpoint)
