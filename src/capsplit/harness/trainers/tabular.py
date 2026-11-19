"""REINFORCE with a moving-average baseline (`tabular_reinforce`), generic over the Policy interface.

RL framing (as `interfaces.py`): an episode is sampled with `policy.sample`, its reward is
r = 1[task.verify] (terminal, binary, so return = reward), and each update is one optimizer step
on the REINFORCE loss

    loss = -mean_j (r_j - b_{p(j)}) * log pi_theta(tau_j)

where log pi_theta(tau_j) sums over generated steps. The baseline b_p is a per-prompt
exponential moving average of past mean rewards on prompt p: it starts at 0 and is updated
*after* each step, b_p <- decay * b_p + (1 - decay) * mean_{j on p} r_j, so it never includes
the current batch (unbiased gradient), as `capsplit.train.tabular.reinforce`. With
`per_prompt_baseline=False` one scalar baseline is shared by all prompts.

Each update samples `batch` episodes for every training prompt. Despite the name (it is the
grid project's tabular REINFORCE), it only uses `Policy.sample`, so it runs on any policy.

Params (unknown ones raise, H14): batch (16), lr (Adam 0.05 / SGD 0.5), optimizer ("adam" or
"sgd"), baseline_decay (0.9), per_prompt_baseline (True), only_tags (None = all given prompts).
All randomness comes from a torch.Generator seeded with the run seed.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch

from ..interfaces import CheckpointHook, Policy, Task, Trainer, call_checkpoints
from ..registry import TRAINERS
from .rl import episode_rewards, select_prompts

_PARAMS = {"batch", "lr", "optimizer", "baseline_decay", "per_prompt_baseline", "only_tags"}
_DEFAULT_LR = {"adam": 0.05, "sgd": 0.5}


@TRAINERS.register("tabular_reinforce")
class TabularReinforce(Trainer):
    def __init__(self, **params: Any):
        unknown = set(params) - _PARAMS
        if unknown:
            raise ValueError(f"tabular_reinforce: unknown params {sorted(unknown)}")
        super().__init__(**params)
        self.batch = int(params.get("batch", 16))
        self.optimizer = str(params.get("optimizer", "adam")).lower()
        if self.optimizer not in _DEFAULT_LR:
            raise ValueError(f"optimizer must be 'adam' or 'sgd', got {self.optimizer!r}")
        self.lr = float(params.get("lr", _DEFAULT_LR[self.optimizer]))
        self.decay = float(params.get("baseline_decay", 0.9))
        self.per_prompt = bool(params.get("per_prompt_baseline", True))
        self.only_tags = params.get("only_tags")
        if self.batch < 1:
            raise ValueError("batch must be at least 1")
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError("baseline_decay must lie in [0, 1]")

    def train(
        self,
        policy: Policy,
        task: Task,
        budget: int,
        seed: int,
        prompt_ids: Sequence[int],
        checkpoints: Sequence[int],
        on_checkpoint: CheckpointHook,
    ) -> None:
        ids = select_prompts(task, prompt_ids, self.only_tags)
        call_checkpoints(0, checkpoints, policy, on_checkpoint)
        if budget <= 0:
            return
        if not ids:
            raise ValueError("no training prompts left after applying only_tags")
        generator = torch.Generator().manual_seed(int(seed))
        params = list(policy.parameters())
        optimizer: torch.optim.Optimizer
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(params, lr=self.lr)
        else:
            optimizer = torch.optim.SGD(params, lr=self.lr)
        baseline = torch.zeros(len(ids))
        episode_ids = [p for p in ids for _ in range(self.batch)]
        slot = torch.arange(len(ids)).repeat_interleave(self.batch)
        for step in range(1, budget + 1):
            rollout = policy.sample(task, episode_ids, generator)
            rewards = episode_rewards(task, rollout)
            advantages = rewards - baseline[slot]
            loss = -(advantages * rollout.log_prob).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            means = rewards.view(len(ids), self.batch).mean(dim=1)
            if not self.per_prompt:
                means = means.mean().expand(len(ids))
            baseline = self.decay * baseline + (1.0 - self.decay) * means
            call_checkpoints(step, checkpoints, policy, on_checkpoint)
