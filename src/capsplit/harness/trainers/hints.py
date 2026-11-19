"""Hint arm: on-policy RLOO that re-rolls all-fail groups from a prefix of a correct solution.

RL framing. The reward is the verifier's outcome, 1 if the finished episode is correct and 0
otherwise. Each update samples a group of G episodes per training prompt from the current
policy and uses the RLOO estimator: episode i's advantage is its reward minus the mean
reward of the other G - 1 episodes in its group. When every episode in a group fails, all
advantages are zero and plain RLOO gets no gradient from that prompt. The hint arm replaces
such a group with a *hinted re-roll*: one teacher demonstration of the prompt is chosen with
the seeded generator, its first `hint_len` actions are forced, and the rest of each of the G
episodes is sampled on-policy. Rewards still come from the verifier, advantages are again
leave-one-out within the (hinted) group, and the policy-gradient loss uses only the
generated actions: the forced prefix is excluded from `Rollout.log_prob` and `entropy`
(DECISIONS H4), so the policy gets no gradient for tokens it did not choose.

This is "hint-guided" or "prefix-guided" RL. NuRL (arXiv 2509.25666) is related work that
also conditions on hints for problems the policy cannot solve; this arm is a simple variant
in the same spirit and does not reproduce it.

No hints at evaluation: the hints exist only inside `train()`. The policy that is measured
afterwards is the ordinary, unconditioned policy pi(a | s).

Parameters
----------
hint_len : int, required. Number of forced demonstration actions, 1 <= hint_len < horizon.
    Different hint lengths are separate arm configs.
group : int, default 16. Episodes per prompt per update (G >= 2).
lr : float, default 1e-3. Adam learning rate; one optimizer per `train()` call.
entropy_coef : float, default 0. Weight of the entropy bonus (over generated steps).
only_tags : dict or None. If given, train only on prompt ids whose tags match every key=value.
hint_when : "all_fail" (default) or "always" (hint every group; for tests).

Each update is exactly one optimizer step. `self.history` holds one summary dict per update
(how many groups were hinted, which prompts, mean rewards), and `self.last_episodes` the
episodes of the most recent update.

`training_log()` returns the `rl.SampleLog` of the last `train()` call with hint fields:
"first_success_update" counts only the un-hinted on-policy groups, "first_hinted_success_update"
only the hinted re-rolls; episode and token counts include both rolls (forced hint tokens are
not counted as sampled tokens).
"""
from __future__ import annotations

from typing import Any, Sequence

import torch

from ..interfaces import CheckpointHook, Policy, Task, Trainer, call_checkpoints
from ..registry import TRAINERS
from .rl import SampleLog

_HINT_WHEN = ("all_fail", "always")


def loo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """(n_groups, G) leave-one-out advantages: r_i minus the mean reward of the other G - 1."""
    g = rewards.shape[-1]
    if g < 2:
        raise ValueError(f"leave-one-out needs a group of at least 2, got {g}")
    return (g * rewards - rewards.sum(dim=-1, keepdim=True)) / (g - 1)


@TRAINERS.register("hints")
class HintedRLOO(Trainer):
    def __init__(self, **params: Any):
        super().__init__(**params)
        if "hint_len" not in params:
            raise ValueError("hints trainer needs hint_len")
        hint_len = params["hint_len"]
        if isinstance(hint_len, bool) or not isinstance(hint_len, int) or hint_len < 1:
            raise ValueError(f"hint_len must be an integer >= 1, got {hint_len!r}")
        self.hint_len: int = hint_len
        self.group = int(params.get("group", 16))
        if self.group < 2:
            raise ValueError(f"group must be at least 2, got {self.group}")
        self.lr = float(params.get("lr", 1e-3))
        self.entropy_coef = float(params.get("entropy_coef", 0.0))
        self.only_tags: dict | None = params.get("only_tags")
        self.hint_when = params.get("hint_when", "all_fail")
        if self.hint_when not in _HINT_WHEN:
            raise ValueError(f"hint_when must be one of {_HINT_WHEN}, got {self.hint_when!r}")
        self.history: list[dict] = []
        self.last_episodes: list[dict] = []
        self.sample_log: SampleLog | None = None

    def training_log(self) -> dict | None:
        """What the last `train()` call sampled (see `rl.SampleLog`); None before any call."""
        return None if self.sample_log is None else self.sample_log.to_dict()

    def _train_ids(self, task: Task, prompt_ids: Sequence[int]) -> list[int]:
        ids = list(prompt_ids)
        if self.only_tags:
            tags = {p.id: p.tags for p in task.prompts()}
            ids = [p for p in ids if all(tags[p].get(k) == v for k, v in self.only_tags.items())]
        if not ids:
            raise ValueError("no training prompts left after the only_tags filter")
        return ids

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
        if self.hint_len >= task.horizon:
            raise ValueError(f"hint_len must be < horizon {task.horizon}, got {self.hint_len}")
        ids = self._train_ids(task, prompt_ids)
        demos = {p: [tuple(d) for d in task.demonstrations(p) if len(d) >= self.hint_len] for p in ids}
        generator = torch.Generator().manual_seed(seed)
        optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr)
        g = self.group
        self.history = []
        self.last_episodes = []
        self.sample_log = log = SampleLog(ids, checkpoints, hints=True)

        call_checkpoints(0, checkpoints, policy, on_checkpoint)
        for step in range(1, budget + 1):
            batch_ids = [p for p in ids for _ in range(g)]
            rollout = policy.sample(task, batch_ids, generator)
            rewards = torch.tensor([float(task.verify(p, a)) for p, a in zip(batch_ids, rollout.actions)])
            log.record(step, rollout, rewards)
            rewards = rewards.view(len(ids), g)

            # Groups to re-roll from a demonstration prefix (prompts with no usable demo are not hinted).
            hint_rows = [
                i
                for i, p in enumerate(ids)
                if demos[p] and (self.hint_when == "always" or float(rewards[i].sum()) == 0.0)
            ]
            log_prob = rollout.log_prob.view(len(ids), g)
            entropy = rollout.entropy.view(len(ids), g)
            actions = [rollout.actions[i * g : (i + 1) * g] for i in range(len(ids))]
            n_forced = [rollout.n_forced[i * g : (i + 1) * g] for i in range(len(ids))]
            unhinted_mean = float(rewards.mean())

            if hint_rows:
                choice = [int(torch.randint(len(demos[ids[i]]), (1,), generator=generator)) for i in hint_rows]
                prefixes = [demos[ids[i]][c][: self.hint_len] for i, c in zip(hint_rows, choice)]
                hint_ids = [ids[i] for i in hint_rows for _ in range(g)]
                hinted = policy.sample(task, hint_ids, generator, prefixes=[pre for pre in prefixes for _ in range(g)])
                hint_rewards = torch.tensor([float(task.verify(p, a)) for p, a in zip(hint_ids, hinted.actions)])
                log.record(step, hinted, hint_rewards, hinted=True)
                rows = torch.tensor(hint_rows)
                mask = torch.zeros(len(ids), 1, dtype=torch.bool)
                mask[rows] = True
                # Rebuild the tensors so the replaced groups carry the hinted rollout's graph.
                scatter = torch.full((len(ids),), -1, dtype=torch.long)
                scatter[rows] = torch.arange(len(hint_rows))
                hl = hinted.log_prob.view(len(hint_rows), g)
                he = hinted.entropy.view(len(hint_rows), g)
                hr = hint_rewards.view(len(hint_rows), g)
                pick = scatter.clamp(min=0)
                log_prob = torch.where(mask, hl[pick], log_prob)
                entropy = torch.where(mask, he[pick], entropy)
                rewards = torch.where(mask, hr[pick], rewards)
                for k, i in enumerate(hint_rows):
                    actions[i] = hinted.actions[k * g : (k + 1) * g]
                    n_forced[i] = hinted.n_forced[k * g : (k + 1) * g]

            advantages = loo_advantages(rewards)
            loss = -(advantages * log_prob).mean() - self.entropy_coef * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            hinted_set = set(hint_rows)
            self.history.append(
                {
                    "step": step,
                    "n_groups": len(ids),
                    "n_hinted": len(hint_rows),
                    "hinted_prompts": [ids[i] for i in hint_rows],
                    "mean_reward_unhinted": unhinted_mean,
                    "mean_reward_hinted": float(rewards[hint_rows].mean()) if hint_rows else None,
                    "loss": float(loss.detach()),
                }
            )
            self.last_episodes = [
                {
                    "prompt_id": p,
                    "actions": tuple(actions[i][j]),
                    "n_forced": int(n_forced[i][j]),
                    "hinted": i in hinted_set,
                    "reward": float(rewards[i, j]),
                }
                for i, p in enumerate(ids)
                for j in range(g)
            ]
            call_checkpoints(step, checkpoints, policy, on_checkpoint)

    @property
    def hinted_per_update(self) -> list[int]:
        """Number of hinted groups in each update of the last `train()` call."""
        return [h["n_hinted"] for h in self.history]
