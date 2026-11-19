"""Reinforcement-learning post-training arms: RLOO, GRPO and RLOO with an entropy bonus.

RL framing (shared with `interfaces.py`):

* **State** s_t: the prompt plus the actions taken so far (for sequence tasks, the prefix).
* **Action** a_t: one of `task.n_actions` choices, drawn from the policy pi_theta(. | s_t).
* **Reward**: terminal and binary, r = 1.0 if `task.verify(prompt_id, actions)` else 0.0.
* **Return**: undiscounted sum of rewards = r, so the expected return on a prompt is the
  success probability J(theta) and the objective is the mean of J over training prompts.
* **Update rule**: a REINFORCE policy-gradient step with a group baseline. For each training
  prompt we sample a group of G episodes from the current policy, compute an advantage A_j
  for each episode from the group's rewards, and take one Adam step on

      loss = -mean_j(A_j * log pi_theta(tau_j)) - entropy_coef * mean_j(H_j)

  where log pi_theta(tau_j) = sum_t log pi_theta(a_t | s_t) over generated steps and H_j is
  the summed per-step policy entropy along tau_j (both from `Policy.sample`). Advantages
  are constants (no gradient flows through them).

The arms differ only in the advantage (see `group_advantages`):

* ``rloo``: leave-one-out baseline, A_j = r_j - mean_{i != j} r_i. Unbiased.
* ``grpo``: group-normalised, A_j = (r_j - mean(r)) / (std(r) + eps), population std.
* ``rl_entropy``: RLOO with entropy_coef = 0.01 by default (the entropy-bonus control).

Both baselines give exactly zero advantage when every episode in a group has the same
reward, so a prompt on which every sample fails (or every sample succeeds) contributes
exactly zero policy gradient. Only the entropy bonus, when on, moves the policy there.

The helpers `episode_rewards`, `group_advantages`, `rl_loss` and `select_prompts` are public
so the hint arm and the SFT-then-RL arm can reuse them.

Training log. After `train`, `training_log()` returns a JSON-ready dict describing what the
arm sampled (see `SampleLog`); the runner writes it to `<cell>/seed-<s>.training.json`.
Logging only reads the sampled episodes and their rewards; it draws no random numbers, so a
run's training and measurements are unchanged by it.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from ..interfaces import CheckpointHook, Policy, Rollout, Task, Trainer, call_checkpoints
from ..registry import TRAINERS

__all__ = [
    "ADVANTAGE_KINDS",
    "GRPO",
    "RLOO",
    "EntropyRLOO",
    "SampleLog",
    "episode_rewards",
    "group_advantages",
    "rl_loss",
    "select_prompts",
]

ADVANTAGE_KINDS = ("rloo", "grpo")


class SampleLog:
    """Per-prompt record of the episodes a trainer sampled; `to_dict()` gives the training log.

    Updates are numbered 1..budget (update k is the k-th optimizer step). Log schema::

        {"updates": int,                       # optimizer updates the log covers
         "checkpoints": [int],                 # the checkpoints passed to train()
         "prompts": {"<prompt id>": {
             "first_success_update": int | null,   # first update whose sampled group held a
                                                   # correct (un-hinted) episode
             "correct_by_checkpoint": {"<c>": int},# correct sampled episodes in updates 1..c
             "episodes_sampled": int,
             "tokens_sampled": int}},              # generated actions; forced hint tokens excluded
         "totals": {"episodes_sampled", "tokens_sampled", "correct_by_checkpoint"}}

    With `hints=True` (the hint arm) each prompt and the totals also carry
    "first_hinted_success_update" (per prompt), "hinted_episodes_sampled" and
    "hinted_correct_by_checkpoint"; `correct_by_checkpoint` and the episode/token counts then
    include the hinted re-rolls, while "first_success_update" counts un-hinted episodes only.
    """

    def __init__(self, prompt_ids: Sequence[int], checkpoints: Sequence[int], hints: bool = False):
        self.prompt_ids: list[int] = []
        self.checkpoints = sorted({int(c) for c in checkpoints})
        self.hints = hints
        self.updates = 0
        self._first: dict[int, int | None] = {}
        self._first_hinted: dict[int, int | None] = {}
        self._episodes: dict[int, int] = {}
        self._tokens: dict[int, int] = {}
        self._hinted_episodes: dict[int, int] = {}
        # prompt -> {update: correct count}; kept per update so phases can be merged.
        self._correct: dict[int, dict[int, int]] = {}
        self._hinted_correct: dict[int, dict[int, int]] = {}
        for pid in prompt_ids:
            self._ensure(int(pid))

    def _ensure(self, pid: int) -> None:
        if pid in self._first:
            return
        self.prompt_ids = sorted(set(self.prompt_ids) | {pid})
        self._first[pid] = self._first_hinted[pid] = None
        self._episodes[pid] = self._tokens[pid] = self._hinted_episodes[pid] = 0
        self._correct[pid] = {}
        self._hinted_correct[pid] = {}

    def record(self, update: int, rollout: Rollout, rewards: Sequence[float] | torch.Tensor, hinted: bool = False) -> None:
        """Add the episodes of `rollout` (with their 0/1 `rewards`) sampled in update `update`."""
        values = rewards.reshape(-1).tolist() if isinstance(rewards, torch.Tensor) else list(rewards)
        if len(values) != len(rollout.actions):
            raise ValueError(f"{len(values)} rewards for {len(rollout.actions)} episodes")
        self.updates = max(self.updates, int(update))
        for pid, actions, forced, reward in zip(rollout.prompt_ids, rollout.actions, rollout.n_forced, values):
            pid = int(pid)
            self._ensure(pid)
            self._episodes[pid] += 1
            self._tokens[pid] += len(actions) - int(forced)
            if hinted:
                self._hinted_episodes[pid] += 1
            if reward > 0:
                counts = self._hinted_correct[pid] if hinted else self._correct[pid]
                counts[int(update)] = counts.get(int(update), 0) + 1
                first = self._first_hinted if hinted else self._first
                if first[pid] is None:
                    first[pid] = int(update)

    def merge(self, other: SampleLog, offset: int) -> None:
        """Add `other`'s records with its update numbers shifted by `offset` (a later phase)."""
        self.hints = self.hints or other.hints
        self.updates = max(self.updates, other.updates + offset)
        for pid in other.prompt_ids:
            self._ensure(pid)
            for mine, theirs in ((self._first, other._first), (self._first_hinted, other._first_hinted)):
                value = theirs[pid]
                if value is not None:
                    current = mine[pid]
                    mine[pid] = value + offset if current is None else min(current, value + offset)
            self._episodes[pid] += other._episodes[pid]
            self._tokens[pid] += other._tokens[pid]
            self._hinted_episodes[pid] += other._hinted_episodes[pid]
            for mine_c, theirs_c in ((self._correct, other._correct), (self._hinted_correct, other._hinted_correct)):
                for update, count in theirs_c[pid].items():
                    mine_c[pid][update + offset] = mine_c[pid].get(update + offset, 0) + count

    def _cumulative(self, *counts: Mapping[int, int]) -> dict[str, int]:
        return {str(c): sum(n for part in counts for u, n in part.items() if u <= c) for c in self.checkpoints}

    def to_dict(self) -> dict:
        prompts: dict[str, dict] = {}
        for pid in self.prompt_ids:
            entry: dict[str, Any] = {
                "first_success_update": self._first[pid],
                "correct_by_checkpoint": self._cumulative(self._correct[pid], self._hinted_correct[pid]),
                "episodes_sampled": self._episodes[pid],
                "tokens_sampled": self._tokens[pid],
            }
            if self.hints:
                entry["first_hinted_success_update"] = self._first_hinted[pid]
                entry["hinted_episodes_sampled"] = self._hinted_episodes[pid]
                entry["hinted_correct_by_checkpoint"] = self._cumulative(self._hinted_correct[pid])
            prompts[str(pid)] = entry

        def total(field: str) -> dict[str, int]:
            return {str(c): sum(e[field][str(c)] for e in prompts.values()) for c in self.checkpoints}

        totals: dict[str, Any] = {
            "episodes_sampled": sum(self._episodes.values()),
            "tokens_sampled": sum(self._tokens.values()),
            "correct_by_checkpoint": total("correct_by_checkpoint"),
        }
        if self.hints:
            totals["hinted_episodes_sampled"] = sum(self._hinted_episodes.values())
            totals["hinted_correct_by_checkpoint"] = total("hinted_correct_by_checkpoint")
        return {"updates": self.updates, "checkpoints": list(self.checkpoints), "prompts": prompts, "totals": totals}


def episode_rewards(task: Task, rollout: Rollout) -> torch.Tensor:
    """(B,) float rewards: 1.0 where `task.verify` accepts the episode, else 0.0."""
    return torch.tensor(
        [1.0 if task.verify(pid, actions) else 0.0 for pid, actions in zip(rollout.prompt_ids, rollout.actions)]
    )


def group_advantages(rewards: torch.Tensor, group: int, kind: str = "rloo", eps: float = 1e-6) -> torch.Tensor:
    """(B,) advantages for rewards laid out as consecutive groups of `group` episodes.

    Episodes [k*group, (k+1)*group) must share one prompt. Within each group:

    * ``kind="rloo"``: A_j = r_j - (sum_i r_i - r_j) / (G - 1) = (G r_j - sum_i r_i) / (G - 1).
    * ``kind="grpo"``: A_j = (r_j - mean_i r_i) / (std_i r_i + eps), population std (ddof 0).

    A group whose rewards are all equal gets exactly zero advantage under both. The result
    is detached (advantages are constants in the policy gradient).
    """
    if group < 2:
        raise ValueError(f"group-baseline advantages need a group of at least 2, got {group}")
    rewards = rewards.detach().to(torch.float32)
    if rewards.ndim != 1 or rewards.numel() % group:
        raise ValueError(f"expected a flat reward vector whose length is a multiple of {group}, got {tuple(rewards.shape)}")
    r = rewards.view(-1, group)
    if kind == "rloo":
        adv = (group * r - r.sum(dim=1, keepdim=True)) / (group - 1)
    elif kind == "grpo":
        centred = r - r.mean(dim=1, keepdim=True)
        adv = centred / (r.std(dim=1, unbiased=False, keepdim=True) + eps)
    else:
        raise ValueError(f"unknown advantage kind {kind!r}; expected one of {ADVANTAGE_KINDS}")
    return adv.reshape(-1)


def rl_loss(
    rollout: Rollout,
    rewards: torch.Tensor,
    group: int,
    kind: str = "rloo",
    entropy_coef: float = 0.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Scalar policy-gradient loss -mean(A * log_prob) - entropy_coef * mean(entropy).

    `rollout` must hold consecutive groups of `group` episodes per prompt (as produced by
    sampling with each prompt id repeated `group` times). Its log_prob/entropy cover only
    generated steps, so forced hint prefixes get no gradient. With entropy_coef == 0 the
    entropy term is omitted entirely.
    """
    advantages = group_advantages(rewards, group, kind, eps)
    loss = -(advantages * rollout.log_prob).mean()
    if entropy_coef:
        loss = loss - entropy_coef * rollout.entropy.mean()
    return loss


def select_prompts(task: Task, prompt_ids: Sequence[int], only_tags: Mapping[str, Any] | None = None) -> list[int]:
    """The ids in `prompt_ids` (order kept) whose tags match every key in `only_tags`.

    A tag value may be a string (exact match) or a list/tuple/set of strings (match any).
    None or {} keeps every given id. Never returns an id outside `prompt_ids`.
    """
    ids = [int(p) for p in prompt_ids]
    if not only_tags:
        return ids
    prompts = task.prompts()

    def matches(pid: int) -> bool:
        tags = prompts[pid].tags
        for key, want in only_tags.items():
            have = tags.get(key)
            if isinstance(want, (list, tuple, set, frozenset)):
                if have not in want:
                    return False
            elif have != want:
                return False
        return True

    return [p for p in ids if matches(p)]


@TRAINERS.register("rloo")
class RLOO(Trainer):
    """RLOO: REINFORCE with a leave-one-out group baseline, one Adam step per sampled batch.

    State, action, reward and return are as in the module docstring. Each update:
    1. choose the update's prompts: all training prompts, or a seeded minibatch of
       `prompts_per_update` of them;
    2. sample G = `group` episodes per prompt in one `policy.sample` call;
    3. reward r_j = 1[task.verify], advantage A_j = r_j - mean of the other G-1 rewards;
    4. one Adam step on -mean(A log pi(tau)) - entropy_coef * mean(H).

    Params: group (16), lr (1e-3), betas ((0.9, 0.999)), entropy_coef (0.0),
    prompts_per_update (None = all), only_tags (None = all given prompt_ids), eps (1e-6,
    GRPO only). One Adam optimizer is created per `train` call. All randomness (sampling and
    minibatches) comes from a torch.Generator seeded with `seed`.
    """

    advantage_kind = "rloo"
    default_entropy_coef = 0.0

    def __init__(self, **params: Any):
        super().__init__(**params)
        self.group = int(params.get("group", 16))
        self.lr = float(params.get("lr", 1e-3))
        self.betas = tuple(float(b) for b in params.get("betas", (0.9, 0.999)))
        self.entropy_coef = float(params.get("entropy_coef", self.default_entropy_coef))
        ppu = params.get("prompts_per_update")
        self.prompts_per_update = None if ppu is None else int(ppu)
        self.only_tags = params.get("only_tags")
        self.eps = float(params.get("eps", 1e-6))
        if self.group < 2:
            raise ValueError(f"{type(self).__name__} needs a group of at least 2, got {self.group}")
        if self.prompts_per_update is not None and self.prompts_per_update < 1:
            raise ValueError("prompts_per_update must be at least 1")
        self.sample_log: SampleLog | None = None

    def training_log(self) -> dict | None:
        """What the last `train` call sampled (see `SampleLog`); None before any `train` call."""
        return None if self.sample_log is None else self.sample_log.to_dict()

    def loss(self, rollout: Rollout, rewards: torch.Tensor) -> torch.Tensor:
        """This arm's loss for one sampled batch (see `rl_loss`)."""
        return rl_loss(rollout, rewards, self.group, self.advantage_kind, self.entropy_coef, self.eps)

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
        self.sample_log = log = SampleLog(ids, checkpoints)
        call_checkpoints(0, checkpoints, policy, on_checkpoint)
        if budget <= 0:
            return
        if not ids:
            raise ValueError("no training prompts left after applying only_tags")
        generator = torch.Generator().manual_seed(int(seed))
        params = list(policy.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr, betas=(self.betas[0], self.betas[1]))
        for step in range(1, budget + 1):
            batch = ids
            if self.prompts_per_update is not None and self.prompts_per_update < len(ids):
                order = torch.randperm(len(ids), generator=generator)[: self.prompts_per_update]
                batch = [ids[i] for i in sorted(order.tolist())]
            episode_ids = [p for p in batch for _ in range(self.group)]
            rollout = policy.sample(task, episode_ids, generator)
            rewards = episode_rewards(task, rollout)
            log.record(step, rollout, rewards)
            loss = self.loss(rollout, rewards)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            call_checkpoints(step, checkpoints, policy, on_checkpoint)


@TRAINERS.register("grpo")
class GRPO(RLOO):
    """GRPO-style arm: RLOO's loop with a group-normalised advantage.

    A_j = (r_j - mean_group(r)) / (std_group(r) + eps), population std, eps = 1e-6, so a group
    with identical rewards gets exactly zero advantage. We take a single on-policy gradient
    step per sampled batch, so the PPO-style importance ratio pi_theta / pi_old equals 1 at
    the step and clipping is inactive; the clipped objective reduces to this loss. There is
    no KL penalty to a reference policy (it can be added later as a parameter).
    """

    advantage_kind = "grpo"


@TRAINERS.register("rl_entropy")
class EntropyRLOO(RLOO):
    """Entropy-bonus control: RLOO with entropy_coef = 0.01 by default.

    The bonus rewards keeping per-step entropy high, which spreads mass without any verifier
    signal; it is the control for "does merely staying diffuse explain the gain?".
    """

    default_entropy_coef = 0.01
