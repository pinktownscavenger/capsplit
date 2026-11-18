"""The four plug-in interfaces of the harness: Task, Policy, Trainer and Backend, plus BaseRecipe.

RL framing, which every implementation must keep:

* A **prompt** starts an episode. The **state** is the prompt plus the actions taken so far
  (for grid tasks, the agent's cell); `Task.initial_state` and `Task.step` define the dynamics.
* An **action** is one of `Task.n_actions` choices (a token, or a move).
* The **reward** is terminal and binary: 1 if `Task.verify(prompt_id, actions)` accepts the
  finished episode, else 0. So the **return** of an episode equals its reward, and a
  policy's expected return on a prompt is its success probability J.
* A **policy** gives a distribution over actions in each state (`Policy.sample`,
  `Policy.log_prob`, `Policy.state_logits`).
* A **trainer** is an **update rule**: it changes a policy's parameters for a fixed number
  of optimizer updates (`Trainer.train`).
* A **backend** measures a base/trained pair: J, the reliability/capability split and E_N.

Each interface has a registry (see `registry.py`), so adding a new one means writing one
class and decorating it.
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable, Mapping, Sequence

import numpy as np
import torch

from capsplit.engine import TrajectoryTable


# --------------------------------------------------------------------------- Task


@dataclass(frozen=True)
class Prompt:
    """One episode start. `tokens` is what a policy conditions on; `tags` label the prompt
    (for example {"difficulty": "hard", "split": "train"}) and define measurement groups."""

    id: int
    tokens: tuple
    tags: Mapping[str, str] = field(default_factory=dict)


class Task(ABC):
    """An episodic task with a 0/1 verifier. Prompt ids must be 0, 1, ..., len(prompts()) - 1."""

    n_actions: int
    horizon: int  # maximum number of actions in an episode

    @abstractmethod
    def prompts(self) -> Sequence[Prompt]:
        """All prompts, in id order."""

    @abstractmethod
    def verify(self, prompt_id: int, actions: Sequence[int]) -> bool:
        """The reward function: True if the finished episode is correct."""

    @abstractmethod
    def initial_state(self, prompt_id: int) -> Hashable:
        """The state before any action."""

    @abstractmethod
    def step(self, prompt_id: int, state: Hashable, action: int) -> tuple[Hashable, bool]:
        """Deterministic transition: (next state, episode done)."""

    def demonstrations(self, prompt_id: int) -> Sequence[tuple]:
        """Correct action sequences a teacher could show (used by supervised arms)."""
        raise NotImplementedError(f"{type(self).__name__} has no demonstrations")

    def table(self, prompt_id: int) -> TrajectoryTable | None:
        """Optional exact enumerator: every trajectory for this prompt, compiled once.

        Row r of `table.index` lists the (state * n_actions + action) slots that trajectory r
        visits, so `Policy.state_logits(task, prompt_id)` must use the same state numbering.
        Return None if the task is too large to enumerate.
        """
        return None

    def prompt_ids(self, **tags: str) -> list[int]:
        """Ids of the prompts whose tags match every given key=value."""
        return [p.id for p in self.prompts() if all(p.tags.get(k) == v for k, v in tags.items())]


# --------------------------------------------------------------------------- Policy


@dataclass
class Rollout:
    """A batch of sampled episodes.

    `log_prob` and `entropy` cover only the *generated* steps (forced prefix steps are
    excluded) and carry gradients, so trainers can build losses from them directly.
    """

    prompt_ids: list[int]
    actions: list[tuple]  # full action sequences, including any forced prefix
    log_prob: torch.Tensor  # (B,) sum of log pi(a_t | s_t) over generated steps
    entropy: torch.Tensor  # (B,) sum over generated steps of the entropy of pi(. | s_t)
    n_forced: list[int]  # forced prefix length for each episode


class Policy(ABC):
    """A stochastic policy pi(a | s). Implementations hold torch parameters."""

    @abstractmethod
    def sample(
        self,
        task: Task,
        prompt_ids: Sequence[int],
        generator: torch.Generator,
        prefixes: Sequence[tuple] | None = None,
    ) -> Rollout:
        """Sample one episode per entry of `prompt_ids` (repeat ids for a group).

        If `prefixes` is given, episode i first takes the actions in prefixes[i] (forced,
        not sampled, not counted in log_prob or entropy), then continues on-policy.
        """

    @abstractmethod
    def log_prob(
        self, task: Task, prompt_ids: Sequence[int], actions: Sequence[tuple], start: int | Sequence[int] = 0
    ) -> torch.Tensor:
        """(B,) teacher-forced sum of log pi(a_t | s_t) over steps t >= start, with gradients."""

    @abstractmethod
    def state_logits(self, task: Task, prompt_id: int) -> np.ndarray:
        """(table.n_states, n_actions) logits, numbered like `task.table(prompt_id)`.

        Needed only by the exact backend. pi(. | s) = softmax(logits[s]); per-step tempering
        at beta multiplies these logits by beta.
        """

    def trajectory_probs(self, task: Task, prompt_id: int) -> np.ndarray:
        """(T,) exact probability of each row of `task.table(prompt_id)`.

        The default chains `state_logits`. Override it for a policy that is a distribution over
        whole trajectories rather than per-state choices (for example a planted case).
        """
        from capsplit.engine import trajectory_probs

        table = task.table(prompt_id)
        if table is None:
            raise ValueError(f"{type(task).__name__} has no exact table")
        return trajectory_probs(table, self.state_logits(task, prompt_id))

    def tempering_logits(self, task: Task, prompt_id: int) -> np.ndarray | None:
        """Per-state logits for per-step tempering, or None if the policy has none (then only
        sequence-level tempering applies). Defaults to `state_logits`."""
        return self.state_logits(task, prompt_id)

    @abstractmethod
    def parameters(self) -> Iterable[torch.nn.Parameter]:
        """Trainable parameters, for an optimizer."""

    @abstractmethod
    def state_dict(self) -> dict:
        """Everything needed to restore the policy's parameters."""

    @abstractmethod
    def load_state_dict(self, state: dict) -> None:
        """Restore parameters saved by `state_dict`."""

    def clone(self) -> "Policy":
        """An independent copy (post-training arms each start from a clone of the base)."""
        return copy.deepcopy(self)


# --------------------------------------------------------------------------- Trainer

CheckpointHook = Callable[[int, Policy], None]


class Trainer(ABC):
    """A post-training arm: an update rule applied for a fixed number of optimizer updates.

    Contract:
    * `budget` is the number of optimizer updates. Every arm in one experiment gets the same.
    * `prompt_ids` are the prompts the arm may train on (the train split). An arm may use a
      subset (for example, supervised fine-tuning on hard train prompts only) but never others.
    * Call `on_checkpoint(step, policy)` after update `step` for every step in `checkpoints`
      (step 0 means before any update). The hook must not modify the policy.
    * All randomness comes from `seed`, so a run is reproducible.
    """

    name: str = "trainer"

    def __init__(self, **params: Any):
        self.params = params

    @abstractmethod
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
        """Update `policy` in place."""


def call_checkpoints(step: int, checkpoints: Sequence[int], policy: Policy, hook: CheckpointHook) -> None:
    """Helper for trainers: call the hook if `step` is a checkpoint."""
    if step in checkpoints:
        hook(step, policy)


# --------------------------------------------------------------------------- Base recipe


class BaseRecipe(ABC):
    """Builds the base policy pi_0 (pre-training) for a task. Returns the policy and a JSON-able info dict."""

    def __init__(self, **params: Any):
        self.params = params

    @abstractmethod
    def build(self, task: Task, seed: int) -> tuple[Policy, dict]:
        """Pre-train and return (pi_0, info)."""


# --------------------------------------------------------------------------- Measurement backend


@dataclass(frozen=True)
class MeasureSpec:
    """What to measure: budgets N, threshold multipliers c, and named prompt groups."""

    ns: tuple
    cs: tuple
    groups: Mapping[str, tuple]  # group name -> prompt ids


@dataclass(frozen=True)
class GroupMetrics:
    """Metrics for one prompt group at one (N, c). Means are over the group's prompts.

    delta_rel + delta_cap == j1 - j0 (Proposition 1) up to floating-point error.
    `excess` maps a tempering family ("per_step", "sequence", "both") to E_N for that family.
    """

    group: str
    n: int
    c: float
    n_prompts: int
    j0: float
    j1: float
    delta_rel: float
    delta_cap: float
    excess: Mapping[str, float]
    label: str
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def delta_total(self) -> float:
        return self.j1 - self.j0

    def to_dict(self) -> dict:
        return {
            "group": self.group,
            "n": self.n,
            "c": self.c,
            "n_prompts": self.n_prompts,
            "j0": self.j0,
            "j1": self.j1,
            "delta_rel": self.delta_rel,
            "delta_cap": self.delta_cap,
            "delta_total": self.delta_total,
            "excess": dict(self.excess),
            "label": self.label,
            "extra": dict(self.extra),
        }


class Backend(ABC):
    """Measures a base/trained pair.

    `prepare` does the work that depends only on the base (for example the tempering
    frontier) and returns a cache; `measure` then scores any trained policy against it.
    """

    def __init__(self, **params: Any):
        self.params = params

    @abstractmethod
    def prepare(self, task: Task, base: Policy, spec: MeasureSpec) -> Any:
        """Base-only precomputation."""

    @abstractmethod
    def measure(self, task: Task, base: Policy, trained: Policy, spec: MeasureSpec, cache: Any) -> list[GroupMetrics]:
        """One GroupMetrics per (group, N, c)."""
