"""LM post-training arms (harness `Trainer`s): `lm_none`, `lm_rl`, `lm_sft`, `lm_sft_then_rl`.

RL framing (docs/lm_design.md): an episode is one problem; the state is the chat-formatted prompt
plus the tokens generated so far; an action is the next token; the reward is terminal and binary,
r = 1.0 if `task.verify(problem, completion_text)` else 0.0, so the return equals the reward and
a policy's expected return on a problem is its success probability. Each trainer is an update rule
on the LoRA parameters of a `LocalLMPolicy`, run for exactly `budget` optimizer updates (H5).

Contract (interfaces.Trainer): `on_checkpoint(k, policy)` is called before any update (k = 0) and
after update k for every k in `checkpoints`; only `prompt_ids` (the train split) are trained on;
all randomness comes from `seed`; unknown params raise (H14). Trainers write no files: the stage
saves adapters from `on_checkpoint` and writes `training_log()` as training.json.

Reuse. The group-normalised advantage is `trainers.rl.group_advantages` (kind "grpo") and the
policy-gradient loss is `rl_loss`'s -mean(A * log_prob), accumulated over micro-batches; rewards go through `rl.episode_rewards` on a text view of the rollout
(LM verifiers take the decoded text, not token ids); the SFT loss and minibatching are
`trainers.sft.sft_loss` / `minibatch` (LM `log_prob` accepts answer strings and appends the stop
token); the SFT-then-RL budget and seed split are `trainers.sft_rl.split_budget` / `phase_seeds`;
the training log is `rl.SampleLog`, extended (`LMSampleLog`) with episodes per checkpoint and
running success rates.

Choices (each also stated where implemented):

* Optimizer: Adam (AdamW with weight_decay = 0 by default), one per `train` call (and per phase in
  `lm_sft_then_rl`, as H25). Gradients are clipped to global norm `max_grad_norm` (default 1.0;
  None disables) before every step, as is standard for LoRA fine-tuning of a pretrained model;
  clipping a zero gradient leaves it zero.
* `lm_rl` problem choice: a seeded "epoch" cycle. The train problems are shuffled; each update
  takes the next `problems_per_update` of them; when fewer remain, the remainder is taken and the
  rest is filled from a fresh shuffle (the remainder's problems moved to the end of that shuffle,
  so no problem repeats within an update). Every problem is used once per cycle, and counts per
  problem never differ by more than one cycle. One torch CPU generator seeded by `seed` drives
  both the shuffles and the sampling (H19).
* `lm_rl` memory: completions are sampled under `torch.no_grad()` (no autograd graph), then the
  teacher-forced log-probs are recomputed with `policy.log_prob` in micro-batches of `micro_batch`
  sequences, each chunk's share of the loss backpropagated at once (`accumulate_pg_gradient`). The
  accumulated gradient equals that of the full-batch loss -mean(A * log_prob) (up to float summation
  order), so the update is unchanged while only one micro-batch's graph is ever alive. Cost: one
  extra (gradient-free) scoring forward per update, since `policy.sample` always scores.
* On MPS each gradient arm sets the allocator's hard cap (`device.limit_mps_memory`, unless
  `memory.mps_hard_cap` is false), and a torch out-of-memory error inside an update is re-raised
  as a `MemoryGuardError` that names the config values to lower (`device.explain_oom`).
* `lm_sft` receives the base classification through `set_base_classes(mapping)`, called by the
  stage before `train` (`needs_base_classes` is True for `lm_sft` and `lm_sft_then_rl`). The
  mapping is {problem_id: "reachable" | "unreachable" | "ambiguous"} (keys int or str, as read
  from the base stage's classification JSON). Reason: config params are static and do not know
  the output directory; the stage already holds the classification and records it.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, ClassVar, cast

import torch

from ..interfaces import CheckpointHook, Policy, Rollout, Task, Trainer, call_checkpoints
from ..registry import TRAINERS
from ..trainers.none import NoTraining
from ..trainers.rl import SampleLog, episode_rewards, group_advantages
from ..trainers.sft import minibatch, sft_loss
from ..trainers.sft_rl import phase_seeds, split_budget
from .device import explain_oom, limit_mps_memory

__all__ = [
    "CLASSES",
    "LMRL",
    "LMSFT",
    "LMNone",
    "LMSFTThenRL",
    "LMSampleLog",
    "ProblemCycle",
    "accumulate_pg_gradient",
    "lm_rewards",
]

CLASSES = ("reachable", "unreachable", "ambiguous")


def _check_params(name: str, params: Mapping[str, Any], known: frozenset[str] | set[str]) -> None:
    unknown = sorted(set(params) - known)
    if unknown:
        raise ValueError(f"{name}: unknown parameters {unknown}; known: {sorted(known)}")


def _guard(policy: Policy, stage: str) -> None:
    """Memory guard between updates (LocalLMPolicy carries one; other policies have none)."""
    guard = getattr(policy, "guard", None)
    device = getattr(policy, "device", None)
    if isinstance(device, torch.device) and device.type == "mps":
        # Release the MPS allocator's cached blocks between updates: otherwise the cache keeps the
        # previous update's peak and the guard (which counts driver memory) reads it as in use.
        torch.mps.empty_cache()
    if guard is not None:
        guard.check_running(stage)


def _hard_cap(policy: Policy) -> None:
    """Set the MPS allocator cap from the policy's memory config (LocalLMPolicy; others have none)."""
    config = getattr(policy, "config", None)
    memory = getattr(config, "memory", None)
    device = getattr(policy, "device", None)
    if memory is not None and isinstance(device, torch.device):
        limit_mps_memory(memory, device)


@contextmanager
def _explain_oom(policy: Policy, stage: str) -> Iterator[None]:
    """Re-raise torch out-of-memory errors as a MemoryGuardError saying which config values to lower."""
    try:
        yield
    except RuntimeError as error:
        explained = explain_oom(error, getattr(getattr(policy, "config", None), "memory", None), stage)
        if explained is None:
            raise
        raise explained from error


def _optimizer(policy: Policy, lr: float, betas: tuple[float, float], weight_decay: float) -> torch.optim.Optimizer:
    params = list(policy.parameters())
    if not params:
        raise ValueError("the policy has no trainable parameters")
    return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)


def _step(optimizer: torch.optim.Optimizer, loss: torch.Tensor | None, max_grad_norm: float | None) -> None:
    """zero_grad, backward `loss`, clip, step. With loss None the gradients are already accumulated."""
    if loss is not None:
        optimizer.zero_grad(set_to_none=False)
        loss.backward()
    if max_grad_norm is not None:
        params = [p for group in optimizer.param_groups for p in group["params"]]
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
    optimizer.step()


def _common(name: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """Optimizer params shared by the gradient arms, validated."""
    lr = float(params.get("lr", 1e-4))
    betas = params.get("betas", (0.9, 0.999))
    weight_decay = float(params.get("weight_decay", 0.0))
    mgn = params.get("max_grad_norm", 1.0)
    if lr <= 0:
        raise ValueError(f"{name}: lr must be positive, got {lr}")
    if mgn is not None and float(mgn) <= 0:
        raise ValueError(f"{name}: max_grad_norm must be positive or null, got {mgn}")
    return {"lr": lr, "betas": (float(betas[0]), float(betas[1])), "weight_decay": weight_decay,
            "max_grad_norm": None if mgn is None else float(mgn)}


def _positive_int(name: str, key: str, value: Any, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name}: {key} must be an integer >= {minimum}, got {value!r}")
    return value


# --------------------------------------------------------------------------- Training log


class LMSampleLog(SampleLog):
    """`rl.SampleLog` plus episodes per checkpoint and the running success rate of training samples.

    Adds to each prompt entry and to the totals::

        "episodes_by_checkpoint": {"<c>": int},        # episodes sampled in updates 1..c
        "success_rate_by_checkpoint": {"<c>": float | null}  # correct / episodes in updates 1..c

    and to the totals "window_success_rate_by_checkpoint" (rate over updates (previous c, c]).
    `extra` holds arm-specific JSON-ready fields (copied to the top level of the dict).
    """

    def __init__(self, prompt_ids: Sequence[int], checkpoints: Sequence[int]):
        super().__init__(prompt_ids, checkpoints)
        self._episodes_at: dict[int, dict[int, int]] = {}
        self.extra: dict[str, Any] = {}

    def record(self, update: int, rollout: Rollout, rewards: Sequence[float] | torch.Tensor,
               hinted: bool = False) -> None:
        super().record(update, rollout, rewards, hinted)
        for pid in rollout.prompt_ids:
            counts = self._episodes_at.setdefault(int(pid), {})
            counts[int(update)] = counts.get(int(update), 0) + 1

    def merge(self, other: SampleLog, offset: int) -> None:
        super().merge(other, offset)
        if isinstance(other, LMSampleLog):
            for pid, counts in other._episodes_at.items():
                mine = self._episodes_at.setdefault(pid, {})
                for update, n in counts.items():
                    mine[update + offset] = mine.get(update + offset, 0) + n
            self.extra.update(other.extra)

    def to_dict(self) -> dict:
        out = super().to_dict()

        def rate(correct: int, episodes: int) -> float | None:
            return correct / episodes if episodes else None

        for key, entry in out["prompts"].items():
            episodes = self._cumulative(self._episodes_at.get(int(key), {}))
            entry["episodes_by_checkpoint"] = episodes
            entry["success_rate_by_checkpoint"] = {
                c: rate(entry["correct_by_checkpoint"][c], episodes[c]) for c in episodes}
        totals = out["totals"]
        episodes = {str(c): sum(e["episodes_by_checkpoint"][str(c)] for e in out["prompts"].values())
                    for c in self.checkpoints}
        correct = totals["correct_by_checkpoint"]
        totals["episodes_by_checkpoint"] = episodes
        totals["success_rate_by_checkpoint"] = {c: rate(correct[c], episodes[c]) for c in episodes}
        window: dict[str, float | None] = {}
        prev = None
        for c in self.checkpoints:
            key = str(c)
            d_eps = episodes[key] - (episodes[prev] if prev else 0)
            d_cor = correct[key] - (correct[prev] if prev else 0)
            window[key] = rate(d_cor, d_eps)
            prev = key
        totals["window_success_rate_by_checkpoint"] = window
        out.update(self.extra)
        return out


# --------------------------------------------------------------------------- Helpers


def lm_rewards(task: Task, rollout: Rollout) -> torch.Tensor:
    """(B,) rewards 1.0/0.0 from `task.verify(pid, text)`: `rl.episode_rewards` on a text view.

    LM verifiers take the decoded completion text, so the rollout's `texts` (an `LMRollout`)
    stand in for its token-id actions. The rewards are put on the log-probs' device (MPS or CPU).
    """
    texts = getattr(rollout, "texts", None)
    if texts is None:
        raise TypeError("LM trainers need rollouts with decoded `texts` (LocalLMPolicy.sample)")
    view = Rollout(list(rollout.prompt_ids), list(texts), rollout.log_prob, rollout.entropy, list(rollout.n_forced))
    return episode_rewards(task, view).to(rollout.log_prob.device)  # rl.py builds them on CPU


def accumulate_pg_gradient(policy: Policy, task: Task, prompt_ids: Sequence[int], actions: Sequence[Any],
                           advantages: torch.Tensor, micro_batch: int) -> float:
    """Backpropagate the policy-gradient loss -mean_i(A_i * log pi(a_i | x_i)) in micro-batches.

    Each chunk c of at most `micro_batch` episodes recomputes its teacher-forced log-probs with
    `policy.log_prob` and backpropagates -sum_{i in c}(A_i * log_prob_i) / B (B = all episodes), so
    the gradients accumulated in `.grad` sum to the full-batch gradient; only one chunk's graph is
    alive at a time. Gradients are added to whatever `.grad` holds (zero them first). Chunks whose
    advantages are all zero contribute exactly zero gradient and are skipped. Returns the loss value.
    """
    n = len(prompt_ids)
    if len(actions) != n or advantages.numel() != n:
        raise ValueError(f"{n} prompt ids, {len(actions)} actions and {advantages.numel()} advantages must match")
    total = 0.0
    for lo in range(0, n, micro_batch):
        hi = min(n, lo + micro_batch)
        adv = advantages[lo:hi]
        if not bool(torch.any(adv != 0)):
            continue
        log_prob = policy.log_prob(task, list(prompt_ids[lo:hi]), list(actions[lo:hi]))
        loss = -(adv.to(log_prob.device, log_prob.dtype) * log_prob).sum() / n
        loss.backward()
        total += float(loss.detach())
        del log_prob, loss
    return total


class ProblemCycle:
    """Seeded epoch-like cycle over `ids` (see the module docstring); `next(k)` returns k distinct ids."""

    def __init__(self, ids: Sequence[int], generator: torch.Generator):
        self.ids = [int(i) for i in ids]
        self.generator = generator
        self.queue: list[int] = []

    def _shuffle(self) -> list[int]:
        return [self.ids[i] for i in torch.randperm(len(self.ids), generator=self.generator).tolist()]

    def next(self, k: int) -> list[int]:
        if k >= len(self.ids):
            return sorted(self.ids)
        if len(self.queue) < k:
            taken = self.queue
            fresh = self._shuffle()
            held = set(taken)
            fresh = [i for i in fresh if i not in held] + [i for i in fresh if i in held]
            batch = taken + fresh[: k - len(taken)]
            self.queue = fresh[k - len(taken):]
        else:
            batch, self.queue = self.queue[:k], self.queue[k:]
        return sorted(batch)


# --------------------------------------------------------------------------- Arms


@TRAINERS.register("lm_none")
class LMNone(NoTraining):
    """Control: no updates; only the checkpoint hook is called. `training_log()` is None."""

    name = "lm_none"
    needs_base_classes = False

    def __init__(self, **params: Any):
        _check_params("lm_none", params, set())
        super().__init__(**params)

    def training_log(self) -> dict | None:
        return None


@TRAINERS.register("lm_rl")
class LMRL(Trainer):
    """GRPO-style policy gradient on LoRA parameters (no KL, one on-policy step per batch, H18).

    Each update: choose `problems_per_update` train problems (`ProblemCycle`); sample G = `group`
    completions each with `policy.sample` under no_grad; reward = 1[task.verify];
    A = (r - mean_group r) / (std_group r + eps) (`rl.group_advantages`, kind "grpo", over the full
    groups), so an all-fail or all-success group gives exactly zero gradient; loss =
    -mean(A * log_prob) as `rl.rl_loss`, backpropagated in micro-batches of `micro_batch` sequences
    with recomputed log-probs (`accumulate_pg_gradient`); clip to `max_grad_norm`; one optimizer
    step; memory guard.

    Params: group (8), problems_per_update (8, capped at the number of train problems), lr (1e-4),
    betas ((0.9, 0.999)), weight_decay (0.0), max_grad_norm (1.0, None disables), eps (1e-6),
    micro_batch (16; sequences per backward pass; peak memory scales with it, the update does not).
    """

    name = "lm_rl"
    needs_base_classes = False
    KNOWN: ClassVar[frozenset[str]] = frozenset({"group", "problems_per_update", "lr", "betas", "weight_decay", "max_grad_norm", "eps",
                                               "micro_batch"})

    def __init__(self, **params: Any):
        _check_params(self.name, params, self.KNOWN)
        super().__init__(**params)
        self.group = _positive_int(self.name, "group", params.get("group", 8), minimum=2)
        self.problems_per_update = _positive_int(self.name, "problems_per_update", params.get("problems_per_update", 8))
        self.opt = _common(self.name, params)
        self.eps = float(params.get("eps", 1e-6))
        self.micro_batch = _positive_int(self.name, "micro_batch", params.get("micro_batch", 16))
        self.sample_log: LMSampleLog | None = None

    def training_log(self) -> dict | None:
        return None if self.sample_log is None else self.sample_log.to_dict()

    def train(self, policy: Policy, task: Task, budget: int, seed: int, prompt_ids: Sequence[int],
              checkpoints: Sequence[int], on_checkpoint: CheckpointHook) -> None:
        ids = sorted({int(p) for p in prompt_ids})
        self.sample_log = log = LMSampleLog(ids, checkpoints)
        log.extra["trainer"] = self.name
        call_checkpoints(0, checkpoints, policy, on_checkpoint)
        if budget <= 0:
            return
        if not ids:
            raise ValueError(f"{self.name}: no training problems")
        generator = torch.Generator().manual_seed(int(seed))
        cycle = ProblemCycle(ids, generator)
        optimizer = _optimizer(policy, self.opt["lr"], self.opt["betas"], self.opt["weight_decay"])
        _hard_cap(policy)
        for step in range(1, int(budget) + 1):
            stage = f"{self.name} update {step}"
            batch = cycle.next(self.problems_per_update)
            with _explain_oom(policy, stage):
                # No graph while sampling: sample()'s log_prob is discarded and recomputed per micro-batch.
                with torch.no_grad():
                    rollout = policy.sample(task, [p for p in batch for _ in range(self.group)], generator)
                rewards = lm_rewards(task, rollout)
                log.record(step, rollout, rewards)
                prompt_ids, actions = list(rollout.prompt_ids), list(rollout.actions)
                del rollout
                advantages = group_advantages(rewards, self.group, "grpo", self.eps)
                optimizer.zero_grad(set_to_none=False)
                accumulate_pg_gradient(policy, task, prompt_ids, actions, advantages, self.micro_batch)
                _step(optimizer, None, self.opt["max_grad_norm"])
            _guard(policy, stage)
            call_checkpoints(step, checkpoints, policy, on_checkpoint)
        log.updates = int(budget)


@TRAINERS.register("lm_sft")
class LMSFT(Trainer):
    """Maximum likelihood on correct answers for train problems in the `target` base class(es).

    Never samples and never calls the verifier (episodes_sampled = 0). Each update: `batch`
    (problem, answer) pairs drawn without replacement by a generator seeded from `seed`
    (`sft.minibatch`; all pairs if batch >= their number); loss = -mean log pi(answer + stop |
    prompt) (`sft.sft_loss`; the answer is `task.demonstrations(pid)[0]`); clip; one step.

    The base classification must be given with `set_base_classes` before `train`.

    Params: target ("unreachable"; a class name or a list of them), batch (8), lr (1e-4), betas,
    weight_decay (0.0), max_grad_norm (1.0).
    """

    name = "lm_sft"
    needs_base_classes = True
    KNOWN: ClassVar[frozenset[str]] = frozenset({"target", "batch", "lr", "betas", "weight_decay", "max_grad_norm"})

    def __init__(self, **params: Any):
        _check_params(self.name, params, self.KNOWN)
        super().__init__(**params)
        target = params.get("target", "unreachable")
        targets = [target] if isinstance(target, str) else list(target)
        if not targets or any(t not in CLASSES for t in targets):
            raise ValueError(f"{self.name}: target must be one or more of {CLASSES}, got {target!r}")
        self.targets = tuple(targets)
        self.batch = _positive_int(self.name, "batch", params.get("batch", 8))
        self.opt = _common(self.name, params)
        self.base_classes: dict[int, str] | None = None
        self.sample_log: LMSampleLog | None = None

    def set_base_classes(self, classes: Mapping[Any, str]) -> None:
        """The base stage's classification {problem_id: class}; keys may be ints or strings."""
        out: dict[int, str] = {}
        for key, value in classes.items():
            if value not in CLASSES:
                raise ValueError(f"{self.name}: problem {key} has class {value!r}, expected one of {CLASSES}")
            out[int(key)] = value
        self.base_classes = out

    def training_log(self) -> dict | None:
        return None if self.sample_log is None else self.sample_log.to_dict()

    def target_ids(self, prompt_ids: Sequence[int]) -> list[int]:
        """Train ids whose base class is a target (every train id must be classified)."""
        if self.base_classes is None:
            raise ValueError(f"{self.name}: call set_base_classes(...) with the base classification before train()")
        ids = sorted({int(p) for p in prompt_ids})
        missing = [p for p in ids if p not in self.base_classes]
        if missing:
            raise ValueError(f"{self.name}: train problems {missing[:5]} are missing from the base classification")
        return [p for p in ids if self.base_classes[p] in self.targets]

    def train(self, policy: Policy, task: Task, budget: int, seed: int, prompt_ids: Sequence[int],
              checkpoints: Sequence[int], on_checkpoint: CheckpointHook) -> None:
        ids = self.target_ids(prompt_ids)
        self.sample_log = log = LMSampleLog(ids, checkpoints)
        log.updates = max(int(budget), 0)
        log.extra.update({"trainer": self.name, "sft_target": list(self.targets), "sft_problem_ids": ids})
        call_checkpoints(0, checkpoints, policy, on_checkpoint)
        if budget <= 0:
            return
        if not ids:
            raise ValueError(f"{self.name}: no train problem is in the target class(es) {list(self.targets)}")
        # Pairs hold answer strings; sft_loss passes them to policy.log_prob, which accepts strings.
        pairs = cast(list[tuple[int, tuple]], [(p, task.demonstrations(p)[0]) for p in ids])
        generator = torch.Generator().manual_seed(int(seed))
        optimizer = _optimizer(policy, self.opt["lr"], self.opt["betas"], self.opt["weight_decay"])
        _hard_cap(policy)
        for step in range(1, int(budget) + 1):
            with _explain_oom(policy, f"{self.name} update {step}"):
                _step(optimizer, sft_loss(policy, task, minibatch(pairs, self.batch, generator)), self.opt["max_grad_norm"])
            _guard(policy, f"{self.name} update {step}")
            call_checkpoints(step, checkpoints, policy, on_checkpoint)


@TRAINERS.register("lm_sft_then_rl")
class LMSFTThenRL(Trainer):
    """`lm_sft` for n_sft = round(sft_fraction * budget) updates, then `lm_rl` for the rest.

    As the toy `sft_then_rl` (H25): `sft_rl.split_budget` / `phase_seeds` (SeedSequence([seed, 0])
    and ([seed, 1])); a fresh optimizer per phase; checkpoints on the global update count (RL local
    step k is global n_sft + k; each step reported once); total updates == budget; the log merges
    both phases on global numbering. `set_base_classes` is forwarded to the SFT phase.

    Params: sft_fraction (0.5), sft (dict of lm_sft params), rl (dict of lm_rl params).
    """

    name = "lm_sft_then_rl"
    needs_base_classes = True
    KNOWN: ClassVar[frozenset[str]] = frozenset({"sft_fraction", "sft", "rl"})

    def __init__(self, **params: Any):
        _check_params(self.name, params, self.KNOWN)
        super().__init__(**params)
        fraction = params.get("sft_fraction", 0.5)
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0.0 <= fraction <= 1.0:
            raise ValueError(f"{self.name}: sft_fraction must be a number in [0, 1], got {fraction!r}")
        self.sft_fraction = float(fraction)
        self.sft_trainer = LMSFT(**dict(params.get("sft") or {}))
        self.rl_trainer = LMRL(**dict(params.get("rl") or {}))
        self.sample_log: LMSampleLog | None = None

    def set_base_classes(self, classes: Mapping[Any, str]) -> None:
        self.sft_trainer.set_base_classes(classes)

    def training_log(self) -> dict | None:
        return None if self.sample_log is None else self.sample_log.to_dict()

    def train(self, policy: Policy, task: Task, budget: int, seed: int, prompt_ids: Sequence[int],
              checkpoints: Sequence[int], on_checkpoint: CheckpointHook) -> None:
        n_sft, n_rl = split_budget(budget, self.sft_fraction)
        sft_seed, rl_seed = phase_seeds(seed)
        wanted = sorted({int(c) for c in checkpoints})
        done: set[int] = set()
        log = LMSampleLog([], wanted)

        def report(step: int, p: Policy) -> None:
            if step in wanted and step not in done:
                done.add(step)
                on_checkpoint(step, p)

        if n_sft > 0:
            self.sft_trainer.train(policy, task, n_sft, sft_seed, prompt_ids, [c for c in wanted if c <= n_sft], report)
            assert self.sft_trainer.sample_log is not None
            log.merge(self.sft_trainer.sample_log, 0)
        if n_rl > 0:
            lo = 1 if n_sft > 0 else 0
            rl_cps = [c - n_sft for c in wanted if lo <= c - n_sft <= n_rl]
            self.rl_trainer.train(policy, task, n_rl, rl_seed, prompt_ids, rl_cps,
                                  lambda local, p: report(local + n_sft, p))
            assert self.rl_trainer.sample_log is not None
            log.merge(self.rl_trainer.sample_log, n_sft)
        if n_sft == 0 and n_rl == 0:
            report(0, policy)
        log.updates = max(int(budget), 0)
        log.extra.update({"trainer": self.name, "n_sft": n_sft, "n_rl": n_rl,
                          "phase_seeds": [sft_seed, rl_seed]})
        self.sample_log = log
