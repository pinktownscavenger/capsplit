"""Base recipe `pretrain_gru`: weighted maximum-likelihood pre-training of one prompt-conditioned GRU.

Pre-training data (H8: every prompt, train and held-out). Each prompt's data is a weighted set
of sequences whose weights sum to 1:

* Easy prompt: the two correct solutions (`easy_correct` in total), the memorised wrong trap
  (`easy_trap`, the mode), `easy_distractors` random wrong sequences (`easy_distractor_mass`
  in total), and the noise set (below).
* Hard prompt: a *noisy trap* around the trap T (the easy rule at every position): T itself
  plus every single-token mutation of T, weighted like a channel that keeps T's token with
  probability `rho` at each position, `(1 - noise_mass)` in total. So at every node on T's path
  the trap token has probability about rho, and each other token about (1 - rho)/(V - 1).
  Mutations that are correct, or that leave T at position k through the hard token h_k, are
  dropped, so the hard subtree holds only hard data. Then the two hard solutions at weight
  `w` each, their single-token mutations after position k at weight `mu * w` each ("near
  misses"), and the noise set.
* Noise set: `n_noise` random wrong sequences sharing `noise_mass` (default off: n_noise = 0;
  each prompt's weights are renormalised, so `noise_mass` then has no effect).

Loss: weighted negative log-likelihood with token-level label smoothing `smoothing` (eps):
each target token is mixed with the uniform distribution, as if every training token were
replaced by a uniform random token with probability eps. This gives every step a floor of
about eps/V and so a broad, structured tail of near-miss sequences.

Why a noisy trap and a smoothed loss rather than one memorised trap: with a near-deterministic
trap, per-step tempering at beta < 1 flattens the one step where a hard solution leaves the
trap and lifts it above 0.5/N; and with a thin tail, sequence-level tempering at beta < 1 does
the same (q_beta(s) = q(s)^beta / sum q^beta is largest when the base's entropy is low). Spread
steps and a smoothed tail keep hard solutions below the line at every beta. A random noise set
also works in an exact fit but makes the GRU fit much worse and slower, so it is off.
See docs/base_verification.md.

Calibration (H11, D13): the hard weight `w` is searched in log space so that the median q_base
of the hard correct solutions is `target_ratio / n`. Each trial trains from the same seeded
initialisation; the search is a secant step in log w safeguarded by a bisection bracket.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from capsplit.engine import solution_probs, trajectory_probs

from ..interfaces import BaseRecipe, Task
from ..policies.prompt_gru import PromptGRUPolicy
from ..registry import BASES

DEFAULTS: dict[str, Any] = {
    # data
    "rho": 0.7,
    "mu": 0.5,
    "n_noise": 0,
    "noise_mass": 0.2,
    "easy_correct": 0.3,
    "easy_trap": 0.4,
    "easy_distractors": 4,
    "easy_distractor_mass": 0.1,
    "data_seed": 0,
    # network
    "embedding": 16,
    "hidden": 64,
    # optimisation
    "adam_steps": 100,
    "lr": 0.01,
    "lbfgs_iter": 500,
    "smoothing": 0.08,
    # calibration
    "n": 64,
    "target_ratio": 0.2,
    "calibration_tol": 1.1,  # stop when median q is within this factor of the target
    "calibration_max_iter": 6,
    "log10_w_range": [-6.0, -0.5],
}


@dataclass
class PretrainData:
    """Rows of (prompt id, sequence, weight); a row's weight is base + w * hard_scale, then each
    prompt's weights are divided by their sum."""

    prompt_ids: np.ndarray  # (R,)
    sequences: np.ndarray  # (R, L)
    base: np.ndarray  # (R,)
    hard_scale: np.ndarray  # (R,)

    def weights(self, w: float) -> np.ndarray:
        raw = self.base + w * self.hard_scale
        totals = np.bincount(self.prompt_ids, weights=raw)
        return raw / totals[self.prompt_ids]


def _random_wrong(rng: np.random.Generator, vocab: int, length: int, count: int, exclude: set) -> list:
    out: list = []
    seen = set(exclude)
    while len(out) < count:
        seq = tuple(int(t) for t in rng.integers(0, vocab, length))
        if seq not in seen:
            seen.add(seq)
            out.append(seq)
    return out


def build_data(task: Task, params: dict) -> PretrainData:
    """The weighted pre-training set for every prompt of a `rule_task`."""
    p = {**DEFAULTS, **params}
    vocab, length, k = task.n_actions, task.horizon, getattr(task, "shared_prefix")
    rng = np.random.default_rng([int(p["data_seed"]), 0xBA5E])
    rows: list = []  # (pid, seq, base, hard_scale)

    for prompt in task.prompts():
        pid = prompt.id
        correct = [tuple(s) for s in getattr(task, "correct_solutions")(pid)]
        correct_set = set(correct)
        trap = tuple(getattr(task, "trap")(pid))
        entries: dict = {}

        def add(seq, base=0.0, hard=0.0, entries=entries):
            b, h = entries.get(seq, (0.0, 0.0))
            entries[seq] = (b + base, h + hard)

        if not getattr(task, "is_hard")(pid):
            for seq in correct:
                add(seq, base=p["easy_correct"] / len(correct))
            add(trap, base=p["easy_trap"])
            n_d = int(p["easy_distractors"])
            for seq in _random_wrong(rng, vocab, length, n_d, correct_set | {trap}):
                add(seq, base=p["easy_distractor_mass"] / max(n_d, 1))
        else:
            rho, trap_mass = float(p["rho"]), 1.0 - float(p["noise_mass"])
            add(trap, base=trap_mass * rho**length)
            hard_k = correct[0][k]
            for t in range(length):
                for x in range(vocab):
                    seq = trap[:t] + (x,) + trap[t + 1 :]
                    if x == trap[t] or seq in correct_set or (t == k and x == hard_k):
                        continue
                    add(seq, base=trap_mass * rho**t * (1 - rho) / (vocab - 1))
            for c in correct:
                add(c, hard=1.0)
                for t in range(k + 1, length):
                    for x in range(vocab):
                        seq = c[:t] + (x,) + c[t + 1 :]
                        if x != c[t] and seq not in correct_set:
                            add(seq, hard=float(p["mu"]))
        n_noise = int(p["n_noise"])
        for seq in _random_wrong(rng, vocab, length, n_noise, correct_set | {trap}):
            add(seq, base=p["noise_mass"] / max(n_noise, 1))
        for seq, (b, h) in entries.items():
            rows.append((pid, seq, b, h))

    return PretrainData(
        prompt_ids=np.array([r[0] for r in rows], dtype=np.int64),
        sequences=np.array([r[1] for r in rows], dtype=np.int64).reshape(len(rows), length),
        base=np.array([r[2] for r in rows]),
        hard_scale=np.array([r[3] for r in rows]),
    )


def pretrain(
    policy: PromptGRUPolicy,
    task: Task,
    data: PretrainData,
    w: float,
    adam_steps: int,
    lr: float,
    lbfgs_iter: int,
    smoothing: float = 0.0,
) -> float:
    """Full-batch weighted maximum likelihood: Adam warm-up, then L-BFGS. Returns the final loss.

    The loss is the weighted negative log-likelihood in nats per prompt (each prompt's weights
    sum to 1). L-BFGS is used because the rare hard-solution mass carries tiny gradients, which
    Adam fits slowly and unevenly across prompts.
    """
    weights = torch.as_tensor(data.weights(w), dtype=torch.float32)
    actions = torch.as_tensor(data.sequences, dtype=torch.long)
    ids = data.prompt_ids.tolist()
    n_prompts = len(task.prompts())
    prompt_tokens = policy._prompt_tokens(task, ids)
    inputs = torch.cat((policy._start(len(ids)), actions[:, :-1]), dim=1)
    params = policy.parameters()

    def loss_fn() -> torch.Tensor:
        logits = policy.net.step(inputs, policy.net.initial_hidden(prompt_tokens))[0]
        log_pi = torch.log_softmax(logits, dim=-1)
        token = log_pi.gather(2, actions.unsqueeze(2)).squeeze(2)
        if smoothing > 0:
            token = (1.0 - smoothing) * token + smoothing * log_pi.mean(dim=2)
        return -(weights * token.sum(dim=1)).sum() / n_prompts

    adam = torch.optim.Adam(params, lr=lr)
    for _ in range(adam_steps):
        loss = loss_fn()
        adam.zero_grad()
        loss.backward()
        adam.step()
    if lbfgs_iter > 0:
        lbfgs = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=lbfgs_iter,
            max_eval=int(lbfgs_iter * 1.25),
            history_size=50,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            value = loss_fn()
            value.backward()
            return value

        lbfgs.step(closure)
    with torch.no_grad():
        return float(loss_fn())


def hard_correct_q(task: Task, policy: PromptGRUPolicy) -> np.ndarray:
    """q_base of every hard correct solution (hard prompts in id order, solutions in table order)."""
    hard = [p.id for p in task.prompts() if p.tags.get("difficulty") == "hard"]
    logits = policy.all_state_logits(task, hard)
    out = []
    for pid, lg in zip(hard, logits):
        table = task.table(pid)
        assert table is not None
        out.append(solution_probs(table, trajectory_probs(table, lg)))
    return np.concatenate(out)


@BASES.register("pretrain_gru")
class PretrainGRU(BaseRecipe):
    """Pre-train a `prompt_gru` policy on `build_data`, calibrating the hard weight (H11)."""

    def __init__(self, **params: Any):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown pretrain_gru params: {sorted(unknown)}")
        super().__init__(**params)

    def build(self, task: Task, seed: int) -> tuple[PromptGRUPolicy, dict]:
        p = {**DEFAULTS, **self.params}
        started = time.perf_counter()
        data = build_data(task, p)
        target = float(p["target_ratio"]) / float(p["n"])
        hard_rows = data.hard_scale > 0

        def train(w: float) -> tuple[PromptGRUPolicy, float, float]:
            policy = PromptGRUPolicy.for_task(task, embedding=int(p["embedding"]), hidden=int(p["hidden"]), seed=seed)
            loss = pretrain(
                policy, task, data, w, int(p["adam_steps"]), float(p["lr"]), int(p["lbfgs_iter"]), float(p["smoothing"])
            )
            return policy, float(np.median(hard_correct_q(task, policy))), loss

        # Start from the weight at which an exact fit of the data would hit the target:
        # q = w / (Z0 + w * H) per hard prompt (Z0, H are the same for every hard prompt).
        pid0 = int(data.prompt_ids[hard_rows][0]) if hard_rows.any() else 0
        z0 = float(data.base[data.prompt_ids == pid0].sum())
        h0 = float(data.hard_scale[data.prompt_ids == pid0].sum())
        low, high = (float(v) for v in p["log10_w_range"])
        log_w = math.log10(target * z0 / max(1e-12, 1.0 - target * h0))
        log_w = min(max(log_w, low), high)

        history: list = []
        best = None
        for _ in range(int(p["calibration_max_iter"])):
            policy, median, loss = train(10.0**log_w)
            error = math.log10(median / target)
            history.append({"log10_w": log_w, "median_hard_q": median, "loss": loss})
            if best is None or abs(error) < abs(best[0]):
                best = (error, policy, log_w, median, loss)
            if abs(error) <= math.log10(float(p["calibration_tol"])):
                break
            if error < 0:
                low = max(low, log_w)
            else:
                high = min(high, log_w)
            # Secant step on (log w, log q); slope 1 until two points exist.
            slope = 1.0
            if len(history) >= 2:
                a, b = history[-2], history[-1]
                dx = b["log10_w"] - a["log10_w"]
                dy = math.log10(b["median_hard_q"]) - math.log10(a["median_hard_q"])
                if dx != 0 and dy / dx > 0.1:
                    slope = dy / dx
            nxt = log_w - error / slope
            log_w = nxt if low < nxt < high else (low + high) / 2

        assert best is not None
        _, policy, log_w, median, loss = best
        info = {
            "recipe": "pretrain_gru",
            "seed": seed,
            "shared_prefix": getattr(task, "shared_prefix", None),
            "hard_weight": 10.0**log_w,
            "median_hard_q": median,
            "target_hard_q": target,
            "final_loss": loss,
            "calibration": history,
            "seconds": time.perf_counter() - started,
            "params": p,
        }
        return policy, info
