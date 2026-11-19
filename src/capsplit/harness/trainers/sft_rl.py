"""SFT-then-RL arm: behaviour cloning for part of the budget, then on-policy RL for the rest.

RL framing. The arm runs two update rules, one after the other, on the **same** policy:

1. **Supervised imitation (no reward).** For the first n_sft = round(sft_fraction * budget)
   updates it runs the SFT arm (`sft.py`): maximum-likelihood behaviour cloning of teacher
   demonstrations, loss -mean log pi_theta(tau | x). It never samples and never calls the
   verifier; by default it imitates only the hard train prompts.
2. **On-policy policy gradient with outcome reward.** For the remaining n_rl = budget - n_sft
   updates it runs an RL arm (`rl.py`, RLOO by default): sample G episodes per prompt from the
   current policy, reward r = 1[task.verify], leave-one-out advantage, one Adam step on
   -mean(A log pi_theta(tau)). The RL phase starts from the policy SFT produced.

Total optimizer updates are exactly `budget` (decision H5), the same as every other arm.

Checkpoints are on the **global** update count. Step 0 (before any update) and steps
1..n_sft come from the SFT phase; steps n_sft+1..budget come from the RL phase, whose local
step k is reported as n_sft + k. Each step is reported at most once (the boundary step n_sft
belongs to the SFT phase). If a phase has zero updates it is skipped entirely, so
sft_fraction = 0 is pure RL and sft_fraction = 1 is pure SFT.

Seeds. The two phases get distinct, deterministic seeds from `phase_seeds(seed)`:
numpy's ``SeedSequence([seed, 0])`` for SFT and ``SeedSequence([seed, 1])`` for RL, each
reduced to one 32-bit word. So a run is reproducible from `seed` and the phases never share
a random stream.

Optimizer. Each phase builds its own Adam optimizer (that is what the SFT and RL trainers
do), so Adam's moment estimates are reset at the phase boundary; the learning rates are set
separately in ``sft`` and ``rl``.

Parameters (unknown keys raise, decision H14):

* ``sft_fraction`` (float in [0, 1], default 0.5): share of the budget given to SFT
  (rounded with Python's ``round``, i.e. half to even).
* ``sft`` (dict, default {}): parameters for the SFT trainer.
* ``rl`` (dict, default {}): parameters for the RL trainer.
* ``rl_type`` (str, default "rloo"): any registered RL trainer name ("rloo", "grpo",
  "rl_entropy", "hints", ...).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..interfaces import CheckpointHook, Policy, Task, Trainer
from ..registry import TRAINERS
from .sft import SFT

__all__ = ["SFTThenRL", "phase_seeds", "split_budget"]

_KNOWN_PARAMS = {"sft_fraction", "sft", "rl", "rl_type"}
# Params understood by the RLOO family (rloo, grpo, rl_entropy), which does not validate itself.
_RLOO_FAMILY = {"rloo", "grpo", "rl_entropy"}
_RLOO_PARAMS = {"group", "lr", "betas", "entropy_coef", "prompts_per_update", "only_tags", "eps"}
_NOT_RL = {"sft", "none", "sft_then_rl"}


def phase_seeds(seed: int) -> tuple[int, int]:
    """(SFT seed, RL seed): distinct deterministic 32-bit seeds derived from `seed`."""
    sft_seed = int(np.random.SeedSequence([int(seed), 0]).generate_state(1)[0])
    rl_seed = int(np.random.SeedSequence([int(seed), 1]).generate_state(1)[0])
    return sft_seed, rl_seed


def split_budget(budget: int, sft_fraction: float) -> tuple[int, int]:
    """(n_sft, n_rl) with n_sft = round(sft_fraction * budget) and n_sft + n_rl = budget."""
    budget = max(int(budget), 0)
    n_sft = min(max(int(round(sft_fraction * budget)), 0), budget)
    return n_sft, budget - n_sft


@TRAINERS.register("sft_then_rl")
class SFTThenRL(Trainer):
    """Supervised imitation then on-policy RL on one policy. See the module docstring."""

    name = "sft_then_rl"

    def __init__(self, **params: Any):
        unknown = set(params) - _KNOWN_PARAMS
        if unknown:
            raise ValueError(f"sft_then_rl: unknown parameters {sorted(unknown)}; known: {sorted(_KNOWN_PARAMS)}")
        super().__init__(**params)
        fraction = params.get("sft_fraction", 0.5)
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0.0 <= float(fraction) <= 1.0:
            raise ValueError(f"sft_then_rl: sft_fraction must be a number in [0, 1], got {fraction!r}")
        self.sft_fraction = float(fraction)
        sft_params = params.get("sft") or {}
        rl_params = params.get("rl") or {}
        if not isinstance(sft_params, Mapping) or not isinstance(rl_params, Mapping):
            raise ValueError("sft_then_rl: 'sft' and 'rl' must be dicts of parameters")
        self.rl_type = str(params.get("rl_type", "rloo"))
        if self.rl_type in _NOT_RL:
            raise ValueError(f"sft_then_rl: rl_type must be an RL trainer, got {self.rl_type!r}")
        if self.rl_type in _RLOO_FAMILY:
            bad = set(rl_params) - _RLOO_PARAMS
            if bad:
                raise ValueError(f"sft_then_rl: unknown rl parameters {sorted(bad)}; known: {sorted(_RLOO_PARAMS)}")
        # Building both sub-trainers now validates their params (and rl_type) at config time.
        self.sft_trainer: Trainer = SFT(**dict(sft_params))
        self.rl_trainer: Trainer = TRAINERS.create(self.rl_type, **dict(rl_params))

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
        n_sft, n_rl = split_budget(budget, self.sft_fraction)
        sft_seed, rl_seed = phase_seeds(seed)
        wanted = sorted({int(c) for c in checkpoints})
        done: set[int] = set()

        def report(step: int, p: Policy) -> None:
            if step not in done:  # never report a global step twice
                done.add(step)
                on_checkpoint(step, p)

        ran_sft = n_sft > 0
        if ran_sft:
            # Global steps 0..n_sft; local step == global step.
            sft_cps = [c for c in wanted if 0 <= c <= n_sft]
            self.sft_trainer.train(policy, task, n_sft, sft_seed, prompt_ids, sft_cps, report)

        if n_rl > 0:
            # Local step k is global step n_sft + k. Local step 0 is already reported by SFT.
            lo = 1 if ran_sft else 0
            rl_cps = [c - n_sft for c in wanted if lo <= c - n_sft <= n_rl]

            def rl_report(local: int, p: Policy) -> None:
                report(local + n_sft, p)

            self.rl_trainer.train(policy, task, n_rl, rl_seed, prompt_ids, rl_cps, rl_report)

        if not ran_sft and n_rl == 0 and 0 in wanted:
            report(0, policy)  # budget 0: only the "before any update" checkpoint exists
