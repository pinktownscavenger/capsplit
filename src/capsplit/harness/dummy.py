"""Dummy plug-ins for interface tests: a tiny prompted task, a uniform base and a J-only backend.

These are deliberately trivial. Real implementations live in tasks/, bases/ and backends/.
"""
from __future__ import annotations

import numpy as np

from capsplit.engine import table_success_prob, trajectory_probs

from .interfaces import Backend, BaseRecipe, GroupMetrics, Prompt
from .policies.tabular import TabularSequencePolicy
from .registry import BACKENDS, BASES, TASKS
from .sequence import SequenceTask

_TAGS = (("easy", "train"), ("hard", "train"), ("easy", "heldout"), ("hard", "heldout"))


@TASKS.register("tiny_prompt")
class TinyPromptTask(SequenceTask):
    """V=3, L=3, four prompts. Prompt i accepts (i%3, i%3, i%3) and ((i+1)%3, i%3, 0)."""

    def __init__(self, vocab: int = 3, length: int = 3):
        self.n_actions, self.horizon = vocab, length
        self._prompts = tuple(
            Prompt(i, (i,), {"difficulty": d, "split": s}) for i, (d, s) in enumerate(_TAGS)
        )

    def prompts(self):
        return self._prompts

    def correct_solutions(self, prompt_id):
        a = prompt_id % self.n_actions
        b = (prompt_id + 1) % self.n_actions
        return ((a,) * self.horizon, (b, a) + (0,) * (self.horizon - 2))


@BASES.register("uniform_tabular")
class UniformTabularBase(BaseRecipe):
    """A tabular policy with all-zero logits (uniform over tokens), or small random ones."""

    def build(self, task, seed):
        policy = TabularSequencePolicy(
            len(task.prompts()), task.n_actions, task.horizon, init_scale=self.params.get("init_scale", 0.0), seed=seed
        )
        return policy, {"recipe": "uniform_tabular"}


@BACKENDS.register("j_only")
class JOnlyBackend(Backend):
    """Exact J per group; puts the whole change in delta_rel. For interface tests only."""

    def prepare(self, task, base, spec):
        return None

    def measure(self, task, base, trained, spec, cache):
        def j(policy, pid):
            table = task.table(pid)
            return table_success_prob(table, trajectory_probs(table, policy.state_logits(task, pid)))

        out = []
        for group, ids in spec.groups.items():
            j0 = float(np.mean([j(base, p) for p in ids]))
            j1 = float(np.mean([j(trained, p) for p in ids]))
            for n in spec.ns:
                for c in spec.cs:
                    out.append(GroupMetrics(group, n, c, len(ids), j0, j1, j1 - j0, 0.0, {}, "unmeasured"))
        return out
