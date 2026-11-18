"""Gate 1: planted cases on the frozen prompt-conditioned base are labelled correctly.

Each planted "trained policy" is built directly from the real base (k = 2, seed 0), so the
true answer is known by construction:

* pure sharpening: the base re-tempered (per-step and sequence-level, beta = 2). It is a
  member of the tempering family, so E_N <= 0, and the base design guarantees that no
  tempering lifts a hard solution over c/N, so nothing crosses.
* capability: on hard train prompts, half of the base's mass is moved onto the prompt's
  correct solutions. They cross c/N, and no tempered base can match it (criterion b).
* reweighting: on easy prompts, half of the incorrect mass is moved onto the correct
  solutions, which were already reachable. E_N > 0, nothing crosses.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from capsplit.engine import trajectory_probs  # noqa: E402
from capsplit.harness import BASES, TASKS, MeasureSpec, Policy  # noqa: E402
from capsplit.harness.backends.exact import ExactBackend  # noqa: E402
from capsplit.harness.config import build_groups, load_config  # noqa: E402
from capsplit.tempering import sequence_probs  # noqa: E402

N, C = 64, 1.0


class PlantedPolicy(Policy):
    """A fixed distribution over whole trajectories, per prompt."""

    def __init__(self, probs: dict):
        self.probs = probs

    def trajectory_probs(self, task, prompt_id):
        return self.probs[prompt_id]

    def tempering_logits(self, task, prompt_id):
        return None

    def sample(self, *a, **k):
        raise NotImplementedError

    def log_prob(self, *a, **k):
        raise NotImplementedError

    def state_logits(self, task, prompt_id):
        raise NotImplementedError

    def parameters(self):
        return []

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


@pytest.fixture(scope="module")
def setup():
    cfg = load_config("configs/main.json")
    params = dict(cfg.task.params, shared_prefix=2)
    task = TASKS.create(cfg.task.type, **params)
    base, _ = BASES.create(cfg.base.type, **cfg.base.params).build(task, seed=0)
    groups = build_groups(task, ["difficulty", "split"])
    spec = MeasureSpec(ns=(N,), cs=(C,), groups=groups)
    backend = ExactBackend()
    cache = backend.prepare(task, base, spec)
    probs0 = {p.id: np.asarray(base.trajectory_probs(task, p.id), dtype=float) for p in task.prompts()}
    return task, base, spec, backend, cache, probs0


def _labels(setup, policy):
    task, base, spec, backend, cache, _ = setup
    return {m.group: m for m in backend.measure(task, base, policy, spec, cache)}


def _move_to_correct(task, pid, probs, share, incorrect_only=False):
    table = task.table(pid)
    correct = table.solution_ids >= 0
    out = probs.copy()
    source = ~correct if incorrect_only else np.ones_like(correct)
    moved = out[source].sum() * share
    out[source] *= 1 - share
    out[correct] += moved * out[correct] / out[correct].sum()
    return out


@pytest.mark.parametrize("family", ["per_step", "sequence"])
def test_planted_pure_sharpening(setup, family):
    task, base, *_ , probs0 = setup
    beta = 2.0
    planted = {}
    for p in task.prompts():
        logits = base.state_logits(task, p.id)
        table = task.table(p.id)
        planted[p.id] = trajectory_probs(table, logits, beta) if family == "per_step" else sequence_probs(table, logits, beta)
    metrics = _labels(setup, PlantedPolicy(planted))
    for group in ("all", "hard_train", "hard_heldout", "easy_train"):
        assert metrics[group].label == "pure_sharpening", (group, metrics[group].label)
        assert metrics[group].extra["crossed_count"] == 0


def test_planted_capability_on_hard_train(setup):
    task, *_, probs0 = setup
    hard_train = set(task.prompt_ids(difficulty="hard", split="train"))
    planted = {pid: (_move_to_correct(task, pid, pr, 0.5) if pid in hard_train else pr) for pid, pr in probs0.items()}
    metrics = _labels(setup, PlantedPolicy(planted))
    assert metrics["hard_train"].label == "capability"
    assert metrics["hard_train"].delta_cap > 0.4 and abs(metrics["hard_train"].delta_rel) < 1e-12
    assert metrics["hard_heldout"].label == "pure_sharpening"  # untouched prompts
    assert metrics["hard_heldout"].delta_cap == 0.0


def test_planted_reweighting_on_easy(setup):
    task, *_, probs0 = setup
    easy = set(task.prompt_ids(difficulty="easy"))
    planted = {pid: (_move_to_correct(task, pid, pr, 0.5, incorrect_only=True) if pid in easy else pr) for pid, pr in probs0.items()}
    metrics = _labels(setup, PlantedPolicy(planted))
    for group in ("easy_train", "easy_heldout"):
        assert metrics[group].label == "reweighting", (group, metrics[group].label)
        assert metrics[group].delta_cap == 0.0 and metrics[group].delta_rel > 0.2


def test_identity_is_zero_gain(setup):
    task, base, *_ = setup
    metrics = _labels(setup, base.clone())
    for m in metrics.values():
        assert m.delta_rel == 0.0 and m.delta_cap == 0.0 and m.j1 == m.j0
        assert m.label == "pure_sharpening"
