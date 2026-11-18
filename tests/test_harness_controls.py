"""T4E: control arms. "none" changes nothing and measures exactly zero gain; "rl_entropy" moves the policy."""
import pytest

torch = pytest.importorskip("torch")

from capsplit.harness import BACKENDS, TASKS, TRAINERS  # noqa: E402
from capsplit.harness.interfaces import MeasureSpec  # noqa: E402
from capsplit.harness.policies.tabular import TabularSequencePolicy  # noqa: E402


def _task():
    return TASKS.create("tiny_prompt")


def _base(task, seed=0):
    return TabularSequencePolicy(len(task.prompts()), task.n_actions, task.horizon, init_scale=1.0, seed=seed)


def _spec(task):
    groups = {
        "all": tuple(p.id for p in task.prompts()),
        "hard_train": tuple(task.prompt_ids(difficulty="hard", split="train")),
        "easy_heldout": tuple(task.prompt_ids(difficulty="easy", split="heldout")),
    }
    return MeasureSpec(ns=(1, 4, 16, 64), cs=(0.5, 1.0, 2.0), groups=groups)


def _none_trained(task, base):
    trained = base.clone()
    seen = []
    TRAINERS.create("none").train(trained, task, 20, 0, task.prompt_ids(split="train"), [0, 10, 20],
                                  lambda s, p: seen.append(s))
    assert seen == [0, 10, 20]
    return trained


def test_none_leaves_parameters_bit_identical():
    task = _task()
    base = _base(task)
    trained = _none_trained(task, base)
    for a, b in zip(base.parameters(), trained.parameters()):
        assert torch.equal(a.detach(), b.detach())
        assert a is not b  # a clone, not the same object
    assert torch.equal(base.state_dict()["logits"], trained.state_dict()["logits"])


def _assert_exact_zero(metrics, spec):
    assert len(metrics) == len(spec.groups) * len(spec.ns) * len(spec.cs)
    for m in metrics:
        assert m.delta_total == 0.0, m
        assert m.delta_rel == 0.0, m
        assert m.delta_cap == 0.0, m
        assert m.j0 == m.j1, m


@pytest.mark.parametrize("backend_name", ["j_only", "exact"])
def test_none_gives_exactly_zero_gain(backend_name):
    if backend_name == "exact":
        pytest.importorskip("capsplit.harness.backends.exact")
    task = _task()
    base = _base(task)
    trained = _none_trained(task, base)
    spec = _spec(task)
    backend = BACKENDS.create(backend_name)
    metrics = backend.measure(task, base, trained, spec, backend.prepare(task, base, spec))
    _assert_exact_zero(metrics, spec)
    if backend_name == "exact":
        for m in metrics:
            assert all(v == 0.0 for v in m.extra["delta_rel"] + m.extra["delta_cap"]), m
            assert m.extra["crossed_count"] == 0, m


def test_entropy_bonus_arm_is_registered_and_moves_parameters():
    assert "rl_entropy" in TRAINERS.names()
    task = _task()
    base = _base(task)
    trained = base.clone()
    trainer = TRAINERS.create("rl_entropy", group=4, lr=0.1)
    assert trainer.entropy_coef > 0
    trainer.train(trained, task, 5, 0, task.prompt_ids(split="train"), [], lambda s, p: None)
    assert not torch.equal(base.logits.detach(), trained.logits.detach())
    # Held-out prompts are never trained.
    for pid in task.prompt_ids(split="heldout"):
        assert torch.equal(base.logits.detach()[pid], trained.logits.detach()[pid])
