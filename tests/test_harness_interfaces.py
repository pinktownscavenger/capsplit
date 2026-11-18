"""T0: the plug-in interfaces, registries and a trivial end-to-end run with dummy plug-ins."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from capsplit.engine import trajectory_probs  # noqa: E402
from capsplit.harness import BACKENDS, BASES, TASKS, TRAINERS, MeasureSpec, Registry, Trainer  # noqa: E402
from capsplit.harness.policies.tabular import TabularSequencePolicy  # noqa: E402
from capsplit.harness.sequence import compile_prefix_tree, prefix_nodes  # noqa: E402
from capsplit.seqtask.verifier import compile_sequence, make_task  # noqa: E402


def _task():
    return TASKS.create("tiny_prompt")


def test_registry_create_and_unknown_name():
    assert "tiny_prompt" in TASKS.names()
    assert "none" in TRAINERS.names()
    with pytest.raises(KeyError, match="unknown trainer"):
        TRAINERS.get("no-such-trainer")


def test_registry_rejects_a_second_class_under_one_name():
    registry = Registry("thing")

    @registry.register("a")
    class A:
        pass

    with pytest.raises(ValueError):

        @registry.register("a")
        class B:
            pass


def test_compile_prefix_tree_matches_the_existing_compiler():
    seq_task = make_task(4, 3, 2, 2, 1, seed=0, n_distractors=5)
    old = compile_sequence(seq_task)
    new = compile_prefix_tree(4, 3, seq_task.common + seq_task.rare)
    np.testing.assert_array_equal(old.index, new.index)
    np.testing.assert_array_equal(old.solution_ids, new.solution_ids)
    assert old.solutions == new.solutions and old.n_states == new.n_states


def test_task_contract():
    task = _task()
    assert [p.id for p in task.prompts()] == list(range(4))
    assert task.prompt_ids(difficulty="hard", split="train") == [1]
    for pid in range(4):
        for solution in task.correct_solutions(pid):
            assert task.verify(pid, solution)
        table = task.table(pid)
        assert (table.solution_ids >= 0).sum() == len(set(task.correct_solutions(pid)))
    state, done = task.step(0, task.initial_state(0), 2)
    assert state == (2,) and not done


def test_sampled_log_prob_matches_teacher_forcing_and_the_exact_table():
    task = _task()
    policy = TabularSequencePolicy(4, 3, 3, init_scale=1.0, seed=1)
    generator = torch.Generator().manual_seed(0)
    rollout = policy.sample(task, [0, 1, 2, 3] * 5, generator)
    forced = policy.log_prob(task, rollout.prompt_ids, rollout.actions)
    torch.testing.assert_close(rollout.log_prob, forced)
    # Exact probabilities from state_logits agree with the torch log-probabilities.
    for pid, actions, lp in zip(rollout.prompt_ids, rollout.actions, forced.tolist()):
        table = task.table(pid)
        probs = trajectory_probs(table, policy.state_logits(task, pid))
        row = int(np.dot(actions, 3 ** np.arange(2, -1, -1)))
        assert np.isclose(np.log(probs[row]), lp, atol=1e-5)
    assert prefix_nodes(3, 3, np.array([[0, 0, 0]])).tolist() == [[0, 1, 4]]


def test_forced_prefixes_are_taken_and_excluded_from_log_prob():
    task = _task()
    policy = TabularSequencePolicy(4, 3, 3, init_scale=1.0, seed=2)
    rollout = policy.sample(task, [1, 1], torch.Generator().manual_seed(0), prefixes=[(2, 1), ()])
    assert rollout.actions[0][:2] == (2, 1) and rollout.n_forced == [2, 0]
    suffix = policy.log_prob(task, [1], [rollout.actions[0]], start=2)
    torch.testing.assert_close(rollout.log_prob[:1], suffix)


def test_sampling_is_reproducible_from_the_generator():
    task = _task()
    policy = TabularSequencePolicy(4, 3, 3, init_scale=1.0, seed=3)
    a = policy.sample(task, [0, 1, 2], torch.Generator().manual_seed(7)).actions
    b = policy.sample(task, [0, 1, 2], torch.Generator().manual_seed(7)).actions
    assert a == b


def test_trivial_end_to_end_with_dummy_plugins():
    task = _task()
    base, info = BASES.create("uniform_tabular").build(task, seed=0)
    trained = base.clone()
    seen = []
    trainer: Trainer = TRAINERS.create("none")
    trainer.train(trained, task, budget=10, seed=0, prompt_ids=task.prompt_ids(split="train"),
                  checkpoints=[0, 5, 10], on_checkpoint=lambda step, p: seen.append(step))
    assert seen == [0, 5, 10]
    spec = MeasureSpec(ns=(4,), cs=(1.0,), groups={"all": tuple(range(4))})
    backend = BACKENDS.create("j_only")
    metrics = backend.measure(task, base, trained, spec, backend.prepare(task, base, spec))
    (m,) = metrics
    assert m.group == "all" and m.j0 == m.j1
    assert np.isclose(m.j0, 2 / 27)  # two correct sequences out of 27 under the uniform policy
    assert m.to_dict()["delta_total"] == 0.0


def test_registry_skips_only_modules_that_do_not_exist():
    from capsplit.harness.registry import _is_missing

    assert _is_missing("capsplit.harness.trainers.sft", "capsplit.harness.trainers.sft")
    assert _is_missing("capsplit.harness.trainers.sft", "capsplit.harness.trainers")
    # sft_rl exists but imports a missing sibling: that is a real error, not a missing plug-in.
    assert not _is_missing("capsplit.harness.trainers.sft_rl", "capsplit.harness.trainers.sft")
    assert not _is_missing("capsplit.harness.trainers.rl", "torch")
    assert not _is_missing("capsplit.harness.trainers.rl", None)


def test_a_broken_plugin_raises_on_every_lookup(monkeypatch, tmp_path):
    import sys

    import capsplit.harness.registry as registry

    (tmp_path / "broken_plugin_xyz.py").write_text("raise RuntimeError('boom')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(registry, "PLUGIN_MODULES", registry.PLUGIN_MODULES + ("broken_plugin_xyz",))
    monkeypatch.setattr(registry, "_loaded", False)
    for _ in range(2):
        sys.modules.pop("broken_plugin_xyz", None)
        with pytest.raises(RuntimeError, match="boom"):
            registry.TRAINERS.names()
