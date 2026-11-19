"""T4C: the hint arm (RLOO with demonstration-prefix re-rolls of all-fail groups)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from capsplit.engine import table_success_prob, trajectory_probs  # noqa: E402
from capsplit.harness import TASKS, TRAINERS  # noqa: E402
from capsplit.harness.policies.tabular import TabularSequencePolicy  # noqa: E402
from capsplit.harness.sequence import n_nodes, node_offsets  # noqa: E402
from capsplit.harness.trainers.hints import loo_advantages  # noqa: E402

HARD = 1  # tiny_prompt prompt 1 (hard/train) accepts (1, 1, 1) and (2, 1, 0)
EASY = 0  # prompt 0 accepts (0, 0, 0) and (1, 0, 0)


def _task():
    return TASKS.create("tiny_prompt")


def _policy(task, root_bias=12.0, easy_bias=0.0):
    """Uniform everywhere, except: on the hard prompt the root strongly prefers token 0, so no
    correct sequence (both start with 1 or 2) is ever sampled; optionally the easy prompt
    strongly prefers its correct sequence (0, 0, 0)."""
    v, length = task.n_actions, task.horizon
    logits = np.zeros((len(task.prompts()), n_nodes(v, length), v), dtype=np.float32)
    logits[HARD, 0, 0] = root_bias
    if easy_bias:
        offsets = node_offsets(v, length)
        for level in range(length):
            logits[EASY, offsets[level], 0] = easy_bias  # node of prefix (0,)*level has code 0
    return TabularSequencePolicy(len(task.prompts()), v, length, logits=logits)


def _exact_j(task, policy, pid):
    table = task.table(pid)
    return table_success_prob(table, trajectory_probs(table, policy.state_logits(task, pid)))


def _train(policy, task, budget=1, seed=0, ids=(HARD,), checkpoints=(), hook=None, **params):
    params.setdefault("group", 32)
    params.setdefault("lr", 0.1)
    trainer = TRAINERS.create("hints", **params)
    trainer.train(policy, task, budget, seed, list(ids), list(checkpoints), hook or (lambda s, p: None))
    return trainer


def test_loo_advantages():
    assert torch.all(loo_advantages(torch.zeros(2, 4)) == 0)  # plain RLOO: no signal when all fail
    adv = loo_advantages(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(adv.numpy(), [[1.0, -1 / 3, -1 / 3, -1 / 3]])


@pytest.mark.parametrize("hint_len", [1, 2])
def test_all_fail_group_is_hinted_and_forced_nodes_get_no_gradient(hint_len):
    task = _task()
    policy = _policy(task)
    before = policy.logits.detach().clone()
    trainer = _train(policy, task, hint_len=hint_len)

    assert trainer.hinted_per_update == [1]
    assert trainer.history[0]["mean_reward_unhinted"] == 0.0  # plain RLOO would get zero gradient
    demo_prefixes = {tuple(d[:hint_len]) for d in task.demonstrations(HARD)}
    episodes = trainer.last_episodes
    assert len(episodes) == 32 and all(e["hinted"] and e["n_forced"] == hint_len for e in episodes)
    assert all(e["actions"][:hint_len] in demo_prefixes for e in episodes)
    assert len({e["actions"][:hint_len] for e in episodes}) == 1  # one demonstration per group
    assert any(e["reward"] == 1.0 for e in episodes)

    after = policy.logits.detach()
    assert not torch.equal(after, before)  # nonzero gradient, unlike plain RLOO
    # The forced prefix nodes are never the source of a generated action: unchanged exactly.
    prefix = episodes[0]["actions"][:hint_len]
    offsets = node_offsets(task.n_actions, task.horizon)
    code = 0
    for level in range(hint_len):
        node = offsets[level] + code
        assert torch.equal(after[HARD, node], before[HARD, node])
        code = code * task.n_actions + prefix[level]
    # Other prompts were not trained on.
    assert torch.equal(after[EASY], before[EASY])


def test_group_with_a_success_is_not_hinted():
    task = _task()
    policy = _policy(task, easy_bias=12.0)
    trainer = _train(policy, task, ids=(EASY, HARD), hint_len=1)
    assert trainer.history[0]["hinted_prompts"] == [HARD]
    assert all(not e["hinted"] and e["n_forced"] == 0 for e in trainer.last_episodes if e["prompt_id"] == EASY)

    always = _train(_policy(task, easy_bias=12.0), task, ids=(EASY, HARD), hint_len=1, hint_when="always")
    assert always.history[0]["hinted_prompts"] == [EASY, HARD]


def test_only_tags_filters_prompts():
    task = _task()
    trainer = _train(_policy(task), task, ids=(0, 1, 2, 3), hint_len=1, only_tags={"difficulty": "hard", "split": "train"})
    assert trainer.history[0]["n_groups"] == 1 and {e["prompt_id"] for e in trainer.last_episodes} == {HARD}


def test_training_raises_unhinted_success_on_the_hard_prompt():
    task = _task()
    policy = _policy(task)
    j0 = _exact_j(task, policy, HARD)
    trainer = _train(policy, task, budget=40, hint_len=1)
    j1 = _exact_j(task, policy, HARD)
    assert j1 > 3 * j0  # measured without any hint
    assert sum(trainer.hinted_per_update) == 40


def test_two_hint_lengths_differ_and_invalid_hint_len_raises():
    task = _task()
    a, b = _policy(task), _policy(task)
    _train(a, task, budget=3, hint_len=1)
    _train(b, task, budget=3, hint_len=2)
    assert not torch.equal(a.logits, b.logits)

    for bad in (0, -1, 1.5, True, "2"):
        with pytest.raises(ValueError):
            TRAINERS.create("hints", hint_len=bad)
    with pytest.raises(ValueError):
        TRAINERS.create("hints")
    with pytest.raises(ValueError):
        _train(_policy(task), task, hint_len=task.horizon)
    with pytest.raises(ValueError):
        TRAINERS.create("hints", hint_len=1, hint_when="never")


def test_checkpoints_and_reproducibility():
    task = _task()
    steps, snaps = [], []

    def hook(step, policy):
        steps.append(step)
        snaps.append(policy.logits.detach().clone())

    p1 = _policy(task)
    _train(p1, task, budget=4, seed=7, ids=(0, 1, 2, 3), checkpoints=(0, 2, 4, 9), hook=hook, hint_len=1, group=8)
    assert steps == [0, 2, 4]
    assert torch.equal(snaps[0], _policy(task).logits.detach())
    assert torch.equal(snaps[-1], p1.logits.detach())

    p2 = _policy(task)
    _train(p2, task, budget=4, seed=7, ids=(0, 1, 2, 3), hint_len=1, group=8)
    assert torch.equal(p1.logits, p2.logits)
    p3 = _policy(task)
    _train(p3, task, budget=4, seed=8, ids=(0, 1, 2, 3), hint_len=1, group=8)
    assert not torch.equal(p1.logits, p3.logits)
