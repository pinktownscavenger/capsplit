"""T4B: the supervised fine-tuning (behaviour cloning) arm."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from capsplit.engine import trajectory_probs  # noqa: E402
from capsplit.harness import TASKS, TRAINERS  # noqa: E402
from capsplit.harness.policies.tabular import TabularSequencePolicy  # noqa: E402
from capsplit.harness.trainers.sft import SFT, demonstration_pairs, sft_loss, sft_step  # noqa: E402


def _task():
    return TASKS.create("tiny_prompt")


def _policy(task, seed=0):
    return TabularSequencePolicy(len(task.prompts()), task.n_actions, task.horizon, init_scale=0.5, seed=seed)


def _train(policy, task, budget=30, seed=0, checkpoints=(), hook=None, **params):
    params.setdefault("lr", 0.1)
    trainer = TRAINERS.create("sft", **params)
    trainer.train(policy, task, budget, seed, task.prompt_ids(split="train"), list(checkpoints), hook or (lambda s, p: None))


def _demo_prob(task, policy, pid):
    """Exact probability mass the policy puts on the prompt's demonstrations."""
    table = task.table(pid)
    probs = trajectory_probs(table, policy.state_logits(task, pid))
    demo_rows = [i for i, sid in enumerate(table.solution_ids) if sid >= 0 and table.solutions[sid] in set(task.demonstrations(pid))]
    return float(probs[demo_rows].sum())


def test_registered_and_selects_hard_train_prompts_only():
    task = _task()
    assert isinstance(TRAINERS.create("sft"), SFT)
    train = task.prompt_ids(split="train")
    pairs = demonstration_pairs(task, train)
    hard_train = [p for p in train if task.prompts()[p].tags["difficulty"] == "hard"]
    assert {pid for pid, _ in pairs} == set(hard_train)
    assert len(pairs) == sum(len(task.demonstrations(p)) for p in hard_train)
    assert len(demonstration_pairs(task, train, demos_per_prompt=1)) == len(hard_train)
    # A filter that matches nothing among the given prompts is a clear error.
    with pytest.raises(ValueError, match="no prompt"):
        demonstration_pairs(task, task.prompt_ids(difficulty="easy"))
    with pytest.raises(ValueError, match="no prompt"):
        _train(_policy(task), task, only_tags={"difficulty": "impossible"})
    with pytest.raises(ValueError, match="unknown parameters"):
        TRAINERS.create("sft", learning_rate=0.1)


def test_only_hard_train_prompts_change():
    task = _task()
    policy = _policy(task)
    before = policy.logits.detach().clone()
    _train(policy, task, budget=20)
    after = policy.logits.detach()
    train = set(task.prompt_ids(split="train"))
    for p in task.prompts():
        if p.tags["difficulty"] == "hard" and p.id in train:
            assert not torch.equal(before[p.id], after[p.id]), p
        else:
            assert torch.equal(before[p.id], after[p.id]), p


def test_demonstration_likelihood_rises():
    task = _task()
    policy = _policy(task)
    hard_train = task.prompt_ids(difficulty="hard", split="train")
    before = [_demo_prob(task, policy, p) for p in hard_train]
    _train(policy, task, budget=50)
    after = [_demo_prob(task, policy, p) for p in hard_train]
    for b, a in zip(before, after):
        assert a > 0.9 and a > b + 0.5, (b, a)


def test_loss_is_negative_mean_teacher_forced_log_likelihood():
    task = _task()
    policy = _policy(task)
    pairs = demonstration_pairs(task, task.prompt_ids(split="train"), only_tags=None)
    by_hand = -np.mean([float(policy.log_prob(task, [pid], [demo]).detach()[0]) for pid, demo in pairs])
    assert float(sft_loss(policy, task, pairs).detach()) == pytest.approx(by_hand, abs=1e-6)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.1)
    before = policy.logits.detach().clone()
    assert sft_step(policy, task, optimizer, pairs) == pytest.approx(by_hand, abs=1e-6)
    assert not torch.equal(before, policy.logits.detach())

    # The trainer's first update is exactly one sft_step on the full batch.
    a, b = _policy(task), _policy(task)
    _train(a, task, budget=1)
    sft_step(b, task, torch.optim.Adam(b.parameters(), lr=0.1), demonstration_pairs(task, task.prompt_ids(split="train")))
    assert torch.equal(a.logits, b.logits)


def test_checkpoints_at_exactly_the_requested_steps():
    task = _task()
    calls = []
    _train(_policy(task), task, budget=5, checkpoints=[0, 2, 5, 9], hook=lambda s, p: calls.append(s))
    assert calls == [0, 2, 5]

    calls.clear()
    seen = []
    policy = _policy(task)
    before = policy.logits.detach().clone()
    _train(policy, task, budget=0, checkpoints=[0, 1, 3], hook=lambda s, p: (calls.append(s), seen.append(p.logits.detach().clone())))
    assert calls == [0]
    assert torch.equal(seen[0], before) and torch.equal(policy.logits, before)


def test_same_seed_same_parameters_and_minibatch_depends_on_seed():
    task = _task()

    def run(seed):
        policy = _policy(task)
        _train(policy, task, budget=8, seed=seed, batch=1)
        return policy.logits.detach().clone()

    assert torch.equal(run(0), run(0))
    assert any(not torch.equal(run(0), run(s)) for s in (1, 2, 3))
    # Full batch ignores the seed entirely.
    full = []
    for seed in (0, 1):
        policy = _policy(task)
        _train(policy, task, budget=3, seed=seed)
        full.append(policy.logits.detach().clone())
    assert torch.equal(full[0], full[1])
