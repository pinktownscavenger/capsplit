"""Tests for the RL post-training arms (task T4A): RLOO, GRPO and the entropy-bonus control."""
import numpy as np
import pytest
import torch

from capsplit.engine import table_success_prob, trajectory_probs
from capsplit.harness import TRAINERS
from capsplit.harness.dummy import TinyPromptTask
from capsplit.harness.policies.tabular import TabularSequencePolicy
from capsplit.harness.trainers.rl import (
    GRPO,
    RLOO,
    EntropyRLOO,
    episode_rewards,
    group_advantages,
    rl_loss,
    select_prompts,
)

ARMS = ("rloo", "grpo", "rl_entropy")


class NoSolutionTask(TinyPromptTask):
    """TinyPromptTask in which prompt 0 has no correct sequence, so every sample fails."""

    def correct_solutions(self, prompt_id):
        return () if prompt_id == 0 else super().correct_solutions(prompt_id)


def _policy(task, init_scale=0.0, seed=0):
    return TabularSequencePolicy(len(task.prompts()), task.n_actions, task.horizon, init_scale=init_scale, seed=seed)


def _exact_j(task, policy, ids):
    out = []
    for pid in ids:
        table = task.table(pid)
        out.append(table_success_prob(table, trajectory_probs(table, policy.state_logits(task, pid))))
    return float(np.mean(out))


def _train(name, policy, task, budget, seed, ids, checkpoints=(), hook=None, **params):
    trainer = TRAINERS.create(name, **params)
    trainer.train(policy, task, budget, seed, ids, list(checkpoints), hook or (lambda s, p: None))
    return trainer


def test_registered_and_defaults():
    assert isinstance(TRAINERS.create("rloo"), RLOO)
    assert isinstance(TRAINERS.create("grpo"), GRPO)
    assert isinstance(TRAINERS.create("rl_entropy"), EntropyRLOO)
    assert TRAINERS.create("rloo").entropy_coef == 0.0
    assert TRAINERS.create("grpo").entropy_coef == 0.0
    assert TRAINERS.create("rl_entropy").entropy_coef == 0.01
    assert TRAINERS.create("rl_entropy", entropy_coef=0.1).entropy_coef == 0.1
    assert TRAINERS.create("rloo").group == 16 and TRAINERS.create("rloo").lr == 1e-3
    with pytest.raises(ValueError):
        TRAINERS.create("rloo", group=1)


# ----------------------------------------------------------------- advantages


def test_loo_advantages_hand_checked():
    r = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0])  # two groups of 4
    adv = group_advantages(r, 4, "rloo")
    # group 1: others' means are 1/3, 2/3, 2/3, 1/3
    # group 2: others' means are 2/3, 2/3, 2/3, 1
    expected = torch.tensor([2 / 3, -2 / 3, -2 / 3, 2 / 3, 1 / 3, 1 / 3, 1 / 3, -1.0])
    assert torch.allclose(adv, expected, atol=1e-6)


def test_grpo_advantages_hand_checked():
    r = torch.tensor([1.0, 0.0, 0.0, 0.0])  # mean 0.25, population std sqrt(3)/4
    adv = group_advantages(r, 4, "grpo")
    std = np.sqrt(3) / 4
    expected = torch.tensor([0.75, -0.25, -0.25, -0.25]) / (std + 1e-6)
    assert torch.allclose(adv, expected.float(), atol=1e-5)
    assert abs(float(adv.mean())) < 1e-6


@pytest.mark.parametrize("kind", ["rloo", "grpo"])
@pytest.mark.parametrize("value", [0.0, 1.0])
def test_all_equal_rewards_give_exactly_zero_advantage(kind, value):
    adv = group_advantages(torch.full((12,), value), 4, kind)
    assert torch.equal(adv, torch.zeros(12))


def test_advantage_input_validation():
    with pytest.raises(ValueError):
        group_advantages(torch.zeros(5), 4)
    with pytest.raises(ValueError):
        group_advantages(torch.zeros(4), 4, "ppo")
    with pytest.raises(ValueError):
        group_advantages(torch.zeros(4), 1)


# ----------------------------------------------------------------- zero gradient when every sample fails


def _grad_on_failing_prompt(kind, entropy_coef):
    task = NoSolutionTask()
    policy = _policy(task, init_scale=1.0, seed=3)
    rollout = policy.sample(task, [0] * 16, torch.Generator().manual_seed(0))
    rewards = episode_rewards(task, rollout)
    assert torch.equal(rewards, torch.zeros(16))
    loss = rl_loss(rollout, rewards, 16, kind, entropy_coef)
    loss.backward()
    return policy.logits.grad


@pytest.mark.parametrize("kind", ["rloo", "grpo"])
def test_zero_gradient_when_every_sample_fails(kind):
    grad = _grad_on_failing_prompt(kind, 0.0)
    assert grad is not None and torch.equal(grad, torch.zeros_like(grad))


@pytest.mark.parametrize("kind", ["rloo", "grpo"])
def test_entropy_bonus_moves_the_policy_even_when_every_sample_fails(kind):
    grad = _grad_on_failing_prompt(kind, 0.01)
    assert grad is not None and grad.abs().sum() > 0


@pytest.mark.parametrize("name,moves", [("rloo", False), ("grpo", False), ("rl_entropy", True)])
def test_training_on_a_failing_prompt(name, moves):
    task = NoSolutionTask()
    policy = _policy(task, init_scale=1.0, seed=3)
    before = policy.logits.detach().clone()
    _train(name, policy, task, 5, 0, [0], group=8, lr=0.05)
    assert torch.equal(policy.logits.detach(), before) is not moves


# ----------------------------------------------------------------- learning


@pytest.mark.parametrize("name", ["rloo", "grpo"])
def test_training_raises_train_j_and_leaves_heldout_untouched(name):
    task = TinyPromptTask()
    train_ids = task.prompt_ids(split="train")
    heldout_ids = task.prompt_ids(split="heldout")
    policy = _policy(task)
    before = policy.logits.detach().clone()
    j0 = _exact_j(task, policy, train_ids)
    _train(name, policy, task, 150, 0, train_ids, group=16, lr=0.05)
    j1 = _exact_j(task, policy, train_ids)
    assert j1 > j0 + 0.3, (j0, j1)
    for pid in heldout_ids:
        assert torch.equal(policy.logits.detach()[pid], before[pid])
        assert np.array_equal(policy.state_logits(task, pid), before[pid].double().numpy())


def test_only_tags_restricts_training_prompts():
    task = TinyPromptTask()
    train_ids = task.prompt_ids(split="train")
    assert select_prompts(task, train_ids, {"difficulty": "hard"}) == [1]
    assert select_prompts(task, train_ids, {"difficulty": ["hard", "easy"]}) == [0, 1]
    assert select_prompts(task, train_ids, None) == train_ids
    # never adds prompts outside prompt_ids, even if their tags match
    assert select_prompts(task, [0], {"difficulty": "hard"}) == []
    policy = _policy(task)
    before = policy.logits.detach().clone()
    _train("rloo", policy, task, 20, 0, train_ids, group=8, lr=0.05, only_tags={"difficulty": "hard"})
    after = policy.logits.detach()
    assert not torch.equal(after[1], before[1])
    for pid in (0, 2, 3):
        assert torch.equal(after[pid], before[pid])


def test_prompts_per_update_minibatch_only_uses_given_prompts():
    task = TinyPromptTask()
    policy = _policy(task)
    before = policy.logits.detach().clone()
    _train("grpo", policy, task, 20, 0, [0, 1, 2], group=8, lr=0.05, prompts_per_update=1)
    after = policy.logits.detach()
    assert torch.equal(after[3], before[3])
    assert sum(not torch.equal(after[p], before[p]) for p in (0, 1, 2)) >= 2


# ----------------------------------------------------------------- checkpoints and reproducibility


@pytest.mark.parametrize("name", ARMS)
def test_checkpoints_called_at_exactly_the_requested_steps(name):
    task = TinyPromptTask()
    seen = []
    _train(name, _policy(task), task, 6, 0, [0, 1], checkpoints=[0, 2, 6, 9], hook=lambda s, p: seen.append(s), group=4)
    assert seen == [0, 2, 6]
    seen.clear()
    _train(name, _policy(task), task, 0, 0, [0, 1], checkpoints=[0, 1, 2], hook=lambda s, p: seen.append(s), group=4)
    assert seen == [0]


def test_checkpoint_sees_pre_update_policy_at_step_zero():
    task = TinyPromptTask()
    policy = _policy(task)
    snaps = {}
    _train("rloo", policy, task, 3, 0, [0, 1], checkpoints=[0, 3], hook=lambda s, p: snaps.__setitem__(s, p.logits.detach().clone()), group=4, lr=0.1)
    assert torch.equal(snaps[0], torch.zeros_like(snaps[0]))
    assert torch.equal(snaps[3], policy.logits.detach())


@pytest.mark.parametrize("name", ARMS)
def test_same_seed_same_parameters_different_seed_different(name):
    task = TinyPromptTask()

    def run(seed):
        policy = _policy(task, init_scale=0.5, seed=1)
        _train(name, policy, task, 10, seed, [0, 1], group=8, lr=0.05, prompts_per_update=1)
        return policy.logits.detach().clone()

    assert torch.equal(run(7), run(7))
    assert not torch.equal(run(7), run(8))
