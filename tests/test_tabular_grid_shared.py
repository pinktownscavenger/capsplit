import numpy as np
import pytest
import torch

from capsplit.engine import table_success_prob, trajectory_probs
from capsplit.harness.policies.tabular_grid import TabularGridPolicy
from capsplit.harness.policies.tabular_grid_shared import TabularGridSharedPolicy
from capsplit.harness.registry import BASES, POLICIES
from capsplit.harness.tasks.grid import GridTask

VARIANTS = [
    {"start": [0, 0], "goal": [4, 4], "walls": [[2, 2]], "difficulty": "easy", "split": "train"},
    {"start": [0, 4], "goal": [4, 0], "walls": [[2, 2]], "difficulty": "hard", "split": "heldout"},
]


def _task():
    return GridTask(n=5, H=8, variants=VARIANTS)


def _j(policy, task, pid):
    return table_success_prob(task.table(pid), policy.trajectory_probs(task, pid))


def _step_on_prompt0(policy, task):
    # An SFT-style step on log-probability; a step on the probability itself is (1/4)^8 too small.
    loss = -policy.log_prob(task, [0], [task.demonstrations(0)[0]]).sum()
    loss.backward()
    with torch.no_grad():
        for p in policy.parameters():
            p.sub_(5.0 * p.grad)


def test_training_one_prompt_moves_the_other():
    task = _task()
    shared = TabularGridSharedPolicy(2, task.n_states(0))
    before = _j(shared, task, 1)
    _step_on_prompt0(shared, task)
    assert abs(_j(shared, task, 1) - before) > 1e-6


def test_per_prompt_policy_does_not_interfere():
    task = _task()
    separate = TabularGridPolicy(2, task.n_states(0))
    before = _j(separate, task, 1)
    _step_on_prompt0(separate, task)
    assert _j(separate, task, 1) == pytest.approx(before, abs=1e-12)


def test_refuses_different_walls():
    task = GridTask(n=5, H=8, variants=[VARIANTS[0], {**VARIANTS[1], "walls": [[1, 1]]}])
    with pytest.raises(ValueError, match="same walls"):
        TabularGridSharedPolicy(2, task.n_states(0)).state_logits(task, 0)


def test_trajectory_probs_match_engine_and_sum_to_one():
    task = _task()
    policy = TabularGridSharedPolicy(2, task.n_states(0), init_scale=0.5, seed=1)
    for pid in (0, 1):
        probs = policy.trajectory_probs(task, pid)
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.allclose(probs, trajectory_probs(task.table(pid), policy.state_logits(task, pid)))
    assert np.array_equal(policy.state_logits(task, 0), policy.state_logits(task, 1))


def test_state_dict_round_trip_and_parameters():
    task = _task()
    a = TabularGridSharedPolicy(2, task.n_states(0), init_scale=0.5, seed=1)
    b = TabularGridSharedPolicy(2, task.n_states(0))
    b.load_state_dict(a.state_dict())
    assert np.array_equal(a.state_logits(task, 1), b.state_logits(task, 1))
    assert [tuple(p.shape) for p in a.parameters()] == [(task.n_states(0), 4)]


def test_shared_base_is_registered_and_reports_q0():
    policy, info = BASES.create("grid_shared_routes").build(_task(), seed=0)
    assert isinstance(policy, TabularGridSharedPolicy)
    assert len(info["favoured_q0"]) == 2 and all(0 < q < 1 for q in info["favoured_q0"])
    assert POLICIES.get("tabular_grid_shared") is TabularGridSharedPolicy
