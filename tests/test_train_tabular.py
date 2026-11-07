import numpy as np
import pytest
from pilot_rq2 import (
    ROUTE_A,
    ROUTE_B,
    competing_base,
    compile_competing,
    compile_isolated,
    isolated_base,
    updates_to_reach,
)

from capsplit.engine import solution_probs, table_success_prob, trajectory_probs
from capsplit.train.tabular import exact_pg_gradient, reinforce, train_exact_pg


@pytest.fixture(scope="module")
def competing():
    return compile_competing()


@pytest.fixture(scope="module")
def isolated():
    return compile_isolated()


def _route_q(table, logits, route):
    return solution_probs(table, trajectory_probs(table, logits))[table.solutions.index(route)]


def test_worlds_have_the_stated_routes(competing, isolated):
    assert set(competing[0].solutions) == {ROUTE_A, ROUTE_B}
    assert len(isolated[0].solutions) == 1


def test_gradient_matches_finite_differences(competing):
    table, _ = competing
    logits = np.random.default_rng(0).normal(size=(table.n_states, table.n_actions))
    gradient, j = exact_pg_gradient(table, logits)
    assert j == pytest.approx(table_success_prob(table, trajectory_probs(table, logits)))

    step = 1e-6
    numeric = np.zeros_like(logits)
    for index in np.ndindex(logits.shape):
        up, down = logits.copy(), logits.copy()
        up[index] += step
        down[index] -= step
        numeric[index] = (
            table_success_prob(table, trajectory_probs(table, up))
            - table_success_prob(table, trajectory_probs(table, down))
        ) / (2 * step)
    assert np.abs(gradient - numeric).max() < 1e-8


def test_gradient_on_a_table_with_padding():
    # The isolated world's table has no padding on its correct route, so check an early-goal world too.
    from capsplit.worlds import GridWorld, compile_grid

    table, _ = compile_grid(GridWorld(n=2, start=(0, 0), goal=(1, 1), H=4))
    logits = np.random.default_rng(1).normal(size=(table.n_states, table.n_actions))
    gradient, _ = exact_pg_gradient(table, logits)
    assert gradient.shape == logits.shape

    step = 1e-6
    index = (0, 3)
    up, down = logits.copy(), logits.copy()
    up[index] += step
    down[index] -= step
    numeric = (
        table_success_prob(table, trajectory_probs(table, up)) - table_success_prob(table, trajectory_probs(table, down))
    ) / (2 * step)
    assert gradient[index] == pytest.approx(numeric, abs=1e-8)


def test_gradient_with_no_correct_trajectories():
    from capsplit.worlds import GridWorld, compile_grid

    table, _ = compile_grid(GridWorld(n=3, start=(0, 0), goal=(2, 2), H=2))
    gradient, j = exact_pg_gradient(table, np.zeros((table.n_states, table.n_actions)))
    assert j == 0.0
    assert (gradient == 0.0).all()


@pytest.mark.parametrize(
    "target,updates",
    [(1e-4, 2358), (1e-3, 362), (1e-2, 63), pytest.param(1e-5, 17837, marks=pytest.mark.slow)],
)
def test_isolated_route_updates(isolated, target, updates):
    table, state_index = isolated
    logits0 = isolated_base(table, state_index, target)
    assert table_success_prob(table, trajectory_probs(table, logits0)) == pytest.approx(target, rel=1e-6)
    assert updates_to_reach(table, logits0, 0.1, lr=1.0, max_steps=30000) == pytest.approx(updates, rel=0.02)


def test_competing_symmetric(competing):
    table, state_index = competing
    final = train_exact_pg(table, competing_base(state_index, (0, 0)), lr=2.0, steps=6000)
    assert _route_q(table, final, ROUTE_A) == pytest.approx(0.4998, abs=0.01)
    assert _route_q(table, final, ROUTE_B) == pytest.approx(0.4998, abs=0.01)


def test_competing_asymmetric(competing):
    table, state_index = competing
    logits0 = competing_base(state_index, (3, 2))
    assert _route_q(table, logits0, ROUTE_B) == pytest.approx(0.0902, abs=5e-5)
    final = train_exact_pg(table, logits0, lr=2.0, steps=6000)
    assert _route_q(table, final, ROUTE_B) < 1e-3
    assert _route_q(table, final, ROUTE_A) > 0.99


def test_train_exact_pg_logs_and_leaves_the_input_alone(competing):
    table, state_index = competing
    logits0 = competing_base(state_index, (1, 0))
    before = logits0.copy()
    seen = []
    final = train_exact_pg(table, logits0, lr=1.0, steps=3, on_step=lambda step, logits: seen.append((step, logits.copy())))
    np.testing.assert_array_equal(logits0, before)
    assert [step for step, _ in seen] == [1, 2, 3]
    np.testing.assert_array_equal(seen[-1][1], final)
    assert not np.array_equal(seen[0][1], seen[1][1])


@pytest.mark.parametrize("seed", range(5))
def test_reinforce_learns(competing, seed):
    table, state_index = competing
    logits0 = competing_base(state_index, (1, 0))
    assert table_success_prob(table, trajectory_probs(table, logits0)) == pytest.approx(0.054, abs=5e-4)
    final = reinforce(table, logits0, seed)
    assert table_success_prob(table, trajectory_probs(table, final)) > 0.9


def test_reinforce_is_deterministic_per_seed(competing):
    table, state_index = competing
    logits0 = competing_base(state_index, (1, 0))
    np.testing.assert_array_equal(reinforce(table, logits0, 1, steps=20), reinforce(table, logits0, 1, steps=20))
    assert not np.array_equal(reinforce(table, logits0, 1, steps=20), reinforce(table, logits0, 2, steps=20))
