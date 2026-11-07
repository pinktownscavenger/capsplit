"""Exact engine: compiled trajectory tables, checked against the pure-Python enumerator."""
import numpy as np
import phase1_rq1 as rq1
import pytest

from capsplit.engine import (
    log_policy,
    solution_probs,
    table_success_prob,
    trajectory_probs,
)
from capsplit.worlds import (
    GridWorld,
    compile_grid,
    enumerate_paths,
    logits_to_array,
    loop_erased,
    raw_path,
)


def open_world(n):
    return GridWorld(n, (0, 0), (n - 1, n - 1), 2 * (n - 1))


@pytest.fixture(scope="module")
def compiled():
    return compile_grid(rq1.WORLD)


def test_table_shape(compiled):
    table, _ = compiled
    assert table.index.shape == (4096, 6)
    assert len(table.solutions) == 8
    assert table.n_states == 15
    assert table.n_actions == 4
    assert table.horizon == 6


def test_index_and_ids_are_int32(compiled):
    table, _ = compiled
    assert table.index.dtype == np.int32
    assert table.solution_ids.dtype == np.int32


def test_matches_oracle(compiled):
    table, states = compiled
    trap = rq1.trap(3.2, 3.2, 0.0)
    oracle = enumerate_paths(rq1.WORLD, trap, 2.0)
    q = solution_probs(table, trajectory_probs(table, logits_to_array(states, trap), 2.0))
    assert max(abs(oracle[s] - q[i]) for i, s in enumerate(table.solutions)) < 1e-12


@pytest.mark.parametrize("n,trajectories,solutions", [(4, 4096, 20), (5, 65536, 70), (6, 1048576, 252)])
def test_sizes(n, trajectories, solutions):
    table, _ = compile_grid(open_world(n))
    assert table.index.shape[0] == trajectories
    assert len(table.solutions) == solutions


def test_loop_erased_path():
    a, b, c = (0, 0), (0, 1), (0, 2)
    assert loop_erased((a, a, b, a, b, c)) == (a, b, c)
    assert raw_path((a, a, b)) == (a, a, b)


def test_loop_erased_merges_detours():
    world = GridWorld(4, (0, 0), (3, 3), 8, frozenset({(1, 1)}))
    assert len(compile_grid(world, phi=raw_path)[0].solutions) == 288
    assert len(compile_grid(world)[0].solutions) == 18


def test_no_solutions():
    table, _ = compile_grid(GridWorld(4, (0, 0), (3, 3), 3))
    probs = trajectory_probs(table, np.zeros((table.n_states, table.n_actions)))
    assert table.solutions == ()
    assert solution_probs(table, probs).shape == (0,)
    assert table_success_prob(table, probs) == 0.0


def test_early_termination_is_padded():
    world = GridWorld(3, (0, 0), (0, 1), 3)
    table, _ = compile_grid(world)
    probs = trajectory_probs(table, np.zeros((table.n_states, table.n_actions)))
    assert probs.sum() == pytest.approx(1.0)
    assert table_success_prob(table, probs) == pytest.approx(
        sum(p for s, p in enumerate_paths(world, {c: [0.0] * 4 for c in world.cells()}).items() if s[-1] == world.goal)
    )


def test_extreme_logits_stable(compiled):
    table, states = compiled
    logits = logits_to_array(states, rq1.trap(3.2, 3.2, 0.0)) * 1000
    probs = trajectory_probs(table, logits)
    assert np.isfinite(probs).all()
    assert probs.sum() == pytest.approx(1.0, abs=1e-9)


def test_float32_logits_are_exact(compiled):
    table, states = compiled
    logits = logits_to_array(states, rq1.trap(3.2, 3.2, 0.0)).astype(np.float32)
    assert trajectory_probs(table, logits).sum() == pytest.approx(1.0, abs=1e-12)


def test_wrong_logits_shape_raises(compiled):
    table, _ = compiled
    with pytest.raises(ValueError):
        trajectory_probs(table, np.zeros((table.n_states + 1, table.n_actions)))


def test_log_policy_normalizes():
    logits = np.array([[0.0, 1.0, 2.0], [5.0, 5.0, 5.0]])
    assert np.exp(log_policy(logits, 2.0)).sum(axis=1) == pytest.approx([1.0, 1.0])
