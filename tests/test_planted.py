import numpy as np
import pytest

from capsplit.engine import solution_probs, trajectory_probs
from capsplit.evaluate import evaluate
from capsplit.planted import EXPECTED, SIZES, PlantedSet, sample_cases, size_world
from capsplit.worlds import compile_grid

SMALL = SIZES[0]
N = SMALL.budget


@pytest.fixture(scope="module")
def compiled():
    return compile_grid(size_world(SMALL))


def _q(table, logits, route):
    return solution_probs(table, trajectory_probs(table, logits))[table.solutions.index(route)]


def test_frozen_sizes():
    assert [(s.grid, s.horizon, s.budget, s.trials, s.quick_trials) for s in SIZES] == [
        (4, 6, 20, 50, 5),
        (5, 8, 50, 20, 2),
        (6, 10, 100, 8, 1),
    ]
    world = size_world(SIZES[1])
    assert (world.n, world.start, world.goal, world.H, world.walls) == (5, (0, 0), (4, 4), 8, frozenset())


def test_returns_every_expected_case(compiled):
    planted = sample_cases(N, 0, *compiled)
    assert isinstance(planted, PlantedSet)
    assert planted.budget == N
    assert set(planted.cases) == set(EXPECTED)


@pytest.mark.parametrize("seed", range(5))
def test_cases_are_distributions(compiled, seed):
    for logits0, probs1 in sample_cases(N, seed, *compiled).cases.values():
        assert logits0.shape == (compiled[0].n_states, compiled[0].n_actions)
        assert (probs1 >= 0).all()
        assert probs1.sum() == pytest.approx(1.0, abs=1e-9)


def test_seed_determinism(compiled):
    first = sample_cases(N, 3, *compiled)
    second = sample_cases(N, 3, *compiled)
    assert first.routes == second.routes
    for name in EXPECTED:
        np.testing.assert_array_equal(first.cases[name][1], second.cases[name][1])


def test_a_d_are_state_disjoint(compiled):
    for seed in range(20):
        routes = sample_cases(N, seed, *compiled).routes
        a, d = routes["a"], routes["d"]
        assert set(a) & set(d) == {a[0], a[-1]}
        assert a[1] == (0, 1)  # first move right
        assert d[1] == (1, 0)  # first move down


@pytest.mark.parametrize("seed", range(5))
def test_trap_preconditions(compiled, seed):
    table = compiled[0]
    planted = sample_cases(N, seed, *compiled)
    trap = planted.cases["b_param"][0]
    routes = planted.routes
    q_a, q_d = _q(table, trap, routes["a"]), _q(table, trap, routes["d"])
    assert min(q_a, q_d) >= 1.2 / N
    assert min(q_a, q_d) == pytest.approx(1.2 / N, abs=1e-6)  # the smallest such bonus
    assert _q(table, trap, routes["rare"]) < 0.5 / N
    assert routes["rare"][1] == (0, 1)
    assert routes["rare"] != routes["a"]
    np.testing.assert_array_equal(trap[compiled[1][(0, 0)]], [3.0, 0.95, 0.0, 0.95])


def test_cases_share_their_base_array(compiled):
    # The sweep computes one frontier per base, keyed on these objects.
    cases = sample_cases(N, 0, *compiled).cases
    assert cases["a_per"][0] is cases["a_seq"][0]
    assert cases["ap_per"][0] is cases["ap_seq"][0]
    assert all(cases[name][0] is cases["b_param"][0] for name in ("b_direct", "c_param", "c_direct"))


@pytest.mark.parametrize("seed", range(5))
def test_b_direct_exact_zero_cap(compiled, seed):
    table = compiled[0]
    logits0, probs1 = sample_cases(N, seed, *compiled).cases["b_direct"]
    assert abs(evaluate(table, logits0, probs1, N).delta_cap) <= 1e-12


def test_direct_cases_move_half_the_incorrect_mass(compiled):
    table = compiled[0]
    planted = sample_cases(N, 0, *compiled)
    for name, route in (("b_direct", "d"), ("c_direct", "rare")):
        logits0, probs1 = planted.cases[name]
        probs0 = trajectory_probs(table, logits0)
        incorrect = table.solution_ids < 0
        np.testing.assert_allclose(probs1[incorrect], probs0[incorrect] / 2)
        target = table.solutions.index(planted.routes[route])
        gain = solution_probs(table, probs1)[target] - solution_probs(table, probs0)[target]
        assert gain == pytest.approx(probs0[incorrect].sum() / 2)


@pytest.mark.parametrize("seed", range(5))
def test_4x4_all_cases_land(compiled, seed):
    table = compiled[0]
    for name, (logits0, probs1) in sample_cases(N, seed, *compiled).cases.items():
        assert evaluate(table, logits0, probs1, N).label == EXPECTED[name], name
