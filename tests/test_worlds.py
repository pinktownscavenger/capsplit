import pytest

from capsplit import split
from capsplit.core import success_prob
from capsplit.worlds import (
    N_ACTIONS,
    GridWorld,
    correct_paths,
    enumerate_actions,
    enumerate_paths,
    policy_from_routes,
)

WORLD = GridWorld(n=3, start=(0, 0), goal=(2, 2), H=5, walls=frozenset({(1, 1)}))
ROUTE_A = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
ROUTE_B = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]


def logits(bonus_a=1.0, bonus_b=0.5):
    return policy_from_routes(WORLD, [ROUTE_A, ROUTE_B], [bonus_a, bonus_b])


def test_probabilities_sum_to_one():
    assert sum(enumerate_paths(WORLD, logits()).values()) == pytest.approx(1.0)
    assert sum(enumerate_actions(WORLD, logits(), beta=3).values()) == pytest.approx(1.0)


def test_routes_are_correct_solutions():
    q = enumerate_paths(WORLD, logits())
    correct = correct_paths(WORLD, q)
    assert tuple(ROUTE_A) in correct and tuple(ROUTE_B) in correct


def test_bonus_dials_route_probability():
    low = enumerate_paths(WORLD, logits(bonus_a=0.0))[tuple(ROUTE_A)]
    high = enumerate_paths(WORLD, logits(bonus_a=2.0))[tuple(ROUTE_A)]
    assert high > low


def test_split_exact_on_grid():
    q0 = enumerate_paths(WORLD, logits(1.0, -1.0))
    q1 = enumerate_paths(WORLD, logits(1.0, 3.0))
    correct = correct_paths(WORLD, q0) | correct_paths(WORLD, q1)
    out = split(q0, q1, correct, n=20)
    assert out["delta_rel"] + out["delta_cap"] == pytest.approx(
        success_prob(q1, correct) - success_prob(q0, correct)
    )


@pytest.mark.parametrize("beta", [1.0, 1.5, 2.0, 4.0])
def test_proposition_2_per_step_bound(beta):
    # ratio P_beta/P0 <= (P0 / U)^(beta-1), U = |A|^-H (episodes may end early, which only loosens it)
    lg = logits()
    p0 = enumerate_actions(WORLD, lg)
    pb = enumerate_actions(WORLD, lg, beta)
    u = N_ACTIONS ** (-WORLD.H)
    for s, p in p0.items():
        assert pb[s] / p <= (p / u) ** (beta - 1) * (1 + 1e-9)
