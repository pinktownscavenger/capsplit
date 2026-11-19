import numpy as np
import pytest
from pilot_rq2 import isolated_base
from rq2_time_to_cross import (
    CROSS_FACTOR,
    ROBUST_SIGMA,
    competing_base,
    competing_world,
    isolated_world,
    noisy_isolated_base,
)

from capsplit.train.tabular import exact_pg_step
from capsplit.worlds import compile_grid


def test_w3_has_exactly_two_routes_and_w2_eight():
    assert len(compile_grid(competing_world("W3")[0])[0].solutions) == 2
    assert len(compile_grid(competing_world("W2")[0])[0].solutions) == 8


@pytest.mark.parametrize("horizon", [9, 10])
def test_longer_horizon_keeps_one_route(horizon):
    assert len(compile_grid(isolated_world(horizon))[0].solutions) == 1


def test_competing_routes_are_correct_solutions():
    for name in ("W1", "W2", "W3"):
        world, common, rare = competing_world(name)
        solutions = compile_grid(world)[0].solutions
        assert tuple(common) in solutions and tuple(rare) in solutions and tuple(common) != tuple(rare)


def test_noise_free_base_equals_pilot():
    world = isolated_world(8)
    table, si = compile_grid(world)
    assert np.allclose(noisy_isolated_base(table, si, world, 1e-4, 0.0, 0), isolated_base(table, si, 1e-4), atol=1e-12)


@pytest.mark.parametrize("seed", [0, 1])
def test_noisy_base_hits_target(seed):
    world = isolated_world(8)
    table, si = compile_grid(world)
    logits = noisy_isolated_base(table, si, world, 1e-5, ROBUST_SIGMA, seed)
    assert exact_pg_step(table, logits)[1][0] == pytest.approx(1e-5, rel=0.01)
    assert not np.allclose(logits, noisy_isolated_base(table, si, world, 1e-5, 0.0, seed))


def test_unreachable_target_raises():
    world = isolated_world(8)
    table, si = compile_grid(world)
    with pytest.raises(ValueError, match="target"):
        noisy_isolated_base(table, si, world, 0.9999999, 0.0, 0)


@pytest.mark.parametrize("name", ["W1", "W2", "W3"])
@pytest.mark.parametrize("ratio", [1.0, 0.01])
def test_competing_ratio_and_common_range(name, ratio):
    world, _, _ = competing_world(name)
    table, si = compile_grid(world)
    _, q_common, q_rare = competing_base(table, si, name, ratio, ROBUST_SIGMA, 3)
    assert q_rare / q_common == pytest.approx(ratio, rel=0.01)
    assert CROSS_FACTOR * q_rare <= 0.5  # the primary rule stays achievable
    if ratio == 1.0:
        _, q_common0, _ = competing_base(table, si, name, 1.0, 0.0, 0)
        assert 0.01 <= q_common0 <= 0.04
