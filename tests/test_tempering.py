"""Tempering families and the re-tempering frontier, on the reference 4x4 trap base."""
import numpy as np
import phase1_rq1 as rq1
import pytest

from capsplit import sequence_temper
from capsplit.engine import table_success_prob, trajectory_probs
from capsplit.tempering import DEFAULT_BETAS, frontier, sequence_probs
from capsplit.worlds import compile_grid, enumerate_actions, logits_to_array


@pytest.fixture(scope="module")
def setup():
    table, states = compile_grid(rq1.WORLD)
    return table, logits_to_array(states, rq1.trap(3.2, 3.2, 0.0)), logits_to_array(states, rq1.lg(2.5, 2.0, 0.0))


@pytest.mark.parametrize(
    "beta,sequence_j,per_step_j",
    [(0.25, 0.00928, 0.00989), (0.5, 0.03250, 0.03970), (2, 0.10089, 0.03124), (4, 0.01133, 0.00055)],
)
def test_family_values(setup, beta, sequence_j, per_step_j):
    table, trap, _ = setup
    assert table_success_prob(table, sequence_probs(table, trap, beta)) == pytest.approx(sequence_j, abs=1e-4)
    assert table_success_prob(table, trajectory_probs(table, trap, beta)) == pytest.approx(per_step_j, abs=1e-4)


def test_sequence_family_tempers_trajectories_not_paths(setup):
    # Blueprint D1: agrees with brute-force tempering over action sequences.
    table, trap, _ = setup
    brute = sequence_temper(enumerate_actions(rq1.WORLD, rq1.trap(3.2, 3.2, 0.0)), 2.0)
    assert np.sort(sequence_probs(table, trap, 2.0)) == pytest.approx(np.sort(list(brute.values())))


def test_sequence_probs_sum_to_one_at_extreme_beta(setup):
    table, trap, _ = setup
    assert sequence_probs(table, trap * 1000, 8.0).sum() == pytest.approx(1.0)


def test_default_betas():
    assert len(DEFAULT_BETAS) == 26
    assert DEFAULT_BETAS.min() == pytest.approx(0.25)
    assert DEFAULT_BETAS.max() == pytest.approx(8.0)
    assert 1.0 in DEFAULT_BETAS


def test_frontier_values(setup):
    table, trap, _ = setup
    best = frontier(table, trap)
    assert best.best_j == pytest.approx(0.13255, abs=1e-4)
    assert best.best_beta == pytest.approx(1.361, abs=0.01)
    assert best.best_family == "sequence"
    assert best.boundary_hit is False


def test_refinement_beats_grid(setup):
    # Blueprint D3: the grid alone is a lower bound on the sup.
    table, trap, _ = setup
    assert frontier(table, trap).best_j > frontier(table, trap, refine=False).best_j


def test_boundary_hit_when_best_beta_is_at_range_end(setup):
    table, _, easy = setup
    best = frontier(table, easy)
    assert best.best_beta == pytest.approx(8.0, abs=0.01)
    assert best.boundary_hit is True


def test_frontier_never_below_base(setup):
    table, trap, _ = setup
    assert frontier(table, trap, betas=np.array([4.0, 8.0]), refine=False).best_j >= table_success_prob(
        table, trajectory_probs(table, trap)
    )
