import numpy as np
import pytest
from phase1_rq1 import WORLD, lg

from capsplit.engine import trajectory_probs
from capsplit.estimator import (
    SplitEstimate,
    chernoff_size,
    estimate_split,
    exact_means,
    hoeffding_size,
    within_bound,
)
from capsplit.evaluate import evaluate
from capsplit.worlds import compile_grid, logits_to_array

N = 20


@pytest.fixture(scope="module")
def reference():
    table, state_index = compile_grid(WORLD)
    logits0 = logits_to_array(state_index, lg(2.5, 2.0, 0.0))
    probs1 = trajectory_probs(table, logits_to_array(state_index, lg(2.0, 2.0, 3.0)))
    return table, logits0, probs1


def test_sizes():
    assert chernoff_size(1e-3, 0.1, 0.05) == 1522553
    assert hoeffding_size(1e-4, 0.05) == 253758691
    assert chernoff_size(1e-2, 0.2, 0.05) == 38064


@pytest.mark.parametrize("eta", [1.0001, 0.0, -0.1])
def test_eta_validated(eta):
    with pytest.raises(ValueError):
        chernoff_size(1e-3, eta, 0.05)


def test_eta_of_one_is_allowed():
    assert chernoff_size(1e-3, 1.0, 0.05) > 0


@pytest.mark.parametrize(
    "call",
    [
        lambda: chernoff_size(0.0, 0.1, 0.05),
        lambda: chernoff_size(1e-3, 0.1, 0.0),
        lambda: chernoff_size(1e-3, 0.1, 1.0),
        lambda: hoeffding_size(0.0, 0.05),
        lambda: hoeffding_size(1e-4, 0.0),
    ],
)
def test_size_arguments_validated(call):
    with pytest.raises(ValueError):
        call()


def test_reference_means(reference):
    means = exact_means(*reference, N)
    np.testing.assert_allclose(means, [0.25019, 0.01281, 0.09489, 0.49799], atol=5e-6)


def test_means_exact(reference):
    table, logits0, probs1 = reference
    means = exact_means(table, logits0, probs1, N)
    report = evaluate(table, logits0, probs1, N)
    assert means[2] - means[0] == pytest.approx(report.delta_rel)
    assert means[3] - means[1] == pytest.approx(report.delta_cap)


def test_threshold_multiplier_moves_mass_between_classes(reference):
    table, logits0, probs1 = reference
    means = exact_means(table, logits0, probs1, N, c=1e6)  # nothing is above the line
    assert means[0] == 0.0 and means[2] == 0.0
    assert means[1] == pytest.approx(0.25019 + 0.01281, abs=1e-4)


def test_coverage(reference):
    table, logits0, probs1 = reference
    mu, eta, delta = 0.01, 0.2, 0.05
    m = chernoff_size(mu, eta, delta)
    exact = exact_means(table, logits0, probs1, N)
    rng = np.random.default_rng(0)
    hits = sum(
        within_bound(estimate_split(table, logits0, probs1, N, m, rng), exact, eta, mu) for _ in range(200)
    )
    assert hits / 200 >= 0.95


def test_estimate_deltas_come_from_means(reference):
    table, logits0, probs1 = reference
    estimate = estimate_split(table, logits0, probs1, N, 1000, np.random.default_rng(1))
    assert estimate.delta_rel == estimate.means[2] - estimate.means[0]
    assert estimate.delta_cap == estimate.means[3] - estimate.means[1]


def test_estimate_converges_to_exact(reference):
    table, logits0, probs1 = reference
    estimate = estimate_split(table, logits0, probs1, N, 10**7, np.random.default_rng(2))
    np.testing.assert_allclose(estimate.means, exact_means(table, logits0, probs1, N), atol=1e-3)


def test_same_rng_seed_same_estimate(reference):
    table, logits0, probs1 = reference
    first = estimate_split(table, logits0, probs1, N, 1000, np.random.default_rng(3))
    second = estimate_split(table, logits0, probs1, N, 1000, np.random.default_rng(3))
    np.testing.assert_array_equal(first.means, second.means)


def test_policies_are_sampled_independently(reference):
    # With pi0 == pi1 the deltas are 0 in expectation, but only exactly 0 if one draw is reused.
    table, logits0, _ = reference
    same = trajectory_probs(table, logits0)
    estimate = estimate_split(table, logits0, same, N, 1000, np.random.default_rng(4))
    assert estimate.delta_rel != 0.0


def test_invalid_estimate_args(reference):
    table, logits0, probs1 = reference
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        estimate_split(table, logits0, probs1, N, 0, rng)
    with pytest.raises(ValueError):
        estimate_split(table, logits0, probs1 * 2, N, 100, rng)


def _estimate(means):
    return SplitEstimate(delta_rel=0.0, delta_cap=0.0, means=np.array(means))


def test_within_bound_uses_max_of_mean_and_mu():
    exact = np.array([0.5, 0.001, 0.5, 0.001])
    # eta * max(m, mu) is 0.05 for the large means and 0.001 for the small ones (mu = 0.01).
    assert within_bound(_estimate([0.54, 0.0019, 0.46, 0.0001]), exact, eta=0.1, mu=0.01)
    assert not within_bound(_estimate([0.56, 0.001, 0.5, 0.001]), exact, eta=0.1, mu=0.01)
    assert not within_bound(_estimate([0.5, 0.0025, 0.5, 0.001]), exact, eta=0.1, mu=0.01)


def test_within_bound_checks_every_mean():
    exact = np.array([0.5, 0.5, 0.5, 0.5])
    assert not within_bound(_estimate([0.5, 0.5, 0.5, 0.6]), exact, eta=0.1, mu=0.01)
