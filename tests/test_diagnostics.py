import numpy as np
import pytest

from capsplit.diagnostics import passk_curve, passk_rule, score_rule

K = [1, 2, 5, 10, 20, 50, 100, 1000]


def test_curve_values():
    ks = [1, 2, 5, 10, 20, 100]
    difference = passk_curve([0.99, 0.01], ks) - passk_curve([0.4, 0.4], ks)
    expected = [0.1000, -0.1301, -0.3977, -0.4461, -0.4089, -0.1830]
    np.testing.assert_allclose(difference, expected, atol=5e-5)


def test_rule_sharpening():
    assert passk_rule([0.4, 0.4], [0.99, 0.01], K, k_large=10) == "sharpening"


def test_rule_capability():
    assert passk_rule([0.2], [0.3], K, k_large=10) == "capability"
    assert passk_rule([0.2, 0.3], [0.4, 0.5], K, k_large=10) == "capability"


def test_rule_undetermined():
    assert passk_rule([0.4, 0.5], [0.5, 0.4], K, k_large=10) == "undetermined"


def test_k_large_is_inclusive():
    assert passk_rule([0.4, 0.4], [0.99, 0.01], [1, 10], k_large=10) == "sharpening"


def test_below_only_under_k_large_is_undetermined():
    # Below at k=10 but no k at or past k_large=11, and above at k=1, so neither verdict holds.
    assert passk_rule([0.4, 0.4], [0.99, 0.01], [1, 10], k_large=11) == "undetermined"


def test_single_prompt_never_crosses():
    rng = np.random.default_rng(0)
    for j0, j1 in rng.uniform(0.0, 1.0, size=(100, 2)):
        difference = passk_curve([j1], K) - passk_curve([j0], K)
        assert (difference >= 0).all() or (difference <= 0).all()


def test_score_primary_secondary():
    rules = ["capability", "capability", "sharpening", "sharpening", "sharpening", "undetermined"]
    cells = [
        "capability",  # agrees on both
        "reweighting",  # agrees on neither
        "pure_sharpening",  # agrees on both
        "compounded_sharpening",  # agrees on both
        "reweighting",  # primary only
        "capability",  # undetermined never agrees
    ]

    score = score_rule(rules, cells)

    assert score["primary"] == pytest.approx(4 / 6)
    assert score["secondary"] == pytest.approx(3 / 6)
    assert score["undetermined_rate"] == pytest.approx(1 / 6)
    assert score["per_cell"]["capability"] == {
        "runs": 2,
        "primary": pytest.approx(1 / 2),
        "secondary": pytest.approx(1 / 2),
        "undetermined_rate": pytest.approx(1 / 2),
    }
    assert score["per_cell"]["reweighting"]["primary"] == pytest.approx(1 / 2)
    assert score["per_cell"]["reweighting"]["secondary"] == 0.0
    assert set(score["per_cell"]) == {"capability", "reweighting", "pure_sharpening", "compounded_sharpening"}


def test_sharpening_on_a_capability_run_disagrees():
    score = score_rule(["sharpening"], ["capability"])
    assert score["primary"] == 0.0
    assert score["secondary"] == 0.0


def test_score_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        score_rule(["capability"], [])


def test_srr_ndr_counts_and_zero_denominators():
    from capsplit.diagnostics import srr_ndr

    r = srr_ndr(np.array([0.2, 0.0, 0.05]), np.array([0.0, 0.3, 0.5]), eps=0.1)
    assert (r["P"], r["S"], r["E"]) == (0, 1, 2) and r["srr"] == 0.0 and r["ndr"] == 1.0
    empty = srr_ndr(np.array([0.0]), np.array([0.0]), eps=0.1)
    assert empty["srr"] is None and empty["ndr"] is None
    boundary = srr_ndr(np.array([0.1]), np.array([0.1]), eps=0.1)  # strict > eps, as the note states
    assert (boundary["P"], boundary["S"], boundary["E"]) == (0, 0, 0)
