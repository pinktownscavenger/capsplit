import math

import pytest

from capsplit import excess_score, label_2x2, pass_at_n, sequence_temper, split
from capsplit.core import success_prob


def test_proposition_1_exact_split():
    q0 = {"a": 0.2, "b": 0.001, "x": 0.799}
    q1 = {"a": 0.4, "b": 0.05, "x": 0.55}
    out = split(q0, q1, {"a", "b"}, n=100)
    assert out["delta_rel"] + out["delta_cap"] == pytest.approx(out["delta_j"])
    assert out["delta_j"] == pytest.approx(success_prob(q1, "ab") - success_prob(q0, "ab"))


def test_proposition_3_example_same_passk_different_split():
    # Two prompts, N = 100. Base probabilities from the project plan.
    base = [{"a": 0.2, "b": 0.001}, {"a": 0.3, "b": 0.002}]
    pi1 = [{"a": 0.4, "b": 0.0}, {"a": 0.5, "b": 0.0}]
    pi1p = [{"a": 0.0, "b": 0.5}, {"a": 0.0, "b": 0.4}]
    correct = {"a", "b"}

    def total(pols):
        rel = cap = 0.0
        for q0, q1 in zip(base, pols):
            s = split(q0, q1, correct, n=100)
            rel += s["delta_rel"]
            cap += s["delta_cap"]
        return rel, cap

    rel1, cap1 = total(pi1)
    rel2, cap2 = total(pi1p)
    assert (rel1, cap1) == pytest.approx((0.4, -0.003))
    assert (rel2, cap2) == pytest.approx((-0.5, 0.897))
    assert rel1 + cap1 == pytest.approx(rel2 + cap2) == pytest.approx(0.397)
    for k in (1, 5, 50, 1000):
        c1 = sum(pass_at_n(success_prob(q, correct), k) for q in pi1)
        c2 = sum(pass_at_n(success_prob(q, correct), k) for q in pi1p)
        assert c1 == pytest.approx(c2)


@pytest.mark.parametrize("beta", [1.0, 1.5, 2, 4, 8])
def test_proposition_2_sequence_level_bound(beta):
    # ratio P_beta/P0 <= (P0 / U)^(beta-1), U = 1/T
    q0 = {i: p for i, p in enumerate([0.5, 0.3, 0.15, 0.04, 0.009, 0.001])}
    t = len(q0)
    qb = sequence_temper(q0, beta)
    for s, p in q0.items():
        assert qb[s] / p <= (p * t) ** (beta - 1) + 1e-12


def test_pure_sharpening_label():
    q0 = {"a": 0.2, "b": 0.8}
    q1 = sequence_temper(q0, 3.0)
    n = 10
    s = split(q0, q1, {"a"}, n)
    e = excess_score(q0, q1, {"a"}, n, betas=[0.25, 0.5, 1, 2, 3, 4, 8])
    assert label_2x2(s["delta_cap"], e) == "pure_sharpening"


def test_capability_label():
    q0 = {"a": 0.5, "rare": 0.0001, "x": 0.4999}
    q1 = {"a": 0.3, "rare": 0.6, "x": 0.1}
    n = 2  # small budget: at large N both pass@N values saturate at 1.0 and E_N = 0
    s = split(q0, q1, {"a", "rare"}, n)
    e = excess_score(q0, q1, {"a", "rare"}, n, betas=[0.01, 0.25, 0.5, 1, 2, 4, 8])
    assert s["delta_cap"] > 0 and e > 0
    assert label_2x2(s["delta_cap"], e) == "capability"
