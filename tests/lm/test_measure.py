"""Tests for the LM experiment's sampled measurement (capsplit.harness.lm.measure)."""
from __future__ import annotations

import math
from decimal import Decimal, getcontext

import numpy as np
import pytest

from capsplit.harness.budget import clopper_pearson_zero_upper
from capsplit.harness.interfaces import GroupMetrics
from capsplit.harness.lm import measure as M

N = 16
ALPHA = 0.05
M_A = 64


# --------------------------------------------------------------------------- helpers


def _decimal_cp(k: int, m: int, alpha: float) -> tuple[Decimal, Decimal]:
    """Clopper-Pearson bounds by brute-force binomial summation and bisection in 60-digit Decimal."""
    getcontext().prec = 60
    target = Decimal(alpha) / 2

    def cdf(kk: int, p: Decimal) -> Decimal:  # P(X <= kk)
        return sum((Decimal(math.comb(m, j)) * p**j * (1 - p) ** (m - j) for j in range(kk + 1)), Decimal(0))

    def solve(f, increasing: bool) -> Decimal:
        lo, hi = Decimal(0), Decimal(1)
        for _ in range(120):
            mid = (lo + hi) / 2
            if (f(mid) < target) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = Decimal(0) if k == 0 else solve(lambda p: 1 - cdf(k - 1, p), increasing=True)
    upper = Decimal(1) if k == m else solve(lambda p: cdf(k, p), increasing=False)
    return lower, upper


def _draw(rng: np.random.Generator, p: np.ndarray, m: int) -> np.ndarray:
    return rng.binomial(m, p)


def _planted(kind: str, seed: int = 0, p_count: int = 200):
    """Half the problems clearly reachable (p0 in [0.3, 0.6]), half with p0 = 0 (unreachable).
    'reliability': p1 = p0 + 0.2 on reachable problems only; 'capability': p1 = 0.3 on unreachable only."""
    rng = np.random.default_rng(seed)
    half = p_count // 2
    p0 = np.concatenate([rng.uniform(0.3, 0.6, half), np.zeros(p_count - half)])
    p1 = p0.copy()
    if kind == "reliability":
        p1[:half] += 0.2
    else:
        p1[half:] = 0.3
    truth = {"reachable": float((p1 - p0)[:half].sum() / p_count),
             "unreachable": float((p1 - p0)[half:].sum() / p_count), "ambiguous": 0.0}
    truth["total"] = float(np.mean(p1 - p0))
    a = M.Counts(_draw(rng, p0, M_A), np.full(p_count, M_A))
    b = M.Counts(_draw(rng, p0, M_A), np.full(p_count, M_A))
    t = M.Counts(_draw(rng, p1, M_A), np.full(p_count, M_A))
    null = M.Counts(_draw(rng, p0, M_A), np.full(p_count, M_A))
    # temperature sweep: T=0.5 sharpens (reproduces the reliability gain), T=1.0 is the base itself
    sharp = np.where(p0 > 0, np.minimum(p0 + 0.2, 1.0), 0.0)
    by_t = {0.5: M.Counts(_draw(rng, sharp, 32), np.full(p_count, 32)),
            1.0: M.Counts(_draw(rng, p0, 32), np.full(p_count, 32))}
    return truth, a, b, t, null, by_t


# --------------------------------------------------------------------------- counts


def test_counts_from_samples():
    rows = [{"problem": 0, "correct": True}, {"problem": 0, "correct": False}, {"problem": 2, "correct": True,
                                                                               "text": "x"}]
    c = M.counts_from_samples(rows, 3)
    assert c.successes.tolist() == [1, 0, 1]
    assert c.trials.tolist() == [2, 0, 1]
    with pytest.raises(ValueError, match="without samples"):
        c.rates()
    with pytest.raises(ValueError):
        M.counts_from_samples([{"problem": 3, "correct": True}], 3)
    with pytest.raises(TypeError):
        M.counts_from_samples([{"problem": 0, "correct": 1}], 3)


# --------------------------------------------------------------------------- Clopper-Pearson


def test_cp_zero_and_full_closed_forms():
    lo, hi = M.clopper_pearson_table(64, 0.05)
    assert hi[0] == pytest.approx(1 - 0.025 ** (1 / 64), abs=1e-15)
    assert hi[0] == clopper_pearson_zero_upper(64, 0.05)
    assert lo[64] == pytest.approx(0.025 ** (1 / 64), abs=1e-15)
    assert lo[0] == 0.0 and hi[64] == 1.0


@pytest.mark.parametrize("k,m,alpha", [(1, 64, 0.05), (5, 20, 0.05), (17, 64, 0.05), (32, 64, 0.1), (63, 64, 0.05),
                                       (3, 200, 0.01), (199, 200, 0.05)])
def test_cp_matches_high_precision_reference(k, m, alpha):
    lo, hi = M.clopper_pearson(k, m, alpha)
    ref_lo, ref_hi = _decimal_cp(k, m, alpha)
    assert abs(float(lo) - float(ref_lo)) < 1e-12
    assert abs(float(hi) - float(ref_hi)) < 1e-12


def test_cp_textbook_value():
    lo, hi = M.clopper_pearson(5, 20, 0.05)
    assert float(lo) == pytest.approx(0.08657, abs=1e-5)
    assert float(hi) == pytest.approx(0.49105, abs=1e-5)


def test_cp_monotone_in_k():
    lo, hi = M.clopper_pearson_table(64, 0.05)
    assert np.all(np.diff(lo) > 0) and np.all(np.diff(hi) > 0) and np.all(lo < hi)


# --------------------------------------------------------------------------- classification and split


def test_classify_constructed_counts():
    # at m=64, N=16 (1/N = 0.0625): 0/64 -> upper 0.056 < 1/N; 1..4/64 straddle 1/N; 10/64 -> lower 0.077
    k = np.array([0, 1, 4, 10, 64])
    classes = M.classify(k, M_A, N, ALPHA)
    assert classes.tolist() == ["unreachable", "ambiguous", "ambiguous", "reachable", "reachable"]
    # boundary consistency with the bounds themselves
    lo, hi = M.clopper_pearson(np.arange(65), 64, ALPHA)
    codes = M.classify(np.arange(65), 64, N, ALPHA)
    assert np.all((codes == "reachable") == (lo >= 1 / N))
    assert np.all((codes == "unreachable") == (hi < 1 / N))
    # mixed trial counts
    assert M.classify([0, 0], [64, 16], N, ALPHA).tolist() == ["unreachable", "ambiguous"]


def test_split_sum_identity_exact():
    rng = np.random.default_rng(1)
    for _ in range(50):
        p = rng.integers(1, 500)
        classes = rng.choice(M.CLASSES, p)
        p0, p1 = rng.random(p), rng.random(p)
        s = M.split(classes, p0, p1)
        assert abs(s["reachable"] + s["unreachable"] + s["ambiguous"] - s["total"]) <= 1e-12
        assert s["total"] == pytest.approx(np.mean(p1 - p0), abs=1e-15)
        assert sum(s["counts"].values()) == p
        for name in M.CLASSES:
            if s["counts"][name]:
                assert s["conditional_means"][name] * s["counts"][name] / p == pytest.approx(s[name], abs=1e-12)
            else:
                assert s["conditional_means"][name] is None and s[name] == 0.0


def test_split_rejects_unknown_class():
    with pytest.raises(ValueError):
        M.split(["reachable", "maybe"], [0, 0], [1, 1])


# --------------------------------------------------------------------------- temperature baseline


def test_pass_at_n_unbiased_matches_combinatorics():
    for m, c, n in [(32, 0, 16), (32, 5, 16), (32, 20, 16), (64, 1, 16), (16, 3, 16)]:
        expected = 1 - math.comb(m - c, n) / math.comb(m, n)
        assert M.pass_at_n_unbiased(c, m, n) == pytest.approx(expected, abs=1e-12)
    with pytest.raises(ValueError):
        M.pass_at_n_unbiased(1, 8, 16)


def _two_temperatures(p_count: int = 100):
    # T=0.7: p=0.4 everywhere (best pass@4); T=1.3: p=1 on 60% of problems, 0 elsewhere (best mean)
    rng = np.random.default_rng(3)
    spread = np.full(p_count, 0.4)
    peaked = np.zeros(p_count)
    peaked[rng.permutation(p_count)[: int(0.6 * p_count)]] = 1.0
    return {0.7: spread, 1.3: peaked}


def test_temperature_picks_right_t():
    base = _two_temperatures()
    p1 = np.full(100, 0.7)
    res = M.temperature_baseline(p1, base, 4, 20, np.random.default_rng(0))
    assert res["t_star_E_1"] == {0.7: 0.0, 1.3: 1.0}
    assert res["t_star_E_N"] == {0.7: 1.0, 1.3: 0.0}
    assert res["pass_at_n_estimator"] == "plug_in"
    assert res["E_1"] == pytest.approx(0.7 - 0.6, abs=1e-12)  # mean over both halves = full mean
    assert res["E_N"] == pytest.approx((1 - 0.3**4) - (1 - 0.6**4), abs=1e-12)
    assert len(res["per_split"]["E_1"]) == 20


def test_temperature_estimator_choice():
    p = np.full(10, 0.25)
    by_t = {1.0: p}
    trials = np.full(10, 32)
    res = M.temperature_baseline(p, by_t, 16, 2, np.random.default_rng(0), trials1=trials, trials_by_T={1.0: trials})
    assert res["pass_at_n_estimator"] == "unbiased"
    assert res["E_N"] == pytest.approx(0.0, abs=1e-15)
    res = M.temperature_baseline(p, by_t, 64, 2, np.random.default_rng(0), trials1=trials, trials_by_T={1.0: trials})
    assert res["pass_at_n_estimator"] == "plug_in"


def test_split_half_symmetric():
    rng = np.random.default_rng(5)
    base = rng.random((4, 51))
    trained = rng.random(51)
    perm = rng.permutation(51)
    h1, h2 = perm[:25], perm[25:]
    v12, a, b = M.split_half_value(trained, base, h1, h2)
    v21, b2, a2 = M.split_half_value(trained, base, h2, h1)
    assert v12 == v21 and (a, b) == (a2, b2)
    manual = 0.5 * (np.mean(trained[h2] - base[a, h2]) + np.mean(trained[h1] - base[b, h1]))
    assert v12 == pytest.approx(manual, abs=1e-15)


def test_temperature_split_removes_selection_bias():
    # many temperatures that are pure noise around the trained rate: in-sample max is biased, split-half is not
    rng = np.random.default_rng(11)
    p_count = 200
    true = rng.uniform(0.2, 0.8, p_count)
    p1 = rng.binomial(32, true) / 32
    es = []
    for rep in range(200):
        by_t = {float(t): rng.binomial(32, true) / 32 for t in range(8)}
        es.append(M.temperature_baseline(p1, by_t, 16, 1, np.random.default_rng(rep))["E_1"])
    # split-half E_1 has mean ~ p1 - true (bias from p1's own noise only), not a systematic negative shift
    assert abs(np.mean(es) - np.mean(p1 - true)) < 0.003


# --------------------------------------------------------------------------- bootstrap


def test_bootstrap_deterministic_and_contains_point():
    _truth, a, b, t, _, _ = _planted("reliability")
    r1 = M.bootstrap(a.successes, a.trials, b.rates(), t.rates(), N, ALPHA, 300, 7)
    r2 = M.bootstrap(a.successes, a.trials, b.rates(), t.rates(), N, ALPHA, 300, 7)
    assert np.array_equal(r1["samples"], r2["samples"])
    for key in M.CLASSES + ("total",):
        assert r1[key]["lo"] <= r1[key]["estimate"] <= r1[key]["hi"]
    np.testing.assert_allclose(r1["samples"][:, :3].sum(axis=1), r1["samples"][:, 3], atol=1e-12)


def test_bootstrap_reclassifies_ambiguous_problems():
    # problems with 2/64 set-A successes sit near 1/N: redraws move them between classes
    p_count = 50
    k = np.full(p_count, 2)
    p0, p1 = np.zeros(p_count), np.full(p_count, 0.5)
    r = M.bootstrap(k, M_A, p0, p1, N, ALPHA, 200, 0)
    assert r["ambiguous"]["estimate"] == pytest.approx(0.5)
    assert r["unreachable"]["hi"] > 0  # some redraws give 0/64 -> unreachable


def test_bootstrap_coverage_of_total():
    """500 simulated datasets with known per-problem rates; the 95% interval for the total gain must cover
    the truth in >= 90% of them."""
    rng = np.random.default_rng(2024)
    p_count = 100
    p0 = np.concatenate([rng.beta(0.6, 2.0, 80), np.zeros(20)])
    p1 = np.clip(p0 + rng.normal(0.05, 0.05, p_count), 0, 1)
    truth = float(np.mean(p1 - p0))
    covered = 0
    for rep in range(500):
        k_a = rng.binomial(M_A, p0)
        p0_hat = rng.binomial(M_A, p0) / M_A
        p1_hat = rng.binomial(M_A, p1) / M_A
        r = M.bootstrap(k_a, M_A, p0_hat, p1_hat, N, ALPHA, 200, rep)
        covered += r["total"]["lo"] <= truth <= r["total"]["hi"]
    print(f"coverage of total gain: {covered / 500:.3f}")
    assert covered / 500 >= 0.90


def test_bootstrap_temperature_intervals():
    _truth, a, b, t, _, by_t = _planted("capability")
    r = M.bootstrap(a.successes, a.trials, b.rates(), t.rates(), N, ALPHA, 200, 1,
                    base_by_T={k: v.rates() for k, v in by_t.items()}, trials1=t.trials,
                    trials_by_T={k: v.trials for k, v in by_t.items()})
    temp = r["temperature"]
    assert temp["pass_at_n_estimator"] == "unbiased"
    assert temp["E_1"]["lo"] > 0  # capability gain on unreachable problems cannot come from temperature


# --------------------------------------------------------------------------- null check


def test_null_check_passes_on_same_p_and_fails_on_shift():
    rng = np.random.default_rng(9)
    p_count = 300
    p = rng.beta(0.7, 2.0, p_count)
    k_a = rng.binomial(M_A, p)
    p0 = rng.binomial(M_A, p) / M_A
    p_null = rng.binomial(M_A, p) / M_A
    ok = M.null_check(k_a, M_A, p0, p_null, N, ALPHA, 1000, 0)
    assert ok["passed"]
    p_shift = rng.binomial(M_A, np.clip(p + 0.1, 0, 1)) / M_A
    bad = M.null_check(k_a, M_A, p0, p_shift, N, ALPHA, 1000, 0)
    assert not bad["passed"] and bad["total"]["lo"] > 0


# --------------------------------------------------------------------------- budget curve


def test_budget_curve_truncation():
    rng = np.random.default_rng(4)
    p = rng.beta(0.5, 3.0, 120)
    k_a = rng.binomial(M_A, p)
    p0, p1 = rng.binomial(M_A, p) / M_A, rng.binomial(M_A, np.clip(p + 0.05, 0, 1)) / M_A
    grid = [1, 2, 4, 8, 16, 32, 64]
    curve = M.budget_curve(k_a, M_A, p0, p1, grid, ALPHA)
    assert curve["n_max_resolvable"] == math.floor(1 / (1 - 0.025 ** (1 / 64))) == 17
    assert curve["n"] == [1, 2, 4, 8, 16] and curve["dropped"] == [32, 64] and curve["truncated"]
    assert "floor(1/u0) = 17" in curve["reason"]
    for i, n in enumerate(curve["n"]):
        parts = curve["reachable"][i] + curve["unreachable"][i] + curve["ambiguous"][i]
        assert abs(parts - curve["total"][i]) <= 1e-12
        assert curve["total"][i] == pytest.approx(np.mean(p1 - p0), abs=1e-15)
        s = M.split(M.classify(k_a, M_A, n, ALPHA), p0, p1)
        assert curve["unreachable"][i] == s["unreachable"]
    # larger N -> threshold 1/N lower -> fewer problems can be confidently unreachable... and more reachable
    reach = [c["reachable"] for c in curve["counts"]]
    unreach = [c["unreachable"] for c in curve["counts"]]
    assert reach == sorted(reach) and unreach == sorted(unreach, reverse=True)
    assert M.budget_curve(k_a, M_A, p0, p1, [4, 8], ALPHA)["truncated"] is False


# --------------------------------------------------------------------------- planted cases


@pytest.mark.parametrize("kind,gain_class,zero_class", [("reliability", "reachable", "unreachable"),
                                                         ("capability", "unreachable", "reachable")])
def test_planted_gain_lands_in_right_class(kind, gain_class, zero_class):
    truth, a, b, t, null, by_t = _planted(kind)
    s = M.split(M.classify(a.successes, a.trials, N, ALPHA), b.rates(), t.rates())
    assert s["counts"] == {"reachable": 100, "unreachable": 100, "ambiguous": 0}
    r = M.bootstrap(a.successes, a.trials, b.rates(), t.rates(), N, ALPHA, 1000, 0)
    for key in M.CLASSES + ("total",):
        assert r[key]["lo"] <= truth[key] <= r[key]["hi"], (key, truth[key], r[key])
    assert r[gain_class]["lo"] > 0
    assert r[zero_class]["lo"] <= 0 <= r[zero_class]["hi"]

    groups = {"all": list(range(200))}
    (gm,) = M.group_metrics(groups, a, b, t, by_t, N, ALPHA, 300, 0, 10, counts_null=null)
    assert isinstance(gm, GroupMetrics)
    assert abs(gm.delta_rel + gm.delta_cap - gm.delta_total) <= 1e-12
    assert gm.extra["null_check"]["passed"]
    if kind == "capability":
        assert gm.extra["beyond_reach"] and gm.extra["beyond_temperature"]
        assert gm.label == "capability"
        assert set(gm.extra["crossed"]) <= set(range(100, 200)) and len(gm.extra["crossed"]) > 90
    else:
        # the T=0.5 base already reproduces the gain: neither beyond reach nor beyond temperature
        assert not gm.extra["beyond_reach"] and not gm.extra["beyond_temperature"]
        assert gm.label == "pure_sharpening"
        assert gm.extra["temperature"]["t_star_E_1"][0.5] == 1.0


def test_group_metrics_groups_and_schema():
    _truth, a, b, t, _null, by_t = _planted("capability")
    groups = {"reach": list(range(100)), "unreach": list(range(100, 200))}
    out = M.group_metrics(groups, a, b, t, by_t, N, ALPHA, 100, 0, 4)
    assert [g.group for g in out] == ["reach", "unreach"]
    assert out[0].n_prompts == 100 and out[0].n == N and out[0].c == 1.0
    assert out[1].delta_cap == pytest.approx(out[1].delta_total, abs=1e-12)
    d = out[1].to_dict()
    assert set(d["excess"]) == {"per_step", "sequence", "both"}
