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


# ------------------------------------------------------------------- arm A and robustness runs

from rq2_time_to_cross import run_arm_a, run_robust  # noqa: E402

from capsplit.runlog import load_seeds  # noqa: E402


def test_arm_a_quick_matches_scalar(tmp_path):
    rows = load_seeds(run_arm_a(quick=True, root=str(tmp_path)))
    assert len(rows) == 3
    for r in rows:
        assert int(r["updates"]) == int(r["updates_scalar"]) > 0
        assert np.max(np.abs(r["q"][:, 0] - r["q_scalar"])) < 1e-9
        assert float(r["q0"]) == pytest.approx(float(r["q0_target"]), rel=0.01)


def test_robust_quick_writes_both_conditions(tmp_path):
    rows = load_seeds(run_robust(quick=True, root=str(tmp_path)))
    assert {str(r["condition"]) for r in rows} == {"R1", "R2"}
    assert all(int(r["updates"]) > 0 for r in rows)
    assert {int(r["horizon"]) for r in rows if str(r["condition"]) == "R2"} == {9}


# ---------------------------------------------------------- arm B and the sampled stretch

from rq2_time_to_cross import run_arm_b, run_sampled  # noqa: E402

from capsplit.crossing import crossing_step  # noqa: E402
from capsplit.train.tabular import trace_exact_pg  # noqa: E402


def test_w1_noise_free_symmetric_crosses_and_rare_does_not():
    world, common, rare = competing_world("W1")
    table, si = compile_grid(world)
    watch = [table.solutions.index(tuple(rare)), table.solutions.index(tuple(common))]
    for ratio, crosses in ((1.0, True), (0.1, False)):
        logits, _, q0_rare = competing_base(table, si, "W1", ratio, 0.0, 0)
        q, _ = trace_exact_pg(table, logits, 2.0, 2000, watch)
        assert (crossing_step(q[:, 0], CROSS_FACTOR * q0_rare) is not None) is crosses


def test_arm_b_quick_writes_expected_fields(tmp_path):
    rows = load_seeds(run_arm_b(quick=True, root=str(tmp_path)))
    assert {float(r["ratio"]) for r in rows} == {1.0, 0.1}
    for r in rows:
        assert r["q_rare"].dtype == np.float32 and r["q_rare"].shape == (2, 2001)
        assert np.all(r["line_applicable"])  # q0(rare) < 0.1 everywhere, given COMMON_BONUS
        assert np.all(CROSS_FACTOR * r["q0_rare"] <= 0.5)
        assert r["crossed_primary"].shape == (2,) and r["peak"].shape == (2,)


def test_sampled_quick_reports_updates_and_episodes(tmp_path):
    rows = load_seeds(run_sampled(quick=True, root=str(tmp_path)))
    assert rows
    for r in rows:
        ok = r["updates"] >= 0
        assert np.array_equal(r["episodes"][ok], 32 * r["updates"][ok])
        assert np.all(r["episodes"][~ok] == -1)


# ------------------------------------------------------------------- report and verdicts

from rq2_time_to_cross import main, p1_verdict, p2_verdict, p3_verdict, p4_verdict, report  # noqa: E402


def test_report_errors_when_runs_missing(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["report", "--root", str(tmp_path)])
    assert "arm-a" in capsys.readouterr().err


def test_report_on_quick_runs(tmp_path):
    root = str(tmp_path)
    for sub in ("arm-a", "robust", "arm-b", "sampled"):
        main([sub, "--quick", "--root", root])
    v = report(root)
    assert v["validity"]["arm_a_matches_scalar"] is True
    assert v["validity"]["arm_b_rule_achievable"] is True
    assert v["P1"]["verdict"] in {"supported", "inconclusive", "inconclusive (censored)"}
    assert v["P3"]["verdict"] == "not computed"  # quick mode runs one learning rate
    out = tmp_path / "rq2_report"
    for name in ("figures/arm_a_loglog.png", "figures/arm_b_fraction.png", "figures/arm_b_traces.png",
                 "tables/arm_a.csv", "tables/robust.csv", "tables/arm_b.csv", "tables/sampled.csv",
                 "verdicts.json", "verdicts.md"):
        assert (out / name).exists(), name


def test_p1_censored_point_gives_inconclusive():
    assert p1_verdict(q0s=[1e-4, 1e-3, 1e-2], updates=[None, 362, 63])["verdict"] == "inconclusive (censored)"


def test_p1_supported_on_the_derived_curve():
    from capsplit.crossing import scalar_isolated_updates

    qs = list(np.logspace(-6, -2, 9))
    v = p1_verdict(qs, [scalar_isolated_updates(q, 8, 1.0, 0.1, 400_000) for q in qs])
    assert v["verdict"] == "supported" and v["fitted"] == pytest.approx(-0.844, abs=0.001)


def test_p1_flat_is_contradicted():
    assert p1_verdict([1e-4, 1e-3, 1e-2], [100, 100, 100])["verdict"] == "contradicted"


def test_p2_rules():
    assert p2_verdict([-0.85] * 18 + [-0.6, None], -0.8)["verdict"] == "supported"
    assert p2_verdict([-0.85] * 17 + [-0.6] * 3, -0.8)["verdict"] == "inconclusive"
    assert p2_verdict([-0.1] * 20, -0.8)["verdict"] == "contradicted"
    assert p2_verdict([-0.85] * 20, -0.2)["verdict"] == "contradicted"


def test_p3_rules():
    assert p3_verdict({1e-4: {0.5: 4716, 1.0: 2358, 2.0: 1179}})["verdict"] == "holds"
    assert p3_verdict({1e-4: {0.5: 6000, 1.0: 2358, 2.0: 1179}})["verdict"] == "does not hold"
    assert p3_verdict({1e-4: {1.0: 2358}})["verdict"] == "not computed"


def test_p4_rules():
    assert p4_verdict({("W1", 0.5, 1.0): 0.0, ("W1", 0.1, 1.0): 0.05})["verdict"] == "supported"
    assert p4_verdict({("W1", 0.5, 1.0): 0.3, ("W1", 0.1, 1.0): 0.6})["verdict"] == "contradicted"
    assert p4_verdict({("W1", 0.5, 1.0): 0.3, ("W1", 0.1, 1.0): 0.2})["verdict"] == "inconclusive"
    assert p4_verdict({("W1", 1.0, 1.0): 1.0})["verdict"] == "not computed"


def _fake_run(root, name, quick, mtime):
    import json
    import os

    d = root / "rq2_arm_a" / name
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"config": {"quick": quick}}))
    os.utime(d / "manifest.json", (mtime, mtime))
    return d


def test_latest_run_prefers_full_runs_over_newer_quick_ones(tmp_path):
    from rq2_time_to_cross import _latest_run

    full = _fake_run(tmp_path, "full", False, 1_000)
    _fake_run(tmp_path, "quick", True, 2_000)
    assert _latest_run(str(tmp_path), "rq2_arm_a", "arm-a") == full


def test_report_flags_quick_only_runs(tmp_path):
    root = str(tmp_path)
    for sub in ("arm-a", "robust", "arm-b"):
        main([sub, "--quick", "--root", root])
    v = report(root)
    assert v["quick_runs_used"] is True
    assert "quick" in (tmp_path / "rq2_report" / "verdicts.md").read_text().lower()
