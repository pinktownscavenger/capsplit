"""T6: report generator — tables, validity checks, sensitivity flags, bootstrap, determinism."""
import csv
import importlib.util
import json
import pathlib

import numpy as np
import pytest

from capsplit.harness.report import bootstrap_ci, build_report

ROOT = pathlib.Path(__file__).resolve().parents[1]
DUMMY = ROOT / "configs" / "dummy.json"
HAS_MPL = importlib.util.find_spec("matplotlib") is not None
GROUPS = ("all", "easy_heldout", "easy_train", "hard_heldout", "hard_train")


# ---------------------------------------------------------------------- synthetic results folder


def _record(group, n, c, step, gain_rel, gain_cap, seed, e_both=0.05, label="capability"):
    j0 = 0.2 + 0.01 * seed
    j1 = j0 + gain_rel + gain_cap
    best_pass = 0.6
    return {
        "group": group, "n": n, "c": c, "n_prompts": 2, "j0": j0, "j1": j1,
        "delta_rel": gain_rel, "delta_cap": gain_cap, "delta_total": j1 - j0,
        "excess": {"per_step": e_both - 0.01, "sequence": e_both - 0.02, "both": e_both},
        "label": label, "step": step,
        "extra": {
            "d2_label": "pure_sharpening", "crossed_count": 1, "saturated": False, "boundary_hit": False,
            "frontier": {"per_step": {"best_pass": best_pass, "best_j": 0.3, "best_beta": 2.0, "boundary_hit": False}},
            "j0": [j0, j0], "j1": [j1, j1], "delta_cap": [gain_cap, gain_cap],
        },
    }


def make_results(root: pathlib.Path, tweak=None, primary=None) -> pathlib.Path:
    """Two arms (none, rl) x factor task.k in {1, 2} x seeds {0, 1, 2}; rl gains on every group.

    `tweak(arm, k, seed, record)` may edit each record in place.
    """
    exp = root / "synth"
    measure = {"ns": [8, 32], "cs": [0.5, 1.0, 2.0], "group_by": ["difficulty", "split"]}
    if primary is not None:
        measure["primary"] = primary
    arms = [{"name": "none", "trainer": "none", "params": {}}, {"name": "rl", "trainer": "rloo", "params": {}}]
    config = {"name": "synth", "task": {"type": "x", "params": {}}, "base": {"type": "y", "params": {}},
              "backend": {"type": "exact", "params": {}}, "factors": {"task.k": [1, 2]}, "arms": arms,
              "seeds": [0, 1, 2], "budget": 5, "checkpoints": [0, 5], "measure": measure}
    exp.mkdir(parents=True)
    (exp / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    runs = []
    for k in (1, 2):
        for arm in arms:
            cell = f"cell-{arm['name']}-{k}"
            for seed in (0, 1, 2):
                records = []
                for step in (0, 5):
                    for group in GROUPS:
                        for n in (8, 32):
                            for c in (0.5, 1.0, 2.0):
                                trained = arm["trainer"] != "none" and step > 0
                                rel = 0.1 * k if trained else 0.0
                                cap = 0.02 * (seed + 1) if trained else 0.0
                                rec = _record(group, n, c, step, rel, cap, seed,
                                              e_both=0.05 if trained else -0.1,
                                              label="capability" if trained else "pure_sharpening")
                                if tweak is not None:
                                    tweak(arm["name"], k, seed, rec)
                                records.append(rec)
                metrics = {"arm": arm["name"], "trainer": arm["trainer"], "seed": seed, "factors": {"task.k": k},
                           "records": records}
                files = {"manifest": f"{cell}/manifest.json", "metrics": f"{cell}/seed-{seed}.metrics.json", "policies": None}
                (exp / cell).mkdir(exist_ok=True)
                (exp / files["metrics"]).write_text(json.dumps(metrics, sort_keys=True))
                runs.append({"factors": {"task.k": k}, "arm": arm["name"], "trainer": arm["trainer"], "seed": seed,
                             "run_dir": cell, "base": "bases/b", "files": files})
    index = {"name": "synth", "config": "config.json", "factors": {"task.k": [1, 2]}, "arms": arms,
             "seeds": [0, 1, 2], "budget": 5, "checkpoints": [0, 5],
             "measure": {k: v for k, v in measure.items() if k != "primary"}, "save_policies": "none", "runs": runs}
    (exp / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    return exp


def _rows(path):
    with open(path) as handle:
        return list(csv.DictReader(handle))


def _warnings(out: pathlib.Path) -> list[str]:
    return [line for line in (out / "report.md").read_text().splitlines() if "**WARNING**" in line]


# ---------------------------------------------------------------------- tests


def test_report_from_dummy_runner_output(tmp_path, capsys):
    pytest.importorskip("torch")
    from capsplit.harness.runner import run_experiment

    exp = run_experiment(DUMMY, out_root=tmp_path)
    out = build_report(exp)
    assert out == exp / "report"
    md = (out / "report.md").read_text()
    for heading in ("## Validity checks", "## Main table (N=64, c=1, step 10)", "## Trajectories", "## Sensitivity"):
        assert heading in md
    assert "exact" in md and "seed-to-seed spread" in md
    assert _warnings(out) == []
    err = capsys.readouterr().err
    assert "PASS: [proposition 1]" in err and "PASS: [zero-gain control]" in err and "WARNING" not in err
    summary = json.loads((out / "summary.json").read_text())
    assert (summary["primary"]["n"], summary["primary"]["c"]) == (64, 1.0)
    assert summary["groups"][:4] == ["hard_heldout", "hard_train", "easy_train", "easy_heldout"]
    for name in ("main", "labels", "trajectory", "frontier", "sensitivity", "sensitivity_flips", "checks"):
        assert (out / "tables" / f"{name}.csv").exists()
    if HAS_MPL:
        assert (out / "figures" / "main_grid.png").stat().st_size > 0
        assert (out / "figures" / "frontier.png").exists()
        assert len(list((out / "figures").glob("training_*.png"))) == 2


def test_report_from_synthetic_folder(tmp_path, capsys):
    exp = make_results(tmp_path)
    out = build_report(exp)
    assert _warnings(out) == []
    assert "WARNING" not in capsys.readouterr().err
    rows = _rows(out / "tables" / "main.csv")
    cell = {(r["arm"], r["factors"], r["group"], r["metric"]): r for r in rows}
    row = cell[("rl", "task.k=2", "hard_heldout", "delta_cap")]
    assert float(row["mean"]) == pytest.approx(0.04)  # seeds give 0.02, 0.04, 0.06
    assert float(row["min"]) == pytest.approx(0.02) and float(row["max"]) == pytest.approx(0.06)
    assert 0.02 <= float(row["ci_low"]) <= 0.04 <= float(row["ci_high"]) <= 0.06
    assert float(cell[("rl", "task.k=2", "hard_heldout", "delta_rel")]["mean"]) == pytest.approx(0.2)
    assert float(cell[("none", "task.k=1", "all", "E_both")]["mean"]) == pytest.approx(-0.1)
    frontier = {(r["arm"], r["factors"], r["group"]): r for r in _rows(out / "tables" / "frontier.csv")}
    f = frontier[("rl", "task.k=1", "all")]
    assert float(f["best_tempered_pass_at_n"]) == pytest.approx(0.6) and f["source"] == "extra.frontier"
    labels = _rows(out / "tables" / "labels.csv")
    assert {"arm": "rl", "factors": "task.k=1", "group": "all", "kind": "label", "label": "capability", "count": "3"} in labels
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n_warnings"] == 0 and summary["sensitivity_flips"] == []
    trajectory = _rows(out / "tables" / "trajectory.csv")
    assert {r["step"] for r in trajectory} == {"0", "5"}


def test_proposition1_violation_and_nonzero_none_arm_warn(tmp_path, capsys):
    def tweak(arm, k, seed, rec):
        if arm == "rl" and seed == 1 and rec["step"] == 5 and rec["group"] == "all":
            rec["delta_rel"] += 1e-6  # breaks delta_rel + delta_cap = j1 - j0
        if arm == "none" and seed == 0 and rec["step"] == 5 and rec["group"] == "hard_train":
            rec["j1"] += 0.01
            rec["delta_rel"] = rec["delta_total"] = 0.01

    out = build_report(make_results(tmp_path, tweak))
    err = capsys.readouterr().err
    assert "WARNING: [proposition 1]" in err and "WARNING: [zero-gain control]" in err
    assert "PASS: [step 0]" in err
    warnings = "\n".join(_warnings(out))
    assert "[proposition 1]" in warnings and "[zero-gain control]" in warnings


def test_step0_nonzero_gain_warns(tmp_path, capsys):
    def tweak(arm, k, seed, rec):
        if arm == "rl" and rec["step"] == 0 and rec["group"] == "easy_train":
            rec["delta_cap"] = 0.001
            rec["j1"] += 0.001
            rec["delta_total"] = 0.001

    build_report(make_results(tmp_path, tweak))
    err = capsys.readouterr().err
    assert "WARNING: [step 0]" in err and "PASS: [proposition 1]" in err


def test_sign_flip_across_c_triggers_sensitivity_flag(tmp_path, capsys):
    def tweak(arm, k, seed, rec):
        if arm == "rl" and rec["step"] == 5 and rec["c"] == 2.0 and rec["group"] == "hard_heldout":
            rec["delta_rel"] += 2 * rec["delta_cap"]
            rec["delta_cap"] = -rec["delta_cap"]

    out = build_report(make_results(tmp_path, tweak))
    err = capsys.readouterr().err
    assert "WARNING: [sensitivity] N=8 c=2" in err and "WARNING: [sensitivity] N=32 c=2" in err
    assert "PASS: [proposition 1]" in err
    flips = _rows(out / "tables" / "sensitivity_flips.csv")
    assert flips and all(r["c"] == "2.0" and r["group"] == "hard_heldout" for r in flips)
    assert {(r["conclusion"], r["primary_value"], r["value"]) for r in flips} == {("delta_cap sign", "+", "-")}
    assert {r["factors"] for r in flips} == {"task.k=1", "task.k=2"}
    sens = {(r["n"], r["c"]): r for r in _rows(out / "tables" / "sensitivity.csv")}
    assert sens[("32", "1.0")]["is_primary"] == "yes" and sens[("32", "1.0")]["n_flips"] == "0"
    assert sens[("8", "2.0")]["n_flips"] == "2"
    assert json.loads((out / "summary.json").read_text())["sensitivity_flips"]


def test_saturation_and_boundary_flags_warn(tmp_path, capsys):
    def tweak(arm, k, seed, rec):
        if arm == "rl" and seed == 2 and rec["group"] == "easy_train":
            rec["extra"]["saturated"] = True
        if arm == "rl" and rec["n"] == 8:  # not at the primary N: reported but no warning
            rec["extra"]["boundary_hit"] = True

    build_report(make_results(tmp_path, tweak))
    err = capsys.readouterr().err
    assert "WARNING: [saturated]" in err and "PASS: [boundary_hit]" in err


def test_primary_from_config_and_missing_fields(tmp_path, capsys):
    def tweak(arm, k, seed, rec):
        del rec["excess"], rec["extra"], rec["label"], rec["delta_total"]

    out = build_report(make_results(tmp_path, tweak, primary={"n": 8, "c": 0.5}))
    summary = json.loads((out / "summary.json").read_text())
    assert (summary["primary"]["n"], summary["primary"]["c"]) == (8, 0.5)
    assert summary["primary"]["rule"] == "config measure.primary"
    cell = next(c for c in summary["cells"] if c["arm"] == "rl" and c["group"] == "all")
    assert cell["metrics"]["E_both"]["mean"] is None and cell["metrics"]["delta_total"]["mean"] is not None
    assert cell["label_majority"] == "n/a"
    assert "PASS: [saturated] not recorded" in capsys.readouterr().err


def test_bootstrap_interval_on_known_data():
    assert bootstrap_ci([0.3, 0.3, 0.3]) == (pytest.approx(0.3), pytest.approx(0.3))
    assert bootstrap_ci([0.0, 1.0]) == (0.0, 1.0)  # resampled means are 0, 0.5, 1 w.p. 1/4, 1/2, 1/4
    data = [0.1, 0.4, 0.2, 0.9, 0.5]
    rng = np.random.default_rng(0)
    means = np.asarray(data)[rng.integers(0, 5, size=(10_000, 5))].mean(axis=1)
    assert bootstrap_ci(data) == (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))
    lo, hi = bootstrap_ci(data)
    assert min(data) < lo < np.mean(data) < hi < max(data)
    assert bootstrap_ci(data) == bootstrap_ci(data)


def test_report_is_deterministic(tmp_path):
    exp = make_results(tmp_path)
    a, b = build_report(exp, tmp_path / "a"), build_report(exp, tmp_path / "b")
    files_a = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    files_b = sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    assert files_a == files_b
    compared = [p for p in files_a if p.suffix != ".png"]
    assert any(p.suffix == ".md" for p in compared) and any(p.suffix == ".json" for p in compared)
    for rel in compared:
        assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel


def test_cli_report(tmp_path, capsys):
    from capsplit.harness.cli import main

    exp = make_results(tmp_path)
    assert main(["report", str(exp)]) == 0
    assert str(exp / "report") in capsys.readouterr().out
    assert (exp / "report" / "report.md").exists()


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_figures_written(tmp_path):
    out = build_report(make_results(tmp_path))
    names = sorted(p.name for p in (out / "figures").glob("*.png"))
    assert names == ["frontier.png", "main_grid.png", "training_00_none.png", "training_01_rl.png"]
    assert "figures/main_grid.png" in (out / "report.md").read_text()
