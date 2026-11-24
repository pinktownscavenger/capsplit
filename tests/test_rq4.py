import json

import numpy as np
import pytest
from rq4_passk import label_distribution, load_units

from capsplit.harness.config import parse_config
from capsplit.harness.runner import run_experiment


def _quick_config(save_policies: str) -> dict:
    with open("configs/grid_example.json") as f:
        data = json.load(f)
    data.update(seeds=[0], budget=4, checkpoints=[0, 2, 4], save_policies=save_policies)
    return data


@pytest.fixture(scope="module")
def quick_grid(tmp_path_factory):
    out = tmp_path_factory.mktemp("rq4_quick")
    return run_experiment(parse_config(_quick_config("final")), out_root=out, workers=1)


def _record(results_dir, unit):
    index = json.loads((results_dir / "index.json").read_text())
    for run in index["runs"]:
        if run["arm"] == unit.arm and run["seed"] == unit.seed:
            metrics = json.loads((results_dir / run["files"]["metrics"]).read_text())
            for rec in metrics["records"]:
                if rec["step"] == unit.step and rec["group"] == unit.group and rec["n"] == 64 and rec["c"] == 1.0:
                    return rec
    raise AssertionError("no matching record")


def test_units_skip_step0_and_none(quick_grid):
    units = load_units(quick_grid)
    assert units and all(u.step > 0 and u.trainer != "none" for u in units)
    assert {u.step for u in units} == {2, 4}


def test_unit_j_lists_average_to_record(quick_grid):
    units = load_units(quick_grid, groups=["all"])
    assert units
    for u in units:
        rec = _record(quick_grid, u)
        assert np.mean(u.j0) == pytest.approx(rec["j0"], abs=1e-9)
        assert np.mean(u.j1) == pytest.approx(rec["j1"], abs=1e-9)
        assert u.label == rec["label"] and u.label_en == rec["extra"]["label_en"]


def test_group_filter(quick_grid):
    units = load_units(quick_grid, groups=["heldout"])
    assert units and {u.group for u in units} == {"heldout"}
    assert all(len(u.j0) == len(u.j1) == 2 for u in units)  # grid_example has 2 held-out prompts


def test_missing_index_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="index.json"):
        load_units(tmp_path)


def test_label_distribution_counts_every_unit(quick_grid):
    units = load_units(quick_grid)
    dist = label_distribution(units)
    assert sum(sum(c.values()) for c in dist.values()) == len(units)


# ------------------------------------------------------------------- scoring and verdicts

import rq4_passk  # noqa: E402
from rq4_passk import Unit, main, q1_verdict, q3_verdict, rule_of, score_units  # noqa: E402

KS = (1, 2, 5, 10, 20, 50, 77)


def _unit(j0, j1, label, seed=0, arm="rloo"):
    return Unit(arm, "rloo", (), seed, 1000, "heldout", tuple(j0), tuple(j1), label, label)


def test_all_prompts_improve_is_capability():
    assert rule_of(_unit([0.1, 0.2], [0.5, 0.6], "reweighting"), KS) == "capability"


def test_one_prompt_drops_is_sharpening():
    assert rule_of(_unit([0.3, 0.05], [0.9, 0.0], "pure_sharpening"), KS) == "sharpening"


def test_saturated_ties_are_undetermined_and_count_as_disagreement():
    u = _unit([1.0, 1.0], [1.0, 1.0], "capability")
    assert rule_of(u, KS) == "undetermined"
    s = score_units([u], KS)
    assert s["primary"] == 0.0 and s["undetermined_rate"] == 1.0 and s["n_units"] == 1


def test_per_seed_rates_and_ci():
    units = [_unit([0.1], [0.5], "capability", seed=s) for s in range(5)] + [
        _unit([0.1], [0.5], "reweighting", seed=s) for s in range(5)
    ]
    s = score_units(units, KS)
    assert s["per_seed_primary"] == {i: 0.5 for i in range(5)}
    assert s["ci"] == pytest.approx((0.5, 0.5))


def test_q1_rules():
    assert q1_verdict((0.4, 0.6)) == "supported"
    assert q1_verdict((0.96, 0.99)) == "contradicted"
    assert q1_verdict((0.9, 0.97)) == "inconclusive"


def _cell(primary, runs=20):
    return {"primary": primary, "runs": runs}


def test_q3_rules_and_small_cells():
    good = {"compounded_sharpening": _cell(0.1), "reweighting": _cell(0.2), "capability": _cell(0.9), "pure_sharpening": _cell(0.8)}
    assert q3_verdict(good)["verdict"] == "supported"
    assert q3_verdict({**good, "reweighting": _cell(0.95)})["verdict"] == "contradicted"
    assert q3_verdict({**good, "capability": _cell(0.9, runs=5)})["verdict"] == "inconclusive"
    missing = {k: v for k, v in good.items() if k != "pure_sharpening"}
    assert q3_verdict(missing)["verdict"] == "inconclusive"


def test_score_refuses_before_freeze(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(rq4_passk, "_tag_is_ancestor", lambda: False)
    with pytest.raises(SystemExit):
        main(["score", "--phase2", str(tmp_path), "--phase1", str(tmp_path)])
    assert "rq4-frozen" in capsys.readouterr().err


# ------------------------------------------------------------------- report outputs and SRR/NDR


@pytest.fixture(scope="module")
def quick_grid_with_policies(tmp_path_factory):
    out = tmp_path_factory.mktemp("rq4_quick_policies")
    return run_experiment(parse_config(_quick_config("checkpoints")), out_root=out, workers=1)


def test_report_writes_everything_and_keeps_srr_out(tmp_path, quick_grid):
    main(["score", "--phase2", str(quick_grid), "--phase1", str(quick_grid), "--out", str(tmp_path), "--allow-unfrozen"])
    for name in ("tables/agreement.csv", "tables/k_large_sensitivity.csv", "tables/units.csv",
                 "figures/agreement_by_cell.png", "figures/disagreement_examples.png", "verdicts.json", "verdicts.md"):
        assert (tmp_path / name).exists(), name
    text = (tmp_path / "verdicts.md").read_text()
    assert "UNFROZEN TEST" in text
    assert "SRR" not in text.upper().replace("SRR/NDR PENDING", "")
    v = json.loads((tmp_path / "verdicts.json").read_text())
    assert v["frozen"] is False and set(v) >= {"Q1", "Q2", "Q3", "validity"}
    assert set(v["Q1"]["per_cell"]) == set(rq4_passk.CELLS)  # every cell reported, even if empty


def test_srr_ndr_cli_writes_only_unverified_file(tmp_path, quick_grid_with_policies):
    main(["srr-ndr", str(quick_grid_with_policies), "--out", str(tmp_path)])
    assert [p.name for p in tmp_path.iterdir()] == ["srr_ndr_UNVERIFIED.csv"]
    header = (tmp_path / "srr_ndr_UNVERIFIED.csv").read_text().splitlines()[0]
    assert header.split(",")[:4] == ["phase", "arm", "seed", "step"]


# ------------------------------------------------------------------- RQ3 reproduction


def test_repro_identical_perturbed_and_tiny_numeric(tmp_path):
    import shutil

    from rq4_passk import repro_compare

    shutil.copytree("docs/experiment_main", tmp_path / "same")
    assert repro_compare(tmp_path / "same")["match"] is True

    shutil.copytree("docs/experiment_main", tmp_path / "bad")
    v = json.loads((tmp_path / "bad/verdicts.json").read_text())
    v["tests"]["H1"]["verdict"] = "supported"
    (tmp_path / "bad/verdicts.json").write_text(json.dumps(v))
    r = repro_compare(tmp_path / "bad")
    assert r["match"] is False and r["files"]["verdicts.json"]["status"] == "different"

    shutil.copytree("docs/experiment_main", tmp_path / "tiny")
    s = json.loads((tmp_path / "tiny/summary.json").read_text())
    s["tau0"] = s.get("tau0", 0.02) + 1e-9 if "tau0" in s else None
    if s["tau0"] is None:
        s.pop("tau0")
        s["_probe"] = 1.0
    (tmp_path / "tiny/summary.json").write_text(json.dumps(s, indent=2))
    r = repro_compare(tmp_path / "tiny")
    assert r["files"]["summary.json"]["status"] in {"numeric", "different"}


# ------------------------------------------------------------------- SRR/NDR report


def _srr_row(seed, prompt, split, p, s, e, arm="rloo", step=4, factors="{}"):
    return ["phase", arm, seed, step, factors, prompt, split, p, s, e, None, None]


def test_srr_ndr_units_pool_heldout_prompts_and_carry_labels():
    from rq4_passk import srr_ndr_units

    rows = [_srr_row(0, 0, "train", 9, 9, 9), _srr_row(0, 1, "heldout", 2, 1, 0), _srr_row(0, 2, "heldout", 1, 0, 1),
            _srr_row(1, 1, "heldout", 0, 2, 0)]
    units = [Unit("rloo", "rloo", (), 0, 4, "heldout", (0.1,), (0.2,), "capability", "capability"),
             Unit("rloo", "rloo", (), 1, 4, "heldout", (0.1,), (0.0,), "pure_sharpening", "pure_sharpening")]
    out = srr_ndr_units(rows, units, group="heldout")
    assert [(u["seed"], u["P"], u["S"], u["E"], u["label"]) for u in out] == [(0, 3, 1, 1, "capability"),
                                                                             (1, 0, 2, 0, "pure_sharpening")]
    assert out[0]["srr"] == 0.75 and out[0]["ndr"] == 0.25 and out[1]["ndr"] is None


def test_srr_ndr_units_refuses_rows_without_a_label():
    from rq4_passk import srr_ndr_units

    with pytest.raises(KeyError):
        srr_ndr_units([_srr_row(5, 1, "heldout", 1, 0, 0)], [], group="heldout")


def test_srr_ndr_summary_pools_per_cell_and_lists_all_cells():
    from rq4_passk import srr_ndr_summary

    units = [{"label": "capability", "P": 1, "S": 1, "E": 2, "prompts": 2},
             {"label": "capability", "P": 3, "S": 1, "E": 0, "prompts": 2},
             {"label": "pure_sharpening", "P": 0, "S": 0, "E": 0, "prompts": 2}]
    s = srr_ndr_summary(units)
    assert s["overall"]["units"] == 3 and s["overall"]["prompts"] == 6 and s["overall"]["srr"] == pytest.approx(4 / 6)
    assert s["per_cell"]["capability"]["srr"] == pytest.approx(4 / 6) and s["per_cell"]["capability"]["ndr"] == 2 / 6
    assert s["per_cell"]["pure_sharpening"]["srr"] is None
    assert s["per_cell"]["reweighting"]["units"] == 0  # every cell reported, even if empty
    assert s["undefined_units"] == {"srr": 1, "ndr": 1}


def test_insert_srr_section_replaces_pending_line_and_is_idempotent(tmp_path):
    from rq4_passk import SRR_PENDING, insert_srr_section

    path = tmp_path / "verdicts.md"
    path.write_text(f"# head\n\nbody\n\n{SRR_PENDING}\n")
    insert_srr_section(path, ["## SRR and NDR", "", "first"])
    text = path.read_text()
    assert SRR_PENDING not in text and "first" in text and text.startswith("# head\n\nbody\n")
    insert_srr_section(path, ["## SRR and NDR", "", "second"])
    text = path.read_text()
    assert "second" in text and "first" not in text and text.count("## SRR and NDR") == 1


def test_srr_ndr_report_cli_end_to_end(tmp_path, quick_grid_with_policies):
    out = tmp_path / "report"
    main(["score", "--phase2", str(quick_grid_with_policies), "--phase1", str(quick_grid_with_policies),
          "--out", str(out), "--allow-unfrozen"])
    main(["srr-ndr-report", "--phase2", str(quick_grid_with_policies), "--phase1", str(quick_grid_with_policies),
          "--out", str(out)])
    text = (out / "verdicts.md").read_text()
    assert "SRR/NDR pending" not in text and "## SRR and NDR" in text
    data = json.loads((out / "srr_ndr.json").read_text())
    assert set(data["phases"]) == {"phase2", "phase1"} and data["settings"]["eps"] == pytest.approx(1 / 64)
    units = (out / "tables/srr_ndr_units.csv").read_text().splitlines()
    assert units[0].split(",")[:4] == ["phase", "arm", "factors", "seed"]
    assert len(units) - 1 == 2 * data["phases"]["phase1"]["overall"]["units"]
