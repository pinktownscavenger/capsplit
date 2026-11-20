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
