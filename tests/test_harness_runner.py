"""T1: config validation, measurement groups, and the experiment runner end to end."""
import copy
import json
import os
import pathlib
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from capsplit.harness import TASKS  # noqa: E402
from capsplit.harness.config import ConfigError, build_groups, load_config, parse_config  # noqa: E402
from capsplit.harness.runner import run_experiment  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
DUMMY = ROOT / "configs" / "dummy.json"


def _dummy() -> dict:
    return json.loads(DUMMY.read_text())


# ---------------------------------------------------------------------- config


def test_dummy_config_parses():
    config = load_config(DUMMY)
    assert [a.name for a in config.arms] == ["none", "none_again"]
    assert config.factor_combos() == [{"task.length": 3}]
    assert config.resolved({"task.length": 4})["task"].params == {"vocab": 3, "length": 4}
    assert config.measure.cs == (0.5, 1.0, 2.0)


def test_factor_grid_is_a_cartesian_product_in_sorted_key_order():
    data = _dummy()
    data["factors"] = {"task.length": [2, 3], "base.init_scale": [0.0, 1.0]}
    combos = parse_config(data).factor_combos()
    assert combos == [
        {"base.init_scale": 0.0, "task.length": 2},
        {"base.init_scale": 0.0, "task.length": 3},
        {"base.init_scale": 1.0, "task.length": 2},
        {"base.init_scale": 1.0, "task.length": 3},
    ]


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda d: d["task"].update(type="no_such_task"), "unknown task"),
        (lambda d: d["base"].update(type="nope"), "unknown base recipe"),
        (lambda d: d["backend"].update(type="nope"), "unknown backend"),
        (lambda d: d["arms"][0].update(trainer="nope"), "unknown trainer"),
        (lambda d: d["arms"][1].update(name="none"), "duplicate arm names"),
        (lambda d: d.update(checkpoints=[0, 5]), "include 0 and budget"),
        (lambda d: d.update(checkpoints=[5, 10]), "include 0 and budget"),
        (lambda d: d.update(checkpoints=[0, 10, 11]), r"lie in \[0, budget"),
        (lambda d: d.update(factors={"arm.lr": [1]}), "factor key"),
        (lambda d: d.update(factors={"task": [1]}), "factor key"),
        (lambda d: d.update(factors={"task.length": []}), "non-empty list"),
        (lambda d: d.update(save_policies="all"), "save_policies"),
        (lambda d: d.update(seeds=[0, 0]), "duplicates"),
        (lambda d: d.update(budget=-1), "budget"),
        (lambda d: d.pop("arms"), "missing config keys"),
        (lambda d: d.update(extra=1), "unknown config keys"),
        (lambda d: d["measure"].update(cs=[0]), "positive"),
        (lambda d: d.update(name="a/b"), "folder name"),
    ],
)
def test_invalid_configs_are_rejected(change, match):
    data = _dummy()
    change(data)
    with pytest.raises(ConfigError, match=match):
        parse_config(data)


# ---------------------------------------------------------------------- groups


def test_groups_from_tags():
    task = TASKS.create("tiny_prompt")  # tags: 0 easy/train, 1 hard/train, 2 easy/heldout, 3 hard/heldout
    groups = build_groups(task, ["difficulty", "split"])
    assert list(groups) == [
        "all", "easy", "hard", "heldout", "train",
        "easy_heldout", "easy_train", "hard_heldout", "hard_train",
    ]
    assert groups["all"] == (0, 1, 2, 3)
    assert groups["hard"] == (1, 3) and groups["train"] == (0, 1)
    assert groups["hard_train"] == (1,) and groups["easy_heldout"] == (2,)
    assert build_groups(task, []) == {"all": (0, 1, 2, 3)}
    assert list(build_groups(task, ["split"])) == ["all", "heldout", "train"]
    with pytest.raises(ConfigError, match="no prompt has the tag"):
        build_groups(task, ["colour"])


# ---------------------------------------------------------------------- runner


def _cli_run(out: pathlib.Path, *extra: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "capsplit.harness", "run", str(DUMMY), "--out", str(out), *extra],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result


def _files(directory: pathlib.Path) -> dict:
    return {str(p.relative_to(directory)): p for p in sorted(directory.rglob("*")) if p.is_file()}


def _load_pt(path: pathlib.Path):
    obj = torch.load(path, weights_only=False)
    return obj.state_dict() if hasattr(obj, "state_dict") and not isinstance(obj, dict) else obj


def _assert_same_outputs(a: pathlib.Path, b: pathlib.Path) -> None:
    fa, fb = _files(a), _files(b)
    assert list(fa) == list(fb)
    for name in fa:
        if name.endswith(".json"):
            assert fa[name].read_bytes() == fb[name].read_bytes(), name
        elif name.endswith(".pt"):
            xa, xb = _load_pt(fa[name]), _load_pt(fb[name])
            assert xa.keys() == xb.keys()
            for key in xa:
                ta, tb = xa[key], xb[key]
                if isinstance(ta, dict):  # {step: state_dict}
                    for k in ta:
                        assert torch.equal(ta[k], tb[k]), (name, key, k)
                else:
                    assert torch.equal(ta, tb), (name, key)
        else:
            raise AssertionError(f"unexpected file {name}")


@pytest.fixture(scope="module")
def cli_runs(tmp_path_factory):
    first = tmp_path_factory.mktemp("first")
    second = tmp_path_factory.mktemp("second")
    _cli_run(first)
    _cli_run(second)
    return first / "dummy", second / "dummy"


def test_cli_runs_end_to_end(cli_runs):
    exp, _ = cli_runs
    index = json.loads((exp / "index.json").read_text())
    assert len(index["runs"]) == 4  # 1 factor combination x 2 arms x 2 seeds
    assert [(r["arm"], r["seed"]) for r in index["runs"]] == [("none", 0), ("none", 1), ("none_again", 0), ("none_again", 1)]
    assert json.loads((exp / "config.json").read_text()) == _dummy()
    for run in index["runs"]:
        manifest = json.loads((exp / run["files"]["manifest"]).read_text())
        assert manifest["config"]["arm"]["name"] == run["arm"]
        assert manifest["config"]["task"]["params"]["length"] == 3
        assert (exp / run["base"] / "info.json").exists() and (exp / run["base"] / "policy.pt").exists()
        metrics = json.loads((exp / run["files"]["metrics"]).read_text())
        assert metrics["train_prompt_ids"] == [0, 1]
        records = metrics["records"]
        # 3 checkpoints x 9 groups x 2 ns x 3 cs; the no-training arm gains exactly zero.
        assert len(records) == 3 * 9 * 2 * 3
        assert {r["step"] for r in records} == {0, 5, 10}
        assert all(r["delta_total"] == 0.0 for r in records)
        saved = torch.load(exp / run["files"]["policies"], weights_only=False)
        assert sorted(saved) == [0, 5, 10]
    # Both arms of one seed share the cached base.
    assert index["runs"][0]["base"] == index["runs"][2]["base"] != index["runs"][1]["base"]


def test_second_run_regenerates_identical_outputs(cli_runs):
    _assert_same_outputs(*cli_runs)


def test_workers_match_serial(cli_runs, tmp_path):
    parallel = run_experiment(DUMMY, out_root=tmp_path, workers=2)
    _assert_same_outputs(cli_runs[0], parallel)


def test_rerun_skips_existing_runs_and_force_reruns(tmp_path):
    config = load_config(DUMMY)
    exp = run_experiment(config, out_root=tmp_path)
    index = json.loads((exp / "index.json").read_text())
    metrics = [exp / r["files"]["metrics"] for r in index["runs"]]
    before = {p: p.stat().st_mtime_ns for p in metrics}
    metrics[0].unlink()
    run_experiment(config, out_root=tmp_path)
    assert metrics[0].exists()
    assert all(p.stat().st_mtime_ns == before[p] for p in metrics[1:])  # untouched
    run_experiment(config, out_root=tmp_path, force=True)
    assert all(p.stat().st_mtime_ns != before[p] for p in metrics[1:])


def test_save_policies_final_and_none(tmp_path):
    data = copy.deepcopy(_dummy())
    data["seeds"] = [0]
    data["save_policies"] = "final"
    exp = run_experiment(parse_config(data), out_root=tmp_path / "final")
    run = json.loads((exp / "index.json").read_text())["runs"][0]
    assert sorted(torch.load(exp / run["files"]["policies"], weights_only=False)) == [10]
    data["save_policies"] = "none"
    exp = run_experiment(parse_config(data), out_root=tmp_path / "none")
    run = json.loads((exp / "index.json").read_text())["runs"][0]
    assert run["files"]["policies"] is None and not list(exp.rglob("seed-*.policies.pt"))


def test_cli_list():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    listed = subprocess.run([sys.executable, "-m", "capsplit.harness", "list"], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=60)
    assert listed.returncode == 0 and "tiny_prompt" in listed.stdout and "none" in listed.stdout
