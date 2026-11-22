"""The data -> base -> train -> measure stages on a tiny random Qwen2 (CPU, temporary folders)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from capsplit.harness.lm import cli
from capsplit.harness.lm import stages as st
from capsplit.harness.lm.config import LMConfig, parse_config
from capsplit.harness.lm.policy import LocalLMPolicy

from .conftest import DEMO_CONFIG

ROOT = Path(__file__).resolve().parents[2]


def tiny_raw() -> dict:
    raw = json.loads(DEMO_CONFIG.read_text())
    raw["name"] = "tiny"
    raw["model"]["device"] = "cpu"
    raw["memory"] = {"min_free_gb": 0.0, "max_process_gb": 1000.0}
    raw["runtime"] = {"nice": 0, "threads": 2}
    raw["task"]["generators"] = [{"type": "arithmetic", "params": {"digits": [2], "ops": ["+", "*"]},
                                  "n_train": 3, "n_heldout": 2}]
    raw["sampling"] = {"n_classify": 4, "n_measure": 4, "n_trained": 3, "temperatures": [0.7, 1.0],
                       "n_temperature": 2, "reference_temperature": 1.0, "batch_size": 3}
    raw["measure"] = {"n": 4, "alpha": 0.05, "n_boot": 20, "boot_seed": 0}
    raw["arms"] = [{"name": "none", "trainer": "lm_none", "params": {}},
                   {"name": "rl", "trainer": "lm_rl", "params": {"group": 2, "problems_per_update": 2, "micro_batch": 2}},
                   {"name": "sft", "trainer": "lm_sft",
                    "params": {"batch": 2, "target": ["reachable", "unreachable", "ambiguous"]}}]
    raw["budget"] = 2
    raw["checkpoints"] = [0, 1, 2]
    raw["seeds"] = [0]
    return raw


@pytest.fixture
def config() -> LMConfig:
    return parse_config(tiny_raw())


@pytest.fixture
def factory(tiny_model_factory, tokenizer, config):
    calls = []

    def build(seed: int) -> LocalLMPolicy:
        calls.append(seed)
        return LocalLMPolicy.from_loaded(tiny_model_factory(), tokenizer, config, seed=seed)

    build.calls = calls  # type: ignore[attr-defined]
    return build


def _all_stages(config, out, factory, all_checkpoints=False):
    st.run_data(config, out)
    st.run_base(config, out, factory=factory)
    st.run_train(config, out, factory=factory)
    st.run_measure(config, out, factory=factory, all_checkpoints=all_checkpoints)


def _snapshot(out: Path) -> dict[str, str]:
    """Contents of every output except manifests (they hold commit/time-free provenance, but skip anyway)."""
    return {str(p.relative_to(out)): p.read_text() if p.suffix != ".safetensors" else p.read_bytes().hex()
            for p in sorted(out.rglob("*")) if p.is_file() and "manifest" not in p.name}


def test_end_to_end(config, factory, tmp_path):
    out = tmp_path / "tiny"
    _all_stages(config, out, factory, all_checkpoints=True)
    assert factory.calls == [0, 0, 0]  # one model load each in base, train (reused across runs) and measure
    problems = list(st.read_jsonl(out / "data" / "problems.jsonl"))
    assert len(problems) == 5
    summary = json.loads((out / "data" / "summary.json").read_text())
    assert summary["by_split"] == {"heldout": 2, "train": 3}
    for set_name, n in (("A", 4), ("B", 4), ("null", 4)):
        assert len(st.read_samples(st.sample_path(out, "base", set_name, 1.0), n)) == 5 * n
    for t in (0.7, 1.0):
        assert len(st.read_samples(st.sample_path(out, "base", "C", t), 2)) == 10
    classes = json.loads((out / "base" / "classes.json").read_text())
    assert set(classes["classes"]) == {str(i) for i in range(5)}
    base_summary = json.loads((out / "base" / "summary.json").read_text())
    assert sum(base_summary["classes"].values()) == 5
    for arm in ("none", "rl", "sft"):
        for step in (0, 1, 2):
            assert st.adapter_path(out, arm, 0, step).exists()
        assert (st.run_dir(out, arm, 0) / "training.json").exists()
        assert json.loads((st.run_dir(out, arm, 0) / "timing.json").read_text())["wall_seconds"] >= 0
    rl_log = json.loads((st.run_dir(out, "rl", 0) / "training.json").read_text())
    train_ids = {str(p["id"]) for p in problems if p["split"] == "train"}
    assert set(rl_log["prompts"]) <= train_ids
    results = json.loads((out / "measure" / "results.json").read_text())
    keys = {(r["arm"], r["step"], r["group"]) for r in results["records"]}
    assert ("rl", 2, "all") in keys and ("sft", 0, "heldout") in keys and ("none", 1, "train") in keys
    assert all(r["n_prompts"] >= 2 for r in results["records"])
    assert "train:d2_add" not in {r["group"] for r in results["records"]} or \
        sum(1 for p in problems if p["split"] == "train" and p["difficulty"] == "d2_add") >= 2
    nulls = json.loads((out / "measure" / "null_check.json").read_text())
    assert {r["group"] for r in nulls["records"]} == {"all", "train", "heldout"}
    curves = json.loads((out / "measure" / "budget_curve.json").read_text())
    rl_curve = next(r for r in curves["records"] if r["arm"] == "rl" and r["step"] == 2 and r["group"] == "all")
    assert rl_curve["n_train"]["value"] == pytest.approx(2 * 2 * 2 / 3)  # 2 updates x 2 problems x group 2 over 3
    none_curve = next(r for r in curves["records"] if r["arm"] == "none")
    assert none_curve["n_train"]["value"] is None
    # lm_none's step-0 and final adapters are the untrained init: its eval samples equal step 0's
    assert (out / "samples" / "none" / "seed0" / "step2" / "eval_T1.0.jsonl").exists()


def test_rerun_skips_and_matches(config, factory, tmp_path):
    out = tmp_path / "tiny"
    _all_stages(config, out, factory)
    first = _snapshot(out)
    calls = len(factory.calls)
    st.run_base(config, out, factory=factory)
    st.run_train(config, out, factory=factory)
    st.run_measure(config, out, factory=factory)
    assert len(factory.calls) == calls  # nothing loaded: every sample and run was already done
    after = _snapshot(out)
    after_timing = {k: v for k, v in after.items() if "timing" not in k}
    assert after_timing == {k: v for k, v in first.items() if "timing" not in k}


def test_gap_fill_is_identical(config, factory, tmp_path):
    full = tmp_path / "full"
    st.run_data(config, full)
    st.run_base(config, full, factory=factory)
    cut = tmp_path / "cut"
    st.run_data(config, cut)
    st.run_base(config, cut, factory=factory)
    path = st.sample_path(cut, "base", "A", 1.0)
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join(lines[:5] + lines[11:]) + '{"problem": 0, "ind')  # plus a truncated last line
    c_path = st.sample_path(cut, "base", "C", 0.7)
    c_path.unlink()
    st.run_base(config, cut, factory=factory)
    rows = [json.loads(x) for x in path.read_text().splitlines()]  # every line parses: the torn tail was dropped
    assert len(rows) == len(lines)

    def keyed(p):
        return {k: (v["text"], v["correct"], v["tokens"]) for k, v in st.read_samples(p, 100).items()}

    assert keyed(path) == keyed(st.sample_path(full, "base", "A", 1.0))
    assert keyed(c_path) == keyed(st.sample_path(full, "base", "C", 0.7))
    assert (cut / "base" / "classes.json").read_text() == (full / "base" / "classes.json").read_text()


def test_model_reuse_matches_fresh_load(config, factory, tmp_path):
    """reseed_policy (unload + new adapter) gives the same adapter init as a freshly loaded model."""
    policy = factory(3)
    reseeded = st.reseed_policy(policy, 7)
    fresh = factory(7)
    a, b = reseeded.state_dict(), fresh.state_dict()
    assert set(a) == set(b) and all((a[k] == b[k]).all() for k in a)


def test_interrupted_training_restarts(config, factory, tmp_path):
    out = tmp_path / "tiny"
    st.run_data(config, out)
    st.run_base(config, out, factory=factory)
    st.run_train(config, out, factory=factory)
    reference = st.adapter_path(out, "rl", 0, 2).read_bytes()
    (st.run_dir(out, "rl", 0) / "training.json").unlink()  # as if killed before the end
    st.adapter_path(out, "rl", 0, 2).unlink()
    st.run_train(config, out, factory=factory)
    assert st.adapter_path(out, "rl", 0, 2).read_bytes() == reference


def test_chunk_seed_depends_on_every_key():
    base = st.chunk_seed(0, "base", "A", 1.0, 3)
    assert base == st.chunk_seed(0, "base", "A", 1.0, 3)
    others = [st.chunk_seed(1, "base", "A", 1.0, 3), st.chunk_seed(0, "rl/seed0/step2", "A", 1.0, 3),
              st.chunk_seed(0, "base", "B", 1.0, 3), st.chunk_seed(0, "base", "A", 0.7, 3),
              st.chunk_seed(0, "base", "A", 1.0, 4)]
    assert base not in others and len(set(others)) == len(others)


def test_measure_groups_skip_small():
    rows = [{"id": 0, "split": "train", "difficulty": "x"}, {"id": 1, "split": "train", "difficulty": "x"},
            {"id": 2, "split": "heldout", "difficulty": "y"}, {"id": 3, "split": "heldout", "difficulty": "x"}]
    groups = st.measure_groups(rows)
    assert groups["all"] == [0, 1, 2, 3] and groups["all:x"] == [0, 1, 3] and groups["train:x"] == [0, 1]
    assert "all:y" not in groups and "heldout:x" not in groups


# --------------------------------------------------------------------------- CLI


@pytest.fixture
def cli_config(tmp_path, factory, monkeypatch):
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps(tiny_raw()))
    monkeypatch.setattr(st, "default_policy_factory", lambda config, device, use_fallback=False: factory)
    return path


def test_cli_stage_messages(cli_config, tmp_path, capsys):
    out = tmp_path / "results"
    for stage, following in (("data", "base"), ("base", "train"), ("train", "measure"), ("measure", "report")):
        assert cli.main([stage, str(cli_config), "--out", str(out)]) == 0
        text = capsys.readouterr().out
        assert "wrote " in text
        assert f"next: python -m capsplit.harness.lm {following} {cli_config} --out {out}" in text
    assert (out / "tiny" / "measure" / "results.json").exists()


def test_cli_errors_exit_2(cli_config, tmp_path, capsys):
    out = tmp_path / "results"
    assert cli.main(["base", str(cli_config), "--out", str(out)]) == 2
    assert "run the `data` stage first" in capsys.readouterr().err
    assert cli.main(["data", str(cli_config), "--out", str(out)]) == 0
    assert cli.main(["measure", str(cli_config), "--out", str(out)]) == 2
    assert "first" in capsys.readouterr().err
    bad = tmp_path / "bad.json"
    raw = tiny_raw()
    raw["arms"][1]["params"]["bogus"] = 1
    bad.write_text(json.dumps(raw))
    st.run_data(parse_config(raw), out / "tiny")
    assert cli.main(["train", str(bad), "--out", str(out)]) == 2
    assert "bogus" in capsys.readouterr().err
    assert cli.main(["data", str(tmp_path / "missing.json"), "--out", str(out)]) == 2


def test_cli_report_missing_module(cli_config, tmp_path, capsys):
    try:
        import importlib

        importlib.import_module("capsplit.harness.lm.report")
    except ImportError:
        assert cli.main(["report", str(cli_config), "--out", str(tmp_path)]) == 2
        assert "report" in capsys.readouterr().err
    else:
        pytest.skip("report module exists")


def test_data_conflict(config, tmp_path):
    out = tmp_path / "tiny"
    st.run_data(config, out)
    st.run_data(config, out)  # same config: fine
    raw = tiny_raw()
    raw["task"]["seed"] = 5
    with pytest.raises(st.StageError, match="differs"):
        st.run_data(parse_config(raw), out)


@pytest.mark.parametrize("name", ["lm_demo.json", "lm_main.json"])
def test_shipped_configs_parse(name):
    from capsplit.harness.lm.config import load_config
    from capsplit.harness.lm.tasks import build_problems
    from capsplit.harness.registry import TRAINERS

    st._plugins()
    config = load_config(ROOT / "configs" / name)
    assert build_problems(config.task)
    for arm in config.arms:
        TRAINERS.create(arm.trainer, **dict(arm.params))


def test_sft_without_targets_fails_before_training(config, factory, tmp_path):
    out = tmp_path / "tiny"
    st.run_data(config, out)
    st.run_base(config, out, factory=factory)
    raw = tiny_raw()
    raw["arms"][2]["params"]["target"] = "reachable"  # the tiny model reaches nothing
    with pytest.raises(st.StageError, match="no train problem"):
        st.run_train(parse_config(raw), out, factory=factory)
    assert not (out / "train").exists()
