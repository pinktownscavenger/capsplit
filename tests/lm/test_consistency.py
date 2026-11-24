"""Stale-output guard (manifest fingerprints), zero-update SFT phases, config validation and CLI
error handling, on the tiny random Qwen2 (CPU, temporary folders)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch

from capsplit.harness.lm import cli
from capsplit.harness.lm import stages as st
from capsplit.harness.lm.config import LMConfig, LMConfigError, parse_config
from capsplit.harness.lm.device import DeviceError, pick_device
from capsplit.harness.lm.policy import LocalLMPolicy

from .test_stages import tiny_raw

ADVICE = "use a new experiment name, or delete"


@pytest.fixture
def config() -> LMConfig:
    return parse_config(tiny_raw())


@pytest.fixture
def factory(tiny_model_factory, tokenizer, config):
    def build(seed: int) -> LocalLMPolicy:
        return LocalLMPolicy.from_loaded(tiny_model_factory(), tokenizer, config, seed=seed)

    return build


@pytest.fixture
def patched(monkeypatch, factory):
    monkeypatch.setattr(st, "default_policy_factory", lambda config, device, use_fallback=False: factory)


def write_config(path: Path, raw: dict) -> str:
    path.write_text(json.dumps(raw))
    return str(path)


def run(stage: str, config_path: str, out: Path, *flags: str) -> int:
    return cli.main([stage, config_path, "--out", str(out), *flags])


def strip_fingerprints(root: Path) -> int:
    """Rewrite every manifest as the older code wrote it: no fingerprint (and no declared model for
    base, which older code recorded from the loaded policy, so keep `model` as written)."""
    n = 0
    for path in root.rglob("manifest.*.json"):
        data = json.loads(path.read_text())
        if data.pop("fingerprint", None) is not None:
            data.pop("status", None)
            path.write_text(json.dumps(data))
            n += 1
    return n


# --------------------------------------------------------------------------- fingerprints


def test_fingerprint_sections_per_stage(config):
    cfg = st.plain_config(config)
    model = st.model_choice(config)
    assert set(st.fingerprint(cfg, None, "data")) == {"task"}
    assert set(st.fingerprint(cfg, model, "base")) == {"task", "model", "sampling"}
    train = set(st.fingerprint(cfg, model, "train"))
    assert train == {"task", "model", "sampling", "lora", "budget", "checkpoints", "seeds",
                     "arms.none", "arms.rl", "arms.sft"}
    assert set(st.fingerprint(cfg, model, "measure")) == train | {"measure"}
    fallback = st.fingerprint(cfg, st.model_choice(config, use_fallback=True), "base")
    assert fallback["model"] != st.fingerprint(cfg, model, "base")["model"]
    assert fallback["task"] == st.fingerprint(cfg, model, "base")["task"]
    raw = tiny_raw()
    raw["arms"][1]["params"]["group"] = 3
    other = st.fingerprint(st.plain_config(parse_config(raw)), model, "train")
    differing = {k for k, v in other.items() if st.fingerprint(cfg, model, "train")[k] != v}
    assert differing == {"arms.rl"}


def test_prompt_template_change_exits_2(patched, tmp_path, capsys):
    out = tmp_path / "results"
    first = write_config(tmp_path / "a.json", tiny_raw())
    assert run("data", first, out) == 0
    assert run("base", first, out) == 0
    capsys.readouterr()
    raw = tiny_raw()
    raw["task"]["prompt_template"] = "What is {question}? Answer with a number."
    second = write_config(tmp_path / "b.json", raw)
    for stage in ("base", "train", "data"):
        assert run(stage, second, out) == 2
        err = capsys.readouterr().err
        assert "task" in err and ADVICE in err and "to recompute" in err, err
    assert "samples/base" not in err  # data's own outputs: delete data only


def test_base_rejects_changed_sampling_mid_run(config, factory, tmp_path):
    out = tmp_path / "tiny"
    st.run_data(config, out)
    st.run_base(config, out, factory=factory)
    raw = tiny_raw()
    raw["sampling"]["n_classify"] = 6
    with pytest.raises(st.StageError, match=f"sampling.*{ADVICE}") as info:
        st.run_base(parse_config(raw), out, factory=factory)
    assert str(out / "samples" / "base") in str(info.value)


def test_arm_param_change_exits_2(patched, tmp_path, capsys):
    out = tmp_path / "results"
    first = write_config(tmp_path / "a.json", tiny_raw())
    for stage in ("data", "base", "train"):
        assert run(stage, first, out) == 0
    capsys.readouterr()
    raw = tiny_raw()
    raw["arms"][1]["params"]["group"] = 3
    second = write_config(tmp_path / "b.json", raw)
    for stage in ("train", "measure"):
        assert run(stage, second, out) == 2
        err = capsys.readouterr().err
        assert "arms.rl" in err and ADVICE in err and str(out / "tiny" / "train" / "rl" / "seed0") in err, err
        assert "arms.sft" not in err and "arms.none" not in err


def test_fallback_change_exits_2(patched, tmp_path, capsys):
    out = tmp_path / "results"
    path = write_config(tmp_path / "a.json", tiny_raw())
    assert run("data", path, out) == 0
    assert run("base", path, out) == 0
    capsys.readouterr()
    assert run("train", path, out, "--fallback") == 2
    err = capsys.readouterr().err
    assert "model" in err and "--fallback" in err and ADVICE in err
    assert run("base", path, out, "--fallback") == 2
    assert ADVICE in capsys.readouterr().err
    assert run("train", path, out) == 0  # same model as base: fine


def test_unchanged_config_reuses(patched, tmp_path, capsys, monkeypatch):
    out = tmp_path / "results"
    path = write_config(tmp_path / "a.json", tiny_raw())
    for stage in ("data", "base", "train", "measure"):
        assert run(stage, path, out) == 0
    def no_load(config, device, use_fallback=False):
        def build(seed: int) -> None:
            raise AssertionError("nothing should be loaded")
        return build

    monkeypatch.setattr(st, "default_policy_factory", no_load)
    for stage in ("data", "base", "train", "measure"):
        assert run(stage, path, out) == 0, capsys.readouterr().err
    assert "has no manifest" not in capsys.readouterr().err


def test_old_style_manifests_accepted_when_matching(config, factory, tmp_path, capsys):
    out = tmp_path / "tiny"
    st.run_data(config, out)
    st.run_base(config, out, factory=factory)
    st.run_train(config, out, factory=factory)
    assert strip_fingerprints(out) > 0
    base = json.loads((out / "base" / "manifest.base.json").read_text())
    assert "fingerprint" not in base and "config" in base
    st.run_data(config, out)
    st.run_base(config, out, factory=factory)
    strip_fingerprints(out)
    st.run_train(config, out, factory=factory)
    strip_fingerprints(out)
    st.run_measure(config, out, factory=factory)
    assert "has no manifest" not in capsys.readouterr().err
    raw = tiny_raw()
    raw["task"]["prompt_template"] = "What is {question}?"
    strip_fingerprints(out)
    with pytest.raises(st.StageError, match=ADVICE):
        st.run_base(parse_config(raw), out, factory=factory)


def test_old_style_manifest_without_model_assumes_primary(config, factory, tmp_path):
    out = tmp_path / "tiny"
    st.run_data(config, out)
    st.run_base(config, out, factory=factory)
    path = out / "base" / "manifest.base.json"
    data = json.loads(path.read_text())
    data.pop("fingerprint")
    data["model"] = None
    path.write_text(json.dumps(data))
    st.run_base(config, out, factory=factory)  # primary model: accepted
    data["model"] = None
    data.pop("fingerprint", None)
    path.write_text(json.dumps(data))
    with pytest.raises(st.StageError, match="model"):
        st.run_base(config, out, factory=factory, use_fallback=True)


def test_stale_eval_samples_rejected(config, factory, tmp_path):
    out = tmp_path / "tiny"
    st.run_data(config, out)
    st.run_base(config, out, factory=factory)
    st.run_train(config, out, factory=factory)
    st.run_measure(config, out, factory=factory)
    raw = tiny_raw()
    raw["arms"][1]["params"]["group"] = 3
    changed = parse_config(raw)
    shutil.rmtree(st.run_dir(out, "rl", 0))  # as the train error says
    st.run_train(changed, out, factory=factory)
    with pytest.raises(st.StageError, match="arms.rl") as info:
        st.run_measure(changed, out, factory=factory)
    assert str(st.eval_dir(out, "rl", 0)) in str(info.value)
    shutil.rmtree(st.eval_dir(out, "rl", 0))
    st.run_measure(changed, out, factory=factory)


def test_measure_section_change_rejected(config, factory, tmp_path):
    out = tmp_path / "tiny"
    st.run_data(config, out)
    st.run_base(config, out, factory=factory)
    st.run_train(config, out, factory=factory)
    st.run_measure(config, out, factory=factory)
    raw = tiny_raw()
    raw["measure"]["n_boot"] = 30
    with pytest.raises(st.StageError, match="in: measure;") as info:
        st.run_measure(parse_config(raw), out, factory=factory)
    assert str(out / "measure") in str(info.value)
    shutil.rmtree(out / "measure")
    st.run_measure(parse_config(raw), out, factory=factory)


def test_unverifiable_outputs_warn(config, factory, tmp_path, capsys):
    out = tmp_path / "tiny"
    st.run_data(config, out)
    st.run_base(config, out, factory=factory)
    (out / "base" / "manifest.base.json").unlink()
    st.run_base(config, out, factory=factory)
    assert "has no manifest" in capsys.readouterr().err
    assert "fingerprint" in json.loads((out / "base" / "manifest.base.json").read_text())


# --------------------------------------------------------------------------- SFT phase with 0 updates


@pytest.mark.parametrize("fraction", [0.0, 0.2])  # 0.2 x budget 2 rounds to 0 SFT updates
def test_sft_then_rl_without_sft_updates_needs_no_targets(fraction, config, factory, tmp_path):
    out = tmp_path / "tiny"
    st.run_data(config, out)
    st.run_base(config, out, factory=factory)
    raw = tiny_raw()
    raw["arms"] = [{"name": "srl", "trainer": "lm_sft_then_rl",
                    "params": {"sft_fraction": fraction, "sft": {"target": "reachable", "batch": 2},
                               "rl": {"group": 2, "problems_per_update": 2, "micro_batch": 2}}}]
    st.run_train(parse_config(raw), out, factory=factory)
    assert st.adapter_path(out, "srl", 0, 2).exists()
    raw["arms"][0]["params"]["sft_fraction"] = 1.0
    raw["name"] = "other"
    with pytest.raises(st.StageError, match="no train problem"):
        st.run_train(parse_config(raw), out, factory=factory)


# --------------------------------------------------------------------------- config validation


@pytest.mark.parametrize("section,key,value", [
    ("sampling", "n_trained", 0), ("sampling", "n_temperature", -1), ("sampling", "n_classify", 1.5),
    ("sampling", "n_measure", True), ("sampling", "batch_size", 0), ("sampling", "top_p", 0.0),
    ("sampling", "top_p", 1.5), ("sampling", "top_p", "1"), ("measure", "n_splits", 0), ("measure", "n_boot", -5),
])
def test_config_rejects_bad_values(section, key, value):
    raw = tiny_raw()
    raw[section][key] = value
    with pytest.raises(LMConfigError, match=f"{section}.{key}"):
        parse_config(raw)


@pytest.mark.parametrize("temps", [[0.0, 1.0], [-0.5, 1.0], [], ["hot", 1.0]])
def test_config_rejects_bad_temperatures(temps):
    raw = tiny_raw()
    raw["sampling"]["temperatures"] = temps
    with pytest.raises(LMConfigError, match="temperatures"):
        parse_config(raw)


def test_config_accepts_edge_values():
    raw = tiny_raw()
    raw["sampling"]["top_p"] = 1
    raw["sampling"]["n_trained"] = 1
    raw["measure"]["n_splits"] = 1
    parse_config(raw)


# --------------------------------------------------------------------------- CLI errors


def test_pick_device_raises_device_error(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(DeviceError, match="model.device"):
        pick_device("mps")
    assert issubclass(DeviceError, RuntimeError)
    assert pick_device("cpu").type == "cpu"


def test_cli_device_error_exits_2(patched, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    raw = tiny_raw()
    raw["model"]["device"] = "mps"
    path = write_config(tmp_path / "a.json", raw)
    out = tmp_path / "results"
    assert run("data", path, out) == 0
    assert run("base", path, out) == 2
    assert "model.device" in capsys.readouterr().err


def test_cli_model_load_oserror_exits_2(tmp_path, monkeypatch, capsys):
    def missing(config, device, use_fallback=False):
        def build(seed: int) -> None:
            raise OSError("Cannot find the requested files in the local cache")
        return build

    monkeypatch.setattr(st, "default_policy_factory", missing)
    path = write_config(tmp_path / "a.json", tiny_raw())
    out = tmp_path / "results"
    assert run("data", path, out) == 0
    assert run("base", path, out) == 2
    err = capsys.readouterr().err
    assert "run the fetch stage first" in err and "network" in err and "Traceback" not in err
