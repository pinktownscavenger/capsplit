"""Separately seeded streams (config keys base_seed / train_seed) and invariance of old configs."""
import copy
import hashlib
import json
import math
import pathlib

import pytest

torch = pytest.importorskip("torch")

from capsplit.harness.config import ConfigError, load_config, parse_config
from capsplit.harness.runner import run_experiment

ROOT = pathlib.Path(__file__).resolve().parents[1]
DUMMY = ROOT / "configs" / "dummy.json"

# A small config whose cells were run with the code from before base_seed/train_seed and
# training logs existed; the hashes below are of those pre-change outputs.
ARMS = {
    "name": "arms",
    "task": {"type": "tiny_prompt", "params": {"vocab": 3}},
    "base": {"type": "uniform_tabular", "params": {"init_scale": 0.5}},
    "backend": {"type": "j_only", "params": {}},
    "arms": [
        {"name": "rloo", "trainer": "rloo", "params": {"group": 4, "lr": 0.1}},
        {"name": "hints", "trainer": "hints", "params": {"hint_len": 1, "group": 4, "lr": 0.1}},
        {"name": "sftrl", "trainer": "sft_then_rl", "params": {"rl": {"group": 4, "lr": 0.1}}},
    ],
    "seeds": [0],
    "budget": 10,
    "checkpoints": [0, 5, 10],
    "measure": {"ns": [16], "cs": [1], "group_by": ["difficulty", "split"]},
    "save_policies": "none",
}

GOLDEN_DUMMY = {  # files with no computed floats: exact bytes
    "index.json": "7ce61e218d9097e164aa59ec980e736570171e7b80a427aa95f5cac6e772fd44",
    "bases/2bbb21621549/info.json": "fd34a3f9328f4ab533e0f10ffca97dc7727894869d01233e7324791590f63009",
    "bases/c71fbc292403/info.json": "35107612cfb4b6a5be02e35d073ceaf52ae483d90cc223b2c94c502fc8fb0cc0",
    "config.json": "538c066442ed7876a516426b4471b04f75fbf62fad7acfe0f5d6dbc972aaf186",
}
# Metrics files hold computed floats, whose last bits differ between platforms (macOS vs
# Linux), so they are compared by fingerprint: (sha256 of the JSON with every finite float
# replaced by a placeholder, float count, sum, position-weighted sum). Values were produced
# by the pre-change code (main before base_seed/train_seed) on Linux. Sums are compared at
# rel 1e-6: CI runners differ from a local Linux machine by about 1e-9 after training.
GOLDEN_DUMMY_METRICS = {
    "40a6eded98d4/seed-0.metrics.json": ("3d0e243180b5ed954ae89ec408e90794498e87db45a25a7ed2dfb2c42b9724fa", 972, 208.48209440812434, 101266.74861237383),
    "40a6eded98d4/seed-1.metrics.json": ("26900dccbf1ab014b10382772b077a4ed86da45f8223e7f1618faebdc26525a4", 972, 219.35484065110614, 106978.46957570134),
    "7328c2f7f94f/seed-0.metrics.json": ("906f0f971a9282f4b990fb9e140ff7cd6e28edbe3d2a861674dbedc4ca403f9c", 972, 208.48209440812434, 101266.74861237383),
    "7328c2f7f94f/seed-1.metrics.json": ("5a08110fb63f84e755cda04f6277a45c75fca6826daf05dfab82eb049d5f6a3c", 972, 219.35484065110614, 106978.46957570134),
}
GOLDEN_ARMS = {  # seed-0 metrics of the rloo, hints and sft_then_rl cells; fingerprints as above
    "9a7b1a4e6bb3/seed-0.metrics.json": ("a5e5bbf8ed2f67117500414bd52242444185f7fcd50f7a4d39a868efa6cc8f9b", 162, 30.671949400823262, 2450.6151115711777),
    "853cf62f4f96/seed-0.metrics.json": ("aa537287e0e2aa177e80d2ee4ec25d5ae5eb1d02070d5d01dd9fe6b363660442", 162, 31.269622703894786, 2528.5697265439926),
    "277a8e79a268/seed-0.metrics.json": ("cc6920c55c80ee192c3485630e43e1a083e999f4a0064e2e77d3f516ef111830", 162, 30.61764196593208, 2449.643168252687),
}


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(path: pathlib.Path) -> tuple[str, int, float, float]:
    floats: list[float] = []

    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items()}
        if isinstance(x, list):
            return [strip(v) for v in x]
        if isinstance(x, float):
            if math.isfinite(x):
                floats.append(x)
                return "<float>"
            return repr(x)
        return x

    skeleton = json.dumps(strip(json.loads(path.read_text())), sort_keys=True)
    weighted = math.fsum(v * (i + 1) for i, v in enumerate(floats))
    return hashlib.sha256(skeleton.encode()).hexdigest(), len(floats), math.fsum(floats), weighted


def _assert_matches(path: pathlib.Path, golden: tuple[str, int, float, float]) -> None:
    skeleton, count, total, weighted = _fingerprint(path)
    assert (skeleton, count) == golden[:2], path.name
    assert total == pytest.approx(golden[2], rel=1e-6), path.name
    assert weighted == pytest.approx(golden[3], rel=1e-6), path.name


def _rloo_config(**extra) -> dict:
    data = copy.deepcopy(ARMS)
    data["name"] = "streams"
    data["arms"] = [data["arms"][0]]
    data["seeds"] = [0, 1]
    data.update(extra)
    return data


def _metrics(exp: pathlib.Path, seed: int) -> dict:
    index = json.loads((exp / "index.json").read_text())
    (run,) = [r for r in index["runs"] if r["seed"] == seed]
    return json.loads((exp / run["files"]["metrics"]).read_text())


def _run(exp: pathlib.Path, seed: int) -> dict:
    index = json.loads((exp / "index.json").read_text())
    (run,) = [r for r in index["runs"] if r["seed"] == seed]
    return run


# ---------------------------------------------------------------------- invariance


def test_dummy_outputs_match_the_pre_change_code(tmp_path):
    exp = run_experiment(DUMMY, tmp_path)
    for rel, digest in GOLDEN_DUMMY.items():
        assert _sha(exp / rel) == digest, rel
    for rel, golden in GOLDEN_DUMMY_METRICS.items():
        _assert_matches(exp / rel, golden)
    assert not list(exp.glob("*/*.training.json"))  # trainer "none" provides no log


def test_trained_metrics_match_the_pre_change_code(tmp_path):
    exp = run_experiment(parse_config(ARMS), tmp_path)
    for rel, golden in GOLDEN_ARMS.items():
        _assert_matches(exp / rel, golden)
    index = json.loads((exp / "index.json").read_text())
    for run in index["runs"]:
        assert set(run) == {"factors", "arm", "trainer", "seed", "run_dir", "base", "files"}
        assert run["files"]["training"] == f"{run['run_dir']}/seed-0.training.json"


def test_cell_config_unchanged_without_the_new_keys():
    config = load_config(DUMMY)
    assert config.base_seed is None and config.train_seed is None and not config.separate_seeds
    cell = config.cell_config(config.factor_combos()[0], config.arms[0])
    assert "base_seed" not in cell and "train_seed" not in cell
    fixed = parse_config({**json.loads(DUMMY.read_text()), "base_seed": 3})
    assert fixed.cell_config(fixed.factor_combos()[0], fixed.arms[0])["base_seed"] == 3


def test_fixing_both_streams_to_the_run_seed_reproduces_the_coupled_run(tmp_path):
    plain = run_experiment(parse_config(_rloo_config(seeds=[1])), tmp_path / "a")
    split = run_experiment(parse_config(_rloo_config(seeds=[1], base_seed=1, train_seed=1)), tmp_path / "b")
    a, b = _metrics(plain, 1), _metrics(split, 1)
    assert a["records"] == b["records"] and a["base"] == b["base"]
    assert "base_seed" not in a
    assert (b["base_seed"], b["train_seed"], b["measure_seed"]) == (1, 1, None)


@pytest.mark.parametrize("value", [-1, 1.5, True, "0", None])
def test_bad_fixed_seeds_raise(value):
    for key in ("base_seed", "train_seed"):
        with pytest.raises(ConfigError, match=key):
            parse_config({**json.loads(DUMMY.read_text()), key: value})


# ---------------------------------------------------------------------- streams


def test_fixed_base_varying_train_shares_the_base_and_changes_training(tmp_path):
    exp = run_experiment(parse_config(_rloo_config(base_seed=7)), tmp_path)
    r0, r1 = _run(exp, 0), _run(exp, 1)
    assert r0["base"] == r1["base"]
    assert len(list((exp / "bases").iterdir())) == 1  # built once, reused
    info = json.loads((exp / r0["base"] / "info.json").read_text())
    assert info["seed"] == 7
    assert (r0["base_seed"], r0["train_seed"], r1["base_seed"], r1["train_seed"]) == (7, 0, 7, 1)
    assert r0["measure_seed"] is None
    m0, m1 = _metrics(exp, 0), _metrics(exp, 1)
    step0 = [r for r in m0["records"] if r["step"] == 0]
    assert step0 == [r for r in m1["records"] if r["step"] == 0]  # identical base
    assert [r for r in m0["records"] if r["step"] == 10] != [r for r in m1["records"] if r["step"] == 10]
    t0 = json.loads((exp / r0["files"]["training"]).read_text())
    t1 = json.loads((exp / r1["files"]["training"]).read_text())
    assert t0["prompts"] != t1["prompts"]
    assert (t0["base_seed"], t0["train_seed"]) == (7, 0)


def test_fixed_train_varying_base_and_same_base_gives_same_training(tmp_path):
    exp = run_experiment(parse_config(_rloo_config(train_seed=3)), tmp_path / "a")
    r0, r1 = _run(exp, 0), _run(exp, 1)
    assert r0["base"] != r1["base"]
    assert (exp / r0["base"] / "policy.pt").read_bytes() != (exp / r1["base"] / "policy.pt").read_bytes()
    assert _metrics(exp, 0)["records"] != _metrics(exp, 1)["records"]
    assert (r0["train_seed"], r1["train_seed"], r0["base_seed"], r1["base_seed"]) == (3, 3, 0, 1)

    # A different run seed with base 1 and training 3 reproduces run seed 1 exactly.
    other = run_experiment(parse_config(_rloo_config(seeds=[5], base_seed=1, train_seed=3)), tmp_path / "b")
    r5 = _run(other, 5)
    assert r5["base"] == r1["base"]
    assert (other / r5["base"] / "info.json").read_text() == (exp / r1["base"] / "info.json").read_text()
    assert _metrics(other, 5)["records"] == _metrics(exp, 1)["records"]
    t5 = json.loads((other / r5["files"]["training"]).read_text())
    t1 = json.loads((exp / r1["files"]["training"]).read_text())
    assert t5["prompts"] == t1["prompts"] and t5["totals"] == t1["totals"]


def test_measure_seed_comes_from_backend_params(tmp_path):
    data = _rloo_config(seeds=[0], train_seed=0)
    data["backend"] = {"type": "j_only", "params": {"seed": 11}}
    exp = run_experiment(parse_config(data), tmp_path)
    assert _run(exp, 0)["measure_seed"] == 11 and _metrics(exp, 0)["measure_seed"] == 11
