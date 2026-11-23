"""Function-space (KL) layer: estimator, bound, scoring path, weight descriptors, stage and report.

Model tests use the tiny random Qwen2 from conftest (CPU, seconds). The end-to-end test builds a
synthetic main run with the real data / base / measure stages (as test_report.py does), writes
tiny-model adapters into it, and runs the `kl` and `kl-report` stages on it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from capsplit.harness.lm import kl
from capsplit.harness.lm import stages as st
from capsplit.harness.lm.config import parse_config
from capsplit.harness.lm.model import encode_prompts
from capsplit.harness.lm.policy import LocalLMPolicy

from .conftest import DEMO_CONFIG, make_config
from .test_report import base_dir, default_runs, make, raw_config  # noqa: F401  (base_dir is a fixture)

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- helpers


def perturb(policy: LocalLMPolicy, seed: int, scale: float = 0.5) -> LocalLMPolicy:
    """Give the adapter non-zero lora_B weights (a fresh adapter has B = 0, i.e. no effect)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in policy.model.named_parameters():
            if "lora_B" in name:
                p.copy_(torch.randn(p.shape, generator=g) * scale)
    return policy


def manual_logprob(model: torch.nn.Module, tokenizer, config, question: str, completion: list[int]) -> float:
    """log pi(completion | prompt) from a plain (no peft) model: one unpadded forward, summed by hand."""
    batch = encode_prompts(tokenizer, config.model.system_prompt, [question], torch.device("cpu"))
    ids = torch.cat([batch["input_ids"], torch.tensor([completion[:-1]], dtype=torch.long)], dim=1)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0].float() / config.sampling.reference_temperature
    logp = torch.log_softmax(logits, dim=-1)
    start = batch["input_ids"].shape[1] - 1
    return float(sum(logp[start + t, tok] for t, tok in enumerate(completion)))


# --------------------------------------------------------------------------- math


def test_bernoulli_kl_known_values():
    assert kl.bernoulli_kl(np.array(0.3), np.array(0.3)) == pytest.approx(0.0)
    expected = 0.9 * math.log(0.9 / 0.5) + 0.1 * math.log(0.1 / 0.5)
    assert kl.bernoulli_kl(np.array(0.9), np.array(0.5)) == pytest.approx(expected)
    # 0 ln 0 = 0: moving all mass onto S costs exactly ln(1 / p0)
    assert kl.bernoulli_kl(np.array(1.0), np.array(0.01)) == pytest.approx(math.log(100))
    assert kl.bernoulli_kl(np.array(0.0), np.array(0.25)) == pytest.approx(math.log(1 / 0.75))
    assert np.isinf(kl.bernoulli_kl(np.array(0.5), np.array(0.0)))


def test_smoothing():
    k, n = np.array([0, 5, 10]), np.array([10, 10, 10])
    np.testing.assert_allclose(kl.smooth(k, n, "jeffreys"), (k + 0.5) / (n + 1))
    np.testing.assert_allclose(kl.smooth(k, n, "laplace"), (k + 1) / (n + 2))
    assert np.all((kl.smooth(k, n, "jeffreys") > 0) & (kl.smooth(k, n, "jeffreys") < 1))
    with pytest.raises(ValueError):
        kl.smooth(k, n, "none")


def test_kl_estimate_mean_and_se():
    x = np.array([1.0, -0.5, 2.0, 0.5])
    mean, se = kl.kl_estimate(x)
    assert mean == pytest.approx(0.75)
    assert se == pytest.approx(np.std(x, ddof=1) / 2)
    assert kl.kl_estimate(np.zeros(5)) == (0.0, 0.0)
    mean1, se1 = kl.kl_estimate(np.array([0.7]))
    assert mean1 == pytest.approx(0.7) and math.isnan(se1)


def test_planted_kl_synthetic_policy():
    """Two categorical 'policies' over 6 sequences: the estimator recovers the known KL, and the
    data-processing bound KL >= d(p1 || p0) holds for every success event S."""
    p1 = np.array([0.40, 0.25, 0.15, 0.10, 0.06, 0.04])
    p0 = np.array([0.10, 0.10, 0.20, 0.30, 0.20, 0.10])
    exact = float(np.sum(p1 * np.log(p1 / p0)))
    rng = np.random.default_rng(0)
    draws = rng.choice(len(p1), size=20000, p=p1)
    mean, se = kl.kl_estimate(np.log(p1[draws]) - np.log(p0[draws]))
    assert abs(mean - exact) < 4 * se
    for mask in range(1, 2 ** len(p1) - 1):
        s = np.array([(mask >> i) & 1 for i in range(len(p1))], dtype=bool)
        assert exact >= float(kl.bernoulli_kl(np.array(p1[s].sum()), np.array(p0[s].sum()))) - 1e-12


# --------------------------------------------------------------------------- model sanity


@pytest.fixture
def tiny(tiny_model_factory, tokenizer, lm_config):
    return LocalLMPolicy.from_loaded(tiny_model_factory(), tokenizer, lm_config, seed=0)


def test_adapter_off_equals_plain_base(tiny_model_factory, tokenizer, lm_config, task):
    """pi0 via disable_adapter() gives the plain base model's log-probs, even with a non-zero adapter."""
    policy = perturb(LocalLMPolicy.from_loaded(tiny_model_factory(), tokenizer, lm_config, seed=0), seed=1)
    plain = tiny_model_factory()
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        rollout = policy.sample(task, [0, 1, 2, 3], gen)
    actions = [tuple(c) for c in rollout.completion_ids]
    lp1, lp0 = kl.score_pair(policy, task, [0, 1, 2, 3], actions)
    for i, (pid, comp) in enumerate(zip([0, 1, 2, 3], rollout.completion_ids, strict=True)):
        ref = manual_logprob(plain, tokenizer, lm_config, task.prompts()[pid].tokens[0], list(comp))
        assert lp0[i] == pytest.approx(ref, abs=1e-4)
    assert np.max(np.abs(lp1 - lp0)) > 1e-3  # the perturbed adapter does change the policy


def test_zero_adapter_gives_exact_zero(tiny, task):
    """A fresh adapter (lora_B = 0, as the `none` arm and every step-0 checkpoint) gives KL exactly 0."""
    rows = kl.kl_chunk(tiny, task, 2, 16, torch.Generator().manual_seed(3))
    assert len(rows) == 16
    assert all(r["logp1"] == r["logp0"] for r in rows)
    assert kl.kl_estimate(np.array([r["logp1"] - r["logp0"] for r in rows])) == (0.0, 0.0)


def test_scoring_uses_policy_log_prob_for_both(tiny_model_factory, tokenizer, lm_config, task, monkeypatch):
    """pi1 and pi0 scores both come from policy.log_prob: two calls per chunk, adapter on then off."""
    policy = perturb(LocalLMPolicy.from_loaded(tiny_model_factory(), tokenizer, lm_config, seed=0), seed=2)
    calls: list[tuple[bool, np.ndarray]] = []
    original = policy.log_prob

    def spy(task_, ids, actions, start=0):
        out = original(task_, ids, actions, start)
        disabled = any(getattr(m, "disable_adapters", False) for m in policy.model.modules()
                       if hasattr(m, "lora_A"))
        calls.append((disabled, out.detach().float().cpu().double().numpy()))
        return out

    monkeypatch.setattr(policy, "log_prob", spy)
    rows = kl.kl_chunk(policy, task, 1, 8, torch.Generator().manual_seed(5))
    assert [d for d, _ in calls] == [False, True]
    np.testing.assert_array_equal([r["logp1"] for r in rows], calls[0][1])
    np.testing.assert_array_equal([r["logp0"] for r in rows], calls[1][1])


def test_planted_kl_tiny_model_one_token(tiny_model_factory, tokenizer, task):
    """With 1-token completions the exact KL is a sum over the vocabulary; the sampled estimator
    recovers it, and the bound holds for the event 'first token is among the 50 likeliest under pi1'."""
    raw = json.loads(DEMO_CONFIG.read_text())
    raw["memory"] = {"min_free_gb": 0.0, "max_process_gb": 1000.0}
    raw["model"]["max_new_tokens"] = 1
    config = parse_config(raw)
    policy = perturb(LocalLMPolicy.from_loaded(tiny_model_factory(), tokenizer, config, seed=0), seed=4, scale=1.0)
    pid = 0
    batch = encode_prompts(tokenizer, config.model.system_prompt, [task.prompts()[pid].tokens[0]], torch.device("cpu"))
    with torch.no_grad():
        l1 = torch.log_softmax(policy.model(**batch).logits[0, -1].double(), -1)
        with policy.model.disable_adapter():
            l0 = torch.log_softmax(policy.model(**batch).logits[0, -1].double(), -1)
    exact = float((l1.exp() * (l1 - l0)).sum())
    assert exact > 0.01
    rows = kl.kl_chunk(policy, task, pid, 3000, torch.Generator().manual_seed(11))
    mean, se = kl.kl_estimate(np.array([r["logp1"] - r["logp0"] for r in rows]))
    assert abs(mean - exact) < 4 * se
    top = torch.topk(l1, 50).indices
    p1s, p0s = float(l1.exp()[top].sum()), float(l0.exp()[top].sum())
    assert exact >= float(kl.bernoulli_kl(np.array(p1s), np.array(p0s)))


# --------------------------------------------------------------------------- weights


def test_delta_w_descriptors_match_explicit():
    rng = np.random.default_rng(0)
    scale = 2.0
    a1, b1 = rng.normal(size=(4, 12)), rng.normal(size=(10, 4))
    a2, b2 = rng.normal(size=(4, 12)), rng.normal(size=(10, 4))
    w1, w2 = scale * b1 @ a1, scale * b2 @ a2
    sv = np.linalg.svd(w1, compute_uv=False)[:4]
    np.testing.assert_allclose(kl.delta_w_singular_values(a1, b1, scale), sv, rtol=1e-10)
    assert kl.delta_w_inner(a1, b1, a2, b2, scale) == pytest.approx(float(np.sum(w1 * w2)))
    assert kl.delta_w_inner(a1, b1, a1, b1, scale) == pytest.approx(float(np.sum(w1 * w1)))


def test_participation_ratio():
    assert kl.participation_ratio(np.array([3.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert kl.participation_ratio(np.array([2.0, 2.0, 2.0, 2.0])) == pytest.approx(4.0)
    assert math.isnan(kl.participation_ratio(np.zeros(3)))


def test_lora_modules_parse(tiny):
    mods = kl.lora_modules({k: v.numpy() for k, v in tiny.state_dict().items()})
    assert len(mods) == 2 * 2  # 2 layers x (q_proj, v_proj) in the tiny model
    (layer, module), (a, b) = next(iter(sorted(mods.items())))
    assert layer == 0 and module in ("q_proj", "v_proj") and a.shape[0] == b.shape[1]


# --------------------------------------------------------------------------- bootstrap and config


def test_cluster_bootstrap():
    values = np.array([1.0, 1.0, 3.0, 3.0])
    clusters = np.array([0, 0, 1, 1])
    res = kl.cluster_bootstrap(values, clusters, n_boot=500, rng=np.random.default_rng(0))
    assert res["estimate"] == pytest.approx(2.0)
    assert 1.0 <= res["lo"] <= res["hi"] <= 3.0
    const = kl.cluster_bootstrap(np.full(6, 0.5), np.arange(6), n_boot=100, rng=np.random.default_rng(0))
    assert const["lo"] == const["hi"] == pytest.approx(0.5)
    assert kl.cluster_bootstrap(np.array([]), np.array([]), 10, np.random.default_rng(0))["n"] == 0


def test_kl_config_strict(tmp_path):
    shipped = kl.load_kl_config(ROOT / "configs" / "lm_kl.json")
    assert shipped.m_final >= 1
    raw = json.loads((ROOT / "configs" / "lm_kl.json").read_text())
    raw["surprise"] = 1
    with pytest.raises(kl.KLConfigError, match="surprise"):
        kl.parse_kl_config(raw)
    with pytest.raises(kl.KLConfigError):
        kl.parse_kl_config({**json.loads((ROOT / "configs" / "lm_kl.json").read_text()), "smoothing": "magic"})


def test_jobs_validate_steps():
    config = make_config()
    base = json.loads((ROOT / "configs" / "lm_kl.json").read_text())
    raw = {**base, "arms": ["rl"], "checkpoints": [999]}
    with pytest.raises(kl.KLConfigError, match="999"):
        kl.kl_jobs(config, kl.parse_kl_config(raw))


# --------------------------------------------------------------------------- end to end on a synthetic run


def _write_tiny_adapters(s, factory) -> None:
    """Zero adapters for `none` and every step 0; perturbed ones for trained arms at the final step."""
    for arm in ("none", "rl", "sft"):
        for seed in (0, 1):
            for step in (0, 2):
                policy = factory(0)
                if arm != "none" and step == 2:
                    perturb(policy, seed=10 * seed + len(arm), scale=0.3)
                policy.save_adapter(st.adapter_path(s.out, arm, seed, step))


def test_kl_stage_and_report_on_synth(base_dir, tmp_path, tiny_model_factory, tokenizer):  # noqa: F811
    s = make(base_dir, tmp_path)
    default_runs(s)
    config = parse_config(raw_config())

    def factory(seed: int) -> LocalLMPolicy:
        return LocalLMPolicy.from_loaded(tiny_model_factory(), tokenizer, config, seed=seed)

    _write_tiny_adapters(s, factory)
    kcfg = kl.parse_kl_config({**json.loads((ROOT / "configs" / "lm_kl.json").read_text()),
                               "arms": ["rl", "sft"], "m_final": 4, "checkpoints": [], "m_zero": 2,
                               "n_boot": 50})
    kl_out = tmp_path / "synth_kl"
    written = kl.run_kl(config, kcfg, s.out, kl_out, device="cpu", factory=factory)
    assert (kl_out / "manifest.kl.json") in written
    n_rows = {p: len(p.read_text().splitlines()) for p in (kl_out / "samples").rglob("*.jsonl")}
    assert n_rows and all(v > 0 for v in n_rows.values())
    kl.run_kl(config, kcfg, s.out, kl_out, device="cpu", factory=factory)  # resume: nothing new
    assert {p: len(p.read_text().splitlines()) for p in (kl_out / "samples").rglob("*.jsonl")} == n_rows

    path = kl.build_kl_report(s.out, kl_out, kcfg)
    res = json.loads(path.read_text())
    assert res["zero_checks"] and all(z["max_abs_log_ratio"] == 0.0 for z in res["zero_checks"])
    assert {r["arm"] for r in res["k1"]} >= {"rl", "sft", "pooled"}
    assert res["weights"]["adapters"]
    for name in ("report.md", "tables/per_problem.csv", "tables/k1.csv", "tables/k2_groups.csv",
                 "tables/trajectory.csv", "tables/weights_modules.csv", "tables/weights_cosine.csv",
                 "figures/kl_by_group.png", "figures/gain_vs_kl.png", "figures/k1_scatter.png"):
        assert (kl_out / "report" / name).exists(), name
    first = (kl_out / "report" / "report.md").read_text()
    kl.build_kl_report(s.out, kl_out, kcfg)
    assert (kl_out / "report" / "report.md").read_text() == first  # deterministic rebuild


def test_sample_completions_match_policy_sample(tiny_model_factory, tokenizer, task):
    """The KL sampler draws exactly what LocalLMPolicy.sample draws (same loop, batches and generator),
    including when m spans several batches."""
    config = make_config(batch_size=5)
    policy = perturb(LocalLMPolicy.from_loaded(tiny_model_factory(), tokenizer, config, seed=0), seed=6)
    with torch.no_grad():
        ref = policy.sample(task, [3] * 12, torch.Generator().manual_seed(21))
    got = kl.sample_completions(policy, task, 3, 12, torch.Generator().manual_seed(21))
    assert [list(a) for a in got] == ref.completion_ids
