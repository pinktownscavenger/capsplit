"""Tests for the tail layer (`lm/tail.py`): checker on prefixes, vocabulary, D, bracket, path
classification, importance sampling and the stage on a synthetic run.

Exactness tests use a toy model over a small fake vocabulary, where p0 can be enumerated.
"""
from __future__ import annotations

import itertools
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from capsplit.harness.lm import stages as st
from capsplit.harness.lm import tail
from capsplit.harness.lm.config import parse_config
from capsplit.harness.lm.policy import LocalLMPolicy
from capsplit.harness.lm.tasks import check

from .conftest import make_config
from .test_kl import _write_tiny_adapters, perturb
from .test_report import base_dir, default_runs, make, raw_config  # noqa: F401  (base_dir is a fixture)

ROOT = Path(__file__).resolve().parents[2]
ALPHABET = "0123456789.,-− ax"


# --------------------------------------------------------------------------- checker on prefixes


def test_prefix_status_never_contradicted_by_extensions():
    rng = random.Random(0)
    for _ in range(4000):
        answer = rng.choice(["12", "120", "7", "1234", "1000"])
        text = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 8)))
        status = tail.prefix_status(text, answer)
        for _ in range(12):
            ext = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 10)))
            correct = check(text + ext, answer)[1]
            if status == "won":
                assert correct, (text, ext, answer)
            elif status == "dead":
                assert not correct, (text, ext, answer)


def test_prefix_status_examples():
    assert tail.prefix_status("", "12") == "open"
    assert tail.prefix_status("1", "12") == "open"
    assert tail.prefix_status("3", "12") == "dead"
    assert tail.prefix_status("-1", "12") == "dead"
    assert tail.prefix_status("12 ", "12") == "won"
    assert tail.prefix_status("12.", "12") == "open"  # "12.0" is accepted, "12.5" is not
    assert tail.prefix_status("12.x", "12") == "won"
    assert tail.prefix_status("1,", "1234") == "open"
    assert tail.prefix_status("0", "12") == "open"  # leading zeros: "012" reads as 12
    assert tail.prefix_status("The answer is 12\n", "12") == "won"
    assert tail.prefix_status("123", "12") == "dead"
    assert tail.terminal_status("12", "12") == "won"
    assert tail.terminal_status("", "12") == "dead"


def test_final_shortcut_agrees_with_probes():
    rng = random.Random(1)
    for _ in range(3000):
        text = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 14)))
        m = tail._NUMBER.search(text)
        if m is None or len(text) - m.end() < 6:
            continue
        assert tail._final_tail(text[m.start():]), text


def test_finality_depends_only_on_character_classes():
    rng = random.Random(2)
    for _ in range(3000):
        text = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 9)))
        m = tail._NUMBER.search(text)
        if m is None:
            continue
        assert tail._final_tail(text[m.start():]) == tail._final_tail(tail.class_rep(text[m.start():])), text


def test_class_rep():
    assert tail.class_rep("a1.,-−") == "x0.,--"


# --------------------------------------------------------------------------- a toy model with a fake vocabulary


class FakeTok:
    """A tiny tokenizer: digits, a few text pieces, a stop token and a special token that decodes to ""."""

    PIECES = (*"0123456789", " ", "ab", ".", ",", "-", "<stop>", "<sp>")

    def __init__(self, pieces: list[str] | None = None):
        self.pieces = pieces or list(self.PIECES)
        self.all_special_ids = [self.pieces.index("<stop>"), self.pieces.index("<sp>")]

    def __len__(self) -> int:
        return len(self.pieces)

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join("" if (skip_special_tokens and i in self.all_special_ids) else self.pieces[i] for i in ids)


TOK = FakeTok()
STOP = {TOK.pieces.index("<stop>")}
V = len(TOK) + 2  # two padded logit rows beyond the tokenizer


def toy_logprobs(prefix: tuple) -> np.ndarray:
    """A deterministic next-token distribution per prefix, with mass on '1', '2' and stop."""
    rng = np.random.default_rng(abs(hash(("toy", prefix))) % (2**32))
    logits = rng.normal(0, 1.5, V)
    logits[[1, 2]] += 1.5
    logits[TOK.pieces.index("<stop>")] += 1.0 + 0.5 * len(prefix)
    return logits - np.log(np.exp(logits).sum())


def enumerate_completions(max_new: int):
    """Every completion (ids incl. the stop token) with its probability."""
    out = []

    def walk(prefix: tuple, logp: float) -> None:
        row = toy_logprobs(prefix)
        for t in range(V):
            lp = logp + row[t]
            child = prefix + (t,)
            if t in STOP or len(child) >= max_new:
                out.append((child, math.exp(lp)))
            else:
                walk(child, lp)

    walk((), 0.0)
    return out


def exact_p0(answer: str, max_new: int) -> float:
    decode = tail.decoder(TOK, len(TOK))
    return sum(p for y, p in enumerate_completions(max_new) if check(decode(y), answer)[1])


def toy_vocab() -> tail.Vocab:
    return tail.build_vocab(TOK, V, STOP)


def test_build_vocab_partition():
    vocab = toy_vocab()
    assert sorted(vocab.digits.values()) == list("0123456789")
    assert set(vocab.empty.tolist()) == {TOK.pieces.index("<sp>"), len(TOK), len(TOK) + 1}
    ends = tail.end_mask(vocab)
    assert ends[TOK.pieces.index(" ")] and ends[TOK.pieces.index("ab")] and ends[TOK.pieces.index("-")]
    assert ends[TOK.pieces.index("<stop>")]
    assert not ends[TOK.pieces.index(".")] and not ends[TOK.pieces.index(",")] and not ends[1]
    with pytest.raises(ValueError, match="single digit"):
        tail.build_vocab(FakeTok([*"0123456789", "12", "<stop>", "<sp>"]), 13, {10 + 1})


def test_d_bound_is_the_d_set_mass_and_a_lower_bound():
    vocab = toy_vocab()
    ends = tail.end_mask(vocab)
    for answer, max_new in (("12", 4), ("1", 3), ("21", 2)):
        digits = tail.answer_ids(vocab, answer)
        path = np.stack([toy_logprobs(tuple(digits[:i])) for i in range(len(digits) + 1)])
        d = tail.d_bound(path, digits, ends, max_new)["D"]
        mass = sum(p for y, p in enumerate_completions(max_new) if tail.in_d_set(y, digits, ends, max_new))
        assert d == pytest.approx(mass, rel=1e-9)
        assert d <= exact_p0(answer, max_new) + 1e-12


@pytest.mark.parametrize("budget", [1, 3, 10, 40, 10_000])
def test_bracket_contains_exact_p0_and_is_exact_when_fully_expanded(budget):
    vocab = toy_vocab()
    decode = tail.decoder(TOK, len(TOK))
    for answer in ("12", "1"):
        b = tail.expand_tree(lambda ps: np.stack([toy_logprobs(tuple(p)) for p in ps]), answer, vocab, decode,
                             max_new=4, budget=budget, children=2, batch=4)
        p0 = exact_p0(answer, 4)
        assert b.won <= p0 + 1e-12 <= b.won + b.open + 2e-12
        assert b.total == pytest.approx(1.0, abs=1e-9)
        if budget == 10_000:
            assert b.open == pytest.approx(0.0, abs=1e-12) or b.open_lumped > 0
            # with every node expandable, only lumped (non-explicit) open mass can remain
            assert b.open_frontier == pytest.approx(0.0, abs=1e-12)


def test_classify_path_remainder_identity():
    """won mass + P(success and in the remainder) = p0 exactly; 'won' paths succeed, 'dead' fail."""
    vocab = toy_vocab()
    decode = tail.decoder(TOK, len(TOK))
    for budget in (1, 5, 30):
        b = tail.expand_tree(lambda ps: np.stack([toy_logprobs(tuple(p)) for p in ps]), "12", vocab, decode,
                             max_new=4, budget=budget, children=2, batch=3)
        expanded = {tuple(x) for x in b.expanded}
        rem_success, rem_mass = 0.0, 0.0
        for y, p in enumerate_completions(4):
            region = tail.classify_path(y, expanded, "12", decode, 4, STOP)
            ok = check(decode(y), "12")[1]
            if region == "won":
                assert ok
            elif region == "dead":
                assert not ok
            else:
                rem_mass += p
                rem_success += p * ok
        assert b.won + rem_success == pytest.approx(exact_p0("12", 4), abs=1e-12)
        assert rem_mass == pytest.approx(b.open, abs=1e-9)


# --------------------------------------------------------------------------- estimator and trust rule


def test_plain_is_estimator_unbiased_on_toy():
    seqs = enumerate_completions(3)
    decode = tail.decoder(TOK, len(TOK))
    p = np.array([q for _, q in seqs])
    success = np.array([check(decode(y), "12")[1] for y, _ in seqs])
    truth = float((p * success).sum())
    rng = np.random.default_rng(0)
    # proposals: pi0 itself and a "success-tilted" one; equal-weight mixture
    tilt = p * np.where(success, 20.0, 1.0)
    tilt /= tilt.sum()
    comps = np.stack([np.log(p), np.log(tilt)], axis=1)
    n, reps = 400, 200
    estimates = []
    for _ in range(reps):
        c = rng.integers(0, 2, n)
        idx = np.where(c == 0, rng.choice(len(p), n, p=p), rng.choice(len(p), n, p=tilt))
        log_w = np.log(p[idx]) - tail.mixture_logq(comps[idx], [0.5, 0.5])
        estimates.append(tail.estimate(success[idx], log_w)["estimate"])
    estimates = np.array(estimates)
    assert abs(estimates.mean() - truth) < 4 * estimates.std(ddof=1) / math.sqrt(reps)


def test_estimate_offset_and_se():
    res = tail.estimate(np.array([1, 0, 1, 0]), np.log(np.array([0.5, 9.0, 0.25, 1.0])), offset=0.1)
    assert res["remainder"] == pytest.approx(0.1875)
    assert res["estimate"] == pytest.approx(0.2875)
    assert res["se"] == pytest.approx(np.std([0.5, 0, 0.25, 0], ddof=1) / 2)


def test_mixture_logq():
    lp = np.log(np.array([[0.2, 0.4, 0.1, 0.3]]))
    assert tail.mixture_logq(lp, [0.25] * 4)[0] == pytest.approx(math.log(0.25))


def test_gpd_khat_recovers_shape():
    rng = np.random.default_rng(3)
    heavy = (rng.pareto(2.0, 4000) + 1)  # GPD shape 1/2
    k = tail.gpd_khat(np.sort(heavy)[-400:] - np.sort(heavy)[-401])
    assert 0.3 < k < 0.7
    light = rng.exponential(1.0, 4000)
    assert tail.gpd_khat(np.sort(light)[-400:] - np.sort(light)[-401]) < 0.2


def test_trust_rule():
    cfg = tail.TailConfig()
    ok = tail.trust(np.ones(40), cfg)
    assert ok["trusted"] and ok["ess"] == pytest.approx(40)
    few = tail.trust(np.ones(10), cfg)
    assert not few["trusted"] and not few["checks"]["n_success"]
    spike = tail.trust(np.r_[np.ones(30), 100.0], cfg)
    assert not spike["checks"]["max_share"] and not spike["trusted"]
    assert tail.pareto_k(np.r_[np.ones(30), 2.0, 3.0]) == -math.inf  # fewer than 4 distinct exceedances
    assert tail.pareto_k(np.ones(40)) == -math.inf
    assert not tail.trust(np.array([]), cfg)["trusted"]


def test_tail_config_strict(tmp_path):
    raw = json.loads((ROOT / "configs" / "lm_tail.json").read_text())
    assert tail.parse_tail_config(raw) == tail.TailConfig(**{k: tuple(v) if isinstance(v, list) else v
                                                             for k, v in raw.items()})
    with pytest.raises(tail.TailConfigError, match="unknown"):
        tail.parse_tail_config({**raw, "budgett": 1})
    with pytest.raises(tail.TailConfigError, match="weights"):
        tail.parse_tail_config({**raw, "weights": [0.5, 0.5]})


def test_select_bf_problems_and_bracket_check():
    cfg = tail.TailConfig(n_bf_problems=3, bf_range=(0.01, 0.05))
    p_hat = np.array([0.0, 0.02, 0.03, 0.04, 0.05, 0.2, 0.011])
    assert tail.select_bf_problems(p_hat, cfg) == [2, 4, 6]
    k, m = np.array([0, 50, 5]), np.array([560, 560, 560])
    res = tail.bracket_check(k, m, lo=np.array([0.0, 0.5, 0.0]), hi=np.array([1e-3, 0.6, 1.0]),
                             d=np.array([0.0, 0.5, 0.5]), alpha=0.05)
    assert res["bracket_fail"] == [1]  # p-hat 50/560 is far below 0.5
    assert res["d_fail"] == [1, 2]


# --------------------------------------------------------------------------- the real tokenizer and a tiny model


def test_real_vocab_groups_match_token_classification(tokenizer):
    stop = {int(tokenizer.eos_token_id), int(tokenizer.convert_tokens_to_ids("<|im_end|>"))}
    vocab = tail.build_vocab(tokenizer, len(tokenizer) + 3, stop)
    rng = random.Random(0)
    grouped = np.flatnonzero(vocab.group_of >= 0)
    texts = ["", "The answer", "12", "1,", "12.", "4", "-", "3 ", "1,23", "0"]
    for _ in range(3000):
        tid = int(rng.choice(grouped))
        piece = tokenizer.decode([tid], skip_special_tokens=True)
        rep = vocab.reps[vocab.group_of[tid]]
        for t in texts:
            assert tail.prefix_status(t + piece, "123") == tail.prefix_status(t + rep, "123"), (t, piece)
            assert tail.terminal_status(t + piece, "123") == tail.terminal_status(t + rep, "123"), (t, piece)


@pytest.fixture
def tiny_policy(tiny_model_factory, tokenizer):
    config = make_config(batch_size=4)
    return LocalLMPolicy.from_loaded(tiny_model_factory(), tokenizer, config, seed=0), config


def test_scale_zero_adapter_reproduces_base_exactly(tiny_policy, task):
    policy, _ = tiny_policy
    perturb(policy, seed=5)
    prompt = tail.prompt_ids(policy, task, 1)
    state = policy.state_dict()
    policy.load_state_dict(tail.scaled_state(state, 0.0))
    on = tail.forward_logprobs(policy, prompt, [[17, 18], [15]], keep=1, use_adapter=True)[0]
    off = tail.forward_logprobs(policy, prompt, [[17, 18], [15]], keep=1, use_adapter=False)[0]
    assert np.array_equal(on, off)
    policy.load_state_dict(state)
    moved = tail.forward_logprobs(policy, prompt, [[17, 18], [15]], keep=1, use_adapter=True)[0]
    assert not np.array_equal(moved, off)


def test_forward_logprobs_matches_policy_log_prob(tiny_policy, task):
    policy, _ = tiny_policy
    prompt = tail.prompt_ids(policy, task, 2)
    completion = [17, 18, 15, int(policy.end_id)]
    lp = tail.forward_logprobs(policy, prompt, [completion[:-1]], keep=len(completion))[0][0]
    total = sum(lp[i, t] for i, t in enumerate(completion))
    with torch.no_grad(), policy.model.disable_adapter():
        ref = float(policy.log_prob(task, [2], [tuple(completion)])[0])
    assert total == pytest.approx(ref, abs=1e-4)


def test_sample_with_logq_matches_sample_batch_and_teacher_forcing(tiny_policy, task):
    from capsplit.harness.lm.model import sample_batch

    policy, _ = tiny_policy
    prompt = tail.prompt_ids(policy, task, 0)
    ids = torch.tensor([prompt] * 6)
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    toks, logq = tail.sample_with_logq(policy.model, batch, 12, 1.3, policy.stop_ids, policy.pad_id,
                                       torch.Generator().manual_seed(4))
    ref = sample_batch(policy.model, batch, 12, 1.3, 1.0, policy.stop_ids, policy.pad_id,
                       torch.Generator().manual_seed(4))
    assert toks == [ref.tokens[i, : int(n)].tolist() for i, n in enumerate(ref.lengths)]
    tf = tail.score_completions(policy, prompt, toks, (1.3,), use_adapter=True)[0]
    for i, t in enumerate(toks):
        assert np.allclose(tf[i, : len(t)], logq[i], atol=1e-4)
        assert np.isnan(tf[i, len(t):]).all()


def test_is_chunk_rows(tiny_policy, task):
    policy, _ = tiny_policy
    perturb(policy, seed=7)
    state = policy.state_dict()
    states = {"sft": state, "rl_half": tail.scaled_state(state, 0.5)}
    prompt = tail.prompt_ids(policy, task, 1)
    rows = tail.is_chunk(policy, prompt, 10, (0.25,) * 4, 1.3, states, torch.Generator().manual_seed(2))
    assert [r["index"] for r in rows] == list(range(10))
    for r in rows:
        assert set(tail.COMPONENTS) <= {k[3:] for k in r if k.startswith("lp_")}
        assert r["gap_max"] < 1e-3  # float32 CPU: sampling-time and teacher-forced log q agree
        own = r[f"lp_{r['component']}"]
        assert own == pytest.approx(sum(r["logq_sample"]), abs=1e-3)
    again = tail.is_chunk(policy, prompt, 10, (0.25,) * 4, 1.3, states, torch.Generator().manual_seed(2))
    assert [r["ids"] for r in again] == [r["ids"] for r in rows]


def test_expand_tree_on_tiny_model_contains_samples(tiny_policy, task):
    """On the tiny model the bracket brackets the fraction of base samples the checker accepts."""
    policy, _ = tiny_policy
    stop = {int(s) for s in policy.stop_ids}
    vocab = tail.build_vocab(policy.tokenizer, int(policy.model.get_output_embeddings().weight.shape[0]), stop)
    decode = tail.decoder(policy.tokenizer, vocab.n_tokens)
    prompt = tail.prompt_ids(policy, task, 0)
    b = tail.expand_tree(lambda ps: tail.forward_logprobs(policy, prompt, ps, keep=1)[0][:, 0, :], "81", vocab,
                         decode, max_new=3, budget=6, children=4, batch=3)
    assert b.total == pytest.approx(1.0, abs=1e-4)
    assert b.expansions == 6 and 0.0 <= b.won <= b.won + b.open <= 1.0 + 1e-6


def test_tail_stage_on_synth(base_dir, tmp_path, tiny_model_factory, tokenizer):  # noqa: F811
    s = make(base_dir, tmp_path)
    default_runs(s)
    config = parse_config(raw_config())

    def factory(seed: int) -> LocalLMPolicy:
        return LocalLMPolicy.from_loaded(tiny_model_factory(), tokenizer, config, seed=seed)

    _write_tiny_adapters(s, factory)
    tcfg = tail.parse_tail_config({**json.loads((ROOT / "configs" / "lm_tail.json").read_text()),
                                   "budget": 3, "children": 2, "batch": 2, "n_is": 6, "n_bf": 5,
                                   "n_bf_problems": 2, "bf_range": [0.0, 1.0], "sft": ["sft", 0, 2],
                                   "rl": ["rl", 0, 2], "max_problems": 6})
    out = tmp_path / "synth_tail"
    real_check = tail.bracket_check

    def failing(*args, **kwargs):
        return {**real_check(*args, **kwargs), "bracket_fail": [0]}

    tail.bracket_check = failing
    try:
        with pytest.raises(st.StageError, match="bug to investigate"):
            tail.run_tail(config, tcfg, s.out, out, device="cpu", factory=factory)
        assert not tail.tail_paths(out)["is"].exists()  # stopped before importance sampling
    finally:
        tail.bracket_check = real_check
    tail.run_tail(config, tcfg, s.out, out, device="cpu", factory=factory)
    paths = tail.tail_paths(out)
    d_rows = list(map(json.loads, paths["d"].read_text().splitlines()))
    assert len(d_rows) == 6 and all(0.0 <= r["D"] <= 1.0 for r in d_rows)
    sel = json.loads(paths["selection"].read_text())
    assert sel["bf"] and all((paths["is"] / f"{p}.jsonl").exists() for p in set(sel["target"]) | set(sel["bf"]))
    first = {p: p.read_text() for p in sorted(out.rglob("*.json*")) if "manifest" not in p.name}
    tail.run_tail(config, tcfg, s.out, out, device="cpu", factory=factory)  # resume: nothing new
    assert {p: p.read_text() for p in sorted(out.rglob("*.json*")) if "manifest" not in p.name} == first
    rows = [json.loads(x) for x in next(iter(sorted(paths["is"].glob("*.jsonl")))).read_text().splitlines()]
    assert {r["region"] for r in rows} <= {"won", "dead", "rem"} and len(rows) == 6

    from capsplit.harness.lm.tail_report import build_tail_report

    path = build_tail_report(config, s.out, out, tcfg, None)
    res = json.loads(path.read_text())
    assert len(res["per_problem"]) == 6 and res["budget_curve"]["n"][-1] == 2 ** tcfg.grid_max_exp
    assert all(r["lo"] <= r["hi"] for r in res["per_problem"])
    for name in ("report.md", "tables/per_problem.csv", "tables/budget_curve.csv", "tables/discovery.csv",
                 "figures/brackets.png", "figures/budget_labels.png"):
        assert (out / "report" / name).exists(), name
    first = (out / "report" / "report.md").read_text()
    build_tail_report(config, s.out, out, tcfg, None)
    assert (out / "report" / "report.md").read_text() == first


def test_label_of_and_budget_curve():
    from capsplit.harness.lm.tail_report import budget_curve_ext, label_of

    assert label_of(0.2, 0.3, 8) == "reachable"
    assert label_of(0.01, 0.05, 16) == "unreachable"
    assert label_of(0.01, 0.1, 16) == "ambiguous"
    curve = budget_curve_ext(np.array([0.5, 0.0]), np.array([0.6, 1e-4]), {"rl": [np.array([0.1, 0.2])]}, [2, 4])
    assert curve["counts"] == [{"reachable": 1, "unreachable": 1, "ambiguous": 0}] * 2
    assert curve["contributions"]["rl"][0] == {"reachable": 0.05, "unreachable": 0.1, "ambiguous": 0.0}


def test_itertools_probe_count():
    assert len(tail._PROBES) == sum(5**n for n in range(6))
    assert list(itertools.islice(tail._PROBES, 1)) == [""]
