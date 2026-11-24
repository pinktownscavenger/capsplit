"""LocalLMPolicy on a tiny random Qwen2 with the real tokenizer (CPU, seconds), plus one
real-model test marked `lm_model`."""
from __future__ import annotations

import time

import pytest
import torch

from capsplit.harness.lm.model import encode_prompts
from capsplit.harness.lm.policy import LMRollout, LocalLMPolicy
from capsplit.harness.registry import POLICIES

from .conftest import make_config


@pytest.fixture
def policy(tiny_model_factory, tokenizer, lm_config):
    return LocalLMPolicy.from_loaded(tiny_model_factory(), tokenizer, lm_config, seed=0)


def _independent_scores(policy, task, pid, completion):
    """log_prob and entropy of one completion from an unbatched, unpadded forward."""
    batch = encode_prompts(policy.tokenizer, policy.config.model.system_prompt, [task.prompts()[pid].tokens[0]],
                           policy.device)
    prompt = batch["input_ids"][0]
    ids = torch.cat([prompt, torch.tensor(completion)]).unsqueeze(0)
    logits = policy.model(input_ids=ids).logits[0].float() / policy.temperature
    logp = torch.log_softmax(logits[len(prompt) - 1 : -1], dim=-1)
    token_lp = logp.gather(-1, torch.tensor(completion).unsqueeze(-1)).sum()
    entropy = -(logp.exp() * logp).sum()
    return token_lp, entropy


def _sgd_step(policy, task, lr=5.0):
    """One large step raising the log-prob of fixed answers, so the adapter clearly moves."""
    loss = -policy.log_prob(task, [0, 1, 2, 3], ["81", "803", "408", "56"]).mean()
    policy.model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for p in policy.parameters():
            p -= lr * p.grad
    policy.model.zero_grad()


def test_sampled_log_prob_matches_unpadded_teacher_forcing(policy, task):
    ids = [0, 1, 2, 3, 0, 1, 3, 2]
    rollout = policy.sample(task, ids, torch.Generator().manual_seed(0))
    assert isinstance(rollout, LMRollout)
    assert rollout.n_forced == [0] * len(ids) and rollout.prompt_ids == ids
    assert [tuple(c) for c in rollout.completion_ids] == rollout.actions
    lengths = {len(a) for a in rollout.actions}
    assert len(lengths) > 1, "the tiny model should end completions at varied lengths"
    for i, (pid, action) in enumerate(zip(ids, rollout.actions)):
        lp, ent = _independent_scores(policy, task, pid, list(action))
        assert rollout.log_prob[i].item() == pytest.approx(lp.item(), abs=1e-4)
        assert rollout.entropy[i].item() == pytest.approx(ent.item(), abs=1e-3)
        stopped = action[-1] in policy.stop_ids
        assert stopped or len(action) == policy.config.model.max_new_tokens
        assert rollout.texts[i] == policy.tokenizer.decode(list(action), skip_special_tokens=True)
    forced = policy.log_prob(task, ids, rollout.actions)
    assert torch.allclose(forced, rollout.log_prob, atol=1e-5)


def test_prompt_and_padding_excluded(policy, task):
    action = tuple(policy.tokenizer("81", add_special_tokens=False)["input_ids"]) + (policy.end_id,)
    alone = policy.log_prob(task, [0], [action])
    long_action = tuple(range(1000, 1009))
    # task prompt 3 is longer (more left padding for prompt 0); the long action adds right padding
    padded = policy.log_prob(task, [3, 0, 1], [long_action, action, action[:1]])
    assert padded[1].item() == pytest.approx(alone.item(), abs=1e-5)
    # strings are tokenized without special tokens, plus the stop token
    assert policy.log_prob(task, [0], ["81"]).item() == pytest.approx(alone.item(), abs=1e-6)
    # start skips the first steps; an empty completion scores 0
    first = policy.log_prob(task, [0], [action[:1]])
    rest = policy.log_prob(task, [0], [action], start=1)
    assert (first + rest).item() == pytest.approx(alone.item(), abs=1e-5)
    assert policy.log_prob(task, [0, 1], [(), ()]).tolist() == [0.0, 0.0]


def test_gradients_flow_only_into_lora(policy, task):
    rollout = policy.sample(task, [0, 1, 2], torch.Generator().manual_seed(1))
    (-(rollout.log_prob.sum()) - rollout.entropy.sum()).backward()
    names = {id(p): n for n, p in policy.model.named_parameters()}
    trainable = list(policy.parameters())
    assert trainable and all("lora_" in names[id(p)] for p in trainable)
    for name, p in policy.model.named_parameters():
        if "lora_" not in name:
            assert p.grad is None and not p.requires_grad, name
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable)


def test_base_and_adapter_sampling(policy, task):
    ids = [0, 1, 2, 3] * 2

    def texts(use_adapter):
        return policy.sample_texts(task, ids, 1.0, torch.Generator().manual_seed(7), use_adapter=use_adapter)

    base_before = texts(False)
    assert texts(True) == base_before  # fresh LoRA (B = 0) is the identity
    assert all(n >= 1 for _, n in base_before)
    _sgd_step(policy, task)
    assert texts(False) == base_before  # the base is untouched by adapter training
    assert texts(True) != base_before


def test_save_load_adapter_round_trip(policy, task, tiny_model_factory, tokenizer, lm_config, tmp_path):
    _sgd_step(policy, task)
    path = policy.save_adapter(tmp_path / "step1" / "adapter.safetensors")
    assert path.exists() and not list(path.parent.glob("*.tmp"))
    other = LocalLMPolicy.from_loaded(tiny_model_factory(), tokenizer, lm_config, seed=99)
    other.load_adapter(path)
    saved, loaded = policy.state_dict(), other.state_dict()
    assert saved.keys() == loaded.keys()
    assert all(torch.equal(saved[k], loaded[k]) and loaded[k].device.type == "cpu" for k in saved)
    actions = [(1, 2, 3), (4, 5)]
    assert torch.equal(policy.log_prob(task, [0, 1], actions), other.log_prob(task, [0, 1], actions))
    with pytest.raises(ValueError):
        other.load_state_dict({k: v for k, v in list(saved.items())[1:]})


def test_deterministic_on_cpu(tiny_model_factory, tokenizer, lm_config, task):
    rng_before = torch.get_rng_state()
    a = LocalLMPolicy.from_loaded(tiny_model_factory(), tokenizer, lm_config, seed=3)
    assert torch.equal(torch.get_rng_state(), rng_before)  # LoRA init does not touch the global RNG
    b = LocalLMPolicy.from_loaded(tiny_model_factory(), tokenizer, lm_config, seed=3)
    c = LocalLMPolicy.from_loaded(tiny_model_factory(), tokenizer, lm_config, seed=4)
    sa, sb, sc = a.state_dict(), b.state_dict(), c.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert any(not torch.equal(sa[k], sc[k]) for k in sa if "lora_A" in k)
    for p in (a, b):  # make the adapters non-trivial before comparing samples
        _sgd_step(p, task, lr=1.0)
    ra = a.sample(task, [0, 1, 2, 3], torch.Generator().manual_seed(5))
    rb = b.sample(task, [0, 1, 2, 3], torch.Generator().manual_seed(5))
    assert ra.actions == rb.actions and torch.equal(ra.log_prob, rb.log_prob)


def test_batch_size_does_not_change_samples_or_scores(tiny_model_factory, tokenizer, task):
    small = LocalLMPolicy.from_loaded(tiny_model_factory(), tokenizer, make_config(batch_size=3), seed=0)
    big = LocalLMPolicy.from_loaded(tiny_model_factory(), tokenizer, make_config(batch_size=64), seed=0)
    actions = [(5, 6, 7), (8,), (9, 10), (11, 12, 13, 14), (15,)]
    ids = [0, 1, 2, 3, 0]
    assert torch.allclose(small.log_prob(task, ids, actions), big.log_prob(task, ids, actions), atol=1e-5)
    assert len(small.sample(task, ids, torch.Generator().manual_seed(0)).actions) == 5


def test_interface_details(policy, task):
    assert POLICIES.get("local_lm") is LocalLMPolicy
    with pytest.raises(NotImplementedError, match="sampled measurement only"):
        policy.state_logits(task, 0)
    assert policy.tempering_logits(task, 0) is None
    with pytest.raises(NotImplementedError, match="prefixes"):
        policy.sample(task, [0], torch.Generator().manual_seed(0), prefixes=[(1,)])
    original = policy.state_dict()
    twin = policy.clone()
    _sgd_step(twin, task)
    assert all(torch.equal(v, original[k]) for k, v in policy.state_dict().items())
    assert any(not torch.equal(v, policy.state_dict()[k]) for k, v in twin.state_dict().items())


@pytest.mark.lm_model
def test_real_model_samples_and_steps(lm_config, task):
    from capsplit.harness.lm.device import pick_device, process_gb
    from capsplit.harness.lm.model import choose, fetch

    try:
        fetch(lm_config.model, choose(lm_config.model, False), local_only=True)
    except (OSError, ValueError) as error:  # not cached: huggingface_hub raises LocalEntryNotFoundError (an OSError)
        pytest.skip(f"{lm_config.model.id} weights are not in the local Hugging Face cache ({error})")
    t0 = time.monotonic()
    policy = LocalLMPolicy(lm_config, device=pick_device(lm_config.model.device), seed=0)
    rollout = policy.sample(task, [0, 1, 2, 3], torch.Generator().manual_seed(0))
    assert len(rollout.texts) == 4 and torch.isfinite(rollout.log_prob).all()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    before = policy.state_dict()
    loss = -(rollout.log_prob.mean())
    loss.backward()
    optimizer.step()
    after = policy.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)
    print(f"\n[lm_model] device={policy.device.type} texts={rollout.texts} "
          f"seconds={time.monotonic() - t0:.1f} process_gb={process_gb():.2f}")
