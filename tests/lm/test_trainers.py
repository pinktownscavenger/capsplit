"""LM trainers on a tiny random Qwen2 with the real tokenizer (CPU, seconds)."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
import torch

from capsplit.harness.lm.policy import LocalLMPolicy
from capsplit.harness.lm.trainers import LMRL, LMSFT, LMNone, LMSFTThenRL, ProblemCycle
from capsplit.harness.registry import TRAINERS

from .conftest import QUESTIONS, StubTask

QUESTIONS6 = QUESTIONS + (
    ("Compute 5 + 6. Reply with only the final number.", "11"),
    ("Compute 9 * 9. Reply with only the final number.", "81"),
)


class ParityTask(StubTask):
    """Correct iff the completion text has even length (the tiny model hits it about half the time);
    records every verify call. Answers serve as demonstrations."""

    def __init__(self, questions: Sequence[tuple[str, str]] = QUESTIONS6, always: bool | None = None):
        super().__init__(questions)
        self.calls: list[tuple[int, str]] = []
        self.always = always

    def verify(self, prompt_id: int, actions: object) -> bool:
        assert isinstance(actions, str), "LM rewards must come from the decoded text"
        self.calls.append((prompt_id, actions))
        return self.always if self.always is not None else len(actions) % 2 == 0

    def demonstrations(self, prompt_id: int) -> Sequence[Any]:
        return (self._answers[prompt_id],)


@pytest.fixture
def make_policy(tiny_model_factory, tokenizer, lm_config):
    def build(seed: int = 0) -> LocalLMPolicy:
        return LocalLMPolicy.from_loaded(tiny_model_factory(), tokenizer, lm_config, seed=seed)

    return build


TRAIN = [0, 1, 2, 4, 5]  # problem 3 is "held out"


def _run(trainer, policy, task, budget=3, checkpoints=(0, 2, 3), seed=0, prompt_ids=TRAIN):
    """Train, counting optimizer steps and snapshotting the adapter at each checkpoint."""
    seen: list[int] = []
    snaps: dict[int, dict] = {}
    steps = {"n": 0, "grads": []}
    original = torch.optim.AdamW.step

    def counting_step(self, *args, **kwargs):
        steps["n"] += 1
        steps["grads"].append([p.grad.detach().clone() if p.grad is not None else None
                               for g in self.param_groups for p in g["params"]])
        return original(self, *args, **kwargs)

    def hook(step, p):
        seen.append(step)
        snaps[step] = p.state_dict()

    torch.optim.AdamW.step = counting_step  # type: ignore[method-assign]
    try:
        trainer.train(policy, task, budget, seed, prompt_ids, list(checkpoints), hook)
    finally:
        torch.optim.AdamW.step = original  # type: ignore[method-assign]
    return seen, snaps, steps


def _classes():
    return {str(i): c for i, c in enumerate(["unreachable", "reachable", "unreachable", "unreachable",
                                             "ambiguous", "unreachable"])}


def _same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# --------------------------------------------------------------------------- contract


def test_registered_and_unknown_params_raise():
    for name in ("lm_none", "lm_rl", "lm_sft", "lm_sft_then_rl"):
        assert name in TRAINERS
        with pytest.raises(ValueError, match="unknown"):
            TRAINERS.create(name, bogus=1)
    with pytest.raises(ValueError, match="unknown"):
        LMSFTThenRL(rl={"grup": 4})
    with pytest.raises(ValueError):
        LMSFT(target="easy")
    with pytest.raises(ValueError):
        LMRL(group=1)


def test_none_calls_checkpoints_only(make_policy):
    policy, task = make_policy(), ParityTask()
    before = policy.state_dict()
    trainer = LMNone()
    seen, _, steps = _run(trainer, policy, task, budget=3)
    assert seen == [0, 2, 3] and steps["n"] == 0
    assert _same(before, policy.state_dict()) and trainer.training_log() is None and task.calls == []


@pytest.mark.parametrize("budget", [0, 1, 4])
def test_rl_checkpoints_and_update_count(make_policy, budget):
    trainer = LMRL(group=2, problems_per_update=2, lr=1e-2)
    checkpoints = sorted({0, budget, min(1, budget)})
    seen, snaps, steps = _run(trainer, make_policy(), ParityTask(), budget=budget, checkpoints=checkpoints)
    assert seen == checkpoints
    assert steps["n"] == budget
    log = trainer.training_log()
    assert log is not None and log["updates"] == budget
    if budget:
        assert not _same(snaps[0], snaps[budget]), "the adapter should move"


def test_rl_rewards_from_verify_and_log_replay(make_policy, monkeypatch):
    policy, task = make_policy(), ParityTask()
    rollouts = []
    original = policy.sample

    def spy(*args, **kwargs):
        r = original(*args, **kwargs)
        rollouts.append(r)
        return r

    monkeypatch.setattr(policy, "sample", spy)
    G, K, B = 3, 2, 5
    trainer = LMRL(group=G, problems_per_update=K, lr=1e-2)
    _run(trainer, policy, task, budget=B, checkpoints=(0, 2, 5))
    log = trainer.training_log()
    assert log is not None and len(rollouts) == B
    # Every reward is task.verify on the decoded text, called once per episode.
    texts = [(pid, t) for r in rollouts for pid, t in zip(r.prompt_ids, r.texts)]
    assert task.calls == texts
    # Replay: first success, correct and episode counts per checkpoint, tokens incl. stop token.
    first: dict[int, int | None] = {p: None for p in TRAIN}
    for update, r in enumerate(rollouts, start=1):
        for pid, text in zip(r.prompt_ids, r.texts):
            if len(text) % 2 == 0 and first[pid] is None:
                first[pid] = update
    for pid in TRAIN:
        entry = log["prompts"][str(pid)]
        assert entry["first_success_update"] == first[pid]
        mine = [(u, t, len(a)) for u, r in enumerate(rollouts, 1) for p, t, a in zip(r.prompt_ids, r.texts, r.actions)
                if p == pid]
        assert entry["episodes_sampled"] == len(mine)
        assert entry["tokens_sampled"] == sum(n for _, _, n in mine)
        for c in (2, 5):
            hits = [t for u, t, _ in mine if u <= c]
            correct = sum(len(t) % 2 == 0 for t in hits)
            assert entry["correct_by_checkpoint"][str(c)] == correct
            assert entry["episodes_by_checkpoint"][str(c)] == len(hits)
            rate = entry["success_rate_by_checkpoint"][str(c)]
            assert rate == (correct / len(hits) if hits else None)
    # Episodes: G per chosen problem per update; the epoch cycle spreads them evenly.
    totals = log["totals"]
    assert totals["episodes_sampled"] == G * K * B
    per_problem = [log["prompts"][str(p)]["episodes_sampled"] for p in TRAIN]
    assert sum(per_problem) == G * K * B and max(per_problem) - min(per_problem) <= G
    assert "3" not in log["prompts"] and all(3 not in r.prompt_ids for r in rollouts)
    assert all(r.prompt_ids == [p for p in sorted(set(r.prompt_ids)) for _ in range(G)] for r in rollouts)


def test_all_zero_rewards_give_zero_gradient(make_policy):
    policy = make_policy()
    before = policy.state_dict()
    trainer = LMRL(group=3, problems_per_update=2, lr=1.0)
    _, _, steps = _run(trainer, policy, ParityTask(always=False), budget=2, checkpoints=(0, 2))
    assert steps["n"] == 2
    for grads in steps["grads"]:
        assert all(g is None or torch.count_nonzero(g) == 0 for g in grads)
    assert _same(before, policy.state_dict())
    assert trainer.training_log()["totals"]["correct_by_checkpoint"] == {"0": 0, "2": 0}


def test_problem_cycle_is_an_epoch():
    cycle = ProblemCycle(list(range(5)), torch.Generator().manual_seed(0))
    batches = [cycle.next(2) for _ in range(5)]  # 10 draws = 2 full epochs
    assert all(len(set(b)) == 2 for b in batches)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == sorted(list(range(5)) * 2)
    assert ProblemCycle([3, 1], torch.Generator()).next(5) == [1, 3]


def test_rl_same_seed_same_weights(make_policy):
    out = []
    for _ in range(2):
        policy = make_policy()
        trainer = LMRL(group=2, problems_per_update=2, lr=1e-2)
        _run(trainer, policy, ParityTask(), budget=2, checkpoints=(0, 2), seed=7)
        out.append((policy.state_dict(), trainer.training_log()))
    assert _same(out[0][0], out[1][0]) and out[0][1] == out[1][1]


# --------------------------------------------------------------------------- SFT


def test_sft_trains_only_target_train_problems(make_policy, monkeypatch):
    policy, task = make_policy(), ParityTask()
    calls = []
    original = policy.log_prob

    def spy(task_, prompt_ids, actions, start=0):
        calls.append((list(prompt_ids), list(actions)))
        return original(task_, prompt_ids, actions, start)

    monkeypatch.setattr(policy, "log_prob", spy)
    trainer = LMSFT(batch=2, lr=1e-2)
    with pytest.raises(ValueError, match="set_base_classes"):
        trainer.train(policy, task, 1, 0, TRAIN, [0, 1], lambda s, p: None)
    trainer.set_base_classes(_classes())
    seen, snaps, steps = _run(trainer, policy, task, budget=4, checkpoints=(0, 4))
    assert seen == [0, 4] and steps["n"] == 4 and not _same(snaps[0], snaps[4])
    # Unreachable train problems: 0, 2, 5 (3 is unreachable but held out; 1 reachable; 4 ambiguous).
    used = {pid for ids, _ in calls for pid in ids}
    assert used == {0, 2, 5}
    assert all(len(ids) == 2 for ids, _ in calls)
    assert all(a == QUESTIONS6[p][1] for ids, acts in calls for p, a in zip(ids, acts))
    assert task.calls == []
    log = trainer.training_log()
    assert log is not None and log["totals"]["episodes_sampled"] == 0 and log["sft_problem_ids"] == [0, 2, 5]
    assert set(log["prompts"]) == {"0", "2", "5"}
    # Another target; and a train problem missing from the classification raises.
    other = LMSFT(target=["reachable", "ambiguous"])
    other.set_base_classes(_classes())
    assert other.target_ids(TRAIN) == [1, 4]
    other.set_base_classes({0: "reachable"})
    with pytest.raises(ValueError, match="missing"):
        other.target_ids(TRAIN)


def test_sft_loss_is_answer_plus_stop(make_policy):
    """One SGD-free check: the SFT loss equals -mean log_prob of the answer strings."""
    from capsplit.harness.trainers.sft import sft_loss

    policy, task = make_policy(), ParityTask()
    loss = sft_loss(policy, task, [(0, "81"), (2, "408")])  # type: ignore[list-item]
    ids = [list(policy.tokenizer("81", add_special_tokens=False)["input_ids"]) + [policy.end_id],
           list(policy.tokenizer("408", add_special_tokens=False)["input_ids"]) + [policy.end_id]]
    expected = -policy.log_prob(task, [0, 2], [tuple(i) for i in ids]).mean()
    assert loss.item() == pytest.approx(expected.item(), abs=1e-6)


# --------------------------------------------------------------------------- SFT then RL


def test_sft_then_rl_split_and_logs(make_policy):
    policy, task = make_policy(), ParityTask()
    trainer = LMSFTThenRL(sft_fraction=0.4, sft={"batch": 2, "lr": 1e-2}, rl={"group": 2, "problems_per_update": 2,
                                                                                "lr": 1e-2})
    trainer.set_base_classes(_classes())
    seen, _, steps = _run(trainer, policy, task, budget=5, checkpoints=(0, 1, 2, 3, 5))
    assert seen == [0, 1, 2, 3, 5] and steps["n"] == 5  # 2 SFT + 3 RL updates
    log = trainer.training_log()
    assert log is not None and log["updates"] == 5 and (log["n_sft"], log["n_rl"]) == (2, 3)
    assert log["totals"]["episodes_sampled"] == 2 * 2 * 3
    assert log["totals"]["episodes_by_checkpoint"] == {"0": 0, "1": 0, "2": 0, "3": 4, "5": 12}
    firsts = [e["first_success_update"] for e in log["prompts"].values() if e["first_success_update"] is not None]
    assert firsts and min(firsts) >= 3  # global numbering: RL updates are 3..5
    assert len(task.calls) == 12
    # Reproducible from the seed.
    again = make_policy()
    trainer2 = LMSFTThenRL(sft_fraction=0.4, sft={"batch": 2, "lr": 1e-2}, rl={"group": 2, "problems_per_update": 2,
                                                                                 "lr": 1e-2})
    trainer2.set_base_classes(_classes())
    _run(trainer2, again, ParityTask(), budget=5, checkpoints=(0, 1, 2, 3, 5))
    assert _same(policy.state_dict(), again.state_dict()) and trainer2.training_log() == log


@pytest.mark.parametrize("fraction,expected", [(0.0, (0, 3)), (1.0, (3, 0))])
def test_sft_then_rl_edge_fractions(make_policy, fraction, expected):
    trainer = LMSFTThenRL(sft_fraction=fraction, sft={"batch": 2}, rl={"group": 2, "problems_per_update": 2})
    trainer.set_base_classes(_classes())
    seen, _, steps = _run(trainer, make_policy(), ParityTask(), budget=3, checkpoints=(0, 3))
    assert seen == [0, 3] and steps["n"] == 3
    log = trainer.training_log()
    assert log is not None and (log["n_sft"], log["n_rl"]) == expected


# --------------------------------------------------------------------------- micro-batched RL gradient


def _grads(policy) -> list[torch.Tensor]:
    return [p.grad.detach().clone() for p in policy.parameters()]


def test_micro_batched_gradient_equals_full_batch(make_policy):
    """accumulate_pg_gradient at any micro_batch == gradient of -mean(A * log_prob) over the whole batch."""
    from capsplit.harness.lm.trainers import accumulate_pg_gradient
    from capsplit.harness.trainers.rl import group_advantages

    policy, task = make_policy(), ParityTask()
    with torch.no_grad():  # sample has no graph under no_grad
        rollout = policy.sample(task, [p for p in (0, 1, 2) for _ in range(4)], torch.Generator().manual_seed(3))
    assert not rollout.log_prob.requires_grad
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    adv = group_advantages(rewards, 4, "grpo", 1e-6)
    ids, actions = list(rollout.prompt_ids), list(rollout.actions)
    params = list(policy.parameters())
    # Reference: full-batch loss and one backward.
    for p in params:
        p.grad = None
    full_loss = -(adv * policy.log_prob(task, ids, actions)).mean()
    full_loss.backward()
    reference = _grads(policy)
    assert any(torch.count_nonzero(g) for g in reference)
    for micro in (1, 3, 5, 12, 64):
        for p in params:
            p.grad = torch.zeros_like(p)
        loss = accumulate_pg_gradient(policy, task, ids, actions, adv, micro)
        assert loss == pytest.approx(full_loss.item(), abs=1e-6)
        for got, want in zip(_grads(policy), reference):
            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_rl_update_gradient_independent_of_micro_batch(make_policy):
    """The first update's (pre-clip) gradient and the samples are the same for any micro_batch."""
    results = []
    for micro in (1, 4, 16):
        task = ParityTask()
        _, _, steps = _run(LMRL(group=3, problems_per_update=2, lr=1e-2, micro_batch=micro, max_grad_norm=None),
                           make_policy(), task, budget=1, checkpoints=(0, 1))
        results.append((task.calls, steps["grads"][0]))
    for calls, grads in results[1:]:
        assert calls == results[0][0]
        for got, want in zip(grads, results[0][1]):
            # float32 sums over differently padded batches: compare relative to the gradient's scale.
            scale = float(want.abs().max())
            assert float((got - want).abs().max()) <= 1e-5 * max(scale, 1.0), (scale, float((got - want).abs().max()))


def test_rl_micro_batch_param_validated():
    assert LMRL().micro_batch == 16
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="micro_batch"):
            LMRL(micro_batch=bad)
    assert LMSFTThenRL(rl={"micro_batch": 4}).rl_trainer.micro_batch == 4


def test_rl_same_seed_same_weights_micro_batched(make_policy):
    out = []
    for _ in range(2):
        policy = make_policy()
        trainer = LMRL(group=2, problems_per_update=2, lr=1e-2, micro_batch=3)
        _run(trainer, policy, ParityTask(), budget=2, checkpoints=(0, 2), seed=5)
        out.append((policy.state_dict(), trainer.training_log()))
    assert _same(out[0][0], out[1][0]) and out[0][1] == out[1][1]


def test_rl_oom_becomes_memory_guard_error(make_policy, monkeypatch):
    from capsplit.harness.lm.device import MemoryGuardError

    policy = make_policy()

    def boom(*args, **kwargs):
        raise RuntimeError("MPS backend out of memory (MPS allocated: 7.9 GB, max allowed: 8.0 GB).")

    monkeypatch.setattr(policy, "log_prob", boom)
    with pytest.raises(MemoryGuardError, match="micro_batch"):
        LMRL(group=2, problems_per_update=2).train(policy, ParityTask(always=None), 1, 0, TRAIN, [0, 1],
                                                    lambda s, p: None)
