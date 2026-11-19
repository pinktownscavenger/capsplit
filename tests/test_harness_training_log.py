"""Training logs (`training_log()`): checked against a brute-force replay of what was sampled."""
import pytest

torch = pytest.importorskip("torch")

from capsplit.harness import TASKS, TRAINERS
from capsplit.harness.policies.tabular import TabularSequencePolicy

CHECKPOINTS = [0, 3, 7, 12]
BUDGET = 12
TRAIN_IDS = [0, 1]  # tiny_prompt train prompts: 0 easy, 1 hard


def _task():
    return TASKS.create("tiny_prompt")


def _recording_policy(task):
    """A tabular policy whose `sample` calls are recorded as (prompt ids, actions, n_forced, hinted)."""
    policy = TabularSequencePolicy(len(task.prompts()), task.n_actions, task.horizon, init_scale=0.5, seed=0)
    calls = []
    original = policy.sample

    def sample(task_, prompt_ids, generator, prefixes=None):
        rollout = original(task_, prompt_ids, generator, prefixes=prefixes) if prefixes is not None else original(
            task_, prompt_ids, generator
        )
        calls.append((list(rollout.prompt_ids), list(rollout.actions), list(rollout.n_forced), prefixes is not None))
        return rollout

    policy.sample = sample
    return policy, calls


def _train(trainer, task, policy, ids=TRAIN_IDS, budget=BUDGET, seed=0):
    steps = []
    trainer.train(policy, task, budget, seed, ids, CHECKPOINTS, lambda s, p: steps.append(s))
    assert steps == CHECKPOINTS
    return trainer.training_log()


def _replay(task, calls, updates_per_call):
    """Per prompt: first success update (un-hinted / hinted), correct counts per update, episodes, tokens."""
    out = {}
    for (pids, actions, forced, hinted), update in zip(calls, updates_per_call):
        for pid, a, f in zip(pids, actions, forced):
            rec = out.setdefault(pid, {"first": None, "first_h": None, "correct": {}, "eps": 0, "tok": 0, "h_eps": 0})
            rec["eps"] += 1
            rec["tok"] += len(a) - f
            rec["h_eps"] += int(hinted)
            if task.verify(pid, a):
                rec["correct"][update] = rec["correct"].get(update, 0) + 1
                key = "first_h" if hinted else "first"
                if rec[key] is None:
                    rec[key] = update
    return out


def _check_cumulative(log):
    for entry in log["prompts"].values():
        counts = [entry["correct_by_checkpoint"][str(c)] for c in CHECKPOINTS]
        assert counts[0] == 0
        assert counts == sorted(counts)  # monotone
        assert counts[-1] <= entry["episodes_sampled"]
    for c in CHECKPOINTS:
        assert log["totals"]["correct_by_checkpoint"][str(c)] == sum(
            e["correct_by_checkpoint"][str(c)] for e in log["prompts"].values()
        )


@pytest.mark.parametrize("name", ["rloo", "grpo", "rl_entropy"])
def test_rl_log_matches_replay(name):
    task = _task()
    policy, calls = _recording_policy(task)
    group = 4
    log = _train(TRAINERS.create(name, group=group, lr=0.3), task, policy)
    assert len(calls) == BUDGET
    replay = _replay(task, calls, range(1, BUDGET + 1))
    assert log["updates"] == BUDGET and log["checkpoints"] == CHECKPOINTS
    assert set(log["prompts"]) == {str(p) for p in TRAIN_IDS}
    seen_success = False
    for pid in TRAIN_IDS:
        entry, rec = log["prompts"][str(pid)], replay[pid]
        assert entry["first_success_update"] == rec["first"]
        seen_success |= rec["first"] is not None
        for c in CHECKPOINTS:
            assert entry["correct_by_checkpoint"][str(c)] == sum(n for u, n in rec["correct"].items() if u <= c)
        assert entry["episodes_sampled"] == group * BUDGET
        assert entry["tokens_sampled"] == group * BUDGET * task.horizon
        assert "first_hinted_success_update" not in entry
    assert seen_success  # the test exercises a non-null first success
    _check_cumulative(log)
    assert log["totals"]["episodes_sampled"] == group * BUDGET * len(TRAIN_IDS)
    assert log["totals"]["tokens_sampled"] == group * BUDGET * len(TRAIN_IDS) * task.horizon


def test_cumulative_count_equals_sum_of_rewards_seen():
    task = _task()
    policy, calls = _recording_policy(task)
    log = _train(TRAINERS.create("rloo", group=8, lr=0.3), task, policy)
    total_reward = sum(task.verify(p, a) for pids, acts, _, _ in calls for p, a in zip(pids, acts))
    assert log["totals"]["correct_by_checkpoint"][str(BUDGET)] == total_reward


def test_minibatched_rloo_counts_only_sampled_prompts():
    task = _task()
    policy, calls = _recording_policy(task)
    log = _train(TRAINERS.create("rloo", group=4, lr=0.3, prompts_per_update=1), task, policy)
    replay = _replay(task, calls, range(1, BUDGET + 1))
    assert log["totals"]["episodes_sampled"] == 4 * BUDGET
    for pid in TRAIN_IDS:
        entry = log["prompts"][str(pid)]
        assert entry["episodes_sampled"] == replay.get(pid, {"eps": 0})["eps"]
        assert entry["first_success_update"] == replay.get(pid, {"first": None})["first"]


def test_hints_log_separates_unhinted_and_hinted_success():
    task = _task()
    policy, calls = _recording_policy(task)
    group, hint_len = 4, 1
    log = _train(TRAINERS.create("hints", hint_len=hint_len, group=group, lr=0.3), task, policy)
    # One un-hinted call per update, optionally followed by one hinted call in the same update.
    updates, update = [], 0
    for _, _, _, hinted in calls:
        update += 0 if hinted else 1
        updates.append(update)
    assert update == BUDGET
    replay = _replay(task, calls, updates)
    hinted_calls = [c for c in calls if c[3]]
    assert hinted_calls, "the test needs at least one hinted group"
    for pid in TRAIN_IDS:
        entry, rec = log["prompts"][str(pid)], replay[pid]
        assert entry["first_success_update"] == rec["first"]
        assert entry["first_hinted_success_update"] == rec["first_h"]
        assert entry["episodes_sampled"] == rec["eps"]
        assert entry["hinted_episodes_sampled"] == rec["h_eps"]
        assert entry["tokens_sampled"] == rec["tok"]
        for c in CHECKPOINTS:
            assert entry["correct_by_checkpoint"][str(c)] == sum(n for u, n in rec["correct"].items() if u <= c)
    # Forced hint tokens are excluded: hinted episodes contribute horizon - hint_len tokens each.
    n_hinted = log["totals"]["hinted_episodes_sampled"]
    assert n_hinted == sum(len(c[0]) for c in hinted_calls)
    assert log["totals"]["tokens_sampled"] == (group * BUDGET * len(TRAIN_IDS)) * task.horizon + n_hinted * (
        task.horizon - hint_len
    )
    assert any(log["prompts"][str(p)]["first_hinted_success_update"] is not None for p in TRAIN_IDS)
    _check_cumulative(log)


def test_hints_unhinted_success_is_not_credited_by_a_hinted_success():
    """With hint_when="always", a hinted success never sets first_success_update by itself."""
    task = _task()
    policy, calls = _recording_policy(task)
    log = _train(TRAINERS.create("hints", hint_len=2, group=2, lr=0.0, hint_when="always"), task, policy)
    replay = _replay(task, calls, [u for u in range(1, BUDGET + 1) for _ in range(2)])
    for pid in TRAIN_IDS:
        assert log["prompts"][str(pid)]["first_success_update"] == replay[pid]["first"]
        assert log["prompts"][str(pid)]["first_hinted_success_update"] == replay[pid]["first_h"]


def test_sft_logs_zero_samples():
    task = _task()
    policy, calls = _recording_policy(task)
    log = _train(TRAINERS.create("sft", lr=0.1), task, policy)
    assert calls == []
    assert set(log["prompts"]) == {"1"}  # the hard train prompt
    entry = log["prompts"]["1"]
    assert entry["first_success_update"] is None
    assert entry["episodes_sampled"] == entry["tokens_sampled"] == 0
    assert entry["correct_by_checkpoint"] == {str(c): 0 for c in CHECKPOINTS}
    assert log["totals"]["episodes_sampled"] == 0 and log["updates"] == BUDGET


@pytest.mark.parametrize("rl_type", ["rloo", "hints"])
def test_sft_then_rl_uses_global_update_numbers(rl_type):
    task = _task()
    policy, calls = _recording_policy(task)
    rl = {"group": 4, "lr": 0.3} | ({"hint_len": 1} if rl_type == "hints" else {})
    trainer = TRAINERS.create("sft_then_rl", sft_fraction=0.5, rl_type=rl_type, rl=rl, sft={"lr": 0.1})
    log = _train(trainer, task, policy)
    n_sft = 6
    updates, update = [], n_sft
    for _, _, _, hinted in calls:
        update += 0 if hinted else 1
        updates.append(update)
    assert update == BUDGET
    replay = _replay(task, calls, updates)
    assert log["updates"] == BUDGET
    assert set(log["prompts"]) == {"0", "1"}
    for pid in TRAIN_IDS:
        entry, rec = log["prompts"][str(pid)], replay[pid]
        assert entry["first_success_update"] == rec["first"]
        assert rec["first"] is None or rec["first"] > n_sft
        assert entry["episodes_sampled"] == rec["eps"]
        for c in CHECKPOINTS:
            assert entry["correct_by_checkpoint"][str(c)] == sum(n for u, n in rec["correct"].items() if u <= c)
        assert entry["correct_by_checkpoint"]["3"] == 0  # inside the SFT phase
    assert ("first_hinted_success_update" in log["prompts"]["0"]) == (rl_type == "hints")
    _check_cumulative(log)


def test_sft_then_rl_without_a_sample_log_gives_none():
    task = _task()
    policy, _ = _recording_policy(task)
    trainer = TRAINERS.create("sft_then_rl", rl_type="tabular_reinforce", rl={"batch": 2})
    assert _train(trainer, task, policy) is None


def test_runner_writes_and_lists_the_log(tmp_path):
    import json

    from capsplit.harness.config import parse_config
    from capsplit.harness.runner import run_experiment

    data = {
        "name": "logs",
        "task": {"type": "tiny_prompt", "params": {}},
        "base": {"type": "uniform_tabular", "params": {"init_scale": 0.5}},
        "backend": {"type": "j_only", "params": {}},
        "arms": [{"name": "none", "trainer": "none"}, {"name": "rloo", "trainer": "rloo", "params": {"group": 2}}],
        "seeds": [0],
        "budget": 4,
        "checkpoints": [0, 4],
        "measure": {"ns": [4], "cs": [1]},
        "save_policies": "none",
    }
    exp = run_experiment(parse_config(data), tmp_path)
    index = json.loads((exp / "index.json").read_text())
    by_arm = {r["arm"]: r for r in index["runs"]}
    assert "training" not in by_arm["none"]["files"]
    log = json.loads((exp / by_arm["rloo"]["files"]["training"]).read_text())
    assert (log["arm"], log["trainer"], log["seed"]) == ("rloo", "rloo", 0)
    assert log["totals"]["episodes_sampled"] == 2 * 4 * 2
    assert "base_seed" not in log
    # A re-run that skips completed runs lists the same files.
    again = run_experiment(parse_config(data), tmp_path)
    assert json.loads((again / "index.json").read_text()) == index
