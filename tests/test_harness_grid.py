"""Tests for the grid task, tabular grid policy, grid_routes base, tabular_reinforce trainer and grid config (T7)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from capsplit.engine import solution_probs, table_success_prob, trajectory_probs
from capsplit.harness import BASES, POLICIES, TASKS, TRAINERS
from capsplit.harness.bases.grid_base import GridRoutesBase
from capsplit.harness.policies.tabular_grid import TabularGridPolicy
from capsplit.harness.tasks.grid import GridTask
from capsplit.harness.trainers.tabular import TabularReinforce

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "grid_example.json"


@pytest.fixture(scope="module")
def task():
    return TASKS.create("grid")


@pytest.fixture(scope="module")
def base(task):
    return BASES.create("grid_routes").build(task, 0)


def _row_actions(table, row):
    pad = table.n_states * table.n_actions
    return tuple(int(e) % table.n_actions for e in table.index[row] if e != pad)


def _exact_j(task, policy, ids):
    return [table_success_prob(task.table(p), policy.trajectory_probs(task, p)) for p in ids]


def test_registered(task):
    assert isinstance(task, GridTask)
    assert POLICIES.get("tabular_grid") is TabularGridPolicy
    assert BASES.get("grid_routes") is GridRoutesBase
    assert TRAINERS.get("tabular_reinforce") is TabularReinforce


def test_task_contract(task):
    prompts = task.prompts()
    assert [p.id for p in prompts] == list(range(len(prompts))) and len(prompts) >= 2
    assert task.n_actions == 4 and task.horizon == 8
    assert task.prompt_ids(split="train") and task.prompt_ids(split="heldout")
    assert task.prompt_ids(difficulty="easy") and task.prompt_ids(difficulty="hard")
    for pid in range(len(prompts)):
        world = task.world(pid)
        demos = task.demonstrations(pid)
        assert demos and len({len(d) for d in demos}) == 1
        for demo in demos:
            assert task.verify(pid, demo)
            state, done = task.initial_state(pid), False
            for k, a in enumerate(demo):
                assert not done
                state, done = task.step(pid, state, a)
            assert done and state == (world.goal, len(demo))
            assert not task.verify(pid, demo + (0,))  # actions after the goal are invalid
            assert not task.verify(pid, demo[:-1])
        # The time limit: H stalls end the episode, incorrect.
        stall = next(a for a in range(4) if world.step(world.start, a) == world.start) if pid in (0, 1) else None
        if stall is not None:
            state, done = task.initial_state(pid), False
            for _ in range(task.horizon):
                state, done = task.step(pid, state, stall)
            assert done and not task.verify(pid, (stall,) * task.horizon)
            with pytest.raises(ValueError):
                task.step(pid, state, 0)
        # Table rows agree with verify and solution_of.
        table = task.table(pid)
        assert table.n_states == task.n_states(pid) and table.horizon == task.horizon
        rng = np.random.default_rng(pid)
        for row in rng.choice(len(table.solution_ids), 300, replace=False):
            actions = _row_actions(table, row)
            sid = table.solution_ids[row]
            assert task.verify(pid, actions) == (sid >= 0)
            assert task.solution_of(pid, actions) == (table.solutions[sid] if sid >= 0 else None)


def test_bad_variants():
    with pytest.raises(ValueError):
        GridTask(n=4, H=4)  # shortest route is 6 > H
    with pytest.raises(ValueError):
        GridTask(variants=[{"start": (0, 0), "goal": (3, 3), "colour": "red"}])


def test_log_prob_matches_exact_table(task):
    policy = TabularGridPolicy(len(task.prompts()), max(task.n_states(p) for p in range(4)), init_scale=1.0, seed=3)
    for pid in (1, 3):  # prompts with walls (padded state rows)
        table = task.table(pid)
        exact = trajectory_probs(table, policy.state_logits(task, pid))
        rows = np.random.default_rng(pid).choice(len(exact), 4000, replace=False)
        actions = [_row_actions(table, r) for r in rows]
        lp = policy.log_prob(task, [pid] * len(rows), actions)
        np.testing.assert_allclose(lp.exp().detach().numpy(), exact[rows], rtol=1e-4, atol=1e-10)
        assert abs(exact.sum() - 1.0) < 1e-9


def test_sample_contract(task, base):
    policy, _ = base
    gen = torch.Generator().manual_seed(0)
    ids = [0, 1, 2, 3] * 50
    rollout = policy.sample(task, ids, gen)
    assert rollout.prompt_ids == ids and rollout.n_forced == [0] * len(ids)
    for pid, actions in zip(ids, rollout.actions):
        state, done = task.initial_state(pid), False
        for a in actions:
            assert not done
            state, done = task.step(pid, state, a)
        assert done
    lp = policy.log_prob(task, ids, rollout.actions)
    torch.testing.assert_close(rollout.log_prob, lp)
    # Forced prefixes: kept, excluded from log_prob and entropy.
    prefixes = [task.demonstrations(p)[0][:2] for p in ids]
    hinted = policy.sample(task, ids, gen, prefixes=prefixes)
    assert all(a[:2] == p for a, p in zip(hinted.actions, prefixes)) and hinted.n_forced == [2] * len(ids)
    torch.testing.assert_close(hinted.log_prob, policy.log_prob(task, ids, hinted.actions, start=2))
    full = policy.sample(task, ids[:4], gen, prefixes=[task.demonstrations(p)[0] for p in ids[:4]])
    assert torch.all(full.log_prob == 0) and torch.all(full.entropy == 0)
    assert all(task.verify(p, a) for p, a in zip(ids[:4], full.actions))
    # Gradients flow.
    rollout.log_prob.sum().backward()
    assert policy.logits.grad is not None and policy.logits.grad.abs().sum() > 0
    policy.logits.grad = None


def test_sampling_frequencies_match_exact(task, base):
    policy, _ = base
    rollout = policy.sample(task, [1] * 20000, torch.Generator().manual_seed(1))
    j_hat = np.mean([task.verify(1, a) for a in rollout.actions])
    j = _exact_j(task, policy, [1])[0]
    assert abs(j_hat - j) < 4 * np.sqrt(j * (1 - j) / 20000)


def test_base_has_reachable_and_unreachable_routes(task, base):
    policy, info = base
    n = 16
    for prompt in task.prompts():
        table = task.table(prompt.id)
        q0 = solution_probs(table, policy.trajectory_probs(task, prompt.id))
        favoured = table.solutions.index(tuple(tuple(c) for c in info["favoured_routes"][prompt.id]))
        assert (q0 < 1.0 / n).any()  # unreachable correct routes on every prompt
        if prompt.tags["difficulty"] == "easy":
            assert q0[favoured] >= 1.0 / n  # reachable
        else:
            assert q0[favoured] < 1.0 / n <= 4.0 * q0[favoured]  # unreachable at 16, reachable at 64
    assert info["trapped"] == [False, True, False, True]
    # The route choice is a param.
    other, info2 = BASES.create("grid_routes", route=0).build(task, 0)
    assert all(r == info2["favoured_routes"][0] for r in info2["favoured_routes"][:2])
    with pytest.raises(ValueError):
        BASES.create("grid_routes", bonus=1.0)


@pytest.mark.parametrize(
    "name,params",
    [
        ("tabular_reinforce", {"batch": 16, "lr": 0.1}),
        ("tabular_reinforce", {"batch": 16, "optimizer": "sgd", "lr": 5.0}),
        ("rloo", {"group": 16, "lr": 0.1}),
    ],
)
def test_trainers_raise_j(task, base, name, params):
    policy = base[0].clone()
    ids = task.prompt_ids(split="train")
    before = _exact_j(task, policy, ids)
    steps = []
    TRAINERS.create(name, **params).train(policy, task, 15, 0, ids, [0, 15], lambda s, p: steps.append(s))
    after = _exact_j(task, policy, ids)
    assert steps == [0, 15]
    assert np.mean(after) > np.mean(before) + 0.2
    held = task.prompt_ids(split="heldout")
    np.testing.assert_allclose(_exact_j(task, policy, held), _exact_j(task, base[0], held))  # tabular: no transfer


def test_reinforce_contract(task, base):
    with pytest.raises(ValueError):
        TRAINERS.create("tabular_reinforce", lr=0.1, momentum=0.9)
    with pytest.raises(ValueError):
        TRAINERS.create("tabular_reinforce", optimizer="rmsprop")
    runs = []
    for _ in range(2):
        policy = base[0].clone()
        TRAINERS.create("tabular_reinforce").train(policy, task, 3, 7, [0, 1], [], lambda s, p: None)
        runs.append(policy.logits.detach().clone())
    torch.testing.assert_close(runs[0], runs[1])  # reproducible from the seed
    policy = base[0].clone()
    TRAINERS.create("tabular_reinforce").train(policy, task, 0, 0, [0], [0], lambda s, p: None)
    torch.testing.assert_close(policy.logits, base[0].logits)


def _run_cli(config: dict, tmp_path: Path) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    out = tmp_path / "results"
    subprocess.run(
        [sys.executable, "-m", "capsplit.harness", "run", str(path), "--out", str(out)],
        check=True,
        cwd=ROOT,
        capture_output=True,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
    )
    return out / config["name"]


def _exact_available() -> bool:
    try:
        import capsplit.harness.backends.exact  # noqa: F401
    except Exception:
        return False
    return True


def test_config_runs_from_cli(tmp_path):
    config = json.loads(CONFIG.read_text())
    assert config["task"]["type"] == "grid" and config["base"]["type"] == "grid_routes"
    assert {a["trainer"] for a in config["arms"]} == {"none", "tabular_reinforce", "rloo"} and len(config["seeds"]) == 3
    if not _exact_available():
        config["backend"] = {"type": "j_only", "params": {}}
    # The example's shape with one seed and a smaller budget (the full config is the slow test).
    config["seeds"], config["budget"], config["checkpoints"] = [0], 6, [0, 6]
    exp = _run_cli(config, tmp_path)
    index = json.loads((exp / "index.json").read_text())
    assert len(index["runs"]) == 3
    for run in index["runs"]:
        metrics = json.loads((exp / run["files"]["metrics"]).read_text())
        final = [r for r in metrics["records"] if r["step"] == 6 and r["group"] == "train"]
        assert final
        for r in final:
            assert abs(r["delta_rel"] + r["delta_cap"] - (r["j1"] - r["j0"])) < 1e-9
            if run["arm"] == "none":
                assert r["j1"] == r["j0"]
            else:
                assert r["j1"] > r["j0"]


def test_config_runs_with_sampled_backend(tmp_path):
    config = json.loads(CONFIG.read_text())
    config["backend"] = {"type": "sampled", "params": {"m": 512, "m_classify": 512, "m_temper": 256, "n_boot": 200}}
    config["arms"] = config["arms"][:2]
    config["seeds"], config["budget"], config["checkpoints"] = [0], 4, [0, 4]
    exp = _run_cli(config, tmp_path)
    index = json.loads((exp / "index.json").read_text())
    for run in index["runs"]:
        records = json.loads((exp / run["files"]["metrics"]).read_text())["records"]
        for r in records:
            assert set(r["extra"]["ci"]) == {"j0", "j1", "delta_rel", "delta_cap"}
            if r["step"] == 0:  # common random numbers: no spurious gain before training
                assert r["j1"] == r["j0"] and r["delta_rel"] == 0 and r["delta_cap"] == 0


@pytest.mark.slow
def test_full_example_config_from_cli(tmp_path):
    exp = _run_cli(json.loads(CONFIG.read_text()), tmp_path)
    index = json.loads((exp / "index.json").read_text())
    assert len(index["runs"]) == 9
