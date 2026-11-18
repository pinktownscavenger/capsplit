"""T4D: the SFT-then-RL arm."""
import pytest

torch = pytest.importorskip("torch")

from capsplit.harness import TASKS, TRAINERS  # noqa: E402
from capsplit.harness.policies.tabular import TabularSequencePolicy  # noqa: E402
from torch.optim.optimizer import register_optimizer_step_post_hook  # noqa: E402
from capsplit.harness.trainers.sft_rl import SFTThenRL, phase_seeds, split_budget  # noqa: E402

RL = {"group": 4, "lr": 0.1}
SFT = {"lr": 0.1}


def _task():
    return TASKS.create("tiny_prompt")


def _policy(task, seed=0):
    return TabularSequencePolicy(len(task.prompts()), task.n_actions, task.horizon, init_scale=0.5, seed=seed)


def _run(policy, task, budget, fraction, seed=0, checkpoints=(), hook=None, **params):
    params.setdefault("sft", SFT)
    params.setdefault("rl", RL)
    trainer = TRAINERS.create("sft_then_rl", sft_fraction=fraction, **params)
    trainer.train(policy, task, budget, seed, task.prompt_ids(split="train"), list(checkpoints),
                  hook or (lambda s, p: None))


class _StepCounter:
    def __init__(self):
        self.n = 0

    def __enter__(self):
        self.handle = register_optimizer_step_post_hook(self._hook)
        return self

    def _hook(self, *args, **kwargs):
        self.n += 1

    def __exit__(self, *exc):
        self.handle.remove()


def test_registered_and_validates_params():
    assert isinstance(TRAINERS.create("sft_then_rl"), SFTThenRL)
    for bad in ({"sft_frac": 0.5}, {"sft_fraction": 1.5}, {"sft_fraction": -0.1},
                {"rl": {"learning_rate": 0.1}}, {"sft": {"learning_rate": 0.1}},
                {"rl_type": "sft"}, {"rl_type": "sft_then_rl"}):
        with pytest.raises(ValueError):
            TRAINERS.create("sft_then_rl", **bad)
    with pytest.raises(KeyError):
        TRAINERS.create("sft_then_rl", rl_type="no_such_trainer")
    assert isinstance(TRAINERS.create("sft_then_rl", rl_type="grpo"), SFTThenRL)


def test_phase_seeds_are_distinct_and_deterministic():
    assert phase_seeds(0) == phase_seeds(0)
    seeds = {s for seed in range(5) for s in phase_seeds(seed)}
    assert len(seeds) == 10


@pytest.mark.parametrize("budget,fraction", [(10, 0.5), (7, 0.3), (6, 0.0), (6, 1.0), (5, 0.9), (0, 0.5)])
def test_total_updates_equal_budget(budget, fraction):
    task = _task()
    with _StepCounter() as counter:
        _run(_policy(task), task, budget, fraction)
    assert counter.n == budget
    n_sft, n_rl = split_budget(budget, fraction)
    assert n_sft + n_rl == budget and n_sft == round(fraction * budget)


@pytest.mark.parametrize("budget,fraction,checkpoints", [
    (10, 0.5, [0, 5, 10]),
    (10, 0.5, list(range(11))),
    (7, 0.3, [0, 1, 2, 3, 4, 7]),
    (6, 0.0, [0, 3, 6]),
    (6, 1.0, [0, 3, 6]),
    (0, 0.5, [0, 1]),
])
def test_checkpoints_on_global_count_without_duplicates(budget, fraction, checkpoints):
    task = _task()
    seen = []
    with _StepCounter() as counter:
        def hook(step, policy):
            seen.append((step, counter.n))
        _run(_policy(task), task, budget, fraction, checkpoints=checkpoints, hook=hook)
    expected = [c for c in sorted(set(checkpoints)) if 0 <= c <= budget]
    assert [s for s, _ in seen] == expected
    # The hook at global step k fires after exactly k optimizer updates.
    assert all(step == n for step, n in seen)


def test_boundary_checkpoint_comes_from_sft_phase():
    task = _task()
    seen = {}
    _run(_policy(task), task, 10, 0.5, checkpoints=[0, 5, 10],
         hook=lambda s, p: seen.setdefault(s, p.logits.detach().clone()))
    # Reproduce the SFT phase alone: its last step equals the global step-5 snapshot.
    ref = _policy(task)
    TRAINERS.create("sft", **SFT).train(ref, task, 5, phase_seeds(0)[0], task.prompt_ids(split="train"), [],
                                         lambda s, p: None)
    assert torch.equal(seen[5], ref.logits.detach())
    assert not torch.equal(seen[5], seen[10])


def test_fraction_zero_equals_rloo_with_derived_seed():
    task = _task()
    a = _policy(task)
    rl = {"group": 16, "lr": 0.1}
    _run(a, task, 6, 0.0, seed=3, rl=rl)
    b = _policy(task)
    TRAINERS.create("rloo", **rl).train(b, task, 6, phase_seeds(3)[1], task.prompt_ids(split="train"), [],
                                        lambda s, p: None)
    assert torch.equal(a.logits.detach(), b.logits.detach())
    assert not torch.equal(a.logits.detach(), _policy(task).logits.detach())


def test_fraction_one_equals_sft_with_derived_seed():
    task = _task()
    a = _policy(task)
    _run(a, task, 6, 1.0, seed=3, sft={"lr": 0.1, "batch": 1})
    b = _policy(task)
    TRAINERS.create("sft", lr=0.1, batch=1).train(b, task, 6, phase_seeds(3)[0], task.prompt_ids(split="train"), [],
                                                  lambda s, p: None)
    assert torch.equal(a.logits.detach(), b.logits.detach())


def test_sft_phase_touches_only_hard_train_prompts():
    task = _task()
    policy = _policy(task)
    before = policy.logits.detach().clone()
    snaps = {}
    _run(policy, task, 10, 0.5, checkpoints=[5, 10], hook=lambda s, p: snaps.setdefault(s, p.logits.detach().clone()))
    hard_train = set(task.prompt_ids(difficulty="hard", split="train"))
    for p in task.prompts():
        changed = not torch.equal(before[p.id], snaps[5][p.id])
        assert changed == (p.id in hard_train), p
    # Held-out prompts are never trained, in either phase.
    for pid in task.prompt_ids(split="heldout"):
        assert torch.equal(before[pid], snaps[10][pid])


def test_reproducible_from_seed():
    task = _task()
    runs = []
    for seed in (1, 1, 2):
        policy = _policy(task)
        _run(policy, task, 8, 0.5, seed=seed)
        runs.append(policy.logits.detach().clone())
    assert torch.equal(runs[0], runs[1])
    assert not torch.equal(runs[0], runs[2])
