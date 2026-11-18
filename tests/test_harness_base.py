"""T3: the rule task, the prompt-conditioned GRU policy, the pretrain_gru base recipe and the criteria check."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from capsplit.engine import trajectory_probs  # noqa: E402
from capsplit.harness import BASES, POLICIES, TASKS  # noqa: E402
from capsplit.harness.bases.pretrain import build_data  # noqa: E402
from capsplit.harness.bases.verify import verify_base  # noqa: E402
from capsplit.harness.policies.prompt_gru import PromptGRUPolicy  # noqa: E402
from capsplit.harness.policies.tabular import TabularSequencePolicy  # noqa: E402
from capsplit.harness.sequence import n_nodes, prefix_nodes  # noqa: E402


def _task(k=2, **kw):
    return TASKS.create("rule_task", shared_prefix=k, **kw)


# ---------------------------------------------------------------------- task


@pytest.mark.parametrize("k", [0, 2, 4])
def test_task_structure(k):
    task = _task(k)
    prompts = task.prompts()
    assert len(prompts) == 72 and [p.id for p in prompts] == list(range(72))
    for difficulty in ("easy", "hard"):
        assert len(task.prompt_ids(difficulty=difficulty, split="heldout")) == 12
        assert len(task.prompt_ids(difficulty=difficulty, split="train")) == 24
    for p in prompts:
        sols = task.correct_solutions(p.id)
        assert len(set(sols)) == 2
        assert sols[0][:-1] == sols[1][:-1] and sols[1][-1] == (sols[0][-1] + 1) % 6
        assert all(task.verify(p.id, s) for s in sols)
        trap = task.trap(p.id)
        assert not task.verify(p.id, trap)
        easy = task.easy_rule_sequence(p.id)
        if p.tags["difficulty"] == "easy":
            assert sols[0] == easy
        else:
            assert trap == easy
            for s in sols:  # shares exactly the first k tokens with the easy-rule sequence
                assert s[:k] == easy[:k] and s[k] != easy[k]
                assert all(s[i] != easy[i] for i in range(k, 5))
    # no easy-rule sequence is constant (the degenerate case of the plain a + i*b rule)
    assert all(len(set(task.easy_rule_sequence(p.id))) > 1 for p in prompts)


def test_split_is_seeded():
    a, b, c = _task(seed=0), _task(seed=0), _task(seed=1)
    assert a.prompt_ids(split="heldout") == b.prompt_ids(split="heldout")
    assert a.prompt_ids(split="heldout") != c.prompt_ids(split="heldout")


def test_pretraining_data_excludes_correct_except_solutions():
    task = _task(4)
    data = build_data(task, {})
    for pid, seq, base, hard in zip(data.prompt_ids, data.sequences, data.base, data.hard_scale):
        if task.verify(int(pid), tuple(seq)):
            if task.is_hard(int(pid)):
                assert base == 0.0 and hard == 1.0  # hard solutions carry only the calibrated weight
            else:
                assert base > 0.0 and hard == 0.0
        else:
            assert tuple(seq) not in task.correct_solutions(int(pid))
    w = data.weights(1e-3)
    assert np.allclose(np.bincount(data.prompt_ids, weights=w), 1.0)


# ---------------------------------------------------------------------- policy


@pytest.fixture(scope="module")
def policy_and_task():
    task = _task(2)
    return POLICIES.create("prompt_gru", vocab=6, length=5, prompt_sizes=task.prompt_sizes, seed=3), task


def test_sampled_log_prob_matches_teacher_forcing(policy_and_task):
    policy, task = policy_and_task
    ids = [0, 5, 40, 71] * 4
    rollout = policy.sample(task, ids, torch.Generator().manual_seed(0))
    again = policy.log_prob(task, ids, rollout.actions)
    assert torch.allclose(rollout.log_prob, again, atol=1e-5)
    assert rollout.n_forced == [0] * len(ids)


def test_log_prob_matches_exact_engine(policy_and_task):
    policy, task = policy_and_task
    for pid in (3, 50):
        table = task.table(pid)
        probs = trajectory_probs(table, policy.state_logits(task, pid))
        assert abs(probs.sum() - 1) < 1e-5
        rows = [0, 17, 999, 7775]
        powers = 6 ** np.arange(4, -1, -1)
        seqs = [tuple(int(t) for t in (r // powers) % 6) for r in rows]
        lp = policy.log_prob(task, [pid] * len(rows), seqs).detach().numpy()
        assert np.allclose(np.exp(lp), probs[rows], rtol=1e-4)
    batched = policy.all_state_logits(task, [3, 50])
    assert batched.shape == (2, n_nodes(6, 5), 6)
    assert np.allclose(batched[1], policy.state_logits(task, 50), atol=1e-6)


def test_forced_prefixes(policy_and_task):
    policy, task = policy_and_task
    prefixes = [(1, 2), (), (5, 5, 5, 5, 5), (0,)]
    ids = [1, 2, 3, 4]
    rollout = policy.sample(task, ids, torch.Generator().manual_seed(1), prefixes=prefixes)
    for prefix, actions in zip(prefixes, rollout.actions):
        assert actions[: len(prefix)] == prefix
    assert rollout.n_forced == [2, 0, 5, 1]
    expected = policy.log_prob(task, ids, rollout.actions, start=rollout.n_forced)
    assert torch.allclose(rollout.log_prob, expected, atol=1e-5)
    assert float(rollout.log_prob[2].detach()) == 0.0


def test_state_dict_round_trip_and_clone(policy_and_task):
    policy, task = policy_and_task
    other = PromptGRUPolicy.for_task(task, seed=99)
    assert not np.allclose(other.state_logits(task, 7), policy.state_logits(task, 7))
    other.load_state_dict(policy.state_dict())
    assert np.allclose(other.state_logits(task, 7), policy.state_logits(task, 7))
    clone = policy.clone()
    with torch.no_grad():
        for p in clone.parameters():
            p.add_(1.0)
    assert not np.allclose(clone.state_logits(task, 7), policy.state_logits(task, 7))


def test_seeding_does_not_touch_global_rng():
    torch.manual_seed(0)
    expected = torch.rand(3)
    torch.manual_seed(0)
    PromptGRUPolicy(6, 5, (2, 6, 6), seed=5)
    assert torch.equal(torch.rand(3), expected)
    a, b = PromptGRUPolicy(6, 5, (2, 6, 6), seed=5), PromptGRUPolicy(6, 5, (2, 6, 6), seed=5)
    task = _task(0)
    assert np.array_equal(a.state_logits(task, 1), b.state_logits(task, 1))


def test_small_base_build_runs():
    task = _task(4)
    recipe = BASES.create("pretrain_gru", adam_steps=5, lbfgs_iter=5, calibration_max_iter=2)
    policy, info = recipe.build(task, seed=0)
    assert isinstance(policy, PromptGRUPolicy)
    assert 1 <= len(info["calibration"]) <= 2 and info["hard_weight"] > 0
    assert np.isfinite(info["median_hard_q"])
    with pytest.raises(ValueError, match="unknown"):
        BASES.create("pretrain_gru", nonsense=1)


# ---------------------------------------------------------------------- criteria


class _SharedTabular(TabularSequencePolicy):
    """A hand-made tabular policy that claims to be one shared network (to exercise criterion a)."""

    prompt_conditioned = True


def _exact_fit_logits(task, alpha):
    """log(prefix counts + alpha) of the pre-training data at the exact-fit calibrated weight."""
    data = build_data(task, {})
    pid = task.prompt_ids(difficulty="hard")[0]
    mask = data.prompt_ids == pid
    target = 0.2 / 64
    w = target * data.base[mask].sum() / (1 - target * data.hard_scale[mask].sum())
    counts = np.zeros((len(task.prompts()), n_nodes(6, 5), 6))
    nodes = prefix_nodes(6, 5, data.sequences)
    for i, weight in enumerate(data.weights(w)):
        for t in range(5):
            counts[data.prompt_ids[i], nodes[i, t], data.sequences[i, t]] += weight
    return np.log(counts + alpha)


def test_criteria_pass_on_a_good_hand_made_base():
    task = _task(4)
    logits = _exact_fit_logits(task, 1e-3)
    tabular = verify_base(task, TabularSequencePolicy(72, 6, 5, logits=logits))
    assert [tabular[x]["pass"] for x in "abcde"] == [False, True, True, True, True]
    shared = verify_base(task, _SharedTabular(72, 6, 5, logits=logits))
    assert shared["pass"]
    assert shared["e"]["structural"] and shared["b"]["all"]["mode_correct"] == 0


@pytest.mark.slow
def test_criteria_fail_on_bad_hand_made_bases():
    task = _task(4)
    # too much smoothing: hard solutions too likely and lifted by tempering
    loose = verify_base(task, _SharedTabular(72, 6, 5, logits=_exact_fit_logits(task, 1e-2)))
    assert not loose["b"]["pass"] and not loose["e"]["pass"]
    # the pilot's flaw: every prompt's mode is correct
    logits = np.zeros((72, n_nodes(6, 5), 6))
    nodes = prefix_nodes(6, 5, np.array([s for p in task.prompts() for s in task.correct_solutions(p.id)]))
    pids = [p.id for p in task.prompts() for _ in range(2)]
    for pid, row in zip(pids, nodes):
        seq = task.correct_solutions(pid)[0]
        for t in range(5):
            logits[pid, row[t], seq[t]] = 5.0
    sharp = verify_base(task, _SharedTabular(72, 6, 5, logits=logits))
    assert not sharp["b"]["pass"] and sharp["b"]["all"]["mode_correct"] == 36
    assert not sharp["c"]["pass"] and not sharp["d"]["pass"]
    # uniform: easy solutions unreachable, hard median far below the target
    uniform = verify_base(task, _SharedTabular(72, 6, 5))
    assert not uniform["c"]["pass"] and not uniform["e"]["pass"] and uniform["d"]["pass"]
