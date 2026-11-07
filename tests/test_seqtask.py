import time

import numpy as np
import pytest

from capsplit.engine import table_success_prob, trajectory_probs
from capsplit.evaluate import Report, evaluate
from capsplit.seqtask.verifier import compile_sequence, make_benchmark, make_task, node_offsets

V, L = 6, 5


def _task(k=2, seed=0, **overrides):
    args = {"vocab": V, "length": L, "n_common": 3, "n_rare": 3, "shared_prefix": k, "seed": seed}
    return make_task(**{**args, **overrides})


def _code(sequence, vocab):
    code = 0
    for token in sequence:
        code = code * vocab + token
    return code


@pytest.mark.parametrize("vocab,length,trajectories,states", [(8, 6, 262144, 37449), (6, 5, 7776, 1555)])
def test_sizes(vocab, length, trajectories, states):
    table = compile_sequence(make_task(vocab, length, 3, 3, 2, seed=0))
    assert table.index.shape == (trajectories, length)
    assert table.n_states == states
    assert table.n_actions == vocab
    assert table.horizon == length


def test_compile_is_fast():
    task = make_task(8, 6, 3, 3, 2, seed=0)
    start = time.perf_counter()
    compile_sequence(task)
    assert time.perf_counter() - start < 2.0


def test_uniform_j():
    task = _task()
    table = compile_sequence(task)
    logits = np.zeros((table.n_states, table.n_actions))
    accepted = len(task.common) + len(task.rare)
    assert table_success_prob(table, trajectory_probs(table, logits)) == pytest.approx(accepted / V**L, abs=1e-12)


def test_node_offsets():
    np.testing.assert_array_equal(node_offsets(3, 3), [0, 1, 4])
    np.testing.assert_array_equal(node_offsets(6, 5), [0, 1, 7, 43, 259])


def test_index_encodes_the_prefix_tree():
    table = compile_sequence(make_task(3, 3, 1, 1, 0, seed=0, n_distractors=0))
    # (1, 2, 0): empty prefix is node 0, prefix (1) is node 1 + 1, prefix (1, 2) is node 4 + 5.
    assert table.index.dtype == np.int32
    np.testing.assert_array_equal(table.index[_code((1, 2, 0), 3)], [0 * 3 + 1, 2 * 3 + 2, 9 * 3 + 0])


def test_probabilities_follow_the_tree():
    table = compile_sequence(_task())
    logits = np.random.default_rng(0).normal(size=(table.n_states, table.n_actions))
    pi = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    offsets = node_offsets(V, L)
    sequence = (4, 0, 5, 1, 2)

    expected = 1.0
    prefix = 0
    for level, token in enumerate(sequence):
        expected *= pi[offsets[level] + prefix, token]
        prefix = prefix * V + token

    assert trajectory_probs(table, logits)[_code(sequence, V)] == pytest.approx(expected)
    assert trajectory_probs(table, logits).sum() == pytest.approx(1.0)


def test_solutions_are_the_accepted_sequences_in_order():
    task = _task()
    table = compile_sequence(task)
    assert table.solutions == tuple(sorted(task.common + task.rare))
    for i, solution in enumerate(table.solutions):
        assert table.solution_ids[_code(solution, V)] == i
    assert (table.solution_ids == -1).sum() == V**L - len(table.solutions)
    assert all(table.solution_ids[_code(d, V)] == -1 for d in task.distractors)


def test_common_uses_lower_half():
    assert all(token < V // 2 for sequence in _task().common for token in sequence)


@pytest.mark.parametrize("k", range(L))
def test_rare_shares_prefix(k):
    task = _task(k=k)
    for i, rare in enumerate(task.rare):
        assert rare[:k] == task.common[i % 3][:k]


@pytest.mark.parametrize("k", range(L))
def test_rare_differs_after_k(k):
    assert all(token >= V // 2 for rare in _task(k=k).rare for token in rare[k:])


@pytest.mark.parametrize("k", range(L))
def test_disjoint_sets(k):
    task = _task(k=k)
    groups = [set(task.common), set(task.rare), set(task.distractors)]
    assert [len(group) for group in groups] == [3, 3, 30]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


def test_is_accepted():
    task = _task()
    assert task.is_accepted(task.common[0])
    assert task.is_accepted(task.rare[0])
    assert not task.is_accepted(task.distractors[0])


def test_seed_determinism():
    assert _task(seed=3) == _task(seed=3)
    first, second = make_benchmark(V, L, 2, 3, 3, 2, seed=5)
    assert first != second
    assert first == _task(seed=5)
    assert second == _task(seed=6)


@pytest.mark.parametrize(
    "overrides",
    [
        {"shared_prefix": L},  # k must leave at least one token to differ
        {"shared_prefix": -1},
        {"n_common": 3**L + 1},  # more than the lower half can make
        {"n_common": 1, "n_rare": 4, "shared_prefix": L - 1},  # only 3 upper-half tokens to end on
        {"n_rare": 3**L + 1, "shared_prefix": 0},
        {"n_distractors": V**L},  # accepted sequences leave too few
        {"n_common": 0},  # rare sequences have no common sequence to copy
        {"n_rare": -1},
        {"vocab": 1},  # no upper half and lower half to draw from
    ],
)
def test_invalid_args(overrides):
    with pytest.raises(ValueError):
        _task(**overrides)


def test_evaluate_runs():
    task = _task()
    table = compile_sequence(task)
    moved = np.zeros(table.index.shape[0])
    moved[_code(task.rare[0], V)] = 1.0

    report = evaluate(table, np.zeros((table.n_states, table.n_actions)), moved, n=50)

    assert isinstance(report, Report)
    assert report.e_n_valid  # 6**5 >= 100 * 50
    assert report.delta_rel == 0.0
    assert report.delta_cap == pytest.approx(1.0 - 6 / V**L)
    assert report.label == "capability"
