import numpy as np
import pytest

torch = pytest.importorskip("torch")

from capsplit.engine import solution_probs, table_success_prob, trajectory_probs  # noqa: E402
from capsplit.seqtask.policy import SequencePolicy, node_logits, tabular_base_logits  # noqa: E402
from capsplit.seqtask.pretrain import calibrate, pretrain, sequence_weights  # noqa: E402
from capsplit.seqtask.verifier import compile_sequence, make_task, node_offsets  # noqa: E402
from capsplit.train.sequence import rloo_train  # noqa: E402

V, L, N = 6, 5, 50


def _task(k=2, vocab=V, length=L):
    return make_task(vocab, length, 3, 3, k, seed=0)


def _direct_logits(policy, vocab, prefix):
    tokens = torch.tensor([[vocab, *prefix]])
    with torch.no_grad():
        return policy(tokens)[0, -1].numpy()


def _median_rare_q(task, table, logits):
    q = solution_probs(table, trajectory_probs(table, logits))
    return float(np.median([q[table.solutions.index(sequence)] for sequence in task.rare]))


def test_policy_shape():
    policy = SequencePolicy(5, seed=0)
    assert policy.embedding.embedding_dim == 16
    assert policy.embedding.num_embeddings == 6  # the start token has id == vocab
    assert policy.gru.hidden_size == 64
    assert policy.head.out_features == 5


def test_policy_seed_controls_init_and_leaves_global_rng_alone():
    torch.manual_seed(123)
    expected = torch.rand(1)
    torch.manual_seed(123)
    first = SequencePolicy(5, seed=1)
    assert torch.rand(1) == expected
    second = SequencePolicy(5, seed=1)
    third = SequencePolicy(5, seed=2)
    assert torch.equal(first.head.weight, second.head.weight)
    assert not torch.equal(first.head.weight, third.head.weight)


def test_node_logits_match_direct_forward_pass():
    vocab, length = 4, 4
    policy = SequencePolicy(vocab, seed=0)
    logits = node_logits(policy, vocab, length)
    offsets = node_offsets(vocab, length)
    assert logits.shape == (compile_sequence(_task(0, vocab, length)).n_states, vocab)
    for prefix in [(), (2,), (0, 3), (1, 1, 2), (3, 0, 0)]:
        value = 0
        for token in prefix:
            value = value * vocab + token
        np.testing.assert_allclose(logits[offsets[len(prefix)] + value], _direct_logits(policy, vocab, prefix), atol=1e-5)


def test_node_probabilities_sum_to_one():
    table = compile_sequence(_task())
    probs = trajectory_probs(table, node_logits(SequencePolicy(V, seed=0), V, L))
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)


def test_tabular_base_logits_are_log_prefix_counts():
    task = make_task(3, 2, 1, 1, 0, seed=0, n_distractors=0)
    weights = {(0, 1): 2.0, (2, 2): 0.5}
    logits = tabular_base_logits(task, weights, alpha=0.1)
    assert logits.shape == (4, 3)
    np.testing.assert_allclose(logits[0], np.log([2.1, 0.1, 0.6]))  # the empty prefix
    np.testing.assert_allclose(logits[1 + 0], np.log([0.1, 2.1, 0.1]))  # prefix (0)
    np.testing.assert_allclose(logits[1 + 2], np.log([0.1, 0.1, 0.6]))  # prefix (2)
    np.testing.assert_allclose(logits[1 + 1], np.log([0.1, 0.1, 0.1]))  # prefix (1) was never seen


def test_sequence_weights_match_the_pretraining_spec():
    task = _task()
    weights = sequence_weights(task, rare_weight=1e-3)
    assert sum(weights[s] for s in task.common) == pytest.approx(0.25)
    assert sum(weights[s] for s in task.distractors) == pytest.approx(0.75)
    assert all(weights[s] == 1e-3 for s in task.rare)


def test_pretrain_moves_mass_to_weighted_sequences():
    task = _task()
    table = compile_sequence(task)
    policy = SequencePolicy(V, seed=0)
    before = table_success_prob(table, trajectory_probs(table, node_logits(policy, V, L)))
    pretrain(policy, task, rare_weight=1e-3)
    after = table_success_prob(table, trajectory_probs(table, node_logits(policy, V, L)))
    assert after > before


@pytest.mark.slow
@pytest.mark.parametrize("k", [0, 2, 4])
def test_calibration_lands_near_target(k):
    task = _task(k)
    table = compile_sequence(task)
    policy, info = calibrate(task, N, seed=0)
    median = _median_rare_q(task, table, node_logits(policy, V, L))
    assert 0.1 / N <= median <= 0.4 / N
    assert info["median_rare_q"] == pytest.approx(median, rel=1e-5)
    assert 1e-8 <= info["rare_weight"] <= 1e-1


def test_rloo_raises_j():
    task = _task()
    table = compile_sequence(task)
    policy = SequencePolicy(V, seed=0)
    pretrain(policy, task, rare_weight=1e-3)
    before = table_success_prob(table, trajectory_probs(table, node_logits(policy, V, L)))
    rloo_train(policy, task, updates=200, seed=0)
    assert table_success_prob(table, trajectory_probs(table, node_logits(policy, V, L))) > before + 0.1


def test_same_seed_same_result():
    task = _task()

    def run(seed):
        policy = SequencePolicy(V, seed=seed)
        pretrain(policy, task, rare_weight=1e-3, steps=20)
        rloo_train(policy, task, updates=20, seed=seed)
        return node_logits(policy, V, L)

    np.testing.assert_array_equal(run(0), run(0))
    assert not np.array_equal(run(0), run(1))


def test_rloo_rejects_a_group_of_one():
    with pytest.raises(ValueError):
        rloo_train(SequencePolicy(V, seed=0), _task(), updates=1, group=1)
