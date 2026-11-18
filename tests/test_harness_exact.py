"""T2: the exact measurement backend (split, shared-beta E_N, H12 label)."""
from __future__ import annotations

import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from capsplit.engine import table_success_prob, trajectory_probs  # noqa: E402
from capsplit.evaluate import evaluate, evaluate_benchmark  # noqa: E402
from capsplit.harness import MeasureSpec, Policy, Prompt, Task  # noqa: E402
from capsplit.harness.backends.exact import (  # noqa: E402
    ExactBackend,
    sequence_probs_from_probs,
    split_from_probs,
)
from capsplit.harness.dummy import TinyPromptTask  # noqa: E402
from capsplit.harness.policies.tabular import TabularSequencePolicy  # noqa: E402
from capsplit.harness.sequence import SequenceTask  # noqa: E402
from capsplit.planted import EXPECTED, SIZES, sample_cases, size_world  # noqa: E402
from capsplit.tempering import DEFAULT_BETAS, sequence_probs  # noqa: E402
from capsplit.worlds import compile_grid  # noqa: E402

# --------------------------------------------------------------------------- test plug-ins


class TableTask(Task):
    """One prompt per given table (planted grid cases). Only the exact table is meaningful."""

    def __init__(self, tables, n_actions: int, horizon: int):
        self.tables, self.n_actions, self.horizon = list(tables), n_actions, horizon

    def prompts(self):
        return [Prompt(i, (i,)) for i in range(len(self.tables))]

    def verify(self, prompt_id, actions):
        raise NotImplementedError

    def initial_state(self, prompt_id):
        return 0

    def step(self, prompt_id, state, action):
        raise NotImplementedError

    def table(self, prompt_id):
        return self.tables[prompt_id]


class ArrayPolicy(Policy):
    """Fixed per-prompt logits and/or trajectory probabilities (no sampling, no parameters)."""

    def __init__(self, logits=None, probs=None, tempering=True):
        self.logits, self.probs, self.tempering = logits, probs, tempering

    def sample(self, task, prompt_ids, generator, prefixes=None):
        raise NotImplementedError

    def log_prob(self, task, prompt_ids, actions, start=0):
        raise NotImplementedError

    def state_logits(self, task, prompt_id):
        if self.logits is None:
            raise NotImplementedError("trajectory-level policy")
        return self.logits[prompt_id]

    def trajectory_probs(self, task, prompt_id):
        if self.probs is not None:
            return self.probs[prompt_id]
        return super().trajectory_probs(task, prompt_id)

    def tempering_logits(self, task, prompt_id):
        return self.state_logits(task, prompt_id) if self.tempering and self.logits is not None else None

    def parameters(self):
        return []

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


class RandomSequenceTask(SequenceTask):
    """V**L prefix tree; each prompt accepts a random set of sequences."""

    def __init__(self, vocab: int, length: int, n_prompts: int, n_correct: int, seed: int):
        self.n_actions, self.horizon = vocab, length
        rng = np.random.default_rng(seed)
        self._prompts = tuple(Prompt(i, (i,), {"split": "train" if i % 2 else "heldout"}) for i in range(n_prompts))
        self._correct = [
            [tuple(int(t) for t in np.unravel_index(code, (vocab,) * length)) for code in
             rng.choice(vocab**length, size=n_correct, replace=False)]
            for _ in range(n_prompts)
        ]

    def prompts(self):
        return self._prompts

    def correct_solutions(self, prompt_id):
        return self._correct[prompt_id]


# --------------------------------------------------------------------------- planted cases


_compiled: dict = {}


def _compile(size):
    if size not in _compiled:
        _compiled[size] = compile_grid(size_world(size))
    return _compiled[size]


def _measure_case(table, logits0, probs1, n, spec_c=1.0, cache=None):
    task = TableTask([table], table.n_actions, table.horizon)
    base = ArrayPolicy(logits={0: logits0})
    trained = ArrayPolicy(probs={0: probs1}, tempering=False)
    spec = MeasureSpec(ns=(n,), cs=(spec_c,), groups={"all": (0,)})
    backend = ExactBackend()
    cache = cache if cache is not None else backend.prepare(task, base, spec)
    (metrics,) = backend.measure(task, base, trained, spec, cache)
    return metrics, cache


def _check_against_evaluate(size, seed):
    table, state_index = _compile(size)
    n = size.budget
    planted = sample_cases(n, seed, table, state_index)
    caches: dict = {}  # cases from one base share the logits0 array, so share the prepare
    landed = {}
    for name, (logits0, probs1) in planted.cases.items():
        metrics, caches[id(logits0)] = _measure_case(table, logits0, probs1, n, cache=caches.get(id(logits0)))
        report = evaluate(table, logits0, probs1, n)
        assert metrics.j0 == pytest.approx(table_success_prob(table, trajectory_probs(table, logits0)), abs=1e-12)
        assert metrics.j1 == pytest.approx(table_success_prob(table, probs1), abs=1e-12)
        assert metrics.delta_rel == pytest.approx(report.delta_rel, abs=1e-12)
        assert metrics.delta_cap == pytest.approx(report.delta_cap, abs=1e-12)
        # One prompt: pass@N is monotone in J, so maximising mean pass@N and maximising J run
        # the same grid search and golden-section path; the sups agree to rounding.
        assert metrics.excess["both"] == pytest.approx(report.excess, abs=1e-6)
        assert metrics.extra["d2_label"] == report.label
        assert metrics.extra["crossed_count"] == len(report.crossed)
        assert metrics.extra["best_family"] == report.best_family
        assert metrics.extra["saturated"] == report.saturated
        assert metrics.extra["e_n_valid"] == report.e_n_valid
        landed[name] = (metrics.label == EXPECTED[name], metrics.extra["d2_label"] == EXPECTED[name])
    return landed


@pytest.mark.parametrize("seed", range(5))
def test_reproduces_evaluate_on_planted_4x4(seed):
    landed = _check_against_evaluate(SIZES[0], seed)
    # The 4x4 D2 landing is 50/50 in the pilot report.
    assert all(d2 for _, d2 in landed.values())


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(2))
def test_reproduces_evaluate_on_planted_5x5(seed):
    landed = _check_against_evaluate(SIZES[1], seed)
    assert all(d2 for _, d2 in landed.values())


@pytest.mark.slow
def test_6x6_b_param_h12_vs_d2():
    """The pilot's 6x6 b_param misses under D2 (seeds 2, 3, 4, 6, 7) and their H12 labels."""
    size = SIZES[2]
    table, state_index = _compile(size)
    d2, h12 = [], []
    for seed in range(size.trials):
        logits0, probs1 = sample_cases(size.budget, seed, table, state_index).cases["b_param"]
        metrics, _ = _measure_case(table, logits0, probs1, size.budget)
        d2.append(metrics.extra["d2_label"] == "reweighting")
        h12.append(metrics.label == "reweighting")
    assert [s for s, ok in enumerate(d2) if not ok] == [2, 3, 4, 6, 7]
    assert all(h12)


# --------------------------------------------------------------------------- Proposition 1


def _spec(task, ns=(1, 4, 16, 64, 256), cs=(0.5, 1.0, 3.0)):
    ids = [p.id for p in task.prompts()]
    return MeasureSpec(ns=ns, cs=cs, groups={"all": tuple(ids), "first": (ids[0],), "odd": tuple(ids[1::2])})


def _check_prop1(task, base, trained):
    spec = _spec(task)
    backend = ExactBackend()
    metrics = backend.measure(task, base, trained, spec, backend.prepare(task, base, spec))
    assert len(metrics) == len(spec.groups) * len(spec.ns) * len(spec.cs)
    for m in metrics:
        assert m.delta_rel + m.delta_cap == pytest.approx(m.j1 - m.j0, abs=1e-12)
        for j0, j1, rel, cap in zip(m.extra["j0"], m.extra["j1"], m.extra["delta_rel"], m.extra["delta_cap"]):
            assert rel + cap == pytest.approx(j1 - j0, abs=1e-12)
        assert m.extra["d2_label"] in EXPECTED.values() and m.label in EXPECTED.values()
    return metrics


@pytest.mark.parametrize("seed", range(3))
def test_proposition_1_tiny_prompt(seed):
    task = TinyPromptTask()
    n = len(task.prompts())
    base = TabularSequencePolicy(n, task.n_actions, task.horizon, init_scale=1.0, seed=seed)
    trained = TabularSequencePolicy(n, task.n_actions, task.horizon, init_scale=2.0, seed=seed + 100)
    _check_prop1(task, base, trained)


def test_proposition_1_larger_prefix_tree():
    task = RandomSequenceTask(vocab=5, length=4, n_prompts=6, n_correct=12, seed=1)
    n = len(task.prompts())
    base = TabularSequencePolicy(n, task.n_actions, task.horizon, init_scale=1.5, seed=3)
    trained = TabularSequencePolicy(n, task.n_actions, task.horizon, init_scale=1.5, seed=4)
    metrics = _check_prop1(task, base, trained)
    assert any(m.delta_cap != 0 for m in metrics) and any(m.delta_rel != 0 for m in metrics)


def test_registered_as_exact():
    # Checked on the class: BACKENDS.get would import every plug-in module, some of which
    # are being written concurrently.
    assert getattr(ExactBackend, "registered_name") == "exact"


def test_split_from_probs_matches_evaluate():
    task = RandomSequenceTask(vocab=4, length=4, n_prompts=1, n_correct=10, seed=2)
    table = task.table(0)
    rng = np.random.default_rng(0)
    logits0 = rng.normal(size=(table.n_states, table.n_actions))
    probs1 = trajectory_probs(table, rng.normal(size=(table.n_states, table.n_actions)) * 2)
    for n, c in [(4, 1.0), (50, 1.0), (50, 3.0)]:
        s = split_from_probs(table, trajectory_probs(table, logits0), probs1, n, c)
        r = evaluate(table, logits0, probs1, n, c=c)
        assert (s.delta_rel, s.delta_cap) == pytest.approx((r.delta_rel, r.delta_cap), abs=1e-12)
        assert s.delta_rel + s.delta_cap == pytest.approx(s.j1 - s.j0, abs=1e-12)
        assert len(s.crossed) == len(r.crossed)


# --------------------------------------------------------------------------- identity, families, groups


def test_identity_is_pure_sharpening():
    task = RandomSequenceTask(vocab=4, length=4, n_prompts=4, n_correct=8, seed=5)
    base = TabularSequencePolicy(4, 4, 4, init_scale=1.0, seed=0)
    for m in _check_prop1(task, base, base.clone()):
        assert m.delta_rel == 0.0 and m.delta_cap == 0.0 and m.j1 == m.j0
        assert all(e <= 1e-12 for e in m.excess.values())
        assert m.label == "pure_sharpening" and m.extra["crossed_count"] == 0


def test_sequence_family_from_probs_matches_sequence_probs():
    task = RandomSequenceTask(vocab=4, length=4, n_prompts=1, n_correct=5, seed=0)
    table = task.table(0)
    logits = np.random.default_rng(1).normal(size=(table.n_states, table.n_actions)) * 2
    probs = trajectory_probs(table, logits)
    for beta in (0.25, 0.7, 1.0, 3.0, 8.0):
        np.testing.assert_allclose(sequence_probs_from_probs(probs, beta), sequence_probs(table, logits, beta), atol=1e-12)


def test_group_excess_matches_evaluate_benchmark_on_grid():
    """Without refinement the group E_N is evaluate_benchmark's grid-only shared-beta excess."""
    task = RandomSequenceTask(vocab=4, length=4, n_prompts=5, n_correct=6, seed=7)
    base = TabularSequencePolicy(5, 4, 4, init_scale=1.5, seed=1)
    trained = TabularSequencePolicy(5, 4, 4, init_scale=1.5, seed=2)
    ids = tuple(range(5))
    spec = MeasureSpec(ns=(8, 64), cs=(1.0,), groups={"all": ids})
    tables = [task.table(i) for i in ids]
    logits0 = [base.state_logits(task, i) for i in ids]
    probs1 = [trained.trajectory_probs(task, i) for i in ids]
    for refine in (False, True):
        backend = ExactBackend(refine=refine)
        for m in backend.measure(task, base, trained, spec, backend.prepare(task, base, spec)):
            report = evaluate_benchmark(tables, logits0, probs1, m.n, c=m.c)
            assert (m.delta_rel, m.delta_cap) == pytest.approx((report.delta_rel, report.delta_cap), abs=1e-12)
            if refine:
                assert m.excess["both"] <= report.excess + 1e-12  # refinement only raises the sup
            else:
                assert m.excess["both"] == pytest.approx(report.excess, abs=1e-12)
            assert m.excess["both"] == pytest.approx(min(m.excess["per_step"], m.excess["sequence"]), abs=1e-15)
            beta = m.extra["best_beta"]
            assert DEFAULT_BETAS[0] <= beta <= DEFAULT_BETAS[-1]


def test_without_tempering_logits_only_sequence_family():
    task = RandomSequenceTask(vocab=3, length=4, n_prompts=2, n_correct=4, seed=3)
    probs = {i: trajectory_probs(task.table(i), np.random.default_rng(i).normal(size=(task.table(i).n_states, 3)))
             for i in range(2)}
    base = ArrayPolicy(probs=probs)
    spec = MeasureSpec(ns=(10,), cs=(1.0,), groups={"all": (0, 1)})
    backend = ExactBackend()
    (m,) = backend.measure(task, base, base, spec, backend.prepare(task, base, spec))
    assert set(m.excess) == {"sequence", "both"} and m.extra["best_family"] == "sequence"


def test_h12_reach_is_a_crossing_not_mass():
    """Tiny leakage onto many unreachable solutions moves delta_cap but crosses nothing."""
    task = RandomSequenceTask(vocab=4, length=3, n_prompts=1, n_correct=20, seed=0)
    table = task.table(0)
    correct = table.solution_ids >= 0
    p0 = np.where(correct, 0.001, 0.0)
    p0[np.flatnonzero(correct)[0]] = 0.5
    p0[~correct] = (1 - p0.sum()) / (~correct).sum()
    p1 = p0.copy()
    p1[correct] *= 1.5  # every unreachable solution grows, none reaches 1/N
    p1[~correct] = (1 - p1[correct].sum()) / (~correct).sum()
    base, trained = ArrayPolicy(probs={0: p0}), ArrayPolicy(probs={0: p1})
    spec = MeasureSpec(ns=(100,), cs=(1.0,), groups={"all": (0,)})
    backend = ExactBackend()
    (m,) = backend.measure(task, base, trained, spec, backend.prepare(task, base, spec))
    assert m.delta_cap > 0 and m.extra["crossed_count"] == 0 and not m.extra["beyond_reach"]
    assert m.label in ("reweighting", "pure_sharpening")


@pytest.mark.slow
def test_measure_is_fast_at_scale():
    """72 prompts x 7776 trajectories: measure must be cheap enough for every checkpoint."""
    task = RandomSequenceTask(vocab=6, length=5, n_prompts=72, n_correct=20, seed=0)
    base = TabularSequencePolicy(72, 6, 5, init_scale=1.0, seed=0)
    trained = TabularSequencePolicy(72, 6, 5, init_scale=1.0, seed=1)
    ids = tuple(range(72))
    spec = MeasureSpec(ns=(1, 8, 64, 512), cs=(1.0, 3.0), groups={"all": ids, "a": ids[:36], "b": ids[36:]})
    backend = ExactBackend()
    cache = backend.prepare(task, base, spec)
    start = time.perf_counter()
    metrics = backend.measure(task, base, trained, spec, cache)
    assert time.perf_counter() - start < 0.5
    for m in metrics:
        assert m.delta_rel + m.delta_cap == pytest.approx(m.j1 - m.j0, abs=1e-12)
