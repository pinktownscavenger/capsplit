"""Tests for the sampled measurement backend (T7): its intervals cover exact values on small tasks.

Exact values are computed directly from engine functions (trajectory_probs, solution_probs and
reachability q0 >= c/N), independent of the exact backend. Observed coverage at the time of
writing, 20 backend seeds, m = m_classify = 4096, pooled over groups and N (c = 1):

* tiny_prompt, random tabular pair (N = 9, 37; 9 groups):
  ci_episodes j0 0.94, j1 0.98, delta_rel 0.97, delta_cap 0.98;
  ci (two-level) j0 0.97, j1 0.99, delta_rel 0.99, delta_cap 1.00
* grid, grid_routes base vs 10 REINFORCE steps (N = 5, 64, 256; 3 groups):
  ci_episodes j0 0.97, j1 0.95, delta_rel 0.98, delta_cap 0.96; ci (two-level) 1.00 for all

The tests assert >= 0.88 on both. The two-level intervals also resample prompts, so they are
wider and conservative for a fixed prompt set.
"""
import numpy as np
import pytest
import torch

from capsplit.engine import solution_probs, table_success_prob, trajectory_probs
from capsplit.harness import BACKENDS, BASES, TASKS, TRAINERS
from capsplit.harness.backends.sampled import SampledBackend, solution_keyer, tempered_policy
from capsplit.harness.config import build_groups
from capsplit.harness.dummy import TinyPromptTask
from capsplit.harness.interfaces import MeasureSpec
from capsplit.harness.policies.tabular import TabularSequencePolicy

METRICS = ("j0", "j1", "delta_rel", "delta_cap")
SEEDS = range(20)


def exact_values(task, base, trained, spec):
    """{(group, n, c): {metric: value}} from engine functions."""
    per = {}
    for pid in sorted({p for ids in spec.groups.values() for p in ids}):
        table = task.table(pid)
        p0 = trajectory_probs(table, base.state_logits(task, pid))
        p1 = trajectory_probs(table, trained.state_logits(task, pid))
        per[pid] = (table_success_prob(table, p0), table_success_prob(table, p1), solution_probs(table, p0), solution_probs(table, p1))
    out = {}
    for group, ids in spec.groups.items():
        for n in spec.ns:
            for c in spec.cs:
                vals = []
                for p in ids:
                    j0, j1, q0, q1 = per[p]
                    reach = q0 >= c / n
                    vals.append((j0, j1, float((q1 - q0)[reach].sum()), float((q1 - q0)[~reach].sum())))
                out[(group, n, c)] = dict(zip(METRICS, np.mean(vals, axis=0)))
    return out


def coverage(task, base, trained, spec, **params):
    """{ci key: {metric: fraction of (seed, group, N, c) cells whose interval covers the exact value}}."""
    truth = exact_values(task, base, trained, spec)
    hits = {key: {m: [] for m in METRICS} for key in ("ci", "ci_episodes")}
    for seed in SEEDS:
        backend = SampledBackend(seed=seed, betas=[], **params)
        for g in backend.measure(task, base, trained, spec, backend.prepare(task, base, spec)):
            for key in hits:
                for m in METRICS:
                    lo, hi = g.extra[key][m]
                    hits[key][m].append(lo - 1e-12 <= truth[(g.group, g.n, g.c)][m] <= hi + 1e-12)
    return {key: {m: float(np.mean(h)) for m, h in per.items()} for key, per in hits.items()}


def well_separated_ns(q0s, m, candidates, k=2):
    """Budgets N whose line 1/N splits the base's correct solutions (some reachable, some not),
    lies at least 5 classification standard deviations from every q0, and gives distinct
    reachable sets. Lines within noise of a q0 are a known weak spot (see the backend docstring)."""
    q = np.concatenate(q0s)
    good, seen = [], set()
    for n in candidates:
        sd = np.sqrt(np.maximum(q * (1 - q), 1.0 / m) / m)
        reachable = int((q >= 1.0 / n).sum())
        if np.all(np.abs(q - 1.0 / n) > 5 * sd) and 0 < reachable < len(q) and reachable not in seen:
            good.append(n)
            seen.add(reachable)
    return tuple(good[:k])


# --------------------------------------------------------------------------- tiny_prompt


@pytest.fixture(scope="module")
def tiny():
    task = TinyPromptTask()
    base = TabularSequencePolicy(4, 3, 3, init_scale=1.0, seed=1)
    trained = TabularSequencePolicy(4, 3, 3, init_scale=1.0, seed=2)
    q0s = [solution_probs(task.table(p), trajectory_probs(task.table(p), base.state_logits(task, p))) for p in range(4)]
    ns = well_separated_ns(q0s, 4096, range(2, 200))
    assert len(ns) == 2
    spec = MeasureSpec(ns, (1.0,), build_groups(task, ["difficulty", "split"]))
    return task, base, trained, spec


def test_registered_and_params():
    assert isinstance(BACKENDS.create("sampled"), SampledBackend)
    with pytest.raises(ValueError):
        BACKENDS.create("sampled", samples=10)
    b = BACKENDS.create("sampled")
    assert (b.m, b.m_classify, b.n_boot, b.alpha) == (4096, 4096, 1000, 0.05) and 1.0 in b.betas


def test_solution_keyer_fallbacks(tiny):
    task = tiny[0]

    class NoTable(TinyPromptTask):
        def table(self, prompt_id):
            return None

    by_table, by_actions = solution_keyer(task, 1), solution_keyer(NoTable(), 1)
    for solution in task.correct_solutions(1):
        assert by_table(solution) == solution and by_actions(solution) == solution
    assert by_table((2, 2, 2)) is None and by_actions((2, 2, 2)) is None


def test_coverage_tiny_prompt(tiny):
    cov = coverage(*tiny, m=4096, m_classify=4096, n_boot=500)
    print("tiny_prompt coverage", cov)
    assert all(v >= 0.88 for v in cov["ci_episodes"].values()), cov
    assert all(v >= 0.88 for v in cov["ci"].values()), cov


def test_point_estimates_converge(tiny):
    task, base, trained, spec = tiny
    truth = exact_values(task, base, trained, spec)
    errors = {}
    for m in (256, 16384):
        errs = []
        for seed in range(5):
            backend = SampledBackend(seed=seed, betas=[], m=m, m_classify=16384, n_boot=50)
            for g in backend.measure(task, base, trained, spec, backend.prepare(task, base, spec)):
                errs += [abs(getattr(g, k) - truth[(g.group, g.n, g.c)][k]) for k in METRICS]
                assert abs(g.delta_rel + g.delta_cap - (g.j1 - g.j0)) < 1e-12
        errors[m] = float(np.mean(errs))
    assert errors[16384] < errors[256] / 4 and errors[16384] < 0.005, errors


def test_identical_policies_give_zero_gain(tiny):
    task, base, _, spec = tiny
    backend = SampledBackend(m=512, m_classify=512, n_boot=50, betas=[1.0, 2.0])
    for g in backend.measure(task, base, base.clone(), spec, backend.prepare(task, base, spec)):
        assert g.j0 == g.j1 and g.delta_rel == 0 and g.delta_cap == 0
        assert g.extra["ci"]["delta_rel"][0] <= 0 <= g.extra["ci"]["delta_rel"][1]


def test_per_step_excess_matches_exact_backend(tiny):
    task, base, trained, spec = tiny
    exact_module = pytest.importorskip("capsplit.harness.backends.exact")
    betas = [0.5, 1.0, 2.0, 4.0]
    exact = exact_module.ExactBackend(betas=betas, refine=False)
    ex = {(g.group, g.n, g.c): g for g in exact.measure(task, base, trained, spec, exact.prepare(task, base, spec))}
    sampled = SampledBackend(betas=betas, m=16384, m_temper=16384, n_boot=50)
    for g in sampled.measure(task, base, trained, spec, sampled.prepare(task, base, spec)):
        e = ex[(g.group, g.n, g.c)]
        assert abs(g.excess["per_step"] - e.excess["per_step"]) < 0.03
        assert np.isnan(g.excess["sequence"]) and np.isnan(g.excess["both"])
        for k in METRICS:  # and the exact backend agrees with the engine-derived values
            assert abs(getattr(g, k) - getattr(e, k)) < 0.03


def test_no_tempering_gives_nan():
    task = TinyPromptTask()
    base = TabularSequencePolicy(4, 3, 3, init_scale=1.0, seed=0)
    assert tempered_policy(base, 2.0) is not None

    class Opaque(TabularSequencePolicy):
        def __getattribute__(self, name):
            if name in ("logits", "tempered"):
                raise AttributeError(name)
            return super().__getattribute__(name)

    assert tempered_policy(Opaque.__new__(Opaque), 2.0) is None
    spec = MeasureSpec((16,), (1.0,), {"all": (0, 1)})
    backend = SampledBackend(m=256, m_classify=256, n_boot=20, betas=[])
    g = backend.measure(task, base, base, spec, backend.prepare(task, base, spec))[0]
    assert np.isnan(g.excess["per_step"]) and g.label == "unmeasured" and "not estimated" in g.extra["e_n_note"]


# --------------------------------------------------------------------------- grid


@pytest.fixture(scope="module")
def grid():
    task = TASKS.create("grid")
    base, _ = BASES.create("grid_routes").build(task, 0)
    trained = base.clone()
    TRAINERS.create("tabular_reinforce", batch=16, lr=0.1).train(trained, task, 10, 0, [0, 1], [], lambda s, p: None)
    # N = 5 is well separated; at N = 64 and 256 the line lies within 1-3 classification
    # standard deviations of some q0 (the grid has many solutions near 1/N), which tests the
    # bootstrap's resampling of the classification sample.
    spec = MeasureSpec((5, 64, 256), (1.0,), build_groups(task, ["difficulty"]))
    return task, base, trained, spec


def test_coverage_grid(grid):
    cov = coverage(*grid, m=4096, m_classify=4096, n_boot=500)
    print("grid coverage", cov)
    assert all(v >= 0.88 for v in cov["ci_episodes"].values()), cov
    assert all(v >= 0.88 for v in cov["ci"].values()), cov


def test_grid_against_exact_backend(grid):
    task, base, trained, spec = grid
    exact_module = pytest.importorskip("capsplit.harness.backends.exact")
    exact = exact_module.ExactBackend()
    ex = {(g.group, g.n, g.c): g for g in exact.measure(task, base, trained, spec, exact.prepare(task, base, spec))}
    truth = exact_values(task, base, trained, spec)
    sampled = SampledBackend(m=4096, m_classify=4096, m_temper=1024, n_boot=200)
    for g in sampled.measure(task, base, trained, spec, sampled.prepare(task, base, spec)):
        e = ex[(g.group, g.n, g.c)]
        for k in METRICS:
            assert abs(getattr(e, k) - truth[(g.group, g.n, g.c)][k]) < 1e-9
            lo, hi = g.extra["ci"][k]
            assert lo - 0.01 <= getattr(e, k) <= hi + 0.01
        assert abs(g.excess["per_step"] - e.excess["per_step"]) < 0.1
