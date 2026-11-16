"""Numerical checks of docs/proofs.md, run on the engine the experiments use.

Each test names the section of proofs.md it checks. These support the proofs; they don't replace reading them.
"""
import math

import numpy as np
import pytest

from capsplit.diagnostics import passk_curve
from capsplit.engine import TrajectoryTable, solution_probs, trajectory_log_probs, trajectory_probs
from capsplit.estimator import chernoff_size
from capsplit.tempering import sequence_probs
from capsplit.worlds import GridWorld, compile_grid

BETAS = (1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
GRIDS = ((3, 4), (3, 6), (4, 6))  # (side, H), no walls, as in proofs.md
POLICIES = 40
LOG_TOL = 1e-9


def _grid(side: int, horizon: int) -> TrajectoryTable:
    world = GridWorld(n=side, start=(0, 0), goal=(side - 1, side - 1), H=horizon)
    return compile_grid(world)[0]


def _random_logits(rng: np.random.Generator, table: TrajectoryTable) -> np.ndarray:
    return rng.normal(scale=rng.uniform(0.5, 3.0), size=(table.n_states, table.n_actions))


def _families(table: TrajectoryTable):
    """(name, beta -> log P_beta over trajectories, log U)."""
    return (
        ("per_step", lambda logits, b: trajectory_log_probs(table, logits, b), -table.horizon * math.log(table.n_actions)),
        ("sequence", lambda logits, b: np.log(sequence_probs(table, logits, b)), -math.log(len(table.index))),
    )


# Section 2b: T <= |A|^H under deterministic dynamics.
@pytest.mark.parametrize("side, horizon", GRIDS)
def test_support_size_at_most_actions_to_the_horizon(side, horizon):
    table = _grid(side, horizon)
    assert len(table.index) <= table.n_actions**table.horizon


# Section 2: log(P_beta / P0) <= (beta - 1) * (log P0 - log U), on every grid trajectory.
@pytest.mark.parametrize("side, horizon", GRIDS)
def test_proposition_2_on_grid_trajectories(side, horizon):
    table = _grid(side, horizon)
    rng = np.random.default_rng(side * 100 + horizon)
    for _ in range(POLICIES):
        logits = _random_logits(rng, table)
        log_p0 = trajectory_log_probs(table, logits)
        for _, log_p_at, log_u in _families(table):
            for beta in BETAS:
                excess = (log_p_at(logits, beta) - log_p0) - (beta - 1.0) * (log_p0 - log_u)
                assert excess.max() <= LOG_TOL


# Section 3: if q0(s) > 0 and every trajectory of s is below U, every beta > 1 strictly lowers q(s).
@pytest.mark.parametrize("side, horizon", GRIDS)
def test_corollary_on_grid_solutions(side, horizon):
    table = _grid(side, horizon)
    rng = np.random.default_rng(side * 1000 + horizon)
    correct = table.solution_ids >= 0
    fired = 0
    for _ in range(POLICIES):
        logits = _random_logits(rng, table)
        log_p0 = trajectory_log_probs(table, logits)
        q0 = solution_probs(table, np.exp(log_p0))
        for _, log_p_at, log_u in _families(table):
            # The highest trajectory log-probability inside each solution.
            top = np.full(len(table.solutions), -np.inf)
            np.maximum.at(top, table.solution_ids[correct], log_p0[correct])
            below = (top < log_u) & (q0 > 0)
            for beta in BETAS[1:]:
                qb = solution_probs(table, np.exp(log_p_at(logits, beta)))
                assert (qb[below] < q0[below]).all()
                fired += int(below.sum())
    assert fired > 0  # the condition is met somewhere, so the check isn't vacuous


# Section 3: the worked example, where q0(s) = 0.4 > U = 1/4 but both trajectories of s are below U.
def test_corollary_example():
    # States: 0 start, 1 after a, 2 after b. Actions: 0 = a, 1 = b. Row entry = state * 2 + action.
    table = TrajectoryTable(
        index=np.array([[0, 2], [0, 3], [1, 4], [1, 5]], dtype=np.int32),  # aa, ab, ba, bb
        solution_ids=np.array([0, 0, -1, -1], dtype=np.int32),
        solutions=("s",),
        n_states=3,
        n_actions=2,
        horizon=2,
    )
    logits = np.log([[0.4, 0.6], [0.5, 0.5], [0.99, 0.01]])
    assert trajectory_probs(table, logits) == pytest.approx([0.2, 0.2, 0.594, 0.006])
    assert solution_probs(table, trajectory_probs(table, logits))[0] == pytest.approx(0.4)
    per_step = solution_probs(table, trajectory_probs(table, logits, 2.0))[0]
    sequence = solution_probs(table, sequence_probs(table, logits, 2.0))[0]
    assert per_step == pytest.approx(0.16 / 0.52)  # 0.308
    assert sequence == pytest.approx(0.08 / (0.08 + 0.594**2 + 0.006**2))  # 0.185


# Section 4, part 2: (1 - 1/N)^N <= 1/e for N >= 1 (0 at N = 1).
def test_lemma_budget_bound():
    n = np.arange(1, 100_001, dtype=float)
    values = (1.0 - 1.0 / n) ** n
    assert values[0] == 0.0
    assert (values <= math.exp(-1.0)).all()


def _newton_roots(power_sums: np.ndarray) -> np.ndarray:
    """Recover x_1..x_P from p_1..p_P via Newton's identities, as in section 5."""
    size = len(power_sums)
    e = [1.0]
    for k in range(1, size + 1):
        e.append(sum((-1) ** (i - 1) * e[k - i] * power_sums[i - 1] for i in range(1, k + 1)) / k)
    coefficients = [(-1) ** k * e[k] for k in range(size + 1)]
    return np.sort(np.roots(coefficients).real)


# Section 5, part 1: the averaged curve at k = 1..P fixes the multiset {J_i}.
def test_proposition_3_newton_reconstruction():
    rng = np.random.default_rng(5)
    prompts = 5
    for _ in range(200):
        js = rng.uniform(0.0, 1.0, size=prompts)
        curve = passk_curve(js, np.arange(1, prompts + 1))
        power_sums = prompts * (1.0 - curve)  # p_k = P (1 - C(k)), with x_i = 1 - J_i
        assert np.sort(1.0 - _newton_roots(power_sums)) == pytest.approx(np.sort(js), abs=1e-6)


def _binomial_log_pmf(m: int, p: float) -> np.ndarray:
    k = np.arange(m + 1)
    log_choose = np.array([math.lgamma(m + 1) - math.lgamma(i + 1) - math.lgamma(m - i + 1) for i in k])
    return log_choose + k * math.log(p) + (m - k) * math.log1p(-p)


# Section 6: each exact binomial tail of |m_hat - m| > eta * max(m, mu) is at most delta / 8.
def test_chernoff_tails_exact():
    mu, eta, delta = 0.05, 0.5, 0.05
    m = chernoff_size(mu, eta, delta)
    assert m == 1219
    for mean in (0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.9):
        pmf = np.exp(_binomial_log_pmf(m, mean))
        estimate = np.arange(m + 1) / m
        t = eta * max(mean, mu)
        assert pmf[estimate > mean + t].sum() <= delta / 8
        assert pmf[estimate < mean - t].sum() <= delta / 8
