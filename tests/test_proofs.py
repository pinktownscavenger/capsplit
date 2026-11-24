"""Numerical checks of docs/proofs.md, run on the engine the experiments use.

Each test names the section of proofs.md it checks. These support the proofs; they don't replace reading them.
"""
import math

import numpy as np
import pytest

from capsplit.crossing import (
    integral_local_slope,
    integral_time,
    recursion_bounds,
    scalar_isolated_trace,
    scalar_isolated_updates,
)
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


# ----------------------------------------------------------------------------------------------
# Section 7: time to cross on an isolated route.


def _route_success(logits: np.ndarray, route: np.ndarray) -> float:
    """J = prod_i softmax(logits[i])[route[i]], the isolated-route success of section 7."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    return float(np.prod(probs[np.arange(len(route)), route]))


def _numeric_grad(logits: np.ndarray, route: np.ndarray, step: float = 1e-6) -> np.ndarray:
    grad = np.zeros_like(logits)
    for idx in np.ndindex(*logits.shape):
        up, down = logits.copy(), logits.copy()
        up[idx] += step
        down[idx] -= step
        grad[idx] = (_route_success(up, route) - _route_success(down, route)) / (2 * step)
    return grad


def _off_route_gaps(logits: np.ndarray, route: np.ndarray) -> np.ndarray:
    """Route logit minus each off-route logit, one row per cell."""
    rows = np.arange(len(route))
    off = np.ones_like(logits, dtype=bool)
    off[rows, route] = False
    return (logits[rows, route][:, None] - logits)[off].reshape(len(route), -1)


# Section 7, statement: full-table gradient ascent, with the gradient taken by finite differences
# (independent of the closed form in the proof), keeps every route cell identical and moves the
# gap exactly as g <- g + lr * A / (A - 1) * q (1 - p).
@pytest.mark.parametrize("n_actions, horizon, lr", [(2, 3, 5.0), (4, 4, 20.0), (5, 6, 50.0)])
def test_section_7_recursion_matches_full_table_ascent(n_actions, horizon, lr):
    rng = np.random.default_rng(n_actions * 10 + horizon)
    route = rng.integers(n_actions, size=horizon)
    q0 = 0.02
    p0 = q0 ** (1.0 / horizon)
    logits = np.zeros((horizon, n_actions))
    logits[np.arange(horizon), route] = math.log((n_actions - 1) * p0 / (1.0 - p0))
    steps = 30
    expected = scalar_isolated_trace(q0, horizon, lr, steps, n_actions)
    for t in range(steps + 1):
        assert _route_success(logits, route) == pytest.approx(expected[t], rel=1e-6)
        assert np.ptp(_off_route_gaps(logits, route)) < 1e-6  # finite-difference noise only
        logits = logits + lr * _numeric_grad(logits, route)
    assert expected[-1] > 2 * q0  # the run moved, so the comparison isn't vacuous


# Section 7, limits: the symmetry is needed. One unequal off-route logit makes the cells drift apart.
def test_section_7_symmetry_is_needed():
    route = np.array([0, 1, 2, 3])
    logits = np.zeros((4, 4))
    logits[np.arange(4), route] = 1.0
    logits[0, 1] = 0.5
    for _ in range(10):
        logits = logits + 20.0 * _numeric_grad(logits, route)
    assert np.ptp(_off_route_gaps(logits, route).mean(axis=1)) > 1e-3


ISOLATED = [
    (n_actions, horizon, lr, line, q0)
    for n_actions in (2, 4, 5)
    for horizon in (2, 4, 8)
    for lr in (0.5, 1.0, 2.0)
    for line in (0.05, 0.1, 0.2)
    for q0 in np.logspace(-6, -2, 5)
    if line <= (horizon / (horizon + 1)) ** horizon and q0 < line
]


# Section 7, addition: the recursion's crossing count N is sandwiched by the integral,
# T <= N <= T + 1 + ln(h(line) / h(q0)) with h(q) = q (1 - q^(1/H)), when line <= (H / (H + 1))^H.
@pytest.mark.parametrize("n_actions, horizon, lr, line, q0", ISOLATED)
def test_section_7_recursion_sandwiched_by_integral(n_actions, horizon, lr, line, q0):
    lower, upper = recursion_bounds(q0, horizon, lr, line, n_actions)
    assert lower == pytest.approx(integral_time(q0, horizon, lr, line, n_actions), rel=1e-12)
    updates = scalar_isolated_updates(q0, horizon, lr, line, 10**8, n_actions)
    assert lower <= updates <= upper


def test_section_7_bounds_refuse_line_past_the_peak():
    with pytest.raises(ValueError, match="peak"):
        recursion_bounds(1e-4, 2, 1.0, 0.5)  # (2/3)^2 = 0.444 < 0.5


# Section 7, asymptotic: T * A lr H q0 / (A - 1) = 1 + (2H / (H - 1)) q0^(1/H) + o(q0^(1/H)).
@pytest.mark.parametrize("horizon", [2, 4, 8])
def test_section_7_asymptotic_leading_and_next_term(horizon):
    ratios = []
    for q0 in (1e-8, 1e-12, 1e-16):
        p0 = q0 ** (1.0 / horizon)
        scaled = integral_time(q0, horizon, 1.0, 0.1) * 4 * horizon * q0 / 3
        ratios.append((scaled - 1.0) / p0)
    target = 2 * horizon / (horizon - 1)
    errors = [abs(r - target) for r in ratios]
    assert errors[-1] < 0.02 * target and errors[0] >= errors[1] >= errors[2]


# Section 7, addition: exact local slope d log T / d log q0 = -(A - 1) / (A lr H q0 (1 - p0)^2 T).
@pytest.mark.parametrize("q0", [1e-6, 1e-4, 1e-2])
def test_section_7_local_slope_formula(q0):
    eps = 1e-5
    numeric = (
        math.log(integral_time(q0 * (1 + eps), 8, 1.0, 0.1)) - math.log(integral_time(q0 * (1 - eps), 8, 1.0, 0.1))
    ) / (math.log1p(eps) - math.log1p(-eps))
    assert integral_local_slope(q0, 8, 1.0, 0.1) == pytest.approx(numeric, rel=1e-6)


def test_section_7_local_slope_tends_to_minus_one():
    slopes = [integral_local_slope(q0, 8, 1.0, 0.1) for q0 in (1e-6, 1e-12, 1e-24)]
    assert slopes[0] == pytest.approx(-0.9346, abs=1e-4)
    assert slopes[0] > slopes[1] > slopes[2] > -1.0 and slopes[2] == pytest.approx(-1.0, abs=1e-2)


# Section 7, "what this means": the numbers quoted in the text.
def test_section_7_quoted_numbers():
    q0, horizon = 1e-6, 8
    p0 = q0 ** (1.0 / horizon)
    exact = integral_time(q0, horizon, 1.0, 0.1)
    leading = 3 / (4 * horizon * q0)
    assert p0 == pytest.approx(0.178, abs=1e-3)
    assert leading == pytest.approx(93_750)
    assert round(exact) == 148_403
    assert 1 - leading / (1 - p0) ** 2 / exact == pytest.approx(0.065, abs=1e-3)  # "within about 7%"
