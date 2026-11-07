"""Frozen planted-case generators for RQ1 (blueprint D8). Don't tune these to make a case land."""
import random
from dataclasses import dataclass

import numpy as np

from .engine import TrajectoryTable, solution_probs, trajectory_probs
from .tempering import sequence_probs
from .worlds import MOVES, GridWorld

EXPECTED = {
    "a_per": "pure_sharpening",
    "a_seq": "pure_sharpening",
    "ap_per": "compounded_sharpening",
    "ap_seq": "compounded_sharpening",
    "b_param": "reweighting",
    "b_direct": "reweighting",
    "c_param": "capability",
    "c_direct": "capability",
}

_RIGHT, _DOWN = (0, 1), (1, 0)
_TRAP_START_LOGITS = [3.0, 0.95, 0.0, 0.95]  # up, down, left, right
_TRAP_MARGIN, _RARE_CEILING = 1.2, 0.5  # multiples of 1/N
_BISECTION_STEPS = 40
_BONUS_LIMIT = 14.0


@dataclass(frozen=True)
class Size:
    grid: int
    horizon: int
    budget: int
    trials: int
    quick_trials: int


SIZES = (Size(4, 6, 20, 50, 5), Size(5, 8, 50, 20, 2), Size(6, 10, 100, 8, 1))


@dataclass(frozen=True)
class PlantedSet:
    budget: int
    cases: dict  # case name -> (logits0, probs1); cases from one base share the logits0 array
    routes: dict  # "a", "d" and "rare" -> route
    trap_bonus: float


def size_world(size: Size) -> GridWorld:
    last = size.grid - 1
    return GridWorld(n=size.grid, start=(0, 0), goal=(last, last), H=size.horizon)


def _first_move(route: tuple) -> tuple:
    return route[1][0] - route[0][0], route[1][1] - route[0][1]


def _route_logits(table: TrajectoryTable, state_index: dict, routes, bonuses) -> np.ndarray:
    """Zero logits, plus each route's bonus on the action it takes at each of its states."""
    logits = np.zeros((table.n_states, table.n_actions))
    for route, bonus in zip(routes, bonuses):
        for here, there in zip(route[:-1], route[1:]):
            logits[state_index[here], MOVES.index((there[0] - here[0], there[1] - here[1]))] += bonus
    return logits


def _route_q(table: TrajectoryTable, logits: np.ndarray, route: tuple) -> float:
    return float(solution_probs(table, trajectory_probs(table, logits))[table.solutions.index(route)])


def _pick_disjoint_pair(rng: random.Random, table: TrajectoryTable) -> tuple:
    right = [r for r in table.solutions if _first_move(r) == _RIGHT]
    down = [r for r in table.solutions if _first_move(r) == _DOWN]
    interior = {r: set(r[1:-1]) for r in right + down}
    pairs = [(a, d) for a in right for d in down if not interior[a] & interior[d]]
    if not pairs:
        raise ValueError("no state-disjoint route pair")
    return rng.choice(pairs)


def _trap_logits(table, state_index, a, d, bonus: float) -> np.ndarray:
    logits = _route_logits(table, state_index, (a, d), (bonus, bonus))
    logits[state_index[a[0]]] = _TRAP_START_LOGITS
    return logits


def _trap_bonus(table, state_index, a, d, n: int) -> float:
    """Smallest bonus putting both routes at or above the margin over 1/N."""

    def clears(bonus: float) -> bool:
        logits = _trap_logits(table, state_index, a, d, bonus)
        return min(_route_q(table, logits, a), _route_q(table, logits, d)) >= _TRAP_MARGIN / n

    low, high = 0.0, _BONUS_LIMIT
    if not clears(high):
        raise ValueError(f"no bonus up to {_BONUS_LIMIT} clears the trap margin at N={n}")
    for _ in range(_BISECTION_STEPS):
        middle = (low + high) / 2
        if clears(middle):
            high = middle
        else:
            low = middle
    return high


def _pick_rare(rng: random.Random, table, trap: np.ndarray, a: tuple, n: int) -> tuple:
    q = solution_probs(table, trajectory_probs(table, trap))
    candidates = [
        route
        for route, mass in zip(table.solutions, q)
        if _first_move(route) == _RIGHT and route != a and mass < _RARE_CEILING / n
    ]
    if not candidates:
        raise ValueError(f"no route with first move right lies below {_RARE_CEILING}/N at N={n}")
    return rng.choice(candidates)


def _move_incorrect_mass(table: TrajectoryTable, logits0: np.ndarray, route: tuple) -> np.ndarray:
    """Move half the base's incorrect mass onto `route`'s trajectories, in proportion to their mass."""
    probs = trajectory_probs(table, logits0)
    moved = probs.copy()
    incorrect = table.solution_ids < 0
    moved[incorrect] /= 2.0
    target = table.solution_ids == table.solutions.index(route)
    moved[target] += probs[incorrect].sum() / 2.0 * probs[target] / probs[target].sum()
    return moved


def sample_cases(n: int, seed: int, table: TrajectoryTable, state_index: dict) -> PlantedSet:
    rng = random.Random(seed)
    a, d = _pick_disjoint_pair(rng, table)
    beta = rng.uniform(3.0, 5.0)

    j = rng.uniform(0.8, 1.2)
    easy = _route_logits(table, state_index, (a, d), (3.0 * j, 2.5 * j))
    weak = _route_logits(table, state_index, (a,), (rng.uniform(1.0, 1.3),))

    b = _trap_bonus(table, state_index, a, d, n)
    trap = _trap_logits(table, state_index, a, d, b)
    rare = _pick_rare(rng, table, trap, a, n)

    b_param = _route_logits(table, state_index, (a, d), (b - 1.0, b + 2.0))
    c_param = _route_logits(table, state_index, (a, d, rare), (b, b, b + 1.0))

    cases = {
        "a_per": (easy, trajectory_probs(table, easy, beta)),
        "a_seq": (easy, sequence_probs(table, easy, beta)),
        "ap_per": (weak, trajectory_probs(table, weak, beta)),
        "ap_seq": (weak, sequence_probs(table, weak, beta)),
        "b_param": (trap, trajectory_probs(table, b_param)),
        "b_direct": (trap, _move_incorrect_mass(table, trap, d)),
        "c_param": (trap, trajectory_probs(table, c_param)),
        "c_direct": (trap, _move_incorrect_mass(table, trap, rare)),
    }
    return PlantedSet(budget=n, cases=cases, routes={"a": a, "d": d, "rare": rare}, trap_bonus=b)
