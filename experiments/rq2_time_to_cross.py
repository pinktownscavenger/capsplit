"""RQ2 time to cross: isolated route (arm A), robustness, competing routes (arm B), sampled stretch.

Design: docs/rq2-time-to-cross-design.md. Plan: docs/rq2-time-to-cross-plan.md. The constants
below are frozen by docs/rq2-preregistration.md; changing one after the freeze is logged in
docs/DECISIONS.md and the affected runs are redone.

Run: python experiments/rq2_time_to_cross.py {arm-a,robust,arm-b,sampled,report} [--quick] [--root DIR]
"""
from __future__ import annotations

import numpy as np
from phase1_rq1 import R_COMMON, R_RARE
from phase1_rq1 import WORLD as PHASE1_WORLD
from pilot_rq2 import COMPETING_WORLD, ISOLATED_WALLS, ROUTE_A, ROUTE_B

from capsplit.train.tabular import exact_pg_step
from capsplit.worlds import MOVES, N_ACTIONS, GridWorld

# ----------------------------------------------------------------------------- frozen settings

LINE = 0.1  # arm A's reach line, c = 1 and N = 10
ARM_A_Q0 = tuple(float(q) for q in np.logspace(-6, -2, 9))
ARM_A_LRS = (0.5, 1.0, 2.0)
ARM_A_CAP = 400_000
ROBUST_SEEDS = tuple(range(20))
ROBUST_SIGMA = 1.0
ROBUST_H = 10
ARM_B_WORLDS = ("W1", "W2", "W3")
ARM_B_RATIOS = (1.0, 0.9, 0.5, 0.1, 0.01)
ARM_B_LRS = (0.5, 1.0, 2.0)
ARM_B_SEEDS = tuple(range(20))
ARM_B_CAP = 20_000
CROSS_FACTOR = 10
COMMON_BONUS = {"W1": 0.75, "W2": 1.0, "W3": 1.75}  # q0(common) about 0.02 at ratio 1, sigma 0 (RQ2-1)
BISECTION_STEPS = 60
BONUS_RANGE = (-8.0, 8.0)

# W3: the isolated world without the (0, 1) wall, which opens a second, top-right route.
W3_WORLD = GridWorld(n=5, start=(0, 0), goal=(4, 4), H=8, walls=frozenset(ISOLATED_WALLS - {(0, 1)}))
W3_TOP_RIGHT = ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (2, 4), (3, 4), (4, 4))
W3_LEFT_BOTTOM = ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4))

# ----------------------------------------------------------------------------- worlds and bases


def isolated_world(horizon: int = 8) -> GridWorld:
    """The pilot's isolated-route world (one correct route of length 8) with horizon `horizon`."""
    return GridWorld(n=5, start=(0, 0), goal=(4, 4), H=horizon, walls=ISOLATED_WALLS)


def competing_world(name: str) -> tuple[GridWorld, tuple, tuple]:
    """(world, common route, rare route) for arm B's worlds W1, W2 and W3."""
    if name == "W1":
        return COMPETING_WORLD, tuple(ROUTE_A), tuple(ROUTE_B)
    if name == "W2":
        return PHASE1_WORLD, tuple(R_COMMON), tuple(R_RARE)
    if name == "W3":
        return W3_WORLD, W3_TOP_RIGHT, W3_LEFT_BOTTOM
    raise ValueError(f"unknown world {name!r}; expected one of {ARM_B_WORLDS}")


def _route_slots(world: GridWorld, state_index: dict, route) -> list[tuple[int, int]]:
    """(state index, action) for each step of `route`."""
    slots = []
    for here, there in zip(route[:-1], route[1:]):
        move = (there[0] - here[0], there[1] - here[1])
        action = MOVES.index(move)
        assert world.step(here, action) == there, f"route step {here} -> {there} is blocked"
        slots.append((state_index[here], action))
    return slots


def _noise(state_index: dict, sigma: float, seed: int, keep: list[tuple[int, int]]) -> np.ndarray:
    """N(0, sigma^2) on every logit except the `keep` slots, which stay 0."""
    logits = np.zeros((len(state_index), N_ACTIONS))
    if sigma:
        logits += sigma * np.random.default_rng(seed).standard_normal(logits.shape)
        for s, a in keep:
            logits[s, a] = 0.0
    return logits


def _with_bonus(base: np.ndarray, slots: list[tuple[int, int]], bonus: float) -> np.ndarray:
    logits = base.copy()
    for s, a in slots:
        logits[s, a] += bonus
    return logits


def _bisect(value, target: float, what: str) -> float:
    """Bonus in BONUS_RANGE with value(bonus) = target, for value increasing in the bonus.

    Same rule as the pilot's `isolated_base`: below the target moves `low` up, and the result is
    `high`. Raises if the target is outside [value(low), value(high)].
    """
    low, high = BONUS_RANGE
    if not value(low) <= target <= value(high):
        raise ValueError(f"{what}: target {target:g} is outside [{value(low):g}, {value(high):g}] over {BONUS_RANGE}")
    for _ in range(BISECTION_STEPS):
        bonus = (low + high) / 2
        if value(bonus) < target:
            low = bonus
        else:
            high = bonus
    return high


def noisy_isolated_base(table, state_index: dict, world: GridWorld, target_q0: float, sigma: float, seed: int) -> np.ndarray:
    """Isolated-route base: off-route noise, then one per-step route bonus bisected to hit target_q0."""
    slots = _route_slots(world, state_index, table.solutions[0])
    base = _noise(state_index, sigma, seed, slots)

    def q0(bonus: float) -> float:
        return float(exact_pg_step(table, _with_bonus(base, slots, bonus))[1].sum())

    return _with_bonus(base, slots, _bisect(q0, target_q0, "isolated base"))


def competing_base(table, state_index: dict, name: str, ratio: float, sigma: float, seed: int) -> tuple[np.ndarray, float, float]:
    """(logits, q0(common), q0(rare)) with q0(rare) / q0(common) = ratio.

    Noise goes on every logit off both bonus routes. The common route's bonus is COMMON_BONUS[name];
    the rare route's is bisected. Both q0 are recomputed at each step, since W2's routes share moves.
    """
    world, common, rare = competing_world(name)
    c_slots, r_slots = _route_slots(world, state_index, common), _route_slots(world, state_index, rare)
    base = _with_bonus(_noise(state_index, sigma, seed, c_slots + r_slots), c_slots, COMMON_BONUS[name])
    i_common, i_rare = table.solutions.index(common), table.solutions.index(rare)

    def ratio_at(bonus: float) -> float:
        q = exact_pg_step(table, _with_bonus(base, r_slots, bonus))[1]
        return float(q[i_rare] / q[i_common])

    logits = _with_bonus(base, r_slots, _bisect(ratio_at, ratio, f"competing base {name}"))
    q = exact_pg_step(table, logits)[1]
    return logits, float(q[i_common]), float(q[i_rare])
