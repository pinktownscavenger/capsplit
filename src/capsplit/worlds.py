"""Phase 1: exact tabular grid worlds.

Dynamics are deterministic: moving into a wall or off the grid leaves the agent
in place. An episode ends on reaching the goal or after H steps. A *solution*
is the visited state path; it is correct iff it ends at the goal. Trajectories
are enumerated exactly, so every probability is computed, not estimated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .engine import TrajectoryTable

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
N_ACTIONS = len(MOVES)


@dataclass(frozen=True)
class GridWorld:
    n: int
    start: tuple
    goal: tuple
    H: int
    walls: frozenset = field(default_factory=frozenset)

    def step(self, pos: tuple, a: int) -> tuple:
        r, c = pos[0] + MOVES[a][0], pos[1] + MOVES[a][1]
        if not (0 <= r < self.n and 0 <= c < self.n) or (r, c) in self.walls:
            return pos
        return (r, c)

    def cells(self):
        return [(r, c) for r in range(self.n) for c in range(self.n) if (r, c) not in self.walls]


def softmax(logits, beta: float = 1.0):
    """Per-step tempering: pi^beta renormalised, computed stably from logits."""
    z = [beta * x for x in logits]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def policy_from_routes(world: GridWorld, routes, bonuses, base: float = 0.0) -> dict:
    """Tabular logits: 0 everywhere, +bonus on the action that follows each route.

    The per-step bonus is the dial for a route's base probability.
    """
    logits = {s: [base] * N_ACTIONS for s in world.cells()}
    for route, b in zip(routes, bonuses):
        for s, s2 in zip(route[:-1], route[1:]):
            for a in range(N_ACTIONS):
                if world.step(s, a) == s2:
                    logits[s][a] += b
                    break
    return logits


def _enumerate(world: GridWorld, logits: dict, beta: float, keep_actions: bool) -> dict:
    out: dict = {}

    def rec(pos, path, acts, p, t):
        if pos == world.goal or t == world.H:
            key = tuple(acts) if keep_actions else tuple(path)
            out[key] = out.get(key, 0.0) + p
            return
        probs = softmax(logits[pos], beta)
        for a, pa in enumerate(probs):
            if pa > 0.0:
                rec(world.step(pos, a), path + [world.step(pos, a)], acts + [a], p * pa, t + 1)

    rec(world.start, [world.start], [], 1.0, 0)
    return out


def enumerate_paths(world, logits, beta=1.0) -> dict:
    """q(path): probability of each visited-state path. Sums to 1."""
    return _enumerate(world, logits, beta, keep_actions=False)


def enumerate_actions(world, logits, beta=1.0) -> dict:
    """P(action sequence); used to check Proposition 2 at trajectory level."""
    return _enumerate(world, logits, beta, keep_actions=True)


def correct_paths(world: GridWorld, q: dict) -> set:
    return {s for s in q if s[-1] == world.goal}


def per_step_family(world, logits):
    """Callable beta -> path distribution under per-step tempering of pi0."""
    return lambda beta: enumerate_paths(world, logits, beta)


def raw_path(path: tuple) -> tuple:
    """Identity solution map: every visited-state path is its own solution."""
    return path


def loop_erased(path: tuple) -> tuple:
    """Route solution map: remove stays and loops from a visited-state path."""
    route: list = []
    for cell in path:
        if cell in route:
            del route[route.index(cell) + 1 :]
        else:
            route.append(cell)
    return tuple(route)


def compile_grid(world: GridWorld, phi=loop_erased) -> tuple[TrajectoryTable, dict]:
    """Enumerate action sequences once. Returns the table and the cell -> state index map."""
    state_index = {cell: i for i, cell in enumerate(world.cells())}
    padding = len(state_index) * N_ACTIONS
    rows: list = []
    solution_ids: list = []
    solutions: dict = {}

    def rec(pos, path, row):
        if pos == world.goal or len(row) == world.H:
            rows.append(row + [padding] * (world.H - len(row)))
            if pos == world.goal:
                solution_ids.append(solutions.setdefault(phi(tuple(path)), len(solutions)))
            else:
                solution_ids.append(-1)
            return
        for a in range(N_ACTIONS):
            nxt = world.step(pos, a)
            rec(nxt, path + [nxt], row + [state_index[pos] * N_ACTIONS + a])

    rec(world.start, [world.start], [])
    table = TrajectoryTable(
        index=np.array(rows, dtype=np.int32).reshape(len(rows), world.H),
        solution_ids=np.array(solution_ids, dtype=np.int32),
        solutions=tuple(solutions),
        n_states=len(state_index),
        n_actions=N_ACTIONS,
        horizon=world.H,
    )
    return table, state_index


def logits_to_array(state_index: dict, logits: dict) -> np.ndarray:
    """Convert a cell -> logits dict, as `policy_from_routes` returns, to an array."""
    out = np.zeros((len(state_index), N_ACTIONS))
    for cell, i in state_index.items():
        out[i] = logits[cell]
    return out
