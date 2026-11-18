"""Grid-world task: `worlds.GridWorld` dynamics with one prompt per world variant.

RL framing:

* **Prompt**: a world variant (start cell, goal cell, walls) on an n x n grid. `Prompt.tokens`
  is (n, start, goal, sorted walls); tags `difficulty` and `split` define measurement groups.
* **State**: `(cell, t)`, the agent's cell and the number of actions taken so far. Tracking t
  in the state makes the time limit part of the dynamics: `step` reports done when the agent
  reaches the goal or when t reaches the horizon H. (Policies condition on the cell only, as
  `compile_grid`'s state numbering does; t is not a policy input.)
* **Action**: one of 4 moves (up, down, left, right). A move into a wall or off the grid leaves
  the agent in place.
* **Reward**: 1 iff the episode ends at the goal. `verify` accepts an action sequence iff it
  reaches the goal for the first time exactly at its last action and has at most H actions, so
  episodes are variable length (they stop at the goal), like the rows of `compile_grid`.
* **Solution identity**: the loop-erased visited path (`worlds.loop_erased`), so two action
  sequences that reach the goal along the same route (with different stalls or detours that
  are erased) are the same solution. `table(pid)` is `compile_grid(world, phi=loop_erased)`.
* **Demonstrations**: every shortest route, as action sequences, in sorted order.

Default variants (n=4, H=8): two easy and two hard prompts, train and held-out, with
different starts, goals and walls. Their difficulty is only a tag here; the base recipe
(`grid_routes`) decides what the base finds easy or hard.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Hashable, Mapping, Sequence, cast

import numpy as np

from capsplit.engine import TrajectoryTable
from capsplit.worlds import MOVES, N_ACTIONS, GridWorld, compile_grid, loop_erased

from ..interfaces import Prompt, Task
from ..registry import TASKS

DEFAULT_VARIANTS: tuple = (
    {"start": (0, 0), "goal": (3, 3), "walls": (), "difficulty": "easy", "split": "train"},
    {"start": (0, 0), "goal": (3, 3), "walls": ((1, 1), (2, 1)), "difficulty": "hard", "split": "train"},
    {"start": (3, 0), "goal": (0, 3), "walls": ((1, 2),), "difficulty": "easy", "split": "heldout"},
    {"start": (3, 0), "goal": (0, 3), "walls": ((1, 1), (2, 2)), "difficulty": "hard", "split": "heldout"},
)
_VARIANT_KEYS = {"start", "goal", "walls", "difficulty", "split"}


def _cell(value: Any) -> tuple:
    r, c = (int(x) for x in value)
    return (r, c)


@TASKS.register("grid")
class GridTask(Task):
    """Grid navigation with a 0/1 verifier. Params: n (4), H (8), variants (see DEFAULT_VARIANTS)."""

    def __init__(self, n: int = 4, H: int = 8, variants: Sequence[Mapping[str, Any]] | None = None):
        self.n, self.horizon, self.n_actions = int(n), int(H), N_ACTIONS
        variants = DEFAULT_VARIANTS if variants is None else variants
        if not variants:
            raise ValueError("grid task needs at least one variant")
        worlds, prompts = [], []
        for i, v in enumerate(variants):
            unknown = set(v) - _VARIANT_KEYS
            if unknown:
                raise ValueError(f"variant {i}: unknown keys {sorted(unknown)}")
            walls = frozenset(_cell(w) for w in v.get("walls", ()))
            world = GridWorld(n=self.n, start=_cell(v["start"]), goal=_cell(v["goal"]), H=self.horizon, walls=walls)
            for cell in (world.start, world.goal):
                if not (0 <= cell[0] < self.n and 0 <= cell[1] < self.n) or cell in walls:
                    raise ValueError(f"variant {i}: start/goal {cell} is off the grid or in a wall")
            if world.start == world.goal:
                raise ValueError(f"variant {i}: start equals goal")
            distance = _shortest_distance(world)
            if distance is None or distance > self.horizon:
                raise ValueError(f"variant {i}: goal not reachable within H={self.horizon} steps")
            tags = {k: str(v[k]) for k in ("difficulty", "split") if k in v}
            worlds.append(world)
            prompts.append(Prompt(i, (self.n, world.start, world.goal, tuple(sorted(walls))), tags))
        self.worlds: tuple = tuple(worlds)
        self._prompts = tuple(prompts)
        self._cache: dict = {}

    # ------------------------------------------------------------------ Task API

    def prompts(self) -> Sequence[Prompt]:
        return self._prompts

    def world(self, prompt_id: int) -> GridWorld:
        return self.worlds[prompt_id]

    def initial_state(self, prompt_id: int) -> Hashable:
        return (self.worlds[prompt_id].start, 0)

    def step(self, prompt_id: int, state: Hashable, action: int) -> tuple[Hashable, bool]:
        world = self.worlds[prompt_id]
        raw_cell, raw_t = cast(tuple, state)
        cell, t = tuple(raw_cell), int(raw_t)
        if cell == world.goal or t >= self.horizon:
            raise ValueError("step called on a finished episode")
        nxt = world.step(cell, int(action))
        return (nxt, t + 1), nxt == world.goal or t + 1 >= self.horizon

    def verify(self, prompt_id: int, actions: Sequence[int]) -> bool:
        return self.solution_of(prompt_id, actions) is not None

    def solution_of(self, prompt_id: int, actions: Sequence[int]) -> tuple | None:
        """The solution (loop-erased route) of a correct episode, or None if it is incorrect."""
        world = self.worlds[prompt_id]
        if not 0 < len(actions) <= self.horizon:
            return None
        cell, path = world.start, [world.start]
        for k, a in enumerate(actions):
            if not 0 <= int(a) < N_ACTIONS:
                return None
            cell = world.step(cell, int(a))
            path.append(cell)
            if cell == world.goal:
                return loop_erased(tuple(path)) if k == len(actions) - 1 else None
        return None

    def demonstrations(self, prompt_id: int) -> Sequence[tuple]:
        key = ("demos", prompt_id)
        if key not in self._cache:
            world = self.worlds[prompt_id]
            self._cache[key] = tuple(sorted(_route_actions(r) for r in shortest_routes(world)))
        return self._cache[key]

    def table(self, prompt_id: int) -> TrajectoryTable:
        return self._compiled(prompt_id)[0]

    # ------------------------------------------------------------------ grid helpers

    def state_index(self, prompt_id: int) -> dict:
        """cell -> state id, the numbering of `table(prompt_id)` (`compile_grid`)."""
        key = ("index", prompt_id)
        if key not in self._cache:
            self._cache[key] = {cell: i for i, cell in enumerate(self.worlds[prompt_id].cells())}
        return self._cache[key]

    def n_states(self, prompt_id: int) -> int:
        return len(self.state_index(prompt_id))

    def dynamics(self, prompt_id: int) -> tuple[np.ndarray, int, int]:
        """(next_state (S, 4) int array, start state id, goal state id) in table numbering."""
        key = ("dyn", prompt_id)
        if key not in self._cache:
            world, index = self.worlds[prompt_id], self.state_index(prompt_id)
            nxt = np.zeros((len(index), N_ACTIONS), dtype=np.int64)
            for cell, i in index.items():
                for a in range(N_ACTIONS):
                    nxt[i, a] = index[world.step(cell, a)]
            self._cache[key] = (nxt, index[world.start], index[world.goal])
        return self._cache[key]

    def _compiled(self, prompt_id: int) -> tuple[TrajectoryTable, dict]:
        key = ("table", prompt_id)
        if key not in self._cache:
            table, index = compile_grid(self.worlds[prompt_id], phi=loop_erased)
            assert index == self.state_index(prompt_id)
            self._cache[key] = (table, index)
        return self._cache[key]


def _shortest_distance(world: GridWorld) -> int | None:
    dist = {world.start: 0}
    queue = deque([world.start])
    while queue:
        cell = queue.popleft()
        for a in range(N_ACTIONS):
            nxt = world.step(cell, a)
            if nxt not in dist:
                dist[nxt] = dist[cell] + 1
                queue.append(nxt)
    return dist.get(world.goal)


def shortest_routes(world: GridWorld) -> list[tuple]:
    """Every shortest start->goal route, as a tuple of cells."""
    to_goal: dict = {world.goal: 0}
    queue = deque([world.goal])
    while queue:  # moves are reversible (except bumps, which are self-loops), so BFS from the goal
        cell = queue.popleft()
        for a in range(N_ACTIONS):
            nxt = world.step(cell, a)
            if nxt not in to_goal:
                to_goal[nxt] = to_goal[cell] + 1
                queue.append(nxt)
    routes: list = []

    def extend(path: list) -> None:
        cell = path[-1]
        if cell == world.goal:
            routes.append(tuple(path))
            return
        for a in range(N_ACTIONS):
            nxt = world.step(cell, a)
            if to_goal.get(nxt, -1) == to_goal[cell] - 1:
                extend(path + [nxt])

    if world.start in to_goal:
        extend([world.start])
    return routes


def _route_actions(route: Sequence[tuple]) -> tuple:
    return tuple(MOVES.index((b[0] - a[0], b[1] - a[1])) for a, b in zip(route[:-1], route[1:]))


# The registry imports a fixed module list; the grid policy and base recipe register here.
from ..bases import grid_base as _grid_base  # noqa: E402,F401
from ..policies import tabular_grid as _tabular_grid  # noqa: E402,F401
