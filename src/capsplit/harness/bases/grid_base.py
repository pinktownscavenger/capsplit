"""Hand-set grid base (`grid_routes`): route bonuses on tabular logits, as `planted.py` does.

The base starts from zero logits (optionally plus Gaussian noise) and, per prompt:

* adds `easy_bonus` (default 2.0) on easy prompts / `hard_bonus` (default 1.0) on hard prompts
  to the action a *favoured* shortest route takes at each of its cells;
* on hard prompts only, adds `hard_trap` (default 2.0) to every action that is a no-op at the
  start cell (it bumps an edge or a wall), so the base mostly stalls at the start and runs out
  of time. If the start has no no-op action the trap is skipped (info records it).

With the defaults on the default 4x4, H=8 grid task (seeds 0-2), the easy favoured route has
base probability 0.26-0.36 (reachable for N >= 4 at c = 1; easy J0 ~ 0.35-0.39) and the hard
favoured route 0.020-0.026 (unreachable at N = 16, reachable at N = 64; hard J0 ~ 0.025-0.03).
Every other correct route has q0 below ~0.02. Values are reported in info (`favoured_q0`,
`j0`). So at N = 16 each prompt has unreachable correct routes, and easy prompts also a
reachable one; at N = 64 the hard favoured route becomes reachable.

Params: easy_bonus, hard_bonus, hard_trap, route ("seeded": a seeded choice among the sorted
shortest routes, or an int index into them), noise (0.0: std of Gaussian noise added to all
logits, seeded).
"""
from __future__ import annotations

import random

import numpy as np

from capsplit.engine import solution_probs, table_success_prob, trajectory_probs
from capsplit.worlds import MOVES, N_ACTIONS

from ..interfaces import BaseRecipe
from ..policies.tabular_grid import TabularGridPolicy
from ..registry import BASES
from ..tasks.grid import GridTask, shortest_routes

_PARAMS = {"easy_bonus", "hard_bonus", "hard_trap", "route", "noise"}


@BASES.register("grid_routes")
class GridRoutesBase(BaseRecipe):
    def __init__(self, **params):
        unknown = set(params) - _PARAMS
        if unknown:
            raise ValueError(f"grid_routes: unknown params {sorted(unknown)}")
        super().__init__(**params)

    def build(self, task, seed):
        if not isinstance(task, GridTask):
            raise TypeError("grid_routes needs the grid task")
        easy = float(self.params.get("easy_bonus", 2.0))
        hard = float(self.params.get("hard_bonus", 1.0))
        trap = float(self.params.get("hard_trap", 2.0))
        route_param = self.params.get("route", "seeded")
        noise = float(self.params.get("noise", 0.0))
        rng = random.Random(seed)
        n_prompts = len(task.prompts())
        n_states = max(task.n_states(p) for p in range(n_prompts))
        logits = np.zeros((n_prompts, n_states, N_ACTIONS))
        if noise:
            logits += noise * np.random.default_rng(seed).standard_normal(logits.shape)
        favoured, trapped = [], []
        for prompt in task.prompts():
            pid, world, index = prompt.id, task.world(prompt.id), task.state_index(prompt.id)
            is_hard = prompt.tags.get("difficulty") == "hard"
            routes = sorted(shortest_routes(world))
            if route_param == "seeded":
                route = rng.choice(routes)
            else:
                route = routes[int(route_param) % len(routes)]
            bonus = hard if is_hard else easy
            for here, there in zip(route[:-1], route[1:]):
                logits[pid, index[here], MOVES.index((there[0] - here[0], there[1] - here[1]))] += bonus
            has_trap = False
            if is_hard and trap:
                for a in range(N_ACTIONS):
                    if world.step(world.start, a) == world.start:
                        logits[pid, index[world.start], a] += trap
                        has_trap = True
            favoured.append(route)
            trapped.append(has_trap)
        policy = TabularGridPolicy(n_prompts, n_states, logits=logits)
        favoured_q0, j0 = [], []
        for pid in range(n_prompts):
            table = task.table(pid)
            probs = trajectory_probs(table, policy.state_logits(task, pid))
            favoured_q0.append(float(solution_probs(table, probs)[table.solutions.index(favoured[pid])]))
            j0.append(table_success_prob(table, probs))
        info = {
            "recipe": "grid_routes",
            "favoured_routes": [[list(c) for c in r] for r in favoured],
            "trapped": trapped,
            "favoured_q0": favoured_q0,
            "j0": j0,
        }
        return policy, info
