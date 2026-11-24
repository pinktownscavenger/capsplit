"""Shared tabular grid policy (`tabular_grid_shared`) and its base recipe (`grid_shared_routes`), for RQ4.

`TabularGridPolicy` gives every prompt its own logits, so training on one prompt can never
change another. This policy keeps **one** logit vector per grid cell, used by every prompt:
pi(a | s) = softmax(shared[cell]), whatever the prompt. Prompts are grid variants with the
same walls and different start or goal cells, like one agent navigating to several
destinations, so training toward some goals moves the cells other goals use. That
interference is what lets a prompt get *worse* under training, which RQ4's pass@k rule needs
(docs/rq4-passk-design.md, section 5.2).

All prompts must share one wall set, so their cell numbering (`compile_grid` numbers the
non-wall cells) is identical; `_check` refuses a task where it isn't.
"""
from __future__ import annotations

import random
from typing import Iterable, Sequence

import numpy as np
import torch

from capsplit.engine import solution_probs, table_success_prob, trajectory_probs
from capsplit.worlds import MOVES, N_ACTIONS

from ..interfaces import BaseRecipe, Task
from ..registry import BASES, POLICIES
from ..tasks.grid import GridTask, shortest_routes
from .tabular_grid import TabularGridPolicy


@POLICIES.register("tabular_grid_shared")
class TabularGridSharedPolicy(TabularGridPolicy):
    """One (n_states, 4) logit table shared by all prompts. Same interface as `TabularGridPolicy`."""

    def __init__(self, n_prompts: int, n_states: int, init_scale: float = 0.0, seed: int = 0, logits=None):
        self.n_prompts, self.n_states = int(n_prompts), int(n_states)
        shape = (self.n_states, N_ACTIONS)
        if logits is None:
            generator = torch.Generator().manual_seed(int(seed))
            logits = init_scale * torch.randn(shape, generator=generator)
        logits = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
        if tuple(logits.shape) != shape:
            raise ValueError(f"shared logits shape {tuple(logits.shape)} != {shape}")
        self.shared = torch.nn.Parameter(logits.clone())

    @property
    def logits(self) -> torch.Tensor:  # type: ignore[override]
        """The shared table viewed per prompt, (n_prompts, n_states, 4); gradients reach `shared`."""
        return self.shared.unsqueeze(0).expand(self.n_prompts, self.n_states, N_ACTIONS)

    def _check(self, task: Task) -> None:
        super()._check(task)
        if hasattr(task, "world"):
            walls = {frozenset(task.world(p.id).walls) for p in task.prompts()}  # type: ignore[attr-defined]
            if len(walls) > 1:
                raise ValueError("tabular_grid_shared needs every prompt to have the same walls")

    def tempered(self, beta: float) -> "TabularGridSharedPolicy":
        return TabularGridSharedPolicy(self.n_prompts, self.n_states, logits=float(beta) * self.shared.detach())

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return [self.shared]

    def state_dict(self) -> dict:
        return {"logits": self.shared.detach().clone()}

    def load_state_dict(self, state: dict) -> None:
        with torch.no_grad():
            self.shared.copy_(state["logits"])


_BASE_PARAMS = {"route_bonus", "trap", "route", "noise"}


@BASES.register("grid_shared_routes")
class GridSharedRoutesBase(BaseRecipe):
    """Shared base: each prompt's favoured shortest route adds `route_bonus` to the shared cells it
    passes; on prompts tagged hard, `trap` is added to the start cell's no-op actions (as
    `grid_routes`). Bonuses from different prompts add up where routes cross, which is the point.

    Params: route_bonus (1.0), trap (2.0), route ("seeded" or an int index into the sorted
    shortest routes), noise (0.0: std of Gaussian noise on all logits, seeded).
    """

    def __init__(self, **params):
        unknown = set(params) - _BASE_PARAMS
        if unknown:
            raise ValueError(f"grid_shared_routes: unknown params {sorted(unknown)}")
        super().__init__(**params)

    def build(self, task, seed):
        if not isinstance(task, GridTask):
            raise TypeError("grid_shared_routes needs the grid task")
        bonus = float(self.params.get("route_bonus", 1.0))
        trap = float(self.params.get("trap", 2.0))
        route_param = self.params.get("route", "seeded")
        noise = float(self.params.get("noise", 0.0))
        rng = random.Random(seed)
        n_prompts, n_states = len(task.prompts()), task.n_states(0)
        logits = np.zeros((n_states, N_ACTIONS))
        if noise:
            logits += noise * np.random.default_rng(seed).standard_normal(logits.shape)
        favoured = []
        for prompt in task.prompts():
            world, index = task.world(prompt.id), task.state_index(prompt.id)
            routes = sorted(shortest_routes(world))
            route = rng.choice(routes) if route_param == "seeded" else routes[int(route_param) % len(routes)]
            for here, there in zip(route[:-1], route[1:]):
                logits[index[here], MOVES.index((there[0] - here[0], there[1] - here[1]))] += bonus
            if prompt.tags.get("difficulty") == "hard" and trap:
                for a in range(N_ACTIONS):
                    if world.step(world.start, a) == world.start:
                        logits[index[world.start], a] += trap
            favoured.append(route)
        policy = TabularGridSharedPolicy(n_prompts, n_states, logits=logits)
        favoured_q0, j0 = [], []
        for pid in range(n_prompts):
            table = task.table(pid)
            probs = trajectory_probs(table, policy.state_logits(task, pid))
            favoured_q0.append(float(solution_probs(table, probs)[table.solutions.index(favoured[pid])]))
            j0.append(table_success_prob(table, probs))
        info = {
            "recipe": "grid_shared_routes",
            "favoured_routes": [[list(c) for c in r] for r in favoured],
            "favoured_q0": favoured_q0,
            "j0": j0,
        }
        return policy, info


__all__: Sequence[str] = ("TabularGridSharedPolicy", "GridSharedRoutesBase")
