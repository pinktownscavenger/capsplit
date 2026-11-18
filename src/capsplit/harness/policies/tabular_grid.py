"""Tabular grid policy: one free logit vector per (prompt, cell).

pi(a | s) = softmax(logits[prompt, cell]). The policy conditions on the cell only (not the time
step), exactly like `compile_grid`'s state numbering, so `state_logits(task, pid)` is
`logits[pid, :n_states(pid)]` and the engine's exact trajectory probabilities apply. Prompts
with walls have fewer cells; their trailing rows are padding that is never visited.

Works with any task that provides `dynamics(pid) -> (next_state (S, 4), start, goal)` and
`n_states(pid)` in table numbering (the `grid` task). Sampling simulates the dynamics for a
batch of episodes in torch: an episode stops at the goal or after `task.horizon` actions, so
`Rollout.actions` are variable length.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch

from ..interfaces import Policy, Rollout, Task
from ..registry import POLICIES
from .tabular import _pad_prefixes


def _stack_dynamics(task: Task, ids: torch.Tensor, n_states: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(B, S, 4) next-state table (padding rows self-loop), (B,) start and (B,) goal state ids."""
    if not hasattr(task, "dynamics"):
        raise TypeError(f"{type(task).__name__} has no grid dynamics; tabular_grid needs the grid task")
    unique = sorted(set(ids.tolist()))
    nxt = torch.arange(n_states).repeat(len(unique), 4, 1).transpose(1, 2).clone()  # (U, S, 4) self-loops
    start = torch.zeros(len(unique), dtype=torch.long)
    goal = torch.zeros(len(unique), dtype=torch.long)
    for k, pid in enumerate(unique):
        table, s0, g = task.dynamics(pid)  # type: ignore[attr-defined]
        if table.shape[0] > n_states:
            raise ValueError(f"prompt {pid} has {table.shape[0]} states; the policy has {n_states}")
        nxt[k, : table.shape[0]] = torch.as_tensor(table, dtype=torch.long)
        start[k], goal[k] = int(s0), int(g)
    where = torch.as_tensor([unique.index(p) for p in ids.tolist()], dtype=torch.long)
    return nxt[where], start[where], goal[where]


@POLICIES.register("tabular_grid")
class TabularGridPolicy(Policy):
    def __init__(self, n_prompts: int, n_states: int, init_scale: float = 0.0, seed: int = 0, logits=None):
        self.n_prompts, self.n_states = int(n_prompts), int(n_states)
        shape = (self.n_prompts, self.n_states, 4)
        if logits is None:
            generator = torch.Generator().manual_seed(int(seed))
            logits = init_scale * torch.randn(shape, generator=generator)
        logits = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
        if tuple(logits.shape) != shape:
            raise ValueError(f"logits shape {tuple(logits.shape)} != {shape}")
        self.logits = torch.nn.Parameter(logits.clone())

    def _check(self, task: Task) -> None:
        if task.n_actions != 4 or len(task.prompts()) != self.n_prompts:
            raise ValueError("policy and task disagree on actions or number of prompts")

    def sample(self, task, prompt_ids, generator, prefixes=None) -> Rollout:
        self._check(task)
        ids = torch.as_tensor(list(prompt_ids), dtype=torch.long)
        batch, horizon = len(ids), task.horizon
        nxt, state, goal = _stack_dynamics(task, ids, self.n_states)
        forced_tokens, forced_lengths = _pad_prefixes(prefixes, batch, horizon)
        rows = torch.arange(batch)
        actions = torch.zeros((batch, horizon), dtype=torch.long)
        lengths = torch.zeros(batch, dtype=torch.long)
        done = torch.zeros(batch, dtype=torch.bool)
        log_prob = torch.zeros(batch)
        entropy = torch.zeros(batch)
        for t in range(horizon):
            forced = t < forced_lengths
            if bool((forced & done).any()):
                raise ValueError("a forced prefix continues after its episode ended")
            if bool(done.all()):
                break
            log_pi = torch.log_softmax(self.logits[ids, state], dim=-1)
            sampled = torch.multinomial(log_pi.detach().exp(), 1, generator=generator).squeeze(1)
            action = torch.where(forced, forced_tokens[:, t], sampled)
            active = ~done
            generated = (active & ~forced).to(log_pi.dtype)
            log_prob = log_prob + generated * log_pi.gather(1, action.unsqueeze(1)).squeeze(1)
            entropy = entropy + generated * -(log_pi.exp() * log_pi).sum(dim=-1)
            actions[:, t] = torch.where(active, action, torch.zeros_like(action))
            lengths = lengths + active.long()
            state = torch.where(active, nxt[rows, state, action], state)
            done = done | (state == goal)
        if bool((forced_lengths > lengths).any()):
            raise ValueError("a forced prefix continues after its episode ended")
        out = [tuple(row[: int(n)]) for row, n in zip(actions.tolist(), lengths.tolist())]
        return Rollout(list(prompt_ids), out, log_prob, entropy, forced_lengths.tolist())

    def log_prob(self, task, prompt_ids, actions, start=0) -> torch.Tensor:
        self._check(task)
        ids = torch.as_tensor(list(prompt_ids), dtype=torch.long)
        batch, horizon = len(ids), task.horizon
        lengths = torch.as_tensor([len(a) for a in actions], dtype=torch.long)
        if batch and int(lengths.max()) > horizon:
            raise ValueError(f"an action sequence is longer than the horizon {horizon}")
        acts = torch.zeros((batch, horizon), dtype=torch.long)
        for i, a in enumerate(actions):
            if len(a):
                acts[i, : len(a)] = torch.as_tensor([int(x) for x in a], dtype=torch.long)
        starts = torch.as_tensor([start] * batch if isinstance(start, int) else list(start), dtype=torch.long)
        nxt, state, goal = _stack_dynamics(task, ids, self.n_states)
        rows = torch.arange(batch)
        total = torch.zeros(batch)
        for t in range(int(lengths.max()) if batch else 0):
            active = t < lengths
            if bool((active & (state == goal)).any()):
                raise ValueError("an action sequence continues after reaching the goal")
            log_pi = torch.log_softmax(self.logits[ids, state], dim=-1)
            step = log_pi.gather(1, acts[:, t : t + 1]).squeeze(1)
            total = total + (active & (t >= starts)).to(step.dtype) * step
            state = torch.where(active, nxt[rows, state, acts[:, t]], state)
        return total

    def state_logits(self, task, prompt_id) -> np.ndarray:
        self._check(task)
        n = task.n_states(prompt_id) if hasattr(task, "n_states") else self.n_states
        return self.logits[prompt_id, :n].detach().double().numpy().copy()

    def tempered(self, beta: float) -> "TabularGridPolicy":
        """Per-step tempering pi^beta: a copy with every logit multiplied by beta."""
        return TabularGridPolicy(self.n_prompts, self.n_states, logits=float(beta) * self.logits.detach())

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return [self.logits]

    def state_dict(self) -> dict:
        return {"logits": self.logits.detach().clone()}

    def load_state_dict(self, state: dict) -> None:
        with torch.no_grad():
            self.logits.copy_(state["logits"])


__all__: Sequence[str] = ("TabularGridPolicy",)
