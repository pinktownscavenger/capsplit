"""Prefix-tree tabular policy: one free logit vector per (prompt, prefix).

It shares nothing between prefixes or prompts, so it is the no-generalisation control for
the prompt-conditioned network, and the policy the interface tests use.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch

from ..interfaces import Policy, Rollout, Task
from ..registry import POLICIES
from ..sequence import n_nodes, node_offsets


def _pad_prefixes(prefixes: Sequence[tuple] | None, batch: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, L) forced tokens (0 where not forced) and (B,) forced lengths."""
    tokens = torch.zeros((batch, length), dtype=torch.long)
    lengths = torch.zeros(batch, dtype=torch.long)
    if prefixes is not None:
        if len(prefixes) != batch:
            raise ValueError(f"got {len(prefixes)} prefixes for {batch} episodes")
        for i, prefix in enumerate(prefixes):
            if len(prefix) > length:
                raise ValueError(f"prefix of length {len(prefix)} exceeds the horizon {length}")
            if len(prefix):
                tokens[i, : len(prefix)] = torch.tensor(prefix, dtype=torch.long)
            lengths[i] = len(prefix)
    return tokens, lengths


def sample_prefix_tree(
    logits_at, vocab: int, length: int, batch: int, generator: torch.Generator, prefixes: Sequence[tuple] | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Shared sampler for sequence policies.

    `logits_at(t, codes, actions_so_far)` returns (B, V) logits for step t, where codes are
    the base-V codes of the prefixes. Returns actions (B, L), log_prob (B,), entropy (B,),
    and forced lengths.
    """
    forced_tokens, forced_lengths = _pad_prefixes(prefixes, batch, length)
    codes = torch.zeros(batch, dtype=torch.long)
    columns: list[torch.Tensor] = []  # never written in place: logits_at may keep them for autograd
    log_prob = torch.zeros(batch)
    entropy = torch.zeros(batch)
    for t in range(length):
        so_far = torch.stack(columns, dim=1) if columns else torch.zeros((batch, 0), dtype=torch.long)
        log_pi = torch.log_softmax(logits_at(t, codes, so_far), dim=-1)
        sampled = torch.multinomial(log_pi.detach().exp(), 1, generator=generator).squeeze(1)
        forced = t < forced_lengths
        action = torch.where(forced, forced_tokens[:, t], sampled)
        generated = (~forced).to(log_pi.dtype)
        log_prob = log_prob + generated * log_pi.gather(1, action.unsqueeze(1)).squeeze(1)
        entropy = entropy + generated * -(log_pi.exp() * log_pi).sum(dim=-1)
        columns.append(action)
        codes = codes * vocab + action
    actions = torch.stack(columns, dim=1) if columns else torch.zeros((batch, 0), dtype=torch.long)
    return actions, log_prob, entropy, forced_lengths.tolist()


@POLICIES.register("tabular_sequence")
class TabularSequencePolicy(Policy):
    def __init__(self, n_prompts: int, vocab: int, length: int, init_scale: float = 0.0, seed: int = 0, logits=None):
        self.n_prompts, self.vocab, self.length = n_prompts, vocab, length
        self.offsets = torch.as_tensor(node_offsets(vocab, length), dtype=torch.long)
        shape = (n_prompts, n_nodes(vocab, length), vocab)
        if logits is None:
            generator = torch.Generator().manual_seed(seed)
            logits = init_scale * torch.randn(shape, generator=generator)
        logits = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
        if tuple(logits.shape) != shape:
            raise ValueError(f"logits shape {tuple(logits.shape)} != {shape}")
        self.logits = torch.nn.Parameter(logits.clone())

    def _check(self, task: Task) -> None:
        if task.n_actions != self.vocab or task.horizon != self.length:
            raise ValueError("policy and task disagree on vocabulary or horizon")

    def sample(self, task, prompt_ids, generator, prefixes=None) -> Rollout:
        self._check(task)
        ids = torch.as_tensor(list(prompt_ids), dtype=torch.long)

        def logits_at(t, codes, _):
            return self.logits[ids, self.offsets[t] + codes]

        actions, log_prob, entropy, forced = sample_prefix_tree(
            logits_at, self.vocab, self.length, len(ids), generator, prefixes
        )
        return Rollout(list(prompt_ids), [tuple(row) for row in actions.tolist()], log_prob, entropy, forced)

    def log_prob(self, task, prompt_ids, actions, start=0) -> torch.Tensor:
        self._check(task)
        ids = torch.as_tensor(list(prompt_ids), dtype=torch.long)
        acts = torch.as_tensor([list(a) for a in actions], dtype=torch.long)
        starts = torch.as_tensor(start if not isinstance(start, int) else [start] * len(ids), dtype=torch.long)
        codes = torch.zeros(len(ids), dtype=torch.long)
        total = torch.zeros(len(ids))
        for t in range(self.length):
            log_pi = torch.log_softmax(self.logits[ids, self.offsets[t] + codes], dim=-1)
            step = log_pi.gather(1, acts[:, t : t + 1]).squeeze(1)
            total = total + (t >= starts).to(step.dtype) * step
            codes = codes * self.vocab + acts[:, t]
        return total

    def state_logits(self, task, prompt_id) -> np.ndarray:
        self._check(task)
        return self.logits[prompt_id].detach().double().numpy().copy()

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return [self.logits]

    def state_dict(self) -> dict:
        return {"logits": self.logits.detach().clone()}

    def load_state_dict(self, state: dict) -> None:
        with torch.no_grad():
            self.logits.copy_(state["logits"])
