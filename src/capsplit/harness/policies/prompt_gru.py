"""One prompt-conditioned GRU policy for every prompt of a sequence task.

Architecture: each prompt token (for the rule task: type, a, b) has its own embedding; their
concatenation goes through tanh(Linear) to the GRU's initial hidden state. The GRU reads a
start token (id V) and then the tokens generated so far; a linear head gives logits over V.
Parameters are shared by all prompts, so whatever the base learns on one prompt can transfer
to another (the shared-structure dial depends on this).

State numbering for `state_logits` is the prefix tree of `harness/sequence.py`, and sampling
uses the shared `sample_prefix_tree`, so forced-prefix semantics match the tabular policy.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch

from ..interfaces import Policy, Rollout, Task
from ..registry import POLICIES
from ..sequence import n_nodes
from .tabular import sample_prefix_tree


class _Net(torch.nn.Module):
    def __init__(self, vocab: int, prompt_sizes: Sequence[int], embedding: int, hidden: int):
        super().__init__()
        self.prompt_embeddings = torch.nn.ModuleList(torch.nn.Embedding(size, embedding) for size in prompt_sizes)
        self.prompt_to_hidden = torch.nn.Linear(embedding * len(prompt_sizes), hidden)
        self.token_embedding = torch.nn.Embedding(vocab + 1, embedding)  # id `vocab` is the start token
        self.gru = torch.nn.GRU(embedding, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, vocab)

    def initial_hidden(self, prompt_tokens: torch.Tensor) -> torch.Tensor:
        """(B, n_fields) prompt tokens -> (1, B, hidden) initial GRU state."""
        parts = [emb(prompt_tokens[:, i]) for i, emb in enumerate(self.prompt_embeddings)]
        return torch.tanh(self.prompt_to_hidden(torch.cat(parts, dim=1))).unsqueeze(0)

    def step(self, tokens: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, S) input tokens -> (B, S, V) logits and the new (1, B, hidden) state."""
        output, hidden = self.gru(self.token_embedding(tokens), hidden)
        return self.head(output), hidden


@POLICIES.register("prompt_gru")
class PromptGRUPolicy(Policy):
    prompt_conditioned = True  # one network for every prompt (checked by bases/verify.py, criterion a)

    def __init__(
        self,
        vocab: int,
        length: int,
        prompt_sizes: Sequence[int],
        embedding: int = 16,
        hidden: int = 64,
        seed: int = 0,
    ):
        self.vocab, self.length = vocab, length
        self.prompt_sizes = tuple(int(s) for s in prompt_sizes)
        self.embedding, self.hidden = embedding, hidden
        with torch.random.fork_rng(devices=[]):  # seed the weights without touching global random state
            torch.manual_seed(seed)
            self.net = _Net(vocab, self.prompt_sizes, embedding, hidden)

    @classmethod
    def for_task(cls, task: Task, **params) -> "PromptGRUPolicy":
        """Build a policy sized for `task` (which must expose `prompt_sizes`)."""
        return cls(task.n_actions, task.horizon, getattr(task, "prompt_sizes"), **params)

    # ------------------------------------------------------------------ helpers

    def _check(self, task: Task) -> None:
        if task.n_actions != self.vocab or task.horizon != self.length:
            raise ValueError("policy and task disagree on vocabulary or horizon")

    def _prompt_tokens(self, task: Task, prompt_ids: Sequence[int]) -> torch.Tensor:
        prompts = task.prompts()
        tokens = [tuple(prompts[int(i)].tokens) for i in prompt_ids]
        if any(len(t) != len(self.prompt_sizes) for t in tokens):
            raise ValueError(f"prompts must have {len(self.prompt_sizes)} tokens")
        return torch.as_tensor(tokens, dtype=torch.long).reshape(len(tokens), len(self.prompt_sizes))

    def _start(self, batch: int) -> torch.Tensor:
        return torch.full((batch, 1), self.vocab, dtype=torch.long)

    def sequence_logits(self, task: Task, prompt_ids: Sequence[int], actions: torch.Tensor) -> torch.Tensor:
        """(B, L, V) teacher-forced logits for full action sequences (B, L)."""
        self._check(task)
        hidden = self.net.initial_hidden(self._prompt_tokens(task, prompt_ids))
        inputs = torch.cat((self._start(actions.shape[0]), actions[:, :-1]), dim=1)
        return self.net.step(inputs, hidden)[0]

    # ------------------------------------------------------------------ Policy API

    def sample(self, task, prompt_ids, generator, prefixes=None) -> Rollout:
        self._check(task)
        ids = list(prompt_ids)
        state = {"hidden": self.net.initial_hidden(self._prompt_tokens(task, ids))}

        def logits_at(t, codes, actions_so_far):
            inputs = self._start(len(ids)) if t == 0 else actions_so_far[:, t - 1 : t]
            logits, state["hidden"] = self.net.step(inputs, state["hidden"])
            return logits[:, -1]

        actions, log_prob, entropy, forced = sample_prefix_tree(
            logits_at, self.vocab, self.length, len(ids), generator, prefixes
        )
        return Rollout(ids, [tuple(row) for row in actions.tolist()], log_prob, entropy, forced)

    def log_prob(self, task, prompt_ids, actions, start=0) -> torch.Tensor:
        ids = list(prompt_ids)
        acts = torch.as_tensor([list(a) for a in actions], dtype=torch.long).reshape(len(ids), self.length)
        log_pi = torch.log_softmax(self.sequence_logits(task, ids, acts), dim=-1)
        steps = log_pi.gather(2, acts.unsqueeze(2)).squeeze(2)
        starts = torch.as_tensor([start] * len(ids) if isinstance(start, int) else list(start), dtype=torch.long)
        mask = torch.arange(self.length).unsqueeze(0) >= starts.unsqueeze(1)
        return (steps * mask.to(steps.dtype)).sum(dim=1)

    def all_state_logits(self, task: Task, prompt_ids: Sequence[int]) -> np.ndarray:
        """(P, n_states, V) float64 logits for every prefix-tree node of each prompt, level by level."""
        self._check(task)
        ids = list(prompt_ids)
        batch, vocab = len(ids), self.vocab
        levels = []
        with torch.no_grad():
            logits, hidden = self.net.step(self._start(batch), self.net.initial_hidden(self._prompt_tokens(task, ids)))
            levels.append(logits[:, -1].reshape(batch, 1, vocab))
            for level in range(1, self.length):
                parents = vocab ** (level - 1)
                hidden = hidden.repeat_interleave(vocab, dim=1)  # parent code c -> children c*V + t
                tokens = torch.arange(vocab).repeat(batch * parents).unsqueeze(1)
                logits, hidden = self.net.step(tokens, hidden)
                levels.append(logits[:, -1].reshape(batch, parents * vocab, vocab))
        out = torch.cat(levels, dim=1).double().numpy()
        assert out.shape[1] == n_nodes(vocab, self.length)
        return out

    def state_logits(self, task, prompt_id) -> np.ndarray:
        return self.all_state_logits(task, [prompt_id])[0]

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return list(self.net.parameters())

    def state_dict(self) -> dict:
        return {name: value.detach().clone() for name, value in self.net.state_dict().items()}

    def load_state_dict(self, state: dict) -> None:
        self.net.load_state_dict(state)
