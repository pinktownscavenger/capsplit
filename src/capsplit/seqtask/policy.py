"""The sequence policy network, and its logits at every node of the prefix tree."""
import numpy as np
import torch

from .verifier import SeqTask, node_offsets

_EMBEDDING, _HIDDEN = 16, 64


class SequencePolicy(torch.nn.Module):
    """Embedding, GRU and a linear head over the vocabulary. The start token has id == vocab."""

    def __init__(self, vocab: int, seed: int = 0):
        super().__init__()
        self.vocab = vocab
        with torch.random.fork_rng(devices=[]):  # seed the weights without touching global random state
            torch.manual_seed(seed)
            self.embedding = torch.nn.Embedding(vocab + 1, _EMBEDDING)
            self.gru = torch.nn.GRU(_EMBEDDING, _HIDDEN, batch_first=True)
            self.head = torch.nn.Linear(_HIDDEN, vocab)

    def step(self, tokens: torch.Tensor, hidden=None) -> tuple[torch.Tensor, torch.Tensor]:
        """(batch, seq) tokens to (batch, seq, vocab) logits, plus the new hidden state."""
        output, hidden = self.gru(self.embedding(tokens), hidden)
        return self.head(output), hidden

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.step(tokens)[0]


def node_logits(policy: SequencePolicy, vocab: int, length: int) -> np.ndarray:
    """(n_states, vocab) logits, in `node_offsets` order. Each level extends the last level's hidden states."""
    levels = []
    with torch.no_grad():
        logits, hidden = policy.step(torch.tensor([[vocab]]))
        levels.append(logits[:, -1])
        for _ in range(length - 1):
            count = levels[-1].shape[0]
            tokens = torch.arange(vocab).repeat(count).unsqueeze(1)
            logits, hidden = policy.step(tokens, hidden.repeat_interleave(vocab, dim=1))
            levels.append(logits[:, -1])
    return torch.cat(levels).numpy()


def tabular_base_logits(task: SeqTask, weights: dict, alpha: float) -> np.ndarray:
    """log(prefix counts + alpha), where a sequence adds its weight at each node it passes through."""
    offsets = node_offsets(task.vocab, task.length)
    counts = np.zeros((int(offsets[-1]) + task.vocab ** (task.length - 1), task.vocab))
    for sequence, weight in weights.items():
        prefix = 0
        for level, token in enumerate(sequence):
            counts[offsets[level] + prefix, token] += weight
            prefix = prefix * task.vocab + token
    return np.log(counts + alpha)
