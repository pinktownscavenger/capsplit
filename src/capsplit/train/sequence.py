"""On-policy RLOO: leave-one-out baseline within each group of sampled sequences."""
import torch

from capsplit.seqtask.policy import SequencePolicy
from capsplit.seqtask.verifier import SeqTask


def _sample(policy: SequencePolicy, task: SeqTask, group: int, generator: torch.Generator):
    """(group, length) tokens and each sequence's log-probability, which carries the gradient."""
    tokens = torch.full((group, 1), task.vocab)
    hidden = None
    sampled = []
    log_prob = torch.zeros(group)
    for _ in range(task.length):
        logits, hidden = policy.step(tokens, hidden)
        log_pi = torch.log_softmax(logits[:, -1], dim=-1)
        tokens = torch.multinomial(log_pi.detach().exp(), 1, generator=generator)
        log_prob = log_prob + log_pi.gather(1, tokens).squeeze(1)
        sampled.append(tokens)
    return torch.cat(sampled, dim=1), log_prob


def rloo_train(policy: SequencePolicy, task: SeqTask, updates: int, group: int = 16, lr: float = 1e-3, seed: int = 0) -> None:
    """Each sequence's advantage is its reward minus the mean reward of the rest of its group."""
    if group < 2:
        raise ValueError(f"RLOO needs a group of at least 2, got {group}")
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(updates):
        sequences, log_prob = _sample(policy, task, group, generator)
        rewards = torch.tensor([float(task.is_accepted(tuple(row))) for row in sequences.tolist()])
        advantages = (group * rewards - rewards.sum()) / (group - 1)
        loss = -(advantages * log_prob).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
