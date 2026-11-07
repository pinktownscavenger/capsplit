"""Tabular softmax policy gradient on a trajectory table: exact, and sampled with a baseline."""
import numpy as np

from capsplit.engine import TrajectoryTable, log_policy, trajectory_probs


def _score(table: TrajectoryTable, rows: np.ndarray, weights: np.ndarray, policy: np.ndarray) -> np.ndarray:
    """Sum of weight * d log P(tau) / d theta over rows: N(s, a) - pi(a | s) * sum_b N(s, b)."""
    size = table.n_states * table.n_actions
    flat = np.repeat(weights, table.horizon)
    counts = np.bincount(rows.ravel(), weights=flat, minlength=size + 1)[:size]  # drop the padding slot
    counts = counts.reshape(table.n_states, table.n_actions)
    return counts - policy * counts.sum(axis=1, keepdims=True)


def exact_pg_gradient(table: TrajectoryTable, logits: np.ndarray) -> tuple[np.ndarray, float]:
    """The gradient of J over all (s, a), and J. Only correct trajectories contribute."""
    log_pi = log_policy(logits)
    rows = table.index[table.solution_ids >= 0]
    flat = np.append(log_pi.ravel(), 0.0)  # the appended zero is what the padding index reads
    probs = np.exp(flat[rows].sum(axis=1))
    return _score(table, rows, probs, np.exp(log_pi)), float(probs.sum())


def train_exact_pg(table: TrajectoryTable, logits0: np.ndarray, lr: float, steps: int, on_step=None) -> np.ndarray:
    """Gradient ascent on J. `on_step(step, logits)` runs after each update, with step counting from 1."""
    logits = np.array(logits0, dtype=float)
    for step in range(1, steps + 1):
        logits += lr * exact_pg_gradient(table, logits)[0]
        if on_step is not None:
            on_step(step, logits)
    return logits


def reinforce(
    table: TrajectoryTable,
    logits0: np.ndarray,
    seed: int,
    steps: int = 400,
    batch: int = 32,
    lr: float = 0.5,
    baseline_decay: float = 0.9,
    on_step=None,
) -> np.ndarray:
    """REINFORCE with a moving-average baseline, which starts at 0 and never includes the current batch."""
    rng = np.random.default_rng(seed)
    logits = np.array(logits0, dtype=float)
    baseline = 0.0
    for step in range(1, steps + 1):
        probs = trajectory_probs(table, logits)
        sampled = rng.choice(len(probs), size=batch, p=probs / probs.sum())
        rewards = (table.solution_ids[sampled] >= 0).astype(float)
        advantages = rewards - baseline
        logits += lr * _score(table, table.index[sampled], advantages, np.exp(log_policy(logits))) / batch
        baseline = baseline_decay * baseline + (1.0 - baseline_decay) * rewards.mean()
        if on_step is not None:
            on_step(step, logits)
    return logits
