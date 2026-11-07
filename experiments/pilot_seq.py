"""Phase 2 pilot: RLOO on the sequence task, a tabular control, and the cost at full size.

Run: python experiments/pilot_seq.py
"""
import platform
import time

import numpy as np

from capsplit.engine import solution_probs, table_success_prob, trajectory_probs
from capsplit.evaluate import evaluate
from capsplit.runlog import run_dir, save_seed
from capsplit.seqtask.policy import SequencePolicy, node_logits, tabular_base_logits
from capsplit.seqtask.pretrain import calibrate, pretrain, sequence_weights
from capsplit.seqtask.verifier import SeqTask, compile_sequence, make_task
from capsplit.tempering import frontier
from capsplit.train.sequence import rloo_train
from capsplit.train.tabular import train_exact_pg

V, L, N = 6, 5, 50
N_COMMON, N_RARE = 3, 3
LEVELS = (0, 2, 4)
SEEDS = 5
TASK_SEED = 0  # one prompt, so "seeds" vary the policy and sampling
UPDATES = 600

# The blueprint doesn't fix the control's settings. These were chosen before its first run.
CONTROL_ALPHA = 1e-3
CONTROL_LR, CONTROL_STEPS = 2.0, 1000

FULL_V, FULL_L, FULL_K = 8, 6, 2
FULL_GROUP = 16
FULL_PROMPTS = 4
COST_UPDATES = 50
COST_LOGGED_UPDATES = 20


def _rare_median(task: SeqTask, table, probs: np.ndarray) -> float:
    q = solution_probs(table, probs)
    return float(np.median([q[table.solutions.index(sequence)] for sequence in task.rare]))


def _row(task, table, logits0, logits1) -> dict:
    probs0, probs1 = trajectory_probs(table, logits0), trajectory_probs(table, logits1)
    report = evaluate(table, logits0, probs1, N, frontier_result=frontier(table, logits0))
    return {
        "delta_rel": report.delta_rel,
        "delta_cap": report.delta_cap,
        "excess": report.excess,
        "j_gap": report.j_gap,
        "label": report.label,
        "e_n_valid": report.e_n_valid,
        "j0": table_success_prob(table, probs0),
        "j1": table_success_prob(table, probs1),
        "rare_q0": _rare_median(task, table, probs0),
        "rare_q1": _rare_median(task, table, probs1),
    }


def _print_row(name: str, row: dict) -> None:
    print(
        f"  {name:9s} J0={row['j0']:.3f} J1={row['j1']:.4f} rare q0={row['rare_q0']:.5f} q1={row['rare_q1']:.2e} "
        f"d_rel={row['delta_rel']:+.4f} d_cap={row['delta_cap']:+.4f} {row['label']}"
    )


def run_level(k: int) -> None:
    task = make_task(V, L, N_COMMON, N_RARE, k, seed=TASK_SEED)
    table = compile_sequence(task)
    print(f"\nk={k}: V={V}, L={L}, N={N}, {UPDATES} updates, line c/N={1 / N}")
    directory = run_dir(
        "pilot_seq",
        {"v": V, "l": L, "n": N, "k": k, "task_seed": TASK_SEED, "updates": UPDATES,
         "control_alpha": CONTROL_ALPHA, "control_lr": CONTROL_LR, "control_steps": CONTROL_STEPS},
    )
    rare_weight = None
    for seed in range(SEEDS):
        start = time.perf_counter()
        policy, info = calibrate(task, N, seed)
        rare_weight = rare_weight if rare_weight is not None else info["rare_weight"]
        logits0 = node_logits(policy, V, L)
        rloo_train(policy, task, UPDATES, seed=seed)
        row = _row(task, table, logits0, node_logits(policy, V, L))
        _print_row(f"seed {seed}", row)
        save_seed(directory, seed, seconds=time.perf_counter() - start, rare_weight=info["rare_weight"], **row)

    # The control uses seed 0's calibrated rare weight, on the same prefix tree with exact PG.
    logits0 = tabular_base_logits(task, sequence_weights(task, rare_weight), CONTROL_ALPHA)
    row = _row(task, table, logits0, train_exact_pg(table, logits0, CONTROL_LR, CONTROL_STEPS))
    _print_row("control", row)
    save_seed(directory, SEEDS, rare_weight=rare_weight, **row)


def _timed(function) -> float:
    start = time.perf_counter()
    function()
    return time.perf_counter() - start


def _logged_updates(policy, task, table, logits0, best) -> None:
    """Updates that also score the policy each time, as logging E_N during training would."""
    for _ in range(COST_LOGGED_UPDATES):
        rloo_train(policy, task, 1, group=FULL_GROUP)
        probs = trajectory_probs(table, node_logits(policy, FULL_V, FULL_L))
        evaluate(table, logits0, probs, FULL_V**FULL_L // 100, frontier_result=best)


def measure_cost() -> None:
    tasks = [make_task(FULL_V, FULL_L, N_COMMON, N_RARE, FULL_K, seed=i) for i in range(FULL_PROMPTS)]
    policies = [SequencePolicy(FULL_V, seed=i) for i in range(FULL_PROMPTS)]
    for policy, task in zip(policies, tasks):
        pretrain(policy, task, rare_weight=1e-3)
    tables = [compile_sequence(task) for task in tasks]
    costs = {}
    for prompts in (1, FULL_PROMPTS):
        pairs = list(zip(policies[:prompts], tasks[:prompts], tables[:prompts]))
        logits = [node_logits(policy, FULL_V, FULL_L) for policy, _, _ in pairs]
        costs[prompts] = {
            "compile_s": _timed(lambda: [compile_sequence(task) for _, task, _ in pairs]),
            "update_ms": 1000
            * _timed(lambda: [rloo_train(p, t, COST_UPDATES, group=FULL_GROUP) for p, t, _ in pairs])
            / COST_UPDATES,
            "node_logits_s": _timed(lambda: [node_logits(p, FULL_V, FULL_L) for p, _, _ in pairs]),
            "frontier_s": _timed(lambda: [frontier(t, z) for (_, _, t), z in zip(pairs, logits)]),
        }
        fronts = [frontier(t, z) for (_, _, t), z in zip(pairs, logits)]
        costs[prompts]["logged_update_ms"] = (
            1000
            * _timed(lambda: [_logged_updates(p, k, t, z, b) for (p, k, t), z, b in zip(pairs, logits, fronts)])
            / COST_LOGGED_UPDATES
        )
    print(f"\nCost at V={FULL_V}, L={FULL_L}, G={FULL_GROUP}, on {platform.machine()} {platform.system()}")
    print("  Each prompt has its own policy, so 4 prompts means 4 policies, run one after another.")
    print(f"  {'prompts':>8} {'ms/update':>10} {'with evaluate':>14} {'node logits s':>14} {'frontier s':>11} {'compile s':>10}")
    for prompts, cost in costs.items():
        print(
            f"  {prompts:8d} {cost['update_ms']:10.1f} {cost['logged_update_ms']:14.1f} {cost['node_logits_s']:14.3f} "
            f"{cost['frontier_s']:11.2f} {cost['compile_s']:10.3f}"
        )
    directory = run_dir("pilot_seq_cost", {"v": FULL_V, "l": FULL_L, "k": FULL_K, "group": FULL_GROUP})
    for prompts, cost in costs.items():
        save_seed(directory, prompts, **cost)


def main() -> None:
    start = time.perf_counter()
    for k in LEVELS:
        run_level(k)
    measure_cost()
    print(f"\n{time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
