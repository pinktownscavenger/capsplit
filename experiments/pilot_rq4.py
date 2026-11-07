"""RQ4: how often does the pass@k rule disagree with the 2x2 label on multi-prompt runs?

Run: python experiments/pilot_rq4.py [--quick]
"""
import argparse
import time

import numpy as np

from capsplit.diagnostics import passk_rule, score_rule
from capsplit.engine import trajectory_probs
from capsplit.evaluate import evaluate_benchmark
from capsplit.planted import SIZES, sample_cases, size_world
from capsplit.runlog import run_dir, save_seed
from capsplit.seqtask.policy import node_logits
from capsplit.seqtask.pretrain import calibrate
from capsplit.seqtask.verifier import compile_sequence, make_benchmark
from capsplit.train.sequence import rloo_train
from capsplit.train.tabular import reinforce, train_exact_pg
from capsplit.worlds import compile_grid

K_LARGE = 10  # frozen before the first run
PROMPTS = 4
K_BASE = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)

GRID = SIZES[0]
GRID_MIXES = {
    "mixed": ("easy", "weak", "trap", "trap"),
    "weak": ("weak",) * PROMPTS,
    "trap": ("trap",) * PROMPTS,
}
GRID_BASE_CASE = {"easy": "a_per", "weak": "ap_per", "trap": "b_param"}
GRID_LR, GRID_STEPS = 1.0, 200
GRID_SEEDS, GRID_QUICK_SEEDS = 5, 2

SEQ_V, SEQ_L, SEQ_N = 6, 5, 50
SEQ_COMMON, SEQ_RARE = 3, 3
SEQ_LEVELS = (0, 2, 4)
SEQ_UPDATES = 600
SEQ_SEEDS, SEQ_QUICK_SEEDS = 3, 1


def k_grid(n_actions: int, horizon: int) -> tuple:
    """K_BASE up to the largest N with n_actions ** horizon >= 100 * N, ending on that N."""
    largest = n_actions**horizon // 100
    return tuple(k for k in K_BASE if k < largest) + (largest,)


def _score(tables, logits0s, probs1s, n: int) -> dict:
    report = evaluate_benchmark(tables, logits0s, probs1s, n)
    ks = k_grid(tables[0].n_actions, tables[0].horizon)
    return {
        "cell": report.label,
        "rule": passk_rule(report.per_prompt_j0, report.per_prompt_j1, ks, K_LARGE),
        "delta_rel": report.delta_rel,
        "delta_cap": report.delta_cap,
        "excess": report.excess,
        "j0": np.array(report.per_prompt_j0),
        "j1": np.array(report.per_prompt_j1),
    }


def grid_runs(seeds: int):
    table, state_index = compile_grid(size_world(GRID))
    n = GRID.budget
    for mix, kinds in GRID_MIXES.items():
        for seed in range(seeds):
            logits0s = [
                sample_cases(n, PROMPTS * seed + i, table, state_index).cases[GRID_BASE_CASE[kind]][0]
                for i, kind in enumerate(kinds)
            ]
            trained = {
                "exact_pg": [train_exact_pg(table, z, GRID_LR, GRID_STEPS) for z in logits0s],
                "reinforce": [reinforce(table, z, PROMPTS * seed + i) for i, z in enumerate(logits0s)],
            }
            for algorithm, finals in trained.items():
                probs1s = [trajectory_probs(table, z) for z in finals]
                name = f"grid {mix} {algorithm} seed={seed}"
                yield name, _score([table] * PROMPTS, logits0s, probs1s, n)


def sequence_runs(seeds: int):
    for k in SEQ_LEVELS:
        for seed in range(seeds):
            tasks = make_benchmark(SEQ_V, SEQ_L, PROMPTS, SEQ_COMMON, SEQ_RARE, k, PROMPTS * seed)
            tables, logits0s, probs1s = [], [], []
            for i, task in enumerate(tasks):
                table = compile_sequence(task)
                policy, _ = calibrate(task, SEQ_N, PROMPTS * seed + i)
                logits0s.append(node_logits(policy, SEQ_V, SEQ_L))
                rloo_train(policy, task, SEQ_UPDATES, seed=PROMPTS * seed + i)
                probs1s.append(trajectory_probs(table, node_logits(policy, SEQ_V, SEQ_L)))
                tables.append(table)
            yield f"seq k={k} rloo seed={seed}", _score(tables, logits0s, probs1s, SEQ_N)


def _print_score(name: str, rows: list) -> None:
    score = score_rule([row["rule"] for row in rows], [row["cell"] for row in rows])
    print(
        f"\n{name}: {len(rows)} runs, primary {score['primary']:.3f}, secondary {score['secondary']:.3f}, "
        f"undetermined {score['undetermined_rate']:.3f}"
    )
    for cell, values in score["per_cell"].items():
        print(
            f"  {cell:22s} runs={values['runs']:<3d} primary={values['primary']:.3f} "
            f"secondary={values['secondary']:.3f} undetermined={values['undetermined_rate']:.3f}"
        )


def main(quick: bool = False) -> None:
    start = time.perf_counter()
    grid_seeds = GRID_QUICK_SEEDS if quick else GRID_SEEDS
    seq_seeds = SEQ_QUICK_SEEDS if quick else SEQ_SEEDS
    directory = run_dir(
        "pilot_rq4",
        {"k_large": K_LARGE, "prompts": PROMPTS, "grid_seeds": grid_seeds, "seq_seeds": seq_seeds,
         "grid_lr": GRID_LR, "grid_steps": GRID_STEPS, "seq_updates": SEQ_UPDATES},
    )
    grid, sequence = [], []
    print(f"{'run':34s} {'rule':13s} {'cell':22s} {'d_rel':>8s} {'d_cap':>8s} {'E_N':>8s}")
    for rows, runs in ((grid, grid_runs(grid_seeds)), (sequence, sequence_runs(seq_seeds))):
        for name, row in runs:
            print(
                f"{name:34s} {row['rule']:13s} {row['cell']:22s} "
                f"{row['delta_rel']:+8.4f} {row['delta_cap']:+8.4f} {row['excess']:+8.4f}"
            )
            save_seed(directory, len(grid) + len(sequence), name=name, **row)
            rows.append(row)
    _print_score("Grid", grid)
    _print_score("Sequence", sequence)
    _print_score("All", grid + sequence)
    print(f"\n{time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help=f"{GRID_QUICK_SEEDS} grid seeds, {SEQ_QUICK_SEEDS} sequence seed")
    main(parser.parse_args().quick)
