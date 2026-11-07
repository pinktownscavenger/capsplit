"""RQ1 sweep: do frozen planted cases land in their predicted 2x2 cell?

Run: python experiments/pilot_rq1.py [--quick]
"""
import argparse
import time

import numpy as np

from capsplit.evaluate import evaluate
from capsplit.planted import EXPECTED, SIZES, sample_cases, size_world
from capsplit.runlog import run_dir, save_seed
from capsplit.tempering import frontier
from capsplit.worlds import compile_grid

CASES = tuple(EXPECTED)
C_VALUES = (0.5, 1.0, 3.0, 5.0)
THRESHOLD_FACTORS = (0.0, 0.1, 1.0)  # reach_threshold as a multiple of c/N


def run_trial(table, state_index, n: int, seed: int) -> dict:
    """One row per case, plus labels over (c, threshold factor)."""
    planted = sample_cases(n, seed, table, state_index)
    frontiers: dict = {}
    rows = {name: [] for name in ("delta_rel", "delta_cap", "delta_j", "excess", "j_gap")}
    labels, sensitivity = [], []
    for case in CASES:
        logits0, probs1 = planted.cases[case]
        best = frontiers.setdefault(id(logits0), frontier(table, logits0))
        report = evaluate(table, logits0, probs1, n, frontier_result=best)
        labels.append(report.label)
        for name in rows:
            rows[name].append(getattr(report, name))
        sensitivity.append(
            [
                [
                    evaluate(
                        table, logits0, probs1, n, c=c, frontier_result=best, reach_threshold=factor * c / n
                    ).label
                    for factor in THRESHOLD_FACTORS
                ]
                for c in C_VALUES
            ]
        )
    return {
        "cases": np.array(CASES),
        "label": np.array(labels),
        "sensitivity": np.array(sensitivity),  # (cases, c values, threshold factors)
        **{name: np.array(values) for name, values in rows.items()},
    }


def print_size(size, results: list, seconds: float) -> None:
    trials = len(results)
    print(f"\n{size.grid}x{size.grid} H={size.horizon} N={size.budget}, {trials} trials, {seconds:.0f} s")
    labels = np.array([r["label"] for r in results])
    for i, case in enumerate(CASES):
        print(f"  {case:9s} {(labels[:, i] == EXPECTED[case]).sum()}/{trials}")
    for seed, result in enumerate(results):
        for i, case in enumerate(CASES):
            if result["label"][i] != EXPECTED[case]:
                print(
                    f"  MISS {case} seed={seed}: {result['label'][i]}, delta_cap={result['delta_cap'][i]:+.4f}, "
                    f"j_gap={result['j_gap'][i]:+.4f}"
                )

    print("  Landing by c and reach threshold (multiple of c/N):")
    sensitivity = np.array([r["sensitivity"] for r in results])
    header = " ".join(f"c={c}/t={f}" for c in C_VALUES for f in THRESHOLD_FACTORS)
    print(f"    {'':9s} {header}")
    for i, case in enumerate(CASES):
        cells = [
            f"{(sensitivity[:, i, ci, fi] == EXPECTED[case]).sum()}/{trials}".rjust(len(f"c={c}/t={f}"))
            for ci, c in enumerate(C_VALUES)
            for fi, f in enumerate(THRESHOLD_FACTORS)
        ]
        print(f"    {case:9s} {' '.join(cells)}")


def main(quick: bool = False) -> None:
    for size in SIZES:
        trials = size.quick_trials if quick else size.trials
        start = time.perf_counter()
        table, state_index = compile_grid(size_world(size))
        directory = run_dir(
            "pilot_rq1",
            {"grid": size.grid, "horizon": size.horizon, "n": size.budget, "c_values": C_VALUES,
             "threshold_factors": THRESHOLD_FACTORS},
        )
        results = []
        for seed in range(trials):
            result = run_trial(table, state_index, size.budget, seed)
            save_seed(directory, seed, **result)
            results.append(result)
        print_size(size, results, time.perf_counter() - start)
        print(f"  wrote {directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="run 5, 2 and 1 trials by size")
    main(parser.parse_args().quick)
