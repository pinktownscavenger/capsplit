"""Coverage of the D11 bound on the reference pair, at a Chernoff-sized and a generous M.

Run: python experiments/pilot_estimator.py
"""
import time

import numpy as np
from phase1_rq1 import WORLD, lg

from capsplit.engine import trajectory_probs
from capsplit.estimator import chernoff_size, estimate_split, exact_means, hoeffding_size, within_bound
from capsplit.runlog import run_dir, save_seed
from capsplit.worlds import compile_grid, logits_to_array

N = 20
MU, ETA, DELTA = 0.01, 0.2, 0.05
REPEATS = 200
GENEROUS_FACTOR = 10
SEED = 0


def coverage(table, logits0, probs1, m: int, rng) -> float:
    exact = exact_means(table, logits0, probs1, N)
    hits = sum(within_bound(estimate_split(table, logits0, probs1, N, m, rng), exact, ETA, MU) for _ in range(REPEATS))
    return hits / REPEATS


def main():
    table, state_index = compile_grid(WORLD)
    logits0 = logits_to_array(state_index, lg(2.5, 2.0, 0.0))
    probs1 = trajectory_probs(table, logits_to_array(state_index, lg(2.0, 2.0, 3.0)))

    chernoff = chernoff_size(MU, ETA, DELTA)
    generous = GENEROUS_FACTOR * chernoff
    hoeffding = hoeffding_size(ETA * MU, DELTA)  # same absolute error at m <= mu; computed only, not sampled

    rng = np.random.default_rng(SEED)
    start = time.perf_counter()
    chernoff_coverage = coverage(table, logits0, probs1, chernoff, rng)
    generous_coverage = coverage(table, logits0, probs1, generous, rng)
    seconds = time.perf_counter() - start

    print(f"exact means: {np.round(exact_means(table, logits0, probs1, N), 5)}")
    print(f"Chernoff M={chernoff}: coverage {chernoff_coverage:.3f} over {REPEATS} repeats")
    print(f"generous M={generous}: coverage {generous_coverage:.3f} over {REPEATS} repeats")
    print(f"Hoeffding M={hoeffding} (not sampled), {hoeffding / chernoff:.1f} times the Chernoff size")
    print(f"{seconds:.1f} s")

    directory = run_dir(
        "pilot_estimator",
        {"case": "reference", "n": N, "mu": MU, "eta": ETA, "delta": DELTA, "repeats": REPEATS,
         "generous_factor": GENEROUS_FACTOR},
    )
    path = save_seed(
        directory,
        SEED,
        chernoff_m=chernoff,
        generous_m=generous,
        hoeffding_m=hoeffding,
        chernoff_coverage=chernoff_coverage,
        generous_coverage=generous_coverage,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
