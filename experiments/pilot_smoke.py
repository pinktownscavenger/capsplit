"""Smoke test: evaluate the 4x4 trap case and write one result through runlog.

Run: python experiments/pilot_smoke.py
"""
from dataclasses import asdict

from phase1_rq1 import WORLD, lg, trap

from capsplit.engine import trajectory_probs
from capsplit.evaluate import evaluate
from capsplit.runlog import run_dir, save_seed
from capsplit.worlds import compile_grid, logits_to_array

N = 20
SEED = 0  # the case is exact, so the seed only names the result file
SCALARS = ("delta_rel", "delta_cap", "delta_j", "excess", "j_gap", "best_beta")


def main():
    table, state_index = compile_grid(WORLD)
    logits0 = logits_to_array(state_index, trap(3.2, 3.2, 0.0))
    logits1 = logits_to_array(state_index, lg(3.2, 3.2, 4.0))
    report = evaluate(table, logits0, trajectory_probs(table, logits1), N)

    directory = run_dir("pilot_smoke", {"case": "trap_boost_rare", "n": N})
    values = asdict(report)
    path = save_seed(directory, SEED, label=report.label, **{name: values[name] for name in SCALARS})
    print(f"label={report.label} e_n_valid={report.e_n_valid} " + " ".join(f"{s}={values[s]:+.4f}" for s in SCALARS))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
