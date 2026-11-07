"""RQ2: time to cross on an isolated route, competing routes, and training dynamics.

Run: python experiments/pilot_rq2.py [--quick]
"""
import argparse
import time

import numpy as np
from phase1_rq1 import WORLD as TRAP_WORLD
from phase1_rq1 import trap

from capsplit.engine import solution_probs, table_success_prob, trajectory_probs
from capsplit.evaluate import evaluate
from capsplit.runlog import run_dir, save_seed
from capsplit.tempering import frontier
from capsplit.train.tabular import exact_pg_gradient, reinforce, train_exact_pg
from capsplit.worlds import GridWorld, compile_grid, logits_to_array, policy_from_routes

ISOLATED_WALLS = frozenset({(r, c) for r in (1, 2, 3) for c in (1, 2, 3)} | {(0, 1)})
ISOLATED_WORLD = GridWorld(n=5, start=(0, 0), goal=(4, 4), H=8, walls=ISOLATED_WALLS)
COMPETING_WORLD = GridWorld(n=3, start=(0, 0), goal=(2, 2), H=4, walls=frozenset({(1, 1)}))
ROUTE_A = ((0, 0), (0, 1), (0, 2), (1, 2), (2, 2))
ROUTE_B = ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2))

LINE = 0.1  # the isolated route's line, N = 10
ARM_A_TARGETS = (1e-5, 1e-4, 1e-3, 1e-2)
ARM_A_LR = 1.0
ARM_A_CAP = 30000
ARM_B_BONUSES = ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (2, 1.5))
ARM_B_LR = 2.0
ARM_B_STEPS, ARM_B_QUICK_STEPS = 6000, 1000
CROSS_FACTOR = 10
DYNAMICS_N = 20
DYNAMICS_SEEDS = 5
DYNAMICS_LR = 1.0  # not fixed by the blueprint; chosen before the first dynamics run
DYNAMICS_STEPS = 1000
BISECTION_STEPS = 60


def compile_isolated():
    return compile_grid(ISOLATED_WORLD)


def compile_competing():
    return compile_grid(COMPETING_WORLD)


def isolated_base(table, state_index, target: float) -> np.ndarray:
    """Logits with one per-step bonus on the only route, bisected on [-8, 8] to hit q = target."""
    route = list(table.solutions[0])
    low, high = -8.0, 8.0
    for _ in range(BISECTION_STEPS):
        bonus = (low + high) / 2
        logits = logits_to_array(state_index, policy_from_routes(ISOLATED_WORLD, [route], [bonus]))
        if table_success_prob(table, trajectory_probs(table, logits)) < target:
            low = bonus
        else:
            high = bonus
    return logits_to_array(state_index, policy_from_routes(ISOLATED_WORLD, [route], [high]))


def competing_base(state_index, bonuses) -> np.ndarray:
    routes = [list(ROUTE_A), list(ROUTE_B)]
    return logits_to_array(state_index, policy_from_routes(COMPETING_WORLD, routes, bonuses))


def updates_to_reach(table, logits0, target: float, lr: float, max_steps: int):
    """Exact-PG updates until J reaches target, or None if it hasn't within max_steps."""
    logits = logits0.copy()
    for step in range(max_steps + 1):
        gradient, j = exact_pg_gradient(table, logits)
        if j >= target:
            return step
        logits += lr * gradient
    return None


def arm_a(quick: bool) -> None:
    targets = ARM_A_TARGETS[1:] if quick else ARM_A_TARGETS
    table, state_index = compile_isolated()
    updates = [
        updates_to_reach(table, isolated_base(table, state_index, q), LINE, ARM_A_LR, ARM_A_CAP) for q in targets
    ]
    slope = np.polyfit(np.log10(targets), np.log10(updates), 1)[0]
    print(f"\nArm A: isolated route, updates to J >= {LINE} at lr={ARM_A_LR}")
    for q, count in zip(targets, updates):
        print(f"  q0={q:.0e}: {count}")
    verdict = "inside" if -1.0 <= slope <= -0.7 else "OUTSIDE"
    print(f"  log-log slope {slope:.3f} ({verdict} [-1.0, -0.7])")
    directory = run_dir("pilot_rq2_arm_a", {"targets": targets, "line": LINE, "lr": ARM_A_LR})
    save_seed(directory, 0, targets=np.array(targets), updates=np.array(updates), slope=slope)


def arm_b(quick: bool) -> None:
    steps = ARM_B_QUICK_STEPS if quick else ARM_B_STEPS
    table, state_index = compile_competing()
    b_index = table.solutions.index(ROUTE_B)
    directory = run_dir("pilot_rq2_arm_b", {"bonuses": ARM_B_BONUSES, "lr": ARM_B_LR, "steps": steps})
    print(f"\nArm B: competing routes, lr={ARM_B_LR}, {steps} steps, crossing {CROSS_FACTOR} x q0(B)")
    print(f"  {'bonuses':>10} {'q0(B)':>9} {'peak':>9} {'final':>9} {'crossed at':>12}")
    for i, bonuses in enumerate(ARM_B_BONUSES):
        logits0 = competing_base(state_index, bonuses)
        q_b = [solution_probs(table, trajectory_probs(table, logits0))[b_index]]

        def log(_step, logits):
            q_b.append(solution_probs(table, trajectory_probs(table, logits))[b_index])

        train_exact_pg(table, logits0, ARM_B_LR, steps, on_step=log)
        q_b = np.array(q_b)
        above = np.nonzero(q_b >= CROSS_FACTOR * q_b[0])[0]
        crossed = int(above[0]) if len(above) else -1  # -1 means censored
        shown = crossed if crossed >= 0 else "censored"
        print(f"  {str(bonuses):>10} {q_b[0]:9.5f} {q_b.max():9.5f} {q_b[-1]:9.2e} {shown!s:>12}")
        save_seed(directory, i, bonuses=np.array(bonuses), q_b=q_b, crossed_at=crossed)


def _dynamics_row(table, logits0, best, logits) -> list:
    report = evaluate(table, logits0, trajectory_probs(table, logits), DYNAMICS_N, frontier_result=best)
    return [report.delta_rel, report.delta_cap, report.excess]


def _print_checkpoints(name: str, rows: np.ndarray) -> None:
    print(f"  {name}")
    print(f"    {'update':>7} {'delta_rel':>10} {'delta_cap':>10} {'excess':>9}")
    for update in (0, 10, 25, 50, 100, 200, 400, 1000):
        if update < len(rows):
            print(f"    {update:7d} {rows[update, 0]:+10.4f} {rows[update, 1]:+10.4f} {rows[update, 2]:+9.4f}")
    line = 1 / DYNAMICS_N
    rel_first = np.nonzero(rows[:, 0] >= line)[0]
    cap_first = np.nonzero(rows[:, 1] >= line)[0]
    print(
        f"    first update at or above c/N={line}: delta_rel "
        f"{rel_first[0] if len(rel_first) else 'never'}, delta_cap {cap_first[0] if len(cap_first) else 'never'}"
    )


def dynamics() -> None:
    table, state_index = compile_grid(TRAP_WORLD)
    logits0 = logits_to_array(state_index, trap(3.2, 3.2, 0.0))
    best = frontier(table, logits0)
    print(f"\nDynamics: 4x4 trap base, N={DYNAMICS_N}")

    directory = run_dir("pilot_rq2_dynamics", {"n": DYNAMICS_N, "lr": DYNAMICS_LR, "steps": DYNAMICS_STEPS})
    rows = [_dynamics_row(table, logits0, best, logits0)]

    def log_exact(_step, logits):
        rows.append(_dynamics_row(table, logits0, best, logits))

    train_exact_pg(table, logits0, DYNAMICS_LR, DYNAMICS_STEPS, on_step=log_exact)
    exact = np.array(rows)
    save_seed(directory, 0, exact_pg=exact)
    _print_checkpoints(f"Exact PG, lr={DYNAMICS_LR}", exact)

    runs = []
    for seed in range(DYNAMICS_SEEDS):
        rows = [_dynamics_row(table, logits0, best, logits0)]

        def log_reinforce(_step, logits, rows=rows):
            rows.append(_dynamics_row(table, logits0, best, logits))

        reinforce(table, logits0, seed, on_step=log_reinforce)
        runs.append(rows)
    runs = np.array(runs)
    save_seed(directory, 1, reinforce=runs)
    _print_checkpoints(f"REINFORCE, mean of {DYNAMICS_SEEDS} seeds", runs.mean(axis=0))
    finals = runs[:, -1]
    print(f"    final per seed (delta_rel, delta_cap): {np.round(finals[:, :2], 3).tolist()}")


def main(quick: bool = False) -> None:
    start = time.perf_counter()
    arm_a(quick)
    arm_b(quick)
    dynamics()
    print(f"\n{time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="skip q0=1e-5 and use 1,000 steps in arm B")
    main(parser.parse_args().quick)
