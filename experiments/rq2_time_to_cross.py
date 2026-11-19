"""RQ2 time to cross: isolated route (arm A), robustness, competing routes (arm B), sampled stretch.

Design: docs/rq2-time-to-cross-design.md. Plan: docs/rq2-time-to-cross-plan.md. The constants
below are frozen by docs/rq2-preregistration.md; changing one after the freeze is logged in
docs/DECISIONS.md and the affected runs are redone.

Run: python experiments/rq2_time_to_cross.py {arm-a,robust,arm-b,sampled,report} [--quick] [--root DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
import time

import numpy as np
from phase1_rq1 import R_COMMON, R_RARE
from phase1_rq1 import WORLD as PHASE1_WORLD
from pilot_rq2 import COMPETING_WORLD, ISOLATED_WALLS, ROUTE_A, ROUTE_B

from capsplit.crossing import crossing_step, integral_time, loglog_slopes, scalar_isolated_trace
from capsplit.runlog import load_seeds, run_dir, save_seed
from capsplit.train.tabular import exact_pg_step, reinforce, trace_exact_pg
from capsplit.worlds import MOVES, N_ACTIONS, GridWorld, compile_grid

# ----------------------------------------------------------------------------- frozen settings

LINE = 0.1  # arm A's reach line, c = 1 and N = 10
ARM_A_Q0 = tuple(float(q) for q in np.logspace(-6, -2, 9))
ARM_A_LRS = (0.5, 1.0, 2.0)
ARM_A_CAP = 400_000
ROBUST_SEEDS = tuple(range(20))
ROBUST_SIGMA = 1.0
ROBUST_H = 10
ARM_B_WORLDS = ("W1", "W2", "W3")
ARM_B_RATIOS = (1.0, 0.9, 0.5, 0.1, 0.01)
ARM_B_LRS = (0.5, 1.0, 2.0)
ARM_B_SEEDS = tuple(range(20))
ARM_B_CAP = 20_000
CROSS_FACTOR = 10
COMMON_BONUS = {"W1": 0.75, "W2": 1.0, "W3": 1.75}  # q0(common) about 0.02 at ratio 1, sigma 0 (RQ2-1)
BISECTION_STEPS = 60
BONUS_RANGE = (-8.0, 8.0)

# W3: the isolated world without the (0, 1) wall, which opens a second, top-right route.
W3_WORLD = GridWorld(n=5, start=(0, 0), goal=(4, 4), H=8, walls=frozenset(ISOLATED_WALLS - {(0, 1)}))
W3_TOP_RIGHT = ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (2, 4), (3, 4), (4, 4))
W3_LEFT_BOTTOM = ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4))

# ----------------------------------------------------------------------------- worlds and bases


def isolated_world(horizon: int = 8) -> GridWorld:
    """The pilot's isolated-route world (one correct route of length 8) with horizon `horizon`."""
    return GridWorld(n=5, start=(0, 0), goal=(4, 4), H=horizon, walls=ISOLATED_WALLS)


def competing_world(name: str) -> tuple[GridWorld, tuple, tuple]:
    """(world, common route, rare route) for arm B's worlds W1, W2 and W3."""
    if name == "W1":
        return COMPETING_WORLD, tuple(ROUTE_A), tuple(ROUTE_B)
    if name == "W2":
        return PHASE1_WORLD, tuple(R_COMMON), tuple(R_RARE)
    if name == "W3":
        return W3_WORLD, W3_TOP_RIGHT, W3_LEFT_BOTTOM
    raise ValueError(f"unknown world {name!r}; expected one of {ARM_B_WORLDS}")


def _route_slots(world: GridWorld, state_index: dict, route) -> list[tuple[int, int]]:
    """(state index, action) for each step of `route`."""
    slots = []
    for here, there in zip(route[:-1], route[1:]):
        move = (there[0] - here[0], there[1] - here[1])
        action = MOVES.index(move)
        assert world.step(here, action) == there, f"route step {here} -> {there} is blocked"
        slots.append((state_index[here], action))
    return slots


def _noise(state_index: dict, sigma: float, seed: int, keep: list[tuple[int, int]]) -> np.ndarray:
    """N(0, sigma^2) on every logit except the `keep` slots, which stay 0."""
    logits = np.zeros((len(state_index), N_ACTIONS))
    if sigma:
        logits += sigma * np.random.default_rng(seed).standard_normal(logits.shape)
        for s, a in keep:
            logits[s, a] = 0.0
    return logits


def _with_bonus(base: np.ndarray, slots: list[tuple[int, int]], bonus: float) -> np.ndarray:
    logits = base.copy()
    for s, a in slots:
        logits[s, a] += bonus
    return logits


def _bisect(value, target: float, what: str) -> float:
    """Bonus in BONUS_RANGE with value(bonus) = target, for value increasing in the bonus.

    Same rule as the pilot's `isolated_base`: below the target moves `low` up, and the result is
    `high`. Raises if the target is outside [value(low), value(high)].
    """
    low, high = BONUS_RANGE
    if not value(low) <= target <= value(high):
        raise ValueError(f"{what}: target {target:g} is outside [{value(low):g}, {value(high):g}] over {BONUS_RANGE}")
    for _ in range(BISECTION_STEPS):
        bonus = (low + high) / 2
        if value(bonus) < target:
            low = bonus
        else:
            high = bonus
    return high


def noisy_isolated_base(table, state_index: dict, world: GridWorld, target_q0: float, sigma: float, seed: int) -> np.ndarray:
    """Isolated-route base: off-route noise, then one per-step route bonus bisected to hit target_q0."""
    slots = _route_slots(world, state_index, table.solutions[0])
    base = _noise(state_index, sigma, seed, slots)

    def q0(bonus: float) -> float:
        return float(exact_pg_step(table, _with_bonus(base, slots, bonus))[1].sum())

    return _with_bonus(base, slots, _bisect(q0, target_q0, "isolated base"))


def competing_base(table, state_index: dict, name: str, ratio: float, sigma: float, seed: int) -> tuple[np.ndarray, float, float]:
    """(logits, q0(common), q0(rare)) with q0(rare) / q0(common) = ratio.

    Noise goes on every logit off both bonus routes. The common route's bonus is COMMON_BONUS[name];
    the rare route's is bisected. Both q0 are recomputed at each step, since W2's routes share moves.
    """
    world, common, rare = competing_world(name)
    c_slots, r_slots = _route_slots(world, state_index, common), _route_slots(world, state_index, rare)
    base = _with_bonus(_noise(state_index, sigma, seed, c_slots + r_slots), c_slots, COMMON_BONUS[name])
    i_common, i_rare = table.solutions.index(common), table.solutions.index(rare)

    def ratio_at(bonus: float) -> float:
        q = exact_pg_step(table, _with_bonus(base, r_slots, bonus))[1]
        return float(q[i_rare] / q[i_common])

    logits = _with_bonus(base, r_slots, _bisect(ratio_at, ratio, f"competing base {name}"))
    q = exact_pg_step(table, logits)[1]
    return logits, float(q[i_common]), float(q[i_rare])


# ----------------------------------------------------------------------------- runs

QUICK_Q0 = (1e-4, 1e-3, 1e-2)


def _updates(q: np.ndarray, line: float) -> int:
    """Time to cross as an int for saving: -1 means censored."""
    step = crossing_step(q, line)
    return -1 if step is None else step


def _log(message: str) -> None:
    print(message, flush=True)


def run_arm_a(quick: bool, root: str = "results") -> pathlib.Path:
    """Arm A main curve: full-table exact PG against the scalar recursion and the integral."""
    q0s = QUICK_Q0 if quick else ARM_A_Q0
    lrs = (1.0,) if quick else ARM_A_LRS
    config = {"arm": "A", "q0": list(q0s), "lrs": list(lrs), "line": LINE, "cap": ARM_A_CAP, "horizon": 8, "quick": quick}
    directory = run_dir("rq2_arm_a", config, root=root)
    world = isolated_world(8)
    table, state_index = compile_grid(world)
    _log(f"arm A -> {directory}")
    i = 0
    for lr in lrs:
        for target in q0s:
            start = time.perf_counter()
            logits0 = noisy_isolated_base(table, state_index, world, target, 0.0, 0)
            q, _ = trace_exact_pg(table, logits0, lr, ARM_A_CAP, [0], stop_line=LINE)
            q0 = float(q[0, 0])
            q_scalar = scalar_isolated_trace(q0, 8, lr, ARM_A_CAP, stop_line=LINE)
            updates, updates_scalar = _updates(q[:, 0], LINE), _updates(q_scalar, LINE)
            integral = integral_time(q0, 8, lr, LINE)
            save_seed(directory, i, lr=lr, q0_target=target, q0=q0, q=q, q_scalar=q_scalar,
                      updates=updates, updates_scalar=updates_scalar, integral=integral)
            _log(f"  lr={lr:<4} q0={target:.1e}: {updates:>7} updates (scalar {updates_scalar}, "
                 f"integral {integral:,.0f}) {time.perf_counter() - start:.1f} s")
            i += 1
    return directory


def run_robust(quick: bool, root: str = "results") -> pathlib.Path:
    """Arm A robustness: R1 off-route noise draws at H = 8, R2 a longer horizon without noise. lr = 1."""
    q0s = QUICK_Q0 if quick else ARM_A_Q0
    draws = ROBUST_SEEDS[:2] if quick else ROBUST_SEEDS
    horizon = 9 if quick else ROBUST_H
    config = {"arm": "robust", "q0": list(q0s), "draws": list(draws), "sigma": ROBUST_SIGMA,
              "r2_horizon": horizon, "lr": 1.0, "line": LINE, "cap": ARM_A_CAP, "quick": quick}
    directory = run_dir("rq2_robust", config, root=root)
    _log(f"robustness -> {directory}")
    jobs = [("R1", 8, ROBUST_SIGMA, d) for d in draws] + [("R2", horizon, 0.0, 0)]
    tables = {h: (isolated_world(h), *compile_grid(isolated_world(h))) for h in {8, horizon}}
    i = 0
    for condition, h, sigma, draw in jobs:
        world, table, state_index = tables[h]
        for target in q0s:
            logits0 = noisy_isolated_base(table, state_index, world, target, sigma, draw)
            q, _ = trace_exact_pg(table, logits0, 1.0, ARM_A_CAP, [0], stop_line=LINE)
            updates = _updates(q[:, 0], LINE)
            save_seed(directory, i, condition=condition, horizon=h, sigma=sigma, draw=draw,
                      q0_target=target, q0=float(q[0, 0]), updates=updates)
            i += 1
        _log(f"  {condition} H={h} draw={draw}: done")
    return directory


def run_arm_b(quick: bool, root: str = "results") -> pathlib.Path:
    """Arm B: competing routes under exact PG, over worlds x ratios x lrs x noise draws.

    One file per (world, ratio, lr), with arrays over draws. Crossing is computed in float64
    before the traces are stored as float32.
    """
    worlds = ("W1",) if quick else ARM_B_WORLDS
    ratios = (1.0, 0.1) if quick else ARM_B_RATIOS
    lrs = (2.0,) if quick else ARM_B_LRS
    draws = ARM_B_SEEDS[:2] if quick else ARM_B_SEEDS
    cap = 2_000 if quick else ARM_B_CAP
    config = {"arm": "B", "worlds": list(worlds), "ratios": list(ratios), "lrs": list(lrs), "draws": list(draws),
              "sigma": ROBUST_SIGMA, "cap": cap, "cross_factor": CROSS_FACTOR, "line": LINE,
              "common_bonus": COMMON_BONUS, "quick": quick}
    directory = run_dir("rq2_arm_b", config, root=root)
    _log(f"arm B -> {directory}")
    i = 0
    for name in worlds:
        world, common, rare = competing_world(name)
        table, state_index = compile_grid(world)
        watch = [table.solutions.index(rare), table.solutions.index(common)]
        for ratio in ratios:
            for lr in lrs:
                fields: dict[str, list] = {k: [] for k in ("q0_common", "q0_rare", "crossed_primary", "crossed_line",
                                                           "line_applicable", "peak", "final", "q_rare", "q_common")}
                for draw in draws:
                    logits0, q0_common, q0_rare = competing_base(table, state_index, name, ratio, ROBUST_SIGMA, draw)
                    q, _ = trace_exact_pg(table, logits0, lr, cap, watch)
                    applicable = q0_rare < LINE
                    fields["q0_common"].append(q0_common)
                    fields["q0_rare"].append(q0_rare)
                    fields["crossed_primary"].append(_updates(q[:, 0], CROSS_FACTOR * q0_rare))
                    fields["crossed_line"].append(_updates(q[:, 0], LINE) if applicable else -1)
                    fields["line_applicable"].append(applicable)
                    fields["peak"].append(float(q[:, 0].max()))
                    fields["final"].append(float(q[-1, 0]))
                    fields["q_rare"].append(q[:, 0].astype(np.float32))
                    fields["q_common"].append(q[:, 1].astype(np.float32))
                arrays = {k: np.array(v) for k, v in fields.items()}
                save_seed(directory, i, world=name, ratio=ratio, lr=lr, draws=np.array(draws), **arrays)
                crossed = int((arrays["crossed_primary"] >= 0).sum())
                _log(f"  {name} ratio={ratio:<5} lr={lr:<4}: {crossed}/{len(draws)} crossed 10 x q0(rare)")
                i += 1
    return directory


SAMPLED_SEEDS = tuple(range(20))
SAMPLED_Q0 = (1e-4, 1e-3, 1e-2)
SAMPLED_RATIOS = (1.0, 0.5, 0.1)
SAMPLED_CAP = 20_000
SAMPLED_BATCH, SAMPLED_LR, SAMPLED_DECAY = 32, 0.5, 0.9  # the pilot's REINFORCE defaults


def _reinforce_trace(table, logits0, seed: int, steps: int, watch: list[int]) -> np.ndarray:
    """q of the watched solutions at updates 0..steps under REINFORCE."""
    rows = [exact_pg_step(table, logits0)[1][watch]]

    def record(_step, logits):
        rows.append(exact_pg_step(table, logits)[1][watch])

    reinforce(table, logits0, seed, steps=steps, batch=SAMPLED_BATCH, lr=SAMPLED_LR,
              baseline_decay=SAMPLED_DECAY, on_step=record)
    return np.array(rows)


def run_sampled(quick: bool, root: str = "results") -> pathlib.Path:
    """Stretch: REINFORCE on arm A's world and on W1, with time to cross in updates and in episodes."""
    q0s = (1e-2,) if quick else SAMPLED_Q0
    ratios = () if quick else SAMPLED_RATIOS
    seeds = SAMPLED_SEEDS[:2] if quick else SAMPLED_SEEDS
    cap = 2_000 if quick else SAMPLED_CAP
    config = {"arm": "sampled", "q0": list(q0s), "ratios": list(ratios), "seeds": list(seeds), "cap": cap,
              "batch": SAMPLED_BATCH, "lr": SAMPLED_LR, "baseline_decay": SAMPLED_DECAY, "line": LINE,
              "cross_factor": CROSS_FACTOR, "common_bonus": COMMON_BONUS, "quick": quick}
    directory = run_dir("rq2_sampled", config, root=root)
    _log(f"sampled -> {directory}")
    i = 0
    world = isolated_world(8)
    table, state_index = compile_grid(world)
    for target in q0s:
        logits0 = noisy_isolated_base(table, state_index, world, target, 0.0, 0)
        updates = np.array([_updates(_reinforce_trace(table, logits0, s, cap, [0])[:, 0], LINE) for s in seeds])
        episodes = np.where(updates >= 0, SAMPLED_BATCH * updates, -1)
        save_seed(directory, i, arm="A", q0_target=target, seeds=np.array(seeds), updates=updates, episodes=episodes)
        _log(f"  A q0={target:.0e}: {int((updates >= 0).sum())}/{len(seeds)} crossed {LINE}")
        i += 1
    if ratios:
        w1, common, rare = competing_world("W1")
        table, state_index = compile_grid(w1)
        watch = [table.solutions.index(rare), table.solutions.index(common)]
        for ratio in ratios:
            logits0, _, q0_rare = competing_base(table, state_index, "W1", ratio, 0.0, 0)
            traces = [_reinforce_trace(table, logits0, s, cap, watch) for s in seeds]
            updates = np.array([_updates(t[:, 0], CROSS_FACTOR * q0_rare) for t in traces])
            episodes = np.where(updates >= 0, SAMPLED_BATCH * updates, -1)
            rare_wins = np.array([t[-1, 0] > t[-1, 1] for t in traces])
            save_seed(directory, i, arm="B", ratio=ratio, q0_rare=q0_rare, seeds=np.array(seeds),
                      updates=updates, episodes=episodes, rare_wins=rare_wins)
            _log(f"  B W1 ratio={ratio}: {int((updates >= 0).sum())}/{len(seeds)} crossed, rare ends ahead in {int(rare_wins.sum())}")
            i += 1
    return directory



# ----------------------------------------------------------------------------- verdicts (design §7)

P1_NOTE = "Settled by the derivation in docs/proofs.md section 7: the full-table run equals the scalar recursion."


def p1_verdict(q0s, updates, local_cutoff: float = -0.9) -> dict:
    """P1: supported if the fitted slope is in [-1.0, -0.7] and the smallest-pair local slope <= cutoff."""
    pairs = sorted(zip(q0s, updates))
    if any(u is None for _, u in pairs):
        return {"verdict": "inconclusive (censored)", "note": P1_NOTE}
    fitted, local = loglog_slopes([q for q, _ in pairs], [u for _, u in pairs])
    if fitted > -0.3:
        verdict = "contradicted"
    elif -1.0 <= fitted <= -0.7 and local[0] <= local_cutoff:
        verdict = "supported"
    else:
        verdict = "inconclusive"
    return {"verdict": verdict, "fitted": fitted, "local": [float(x) for x in local], "note": P1_NOTE}


def p2_verdict(r1_slopes, r2_slope) -> dict:
    """P2: supported if >= 90% of R1 draws (18/20) and R2 have slope <= -0.7; contradicted if R1's median or R2 > -0.3.

    A censored draw (None) counts as not <= -0.7, and is left out of the median.
    """
    n = len(r1_slopes)
    good = sum(1 for x in r1_slopes if x is not None and x <= -0.7)
    finite = [x for x in r1_slopes if x is not None]
    median = float(np.median(finite)) if finite else None
    result = {"r1_at_most_-0.7": good, "r1_draws": n, "r1_median": median, "r2_slope": r2_slope}
    if (median is not None and median > -0.3) or (r2_slope is not None and r2_slope > -0.3):
        result["verdict"] = "contradicted"
    elif good >= math.ceil(0.9 * n) and r2_slope is not None and r2_slope <= -0.7:
        result["verdict"] = "supported"
    else:
        result["verdict"] = "inconclusive"
    return result


def p3_verdict(table: dict) -> dict:
    """P3: lr x updates varies by < 5% across learning rates at every q0. Reported, not a contradiction test."""
    spreads = {}
    for q0, by_lr in table.items():
        if len(by_lr) < 2 or any(u is None for u in by_lr.values()):
            return {"verdict": "not computed", "reason": "needs two or more learning rates and no censored points"}
        products = [lr * u for lr, u in by_lr.items()]
        spreads[q0] = (max(products) - min(products)) / float(np.mean(products))
    if not spreads:
        return {"verdict": "not computed", "reason": "no data"}
    worst = max(spreads.values())
    return {"verdict": "holds" if worst < 0.05 else "does not hold", "worst_spread": worst,
            "spreads": {str(k): v for k, v in spreads.items()}}


def p4_verdict(fractions: dict) -> dict:
    """P4 (primary rule): supported if every cell with ratio <= 0.5 has fraction <= 0.1;
    contradicted if any cell with ratio <= 0.1 has fraction >= 0.5."""
    low = {k: v for k, v in fractions.items() if k[1] <= 0.5}
    if not low:
        return {"verdict": "not computed", "reason": "no cells with ratio <= 0.5"}
    worst = max(low.values())
    if any(v >= 0.5 for k, v in low.items() if k[1] <= 0.1):
        verdict = "contradicted"
    elif worst <= 0.1:
        verdict = "supported"
    else:
        verdict = "inconclusive"
    return {"verdict": verdict, "worst_fraction_at_ratio_le_0.5": worst}


# ----------------------------------------------------------------------------- report


def _latest_run(root: str, experiment: str, command: str, required: bool = True) -> pathlib.Path | None:
    runs = sorted(pathlib.Path(root, experiment).glob("*/manifest.json"), key=lambda m: m.stat().st_mtime)
    if not runs:
        if required:
            print(f"error: no results under {pathlib.Path(root, experiment)}; "
                  f"run: python experiments/rq2_time_to_cross.py {command} --root {root}", file=sys.stderr)
            raise SystemExit(2)
        return None
    full = [m for m in runs if not json.loads(m.read_text()).get("config", {}).get("quick", False)]
    pool = full or runs  # a --quick run is used only when no full run exists, and the report flags it
    if len(runs) > 1:
        _log(f"note: {len(runs)} runs under {experiment} ({len(full)} full); using {pool[-1].parent.name}")
    return pool[-1].parent


def _is_quick(directory: pathlib.Path | None) -> bool:
    if directory is None:
        return False
    return bool(json.loads((directory / "manifest.json").read_text()).get("config", {}).get("quick", False))


def _none(value) -> int | None:
    value = int(value)
    return None if value < 0 else value


def _write_csv(path: pathlib.Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _slope_or_none(q0s, updates):
    try:
        return loglog_slopes(q0s, updates)[0]
    except ValueError:
        return None


def report(root: str = "results", out: str | None = None) -> dict:
    """Tables, figures, verdicts and validity checks from the newest run of each arm."""
    dirs = {name: _latest_run(root, f"rq2_{name}", command, required=name != "sampled")
            for name, command in (("arm_a", "arm-a"), ("robust", "robust"), ("arm_b", "arm-b"), ("sampled", "sampled"))}
    out_dir = pathlib.Path(out) if out else pathlib.Path(root, "rq2_report")
    arm_a, robust, arm_b = (load_seeds(dirs[k]) for k in ("arm_a", "robust", "arm_b"))
    sampled = load_seeds(dirs["sampled"]) if dirs["sampled"] else []

    # Arm A
    rows, by_lr = [], {}
    for r in arm_a:
        by_lr.setdefault(float(r["lr"]), []).append(r)
    for lr, runs in sorted(by_lr.items()):
        runs.sort(key=lambda r: float(r["q0_target"]))
        ups = [_none(r["updates"]) for r in runs]
        local = [None] + list(loglog_slopes([float(r["q0_target"]) for r in runs], ups)[1]) if None not in ups and len(runs) > 1 else [None] * len(runs)
        for r, loc in zip(runs, local):
            rows.append([lr, float(r["q0_target"]), float(r["q0"]), int(r["updates"]), int(r["updates_scalar"]),
                         float(r["integral"]), "" if loc is None else loc])
    _write_csv(out_dir / "tables/arm_a.csv", ["lr", "q0_target", "q0", "updates", "updates_scalar", "integral",
                                               "local_slope_from_previous"], rows)
    primary = sorted(by_lr.get(1.0, []), key=lambda r: float(r["q0_target"]))
    p1 = p1_verdict([float(r["q0_target"]) for r in primary], [_none(r["updates"]) for r in primary])
    p3 = p3_verdict({float(r["q0_target"]): {lr: _none(x["updates"]) for lr, runs in by_lr.items() for x in runs
                                              if float(x["q0_target"]) == float(r["q0_target"])} for r in primary})

    # Robustness
    groups: dict = {}
    for r in robust:
        groups.setdefault((str(r["condition"]), int(r["draw"]), int(r["horizon"])), []).append(r)
    rob_rows, r1_slopes, r2_slope = [], [], None
    for (condition, draw, horizon), runs in sorted(groups.items()):
        runs.sort(key=lambda r: float(r["q0_target"]))
        slope = _slope_or_none([float(r["q0_target"]) for r in runs], [_none(r["updates"]) for r in runs])
        if condition == "R1":
            r1_slopes.append(slope)
        else:
            r2_slope = slope
        for r in runs:
            rob_rows.append([condition, draw, horizon, float(r["q0_target"]), float(r["q0"]), int(r["updates"]),
                             "" if slope is None else slope])
    _write_csv(out_dir / "tables/robust.csv", ["condition", "draw", "horizon", "q0_target", "q0", "updates",
                                                "fitted_slope_of_draw"], rob_rows)
    p2 = p2_verdict(r1_slopes, r2_slope)

    # Arm B
    b_rows, fractions = [], {}
    for r in sorted(arm_b, key=lambda r: (str(r["world"]), -float(r["ratio"]), float(r["lr"]))):
        key = (str(r["world"]), float(r["ratio"]), float(r["lr"]))
        crossed = r["crossed_primary"] >= 0
        fractions[key] = float(crossed.mean())
        applicable = r["line_applicable"].astype(bool)
        line_frac = float((r["crossed_line"][applicable] >= 0).mean()) if applicable.any() else float("nan")
        median_up = float(np.median(r["crossed_primary"][crossed])) if crossed.any() else float("nan")
        b_rows.append([*key, fractions[key], line_frac, median_up, float(np.median(r["peak"])),
                       float(np.median(r["final"])), int((~crossed).sum()), len(crossed)])
    _write_csv(out_dir / "tables/arm_b.csv", ["world", "ratio", "lr", "fraction_crossing_primary", "fraction_crossing_line",
                                               "median_updates_among_crossing", "median_peak", "median_final",
                                               "censored", "draws"], b_rows)
    p4 = p4_verdict(fractions)

    # Sampled (stretch)
    s_rows = []
    for r in sampled:
        ups = r["updates"]
        ok = ups >= 0
        s_rows.append([str(r["arm"]), float(r["q0_target"]) if "q0_target" in r else "", float(r["ratio"]) if "ratio" in r else "",
                       int(ok.sum()), len(ups), float(np.median(ups[ok])) if ok.any() else float("nan"),
                       float(np.median(r["episodes"][ok])) if ok.any() else float("nan"),
                       int(r["rare_wins"].sum()) if "rare_wins" in r else ""])
    _write_csv(out_dir / "tables/sampled.csv", ["arm", "q0_target", "ratio", "crossed", "seeds", "median_updates",
                                                 "median_episodes", "rare_ends_ahead"], s_rows)

    # Validity
    validity = {
        "arm_a_matches_scalar": all(int(r["updates"]) == int(r["updates_scalar"])
                                    and len(r["q"]) == len(r["q_scalar"])
                                    and float(np.max(np.abs(r["q"][:, 0] - r["q_scalar"]))) <= 1e-9 for r in arm_a),
        "bases_within_1pct": all(abs(float(r["q0"]) / float(r["q0_target"]) - 1) <= 0.01 for r in [*arm_a, *robust])
        and all(np.all(np.abs(r["q0_rare"] / r["q0_common"] / float(r["ratio"]) - 1) <= 0.01) for r in arm_b),
        "arm_b_rule_achievable": all(bool(np.all(CROSS_FACTOR * r["q0_rare"] <= 0.5)) for r in arm_b),
    }
    verdicts = {"P1": p1, "P2": p2, "P3": p3, "P4": p4, "validity": validity,
                "overall": "valid" if all(validity.values()) else "invalid",
                "quick_runs_used": any(_is_quick(d) for d in dirs.values()),
                "runs": {k: str(v) if v else None for k, v in dirs.items()}}

    _figures(out_dir, by_lr, groups, arm_b)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "verdicts.json").write_text(json.dumps(verdicts, indent=2, default=float) + "\n")
    (out_dir / "verdicts.md").write_text(_verdicts_md(verdicts))
    _log(f"report -> {out_dir}: P1 {p1['verdict']}, P2 {p2['verdict']}, P3 {p3['verdict']}, P4 {p4['verdict']}, "
         f"{verdicts['overall']}")
    return verdicts


def _verdicts_md(v: dict) -> str:
    lines = ["# RQ2 time-to-cross verdicts", "", f"**Validity:** {v['overall']}", ""]
    if v["quick_runs_used"]:
        lines += ["**Warning: built from --quick runs. These are smoke-test numbers, not the pre-registered result.**", ""]
    lines += [f"- {k}: {'pass' if ok else 'FAIL'}" for k, ok in v["validity"].items()]
    lines += ["", "| Prediction | Verdict | Numbers |", "|---|---|---|"]
    for key in ("P1", "P2", "P3", "P4"):
        numbers = {k: x for k, x in v[key].items() if k not in ("verdict", "note")}
        lines.append(f"| {key} | {v[key]['verdict']} | `{json.dumps(numbers, default=float)}` |")
    lines += ["", f"P1 note: {P1_NOTE}", ""]
    return "\n".join(lines)


def _figures(out_dir: pathlib.Path, by_lr: dict, groups: dict, arm_b: list) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Arm A: measured points, the integral, and the R1 min-max band.
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for lr, runs in sorted(by_lr.items()):
        runs = sorted(runs, key=lambda r: float(r["q0_target"]))
        pts = [(float(r["q0"]), int(r["updates"])) for r in runs if int(r["updates"]) >= 0]
        if pts:
            ax.plot(*zip(*pts), "o", label=f"measured, lr={lr:g}")
        dense = np.logspace(np.log10(min(float(r["q0"]) for r in runs)), np.log10(max(float(r["q0"]) for r in runs)), 100)
        ax.plot(dense, [integral_time(q, 8, lr, LINE) for q in dense], "-", lw=1, alpha=0.6, label=f"integral, lr={lr:g}")
    r1: dict = {}
    for (condition, _draw, _h), runs in groups.items():
        if condition == "R1":
            for r in runs:
                if int(r["updates"]) >= 0:
                    r1.setdefault(float(r["q0_target"]), []).append(int(r["updates"]))
    if r1:
        qs = sorted(r1)
        ax.fill_between(qs, [min(r1[q]) for q in qs], [max(r1[q]) for q in qs], alpha=0.2, label="R1 noise draws, min-max")
    ax.set(xscale="log", yscale="log", xlabel="base probability of the route, q0", ylabel=f"updates to reach J = {LINE}",
           title="Arm A: time to cross")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "arm_a_loglog.png", dpi=150)
    plt.close(fig)

    # Arm B: fraction crossing (primary rule), one panel per lr; rows are worlds, columns ratios.
    worlds = sorted({str(r["world"]) for r in arm_b})
    ratios = sorted({float(r["ratio"]) for r in arm_b}, reverse=True)
    lrs = sorted({float(r["lr"]) for r in arm_b})
    fig, axes = plt.subplots(1, len(lrs), figsize=(3.2 * len(lrs) + 1, 2.8), squeeze=False)
    for ax, lr in zip(axes[0], lrs):
        grid = np.full((len(worlds), len(ratios)), np.nan)
        for r in arm_b:
            if float(r["lr"]) == lr:
                grid[worlds.index(str(r["world"])), ratios.index(float(r["ratio"]))] = float((r["crossed_primary"] >= 0).mean())
        image = ax.imshow(grid, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set(xticks=range(len(ratios)), xticklabels=[f"{x:g}" for x in ratios], yticks=range(len(worlds)),
               yticklabels=worlds, xlabel="q0(rare) / q0(common)", title=f"lr = {lr:g}")
        for (i, j), value in np.ndenumerate(grid):
            if not np.isnan(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="w" if value < 0.5 else "k", fontsize=8)
    fig.colorbar(image, ax=axes[0].tolist(), label="fraction crossing 10 x q0(rare)")
    fig.savefig(fig_dir / "arm_b_fraction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Arm B: median q(rare)/q0(rare) over updates, at the lr closest to 1.
    lr_show = min(lrs, key=lambda x: abs(x - 1.0))
    fig, axes = plt.subplots(1, len(worlds), figsize=(4 * len(worlds), 3.2), squeeze=False)
    for ax, world in zip(axes[0], worlds):
        for r in sorted(arm_b, key=lambda r: -float(r["ratio"])):
            if str(r["world"]) == world and float(r["lr"]) == lr_show:
                rel = r["q_rare"].astype(float) / r["q0_rare"][:, None]
                ax.plot(np.median(rel, axis=0), label=f"ratio {float(r['ratio']):g}")
        ax.axhline(CROSS_FACTOR, ls="--", color="grey", lw=0.8)
        ax.set(yscale="log", xlabel="update", ylabel="median q(rare) / q0(rare)", title=f"{world}, lr = {lr_show:g}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "arm_b_traces.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- command line

RUNNERS = {"arm-a": run_arm_a, "robust": run_robust, "arm-b": run_arm_b, "sampled": run_sampled}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=sorted([*RUNNERS, "report"]))
    parser.add_argument("--quick", action="store_true", help="small settings for a smoke run; never use with --root results")
    parser.add_argument("--root", default="results", help="results root (default: results)")
    args = parser.parse_args(argv)
    start = time.perf_counter()
    if args.command == "report":
        report(args.root)
    else:
        RUNNERS[args.command](quick=args.quick, root=args.root)
    _log(f"{args.command}: {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1:])
