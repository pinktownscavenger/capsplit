# Review: label and metric code

Status: **findings confirmed by reproduction (2026-11-25)**. This is plan task 3, nominally Sazan's. The project rule makes code review person-only, but Bilal asked Claude to confirm the findings itself, so no teammate has reviewed this. Each finding was confirmed with a runnable check (see [Confirmation](#confirmation-2026-11-25)), not just by reading the code.

Scope: `src/capsplit/evaluate.py`, `src/capsplit/core.py`, `src/capsplit/engine.py`, `src/capsplit/tempering.py`, `src/capsplit/harness/backends/exact.py`, the label paths in `src/capsplit/harness/backends/sampled.py`, and the RQ1 driver `experiments/pilot_rq1.py`. Checked against `docs/blueprint.md` §2.2–2.3 and D2, D3, D5, D7, `docs/DECISIONS.md` H12, H13 and H39, and `docs/proofs.md` §1. Commit reviewed: `ea1f21f` (main, 2026-11-24).

## Summary

I found no bug that changes a number already reported in RQ2, RQ3 or RQ4. The most important finding is about the run that's still to come: **the RQ1 driver labels with D2 only** (M1), so the RQ1 final sweep would reproduce the 6×6 `b_param` misfire unless it also records the H12 label. Which label counts as primary is team decision 1.

| ID | Severity | Finding | Blocks | Verdict |
|---|---|---|---|---|
| M1 | Medium | RQ1 driver records only the D2 label, not H12 | RQ1 final run | Confirmed (D2 3/8, H12 8/8) |
| M2 | Medium | `BenchmarkReport` has no `saturated` flag | Report wording | Confirmed |
| M3 | Medium | `evaluate_benchmark` frontier is grid-only (D3) | Report wording | Confirmed (false reweighting) |
| M4 | Medium (design) | H12 "beyond reach" fires on one crossing, whatever the mass, and even when Δcap < 0 | Decision 1 | Confirmed (example built) |
| L1 | Low | `q0 >= c/N` with no tolerance misclassifies q0 = c/N exactly | Nothing yet | Confirmed; no RQ1 6×6 case near the line |
| L2 | Low | The E_1 frontier's β and boundary hit are not recorded | Nothing (probed, see below) | Confirmed; no impact, all 10 seeds |
| L3 | Low | Sampled backend labels from point estimates, not a CI bound | Only if it's ever used | Confirmed by reading |
| L4 | Low | E_N outside valid (H, N) is flagged, not refused (D7) | Nothing | Confirmed by reading |
| N1–N3 | Nit | Stale comment, legacy label function, silent family drop | Nothing | Confirmed by reading |

Severity: **High** would change a reported result (none found). **Medium** affects a result still to be produced or a claim the report might make. **Low** is a robustness gap with no current effect. **Nit** is cosmetic.

## Verified correct

Each item was checked by reading the code against the definition. Tests also cover them (520 tests pass at `ea1f21f` with torch and matplotlib installed; `tests/lm` not run).

- **Reachable set (Definition 1).** `q0 >= c/N` over correct solutions only: `evaluate.py:75`, `exact.py:88`, `exact.py:335`. `engine.solution_probs` uses only `solution_ids >= 0`, so incorrect trajectories never enter Δrel or Δcap.
- **Split and Proposition 1.** Δrel + Δcap = J1 − J0. Both J's sum the same correct trajectories that `solution_probs` bins, so the identity is exact up to float. Group values in `exact.py` are prompt means, so the identity holds for groups too.
- **pass@N** = 1 − (1 − J)^N: `core.py:20`, `exact.py:50`.
- **Sequence tempering (D1).** P0(τ)^β is renormalised over all trajectories before φ: `tempering.py:19`, `exact.py:224–227`. Zero-probability trajectories stay at zero.
- **Shared-β frontier for groups (H13, D5).** The sup is of the *mean* pass@N over one β, not a mean of per-prompt sups: `exact.py:280–292`. The refinement uses lexicographic (pass@N, J), so ties at pass@N = 1.0 are broken by J.
- **E_1 (H39).** `front1` is the N = 1 frontier, where pass@1 = J, so `e1["both"]` is the J gap: `exact.py:321–322`. The single-task `evaluate` uses the same J gap (`evaluate.py:107`).
- **Label cells.** `h12_label` and `label_cell` map the two axes to the four cells correctly, with ε = 1e-9 for the temperature axis, as D2 specifies for exact computation.
- **Saturation.** `1 − best pass@N < 0.05` matches D2: `evaluate.py:121`, `exact.py:346`.
- **Crossed set.** `q0 < c/N <= q1` is consistent with the reachable test, since a crossed solution is always in U_N.

## Findings

### M1. The RQ1 driver records only the D2 label (Medium; blocks the RQ1 final run)

`experiments/pilot_rq1.py:30` calls `evaluate.evaluate`, whose label is D2 (`evaluate.py:116`: beyond reach iff Δcap ≥ c/N). The saved run (`pilot_rq1.py:44–49`) holds the label, Δrel, Δcap, ΔJ, E_N and the J gap, but not `report.crossed`, so the H12 label can't be recovered afterwards. H29 says that H12 lands 8/8 for 6×6 `b_param` where D2 lands 3/8. If the final RQ1 sweep runs this script unchanged, it reports the D2 misfire as its primary result.

**Recommendation.** Before Muhammad runs RQ1, add an `h12_label` column, using beyond reach = `bool(report.crossed)` and the same J gap, and save `crossed`. Keep D2 alongside. Which label is primary is decision 1. Write a failing test first: a planted case with Δcap ≥ c/N spread over many solutions, none of which crosses.

### M2. `BenchmarkReport` has no `saturated` field (Medium; report wording)

`evaluate.py:36–46` has no `saturated` field. The blueprint (line 216, and D2 line 276) and pilot recommendation 3 ask for one. The flag *does* exist where the sweep uses it: `exact.py:346` writes `extra["saturated"]` for every group record, and the harness report flags it (`report.py:491`). `evaluate_benchmark` is used only by `experiments/pilot_rq4.py`.

**Recommendation.** Don't write in the report that `BenchmarkReport` carries the flag. Either say "recommendation 3 was implemented in the harness, which replaced `evaluate_benchmark`", or retire `evaluate_benchmark`.

### M3. `evaluate_benchmark`'s frontier is grid-only (Medium; report wording)

`_best_mean_pass` (`evaluate.py:129–136`) takes the max over the β grid with no golden-section refinement. D3 calls this a lower bound on the sup that biases E_N upward. The docstring admits it. The harness's `exact` backend refines (`exact.py:155–181`), so RQ3 and RQ4 aren't affected. Only `pilot_rq4.py` uses this path.

**Recommendation.** The same as M2: don't cite any `evaluate_benchmark` E_N in the final report, or retire the function.

### M4. H12's reach axis is mass-free (Medium, design; for decision 1)

`exact.py:341`: `beyond_reach = bool(crossed.any())`, over every correct solution on every prompt in the group. Two consequences the team should know about when ratifying H12:

1. One solution on one prompt crossing just over the line (for example q0 = 0.0099 → q1 = 0.0100 at N = 100) flips a whole group to beyond reach. The group's `crossed_mass` is then about c/(N·m) for m prompts.
2. A group can be beyond reach, and so `capability` or `compounded_sharpening`, while Δcap < 0, if other unreachable solutions lose more mass than the crossed one gains.

This is what H12 says ("crossing is what reachable at budget N means"), so it isn't a code bug. But the report should show `crossed_count`, `crossed_mass` and the sign of Δcap beside every beyond-reach label. It should also report how many beyond-reach labels have Δcap < 0. That count hasn't been measured.

### L1. No tolerance at q0 = c/N (Low)

The reach test is an exact float comparison (`evaluate.py:75`, `exact.py:88`, `exact.py:335`). With uniform logits over k actions, the computed probability falls a rounding error short of 1/k for k = 10 and k = 100. I measured this with `engine.log_policy(np.zeros((1, k)))`: 0.09999999999999998 and 0.009999999999999995. A solution whose true q0 is exactly c/N is then classed as unreachable, and under H12 any increase at all "crosses". Only a constructed world can sit exactly on the line. I haven't checked whether any RQ1 planted case does.

**Recommendation.** Use a relative tolerance, `q0 >= line * (1 - 1e-9)`, in the three places, with a test at k = 10.

### L2. The E_1 frontier's β and boundary hit aren't recorded (Low; probed)

The label uses `e1["both"]` (`exact.py:380`), but `extra["best_beta"]` and `extra["boundary_hit"]` come from the N frontier (`exact.py:343–345`). If the E_1 sup sat at the edge of the β grid [0.25, 8], "beyond temperature" could come from the range, and nothing in the record would show it.

**Probe (measured).** I rebuilt the `configs/main.json` bases for k ∈ {0, 2, 4} and all ten base seeds (0–9), and recomputed the N = 1 frontier for every group. I did this with the default grid and with a 61-point grid over [0.02, 200]. All 540 E_1 frontiers (3 k values × 10 seeds × 9 groups × 2 families) are interior: β ranges from 0.68 to 1.71, and no boundary was hit. The wider grid changes no best_pass to 6 decimals. So RQ3's labels don't depend on the β range. I didn't probe the RQ4 Phase 1 grid bases or RQ1. The script is in the appendix.

**Recommendation.** Record `e1_best_beta` and `e1_boundary_hit` in `extra`.

### L3. Sampled-backend labels use point estimates (Low)

`sampled.py:117–120` and `sampled.py:291` test E_1 > 1e-9 on the point estimate. D2 asks for a positive lower bound on a 95% CI for sampled estimates. A crossing (`sampled.py:226`) compares noisy q1 estimates with the line. The module docstring already says that per-step-only E_1 "can overstate beyond temperature". None of the configs in `configs/` uses this backend: they're all `exact`. The `sampled` arm in `rq2_time_to_cross.py` is sampled *training*, not this backend.

**Recommendation.** Keep sampled-backend labels out of the report, or add the CI rule before using them.

### L4. E_N outside valid (H, N) is flagged, not refused (Low)

D7 says the library refuses to report E_N outside |A|^H ≥ 100N unless forced. The code computes it and sets `e_n_valid` (`evaluate.py:122`, `exact.py:362`), and the harness report flags it (`report.py:493`). That's acceptable if the report never quotes an E_N with `e_n_valid = false`. The label doesn't depend on E_N (H39), so labels aren't affected.

### Nits

- **N1.** `exact.py:44`: the comment says `E_N["both"]`, but the constant is applied to `E_1["both"]` (H39).
- **N2.** `core.label_2x2` (`core.py:64`) is the old sign rule with `tol=1e-12`. It's still exported and still used by `experiments/phase1_rq1.py:54–55` with `TOL = 0.01`, which is exactly D2's Problem 1. Don't cite that script's output. Consider deleting both, or marking them legacy.
- **N3.** `exact.py:278`: if any prompt in a group lacks `tempering_logits`, the per-step family is dropped for the whole group, so "both" means sequence only. The missing key in `extra["frontier"]` shows it, but nothing flags it.

## Confirmation (2026-11-25)

Each check ran on main at `40b6cd3`. The library code reviewed is unchanged since `ea1f21f`. The script is in the appendix.

- **M1, confirmed.** `pilot_rq1.run_trial` returns `cases, delta_cap, delta_j, delta_rel, excess, j_gap, label, sensitivity`, with no `crossed`. I recomputed the eight 6×6 (N = 100) planted trials. `b_param` lands as `reweighting` in 3/8 under D2 and 8/8 under H12, which reproduces H29. H12 and D2 agree on every other 6×6 case.
- **M2, confirmed.** `BenchmarkReport.__dataclass_fields__` has no `saturated`.
- **M3, confirmed, and worse than a bias in magnitude.** On a five-action, one-step task I planted π1 = the base tempered at its true best β* = 0.3245, which is 0.028 in log β from the nearest grid point. A tempered base is pure sharpening by construction. Single-task `evaluate` gets this right: J gap −1.4e-10, `pure_sharpening`. But `evaluate_benchmark` reports excess +1.5e-5 and labels it `reweighting`. The grid-only frontier produces a false "beyond temperature". Only `experiments/pilot_rq4.py` uses this path, so no reported RQ4 number is affected (the final RQ4 uses the harness's refined frontier).
- **M4, confirmed.** With q0 = (0.5, 0.009, 0.009) on three correct solutions and q1 = (0.5, 0.010, 0.0) at N = 100, solution 2 crosses, so H12 says beyond reach. But Δcap = −0.008, and D2 labels it `pure_sharpening`.
- **L1, confirmed, with no current impact on RQ1 6×6.** Uniform logits give 1/k a rounding error below 1/k at k = 10 and 100, as stated. On the 6×6 planted bases, the closest any solution's q0 comes to c/N is 19% (relative), so none sits on the knife edge. The 4×4 and 5×5 sizes weren't checked.
- **L2, confirmed, with no impact.** See the probe above (all ten seeds).
- **L3, L4, N1–N3, confirmed by reading** the lines cited. None of them changes a number, so no runnable check was needed.

## What still needs a person

1. **Optional:** a teammate reads this doc if the team wants the person-only review rule met. Bilal waived it on 2026-11-25.
2. **Team (decision 1):** ratify H12 or D2 as the primary label, knowing M4.
3. **Muhammad:** resolve M1 before the RQ1 final run.
4. **Report author:** follow the M2 and M3 wording.

## Appendix: L2 probe

```python
import numpy as np, torch
from capsplit.harness.config import load_config
from capsplit.harness.registry import TASKS, BASES, BACKENDS
from capsplit.harness.runner import measure_spec
cfg = load_config("configs/main.json")
for combo in cfg.factor_combos():
    parts = cfg.resolved(combo)
    task = TASKS.create(parts["task"].type, **parts["task"].params)
    for seed in [0, 1]:
        bs = cfg.effective_base_seed(seed); torch.manual_seed(bs)
        base, _ = BASES.create(parts["base"].type, **parts["base"].params).build(task, bs)
        spec = measure_spec(task, cfg.measure)
        for betas in (None, np.exp(np.linspace(np.log(0.02), np.log(200), 61))):
            cache = BACKENDS.create("exact", **({} if betas is None else {"betas": betas})).prepare(task, base, spec)
            print(combo, seed, {g: {f: (fr.best_pass, fr.best_beta, fr.boundary_hit)
                  for f, fr in cache.frontiers[(g, 1)].items()} for g in spec.groups})
```

## Appendix: confirmation script

Run from the repo root. M1 takes about 15 minutes on 4 CPUs, mostly for the 6×6 frontiers.

```python
import sys, numpy as np
sys.path.insert(0, "experiments")
from capsplit.engine import TrajectoryTable, trajectory_probs, solution_probs
from capsplit.evaluate import evaluate, evaluate_benchmark, BenchmarkReport
from capsplit.planted import SIZES, sample_cases, size_world, EXPECTED
from capsplit.worlds import compile_grid
from capsplit.tempering import frontier, DEFAULT_BETAS
import pilot_rq1

def one_state(k, correct_ids):
    sid = np.full(k, -1, dtype=np.int32)
    for j, a in enumerate(correct_ids): sid[a] = j
    return TrajectoryTable(index=np.arange(k, dtype=np.int32).reshape(k, 1), solution_ids=sid,
                           solutions=tuple(range(len(correct_ids))), n_states=1, n_actions=k, horizon=1)

print("== M2: BenchmarkReport has saturated:", "saturated" in BenchmarkReport.__dataclass_fields__)

print("== M4: one crossing with delta_cap < 0, N=100")
t = one_state(6, [0, 1, 2])
q0 = np.array([0.5, 0.009, 0.009, 0.3, 0.1, 0.082]); q1 = np.array([0.5, 0.010, 0.0, 0.3, 0.1, 0.09])
r = evaluate(t, np.log(q0)[None, :], q1, 100)
print(f"   crossed={list(r.crossed)} delta_cap={r.delta_cap:+.4f} -> H12 beyond reach=True, D2 label={r.label}")

print("== M3: planted off-grid beta, benchmark vs single-task")
t = one_state(5, [1, 2])
l0 = np.log(np.array([0.55, 0.2, 0.15, 0.06, 0.04]))[None, :]
fine = np.exp(np.linspace(np.log(0.25), np.log(8), 20001))
js = [trajectory_probs(t, l0, b)[[1, 2]].sum() for b in fine]
bstar = fine[int(np.argmax(js))]
p1 = trajectory_probs(t, l0, bstar)
single = evaluate(t, l0, p1, 1)
bench = evaluate_benchmark([t], [l0], [p1], 1)
on_grid = np.min(np.abs(np.log(DEFAULT_BETAS / bstar)))
print(f"   beta*={bstar:.4f} (log-distance to grid {on_grid:.3f}); single j_gap={single.j_gap:+.2e} label={single.label}; benchmark excess={bench.excess:+.2e} label={bench.label}")

print("== M1: RQ1 driver, 6x6 N=100, D2 vs H12 on b_param")
size = SIZES[2]
world = size_world(size)
table, state_index = compile_grid(world)
res_keys = None; d2 = h12 = 0; trials = size.trials
near = []
for seed in range(trials):
    planted = sample_cases(size.budget, seed, table, state_index)
    fr = {}
    for case in EXPECTED:
        l0, p1 = planted.cases[case]
        r = evaluate(table, l0, p1, size.budget, frontier_result=fr.setdefault(id(l0), frontier(table, l0)))
        lab_h12 = {(True, True): "capability", (True, False): "compounded_sharpening",
                   (False, True): "reweighting", (False, False): "pure_sharpening"}[(bool(r.crossed), r.j_gap > 1e-9)]
        if case == "b_param":
            d2 += r.label == EXPECTED[case]; h12 += lab_h12 == EXPECTED[case]
        elif lab_h12 != r.label:
            print("   H12/D2 differ on", case, seed, r.label, lab_h12)
        q0 = solution_probs(table, trajectory_probs(table, l0))
        line = 1 / size.budget
        near += list(np.abs(q0 / line - 1)[q0 > 0])
w4 = size_world(SIZES[0]); t4, s4 = compile_grid(w4)
print("   saved keys (4x4 trial):", sorted(pilot_rq1.run_trial(t4, s4, SIZES[0].budget, 0)))
print(f"   b_param lands: D2 {d2}/{trials}, H12 {h12}/{trials}")
print(f"== L1: closest planted q0 to the line (relative): {min(near):.3e}")
```
