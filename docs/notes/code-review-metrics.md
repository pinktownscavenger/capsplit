# Pre-review: label and metric code

Status: **pre-review for a person to confirm** (plan task 3, owner Sazan). Code review is person-only in this project, so nothing here counts as reviewed until Sazan signs off on each finding.

Scope: `src/capsplit/evaluate.py`, `src/capsplit/core.py`, `src/capsplit/engine.py`, `src/capsplit/tempering.py`, `src/capsplit/harness/backends/exact.py`, the label paths in `src/capsplit/harness/backends/sampled.py`, and the RQ1 driver `experiments/pilot_rq1.py`. Checked against `docs/blueprint.md` §2.2–2.3 and D2, D3, D5, D7, `docs/DECISIONS.md` H12, H13 and H39, and `docs/proofs.md` §1. Commit reviewed: `ea1f21f` (main, 2026-11-24).

## Summary

I found no bug that changes a number already reported in RQ2, RQ3 or RQ4. The most important finding is about the run that's still to come: **the RQ1 driver labels with D2 only** (M1), so the RQ1 final sweep would reproduce the 6×6 `b_param` misfire unless it also records the H12 label. Which label counts as primary is team decision 1.

| ID | Severity | Finding | Blocks |
|---|---|---|---|
| M1 | Medium | RQ1 driver records only the D2 label, not H12 | RQ1 final run |
| M2 | Medium | `BenchmarkReport` has no `saturated` flag | Report wording |
| M3 | Medium | `evaluate_benchmark` frontier is grid-only (D3) | Report wording |
| M4 | Medium (design) | H12 "beyond reach" fires on one crossing, whatever the mass, and even when Δcap < 0 | Decision 1 |
| L1 | Low | `q0 >= c/N` with no tolerance misclassifies q0 = c/N exactly | Nothing yet |
| L2 | Low | The E_1 frontier's β and boundary hit are not recorded | Nothing (probed, see below) |
| L3 | Low | Sampled backend labels from point estimates, not a CI bound | Only if it's ever used |
| L4 | Low | E_N outside valid (H, N) is flagged, not refused (D7) | Nothing |
| N1–N3 | Nit | Stale comment, legacy label function, silent family drop | Nothing |

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

**Probe (measured).** I rebuilt the `configs/main.json` bases for k ∈ {0, 2, 4} and base seeds 0 and 1, and recomputed the N = 1 frontier for every group. I did this with the default grid and with a 61-point grid over [0.02, 200]. Every E_1 frontier is interior: β ranges from 0.68 to 1.71, and no boundary was hit. The wider grid changes no best_pass to 6 decimals. So for those seeds, RQ3's labels don't depend on the β range. I didn't probe seeds 2–9, the RQ4 Phase 1 grid bases, or RQ1. The script is in the appendix.

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

## What a person still needs to do

1. **Sazan:** confirm or reject each finding above, and sign off on the "Verified correct" list. This doc is a pre-review, not that sign-off.
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
