# Pre-registration: RQ4, pass@k misdiagnosis

**Status:** frozen before the pass@k rule was applied to any real unit (git tag `rq4-frozen`). Changes after the freeze go in `docs/DECISIONS.md`, and anything affected is redone. Design: `docs/rq4-passk-design.md`. Code: `experiments/rq4_passk.py`, whose `score` command refuses to run unless this tag is an ancestor of HEAD.

**What was looked at before the freeze:**
- Phase 1: labels only, in the label check (RQ4-1).
- Phase 2: the RQ3 rerun's harness report and its reproduction check against `docs/experiment_main/` (RQ4-2).
- Neither phase's rule outputs or per-prompt success rates were looked at.

## 1. Question and prediction

How often does the practical pass@k rule (blueprint D10) disagree with the exact 2×2 label?
- **Prediction (blueprint §2.6):** it disagrees in a sizable share of runs, most often in the compounded-sharpening and reweighting cells.
- **Contradicted if:** primary agreement is above 95%.

## 2. Data

| Phase | Source | Primary units |
|---|---|---|
| **2 (headline)** | **The RQ3 rerun** (Bilal's decision, 2026-11-20). `configs/main.json` run at commit `46490ae` on this machine with a clean tree, copied to `results/main`. It reproduces all four RQ3 verdicts but not the committed numbers (RQ4-2), and RQ4's report says so. | 7 trained arms × k ∈ {0, 2, 4} × seeds 0–9 = **210** |
| 1 | `configs/rq4_grid.json` (sha256 prefix `848ad399ff89ca83`), run at the tagged commit. Shared tabular policy, 8 prompts, 3 trained arms, seeds 0–19, 400 updates. | 3 arms × seeds 0–19 = **60** |

- **A unit** is one (run, checkpoint, prompt group).
- **Excluded:** update 0, and arms whose trainer is `none`.
- **Primary units:** group `heldout`, final checkpoint (update 1,000 in Phase 2, 400 in Phase 1), at N = 64, c = 1.
- **Inputs:** per-prompt J₀ and J₁ come from each record's `extra.j0` and `extra.j1`.
- **Ground truth:** the record's `label`, the E₁ rule (RQ3's H39). `extra.label_en` is reported beside it.

## 3. The rule (frozen)

- **Rule:** `capsplit.diagnostics.passk_rule` (D10). It says **sharpening** if the trained mean pass@k is below the base's at any k ≥ k_large; **capability** if it is never below and above somewhere; **undetermined** otherwise.
- **k_large = 10.** Tie tolerance `1e-9`.
- **k grids:** `pilot_rq4.k_grid` gives (1, 2, 5, 10, 20, 50, 77) for Phase 2 and (1, 2, 5, 10, 20, 50, 100, 200, 500, 655) for Phase 1.
- **Sensitivity only:** k_large ∈ {5, 20, 50} (only where k_large ≤ k_max).

## 4. Scoring and decision rule

**Primary agreement (D10):**
- It counts as agreement when the rule says capability and the label is capability, or the rule says sharpening and the label is not capability.
- "Undetermined" counts as disagreement, and its rate is reported separately.
- Secondary agreement (sharpening must match a sharpening cell) is reported per cell, never as a headline.

**Interval:** `capsplit.harness.report.bootstrap_ci` (10,000 resamples, RNG seed 0) over per-seed primary agreement. Each seed pools its primary units over arms and k.

| ID | Claim | Supported if | Contradicted if |
|---|---|---|---|
| Q1 | Phase 2: the rule misdiagnoses a sizable share | CI upper bound < 0.95 | CI lower bound > 0.95 |
| Q2 | Same, Phase 1 | As Q1 | As Q1 |
| Q3 | Disagreement concentrates in compounded sharpening and reweighting (pooled Phase 1 + 2 primary units, each scored with its own k grid) | Each of those two cells' disagreement (1 − primary) exceeds both other cells' | Either of those two cells has the lowest disagreement of the four |

- Anything else is inconclusive.
- Q3 is inconclusive if any of the four cells is missing or has fewer than 10 pooled primary units.

## 5. Validity checks

These are reported with the verdicts. A failure marks the result "invalid".
- Every unit's J₀ and J₁ lists are non-empty and of equal length.
- The primary unit counts equal the expected 210 (Phase 2) and 60 (Phase 1).
- Every one of the four cells is reported per phase, even if empty.

Phase 1's manifests must show the tagged commit and `dirty: false`.

## 6. Reporting

- Negative, null and inconclusive verdicts are reported as prominently as positive ones.
- Every overall rate comes with its per-cell rates, and the report says how each phase's run mix was chosen: Phase 2 is RQ3's natural runs; Phase 1 was tuned on labels only (RQ4-1).
- **SRR/NDR** are computed per prompt into `srr_ndr_UNVERIFIED.csv` and are **not reported** until a person has checked `docs/notes/invisible-leash-scores.md` against the paper and the team has settled its open points.
