# RQ2 time-to-cross: design

> **Status: approved by Bilal on 2026-11-19, with the recommended defaults.** Nothing here is built yet. The items marked **(team decision)** were settled as follows:
> - Arm B crossing rule: primary q(rare) ≥ 10 × q₀(rare); q(rare) ≥ 0.1 reported as secondary (§6.3).
> - Derivation: a new section in `docs/proofs.md`, needing a second-person check (§5).
> - Sampled REINFORCE stretch: included, and cut first if time runs short (§6.4).
> - Prediction bands: as written in §7.
>
> Items marked **(builder's call)** may be settled during implementation and logged in `docs/DECISIONS.md`.

## 1. Intent

RQ2 asks what standard on-policy training changes (blueprint §2.6). Its time-to-cross part asks one narrow question: **how long does policy-gradient training take to make a rare correct solution reachable, and how does that time depend on the solution's starting probability q₀?**

- **Prediction:** time to cross grows roughly as 1/q₀.
- **Contradicted if:** time to cross is flat in q₀.

This matters for the project's main argument. A solution crossing the reach line c/N is exactly the "beyond reach" event that the split counts as capability. So the time-to-cross curve says how much training it takes before on-policy RL can show capability gain at all, and the competing-route arm says whether it does so when a more likely correct solution is present. It also connects to the RQ3 Phase 2 finding that "unreachable at N" depends on the budget: crossing is driven by how much probability the training signal can find.

This is one piece of RQ2. The RQ2 *training dynamics* part (3 algorithms, Δrel and Δcap every update, 20+ seeds) is out of scope here.

## 2. What exists

The pilot (`docs/pilot-report.md` §4) built a first version:

| Piece | Where | State |
|---|---|---|
| Exact-gradient softmax PG, REINFORCE | `src/capsplit/train/tabular.py` | Done, tested (`tests/test_train_tabular.py`) |
| Isolated-route world (5×5, H = 8, one route) and its base, bisected to a target q₀ | `experiments/pilot_rq2.py` | Done |
| Competing world (3×3 ring, two routes) and its base from route bonuses | `experiments/pilot_rq2.py` | Done |
| Arm A at q₀ ∈ {10⁻⁵ … 10⁻²}, lr = 1, cap 30,000 | `experiments/pilot_rq2.py` | Slope −0.817 |
| Arm B, 7 bonus pairs, lr = 2, 6,000 steps, crossing = 10 × q₀(B) | `experiments/pilot_rq2.py` | Only the symmetric start crossed |

**Gaps against the blueprint (§6.3, D6, Target level):**

1. Arm A stops at 10⁻⁵; the blueprint asks for 10⁻⁶ (about 150,000 updates, above the pilot's cap).
2. No written time-to-cross derivation, although the schedule asked for one. The acceptance band [−1.0, −0.7] has no stated basis.
3. Arm A uses the line c/N = 0.1, while arm B uses 10 × q₀. The two arms define "crossing" differently.
4. Arm B covers one world and one learning rate; D6 says it "needs a proper sweep across worlds and learning rates".
5. No learning-rate check, and nothing frozen before running.

## 3. Goals, non-goals, anti-goals

**Goals**
- G1. Arm A over q₀ from 10⁻⁶ to 10⁻², with a log-log curve, fitted slope and local slopes.
- G2. A written derivation that predicts arm A's curve, and a check of the measured curve against it.
- G3. Robustness of arm A beyond the one symmetric world: 20 random draws of the off-route logits, a longer horizon and three learning rates.
- G4. Arm B swept across 3 worlds, 5 starting ratios and 3 learning rates, with fraction crossing, peak, final mass and censoring.
- G5. A pre-registration frozen before the full runs, and a short RQ2 time-to-cross report.

**Non-goals**
- RQ2 training dynamics (Δrel and Δcap curves for 3 algorithms) and Q-learning.
- Moving RQ2 into the harness (`src/capsplit/harness/`). That was decided against: the harness fixes one update budget per experiment, measures only at checkpoints and stores tabular logits in float32. All three conflict with step-exact crossing at q₀ = 10⁻⁶.
- Any change to the harness, RQ3 results, or the pilot record.

**Anti-goals** (would count as failure even if the code works)
- Tuning worlds, ratios, caps or learning rates after seeing crossing results. Everything in §6 is frozen before the full run (D8's rule).
- Dropping runs that never cross. They are censored and reported.
- Presenting arm A's agreement with the scalar recursion as evidence for the prediction. It is an implementation check: the recursion is the same mathematics. The evidence is the shape of the curve, the derivation's asymptotics, and the robustness draws, where the recursion no longer holds exactly.

## 4. Approaches considered

| Approach | Trade-off | Decision |
|---|---|---|
| **A. Extend `experiments/` and `train/tabular.py`** (numpy, float64) | Small, fast, reuses the pilot's tested code, and matches the blueprint's traceability table (`train/tabular`). No harness report machinery, so the RQ2 report is written by its own script. | **Chosen by the team** |
| B. Build it in the harness | Config-driven runs and manifests, but needs a shared-interface change (early stop or per-step trace), an exact-PG trainer, a float64 tabular policy and new report tables. That touches RQ3's frozen code. | Rejected |
| C. Edit `pilot_rq2.py` in place | Least new code, but `tests/test_train_tabular.py` imports it and it is the pilot's record of what was run. | Rejected: import from it, don't change it |

## 5. Theory: the scalar reduction

In the isolated-route world, the route has length H, equal to the horizon, so any off-route action fails. The base gives the route action at every route cell a logit θ, and the other 3 actions 0.

- The per-step route probability is p = e^θ / (e^θ + 3), and q = J = p^H.
- Each route cell is visited once by the only correct trajectory, so at each route cell the exact gradient is q(1 − p) on the route action and −q·(1 − p)/3 on each of the 3 other actions. It is zero at off-route cells and the same at every route cell, so all H cells stay identical.
- Softmax depends only on logit differences, so the state is the gap g between the route logit and the (equal) off-route logits, with p = e^g / (e^g + 3). The gap grows by the route action's gain plus the off-route actions' loss: **g_{t+1} = g_t + η · (4/3) · p_t^H (1 − p_t)**. In general the factor is |A| / (|A| − 1).
- In continuous time, dp/dt = (4/3) η p^{H+1}(1 − p)², so the time from q₀ to the line ℓ is

  **T(q₀) = (3 / 4η) ∫ from q₀^{1/H} to ℓ^{1/H} of dp / (p^{H+1}(1 − p)²)**.

- For small q₀ the lower limit dominates and T ≈ 3 / (4 η H q₀) · (1 + O(q₀^{1/H})). So the local log-log slope tends to −1 as q₀ → 0, and is shallower at larger q₀, where the fixed cost of climbing from about q₀ to ℓ matters. This explains the pilot's −0.817 and D6's steepening toward −1.

**Checked numerically on 2026-11-19** (scratch script, not committed):
- The recursion reproduces the pilot's full-table counts exactly: 17,842, 2,358, 362 and 63 updates at q₀ = 10⁻⁵ … 10⁻².
- Over 200 updates from q₀ = 10⁻⁴, the recursion's q(t) and the full table's agree to 4 × 10⁻¹⁹.
- The integral gives exactly the blueprint's D6 numbers (148,403, 17,837, 2,354, 359, 61), so D6's "scalar logit gap" model is this one.
- Predicted at η = 1: 148,409 updates at q₀ = 10⁻⁶; fitted slope over the 9 points of §6.1 is −0.844; local slope between the two smallest is −0.928.

**Deliverable:** this derivation written up in `docs/proofs.md` as a new section (or `docs/notes/time-to-cross.md` if the team prefers proofs.md to stay frozen; **team decision**), with the fitted slope over [10⁻⁶, 10⁻²] at H = 8, ℓ = 0.1 computed from the integral. It needs a second-person check like the other proofs.

**What the reduction does not cover:** it breaks when the route cells are not symmetric (random off-route logits change the denominator per cell), and when H is longer than the route, because then detours that loop back are erased onto the same route (`loop_erased`) and add correct trajectories. Those are the robustness conditions in §6.2.

## 6. Experiments

All runs use exact-gradient softmax PG from `train/tabular.py`, in float64. Each run records q(t) for the solutions of interest at **every** update, so crossing at any line can be read off afterwards without rerunning.

### 6.1 Arm A, main curve

| Setting | Value |
|---|---|
| World | Pilot's isolated world: 5×5, H = 8, walls at rows 1–3 × columns 1–3 plus (0, 1) |
| q₀ | 9 points log-spaced over 10⁻⁶ to 10⁻² (10⁻⁶, 10^−5.5, …, 10⁻²) |
| Base | Pilot's `isolated_base`: one per-step bonus on the route, bisected to the target q₀ |
| Line | ℓ = 0.1 (c = 1, N = 10) |
| Learning rate | η ∈ {0.5, 1, 2}; η = 1 is primary |
| Cap | 400,000 updates (predicted worst case: about 297,000 for η = 0.5 at 10⁻⁶) |
| Outputs | Updates to cross per (q₀, η); scalar-recursion and integral predictions beside them; fitted slope over all 9 points; local slopes between neighbours |

The full-table run, the scalar recursion and the integral should agree: the first two exactly (to rounding), the integral to within a few percent (D6 found 3%).

### 6.2 Arm A, robustness

| Condition | What changes | Draws |
|---|---|---|
| R1. Off-route noise | Add N(0, σ²) to every non-route logit, σ = 1 (**builder's call** on σ within [0.5, 1.5], fixed before running), then re-bisect the route bonus to hit q₀ | 20 seeds per q₀ |
| R2. Longer horizon | Same walls, H = 9 or 10, so one or two wasted moves still reach the goal and loop-erase onto the route (checked: the world still has exactly 1 correct route at H = 9 and 10) | 1 per q₀ (deterministic) |

- R2 is limited by cost: the table has 262,141 rows at H = 9 and 1,048,510 at H = 10, against 65,536 at H = 8. **Builder's call:** measure one gradient step's time first; use H = 10 if a full q₀ sweep fits in under 2 CPU-hours, otherwise H = 9, and log the choice.
- η = 1 only for R1 and R2.
- Outputs: per-draw slope, the spread of updates-to-cross per q₀, and how far each condition departs from the scalar prediction.

### 6.3 Arm B, competing routes

| Setting | Value |
|---|---|
| Worlds | W1: pilot's 3×3 ring, H = 4, 2 routes. W2: `phase1_rq1.WORLD`, 4×4, H = 6, wall (1, 1). It has 8 correct routes; the base puts bonuses on two of them, `phase1_rq1.R_COMMON` and `R_RARE`, and the other 6 are background routes with no bonus. These two share their first two moves, so W2 is the overlapping-route case; W1 and W3 are disjoint. The report should compare them, since shared steps let the common route's gradient also raise the rare route's early steps. W3: the isolated world *without* the (0, 1) wall, so 5×5, H = 8, with exactly 2 correct routes (left-bottom and top-right, checked). |
| Starting ratios | q₀(rare) / q₀(common) ∈ {1, 0.9, 0.5, 0.1, 0.01} between the two bonus routes, set by bisecting the rare route's bonus with the common route's bonus fixed (**builder's call** on the fixed bonus, chosen so q₀(common) is between 0.01 and 0.04). The upper limit keeps the primary rule achievable: q₀(rare) ≤ 0.04, so 10 × q₀(rare) ≤ 0.4, which a route can reach even when it only ties. A cap of 0.3, as first drafted, would make the rule impossible at ratio 1 and 0.9 and bias P4. |
| Learning rates | η ∈ {0.5, 1, 2} |
| Draws | 20 off-route noise seeds per (world, ratio, η), as R1, so "fraction crossing" is a fraction over draws |
| Cap | 20,000 updates |
| Outputs | Per cell: fraction crossing, median updates to cross among those that cross, peak q(rare), final q(rare), and the censored count |

**Crossing rule (team decision).** Recommendation: the **primary** rule is the pilot's, q(rare) ≥ 10 × q₀(rare), because the c/N line sits below or near the start for the larger ratios. Arm A's line, q(rare) ≥ 0.1, is reported as secondary for every cell where q₀(rare) < 0.1. Both come from the same stored q(t). Whichever is chosen goes into `docs/DECISIONS.md` and the pre-registration.

### 6.4 Stretch, cut first: sampled REINFORCE

REINFORCE (`train/tabular.reinforce`) on arm A at q₀ ∈ {10⁻⁴, 10⁻³, 10⁻²} and on arm B's W1, 20 seeds, with time to cross measured both in updates and in sampled episodes. Under sampling a rare route may never be drawn, so censoring is real here, and arm B can flip between routes, which exact PG cannot. This ties directly to RQ3's budget finding. **Team decision:** include or drop.

## 7. Pre-registered predictions and decision rule

Frozen in `docs/rq2-preregistration.md` and tagged `rq2-frozen` before any full run. A `--quick` smoke run may happen before freezing; it uses only q₀ ≥ 10⁻⁴, and nobody looks at arm B crossing outcomes before the freeze.

| ID | Claim | Supported if | Contradicted if |
|---|---|---|---|
| P1 | Time to cross grows roughly as 1/q₀ (arm A main, η = 1) | Fitted slope over all 9 points in [−1.0, −0.7], and the local slope between the two smallest q₀ is ≤ −0.9 | Fitted slope > −0.3 ("flat") |
| P2 | It holds off the symmetric world (R1, R2) | ≥ 18/20 R1 draws and R2 have fitted slope ≤ −0.7 | Any condition's median slope > −0.3 |
| P3 | Time scales as 1/η (arm A main) | η · updates varies by < 5% across η ∈ {0.5, 1, 2} at every q₀ | — (reported, not a contradiction test) |
| P4 | A rarer competing route does not cross under exact PG (arm B, primary rule) | For every ratio ≤ 0.5, fraction crossing ≤ 0.1 in every world and η | Fraction crossing ≥ 0.5 in any world at ratio ≤ 0.1 |

Anything between supported and contradicted is inconclusive. The bands for P1 and P2 are **team decisions**.

**P1 is already settled by §5.** The derivation predicts a fitted slope of −0.844 and a smallest-pair local slope of −0.928, and the full-table run equals the recursion, so P1 will pass. It stays in the table so the report states it, but the report must present it as a confirmed derivation, not as an empirical test. The tests that can fail are P2 (where the reduction no longer holds) and P4.

**Validity checks before any verdict:** the full-table run matches the scalar recursion in arm A main (same update count, q(t) within 1e-9); finite-difference gradient checks pass on W2 and W3; each base hits its target q₀ within 1% relative; and every arm B run has 10 × q₀(rare) ≤ 0.5, so the primary rule is achievable.

## 8. Where the changes go

| File | Change |
|---|---|
| `src/capsplit/crossing.py` (new) | `crossing_step(series, line)` returning the first index at or above the line, or None if censored; `scalar_isolated_updates(q0, H, eta, line, cap)` (the recursion); `integral_time(q0, H, eta, line)`; `loglog_slopes(q0s, times)` returning the fitted slope and local slopes |
| `src/capsplit/train/tabular.py` | Optional backward-compatible `stop_when` argument on `train_exact_pg`, so arm A stops at the crossing instead of running to the cap. Existing behaviour unchanged when it isn't given. |
| `experiments/rq2_time_to_cross.py` (new) | Subcommands `arm-a`, `robust`, `arm-b`, `sampled` (stretch) and `report`, plus `--quick`. Imports worlds and bases from `pilot_rq2` and `phase1_rq1`; defines W3 and the noise and ratio bisection. All frozen constants sit at the top of the file, as in the pilot. Writes through `capsplit.runlog` to `results/rq2_*/<hash>/`. |
| `experiments/pilot_rq2.py`, `run_pilot.py` | Unchanged |
| `tests/test_crossing.py` (new) | Crossing and censoring edge cases; the scalar recursion (with the 4/3 factor) reproduces the pilot's arm A counts exactly (17,842 / 2,358 / 362 / 63); the integral reproduces D6's numbers (17,837 / 2,354 / 359 / 61) and is within 5% of the recursion; slope helper on synthetic power laws |
| `tests/test_rq2.py` (new) | Full-table PG equals the scalar recursion on the isolated world; W3 has exactly 2 correct routes; ratio bisection hits its target; `--quick` runs end to end. Long runs marked `slow`. |
| `docs/rq2-preregistration.md` (new) | §6 settings and §7 table, frozen |
| `docs/proofs.md` or `docs/notes/time-to-cross.md` | The §5 derivation |
| `docs/rq2-time-to-cross.md` (new) | Results report, generated numbers only, written after the run |
| `docs/DECISIONS.md` | Crossing rule, σ, H choice, cap and any bug fixes |
| `README.md` | Status line and layout rows once the report exists |

The `viz` extra (matplotlib) is already declared; figures go to `results/` and are copied into `docs/rq2/` for the report, as RQ3 did.

## 9. Constraints

- **Precision:** float64 throughout. At q₀ = 10⁻⁶ the first updates move logits by about 10⁻⁶, which float32 would round away.
- **Cost:** arm A main is about 1.5M updates in total on a small table, and arm B is about 900 runs × 20,000 updates. Both should be minutes, based on the pilot's timings. R2 at H = 10 is the only expensive piece, and §6.2 bounds it.
- **Schedule:** the report freezes on Dec 2. The order is §5 derivation and §8 code with tests, then the pre-registration freeze, then the full run, then the report. The stretch (§6.4) is cut first if time runs short.
- **Review:** the README's workflow is a branch (`exp/rq2-time-to-cross`), a PR, and one teammate's review before merging.

## 10. Left to the builder

- Names and signatures inside the new files, beyond those listed in §8.
- σ for the noise draws within [0.5, 1.5], the common route's fixed bonus in arm B, and H = 9 or 10 for R2, each fixed before the full run and logged.
- Figure layout: arm A's log-log curve with the integral and the R1 spread; arm B's fraction-crossing heatmap per world and η; and a few q(rare) trajectories.
