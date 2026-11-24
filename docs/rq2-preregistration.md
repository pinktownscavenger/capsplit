# Pre-registration: RQ2 time to cross

**Status:** frozen before any full run of `experiments/rq2_time_to_cross.py` (git tag `rq2-frozen`). Changes after the freeze go in `docs/DECISIONS.md`, and affected runs are redone. Design: `docs/rq2-time-to-cross-design.md`.

Only `--quick` smoke runs happened before the freeze. They used q₀ ≥ 10⁻⁴, world W1 only, ratios 1 and 0.1, and 2 draws, and they wrote to a scratch directory, not `results/`.

## 1. Question and prediction

How many exact policy-gradient updates does it take for a rare correct route to cross the reach line, and how does that depend on its base probability q₀? The plan (blueprint §2.6) predicts that time to cross grows roughly as 1/q₀, and is contradicted if it is flat in q₀.

## 2. Settings

All values are the constants at the top of `experiments/rq2_time_to_cross.py` at the tagged commit.

| Experiment | Settings |
|---|---|
| Arm A main | Isolated world (5×5, H = 8, one route). q₀ ∈ 9 log-spaced points over [10⁻⁶, 10⁻²]. lr ∈ {0.5, 1, 2}, primary lr = 1. Line ℓ = 0.1. Cap 400,000 updates. Exact PG. |
| R1 (robustness) | As arm A at lr = 1, plus N(0, σ²) noise with σ = 1.0 on every logit except the route actions at route cells, then the route bonus is re-bisected to hit q₀. Draws 0–19. |
| R2 (robustness) | As arm A at lr = 1, σ = 0, horizon H = 10 (the world still has exactly one route). |
| Arm B | Worlds W1 (3×3 ring, H = 4), W2 (4×4, H = 6, wall (1, 1), 8 routes, bonus routes `R_COMMON` and `R_RARE`, which share two moves), W3 (5×5, H = 8, two disjoint routes). Common-route bonus fixed at W1 0.75, W2 1.0, W3 1.75 (q₀(common) ≈ 0.02). Rare-route bonus bisected so q₀(rare)/q₀(common) ∈ {1, 0.9, 0.5, 0.1, 0.01}. Noise σ = 1.0 off both routes, draws 0–19. lr ∈ {0.5, 1, 2}. Cap 20,000 updates. Exact PG. |
| Sampled (stretch) | REINFORCE, batch 32, lr 0.5, baseline decay 0.9, seeds 0–19, cap 20,000 updates. Arm A at q₀ ∈ {10⁻⁴, 10⁻³, 10⁻²}, and W1 at ratios {1, 0.5, 0.1}, both with σ = 0. |

**Crossing.** The time to cross is the first update at which q is at or above the line, counting the base as update 0. Runs that never cross are censored, and reported, never dropped.
- Arm A, R1, R2 and sampled arm A: q ≥ 0.1.
- Arm B and sampled W1, **primary:** q(rare) ≥ 10 × q₀(rare). **Secondary:** q(rare) ≥ 0.1, reported where q₀(rare) < 0.1.

## 3. Predictions and decision rule

Anything between supported and contradicted is inconclusive.

| ID | Claim | Supported if | Contradicted if |
|---|---|---|---|
| P1 | Time to cross grows roughly as 1/q₀ (arm A main, lr = 1) | Fitted slope over all 9 points in [−1.0, −0.7], and the local slope between the two smallest q₀ is ≤ −0.9 | Fitted slope > −0.3 |
| P2 | It holds off the symmetric world (R1, R2) | ≥ 18/20 R1 draws and R2 have fitted slope ≤ −0.7 | Median R1 slope > −0.3, or R2 slope > −0.3 |
| P3 | Time scales as 1/lr (arm A main) | lr × updates varies by < 5% across lr ∈ {0.5, 1, 2} at every q₀ ("holds") | Reported only, not a contradiction test |
| P4 | A rarer competing route does not cross under exact PG (arm B, primary rule) | Fraction crossing ≤ 0.1 for every world and lr at every ratio ≤ 0.5 | Fraction crossing ≥ 0.5 for any world and lr at ratio ≤ 0.1 |

- A censored arm A point at lr = 1 makes P1 "inconclusive (censored)".
- A censored R1 draw counts as not ≤ −0.7, and is left out of the median.

**P1 is already settled by the derivation** in `docs/proofs.md` §7. The full-table run equals the scalar recursion, which predicts a fitted slope of −0.844 and a smallest-pair local slope of −0.928. P1 is reported as a confirmed derivation, not as an empirical test. The tests that can fail are P2 and P4.

## 4. Validity checks before any verdict

If any of these fails, the report marks the whole result "invalid".
- Arm A main: the full-table run and the scalar recursion give the same update count, and their q(t) agree within 1e-9.
- Every base hits its target within 1% relative: q₀ for arm A, R1 and R2, and the ratio for arm B.
- Every arm B run has 10 × q₀(rare) ≤ 0.5, so the primary rule is achievable.

Gradient correctness is covered by the existing finite-difference test on W1 (`tests/test_train_tabular.py`) and by the exact match to the scalar recursion.

## 5. Reporting

`python experiments/rq2_time_to_cross.py report` writes tables, figures and `verdicts.{json,md}` to `results/rq2_report/`. Negative, null and inconclusive results are reported as prominently as positive ones. The W2 (overlapping routes) results are compared with W1 and W3 (disjoint).
