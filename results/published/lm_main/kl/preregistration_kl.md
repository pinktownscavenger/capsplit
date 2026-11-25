# Pre-registration: function-space (KL) layer on the completed LM run

Written and committed **before any KL number for a trained adapter was computed**, together with `configs/lm_kl.json` and the stage code (`src/capsplit/harness/lm/kl.py`). Decisions: `docs/DECISIONS.md` L23–L31.

## What had been seen when this was written

- **The full `lm_main` results:** report, verdicts L1–L3, null-check diagnostic, discovery comparison, training logs.
- **The Step 0 audit** (`audit.md`).
- **The re-run speedtest** and two timing pilots of the `kl` stage on the untrained `none` arm. Its adapter is exactly zero, so its KL is 0 by construction and carries no information about the trained arms.
- **Unit tests** on a tiny random model.

**This is not a blind design.** The output-metric results were known, and the predictions below are written with them in view. The layer is exploratory and gives no verdict on L1–L3.

## Measurement

- **Policies:**
  - π0 is the base model (LoRA adapter disabled).
  - π1 is the trained adapter at the final checkpoint (200 updates).
  - Both run at the reference temperature T = 1, with top_p = 1.
- **Completion:** the generated token sequence including its stop token, truncated at 12 tokens.
- **Scope:** arms `rl`, `sft`, `sft_then_rl` × seeds 0, 1, 2 × all 180 problems.
- **KL estimate:**
  - m = 64 completions y ~ π1 per problem.
  - KL_hat is the mean of ln π1(y) − ln π0(y). No term is clipped.
  - SE = sd / √m.
  - Both log-probs come from `policy.log_prob`, with the adapter on and then off.
- **p̂0:** base sets A and B pooled (400 + 64 samples per problem).
- **p̂1:** the arm's saved eval samples (64 per problem). Neither p̂0 nor p̂1 is resampled.
- **Bound:** d(p̂1 ‖ p̂0) with Jeffreys smoothing, (k + ½)/(n + 1). The headline K1 numbers are also shown with Laplace smoothing, (k + 1)/(n + 2).
- **Intermediate checkpoints (25, 50, 100):** m = 16, and gain uses p̂1 from these KL samples.
- **Exact-zero checks:** every step-0 adapter and the `none` arm at step 200, with m = 4.

## K1: bound check

**Statistic:** the fraction of (arm, seed, problem) cells with KL_hat + 2·SE < d(p̂1 ‖ p̂0). It is reported pooled and per arm, with both smoothings.

**Prediction:**
- Violations are rare, **at most 5% pooled**. Above 5% means a bug or a bad estimator, not a discovery; investigate before interpreting K2 and K3.
- The exact-zero checks give KL_hat = 0 exactly (maximum |log-ratio| = 0).

## K2: does KL separate the classes the output metric found?

**Groups** (final checkpoint; seeds pooled; one observation = one problem in one seed):

| Group | Definition |
|---|---|
| G1 | train, reachable |
| G2 | train, unreachable, sampled correct at least once in training (`first_success_update` not null) |
| G3 | train, unreachable, never sampled correct in training |
| G4 | held-out, reachable |
| G5 | held-out, unreachable |

**Rules for the groups and statistic:**
- Classes come from set A, N = 16, as in the main run.
- G2 and G3 apply only to arms that sample during training (`rl`, `sft_then_rl`). For `sft`, all train-unreachable problems are reported in the G3 row.
- **Statistic:** mean KL_hat per group, with a 95% bootstrap interval that resamples problems, keeping all seeds of a problem together.

**The user's prediction (from the brief)**, for `rl`:
- **(a)** G2 has the largest mean KL of the five groups.
- **(b)** G3 is near the level of the reachable problems, not zero, because RL still moves the model there, only not toward success. "Near" is made concrete here as the ratio mean(G3) / mean(G1) lying in [0.5, 2]. "Not zero" means the G3 interval's lower bound is > 0.

**My prediction where it differs:**
- I agree with (a). Since p̂0 is small and p̂1 is near 1 for most G2 problems, the bound alone forces several nats there.
- I agree G3 is clearly above zero.
- **I predict G3 is lower than G1:** mean(G3) / mean(G1) < 1, possibly below 0.5. The policy-gradient update for an all-wrong group is exactly zero, so G3 moves only through parameters shared with other problems.
- **For `sft`**, the train-unreachable row (trained on directly) has the largest KL, and G5 exceeds G4.

## K3: the held-out gap (descriptive; no new verdict on L2)

**Reported:** per arm, KL_hat and the efficiency Δp / KL_hat (mean Δp over mean KL_hat, Δp = p̂1 − p̂0) for each group, with bootstrap intervals.

**Prediction:**
- For every arm, G5's KL interval lies above 0, even though the L2 success-metric interval touches 0. The distribution on held-out unreachable problems moves measurably.
- G5's efficiency is lower than G4's.

This would show movement, not capability. **It does not turn L2 into a verdict.**

## Over updates (descriptive)

**Reported:** mean KL_hat and total gain at checkpoints 0, 25, 50, 100, 200, and the fraction of the step-200 mean KL reached at step 25.

**Prediction:**
- Mean KL increases with updates for `rl` and `sft`.
- For `rl`, the step-25 fraction is below 0.5.
- For `sft`, the step-25 fraction is larger than for `rl`, because SFT fits its 47 target answers quickly.

## Weight descriptors (descriptive only; no prediction, no verdict)

- **Quantities:** ‖ΔW‖_F and the participation-ratio effective rank per layer and module, with ΔW = (α/r)·B·A.
- **Cosines:** between flattened ΔW across seeds and arms.
- **Caveat:** these are coordinates. Many weight settings give the same behaviour, and the numbers do not define capability.

## What this cannot show

- **Scope:** one 0.5B model with LoRA, generated arithmetic only, and 3 seeds.
- **Small group:** only 26 held-out unreachable problems.
- **Estimation:** a sampled estimator with m = 64, and 12-token completions.
- **Meaning:** KL measures how far the output distribution moved, not whether the model gained capability. A large KL is not evidence of new capability, and a small KL is not evidence of none.
