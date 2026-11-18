# Pre-registration: which post-training signals produce capability gain?

**Status:** frozen at Gate 1, before any run of `configs/main.json`. This file and the frozen base design (`configs/base_v1.json`, verified in `docs/base_verification.md`) must not change after Gate 1. If a bug forces a change, it is logged in `docs/DECISIONS.md`, and everything affected is re-run.

## 1. Setting (RL framing)

| RL object | Here |
|---|---|
| Episode | One prompt (type, a, b); the policy writes L = 5 tokens from a vocabulary of V = 6 |
| State | The prompt plus the tokens written so far |
| Action | The next token |
| Reward | 1 at the end if the verifier accepts the sequence, else 0 |
| Return | Equal to the reward, so the expected return on a prompt is J = P(correct) |
| Policy | One prompt-conditioned GRU (`prompt_gru`) for all 72 prompts |
| π0 | Pre-trained by weighted maximum likelihood (`pretrain_gru`, frozen in `configs/base_v1.json`) |
| Update rules (arms) | See §2 |

- **Prompts:** easy and hard prompts, each split into train and held-out.
- **Hard prompts:** a hard solution follows the easy rule for its first k tokens and a hard rule after that. k ∈ {0, 2, 4} is the **shared-structure** factor.
- **Base properties:** every hard solution is unreachable under π0, and no tempering fixes it. Easy solutions are reachable but not saturated. Verified in `docs/base_verification.md`.

## 2. Design

`configs/main.json` runs arms × k ∈ {0, 2, 4} × seeds 0–9 through the harness. All arms start from the same base for a given (k, seed). Every arm gets the same budget of optimizer updates, and all train only on train prompts.

| Arm | Trainer | Signal |
|---|---|---|
| `none` | none | No post-training (control) |
| `rloo` | rloo | RL only: outcome reward, leave-one-out baseline (A) |
| `grpo` | grpo | RL only: group-normalised advantage (A, robustness) |
| `rl_entropy` | rl_entropy | RL + entropy bonus 0.01 (E) |
| `sft` | sft | Supervised fine-tuning on teacher demonstrations, hard train prompts only (B) |
| `hints_h1` | hints, hint_len 1 | RL; all-fail groups are re-rolled from a 1-token prefix of a correct solution (C) |
| `hints_h3` | hints, hint_len 3 | Same, with a 3-token prefix (C) |
| `sft_then_rl` | sft_then_rl, fraction 0.5 | Arm B for half the budget, then RLOO (D) |

**Shared hyperparameters:** Adam with lr 1e-3, RL group size G = 16, all train prompts in every update, SFT full batch. The budget and checkpoints are in `configs/main.json`.

**Pilot rule** (pilot result in `docs/DECISIONS.md` H40): the budget was fixed by a single pilot run using seed 100, which is not one of the experiment seeds. The pilot looked **only** at the `rloo` arm's J on easy-train prompts. The rule: the budget must let it reach J ≥ 0.9 at k = 0. No hard-prompt quantity was looked at.

**Measurement:**
- Exact enumeration with the `exact` backend at every checkpoint.
- N ∈ {16, 64} and c ∈ {0.5, 1, 2}. **Primary: N = 64, c = 1.**
- Groups by difficulty × split.

## 3. Hypotheses

Notation. All quantities come from the final checkpoint at primary (N, c), unless stated.
- **Δcap(G)** and **Δrel(G):** the capability and reliability contributions on group G. They are seed-level values.
- **CI:** the 95% percentile bootstrap interval of the seed mean over the 10 seeds (10,000 resamples, RNG seed 0), as produced by the report generator.
- **Paired difference:** the difference between two arms computed seed by seed (same base), then bootstrapped the same way.

**Decision rule** for a claim "X > τ":
- **supported** if CI lower bound > τ;
- **not supported** if CI upper bound ≤ τ;
- **inconclusive** otherwise.

For "X ≤ τ" it is the mirror image. The "near zero" tolerance is τ0 = 0.02 in probability mass. That is about 3× the base's hard-prompt success rate, and 2% of each hard prompt's mass.

**H1.** RL only produces mostly Δrel, and Δcap on hard prompts stays near zero when shared structure is zero.
- *Test* (arm `rloo`, k = 0):
  - (a) Δrel(all) ≥ 0.9 · (Δrel(all) + Δcap(all)), seed by seed. Supported if it holds in ≥ 9/10 seeds.
  - (b) Δcap(hard_heldout) ≤ τ0;
  - (c) Δcap(hard_train) ≤ τ0.
- H1 is supported if (a), (b) and (c) are all supported, and not supported if any is not supported. Otherwise it is inconclusive. `grpo` is reported as a robustness check, not a separate test.

**H2.** Δcap under RL only rises with shared structure. This is the genuinely uncertain hypothesis.
- *Test* (arm `rloo`, primary group hard_heldout):
  - the paired-by-seed difference Δcap(k = 4) − Δcap(k = 0) > τ0 (seeds pair across k because they share a seed index, not a base, so the bootstrap resamples seeds jointly);
  - and seed-mean Δcap is non-decreasing over k = 0, 2, 4.
- Supported if both hold; not supported if the difference's CI upper bound ≤ τ0; else inconclusive.
- Secondary: the same test on hard_train.

**H3.** SFT and hints produce Δcap on hard train prompts, and whether it transfers to held-out hard prompts depends on shared structure.
- *Test part (a):* for each arm in {sft, hints_h1, hints_h3} and each k, Δcap(hard_train) > τ0. H3a holds for an arm if it is supported at every k.
- *Test part (b):* for each of these arms, the difference Δcap(hard_heldout, k = 4) − Δcap(hard_heldout, k = 0) has a CI excluding 0.
  - "Depends on shared structure" is supported if this holds for at least one arm.
  - Report the direction and the per-k values of Δcap(hard_heldout) for every arm.
- H3 overall is supported if (a) holds for sft and at least one hint arm, and (b) is supported.

**H4.** SFT then RL beats either alone on held-out hard prompts.
- *Test:* metric J1(hard_heldout). At each k, compute the paired differences sft_then_rl − sft and sft_then_rl − rloo.
- Supported if both CIs exclude 0 on the positive side at every k. Not supported if, at any k, either CI lies entirely ≤ 0. Otherwise inconclusive, with per-k results reported.

## 4. Robustness and reporting (no new hypotheses)

- **Sensitivity:** every verdict is recomputed at c ∈ {0.5, 1, 2} and N ∈ {16, 64}. Any flip is reported next to the verdict (report generator flags).
- **Labels:** these are reported per arm × k:
  - the H12 label: beyond reach = a solution crosses c/N; beyond temperature = E_1 > 0 (decision H39);
  - the same label with E_N in place of E_1;
  - the D2 label.
- **E_N:** E_N for the per-step, sequence and both families is reported for every arm. A positive E_N means the trained policy beats every tempered base.
- **Samples:** the number of sampled episodes per arm is reported. Arms are matched on updates, not samples (decision H20).
- **Prominence:** negative, null and inconclusive verdicts are reported as prominently as positive ones.
- **Validity checks before any verdict:** Proposition 1 on every record, and exactly zero gain for `none` and at step 0.

## 5. What would change this plan

Nothing about the analysis changes after Gate 1. If a run fails or a bug is found, the fix and the re-run go in `docs/DECISIONS.md`, and the affected verdicts are recomputed under this same plan.
