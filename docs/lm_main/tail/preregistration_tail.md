# Pre-registration: the base model's tail success probability (lm_main)

Written and committed **before any number from the new tail code existed**: no lower bound D, no bracket, no importance-sampling estimate. Tagged `lm-tail-prereg`. Decisions are in `docs/DECISIONS.md` (L36 onward).

**What had already been seen:** all `lm_main` results (report, verdicts, discovery comparison, budget curve, null check) and the KL layer. The design below was fixed with those in view, so this is not a blind design.

**Method credit:** sequence-level importance sampling for rare language-model outputs is not new. See Angell, Singhal, Horvitz, Yu, Ranganath, McKeown and He, *Estimating Tail Risks in Language Model Output Distributions*, ICML 2026, arXiv:2604.22167. Here it is only applied to per-problem correctness and reachability budgets.

## Target quantity

**Definition.** p0(x) is the probability that the base model, with the LoRA adapter disabled, writes a completion the strict checker accepts for problem x.
- **Sampling:** at temperature 1, with top-p 1.
- **Completion:** at most 12 tokens, stopping at the first stop token. The decoded text skips special tokens.
- **Acceptance:** the first number in the text is an integer equal to the answer (`lm/tasks.py: check`).

**The model.** p0 is that of the run's model, i.e. float16 weights on MPS, with log-probabilities taken as log-softmax in float32 of the logits.
- Every "exact" or "certified" bound below is exact up to the numerics of that forward pass.
- A float32 CPU recomputation of one D is part of the independent re-derivation.

**Base samples used as a reference.** Only temperature-1 base samples are draws from π0: sets A (400), B (64), C at T = 1 (32) and null (64), so 560 per problem.
- The brief says 752; the saved files hold 560 temperature-1 samples per problem (DECISIONS L36).
- Clopper-Pearson (CP) intervals use these 560.
- **"Never-correct"** means 0 successes in all 560.

**Problem sets:**
- **Target set:** the 73 problems classed unreachable and the 3 classed ambiguous on set A at N = 16, 76 in all.
- **D and the bracket** are computed for all 180 problems.

## Step 1: exact lower bound D

D = π0(answer digits | prompt) × π0(E | prompt, answer digits).
- **The digit path.** Teacher-force the answer's digit tokens directly after the prompt. Qwen2.5 writes every digit as its own token, and no other vocabulary entry contains a digit (checked before writing this).
- **E, the tokens that end the integer:**
  - the stop tokens;
  - every non-special token whose decoded text starts with a character other than `0-9`, `.` or `,`.
- **Excluded from E:** `.` and `,`, which can continue the number; tokens that decode to an empty string; and the padded logit rows beyond the tokenizer's vocabulary.
- If the answer's digits use all 12 tokens, the end factor is 1.

D ≤ p0 exactly, because every such completion is accepted.

**Check.** For each problem, the one-sided CP upper bound at level 1 − 0.05/180 (Bonferroni over the 180 problems) must not lie below D. A failure is a bug to investigate. The count of failures at the unadjusted 95% level is reported too.

**GATE 1.**
- **Rule:** on problems with p̂ ≥ 1e-2 (p̂ from the 560 samples), D/p̂ must be ≥ 0.1.
- **If not:** report how many problems fail and the ratios, flag that the bracket will be loose, and continue.

## Step 2: exact bracket by partial expansion

**The tree.** It is expanded from the prompt, best first by base prefix probability. A prefix is classified as follows.
- **won:** the first number is already final and equals the answer, so every continuation is accepted.
- **dead:** the first number is already negative, or its leading digits (with leading zeros stripped) are not a prefix of the answer, or it is final and not equal to the answer.
- **terminal:** a stop token was drawn or 12 tokens were reached. It is then won or dead by the checker itself.
- **open:** otherwise.

"Final" is decided by checking that the checker's regular expression gives the same match for every extension of up to 5 characters, drawn from the classes {digit, `.`, `,`, `-`, other}. The regular expression looks at most 5 characters past a match.

**Children.**
- All 10 digit tokens and the stop tokens are explicit children of every expanded node.
- The other tokens are grouped by the character classes of their first 6 decoded characters, and each group's mass is classified as a whole.
- Of the open children, the 64 most probable become explicit nodes. The rest of the open mass stays in the open total, unexpanded.

**Budget.** 4,096 node expansions per problem, run as batched forwards of 64 prefixes.

**Bracket.** p0 ∈ [max(D, W), W + O], where W is the won mass and O the open mass.

**Check.** For every problem, the two-sided CP interval at level 1 − 0.05/180 must intersect the bracket. If one does not, that is a bug: investigate before continuing.

## Step 3: importance sampling for the open remainder only

**Proposal.** A defensive mixture with equal weights, one component chosen per episode:
1. π0 at T = 1;
2. the base at T = 1.3;
3. the SFT adapter (seed 0, step 200) at T = 1;
4. the RL adapter (seed 0, step 200) with lora_B scaled by 0.5, at T = 1.

The RL adapter at full scale is never used alone.

**Samples.** n = 1,024 completions per problem, 12 tokens at most.
- The token ids are recorded, with each chosen token's log q at the component's own temperature at sampling time.
- Every sequence is then scored teacher-forced under π0 and all 4 components. The per-token gap between the sampling-time and teacher-forced log q of the sampled component is reported.
- All scoring runs under no_grad.

**Estimator (plain, never self-normalised).**
- p̂0 = W + (1/n) Σ 1(y ∈ S) · 1(y ∈ R) · π0(y) / q(y), where q is the mixture density from the teacher-forced log-probs.
- R is the set of completions not resolved by the tree, i.e. whose path leaves the explicit tree through open mass. Completions in the won set are already counted in W.
- SE = sd / √n of the summands.
- The same estimator, mixture and n are used for every problem: discovered, never-discovered, reachable and control alike.

**Trust rule** (per problem, on the successful remainder samples with weights w):
- at least 20 such samples;
- effective sample size (Σw)² / Σw² ≥ 10;
- no single w above 20% of Σw;
- Pareto tail shape k̂ ≤ 0.7, from the Zhang–Stephens generalised Pareto fit to the largest ⌈min(0.2·n_s, 3·√n_s)⌉ weights.

If the open mass O is 0, the bracket is exact and IS is not needed. An untrusted problem is reported as bracket-only.

**Controls** (failing either one is a **STOP GATE**):
- **Known-answer control.** On every problem in the IS sets, estimate D from the same samples: a full, not remainder, IS estimate of the mass of the D-set. Here z = (D̂ − D)/SE. The control fails if any problem whose D-set estimate passes the trust rule has |z| > 3.5, or if fewer than 5 problems pass the trust rule.
- **Brute-force control.**
  - *Problems:* 12, chosen by rank before any tail number exists. Among problems with p̂ (560 samples) in [0.005, 0.06], sorted by p̂, take 12 evenly spaced ranks.
  - *Samples:* 4,000 fresh π0 samples each (T = 1, a new seed stream), plus the IS estimate at the same settings.
  - *Rule:* z = (p̂_IS − p̂_BF) / √(SE_IS² + SE_BF²). The control fails if any trusted problem has |z| > 3, or if fewer than 6 of the 12 are trusted.

If the gate fails: stop, write up what failed and why, and report only the brackets from steps 1–2.

Nothing (thresholds, mixture, n, budget) is tuned on the samples that are reported.

## Step 4: use

**Discovery.**
- Rerun the waiting-time comparison of `lm/discovery.py` (rl arm, G = 8, the logged draws).
- For the problems with p̂_A = 0, use p0 at the bracket's lower and upper ends, and at the trusted IS estimate where one exists. That gives the predicted number of discoveries as an interval, compared with the 0 observed.
- This is shown next to the earlier comparison: 20.10 predicted vs 18 observed over all unreachable observations.

**Budget curve past N = 108.**
- **Labels** for N on a grid of powers of 2 up to 2^20:
  - reachable if the bracket's lower end ≥ 1/N;
  - unreachable if its upper end < 1/N;
  - ambiguous otherwise.
- **Contributions:** as in `measure.budget_curve`, with gain = p̂1 (eval) − p̂0 (set B), per arm and seed.
- **Two versions:** brackets only (primary), and brackets narrowed to the IS 95% interval where IS is trusted (secondary).

**KL.** One number per arm, Σ d(p̂1 ‖ p0_hi) / Σ KL̂, over the target problems.
- p0_hi is the bracket's upper end, which gives the smallest d consistent with the bracket.
- d is undefined (infinite) when p0 = 0 and p̂1 > 0.

## Predictions

| # | Prediction | Made concrete as |
|---|---|---|
| P1 (user) | The exact bracket is tight for 3- and 4-digit multiplication only if the model reliably starts with a digit | Tight means upper / lower ≤ 10. Among 3- and 4-digit multiplication target problems, the tight ones have a higher mean first-token digit probability π0(first token is a digit) than the loose ones |
| P2 (user) | The never-correct problems' bracketed p0 is mostly below 1e-3 | More than half of them have p0 < 1e-3: by the bracket's upper end (certified), or by the trusted IS estimate (estimated). Both are reported |
| P3 (user) | The predicted discovery count among the p̂_A = 0 problems is small, near 0 | The predicted count at the bracket's upper end, pooled over 3 seeds, is ≤ 2 |
| P1' (mine) | P1's mechanism is right, but at this budget most unreachable multiplication brackets will **not** be tight | The open mass from non-digit openings and digit paths left unexpanded will exceed these problems' small p0, so fewer than half of the target problems have upper / lower ≤ 10 |
| P2' (mine) | p0 < 1e-3 for most never-correct problems will hold by D and by the IS estimates, but will **not** be certified by the bracket's upper end for most of them | — |
| P3' (mine) | The count is near 0 at the lower end and at trusted estimates, but the upper-end count will not be ≤ 2. The bracket will be too loose to rule discoveries out | — |

## Caveats (stated in advance)

- **Scope:** one 0.5B model, and 12-token arithmetic completions.
- **Unresolved:** the null-check failure of the main run (z = -3.2).
- **Silent misses:** importance sampling can miss success strings that the proposal never visits. The bracket does not have that problem, but it is only as tight as its budget allows.
