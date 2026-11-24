# Function-space (KL) layer: the answer in plain words

**Question:** does the KL layer show anything the output metric alone does not?

**Answer: yes, two things, both descriptive.**

1. **Movement without progress.** On training problems that stayed unreachable and were never once sampled correct during RL, the success metric shows no change. Yet the output distribution moved far from the base there, by roughly as much as on the problems RL did discover. It moved much more than on problems that were already reachable. The same holds for held-out unreachable problems, where the success gain is near zero but the KL is large for every arm.
   - The success metric cannot see this.
   - It also cannot say what the movement is. An informal diversity check on these problems (RL, one seed, done on interim data and not part of the generated report) found the wrong answers being reshuffled, not collapsed onto one answer. The KL there is large partly because the base spreads its mass over many unlikely wrong answers, so moving that mass costs many nats.
2. **The methods differ in how far they move the model per unit of gain.**
   - RL reaches its gain with much less movement of the output distribution than SFT.
   - SFT moves the distribution furthest of all, including on reachable problems, where its accuracy went down.
   - In SFT-then-RL, the RL phase raises the gain while the total KL stays roughly flat. That fits the RL phase redistributing mass within distance the SFT phase had already moved, though the over-updates estimates are noisy.

**What this does not show.** Nothing here says whether the models learned "little" or "a lot", or gained or lost capability.
- KL measures how far the output distribution moved. It does not measure direction, usefulness or capability.
- The weight descriptors are coordinates and define nothing about capability.

## The pre-registered checks, in order

**K1, the bound check: failed as registered.**
- The violation fraction is well above the pre-registered 5% line, for every arm and with both smoothings.
- Diagnosis: almost all violations are an artefact of the bound check, not of the KL estimator.
  - **Smoothing artefact.** Many violations are cells where the raw success rates are both 0 or both 1. There, smoothing two different sample sizes produces a small positive bound while the KL is about zero.
  - **Bound noise.** The rest are cells where the trained model is near-deterministic, so the KL's standard error is about zero, and the KL sits just under a bound that has sampling noise of its own. The registered rule ignores that noise.
- The post-hoc variant (DECISIONS L34) handles these: it excludes the exactly-degenerate cells and includes the bound's noise. With it, violations are essentially absent.
- The exact-zero checks passed: every fresh adapter gives exactly zero log-ratio.
- The p̂0-from-B-only sensitivity (L33) also lowers the violation rate. This shows the selection effect of classifying with set A.
- The registered K1 result is reported first and unchanged in the generated report below.

**K2, KL by class group:**

| Prediction | Who | Outcome |
|---|---|---|
| (a) For RL, problems discovered in training have the largest KL | user | Supported |
| (b) Never-discovered unreachable problems sit near the reachable problems, and above zero | user | **Contradicted.** They are above zero, but far above the reachable problems, not near them |
| Never-discovered unreachable problems are below reachable ones | mine | **Contradicted** |
| For SFT, the trained-on unreachable problems have the largest KL | mine | Supported |
| For SFT, held-out unreachable exceeds held-out reachable | mine | Supported |

**K3, the held-out gap (descriptive, no new verdict on L2):**
- **Supported:** the held-out unreachable KL interval lies above zero for every arm.
- **Mixed:** the prediction that held-out unreachable efficiency is below held-out reachable efficiency.
  - It holds for RL.
  - For SFT-then-RL it holds in the point estimates, with overlapping intervals.
  - For SFT it does not hold. SFT's held-out reachable gain is negative, so its efficiency there is negative.

**Over updates:**
- **KL rises with updates.** This holds for SFT on every seed. For RL it holds on two seeds; on the third, the step-100 estimate (from fewer samples) is above the step-200 one.
- **RL reaches under half of its final KL by update 25.** Supported on every seed.
- **SFT reaches a larger fraction of its final KL by update 25 than RL.** Not supported consistently; it depends on the seed.

## Independent re-derivation

One KL, one bound and one effective rank were recomputed by `scripts/rederive_kl.py`. That script does not import the KL stage code, rescores the saved token ids in float32 on CPU and reads the raw sample and adapter files. Its output, with the differences from the stage's values, is in `rederivation.txt`.

- **Bound and effective rank:** both match the stage to floating-point rounding.
- **KL:** the independent value is a few percent lower than the stage's.
  - `scripts/kl_precision_check.py` re-scored the same samples with the stage's own code. In float32 on CPU it reproduces the independent value; in float16 on MPS, the run's setting, it reproduces the stored samples exactly.
  - So the gap comes from numeric precision, not from a code difference (`precision_check.txt`, DECISIONS L35).
  - The reported KLs are those of the float16 model that was trained and sampled. Individual log-ratios can differ between precisions by a noticeable fraction of a nat.

## Caveats

- **Scope:** one 0.5B model with LoRA, generated arithmetic only, and 3 seeds.
- **Small group:** only 26 held-out unreachable problems.
- **Estimation:**
  - KL is a sampled estimate from 64 completions per problem at the final checkpoint, and from fewer at the intermediate ones.
  - Completions are truncated at 12 tokens.
  - Heavy-tailed log-ratios can make a few standard errors unreliable.
  - KL values are for the float16 model on MPS. A float32 re-scoring of one problem gave a KL a few percent lower.
- **Interpretation:**
  - Large KL on unreachable problems reflects how spread out the base distribution is there, as much as how much the model changed.
  - The pre-registration was written after the output-metric results were known, so this layer is exploratory.
