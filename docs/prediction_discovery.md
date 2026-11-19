# Pre-registered prediction: is RL discovery a waiting time? (addendum B4)

**Committed before any seed-study run** (configs `seedstudy_*`, results `results/seedstudy_*`).

**What had already been seen when this was written:** the aggregate results of the frozen main toy run (`docs/experiment_main/`). In particular, for `rloo` at k = 0, hard-train Δcap ranged from 0.49 to 0.86 across seeds, and in seed 0, 21 of 24 hard-train prompts ended with Δcap > 0.98. No per-prompt first-success times had been logged or seen; that logging did not exist until this addendum.

## Model of discovery (no drift)

Take one hard training prompt, whose base success probability (both correct solutions together) is p, and train with RLOO using G samples of that prompt per update.

1. While no correct sample has appeared on this prompt, every advantage in its group is exactly 0. So the prompt gets no direct gradient. Its p changes only through shared parameters, i.e. through learning on other prompts.
2. If p stays at its base value (**no drift**), each update independently contains at least one correct sample with probability π = 1 − (1 − p)^G.
3. So the first-success update T is **geometric**:

   P(T > t) = (1 − p)^{G·t},  E[T] = 1/π,  median(T) = ⌈ ln 2 / (−G·ln(1 − p)) ⌉.

4. The probability that a prompt is **never discovered** within a budget of B updates is (1 − p)^{G·B}.
5. Summed over prompts, the predicted fraction of hard-train prompts never discovered is the mean over those prompts of (1 − p_i)^{G·B}.

Here p_i is computed exactly from the frozen base for each hard-train prompt. G = 16 and B = 1,000 in the seed-study configs, which use the same trainer settings as the main run.

**Numeric predictions:** `python -m capsplit.harness discovery predict <config>` writes per-prompt p_i, E[T], median T and P(undiscovered by B), computed from the bases alone. Its outputs are committed under `docs/seedstudy/predictions/` **before** the seed-study runs.

**Expected order of magnitude:** base hard-solution probabilities are calibrated so the median hard solution has q ≈ 0.2/64 ≈ 0.003, giving p ≈ 0.006 for two solutions. With G = 16, π ≈ 0.09, so E[T] ≈ 11 updates, and P(undiscovered by 1,000 updates) ≈ e^{−96} ≈ 0. **Without drift, the model predicts essentially every hard-train prompt is discovered within the first ~100 updates.**

## How drift should bend the prediction

Learning on other prompts changes p before discovery. The direction depends on shared structure k:

- **k = 0.** A hard prompt's most likely output is the easy rule applied everywhere (the trap). RL on easy prompts reinforces the easy rule, which should *raise* the trap's probability on hard prompts and *lower* p. Prediction:
  - first successes later than the geometric model;
  - a heavier right tail;
  - a non-zero fraction of prompts never discovered, because p can fall toward 0 before the first success arrives.
  
  The no-drift model is then an **upper bound on discovery speed**.
- **k = 4.** The hard solution shares its first four tokens with the easy rule, so learning the easy rule raises the probability of the shared prefix. That may raise p, which predicts first successes no later than the geometric model.
- **k = 2:** in between.
- **After discovery:** a discovered prompt's p rises quickly, because RLOO then has non-zero advantage there. "Discovered" is necessary but not sufficient for a large final gain, since later learning can still erode it.

## What would count as agreement

- **Agreement:** at k = 4 (where drift is predicted to be weakest or helpful), the observed first-success times follow the geometric prediction. Concretely, the observed median T is within a factor of 2 of the predicted median for most prompts, and the observed fraction never discovered is ≈ 0.
- **Disagreement in the predicted direction:** observed T is systematically later than predicted at k = 0, with a non-zero never-discovered fraction. This is evidence of negative drift, not a failure of the waiting-time picture.
- **Disagreement against the model:** first successes much *earlier* than predicted at k = 0, or prompts with high base p never discovered across most seeds while low-p prompts are discovered. That would mean something other than chance sampling of the base distribution determines discovery.

## Determined by the base, or by luck?

The seed study holds the base fixed and varies only the training-sampling seed (30 seeds), and separately varies only the base seed (30 seeds). Predictions:

- **Under the waiting-time picture,** with a fixed base and B·π ≫ 1, almost every prompt is discovered in every seed. Then the training-sampling seed should explain little of the variance in a prompt's discovery, and differences in final gain should come mainly from drift and post-discovery dynamics.
- **If drift at k = 0 drives p toward 0 for some prompts,** the outcome on those prompts becomes seed-dependent: the same base gives different outcomes under different sampling seeds.
- **The verdict will be stated per k** as one of: "outcome determined by the base", "outcome depends on sampling luck", or "mixed". It comes from the variance decomposition of final per-prompt Δcap into a training-seed component and a base-seed component.

## Addendum: operational thresholds and a disclosed preview (committed before the seed study)

The criteria above were turned into fixed constants in `src/capsplit/harness/discovery.py` before any seed-study run:

- **"Most prompts within ×2 of the predicted median":** this counts only prompts with at least 5 runs. Otherwise the pooled median ratio must lie in [0.5, 2]. A single geometric draw falls within ×2 of its median only about 46% of the time.
- **"≈ 0 never discovered":** at most 1% of prompt-runs.
- **"Later" or "much earlier" than predicted:** a randomised probability-integral-transform z-score beyond ±3, together with the direction of the pooled median ratio.
- **"Determined by the base" / "depends on sampling luck":** the ratio of variance across training seeds to variance across base seeds, V_train / V_base, at ≤ 1/3 or ≥ 2/3 respectively; anything in between is "mixed".

**Disclosure.** While testing the analysis code, the implementer ran a small scratch study (k = 4, 3 seeds, B = 100 updates; not committed). It saw 2 of 72 prompt-runs never discovered, against 3×10⁻⁵ predicted, and an observed median first-success update of 6 against 7 predicted. The thresholds above were written by the same implementer; they are stated here so that readers can judge them. Neither the prediction formulas nor the expected directions in this file changed.
