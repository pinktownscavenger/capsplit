# Pre-registration: local language-model experiment (L1–L3)

**Frozen** together with `configs/lm_main.json` and `configs/lm_prereg.json` at git tag `lm-prereg-frozen`, after the `data` and `base` stages and **before any training** (decision L17).

**What had been seen at freezing time:**
- the base-model outputs: `data/summary.json`, `base/classes.json`, `base/summary.json`;
- the speedtest;
- the 10-problem demo run (`configs/lm_demo.json`, 4 updates; demo numbers are not used here);
- all toy-experiment results, including the toy seed study.

No training of `configs/lm_main.json` had run.

## Setting (RL framing)

| | |
|---|---|
| Episode | One generated arithmetic problem: 2–4-digit operands, + or ×, 180 problems, 120 train and 60 held out |
| Action | The next token of the completion (at most 12 tokens) |
| Reward | 1 if the first integer in the completion equals the answer (program checker), else 0 |
| π0 | `Qwen/Qwen2.5-0.5B-Instruct` at the pinned revision, LoRA adapter disabled |
| π1 | The same model with a trained LoRA adapter (r = 8, on q and v projections) |
| Arms | `none`; `rl` (GRPO-style, 8 problems × 8 samples per update); `sft` (on train problems the base cannot reach); `sft_then_rl` (half and half); 200 updates each, seeds 0, 1, 2 |

**Measurement** (decision L11 and `docs/lm_design.md`):

- **Classification.** Set A (400 base samples per problem) classifies each problem. With a 95% Clopper–Pearson interval and N = 16:
  - **reachable:** the interval's lower bound is ≥ 1/16;
  - **unreachable:** its upper bound is < 1/16;
  - **ambiguous:** otherwise.
- **Base success.** Set B (64 samples) gives p̂0. Every trained arm is sampled 64 times per problem for p̂1.
- **Contributions.** Each class's contribution is its summed gain divided by the number of problems in the group.
- **Intervals.** The bootstrap over problems redoes the classification inside every resample.
- **Temperature baseline.** Split-half choice of T among {0.5, 0.7, 1.0, 1.3}.
- **Null check.** A fresh base sample is measured as if it were a trained policy.

## Hypotheses and exact tests

Thresholds are in `configs/lm_prereg.json`, applied by `python -m capsplit.harness.lm verdicts` (`src/capsplit/harness/lm/verdicts.py`). τ = 0.02 matches the toy experiment's near-zero threshold.

**Decision rule** for "X ≤ τ":
- **supported** if the interval's upper bound is ≤ τ;
- **not supported** if its lower bound is > τ;
- **inconclusive** otherwise.

**L1. RL gains on training problems classified unreachable occur only where training itself sampled a correct answer.**
- **Data:** arm `rl`, each seed separately. Take the train problems classified unreachable, and split them by whether the training log shows any correct sampled completion on that problem (`first_success_update` not null).
- **Statistic:** the contribution to the train group of the never-correct subset's gains.
- **Rule:** supported if, in every seed, its upper bound is ≤ τ_l1 = 0.02. Not supported if, in any seed, its lower bound is > 0.02. Otherwise inconclusive.
- **Guard:** inconclusive if fewer than 3 problems are in the never-correct subset.
- **Also reported:** the ever-correct subset's contribution, and per-problem tables.

**L2. No arm produces capability gain on held-out unreachable problems beyond a near-zero threshold.**
- **Statistic:** for every trained arm, seeds pooled, the unreachable-class contribution on the held-out group.
- **Rule:** supported if every arm's upper bound is ≤ τ_l2 = 0.02. Not supported if any arm's lower bound is > 0.02. Otherwise inconclusive.
- **Guard:** inconclusive if no held-out problem is unreachable.

**L3. SFT on unreachable training problems lowers success on reachable problems.**
- **Statistic:** for arm `sft`, seeds pooled, the reachable-class contribution on all problems.
- **Rule:** supported if its upper bound is < 0 (τ_l3_effect = 0). Not supported if its lower bound is ≥ 0. Otherwise inconclusive.
- **Also reported:** the train and held-out splits separately; they do not enter the rule.

## Also reported (no hypothesis attached)

- **Totals and labels:** total gain, the three contributions, E_1 and E_N against the temperature baseline, and labels for every arm × seed.
- **Budget curve (addendum A):** the split as a function of N, truncated at the largest N the 400-sample classification can resolve (about 108), with each arm's logged N_train marked.
- **Discovery (addendum B):** per-problem first-success updates and correct counts from the training logs. These are compared with the waiting-time model P(no success in t updates) ≈ (1 − p)^(8·v(t)), where v(t) is the number of updates in which the problem was drawn. This comparison is descriptive; with one model and 3 seeds it is not a hypothesis test.
- **Format validity** for every arm.
- **Null check:** a failure is reported at the top of the report.

## What this experiment cannot show

- **Scale.** One small model (0.5B) with LoRA, generated arithmetic only, and 3 seeds. Nothing here generalises to larger models, other tasks or full fine-tuning.
- **Measurement is sampled, not exact.** Classification uncertainty is carried into the intervals, but problems near the 1/N line can be misclassified.
- **The checker is strict about format.** A correct number that comes after other text (e.g. an echoed equation) counts as wrong. Format validity is reported separately.
