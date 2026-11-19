# Gate 2: independent re-derivation and hypothesis verdicts

**Result: PASSED.** A subagent that did not build or run the experiment recomputed results directly from the saved base models and step-1000 checkpoints in `results/main/`. It used its own forward pass, and its own construction of the correct solutions from the task rules (decision H33). It did not use `report.py`, `verdicts.py` or the exact backend. Its scripts are kept outside the repo.

## Three headline numbers (N = 64, c = 1, update 1000, seeds 0–9)

| # | Quantity | Recorded by the harness | Re-derived | Max per-seed difference |
|---|---|---|---|---|
| 1 | rloo, k = 0, hard_train, Δcap: mean (min..max) | +0.5819 (+0.4908..+0.8639) | +0.5819 (+0.4908..+0.8639) | 5.6e-9 |
| 2 | sft, k = 4, hard_heldout, Δcap | +0.0107 (−0.0052..+0.0637) | +0.0107 (−0.0052..+0.0637) | 2.0e-7 |
| 3 | rloo, k = 0, all: Δrel / Δcap | +0.2264 / +0.1930 | +0.2264 / +0.1930 | 3.2e-8 / 2.0e-9 |

Beyond these three, the verifier recomputed **every** run, group and metric (Δcap, Δrel, J1). All 240 runs agree with the metrics files to within 3.4e-7.

## Verdicts (from `docs/experiment_main/verdicts.md`; independently re-checked with a fresh bootstrap)

| Hypothesis | Verdict | Independent check | Caveat that must accompany it |
|---|---|---|---|
| H1 | **not supported** | agree | It fails on hard **train** prompts only (Δcap +0.58), where RL found rare solutions over about 16,000 training samples per prompt. Held-out hard Δcap is −0.006, which meets the "near zero" part. This is not evidence that RL creates capability in general |
| H2 | **not supported** (primary, hard held-out) | agree | The effect is real but tiny: +0.0060 [+0.0040, +0.0081], below τ0 = 0.02. The secondary test on hard train is supported (+0.37), and must not be presented ahead of the primary |
| H3 | **supported** | agree (literal reading, H45) | Part (b)'s transfer effects are 0.008–0.015, statistically detectable but **below** the plan's own near-zero threshold. The sft result leans on one seed (+0.069). A percentile bootstrap over 10 seeds gives narrow intervals |
| H4 | **inconclusive** | agree | All differences are within ±0.007 and all intervals include 0. sft_then_rl − sft is negative on average at every k |

**No verdict flips** across N ∈ {16, 64} and c ∈ {0.5, 1, 2}.

**Validity:** completeness, coverage, Proposition 1 (worst residual 3.9e-16), exact zero gain for the `none` arm, and exact zero gain at step 0 all pass. The remaining warnings are sensitivity flips in labels and E_N signs, plus saturation on easy prompts at N = 64. They are listed in full in `docs/experiment_main/report.md`.
