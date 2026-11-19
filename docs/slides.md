# Slide outline: capability vs reliability (~12 minutes)

13 slides, about 55 seconds each. Every number comes from `docs/experiment_main/` or `docs/gate2.md`. Primary setting throughout: N = 64, c = 1, update 1000, seeds 0–9. Each value is a seed mean, with a bootstrap 95% interval over seeds where given.

---

### 1. Does RL teach new things, or just sharpen old ones?
- Post-training (RL, SFT) raises success rates. But is the model solving *new* problems, or just finding solutions it already had more often?
- The debate: "RL only sharpens the base" versus "RL expands what the model can do".
- Our approach: a small world where every probability is **exact**, so the question has a clean answer.
- Figure: none

### 2. The RL framing
- Episode = one prompt. The policy writes 5 tokens from a vocabulary of 6. Reward = 1 if the verifier accepts the sequence.
- State = prompt + tokens so far. Action = next token. Return = reward, so J = P(correct).
- Policy: one small GRU shared by all 72 prompts. The update rule depends on the arm.
- Easy and hard prompts, each split into **train** and **held-out**. Held-out prompts are never trained on.
- Figure: none

### 3. The split, in one equation
- In words: **gain in success rate = gain on solutions the base could already find in N tries + gain on solutions it couldn't.**
- "Could find" means base probability ≥ c/N (reachable). At N = 64, c = 1 the line is 1/64.
- J1 − J0 = Δrel + Δcap, exactly, on every prompt (Proposition 1). The worst residual over all records is 3.9e-16.
- Either part can be negative.
- Second check: E_N / E_1 compare the trained policy with the best *tempered* base, the "free" improvement that needs no training.
- Figure: none (optionally the worked example from `docs/explainer.md` §2)

### 4. The base design fix
- Pilot flaw: if the base's most likely sequence is already correct, tempering alone gets close to perfect. Then there is nothing for training to explain.
- Fix: every prompt's most likely sequence is a wrong **trap**. Hard solutions are made **unreachable**, and no amount of tempering makes them reachable (decisions H10, H34–H36).
- **Shared-structure dial k ∈ {0, 2, 4}:** a hard solution follows the easy rule for its first k tokens.
- These properties are verified for all 30 bases (`docs/base_verification.md`) and frozen before the main run.
- Figure: none

### 5. The training arms
- `none` (control) · `rloo`, `grpo`, `rl_entropy` (RL only) · `sft` (demonstrations, hard-train only) · `hints_h1`, `hints_h3` (RL with a solution prefix on all-fail groups) · `sft_then_rl`.
- Same budget for every arm: 1,000 optimizer updates. Arms are matched on **updates, not samples**: SFT draws no samples and hint arms draw extra (H20).
- 8 arms × 3 values of k × 10 seeds = 240 runs, all measured exactly at 9 checkpoints.
- Figure: none

### 6. Pre-registration
- Hypotheses, tests and the tolerance τ0 = 0.02 were frozen (tag `gate1-frozen`) before any main run.
- Rule for a claim "X > τ": **supported** if the CI's lower bound is > τ, **not supported** if its upper bound is ≤ τ, otherwise **inconclusive**.
- **H1:** RL gives mostly reliability, and ~0 capability at k = 0.
- **H2:** RL's capability rises with shared structure.
- **H3:** SFT and hints give capability, and transfer depends on k.
- **H4:** SFT then RL beats either alone on held-out hard prompts.
- Verdicts are computed by code (`python -m capsplit.harness verdicts`). An independent re-derivation agreed on all 240 runs to within 3.4e-7 (Gate 2).
- Figure: none

### 7. Verdicts (each with its caveat)

| | Verdict | Caveat |
|---|---|---|
| H1 | **not supported** | Fails only on hard **train** prompts: Δcap +0.58, where RL found rare solutions over about 16,000 training samples per prompt. Held-out hard Δcap is −0.006, which meets "near zero". This is not evidence that RL creates capability in general |
| H2 | **not supported** (primary, hard held-out) | The effect is real but tiny: +0.0060 [+0.0040, +0.0081], below τ0 = 0.02. The secondary test (hard train, +0.37) is supported, but it is secondary |
| H3 | **supported** | Part (b)'s transfer effects are 0.008–0.015: detectable, but below τ0. The sft result leans on one seed (+0.069). Ten seeds give narrow intervals |
| H4 | **inconclusive** | All differences are within ±0.007 and every CI includes 0. sft_then_rl − sft is negative on average at every k |

- No verdict flips across N ∈ {16, 64} and c ∈ {0.5, 1, 2}.
- Two negatives, one null and one positive. The positive one is the weakest-looking.
- Figure: none (the table is the slide)

### 8. Where RL's gain goes (H1)
- `rloo`, k = 0, all prompts: Δrel +0.2264 and Δcap +0.1930. That is far from "≥ 90% reliability" (the condition held in 0/10 seeds).
- All of the capability gain is on **hard train** prompts: Δcap +0.5819 [+0.5317, +0.6564].
- Hard **held-out** Δcap is −0.0058 [−0.0063, −0.0052]: slightly negative.
- GRPO gives the same verdict (robustness check).
- Figure: `training_01_rloo.png` (rows: all / hard_train / hard_heldout)

### 9. Key insight: reachability is budget-relative
- "Unreachable" means unreachable **in N = 64 tries**. RL trained with about 16,000 samples per hard train prompt (1,000 updates × G = 16).
- With 250× more tries than the measurement budget, RL can find "unreachable" solutions by brute exposure, then reinforce them.
- So Δcap on train prompts measures **search budget** as much as new skill. This is what sinks H1(c).
- The honest reading: RL turned rare-but-possible into reliable on the prompts it practised. There is no sign it learned the general hard rule (see slide 10).
- Figure: none (optionally the N = 16 / 64 / 256 table from `docs/explainer.md` §2, step 3)

### 10. No held-out transfer, and some damage
- Δcap on hard held-out prompts is below τ0 = 0.02 for every arm and every k. The largest is sft at k = 4: +0.0107 (−0.0052..+0.0637).
- At k = 0 and 2, every trained arm is slightly negative on held-out hard prompts.
- Frontier: on hard held-out, every trained arm falls **below** the tempered base. Its pass@64 drops from ~0.34 (untrained) to ~0.03–0.18.
- Costs elsewhere: SFT lowers easy-train reliability (Δrel −0.2443 at k = 0), and RL lowers easy held-out J (rloo k = 0: −0.0884).
- Figures: `main_grid.png` (mind the ±0.01 colour scale), `frontier.png`

### 11. SFT, hints and combinations (H3, H4)
- On hard train, SFT reaches Δcap ≈ +0.989 at every k. Hints reach +0.74 to +0.99, rising with k for hints_h3.
- Transfer rises with k, but stays tiny: sft k = 4 − k = 0 is +0.0146 [+0.0043, +0.0287], driven by one seed.
- SFT then RL is no better than either alone on held-out hard J1 (H4 inconclusive).
- Figure: `training_04_sft.png`

### 12. Limitations
- A toy world: 72 prompts, 5 tokens, vocabulary 6, one small GRU. **These results do not transfer to LLMs** without new evidence.
- The base is engineered (traps, calibrated hard mass), so the "hard" here is hard by construction.
- Arms are matched on updates, not samples or compute.
- The saturation warning: on easy prompts at N = 64, the tempered base already has pass@N ≈ 1, so E_N is blind there. Labels use E_1, but the E_N signs and labels flip under other (N, c).
- Ten seeds and a percentile bootstrap give narrow intervals, which describe seed spread only.
- Figure: none

### 13. What's reusable: the harness
- Plug-in tasks, bases, policies, trainers and backends, selected by name in a JSON config. Typos fail loudly.
- An exact backend (enumeration) and a sampled backend for bigger worlds. Validity checks are built in (Proposition 1, zero-gain control, sensitivity, saturation).
- `run` → `report` → `verdicts`: deterministic, resumable, and every number traceable to a metrics file.
- Next steps: measure reachability at the *training* budget as well as at N. Add held-out structure that RL could plausibly learn.
- Figure: none
