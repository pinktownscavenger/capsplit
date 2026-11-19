# Explainer: capability vs reliability, for a new teammate

This is the plain-language companion to the RQ3 Phase 2 piece of the project (the harness and its experiment), not to the whole project. Read it before `docs/preregistration.md`, `docs/gate2.md` and `docs/experiment_main/report.md`. It has five parts:
- what each piece of the code is;
- a small worked example of the split;
- how to read the figures;
- how to run the code;
- a glossary.

**One-sentence version.** When post-training raises a policy's success rate, we split the gain exactly into two parts:
- **reliability**: more probability on solutions the base model could already find within N tries;
- **capability**: more probability on solutions it practically could not find within N tries.

We then ask which training signals (RL, SFT, hints, SFT then RL) produce which part, in a small world where every probability can be computed exactly.

---

## 1. The pieces

### The RL framing

| RL object | In this project |
|---|---|
| Episode | One prompt. The policy writes a sequence of L = 5 tokens, each from a vocabulary of V = 6 |
| State | The prompt plus the tokens written so far |
| Action | The next token |
| Reward | 1 at the end if the verifier accepts the whole sequence, otherwise 0 |
| Return | Equal to the reward (one terminal reward, no discount), so the expected return on a prompt is J = P(correct) |
| Policy | A neural network that gives a probability for each next token, given the state |
| Update rule | Depends on the training arm: a policy gradient (RLOO, GRPO), supervised learning (SFT), or a mix |

For a prompt, write q(s) for the probability that the policy writes the whole sequence s. Then J = Σ q(s), summed over the correct sequences s. A subscript says which policy is meant: **q0** is the base model before post-training, and **q1** is the policy after training.

### Each component

These live in `src/capsplit/harness/`. Each kind of component has an abstract class in `interfaces.py` and registers itself by name in `registry.py`. The config refers to components by these names.

- **Task** (`tasks/rule_task.py`, name `rule_task`)
  - The environment and verifier.
  - A prompt is (type, a, b), with type easy or hard and a, b in 0..5, so there are 72 prompts. Each prompt has exactly two correct sequences.
  - Easy prompts follow an "easy rule".
  - Hard prompts follow the easy rule for their first **k** tokens, then a different "hard rule". k ∈ {0, 2, 4} is the **shared-structure dial**: how much of a hard solution uses a skill the base already has.
  - A third of each type's prompts is held out. Training never sees held-out prompts (decisions H9, H33).
- **Base model** (`bases/pretrain.py`, name `pretrain_gru`)
  - The starting policy π0, pre-trained by weighted maximum likelihood on all prompts.
  - It is built so that:
    - every hard solution is *unreachable* (very low probability);
    - easy solutions are reachable but not saturated;
    - each prompt's most likely sequence is a wrong "trap".
  - See "the base design fix" in the slides and decisions H10, H34–H36.
- **Policy** (`policies/prompt_gru.py`, name `prompt_gru`)
  - One small GRU network for all 72 prompts (embedding 16, hidden 64). Because it is shared, training on some prompts can change behaviour on held-out prompts. That is how we can measure transfer.
- **Trainers / arms** (`trainers/`)
  - An *arm* is a named trainer with its settings. Every arm gets the same budget: 1,000 optimizer updates, Adam, lr 1e-3. Arms are matched on updates, not on samples (H20).
    - `none`: no training (control). Its gain must be exactly 0.
    - `rloo`: RL with outcome reward. Groups of G = 16 samples per prompt, with a leave-one-out average of the other samples as the baseline.
    - `grpo`: RL with a group-normalised advantage (robustness check).
    - `rl_entropy`: RLOO plus an entropy bonus of 0.01.
    - `sft`: supervised fine-tuning on correct demonstrations of the **hard train** prompts only. It draws no samples.
    - `hints_h1`, `hints_h3`: RL, but when all 16 samples on a prompt fail, the group is re-rolled starting from a 1- or 3-token prefix of a correct solution. The forced tokens get no gradient (H17).
    - `sft_then_rl`: SFT for half the budget, then RLOO.
- **Measurement backend** (`backends/exact.py`, name `exact`)
  - For a policy, it lists all 6^5 sequences of every prompt and computes their exact probabilities.
  - It returns J, Δrel, Δcap and E_N for each group of prompts, at each budget N and multiplier c.
  - Because nothing is sampled, the numbers have no measurement noise. A `sampled` backend exists for worlds too big to enumerate.
- **Runner** (`runner.py`, `cli.py run`)
  - Reads a JSON config (`configs/main.json`).
  - For every (k, seed), it builds the base once and trains every arm from it, saving checkpoints at updates 0, 25, 50, 100, 200, 300, 500, 750 and 1000.
  - It measures each checkpoint and writes one metrics file per run, plus `index.json`.
- **Report** (`report.py`, `cli.py report`)
  - Reads every metrics file and runs the validity checks: completeness, Proposition 1, exactly zero gain for `none` and at step 0, sensitivity, and saturation.
  - Writes `report.md`, CSV tables, `summary.json` and figures.
  - Each cell shows the seed mean, then a bootstrap 95% interval over the 10 seeds, then (min..max).
- **Verdicts** (`verdicts.py`, `cli.py verdicts`)
  - Applies the frozen pre-registered tests (`docs/preregistration.md`) mechanically. Each hypothesis gets supported / not supported / inconclusive, with tolerance τ0 = 0.02.
  - It also re-runs the tests at every (N, c) to look for verdicts that flip.

---

## 2. A small worked example of the split

> **This example is invented to illustrate the arithmetic. None of these numbers come from the experiment.** (Real prompts in our task have two correct sequences. We use three here to show more cases.)

### Step 1: one prompt, N = 64, c = 1

Prompt A has three correct solutions s1, s2, s3.

| | s1 | s2 | s3 | J (sum) |
|---|---|---|---|---|
| base q0 | 0.30 | 0.05 | 0.004 | 0.354 |
| trained q1 | 0.55 | 0.10 | 0.20 | 0.85 |

**The reachability line.** A solution counts as reachable at budget N if the base gives it at least c/N probability. Roughly, the base would be expected to produce it about c times or more in N tries. Here the line is c/N = 1/64 = **0.015625**.

- s1: 0.30 ≥ 0.015625, so **reachable**
- s2: 0.05 ≥ 0.015625, so **reachable**
- s3: 0.004 < 0.015625, so **unreachable** (the base would produce it about 64 × 0.004 ≈ 0.26 times in 64 tries)

Reachability is always judged with the **base** q0, never the trained q1.

**The two parts.**
- Δrel = change on reachable solutions = (0.55 − 0.30) + (0.10 − 0.05) = 0.25 + 0.05 = **0.30**
- Δcap = change on unreachable solutions = 0.20 − 0.004 = **0.196**
- Total = J1 − J0 = 0.85 − 0.354 = **0.496**

**Check:** 0.30 + 0.196 = 0.496. This always holds, because every correct solution is either reachable or unreachable. That is Proposition 1 (`docs/proofs.md`). The report checks it on every record.

### Step 2: a second prompt, and averaging

Prompt B has two correct solutions.

| | s1 | s2 | J |
|---|---|---|---|
| base q0 | 0.40 | 0.008 | 0.408 |
| trained q1 | 0.70 | 0.002 | 0.702 |

The line is again 0.015625. s1 is reachable and s2 is not.
- Δrel = 0.70 − 0.40 = **0.30**
- Δcap = 0.002 − 0.008 = **−0.006**. Training *removed* mass from the rare solution. **Both parts can be negative.**
- Total = 0.702 − 0.408 = **0.294** = 0.30 + (−0.006). ✓

**Group value = mean over prompts:**
- Δrel = (0.30 + 0.30) / 2 = **0.30**
- Δcap = (0.196 + (−0.006)) / 2 = **0.095**
- Total = (0.496 + 0.294) / 2 = **0.395**. Check against mean J: (0.85 + 0.702)/2 − (0.354 + 0.408)/2 = 0.776 − 0.381 = 0.395 ✓

### Step 3: the split depends on the budget N

The same two policies give different splits at different N, because the line c/N moves.

| N (c = 1) | line c/N | Prompt A: Δrel / Δcap | Prompt B: Δrel / Δcap |
|---|---|---|---|
| 16 | 0.0625 | 0.25 / 0.246 (s2 at 0.05 is now *below* the line) | 0.30 / −0.006 |
| 64 | 0.015625 | 0.30 / 0.196 | 0.30 / −0.006 |
| 256 | 0.00390625 | 0.496 / 0 (s3 at 0.004 is now *above* the line) | 0.294 / 0 (s2 at 0.008 is above) |

The totals never change. Only the labels on the mass do.
- "Capability" here means "beyond what the base finds **in N tries**".
- With a big enough sampling budget, almost everything is reachable.

This is the key idea behind our H1 result (section 3).

### Step 4: E_N vs E_1, and why pass@N saturates

**pass@N.** With success probability J per try, the chance of at least one success in N independent tries is pass@N = 1 − (1 − J)^N.

**Tempering** reshapes the base without new learning. Raise the base's probabilities to a power β and renormalise:
- per token (*per-step* family), or
- over whole sequences (*sequence* family).

This sharpens or flattens the base. Tempering is the "free" improvement that needs no training.

**The two excess measures.**
- **E_N** = pass@N(trained) − max over β of pass@N(tempered base).
- **E_1** = J(trained) − max over β of J(tempered base), with one shared β for the group (decision H39).

Positive means the trained policy beats every re-tempered version of the base.

**Invented example.** An easy-like prompt with base J0 = 0.275 and trained J1 = 0.998. Suppose the best tempered base reaches J = 0.45.
- pass@64 of the base = 1 − 0.725^64 = 1 − 1.15×10⁻⁹ ≈ **0.999999999**. It is already effectively 1.
- pass@64 of the best tempered base (J = 0.45) = 1 − 0.55^64 ≈ 1 − 2.4×10⁻¹⁷ ≈ **1**.
- pass@64 of the trained policy (J = 0.998) ≈ **1**.
- So **E_64 ≈ 0**, even though J went from 0.275 to 0.998.
- **E_1** = 0.998 − 0.45 = **0.548**.

E_N is blind to reliability gains on prompts the base already solves within N tries. That is **saturation**. It is why the label's "beyond temperature" axis uses E_1 (decision H39). The report flags saturated records: easy prompts at N = 64.

**Compare an unsaturated, hard-like prompt** with J = 0.0065. Then pass@64 = 1 − 0.9935^64 ≈ **0.341**, well below 1. That is close to the untrained base's real hard-prompt pass@64 in the report (about 0.34 for `none`, k = 0). There, pass@N still has room to move.

---

## 3. How to read the figures (`docs/experiment_main/figures/`)

All figures use the primary budget **N = 64, c = 1**, and the seed mean over 10 seeds.

**Group names:**
- `hard_heldout`: hard prompts never trained on, which tests transfer;
- `hard_train`: hard prompts the arm trained on;
- `easy_train` and `easy_heldout`: the same split for easy prompts;
- `all`: every prompt.

### `main_grid.png`: heatmap of Δcap on hard held-out prompts at step 1000

- **Rows:** arms. **Columns:** shared structure k = 0, 2, 4.
- **Colour:** red is negative, blue is positive, white is about 0. The colour scale only reaches ±0.01.
- **What it shows:**
  - At k = 0 and k = 2, every trained arm is slightly **negative** on held-out hard prompts. For example, `rloo` is −0.00584 at k = 0. Training took a little mass *away* from unseen hard solutions.
  - At k = 4, most arms are slightly positive. The largest is `sft` at +0.0107.
- **Watch the scale.** All of these values are below the pre-registered "near zero" tolerance τ0 = 0.02. The bright blue is a small effect drawn on a small scale.

### `training_<arm>.png`: one per arm, Δrel (blue) and Δcap (orange) against update step

- **Columns:** k = 0, 2, 4.
- **Rows:** groups `all`, `hard_train`, `hard_heldout`.
- **Lines:** seed means. **Shaded bands:** min..max over the 10 seeds, not a confidence interval.
- **Note:** each panel has its own y-axis, so the `hard_heldout` row is zoomed in by roughly 100×.
- **Δrel on hard prompts is always exactly 0.** No hard solution is reachable under the base, so all hard-prompt change is Δcap by construction.
- **`training_01_rloo.png`:**
  - On `all`, Δrel jumps within the first ~50 updates (easy prompts are learned fast). Δcap rises more slowly but keeps growing.
  - On `hard_train`, Δcap climbs to about 0.6 (k = 0) up to about 0.95 (k = 4).
  - On `hard_heldout`, Δcap drifts slightly **negative** at k = 0 and k = 2. At k = 4 it bumps up early, then returns to about 0 by step 1000.
- **`training_04_sft.png`:**
  - Δcap on `hard_train` reaches about 0.99 within a few hundred updates.
  - Δrel on `all` goes **negative**. SFT on hard demonstrations costs easy-prompt reliability (the report gives `easy_train` Δrel −0.2443 at k = 0).
  - Held-out hard Δcap is tiny, with a wide seed band at k = 4.
- **`training_00_none.png`** must be flat at 0. It is: this is the zero-gain control.

### `frontier.png`: trained pass@N against the best tempered-base pass@N

- **One panel per group.** Each marker is one arm × k:
  - **colour** = arm;
  - **shape** = k (circle 0, square 2, triangle 4).
- **x-axis:** the best pass@64 any re-tempered base reaches. **y-axis:** the trained policy's pass@64.
- **Reading position:**
  - **Above the dashed diagonal:** the trained policy beats every tempered base (E_N > 0).
  - **Below:** some tempered base does better.
- **What it shows:**
  - **`hard_train`:** all trained arms sit far above the diagonal. Training found solutions tempering cannot.
  - **`hard_heldout`:** every trained arm sits **below** the diagonal, and below the untrained `none` markers. Trained pass@64 falls from about 0.34 to roughly 0.03–0.18. RL and SFT on train prompts made the policy *worse* at stumbling onto unseen hard solutions.
  - **`easy_*`:** everything sits at x ≈ 1. The tempered base already has pass@64 ≈ 1, so E_64 cannot detect any gain (the saturation of section 2, step 4).
  - **`all`:** a mix, near the diagonal.

---

## 4. How to run the code

```bash
cd capsplit
python3.12 -m venv .venv && source .venv/bin/activate   # Python 3.10–3.12; torch may not install on 3.13+
pip install -e ".[dev,deep,viz]"     # numpy + pytest + torch + matplotlib

# list registered plug-ins (tasks, bases, policies, trainers, backends)
python -m capsplit.harness list

# tests (add -m "not slow" to skip the slow ones)
pytest

# a tiny smoke run (dummy task, two do-nothing arms)
python -m capsplit.harness run configs/dummy.json --out results
python -m capsplit.harness report results/dummy

# the main experiment: 240 runs (8 arms x 3 k x 10 seeds); takes hours, use several workers
python -m capsplit.harness run configs/main.json --out results --workers 6
python -m capsplit.harness report   results/main
python -m capsplit.harness verdicts results/main
```

**Where results land:**
- `results/<name>/`, which git ignores:
  - `index.json`: every run;
  - `bases/`: cached base models;
  - `<cell>/seed-<s>.metrics.json`: one metrics file per run;
  - `<cell>/manifest.json`: the commit and config used.
- A run is complete once its metrics file exists. Re-running skips completed runs unless you pass `--force` (H23).
- `results/<name>/report/` holds:
  - `report.md`, `summary.json`;
  - `tables/*.csv` (main, labels, trajectory, frontier, sensitivity, sensitivity_flips, checks);
  - `figures/*.png`;
  - `verdicts.md` and `verdicts.json`, written by the `verdicts` command.
- `docs/experiment_main/` is the committed copy of `results/main/report/` for the main run.

**To add a new task, trainer or backend:** subclass the abstract class in `interfaces.py`, decorate it with the registry, and name it in a JSON config. Unknown config keys or trainer parameters raise an error, so a typo fails loudly (H14, H24).

---

## 5. Glossary

- **Arm.** One training method with fixed settings (for example `hints_h3`). All arms start from the same base for a given (k, seed).
- **Base (π0, q0).** The pre-trained starting policy. All reachability is judged against it.
- **Bootstrap interval.** The 95% percentile interval of the seed mean, from resampling the 10 seeds 10,000 times. Results are exact, so it describes seed-to-seed spread only, not measurement noise.
- **c (multiplier).** Scales the reachability line c/N. We use c ∈ {0.5, 1, 2}; the primary is c = 1.
- **Capability gain (Δcap).** The change in probability on correct solutions with q0(s) < c/N. It can be negative.
- **D2 label.** An older labelling rule (Δcap ≥ c/N), reported alongside the H12 label.
- **E_N / E_1.** The excess of trained pass@N (or J) over the best tempered base. Positive means the gain is "beyond temperature".
- **Held-out.** Prompts no arm ever trains on. They measure transfer.
- **Labels (H12 / H39).**
  - Each record gets one label from two yes/no questions:
    - *beyond reach*: did some unreachable solution cross the line?
    - *beyond temperature*: is E_1 > 0?
  - The four outcomes are `capability` (both yes), `compounded_sharpening` (reach only), `reweighting` (temperature only) and `pure_sharpening` (neither).
- **N (budget).** The number of tries the reachability line refers to. We use N ∈ {16, 64}; the primary is 64.
- **pass@N.** 1 − (1 − J)^N, the chance of at least one correct answer in N tries.
- **Pre-registration.** Hypotheses, tests and thresholds frozen before the main run (`docs/preregistration.md`, tag `gate1-frozen`).
- **Proposition 1.** Δrel + Δcap = J1 − J0 exactly, on every prompt and every group.
- **Reachable.** q0(s) ≥ c/N. Roughly, the base would produce s about c or more times in N tries.
- **Reliability gain (Δrel).** The change in probability on reachable correct solutions.
- **Saturation.** The base's pass@N is already ≈ 1, so E_N cannot register any further improvement.
- **Shared structure k.** How many leading tokens a hard solution shares with the easy rule (0, 2 or 4).
- **τ0 = 0.02.** The pre-registered "near zero" tolerance, in probability mass.
- **Tempering.** Reshaping the base by raising its probabilities to a power β (per token or per sequence) and renormalising. It is a free baseline that needs no training.
- **Trap.** A wrong sequence that the base makes its most likely output, so that tempering alone cannot solve the prompt.
