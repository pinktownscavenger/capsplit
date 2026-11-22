# Explainer: capability vs reliability, for a new teammate

This is the plain-language companion to the project. Read it before `docs/preregistration.md`, `docs/gate2.md` and `docs/experiment_main/report.md`. It has six parts:
- what each piece of the code is;
- a small worked example of the split;
- how to read the figures;
- how to run the code;
- a glossary;
- the follow-up work: the budget curve, the seed study and the language-model (LM) experiment (section 6).

Sections 1–5 describe the frozen toy experiment and are unchanged.

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

---

## 6. Follow-up work: the budget curve, the seed study and the LM experiment

> **No result of the LM experiment is stated here: when this was written, none existed.** Every LM
> number below is either a design setting from the config or the pre-registration, or an invented
> example marked as such.

### 6.1 The LM experiment in RL terms

The toy experiment used a world small enough to compute every probability exactly. The LM
experiment asks the same question about a real (small) language model. Here sampling is the
only way to measure anything.

| RL object | In the LM experiment |
|---|---|
| Episode | One generated arithmetic problem, e.g. "Compute 347 * 29. Reply with only the final number." 180 problems: 2–4-digit operands, + or ×, 120 train and 60 held out |
| State | The chat-formatted prompt plus the tokens the model has written so far |
| Action | The next token, from the model's whole vocabulary; at most 12 tokens per answer |
| Reward | 1 if the first integer in the answer equals the true result (a program checks it), otherwise 0. Return = reward, so the expected return on a problem is its success probability p |
| Policy | π0 = the base model `Qwen/Qwen2.5-0.5B-Instruct` (about 0.5 billion parameters, pinned revision); π1 = the same model plus a trained LoRA adapter |
| Update rule | `rl`: GRPO-style policy gradient (8 problems × 8 sampled answers per update); `sft`: supervised learning on the correct answers of train problems the base cannot reach; `sft_then_rl`: half and half; `none`: control. 200 updates each, seeds 0, 1, 2 |

**LoRA in one paragraph.** Retraining all 0.5 billion weights would need far more memory than a
laptop has. LoRA ("low-rank adaptation") freezes every original weight. Next to a few chosen
weight matrices (here the attention layers' query and value projections), it adds a small
trainable correction: the product of two thin matrices, of rank r = 8. Only these thin matrices
are trained, which is well under 1% of the parameter count. The correction starts at zero, so a
fresh adapter leaves the model unchanged. Turning the adapter off gives back the base model
exactly. That is why one loaded model can serve as both π0 (adapter off) and π1 (adapter on).
The cost is that LoRA can change less than full fine-tuning, which the report lists as a limit.

The checker is strict about format: an answer that repeats the equation ("347 * 29 = 10063")
reads 347 and counts as wrong. The report therefore shows **format validity** (did the answer
start with an integer at all?) separately from correctness.

### 6.2 Reachable, unreachable and ambiguous, with samples

In the toy world we knew q0 exactly. Here we only see how often the base got a problem right in
a finite number of tries, so each problem's success rate p is only known up to an interval.

- **The line.** Budget N = 16, so the reachability line is 1/N = 1/16 = **0.0625**: the base
  would find the answer about once or more in 16 tries.
- **Set A.** 400 base samples per problem, used **only** to classify. Each problem gets a 95%
  Clopper–Pearson interval on p, an exact interval for a binomial rate.
- **The rule.**
  - **reachable** if the whole interval is at or above 0.0625 (lower bound ≥ 1/16);
  - **unreachable** if the whole interval is below 0.0625 (upper bound < 1/16);
  - **ambiguous** if the interval straddles the line: the data cannot tell.

> **Invented example (illustrative counts, not experiment data).** The intervals were computed
> with `capsplit.harness.lm.measure.clopper_pearson` (α = 0.05).

| Problem | Set A successes | Observed rate | 95% interval on p | Against 0.0625 | Class |
|---|---|---|---|---|---|
| P1 | 0 / 400 | 0 | [0, 0.0092] | entirely below | **unreachable** |
| P2 | 30 / 400 | 0.075 | [0.0512, 0.1053] | straddles | **ambiguous** |
| P3 | 6 / 400 | 0.015 | [0.0055, 0.0324] | entirely below | **unreachable** |
| P4 | 48 / 400 | 0.12 | [0.0898, 0.1559] | entirely above | **reachable** |

P2 shows why "ambiguous" exists. Its observed rate, 0.075, is above the line. But a true rate of
0.055, just below the line, would also produce 30 successes in 400 tries quite plausibly. Calling
it reachable would be a guess. With 400 samples and N = 16, 0–15 successes classify as
unreachable, 16–35 as ambiguous and 36 or more as reachable.

**Then the split.** Fresh samples, never set A, measure success: set B (64 base samples per
problem) gives p̂0, and 64 samples of each trained policy give p̂1. Keeping set A apart matters.
If the same samples both chose the class and measured the rate, problems that were unlucky in
set A would be filed as unreachable and would then "improve" by luck alone.
Continuing the invented example:

| Problem | Class | p̂0 (set B) | p̂1 (trained) | gain |
|---|---|---|---|---|
| P1 | unreachable | 0/64 | 3/64 | +0.046875 |
| P2 | ambiguous | 5/64 | 9/64 | +0.0625 |
| P3 | unreachable | 1/64 | 1/64 | 0 |
| P4 | reachable | 8/64 | 20/64 | +0.1875 |

Each class's contribution is the sum of its problems' gains divided by **all 4** problems:
- reachable: 0.1875 / 4 = **0.046875**
- ambiguous: 0.0625 / 4 = **0.015625**
- unreachable: (0.046875 + 0) / 4 = **0.01171875**
- total = mean gain = (3 + 4 + 0 + 12) / 64 / 4 = **0.07421875**, and the three add up to it exactly.

The ambiguous part is reported on its own. The headline capability number (`delta_cap`) counts
only problems confidently unreachable: a capability claim needs evidence that the base could not
reach the problem. Even P1 does not count as "beyond reach" here. After training it has 3/64,
whose interval [0.0098, 0.1309] is not wholly above 0.0625. To count, the trained policy must
be confidently above the line too.

Intervals in the report come from a bootstrap that resamples problems and also redraws each
problem's set-A count and reclassifies it, so the chance of misclassification is inside the
interval.

### 6.3 The budget curve, and why the training budget N_train matters

"Unreachable" always means "unreachable **in N tries**". Section 2, step 3 showed the split
moving with N. The **budget curve** (`python -m capsplit.harness budget-curve`) draws the
reliability and capability parts against N on a log axis, from 1 up to very large N. As N grows,
more solutions cross the line and gain moves from capability to reliability. The total never
changes.

The natural question is which N matters. During training an RL arm draws many samples of each
training prompt. **N_train** is that number: episodes sampled per training prompt, read from the
training logs that trainers now write. If a solution is unreachable at N = 64 but reachable at
N_train, RL did not need to create anything: it sampled the solution by chance during training
and then reinforced it.

**The toy finding** (`docs/budget_curve_main/`). The frozen main run predates the logs, so its
N_train is computed from the config and marked "from_config (not logged)". For `rloo`, that is
1,000 updates × G = 16 = **16,000 samples per training prompt**. Hard solutions were built to be
unreachable at N = 64: their base probability is around 0.2/64 ≈ 0.003. But 1/16,000 is about
0.00006, far below 0.003. In 16,000 tries the base would hit such a solution about
16,000 × 0.003 ≈ 48 times (a rough figure, for intuition). The budget curve agrees. For `rloo`,
`grpo` and `rl_entropy` at every k, the hard-train capability part at N_train is +0.0000, and the
crossover N\* (where reliability holds half the gain) is 512 on hard-train prompts. So the toy's
"capability at N = 64" by RL was all reachable at the budget RL actually spent.

The LM experiment was sized with this in mind (decision L16). The 400 set-A samples set the
largest N the experiment can resolve. A problem with 0 successes in 400 still has an upper bound
of 0.0092, so nothing can be called unreachable beyond N = floor(1/0.0092) = **108**. The
planned `rl` arm samples about 200 updates × 64 answers / 120 problems ≈ **107** per training
problem, so the curve can just reach N_train. The report draws the curve only up to 108 and
marks each arm's logged N_train.

Pre-registered hypothesis L1 tests the same idea problem by problem: RL gains on train problems
classified unreachable should occur only on problems where training itself sampled a correct
answer at least once (`first_success_update` in the training log is not empty).

### 6.4 The seed study: luck or the base?

The toy runs showed RL's hard-train capability gain varying a lot between seeds. In the old setup
one seed controlled two things at once: how the base was built and which samples training drew.
The seed study pulls them apart with two configs that each fix one stream (config keys
`base_seed` and `train_seed`):

- `configs/seedstudy_base.json`: **one fixed base**, 30 training seeds. Any spread is sampling
  luck during training.
- `configs/seedstudy_train.json`: **one fixed training seed**, 30 different bases. Spread comes
  from the base (and from how a different base samples).

The question: **is a prompt's outcome decided by the base, or by luck in which samples RL happened
to draw?** The analysis (`python -m capsplit.harness discovery analyze`) compares the variance of
each prompt's final capability gain across training seeds (V_train) with the variance across
bases (V_base). A ratio V_train / V_base of at most 1/3 reads "determined by the base"; at least
2/3 reads "depends on sampling luck"; anything between is "mixed".

The study also tests a pre-registered model of *discovery* (`docs/prediction_discovery.md`).
Until RL samples a correct answer on a prompt, every sample in that prompt's group gets reward 0,
so the group's advantage is exactly zero and that prompt gives no learning signal. The first
success is then a waiting time: with base success probability p and G samples per update, the
chance of still waiting after t updates is (1 − p)^(G·t). Predictions were computed from the bases
and committed before the runs.

### 6.5 How to read the LM report

`python -m capsplit.harness.lm report CONFIG` writes `results/<name>/report/report.md`. Every
number in it is read from files; nothing is typed by hand. Its sections, in order:

- **WARNINGS** (only if something is wrong). Read this first. It flags a failed null check, an
  empty base class, missing training logs or measurements, and uncommitted code. A failed null
  check means the measurement finds "gains" where none can exist, so treat every other number
  with suspicion.
- **Setup.** Model and revision actually loaded, device, git commit, library versions, and the
  sample sizes (set A, set B, trained, temperature sweep). Check that the model and revision are
  the pinned ones.
- **Data.** Problems per split and difficulty (for example `d3_mul`).
- **Base classification.** How many problems are reachable, unreachable and ambiguous, overall
  and per split and difficulty, plus the base success rate. If a class is empty, every statement
  about it is vacuous, and the report says so.
- **Split of the gain (final checkpoint).** One row per arm × seed × group (`all`, `train`,
  `heldout`): total gain, the three contributions with intervals, E_1 and E_N, the chosen
  temperature T\*, and the label. E_1 > 0 means the trained policy beats the base at its best
  temperature, where the temperature is chosen on one half of the problems and scored on the
  other. This is the LM version of "beyond temperature".
- **Format validity.** The share of answers that contain an integer, and the share correct. A
  "gain" that is really the model learning to answer with a bare number shows up here.
- **Training.** Per arm × seed: updates, episodes and tokens sampled, how many problems were ever
  answered correctly during training, N_train and wall-clock time. Arms are matched on updates,
  not samples, so SFT samples nothing and RL samples a lot. The first-success histogram shows
  when each training problem was first solved during training.
- **Null check.** A fresh base sample is measured as if it were a trained policy. Every row
  should read PASS, meaning its total interval contains 0.
- **Budget curve.** Contributions against N up to the resolvable limit, with N_train marked.
- **What this cannot show.** One small model, synthetic arithmetic, sampled rather than exact
  measurement, LoRA only, few seeds.

The pre-registered tests are in `report/verdicts.md` (`python -m capsplit.harness.lm verdicts
CONFIG`). Each of L1–L3 is "supported", "not supported" or "inconclusive", with the rule and the
numbers used. The rules and thresholds (τ = 0.02) are in `docs/preregistration_lm.md`.

**New terms:**
- **Ambiguous.** A problem whose set-A interval straddles 1/N. Reported separately, never counted
  as capability.
- **Clopper–Pearson interval.** An exact confidence interval for a success rate from k successes
  in m tries.
- **Sets A / B / C / null / eval.** Base samples for classification (A), base success (B), the
  temperature sweep (C), the null check (null); trained-policy samples (eval).
- **N_train.** Episodes sampled per training prompt during training, from the training log.
- **Largest resolvable N.** floor(1/u0), with u0 the upper bound after 0 successes in set A; 108
  for 400 samples.
- **LoRA adapter.** The small trainable add-on that turns π0 into π1.
- **base_seed / train_seed.** Config keys that fix one random stream (base building or training
  sampling) so that the run seeds vary only the other.
