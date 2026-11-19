# LM experiment: design note (C0)

This note fixes the interfaces before C1–C6 are written in parallel. Each code module has one owner (see the table at the end).

## RL framing

| RL object | Here |
|---|---|
| Episode | One generated arithmetic problem. The policy writes a short completion, at most `model.max_new_tokens` tokens. |
| State | The chat-formatted prompt plus the tokens generated so far |
| Action | The next token |
| Reward | 1 if the checker accepts the completion, else 0 |
| Policy | π0 = the base model (LoRA adapter disabled); π1 = the base model with a trained LoRA adapter |
| Update rules | `lm_rl`: group-normalised policy gradient (GRPO-style); `lm_sft`: maximum likelihood on correct answers; `lm_none` |

## Package layout (`src/capsplit/harness/lm/`)

| Module | Job | Owner |
|---|---|---|
| `config.py` | Config schema; unknown keys raise | C0 (done) |
| `device.py` | Device choice, memory guards, niceness | C0 (done) |
| `storage.py` | Atomic JSON, resumable JSONL, manifests, progress with ETA | C0 (done) |
| `model.py` | Fetch and load the pinned checkpoint; LoRA; chat encoding; `sample_batch`; `stop_token_ids` | C0 (done) |
| `setup.py` | doctor, fetch, speedtest | C0 (done) |
| `policy.py` | `LocalLMPolicy` (harness `Policy`) | C1 |
| `tasks.py` | Problem generators (registry), `LMTask` (harness `Task`), checker | C2 |
| `trainers.py` | `lm_none`, `lm_rl`, `lm_sft`, `lm_sft_then_rl` (harness `Trainer`) with training logs | C3 |
| `measure.py` | Pure measurement from per-problem success counts: classification, split, temperature baseline, bootstrap, null check, budget curve | C4 |
| `stages.py`, `cli.py` | data, base, train, measure, report stages and commands | C5 |
| `report.py`, `verdicts.py` | Generated report, figures and L1–L3 verdicts | C6 |

Tests go in `tests/lm/`. Tests that need the real model weights or tokenizer use the `lm_model` marker and skip, with a message, when the files are not in the local Hugging Face cache.

## Data contracts

**Problems** live in `<out>/<name>/data/problems.jsonl`, one row per problem:

```json
{"id": 0, "generator": "arithmetic", "question": "347 * 29", "answer": "10063",
 "split": "train", "difficulty": "d3_mul"}
```

- `id`s are 0..P-1.
- `LMTask.prompts()` returns `Prompt(id, tokens=(prompt_text,), tags={"split", "generator", "difficulty"})`, where `prompt_text = task.prompt_template.format(question=...)`.
- For LM tasks, `Task.verify(pid, actions)` takes the decoded completion **text** (a `str`).
- The checker extracts the first integer in the text (an optional minus sign, digits, commas allowed). The completion is correct iff that integer equals the answer.
- `LMTask.parse(pid, text) -> (valid: bool, correct: bool)`, where valid means an integer was found.

**Samples** are stored per policy tag and set:

- **Path:** `<out>/<name>/samples/<tag>/<set>_T<temperature>.jsonl`, one row per sample: `{"problem": id, "index": i, "text": str, "valid": bool, "correct": bool, "tokens": int}`.
- **Tags:**
  - `base`
  - `<arm>/seed<s>/step<k>` (a trained policy at a checkpoint)
- **Sets:**
  - `A`: base only, classification
  - `B`: base only, measurement
  - `C`: base only, temperature sweep, one file per temperature
  - `null`: base only, a fresh reference-temperature set used as a fake "trained" policy for the null check
  - `eval`: trained policies
- **Resuming:** sampling skips any (problem, index) already present.
- **Seeds:** the generator for a chunk is seeded from `numpy.random.SeedSequence([sampling.seed, crc32(tag), crc32(set), round(T*1000), problem])`, so every chunk is reproducible on its own and resumable in any order.

**Counts** (`measure.py` input) are a per-problem success count plus a sample count, one array per (tag, set, T). `measure.py` reads only counts derived from the sample files, never the model.

## Seed streams (addendum B1)

- **Data seed:** `task.seed`.
- **Base:** no seed. It is a pretrained checkpoint, recorded by revision.
- **Training sampling and LoRA init:** the run seed from `seeds`. Optional `train_seed` and `base_seed` keys are not needed here, because there is a single base.
- **Measurement sampling:** `sampling.seed`.

All three are recorded in every manifest.

## Measurement (C4)

- **Classification.**
  - Set A gives each problem a Clopper–Pearson interval (level 1−α) on its success rate p.
  - **reachable** if the lower bound ≥ 1/N; **unreachable** if the upper bound < 1/N; **ambiguous** otherwise.
  - Set A is never used for measurement.
- **Contributions.** For each problem, gain = p̂1 − p̂0, where p̂0 comes from set B and p̂1 from `eval`.
  - Each class's contribution is the sum of its problems' gains divided by the **total** number of problems.
  - The three contributions sum exactly to the mean gain.
- **Temperature baseline.**
  - Split the problems in half at random. On half 1, choose T* maximising the base's mean success (and mean pass@N), using set C. On half 2, E = trained − base(T*). Swap the halves and average. Repeat over `measure.n_splits` random splits.
  - Set C is not used for the main estimate.
- **Bootstrap.**
  - Resample problems with replacement.
  - In each resample, redo the classification from a binomial redraw of that problem's set-A count (rate = observed set-A rate), so classification uncertainty enters the interval.
  - Percentile intervals; `measure.n_boot` resamples.
- **Null check.**
  - The same pipeline with π1 := the `null` set.
  - If the total gain's interval excludes 0, the report prints a prominent warning.
- **Budget curve (addendum A).**
  - Contributions as a function of N on a log grid, from the same counts.
  - The largest resolvable N is floor(1/u0), where u0 = 1 − (α/2)^(1/n_A) is the upper bound for zero successes. The curve is not reported beyond it.
  - N_train per arm comes from the training log.

## Training logs (C3; addendum B2)

Each trainer writes `training.json` for every (arm, seed) with:

- **Per training problem:**
  - `first_success_update`: the first update in which a sampled completion was correct, or null;
  - `correct_by_checkpoint`;
  - `episodes_sampled` and `tokens_sampled`;
  - the running success rate of training samples at each checkpoint.
- **Totals:** `episodes_sampled` and `tokens_sampled`.

LoRA adapters are saved at every checkpoint as `<out>/<name>/train/<arm>/seed<s>/step<k>/adapter.safetensors`.
