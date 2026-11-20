# RQ4 pass@k misdiagnosis: design

> **Status: approved by Bilal on 2026-11-20, with the recommended defaults.** Nothing here is built yet. Decided with Bilal: use both data sources, with RQ3's runs as the headline; approach C (harness runs plus one analysis script); and a new shared tabular policy for the grid sweep. The items marked **(team decision)** were settled as written:
> - primary group `heldout`;
> - k_large = 10, with a sensitivity table;
> - the 0.95 agreement line;
> - Q3 as stated in §7.
>
> Items marked **(builder's call)** may be settled during implementation and logged in `docs/DECISIONS.md`.

## 1. Intent

RQ4 asks how often the common way of reading pass@k curves gets a run wrong (blueprint §2.6, §9, D10). The rule reads the trained model's pass@k curve against the base's:
- below it at large k means **sharpening**;
- never below it, and above it somewhere, means **capability**;
- anything else is **undetermined**.

RQ4 compares that call with the exact 2×2 label and reports a misdiagnosis rate.

- **Prediction:** the rule disagrees with the label in a sizable share of runs, most often in the compounded-sharpening and reweighting cells.
- **Contradicted if:** primary agreement is above 95%.

**Why it matters.** Proposition 3 proves that pass@k cannot determine the split in principle. RQ4 measures how often that matters in practice. It is also the project's result closest to real LLM evaluation, where papers on Qwen-style post-training reason from pass@k curves. A teammate is post-training Qwen now, and RQ4's rate tells them how much to trust that kind of reading.

**Why the pilot failed.** The pilot's rule said "capability" in 39 of 39 runs, because every run improved every prompt, so the trained curve could never fall below the base. RQ4 needs runs in which some prompts get worse, and both data sources below are chosen to provide them.

## 2. What exists

| Piece | Where | State |
|---|---|---|
| The rule (`passk_rule`), curves (`passk_curve`) and scoring (`score_rule`, with primary and secondary agreement and per-cell rates) | `src/capsplit/diagnostics.py` | Done and tested; pilot only |
| Pilot RQ4 script, k grid (`k_grid`: K_BASE up to the largest valid N) and k_large = 10 | `experiments/pilot_rq4.py` | Ran; 39/39 "capability" |
| RQ3 Phase 2 experiment: 8 arms × k ∈ {0, 2, 4} × 10 seeds, 72 prompts, policies saved at 9 checkpoints, per-group labels at N ∈ {16, 64}, c ∈ {0.5, 1, 2} | `configs/main.json`, harness | Run once, on a teammate's machine. Results are not on this one. Rerun takes about 6.5 hours on 6 workers |
| Harness grid task, tabular grid policy (separate logits per prompt), `tabular_reinforce` and `rloo` trainers with `only_tags`, checkpoint saving | `src/capsplit/harness/` | Done |
| SRR/NDR definitions | `docs/notes/invisible-leash-scores.md` | **Unverified draft.** The team requires a person to check it against the paper first |

**Groups in the RQ3 runs.** Each run is measured on 9 overlapping prompt groups: `all`, `easy`, `hard`, `train`, `heldout`, `easy_train`, `easy_heldout`, `hard_train` and `hard_heldout`. Each group's record, per (checkpoint, N, c), has:
- a label (the E₁ rule, H39) and `extra.label_en` (the E_N rule);
- **the per-prompt success rates, `extra.j0` and `extra.j1`**, as lists in the group's prompt order (`harness/backends/exact.py`).

So the rule's inputs are already in the metrics files. No policy needs reloading. (Corrected 2026-11-20 while planning; the first draft said they had to be recomputed.)

## 3. Goals, non-goals, anti-goals

**Goals**
- G1. **Phase 2 (headline):** apply the frozen rule to RQ3's runs and score it against the harness's existing labels, reading per-prompt J from the metrics files.
- G2. **Phase 1:** a new grid sweep with a shared tabular policy, so that training on some prompts can make others worse. Score the rule the same way.
- G3. A full rerun of RQ3 here, which also checks it reproduces against the committed `docs/experiment_main/`. The RQ3 PR listed this check as never done.
- G4. SRR and NDR computed on the same runs, **held out of the report** until a person has verified the note.
- G5. A pre-registration frozen before any rule output is looked at, and a short RQ4 report.

**Non-goals**
- Changing the rule (D10), its scoring, or k_large after any rule output has been seen.
- Changing RQ3's frozen harness code. The exception is one line in the registry's module list, to register the new policy (§6.2).
- Running the rule on the Qwen pair. The frozen rule can be handed to the teammate afterwards, but the report doesn't depend on it.

**Anti-goals**
- **Tuning the run mix to the rule's output.** The Phase 1 design may be checked against its *labels* before the freeze (to make sure all four cells occur), but never against rule outputs.
- **Quoting a rate that the run mix produces by construction.** That was the pilot's failure. Report per-cell rates beside the overall rate, and say how the mix was chosen.
- **Reporting SRR/NDR from the unverified note.**

## 4. Approaches considered

| Approach | Trade-off | Decision |
|---|---|---|
| A. One script in `experiments/` that also trains the grid sweep by hand | No harness changes, but the grid runs are hand-rolled, with no configs or manifests | Rejected |
| B. RQ4 inside the harness (backend, report, verdicts) | Most reusable, but changes RQ3's frozen code and forces re-validating RQ3 | Rejected |
| **C. Harness configs for the grid sweep, one analysis script for both sources** | Uses the harness's runner, manifests and resumability; no change to RQ3's code paths | **Chosen** |

Grid-policy options:
- **Separate logits per prompt:** untrained prompts can't change, so this repeats the pilot. Rejected.
- **"Nothing degrades" control only:** rejected as the main design.
- **Shared tabular policy:** chosen.

## 5. Data and units

### 5.1 Phase 2: RQ3's runs (headline)

- **Source:** a fresh run of `configs/main.json` on this machine, at the current `main` commit, with a clean tree.
  - It uses `python -m capsplit.harness run configs/main.json --out results --workers 8` (this machine has 10 cores).
  - The `report` output is then compared with the committed `docs/experiment_main/` (`summary.json`, `tables/main.csv`, `verdicts.json`), and that comparison is RQ4's G3 deliverable.
  - If it doesn't match, RQ4 stops and the mismatch is reported first.
- **Unit:** one (run, checkpoint, prompt group).
  - The rule gets the group's per-prompt J₀ and J₁ from the record's `extra.j0` and `extra.j1` at N = 64, c = 1. J doesn't depend on N or c.
  - The label is the same record's `label`.
- **Primary group: `heldout`, all 24 held-out prompts (team decision).** This is what a practitioner reports: a benchmark the model wasn't trained on. Held-out prompts are also where RQ3 saw degradation, and they mix easy and hard prompts, so the curves can cross.
  - **Secondary groups:** `all`, and the four disjoint `difficulty_split` groups, each reported separately.
- **Primary checkpoint:** update 1,000.
  - **Secondary:** all checkpoints from 25 to 750, read as early-stopped runs.
  - Update 0 and the `none` arm are excluded, since π₁ = π₀ there and there is nothing to diagnose.
- **Primary dataset:** 7 arms × 3 values of k × 10 seeds = 210 units.

### 5.2 Phase 1: shared-policy grid sweep

- **Policy (new plug-in, `tabular_grid_shared`):** one logit vector per grid cell, shared by all prompts. All prompts must have the same walls, so their cell numbering is identical; the policy refuses a task where it isn't.
- **Task:** the existing `grid` task with custom `variants`: same walls, different start or goal cells, tagged `difficulty` and `split`.
  - About 8 prompts, so the curves can cross (Proposition 3 needs several prompts).
  - At least two prompts per split, and goals placed so that their routes pull the shared cells in different directions.
  - Exact layout is the builder's call, frozen in the config.
- **Base:** a base recipe for the shared policy (new, or `grid_routes` extended, **builder's call**), set so that:
  - every prompt has at least one correct route below the reach line c/N at N = 64;
  - at least one prompt has a reachable correct route;
  - the most likely behaviour is wrong on at least one prompt, so sharpening can hurt.
- **Arms (all `tabular_reinforce` or `rloo`, lr and batch from `configs/grid_example.json` unless a pre-freeze label check says otherwise):**
  1. Train on all train prompts.
  2. Train on a subset of train prompts (`only_tags`), so the shared cells drift away from the rest.
  3. RLOO on all train prompts.
- **Checkpoints:** early (25, 50, 100) through final (budget fixed before the freeze), so early-stopped runs are included.
- **Seeds:** 0–19 (the blueprint's 20-seed rule).
- **Measurement:** N ∈ {16, 64}, c ∈ {0.5, 1, 2}; primary N = 64, c = 1. Same groups as RQ3.
- **Pre-freeze label check (allowed):** run the sweep at reduced seeds and look **only at the label distribution** (never at rule outputs). Adjust the base or arms until each of the four cells appears in at least 10% of units. Log every adjustment in DECISIONS. Then freeze, and run the full sweep from scratch.

### 5.3 The rule's settings

- **k grid:** `pilot_rq4.k_grid(n_actions, horizon)`, which is K_BASE up to the largest valid N (`n_actions ** horizon // 100`). That is 77 on the sequence task (V = 6, L = 5) and 655 on a grid with H = 8.
- **k_large = 10:** the value committed before the pilot, re-frozen here (team decision). The report adds a sensitivity table at k_large ∈ {5, 20, 50} (k ≤ k_max), marked as sensitivity only.
- **Tie tolerance:** `tol = 1e-9`, as in `passk_rule`.

## 6. Code changes

### 6.1 Files

| File | Change |
|---|---|
| `src/capsplit/harness/policies/tabular_grid_shared.py` (new) | `TabularGridSharedPolicy`, registered as `tabular_grid_shared`. It has the same interface as `TabularGridPolicy`, but one logits tensor of shape (n_states, 4), used by every prompt. The same module holds the shared-policy base recipe `grid_shared_routes` (§5.2), so the registry change stays at one line. |
| `src/capsplit/harness/registry.py` | One line: add the new module to `PLUGIN_MODULES`. No other harness change. |
| `configs/rq4_grid.json` (new) | The Phase 1 sweep, frozen. |
| `src/capsplit/diagnostics.py` | Unchanged. If needed, add `srr_ndr(q0, q1, eps)` per prompt with explicit options for the note's open points (aggregation, zero denominators, boundary). |
| `experiments/rq4_passk.py` (new) | Subcommands: `repro` (compare a rerun RQ3 report with `docs/experiment_main/`), `labels` (label distribution only, for the pre-freeze check; never calls the rule), `score` (rule, agreement, per-cell rates, bootstrap intervals, verdicts, tables and figures). Units are read straight from the harness metrics files. |
| `tests/test_tabular_grid_shared.py`, `tests/test_rq4.py` (new) | See §6.3. |
| `docs/rq4-preregistration.md`, `docs/rq4-passk.md` (new); `docs/DECISIONS.md`, `README.md` | Pre-registration, report, decision log, status. |

### 6.2 Why the registry line is acceptable

The harness imports a fixed list of plug-in modules (H6). Registering in the analysis script instead would break the harness's parallel workers, which re-import modules. One added module name doesn't change any existing behaviour, and the RQ3 tests must still pass unchanged.

### 6.3 Tests that matter

- **Shared-policy interference.**
  - Two prompts with the same walls and different goals: an exact gradient step on one changes the other's J.
  - The same step under `tabular_grid` leaves the other's J unchanged.
  - The policy refuses prompts with different walls.
- **Shared-policy exactness.** `trajectory_probs` matches the engine on each prompt's table.
- **Units are read right.** On a quick harness run in a temp directory, every unit's per-prompt J₀ and J₁ average to the record's `j0` and `j1` to 1e-9, and units at update 0 and in the `none` arm are excluded.
- **Rule degeneracy is visible.**
  - A synthetic unit where every prompt improves gives "capability".
  - One where a held-out prompt drops below the base at k ≥ 10 gives "sharpening".
- **`repro`** reports a match on identical inputs and a clear mismatch on a perturbed copy.
- **SRR/NDR** on hand-built q₀ and q₁ vectors, including the zero-denominator cases.

## 7. Pre-registered predictions and decision rule

Frozen in `docs/rq4-preregistration.md` and tagged `rq4-frozen` **before any rule output is computed on full data**. Collecting J values, and the Phase 1 label check, may happen before the freeze. Calling `passk_rule` on real units may not, except in unit tests on synthetic data.

**Primary agreement:** D10's primary score, "capability or not". "Undetermined" counts as disagreement, and its rate is reported separately. Intervals come from RQ3's `harness.report.bootstrap_ci` applied to per-seed agreement rates. That is, each seed's rate is pooled over arms and k, so seeds are resampled jointly, as RQ3 did (10,000 resamples, RNG seed 0).

| ID | Claim | Supported if | Contradicted if |
|---|---|---|---|
| Q1 | Phase 2 headline: the rule misdiagnoses a sizable share (`heldout`, update 1,000) | CI upper bound of primary agreement < 0.95 | CI lower bound > 0.95 |
| Q2 | Same, Phase 1 grid sweep (`heldout`, final checkpoint) | As Q1 | As Q1 |
| Q3 | Disagreement concentrates in compounded sharpening and reweighting | In the pooled Phase 1 + 2 primary units, those two cells' disagreement rates each exceed both other cells' | Either of those two cells has the lowest disagreement rate of the four |

- Anything else is inconclusive.
- Cells with fewer than 10 units are reported but can't decide Q3, which is then inconclusive.
- The 0.95 comes from the blueprint (team decision to keep). Q3's operationalization is my proposal (team decision).

**Validity checks before any verdict:**
- G3 reproduction matches.
- Every unit's collected group mean equals its record's j0 and j1 to 1e-9.
- Every Phase 1 unit has a label from the harness at the primary (N, c).
- The secondary agreement and the undetermined rate are reported for every cell.

## 8. SRR and NDR

These are computed per prompt from exact solution probabilities, with ε = c/N, following the note's definitions. They are written to `results/rq4_report/srr_ndr_UNVERIFIED.csv` and kept out of `verdicts.*` and the report until all of the following hold:
1. A person checks the note against the paper and removes its status banner.
2. The team settles the note's four open points: measurement, aggregation, zero denominators and boundary.
3. A DECISIONS entry records those choices.

Until then the report says they are pending.

## 9. Constraints

- **Cost:**
  - The RQ3 rerun is about 5 hours on 8 workers, so it runs overnight in the background.
  - Reading units is just parsing the metrics JSON, so it takes seconds.
  - The grid sweep is small.
- **Disk:** RQ3 with `save_policies: checkpoints` writes 240 × 9 GRU state dicts, small but not tiny. `results/` is git-ignored.
- **Schedule:** the report freezes Dec 2. Order of work:
  1. Shared policy and tests; start the RQ3 rerun in parallel.
  2. Phase 1 label check, then the freeze.
  3. Full runs, then `score`.
  4. Report.
- **Branch:** `exp/rq4-passk` from `main`. RQ4 doesn't need the RQ2 branch.

## 10. Left to the builder

- The Phase 1 grid layout, base recipe, budget, lr and batch, within §5.2's requirements, settled by the label check and logged.
- Where the shared-policy base lives.
- File and function names inside `experiments/rq4_passk.py` beyond the subcommands.
- Figure layout: a per-cell agreement bar chart for each phase, and a few example curve pairs where the rule and label disagree.
