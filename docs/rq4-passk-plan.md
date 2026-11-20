# RQ4 Pass@k Misdiagnosis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how often the frozen pass@k rule (D10) disagrees with the exact 2×2 label, on RQ3's rerun runs (headline) and on a new shared-policy grid sweep. Reproduce RQ3 along the way, and compute SRR/NDR without reporting them.

**Architecture:** One harness plug-in module, `tabular_grid_shared` (a shared-logit grid policy plus its base recipe), registered with one line in the registry. One analysis script, `experiments/rq4_passk.py`, reads units straight from harness metrics files (per-prompt `extra.j0` and `extra.j1`), applies `capsplit.diagnostics.passk_rule`, and scores it. RQ3's code is otherwise untouched, and its rerun is already running in the pinned worktree `../capsplit-rq3-rerun` at `46490ae`.

**Tech Stack:** Python ≥ 3.10, numpy, torch (the harness), matplotlib (`viz`), pytest. No new dependencies.

**Spec:** `docs/rq4-passk-design.md` (approved 2026-11-20 with defaults). Section numbers below refer to it.

## Global Constraints

- Do not change RQ3's harness behaviour. The only harness edit is one added entry in `PLUGIN_MODULES` (`src/capsplit/harness/registry.py`). All existing tests must pass unchanged.
- Primary measurement: N = 64, c = 1.0, unit group `heldout`, final checkpoint. Excluded units: step 0, and arms whose trainer is `none`.
- k grid: `pilot_rq4.k_grid(n_actions, horizon)`, which is 77 for RQ3 (V = 6, L = 5) and `4**H // 100` for the grid. k_large = 10; tie tolerance `1e-9`.
- Intervals: `capsplit.harness.report.bootstrap_ci` on per-seed primary agreement (10,000 resamples, seed 0).
- **No call to `passk_rule` on real (non-synthetic) units before the tag `rq4-frozen` is an ancestor of HEAD.** `score` enforces this.
- SRR/NDR outputs go only to files named `*_UNVERIFIED*` and never into `verdicts.*` or the report text.
- Branch `exp/rq4-passk`; Conventional Commits; no AI attribution. Decisions go in `docs/DECISIONS.md` under `## RQ4 pass@k`, with ids `RQ4-1`, `RQ4-2`, …
- Quick or test runs write to `tmp_path` or the scratchpad, never to `results/`.

## Review Focus

1. **`score` run before the freeze**, or on a checkout without the tag: refuse with a message naming `rq4-frozen`, not silently score. Pinned in Task 3.
2. **A group whose prompts all saturate** (J₀ and J₁ near 1, so the curves tie everywhere): the rule returns "undetermined", and the unit counts as a disagreement and in the undetermined rate, never dropped. Pinned in Task 3.
3. **A metrics directory with quick and full runs mixed, or missing `index.json`**: `load_units` reads only what `index.json` lists, and fails with the path named if it is missing. Pinned in Task 2.
4. **The shared policy given prompts with different walls**: refuse at construction or first use, because the cell numbering would silently disagree. Pinned in Task 1.
5. **A cell with fewer than 10 primary units in Q3**: it is reported, and Q3 becomes inconclusive rather than deciding on a handful of units. Pinned in Task 3.

---

### Task 0: Commit design update and plan

- [ ] **Step 1:** On `exp/rq4-passk`, commit the corrected design and this plan: `git add docs/rq4-passk-design.md docs/rq4-passk-plan.md && git commit -m "docs: update the RQ4 design and add its plan"`.

---

### Task 1: Shared tabular grid policy and its base

**Files:**
- Create: `src/capsplit/harness/policies/tabular_grid_shared.py`
- Modify: `src/capsplit/harness/registry.py` (append `"capsplit.harness.policies.tabular_grid_shared"` to `PLUGIN_MODULES`)
- Test: `tests/test_tabular_grid_shared.py`

**Interfaces:**
- Consumes: `TabularGridPolicy` (`harness/policies/tabular_grid.py`), `GridTask` (`harness/tasks/grid.py`), `BaseRecipe`, `BASES` and `POLICIES` registries.
- Produces:
  - `TabularGridSharedPolicy(n_prompts: int, n_states: int, init_scale=0.0, seed=0, logits=None)`, registered as `"tabular_grid_shared"`. It has one `torch.nn.Parameter` of shape `(n_states, 4)`, and its `parameters()`, `state_dict()` (key `"logits"`) and `load_state_dict` act on it. `sample`, `log_prob`, `state_logits` and `tempered` behave like `TabularGridPolicy` with every prompt reading the same rows. Subclassing `TabularGridPolicy` and exposing the shared tensor through the parent's per-prompt indexing is the expected route. `_check(task)` additionally raises `ValueError("... same walls ...")` if any two prompts' `task.world(pid).walls` differ.
  - Base recipe `"grid_shared_routes"`, with params `route_bonus` (default 1.0), `trap` (default 2.0), `route` (`"seeded"` or int), and `noise` (default 0.0). It sums each prompt's favoured-route bonus into the shared cells, adds `trap` to the start cell's no-op actions on prompts tagged `hard` (as `grid_routes` does), and returns `(TabularGridSharedPolicy, info)` with `info` keys `favoured_routes`, `favoured_q0` and `j0`, as `grid_routes` does.

- [ ] **Step 1: Write the failing tests**

```python
VARIANTS = [
    {"start": [0, 0], "goal": [4, 4], "walls": [[2, 2]], "difficulty": "easy", "split": "train"},
    {"start": [0, 4], "goal": [4, 0], "walls": [[2, 2]], "difficulty": "hard", "split": "heldout"},
]

def _task():
    return GridTask(n=5, H=8, variants=VARIANTS)

def _j(policy, task, pid):
    return table_success_prob(task.table(pid), policy.trajectory_probs(task, pid))

def test_training_one_prompt_moves_the_other():
    task = _task()
    shared = TabularGridSharedPolicy(2, task.n_states(0))
    before = _j(shared, task, 1)
    loss = -torch.exp(shared.log_prob(task, [0], [task.demonstrations(0)[0]]))
    loss.backward()
    with torch.no_grad():
        shared.logits.sub_(5.0 * shared.logits.grad)
    assert abs(_j(shared, task, 1) - before) > 1e-6

def test_per_prompt_policy_does_not_interfere():
    task = _task()
    separate = TabularGridPolicy(2, task.n_states(0))
    before = _j(separate, task, 1)
    loss = -torch.exp(separate.log_prob(task, [0], [task.demonstrations(0)[0]]))
    loss.backward()
    with torch.no_grad():
        separate.logits.sub_(5.0 * separate.logits.grad)
    assert _j(separate, task, 1) == pytest.approx(before, abs=1e-12)

def test_refuses_different_walls():
    task = GridTask(n=5, H=8, variants=[VARIANTS[0], {**VARIANTS[1], "walls": [[1, 1]]}])
    with pytest.raises(ValueError, match="same walls"):
        TabularGridSharedPolicy(2, task.n_states(0)).state_logits(task, 0)

def test_trajectory_probs_match_engine_and_sum_to_one():
    task = _task()
    policy = TabularGridSharedPolicy(2, task.n_states(0), init_scale=0.5, seed=1)
    for pid in (0, 1):
        probs = policy.trajectory_probs(task, pid)
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.allclose(probs, trajectory_probs(task.table(pid), policy.state_logits(task, pid)))

def test_shared_base_is_registered_and_reports_q0():
    policy, info = BASES.create("grid_shared_routes").build(_task(), seed=0)
    assert isinstance(policy, TabularGridSharedPolicy)
    assert len(info["favoured_q0"]) == 2 and all(0 < q < 1 for q in info["favoured_q0"])
    assert POLICIES.get("tabular_grid_shared") is TabularGridSharedPolicy
```

`POLICIES.get` stands for however the registry looks up a name. Use its real accessor, as the existing registry tests do.

- [ ] **Step 2:** Run `pytest tests/test_tabular_grid_shared.py -v`. Expected: FAIL (import error).
- [ ] **Step 3:** Implement the module and add the registry line.
- [ ] **Step 4:** Run `pytest tests/test_tabular_grid_shared.py tests/test_harness_interfaces.py tests/test_harness_grid.py -v`. Expected: PASS, with the existing harness tests unchanged.
- [ ] **Step 5:** Commit: `feat(harness): add a shared-logit tabular grid policy and base for RQ4`.

---

### Task 2: Reading units from harness results

**Files:**
- Create: `experiments/rq4_passk.py` (data model and loading only in this task)
- Test: `tests/test_rq4.py`

**Interfaces:**
- Consumes: the harness results layout (`<results_dir>/index.json` with `runs[]` entries holding `arm`, `trainer`, `seed`, `factors` and `files.metrics`; metrics JSON with `records[]` holding `step`, `group`, `n`, `c`, `label`, `extra.label_en`, `extra.j0` and `extra.j1`).
- Produces:
  - `@dataclass(frozen=True) class Unit: arm: str; trainer: str; factors: tuple; seed: int; step: int; group: str; j0: tuple; j1: tuple; label: str; label_en: str`, where `factors` is a sorted tuple of items, for hashing.
  - `load_units(results_dir: str | Path, n: int = 64, c: float = 1.0, groups: Sequence[str] | None = None) -> list[Unit]`. It skips step 0 and trainer `"none"`, keeps only the given groups if any, and raises `FileNotFoundError` naming the path if `index.json` is missing.
  - `label_distribution(units: Sequence[Unit]) -> dict[str, dict[str, int]]`, giving group → label → count. **It never calls the rule.**
  - CLI subcommand `labels <results_dir> [--group G]`, which prints `label_distribution`.

- [ ] **Step 1: Write the failing tests.** A module-scoped fixture runs `capsplit.harness.runner.run_experiment(cfg, out_root=tmp, workers=1)` on a tiny grid config: `configs/grid_example.json` with seeds [0], budget 4, checkpoints [0, 2, 4] and `save_policies: "final"`.

```python
def test_units_skip_step0_and_none(quick_grid):
    units = load_units(quick_grid)
    assert units and all(u.step > 0 and u.trainer != "none" for u in units)

def test_unit_j_lists_average_to_record(quick_grid):
    for u in load_units(quick_grid, groups=["all"]):
        rec = _record(quick_grid, u)  # helper in the test file: finds the matching metrics record
        assert np.mean(u.j0) == pytest.approx(rec["j0"], abs=1e-9)
        assert np.mean(u.j1) == pytest.approx(rec["j1"], abs=1e-9)

def test_missing_index_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="index.json"):
        load_units(tmp_path)

def test_label_distribution_counts_every_unit(quick_grid):
    units = load_units(quick_grid)
    dist = label_distribution(units)
    assert sum(sum(c.values()) for c in dist.values()) == len(units)
```

- [ ] **Step 2:** Run `pytest tests/test_rq4.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement the dataclass, `load_units`, `label_distribution` and the `labels` CLI. Use argparse subcommands, as `rq2_time_to_cross.py` does, with `main(argv)`.
- [ ] **Step 4:** Run `pytest tests/test_rq4.py -v`. Expected: PASS.
- [ ] **Step 5:** Commit: `feat(rq4): read pass@k units from harness metrics`.

---

### Task 3: Scoring, verdicts and the freeze guard

**Files:**
- Modify: `experiments/rq4_passk.py`
- Test: `tests/test_rq4.py`

**Interfaces:**
- Consumes: `Unit` and `load_units` (Task 2); `capsplit.diagnostics.passk_rule` and `score_rule`; `pilot_rq4.k_grid`; `capsplit.harness.report.bootstrap_ci`.
- Produces:
  - `K_LARGE = 10`, `TOL = 1e-9`, `PRIMARY_GROUP = "heldout"`, `MIN_CELL_UNITS = 10`, `AGREEMENT_LINE = 0.95`.
  - `rule_of(unit: Unit, ks: Sequence[int], k_large: int = K_LARGE) -> str`.
  - `score_units(units: Sequence[Unit], ks, k_large: int = K_LARGE) -> dict`. Its keys are the `score_rule` output (`primary`, `secondary`, `undetermined_rate`, `per_cell`) plus `per_seed_primary: dict[int, float]`, `ci: tuple[float, float]` and `n_units: int`.
  - `q1_verdict(ci: tuple[float, float]) -> str`: "supported" if `ci[1] < 0.95`, "contradicted" if `ci[0] > 0.95`, else "inconclusive". Q2 uses the same function on Phase 1.
  - `q3_verdict(per_cell: dict) -> dict`, with disagreement = 1 − primary. "supported" if `compounded_sharpening` and `reweighting` each exceed both other cells; "contradicted" if either of them has the lowest disagreement of the four; "inconclusive" otherwise, or if any of the four cells is missing or has `runs < 10`.
  - `require_frozen() -> None`: raises `SystemExit` with a message containing `rq4-frozen` unless `git merge-base --is-ancestor rq4-frozen HEAD` succeeds.
  - CLI `score --phase2 <rq3_results_dir> --phase1 <grid_results_dir> [--out DIR]` calls `require_frozen()` first, then writes the outputs of Task 4. `--allow-unfrozen` exists for tests only and stamps every output with `"UNFROZEN TEST"`.

- [ ] **Step 1: Write the failing tests.** These use synthetic units, built with a helper `_unit(j0, j1, label, seed=0, arm="rloo")`.

```python
KS = (1, 2, 5, 10, 20, 50, 77)

def test_all_prompts_improve_is_capability():
    assert rule_of(_unit([0.1, 0.2], [0.5, 0.6], "reweighting"), KS) == "capability"

def test_one_prompt_drops_is_sharpening():
    assert rule_of(_unit([0.3, 0.05], [0.9, 0.0], "pure_sharpening"), KS) == "sharpening"

def test_saturated_ties_are_undetermined_and_count_as_disagreement():
    u = _unit([1.0, 1.0], [1.0, 1.0], "capability")
    assert rule_of(u, KS) == "undetermined"
    s = score_units([u], KS)
    assert s["primary"] == 0.0 and s["undetermined_rate"] == 1.0

def test_per_seed_rates_and_ci():
    units = [_unit([0.1], [0.5], "capability", seed=s) for s in range(5)] + \
            [_unit([0.1], [0.5], "reweighting", seed=s) for s in range(5)]
    s = score_units(units, KS)
    assert s["per_seed_primary"] == {i: 0.5 for i in range(5)} and s["ci"] == (0.5, 0.5)

def test_q1_rules():
    assert q1_verdict((0.4, 0.6)) == "supported"
    assert q1_verdict((0.96, 0.99)) == "contradicted"
    assert q1_verdict((0.9, 0.97)) == "inconclusive"

def test_q3_rules_and_small_cells():
    cell = lambda p, n=20: {"primary": p, "runs": n}
    good = {"compounded_sharpening": cell(0.1), "reweighting": cell(0.2), "capability": cell(0.9), "pure_sharpening": cell(0.8)}
    assert q3_verdict(good)["verdict"] == "supported"
    bad = {**good, "reweighting": cell(0.95)}
    assert q3_verdict(bad)["verdict"] == "contradicted"
    assert q3_verdict({**good, "capability": cell(0.9, n=5)})["verdict"] == "inconclusive"

def test_score_refuses_before_freeze(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(rq4_passk, "_tag_is_ancestor", lambda: False)
    with pytest.raises(SystemExit):
        main(["score", "--phase2", str(tmp_path), "--phase1", str(tmp_path)])
    assert "rq4-frozen" in capsys.readouterr().err
```

`require_frozen` calls a module-level `_tag_is_ancestor()`, so the test can replace it.

- [ ] **Step 2:** Run `pytest tests/test_rq4.py -v`. Expected: the new tests FAIL.
- [ ] **Step 3:** Implement the constants, `rule_of`, `score_units`, the verdict functions, `require_frozen` and the CLI wiring. Report-writing is a stub until Task 4.
- [ ] **Step 4:** Run `pytest tests/test_rq4.py -v`. Expected: PASS.
- [ ] **Step 5:** Commit: `feat(rq4): score the pass@k rule with pre-registered verdicts and a freeze guard`.

---

### Task 4: Report outputs, sensitivity and SRR/NDR (unverified)

**Files:**
- Modify: `experiments/rq4_passk.py`, `src/capsplit/diagnostics.py` (append `srr_ndr`)
- Test: `tests/test_rq4.py`, `tests/test_diagnostics.py` (append)

**Interfaces:**
- Consumes: Tasks 2 and 3; saved policies (`<results>/<run_dir>/seed-<s>.policies.pt`, `{step: state_dict}`) and bases (`<results>/bases/<key>/policy.pt`, a pickled Policy), via `capsplit.harness.runner._torch_load`, and the task rebuilt from the run's `manifest.json` config through `capsplit.harness.config` and the `TASKS` registry.
- Produces:
  - `srr_ndr(q0: np.ndarray, q1: np.ndarray, eps: float) -> dict`, with keys `P`, `S`, `E` (counts, using a strict `> eps` as the note states), `srr` (`P/(P+S)` or `None` if 0) and `ndr` (`E/(P+E)` or `None`).
  - `write_report(phase2: dict, phase1: dict, out: Path) -> dict` writes:
    - `tables/agreement.csv`: phase, group, step, cell, units, primary, secondary, undetermined_rate;
    - `tables/k_large_sensitivity.csv`: k_large ∈ {5, 10, 20, 50}, with only values ≤ the phase's k_max;
    - `tables/units.csv`: every primary unit's rule and label;
    - `figures/agreement_by_cell.png`;
    - `figures/disagreement_examples.png`: three disagreeing units' pass@k curves, base against trained;
    - `verdicts.json` and `verdicts.md`, with Q1–Q3, the validity checks and `"frozen": true|false`.
  - CLI `srr-ndr <results_dir> [--out DIR]` writes only `srr_ndr_UNVERIFIED.csv`: phase, run, seed, step, group, per-prompt P, S, E, srr and ndr, at ε = c/N for N = 64, c = 1.

- [ ] **Step 1: Write the failing tests**

```python
def test_srr_ndr_counts_and_zero_denominators():
    r = srr_ndr(np.array([0.2, 0.0, 0.05]), np.array([0.0, 0.3, 0.5]), eps=0.1)
    assert (r["P"], r["S"], r["E"]) == (0, 1, 2) and r["srr"] == 0.0 and r["ndr"] == 1.0
    empty = srr_ndr(np.array([0.0]), np.array([0.0]), eps=0.1)
    assert empty["srr"] is None and empty["ndr"] is None

def test_report_writes_everything_and_keeps_srr_out(tmp_path, quick_grid):
    main(["score", "--phase2", str(quick_grid), "--phase1", str(quick_grid), "--out", str(tmp_path), "--allow-unfrozen"])
    for name in ("tables/agreement.csv", "tables/k_large_sensitivity.csv", "tables/units.csv",
                 "figures/agreement_by_cell.png", "verdicts.json", "verdicts.md"):
        assert (tmp_path / name).exists(), name
    text = (tmp_path / "verdicts.md").read_text()
    assert "UNFROZEN TEST" in text and "SRR" not in text.upper().replace("SRR/NDR PENDING", "")

def test_srr_ndr_cli_writes_only_unverified_file(tmp_path, quick_grid_with_policies):
    main(["srr-ndr", str(quick_grid_with_policies), "--out", str(tmp_path)])
    assert [p.name for p in tmp_path.iterdir()] == ["srr_ndr_UNVERIFIED.csv"]
```

`quick_grid_with_policies` is the Task 2 fixture with `save_policies: "checkpoints"`. `verdicts.md` carries the single line "SRR/NDR pending a person's check of docs/notes/invisible-leash-scores.md".

- [ ] **Step 2:** Run `pytest tests/test_rq4.py tests/test_diagnostics.py -v`. Expected: the new tests FAIL.
- [ ] **Step 3:** Implement `srr_ndr`, `write_report` and the `srr-ndr` CLI. Validity checks in `verdicts.json`:
  - every unit's J lists have the group's prompt count;
  - the primary dataset sizes are 210 for Phase 2 and the Phase 1 equivalent;
  - every one of the four cells is reported, even if empty.
- [ ] **Step 4:** Run `pytest -q` (the whole suite). Expected: all pass.
- [ ] **Step 5:** Commit: `feat(rq4): write RQ4 tables, figures and verdicts; compute SRR/NDR as unverified`.

---

### Task 5: Reproduce RQ3

**Files:**
- Modify: `experiments/rq4_passk.py`
- Test: `tests/test_rq4.py`

**Interfaces:**
- Produces: `repro_compare(new_report_dir: Path, committed_dir: Path = Path("docs/experiment_main")) -> dict`. It compares `summary.json`, `tables/main.csv` and `verdicts.json`. For each file it reports `"identical"` (bytes equal), or `"numeric"` with `max_abs_diff` over all numbers if the files parse to the same structure, or `"different"` with the first differing key. Overall `match` is true only if every file is identical, or numeric with `max_abs_diff ≤ 1e-6`. CLI: `repro <new_report_dir>`.

- [ ] **Step 1: Write the failing tests**

```python
def test_repro_identical_and_perturbed(tmp_path):
    shutil.copytree("docs/experiment_main", tmp_path / "same")
    assert repro_compare(tmp_path / "same")["match"] is True
    shutil.copytree("docs/experiment_main", tmp_path / "bad")
    s = json.loads((tmp_path / "bad/verdicts.json").read_text())
    s["H1"]["verdict"] = "supported"  # use the real key path in verdicts.json
    (tmp_path / "bad/verdicts.json").write_text(json.dumps(s))
    r = repro_compare(tmp_path / "bad")
    assert r["match"] is False and r["files"]["verdicts.json"]["status"] == "different"
```

- [ ] **Step 2:** Run it. Expected: FAIL.
- [ ] **Step 3:** Implement `repro_compare` and the CLI.
- [ ] **Step 4:** Run `pytest tests/test_rq4.py -v`. Expected: PASS.
- [ ] **Step 5:** Commit: `feat(rq4): compare a rerun RQ3 report with the committed one`.

---

### Task 6: Phase 1 config and the label check

**Files:**
- Create: `configs/rq4_grid.json`
- Modify: `docs/DECISIONS.md`

- [ ] **Step 1:** Write `configs/rq4_grid.json` (name `rq4_grid`):
  - Task: `grid`, n = 5, H = 8, with 8 variants sharing one wall set. 2 easy and 2 hard prompts in each split, and goals in different corners and edges, so routes pull the shared cells apart.
  - Base: `grid_shared_routes`.
  - Backend: `exact`.
  - Arms:
    - `reinforce_all`: `tabular_reinforce`, `optimizer: "sgd"`, lr 0.5, batch 32;
    - `reinforce_subset`: the same, with `only_tags: {"difficulty": "easy"}`;
    - `rloo_all`: `rloo`, group 32, lr 0.1.
  - Seeds 0–19; budget 400; checkpoints [0, 25, 50, 100, 200, 400].
  - Measure: ns [16, 64], cs [0.5, 1, 2], group_by [difficulty, split].
  - `save_policies: "checkpoints"`.
  - Exact values are the builder's call where this step doesn't pin them.
- [ ] **Step 2: Label check.** Run the config with seeds 0–4 to a scratch root, then `python experiments/rq4_passk.py labels <scratch>/rq4_grid --group heldout`.
  - Requirement: each of the four cells covers ≥ 10% of `heldout` units over all steps > 0.
  - If not, adjust the base params, the wall set or the arms, rerun, and log each change as an `RQ4-n` entry.
  - **Do not run `score` or call `passk_rule` here.**
- [ ] **Step 3:** Log the final config and the label distribution as an RQ4 decision. Commit: `feat(rq4): add the Phase 1 shared-policy grid sweep config`.

---

### Task 7: Freeze the pre-registration

**Files:**
- Create: `docs/rq4-preregistration.md`

- [ ] **Step 1:** Write it from design §5 and §7, with every frozen value:
  - k_large 10, tol 1e-9, k grids (77 and 655);
  - primary group, checkpoint and (N, c);
  - excluded units;
  - Q1–Q3 rules verbatim, and the 0.95 line;
  - the bootstrap;
  - the validity checks;
  - the SRR/NDR hold;
  - the Phase 1 config hash, the rerun commit `46490ae`, and the statement that only labels (never rule outputs) were inspected before the freeze.
- [ ] **Step 2:** Check that no rule output exists: `ls results/rq4_report` fails or is empty.
- [ ] **Step 3:** Commit: `docs(rq4): freeze the pass@k misdiagnosis pre-registration`, then `git tag rq4-frozen`.

---

### Task 8: Full runs, reproduction, scoring and the report

**Files:**
- Create: `docs/rq4-passk.md`, `docs/rq4/` (copy of `results/rq4_report/`)
- Modify: `README.md`, `docs/DECISIONS.md`

- [ ] **Step 1: Phase 1 sweep.** From a clean tree at the tag: `python -m capsplit.harness run configs/rq4_grid.json --out results --workers 8` (in the background). Then confirm every manifest is `dirty: false` and records the tag's commit.
- [ ] **Step 2: Phase 2.** When the rerun in `../capsplit-rq3-rerun` finishes:
  - Run `python -m capsplit.harness report results/main` and `python -m capsplit.harness verdicts results/main` there.
  - Then run `python experiments/rq4_passk.py repro ../capsplit-rq3-rerun/results/main/report`.
  - Record the result as an RQ4 decision. **If `match` is false, stop and report the mismatch to the user before scoring.**
- [ ] **Step 3: Score.** `python experiments/rq4_passk.py score --phase2 ../capsplit-rq3-rerun/results/main --phase1 results/rq4_grid --out results/rq4_report`.
- [ ] **Step 4: SRR/NDR.** Run `python experiments/rq4_passk.py srr-ndr` on each phase into `results/rq4_report/`. Do not report the numbers.
- [ ] **Step 5: Write `docs/rq4-passk.md`.**
  - Numbers come from `docs/rq4/`; say which, if any, come from ad-hoc analysis.
  - Give per-cell rates beside every overall rate, and say how each phase's run mix was chosen.
  - Report negative and inconclusive verdicts as prominently as positive ones.
  - SRR/NDR are marked pending.
  - Include a note for the Qwen comparison: the frozen rule and k_large, and that it needs per-prompt J for both models.
  - Copy `results/rq4_report/` to `docs/rq4/`, without the UNVERIFIED file.
  - Update the README status and layout.
- [ ] **Step 6:** Run `pytest -q`. Expected: all pass. Commit: `docs(rq4): add the pass@k misdiagnosis report and results`.
- [ ] **Step 7:** Ask the user before pushing or opening a PR.
