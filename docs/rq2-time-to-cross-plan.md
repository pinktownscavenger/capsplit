# RQ2 Time-to-Cross Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure, derive and report how many exact policy-gradient updates it takes for a rare correct route to cross the reach line, on isolated and competing routes, as specified in the approved design.

**Architecture:** Pure helpers (crossing detection, slopes, the scalar recursion and its integral) go in a new `src/capsplit/crossing.py`. A fast tracing loop goes in `src/capsplit/train/tabular.py`, next to the existing exact PG. Experiments, the report and the verdicts go in one new script, `experiments/rq2_time_to_cross.py`, which imports the pilot's worlds and bases without changing `pilot_rq2.py`. Results go through `capsplit.runlog` into `results/rq2_*`.

**Tech Stack:** Python ≥ 3.10, numpy (float64), matplotlib (`viz` extra) for figures, pytest. No new dependencies: in particular **no scipy**, so integrals use numpy Gauss–Legendre.

**Spec:** `docs/rq2-time-to-cross-design.md` (approved 2026-11-19 with the recommended defaults). Read it first; section numbers below (§5, §6.1, …) refer to it.

## Global Constraints

- All tabular arithmetic in float64; inputs are cast with `np.asarray(x, dtype=float)`.
- Do not modify `experiments/pilot_rq2.py`, `experiments/run_pilot.py`, or anything under `src/capsplit/harness/`.
- Existing behaviour of `train_exact_pg`, `exact_pg_gradient` and `reinforce` stays unchanged; new functions only.
- Crossing convention (matches the pilot's `updates_to_reach`): a series starts with the step-0 value, and the time to cross is the first index i with `series[i] >= line`. `None` means censored.
- Reach line for arm A: ℓ = 0.1. Arm B primary rule: q(rare) ≥ 10 × q₀(rare); secondary: q(rare) ≥ 0.1, reported only where q₀(rare) < 0.1.
- Work on branch `exp/rq2-time-to-cross`. Commit messages are Conventional Commits with no AI attribution trailers.
- Decisions made during implementation go in `docs/DECISIONS.md` under a new heading `## RQ2 time-to-cross`, with ids `RQ2-1`, `RQ2-2`, … in the existing table format.
- Long tests are marked `@pytest.mark.slow` (the marker already exists in `pyproject.toml`).
- Full runs (Task 9) happen only after the pre-registration is committed and tagged `rq2-frozen` (Task 8).

## Review Focus

1. **A point never crosses within the cap** (arm A at small q₀ with a low cap, or `--quick`): the slope helpers must refuse cleanly, and the report must mark P1/P2 "inconclusive (censored)" rather than crash or fit around the gap. Pinned in Task 1 and Task 7.
2. **The series already starts at or above the line** (arm B ratio 1 at large q₀, or a base above ℓ): the time to cross is 0, not None. Pinned in Task 1.
3. **Bisection cannot reach its target in [−8, 8]** (a noise draw that makes the target impossible): raise `ValueError` naming the target, instead of silently returning the boundary bonus as the pilot's `isolated_base` would. Pinned in Task 4.
4. **float32 logits passed in** (for example loaded from elsewhere): the trace must upcast and still match the scalar recursion at q₀ = 10⁻⁶, where updates are about 10⁻⁶. Pinned in Task 3.
5. **`report` run before the run subcommands have written results:** exit with a message naming the missing subcommand, not a stack trace. Pinned in Task 7.

---

### Task 0: Branch and docs

**Files:**
- Add: `docs/rq2-time-to-cross-design.md`, `docs/rq2-time-to-cross-plan.md` (already written, uncommitted)

- [ ] **Step 1:** `git switch -c exp/rq2-time-to-cross` from an up-to-date `main`.
- [ ] **Step 2:** Commit both docs: `git commit -m "docs: add the RQ2 time-to-cross design and plan"`.

---

### Task 1: Crossing and slope helpers

**Files:**
- Create: `src/capsplit/crossing.py`
- Test: `tests/test_crossing.py`

**Interfaces:**
- Produces:
  - `crossing_step(series: np.ndarray, line: float) -> int | None`. 1-D series; first index with `series[i] >= line`, else `None`.
  - `loglog_slopes(xs: Sequence[float], ys: Sequence[float | None]) -> tuple[float, np.ndarray]`. Returns (least-squares slope of log10 y on log10 x, local slopes between neighbours with length n − 1). Raises `ValueError` if any y is `None` or ≤ 0, or if n < 2.

- [ ] **Step 1: Write the failing tests**

```python
def test_crossing_first_index_at_or_above():
    assert crossing_step(np.array([0.01, 0.05, 0.1, 0.2]), 0.1) == 2

def test_crossing_censored():
    assert crossing_step(np.array([0.01, 0.05]), 0.1) is None

def test_crossing_already_above_at_step_zero():
    assert crossing_step(np.array([0.3, 0.2]), 0.1) == 0

def test_slopes_recover_power_law():
    xs = np.logspace(-6, -2, 9)
    fitted, local = loglog_slopes(xs, 5.0 * xs ** -0.9)
    assert fitted == pytest.approx(-0.9, abs=1e-12)
    assert local.shape == (8,) and np.allclose(local, -0.9)

def test_slopes_refuse_censored_point():
    with pytest.raises(ValueError, match="censored"):
        loglog_slopes([1e-3, 1e-2], [None, 10])
```

- [ ] **Step 2:** Run `pytest tests/test_crossing.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3:** Implement both functions in `src/capsplit/crossing.py`. The `ValueError` message for `None` contains "censored".
- [ ] **Step 4:** Run `pytest tests/test_crossing.py -v`. Expected: PASS.
- [ ] **Step 5:** Commit: `feat(crossing): add crossing-step and log-log slope helpers`.

---

### Task 2: Scalar recursion, integral and the written derivation

**Files:**
- Modify: `src/capsplit/crossing.py`
- Modify: `docs/proofs.md` (new section `## 7. Time to cross on an isolated route`, inserted before `## Numerical checks`, plus rows in the numerical checks table)
- Test: `tests/test_crossing.py`

**Interfaces:**
- Consumes: `crossing_step` (Task 1).
- Produces (all with `n_actions: int = 4`):
  - `scalar_isolated_trace(q0: float, horizon: int, lr: float, steps: int, n_actions=4, stop_line: float | None = None) -> np.ndarray`. q at steps 0..T. Stops after the first row ≥ `stop_line` if given.
  - `scalar_isolated_updates(q0, horizon, lr, line, cap, n_actions=4) -> int | None`.
  - `integral_time(q0, horizon, lr, line, n_actions=4) -> float`.

The recursion (design §5): p₀ = q₀^{1/H}, g₀ = log((A − 1)p₀ / (1 − p₀)), then repeat p = eᵍ / (eᵍ + A − 1), q = p^H, g ← g + lr · A/(A − 1) · q(1 − p). The integral is T = ((A − 1)/A)/lr · ∫ from q₀^{1/H} to ℓ^{1/H} of dp / (p^{H+1}(1 − p)²). Evaluate it with `np.polynomial.legendre.leggauss(400)` after substituting u = log p, which handles the steep integrand.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("q0,updates", [(1e-5, 17842), (1e-4, 2358), (1e-3, 362), (1e-2, 63)])
def test_scalar_recursion_reproduces_pilot_counts(q0, updates):
    assert scalar_isolated_updates(q0, 8, 1.0, 0.1, 30000) == updates

@pytest.mark.parametrize("q0,updates", [(1e-6, 148403), (1e-5, 17837), (1e-4, 2354), (1e-3, 359), (1e-2, 61)])
def test_integral_reproduces_blueprint_d6(q0, updates):
    assert round(integral_time(q0, 8, 1.0, 0.1)) == updates

def test_integral_within_5pct_of_recursion():
    for q0 in np.logspace(-6, -2, 9):
        assert integral_time(q0, 8, 1.0, 0.1) == pytest.approx(scalar_isolated_updates(q0, 8, 1.0, 0.1, 400_000), rel=0.05)

def test_scalar_fitted_and_local_slopes():
    qs = np.logspace(-6, -2, 9)
    fitted, local = loglog_slopes(qs, [scalar_isolated_updates(q, 8, 1.0, 0.1, 400_000) for q in qs])
    assert fitted == pytest.approx(-0.844, abs=0.001)
    assert local[0] == pytest.approx(-0.928, abs=0.001)

def test_scalar_time_scales_inversely_with_lr():
    t1 = scalar_isolated_updates(1e-4, 8, 1.0, 0.1, 100_000)
    assert abs(0.5 * scalar_isolated_updates(1e-4, 8, 0.5, 0.1, 100_000) - t1) / t1 < 0.05
```

- [ ] **Step 2:** Run `pytest tests/test_crossing.py -v`. Expected: the new tests FAIL.
- [ ] **Step 3:** Implement the three functions.
- [ ] **Step 4:** Run `pytest tests/test_crossing.py -v`. Expected: PASS. If `test_integral_reproduces_blueprint_d6` misses by 1 on any point, raise the node count and keep the test exact.
- [ ] **Step 5:** Write `docs/proofs.md` §7. Cover: setup, the gradient on route and off-route actions, the gap recursion with the A/(A − 1) factor, the integral, the small-q₀ asymptotic T ≈ (A − 1)/(A · lr · H · q₀), why the slope tends to −1, and where the reduction fails (asymmetric off-route logits; H longer than the route). Open the section with: "**Status: written 2026-11-19; not yet re-derived by a second teammate.**" Add rows to the numerical checks table for the pilot-count match and the D6 match.
- [ ] **Step 6:** Commit: `feat(crossing): add the isolated-route scalar recursion and integral; derive it in proofs §7`.

---

### Task 3: Fast exact-PG trace

**Files:**
- Modify: `src/capsplit/train/tabular.py` (append; existing functions untouched)
- Test: `tests/test_train_tabular.py` (append)

**Interfaces:**
- Consumes: `_score`, `log_policy` (existing).
- Produces:
  - `exact_pg_step(table: TrajectoryTable, logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]`. Returns (gradient identical to `exact_pg_gradient`'s, q per solution as an array of length `len(table.solutions)`), computed from the correct rows only. Do not call `trajectory_probs`: it scores every row and is about 20× slower (1 ms against 0.055 ms per step at H = 8).
  - `trace_exact_pg(table, logits0, lr: float, steps: int, watch: Sequence[int], stop_line: float | None = None) -> tuple[np.ndarray, np.ndarray]`. Returns (q array of shape (T + 1, len(watch)) for steps 0..T in float64, final logits). If `stop_line` is given, it stops after the first row where `q[:, 0] >= stop_line`.

- [ ] **Step 1: Write the failing tests**

```python
def test_exact_pg_step_matches_existing(competing):
    table, state_index = competing
    logits = competing_base(state_index, (2, 1.5))
    gradient, q = exact_pg_step(table, logits)
    assert np.allclose(gradient, exact_pg_gradient(table, logits)[0], atol=1e-15)
    assert np.allclose(q, solution_probs(table, trajectory_probs(table, logits)), atol=1e-15)

def test_trace_equals_scalar_recursion(isolated):
    table, state_index = isolated
    q, _ = trace_exact_pg(table, isolated_base(table, state_index, 1e-4), 1.0, 200, [0])
    assert np.max(np.abs(q[:, 0] - scalar_isolated_trace(1e-4, 8, 1.0, 200))) < 1e-9

def test_trace_stops_at_line_and_matches_pilot(isolated):
    table, state_index = isolated
    q, _ = trace_exact_pg(table, isolated_base(table, state_index, 1e-3), 1.0, 30000, [0], stop_line=0.1)
    assert len(q) - 1 == 362 and q[-1, 0] >= 0.1 > q[-2, 0]

@pytest.mark.slow
def test_trace_upcasts_float32_at_1e6(isolated):
    table, state_index = isolated
    logits = isolated_base(table, state_index, 1e-6).astype(np.float32)
    q, _ = trace_exact_pg(table, logits, 1.0, 400_000, [0], stop_line=0.1)
    assert q.dtype == np.float64
    assert abs((len(q) - 1) - scalar_isolated_updates(float(q[0, 0]), 8, 1.0, 0.1, 400_000)) <= 1
```

The last test uses the float32 base's own q₀ (`q[0, 0]`), because casting changes q₀ slightly, and allows one update of slack for that rounding over about 148,000 steps. A float32 accumulation bug would be off by thousands.

- [ ] **Step 2:** Run `pytest tests/test_train_tabular.py -v -m "slow or not slow"`. Expected: the new tests FAIL.
- [ ] **Step 3:** Implement both functions. `exact_pg_step` gets per-solution q with `np.bincount(table.solution_ids[correct], weights=probs, minlength=len(table.solutions))`.
- [ ] **Step 4:** Run the same command. Expected: PASS, including the pilot's existing tests.
- [ ] **Step 5:** Commit: `feat(train): add a fast exact-PG trace that records solution probabilities`.

---

### Task 4: Worlds and bases for RQ2

**Files:**
- Create: `experiments/rq2_time_to_cross.py` (constants and setup functions only in this task)
- Test: `tests/test_rq2.py`

**Interfaces:**
- Consumes: `pilot_rq2.ISOLATED_WALLS`, `COMPETING_WORLD`, `ROUTE_A`, `ROUTE_B`; `phase1_rq1.WORLD`, `R_COMMON`, `R_RARE`; `capsplit.worlds` (`GridWorld`, `compile_grid`, `policy_from_routes`, `logits_to_array`); `exact_pg_step` (Task 3).
- Produces:
  - Frozen constants at the top of the file, copied from design §6: `LINE = 0.1`, `ARM_A_Q0 = tuple(np.logspace(-6, -2, 9))`, `ARM_A_LRS = (0.5, 1.0, 2.0)`, `ARM_A_CAP = 400_000`, `ROBUST_SEEDS = range(20)`, `ROBUST_SIGMA` (builder's call within [0.5, 1.5], default 1.0), `ROBUST_H` (Task 5 decides 9 or 10), `ARM_B_WORLDS = ("W1", "W2", "W3")`, `ARM_B_RATIOS = (1.0, 0.9, 0.5, 0.1, 0.01)`, `ARM_B_LRS = (0.5, 1.0, 2.0)`, `ARM_B_SEEDS = range(20)`, `ARM_B_CAP = 20_000`, `CROSS_FACTOR = 10`, `COMMON_BONUS = {"W1": …, "W2": …, "W3": …}` (builder's call: chosen so q₀(common) is in [0.01, 0.04] at ratio 1 with σ = 0, which keeps 10 × q₀(rare) ≤ 0.4 reachable; logged as a DECISIONS entry), `BISECTION_STEPS = 60`, `BONUS_RANGE = (-8.0, 8.0)`.
  - `isolated_world(horizon: int = 8) -> GridWorld`, the pilot's walls with that H.
  - `competing_world(name: str) -> tuple[GridWorld, tuple, tuple]`, returning (world, common route, rare route). W1 = (`COMPETING_WORLD`, `ROUTE_A`, `ROUTE_B`); W2 = (`phase1_rq1.WORLD`, `R_COMMON`, `R_RARE`); W3 = 5×5, H = 8, `ISOLATED_WALLS - {(0, 1)}`, with common = the top-right route and rare = the left-bottom route.
  - `noisy_isolated_base(table, state_index, world, target_q0: float, sigma: float, seed: int) -> np.ndarray`. Adds N(0, σ²) noise (`np.random.default_rng(seed)`) to every logit that is not a route action at a route cell, then bisects one per-step route bonus over `BONUS_RANGE` to hit `target_q0`. Raises `ValueError` if the target lies outside what the range reaches. With σ = 0 at H = 8 it equals `pilot_rq2.isolated_base` to 1e-12.
  - `competing_base(table, state_index, name: str, ratio: float, sigma: float, seed: int) -> tuple[np.ndarray, float, float]`. Returns (logits, q₀(common), q₀(rare)). Noise goes on every logit not on either bonus route. The common bonus is fixed at `COMMON_BONUS[name]`, and the rare bonus is bisected so q₀(rare)/q₀(common) = ratio. In W2 the two routes share moves, so recompute both q₀ at each bisection step. Raises `ValueError` if unreachable.

- [ ] **Step 1: Write the failing tests**

```python
def test_w3_has_exactly_two_routes_and_w2_eight():
    assert len(compile_grid(competing_world("W3")[0])[0].solutions) == 2
    assert len(compile_grid(competing_world("W2")[0])[0].solutions) == 8

@pytest.mark.parametrize("horizon", [9, 10])
def test_longer_horizon_keeps_one_route(horizon):
    assert len(compile_grid(isolated_world(horizon))[0].solutions) == 1

def test_noise_free_base_equals_pilot():
    table, si = compile_grid(isolated_world(8))
    assert np.allclose(noisy_isolated_base(table, si, isolated_world(8), 1e-4, 0.0, 0), isolated_base(table, si, 1e-4), atol=1e-12)

@pytest.mark.parametrize("seed", [0, 1])
def test_noisy_base_hits_target(seed):
    world = isolated_world(8); table, si = compile_grid(world)
    q0 = exact_pg_step(table, noisy_isolated_base(table, si, world, 1e-5, ROBUST_SIGMA, seed))[1][0]
    assert q0 == pytest.approx(1e-5, rel=0.01)

def test_unreachable_target_raises():
    world = isolated_world(8); table, si = compile_grid(world)
    with pytest.raises(ValueError, match="target"):
        noisy_isolated_base(table, si, world, 0.9999999, 0.0, 0)

@pytest.mark.parametrize("name", ["W1", "W2", "W3"])
@pytest.mark.parametrize("ratio", [1.0, 0.01])
def test_competing_ratio_and_common_range(name, ratio):
    world, _, _ = competing_world(name); table, si = compile_grid(world)
    _, q_common, q_rare = competing_base(table, si, name, ratio, ROBUST_SIGMA, 3)
    assert q_rare / q_common == pytest.approx(ratio, rel=0.01)
    assert CROSS_FACTOR * q_rare <= 0.5  # the primary rule stays achievable
    if ratio == 1.0:
        _, q_common0, _ = competing_base(table, si, name, 1.0, 0.0, 0)
        assert 0.01 <= q_common0 <= 0.04
```

- [ ] **Step 2:** Run `pytest tests/test_rq2.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement the constants and functions. Choose `COMMON_BONUS` per world to satisfy the range test, and add a DECISIONS entry (RQ2-1) recording the values and σ.
- [ ] **Step 4:** Run `pytest tests/test_rq2.py -v`. Expected: PASS.
- [ ] **Step 5:** Commit: `feat(rq2): add RQ2 worlds and noise-aware bases`.

---

### Task 5: Arm A and robustness runs

**Files:**
- Modify: `experiments/rq2_time_to_cross.py`
- Test: `tests/test_rq2.py`

**Interfaces:**
- Consumes: Tasks 2–4.
- Produces:
  - `run_arm_a(quick: bool, root: str = "results") -> pathlib.Path`. For each (lr, q₀): call `trace_exact_pg` with `stop_line=LINE` and cap `ARM_A_CAP`, and the scalar trace with the same stop. Save with `save_seed(dir, i, lr=, q0_target=, q0=, q=, q_scalar=, updates=, updates_scalar=, integral=)`, using `updates = -1` when censored. The directory is `run_dir("rq2_arm_a", config)`.
  - `run_robust(quick: bool, root: str = "results") -> pathlib.Path`. R1: lr = 1, H = 8, `ROBUST_SIGMA`, seeds `ROBUST_SEEDS`, every q₀. R2: H = `ROBUST_H`, σ = 0, every q₀. Saved as `run_dir("rq2_robust", config)` with fields `condition` ("R1" or "R2"), `seed`, `q0_target`, `q0`, `updates`.
  - Quick mode: q₀ ∈ {1e-4, 1e-3, 1e-2}, lr = 1 only, 2 R1 seeds, and R2 at H = 9.
- `main(argv)` with argparse subcommands `arm-a`, `robust`, `arm-b`, `sampled`, `report` (later tasks fill in the last three), each taking `--quick` and `--root`.

- [ ] **Step 1:** Before choosing `ROBUST_H`, time one `exact_pg_step` at H = 10 and estimate a full R2 sweep (sum of the predicted updates over the 9 q₀ × step time). Set `ROBUST_H = 10` if the estimate is under 2 CPU-hours (expected: minutes, since the step is about 0.67 ms), otherwise 9. Log it as RQ2-2.
- [ ] **Step 2: Write the failing tests**

```python
def test_arm_a_quick_matches_scalar(tmp_path):
    rows = load_seeds(run_arm_a(quick=True, root=str(tmp_path)))
    assert len(rows) == 3
    for r in rows:
        assert int(r["updates"]) == int(r["updates_scalar"]) > 0
        assert np.max(np.abs(r["q"][:, 0] - r["q_scalar"])) < 1e-9

def test_robust_quick_writes_both_conditions(tmp_path):
    rows = load_seeds(run_robust(quick=True, root=str(tmp_path)))
    assert {str(r["condition"]) for r in rows} == {"R1", "R2"}
    assert all(int(r["updates"]) > 0 for r in rows)
```

- [ ] **Step 3:** Run `pytest tests/test_rq2.py -v`. Expected: the new tests FAIL.
- [ ] **Step 4:** Implement `run_arm_a`, `run_robust` and `main`.
- [ ] **Step 5:** Run `pytest tests/test_rq2.py -v` and `python experiments/rq2_time_to_cross.py arm-a --quick --root <a scratch directory>`. Quick runs never go in `results/`, so the report never mixes them with the full run. Expected: PASS, and a printed table of updates against the scalar predictions.
- [ ] **Step 6:** Commit: `feat(rq2): add arm A and robustness runs`.

---

### Task 6: Arm B and the sampled stretch

**Files:**
- Modify: `experiments/rq2_time_to_cross.py`
- Test: `tests/test_rq2.py`

**Interfaces:**
- Consumes: Tasks 3–4; `capsplit.train.tabular.reinforce` (existing; its `on_step(step, logits)` callback is used to record q via `exact_pg_step(table, logits)[1]`).
- Produces:
  - `run_arm_b(quick: bool, root: str = "results") -> pathlib.Path`. For each (world, ratio, lr, seed): build `competing_base`, then `trace_exact_pg` for `ARM_B_CAP` steps watching [rare, common]. Compute in float64: `crossed_primary = crossing_step(q_rare, CROSS_FACTOR * q0_rare)`; `crossed_line = crossing_step(q_rare, LINE) if q0_rare < LINE else None`, with `line_applicable` saved; and peak and final q(rare). Save one `save_seed` file per (world, ratio, lr) holding arrays over seeds, with traces `q_rare` and `q_common` of shape (seeds, cap + 1) as **float32** to bound disk use (about 70 MB in total), and the crossing results as int arrays with −1 for censored. Quick mode: W1 only, ratios {1.0, 0.1}, lr = 2, 2 seeds, cap 2,000.
  - `run_sampled(quick: bool, root: str = "results") -> pathlib.Path`. REINFORCE with the pilot defaults (batch 32, lr 0.5, baseline decay 0.9), seeds 0–19, cap 20,000 updates. Arm A at q₀ ∈ {1e-4, 1e-3, 1e-2} with σ = 0, and arm B on W1 at ratios {1.0, 0.5, 0.1} with σ = 0. Record per seed the time to cross in updates and in sampled episodes (updates × 32), −1 if censored, and for arm B which route ends with more mass. Quick mode: arm A q₀ = 1e-2 only, 2 seeds, cap 2,000.

- [ ] **Step 1: Write the failing tests**

```python
def test_w1_noise_free_symmetric_crosses_and_rare_does_not():
    world, common, rare = competing_world("W1"); table, si = compile_grid(world)
    watch = [table.solutions.index(tuple(rare)), table.solutions.index(tuple(common))]
    for ratio, crosses in ((1.0, True), (0.1, False)):
        logits, _, q0_rare = competing_base(table, si, "W1", ratio, 0.0, 0)
        q, _ = trace_exact_pg(table, logits, 2.0, 2000, watch)
        assert (crossing_step(q[:, 0], CROSS_FACTOR * q0_rare) is not None) is crosses

def test_arm_b_quick_writes_expected_fields(tmp_path):
    rows = load_seeds(run_arm_b(quick=True, root=str(tmp_path)))
    assert {float(r["ratio"]) for r in rows} == {1.0, 0.1}
    for r in rows:
        assert r["q_rare"].dtype == np.float32 and r["q_rare"].shape == (2, 2001)
        assert np.all(r["line_applicable"])  # q0(rare) < 0.1 everywhere, given COMMON_BONUS

def test_sampled_quick_reports_updates_and_episodes(tmp_path):
    rows = load_seeds(run_sampled(quick=True, root=str(tmp_path)))
    for r in rows:
        ok = r["updates"] >= 0
        assert np.array_equal(r["episodes"][ok], 32 * r["updates"][ok])
```

The first test encodes the pilot's W1 result: only the symmetric start crosses. It uses σ = 0, because with off-route noise even ratio 1 is no longer an exact tie, and either route may win. That is what arm B measures, so the noisy runs are not asserted.

- [ ] **Step 2:** Run `pytest tests/test_rq2.py -v`. Expected: the new tests FAIL.
- [ ] **Step 3:** Implement both runners and wire them into `main`.
- [ ] **Step 4:** Run `pytest tests/test_rq2.py -v`. Expected: PASS.
- [ ] **Step 5:** Commit: `feat(rq2): add competing-route sweep and sampled REINFORCE stretch`.

---

### Task 7: Report, figures and verdicts

**Files:**
- Modify: `experiments/rq2_time_to_cross.py`
- Test: `tests/test_rq2.py`

**Interfaces:**
- Consumes: the four run directories (found as the single hash directory under each of `results/rq2_arm_a`, `rq2_robust`, `rq2_arm_b` and `rq2_sampled`, or the newest by manifest mtime if several exist, which the report prints); `loglog_slopes`.
- Produces: `report(root: str = "results", out: str | None = None) -> dict`, writing to `<root>/rq2_report/` by default:
  - `tables/arm_a.csv` (lr, q0, updates, scalar, integral, local slope), `tables/robust.csv`, `tables/arm_b.csv` (world, ratio, lr, fraction crossing on each rule, median updates among those crossing, median peak, median final, censored count), `tables/sampled.csv`.
  - `figures/arm_a_loglog.png` (measured points, the integral curve, and the R1 min–max band), `figures/arm_b_fraction.png` (fraction crossing heatmap per world × lr over ratios), `figures/arm_b_traces.png` (median q(rare)/q₀(rare) over updates for W1–W3 at lr = 1).
  - `verdicts.json` and `verdicts.md` with P1–P4 computed exactly as design §7 states, plus the validity checks (full table equals scalar in arm A; each base's q₀ within 1% of target; every arm B run has 10 × q₀(rare) ≤ 0.5). A failed validity check sets the overall verdict to "invalid".
  - Returns the verdicts dict.

Verdict rules, copied from design §7:
- **P1:** supported if fitted slope ∈ [−1.0, −0.7] and local slope between the two smallest q₀ ≤ −0.9; contradicted if fitted slope > −0.3; else inconclusive. "Inconclusive (censored)" if any arm A point at lr = 1 is censored. The `verdicts.md` text for P1 states that it is settled by the §5 derivation.
- **P2:** supported if ≥ 18/20 R1 draws have fitted slope ≤ −0.7 and R2's fitted slope ≤ −0.7; contradicted if the median R1 slope > −0.3 or R2's slope > −0.3; else inconclusive. Censored draws count as not ≤ −0.7.
- **P3:** "holds" if, at every q₀, (max − min)/mean of lr × updates over the three lrs < 0.05; otherwise "does not hold". Reported, not a contradiction test.
- **P4:** supported if fraction crossing (primary rule) ≤ 0.1 for every world and lr at every ratio ≤ 0.5; contradicted if it is ≥ 0.5 for any world and lr at any ratio ≤ 0.1; else inconclusive.

- [ ] **Step 1: Write the failing tests**

```python
def test_report_errors_when_runs_missing(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["report", "--root", str(tmp_path)])
    assert "arm-a" in capsys.readouterr().err

def test_report_on_quick_runs(tmp_path):
    root = str(tmp_path)
    for sub in ("arm-a", "robust", "arm-b", "sampled"):
        main([sub, "--quick", "--root", root])
    v = report(root)
    assert v["validity"]["arm_a_matches_scalar"] is True
    assert v["P1"]["verdict"] in {"supported", "inconclusive", "inconclusive (censored)"}
    assert (tmp_path / "rq2_report" / "figures" / "arm_a_loglog.png").exists()

def test_p1_censored_point_gives_inconclusive():
    assert p1_verdict(q0s=[1e-4, 1e-3, 1e-2], updates=[None, 362, 63])["verdict"] == "inconclusive (censored)"

def test_p4_rules():
    assert p4_verdict({("W1", 0.5, 1.0): 0.0, ("W1", 0.1, 1.0): 0.05})["verdict"] == "supported"
    assert p4_verdict({("W1", 0.5, 1.0): 0.3, ("W1", 0.1, 1.0): 0.6})["verdict"] == "contradicted"
```

So that these tests can call them directly, the P1–P4 rules are separate pure functions: `p1_verdict(q0s, updates, local_cutoff=-0.9) -> dict`, `p2_verdict(r1_slopes: list[float | None], r2_slope: float | None) -> dict`, `p3_verdict(table: dict[float, dict[float, int]]) -> dict` (q₀ → lr → updates), and `p4_verdict(fractions: dict[tuple[str, float, float], float]) -> dict` ((world, ratio, lr) → fraction). Each returns `{"verdict": str, ...numbers used}`.

- [ ] **Step 2:** Run `pytest tests/test_rq2.py -v`. Expected: the new tests FAIL.
- [ ] **Step 3:** Implement `report` and the four verdict functions. matplotlib is imported inside `report`, so the run subcommands work without the `viz` extra.
- [ ] **Step 4:** Run `pytest -v` (the whole suite). Expected: all tests pass, including the 410 existing ones.
- [ ] **Step 5:** Commit: `feat(rq2): add the RQ2 report, figures and pre-registered verdicts`.

---

### Task 8: Freeze the pre-registration

**Files:**
- Create: `docs/rq2-preregistration.md`
- Modify: `docs/DECISIONS.md`

- [ ] **Step 1:** Write `docs/rq2-preregistration.md`. It contains design §6's settings with the final values of every builder's call (σ, `COMMON_BONUS`, `ROBUST_H`), §7's table and decision rules verbatim, the crossing rules, the validity checks, and the statement that P1 is settled by the derivation. Header: "**Status:** frozen before any full run of `experiments/rq2_time_to_cross.py`. Changes after the freeze go in `docs/DECISIONS.md`, and affected runs are redone."
- [ ] **Step 2:** Check that no full-run results exist yet: `ls results/rq2_*` should show nothing, or only directories from `--quick` runs, which used only q₀ ≥ 10⁻⁴ and W1.
- [ ] **Step 3:** Commit: `docs(rq2): freeze the time-to-cross pre-registration`, then `git tag rq2-frozen`.

---

### Task 9: Full run and the written report

**Files:**
- Create: `docs/rq2-time-to-cross.md`, `docs/rq2/` (copied tables, figures and verdicts)
- Modify: `README.md`, `docs/DECISIONS.md`

- [ ] **Step 1:** Run all four subcommands without `--quick` in the background, with `--root results`. Then run `python experiments/rq2_time_to_cross.py report`. Record the wall time of each as a DECISIONS entry.
- [ ] **Step 2:** Check that every `manifest.json` under `results/rq2_*` has `"dirty": false` and the commit tagged `rq2-frozen` or a later code-free commit. If not, rerun from a clean tree.
- [ ] **Step 3:** Copy `results/rq2_report/` to `docs/rq2/`, and write `docs/rq2-time-to-cross.md`. Numbers come only from those files, with negative and inconclusive verdicts as prominent as positive ones. Present P1 as a confirmed derivation. Compare W2 (overlapping routes) with W1 and W3 (disjoint). Link the sampled results to RQ3's training-budget finding in `docs/report.md` §4.3 (i). Include a limitations section: toy worlds, exact PG only for the main arms, and the 4/3 factor's dependence on |A| = 4.
- [ ] **Step 4:** Update the README's status paragraph, which says RQ2's time-to-cross curve is still to do, and add layout rows for `src/capsplit/crossing.py`, `experiments/rq2_time_to_cross.py` and the three new docs.
- [ ] **Step 5:** Run `pytest -v`. Expected: all tests pass.
- [ ] **Step 6:** Commit: `docs(rq2): add the time-to-cross report and results`.
- [ ] **Step 7:** Ask the user before pushing the branch and tag and opening a PR. The README's workflow requires one teammate's review before merging, and the `docs/proofs.md` §7 derivation needs its own second-person check.
