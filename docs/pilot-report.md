# capsplit pilot report

> **Status: draft for the Nov 15 review.** The recommendation below is a proposal, and the go, change or cut decision belongs to the team.

**Result:** six of the eight exit criteria are met, one is partly met and one is not. The pipeline works from task to label, and the sweep is cheaper than estimated, but the current run designs can't yet answer RQ4 or say much about capability gain in Phase 2. **The proposed recommendation is to change the sweep before running it.**

| # | Exit criterion | Result |
|---|---|---|
| 1 | `pytest` passes on a clean clone, and CI runs on every PR | **Partly met.** `pytest` passes and CI passes on pushes, but CI hasn't run on a PR yet |
| 2 | `run_pilot.py --quick` regenerates every result from a clean clone | **Met**, 109 s |
| 3 | RQ1: gated families land in all 3 sizes, misses listed and explained | **Met**, with 5 expected misses listed |
| 4 | RQ2: isolated slope in range, and the competing table exists | **Met**, slope −0.817 |
| 5 | The estimator meets its bound in at least 95% of repeats | **Met**, 100% on both pairs |
| 6 | Propositions 1 to 3 written up and re-derived by a second teammate | **Not met.** Written, but nobody has checked them |
| 7 | Sequence task and RLOO use the same `evaluate`, with measured cost | **Met** |
| 8 | The report gives a go, change or cut recommendation | **Met, as a proposal** |

**How these numbers were produced:** a clean clone of commit `c901016` on Python 3.12, numpy from a fresh install, torch 2.15.0, on an Apple Silicon (arm64) laptop. All 14 run manifests record that commit with `dirty: false`. To reproduce them:

```bash
pip install -e ".[dev,deep]"
python experiments/run_pilot.py
```

## How the pilot was run

The blueprint split this work across four people. **One person, Bilal, working with a coding agent, completed all of it instead.** So these parts of the plan weren't followed:

- **File ownership and review pairs:** neither applied, and only F1 went through a PR.
- **PR reviews:** no teammate approved any task. F1 merged without Muhammad's review, and the other tasks went straight to `main`.
- **No fresh code review:** no independent reviewer has read this code, so the team should treat it as unreviewed.
- **Agent-written material:** the proofs in `docs/proofs.md` and the SRR/NDR note in `docs/notes/invisible-leash-scores.md` are unverified drafts. Both need a person to check them.

## Exit criteria

### 1. Tests and CI

- **Clean clone:** 217 tests pass with `torch` installed.
- **CI:** it passed on the pushes to `main` that were checked (`0c8ed1e`, `50f4bf1`, `2cca2f1`). The workflow also triggers on PRs, but no PR has opened since it was added, so that hasn't been exercised.
- **Gap:** CI installs `.[dev]` only, so it skips `tests/test_seq_training.py` and the 3 test functions in `tests/test_integration.py` that need `torch`. Nothing checks the Phase 2 code automatically.

### 2. Quick run

| Experiment | `--quick` | Full |
|---|---|---|
| `pilot_smoke` | 0 s | 0 s |
| `pilot_rq1` | 32 s | 258 s |
| `pilot_rq2` | 1 s | 3 s |
| `pilot_estimator` | 0 s | 0 s |
| `pilot_seq` | 49 s | 46 s |
| `pilot_rq4` | 27 s | 79 s |
| **Total** | **109 s** | **386 s** |

The limit was 10 minutes. `pilot_seq` has no quick mode, because a full run is already fast.

### 3. RQ1: planted cases

The generators were committed (`d4068cc`) before any sweep ran, and weren't changed afterward. One caveat is that the 5-seed 4×4 landing test had already passed by then.

| Case | 4×4, N=20 | 5×5, N=50 | 6×6, N=100 |
|---|---|---|---|
| `a_per`, `a_seq`, `ap_per`, `ap_seq`, `b_direct`, `c_direct`, `c_param` | 50/50 | 20/20 | 8/8 |
| `b_param` | 50/50 | 20/20 | **3/8** |

These counts match the prototype's exactly.

**All 5 misses** are 6×6 `b_param`, labelled `capability` instead of `reweighting`:

| Seed | `delta_cap` | $J$ gap |
|---|---|---|
| 2 | +0.0168 | +0.6490 |
| 3 | +0.0138 | +0.6476 |
| 4 | +0.0124 | +0.6454 |
| 6 | +0.0162 | +0.6509 |
| 7 | +0.0110 | +0.6438 |

**Explanation:** moving mass between reachable routes leaks a little onto other routes, and at $N=100$ the $c/N = 0.01$ threshold is low enough for that leak to pass. The blueprint predicted this (0.011 to 0.017) and said to report it, not fix it.

**Threshold sensitivity.** The landing counts below are for $c = 1$, where the columns set the reach threshold:

| Case | Size | Threshold 0 | $0.1\,c/N$ | $c/N$ |
|---|---|---|---|---|
| `b_param` | 4×4 | 10/50 | 50/50 | 50/50 |
| `b_param` | 5×5 | 0/20 | 1/20 | 20/20 |
| `b_param` | 6×6 | 0/8 | 0/8 | 3/8 |

Across the wider sensitivity table:

- **`c` of 3 or 5:** `b_param` and `b_direct` land 0 at every size. Their routes sit at about $1.2/N$, so a larger $c$ moves those routes below the line, and the gain then counts as beyond reach.
- **`c_param` and `c_direct`:** they land in every trial at every setting.
- **`a_per`:** at $c=5$ with the $c/N$ threshold, it lands 36/50, 20/20 and 4/8.

### 4. RQ2: what training changes

**Arm A, isolated route, at lr=1:**

| $q_{\pi_0}$ | $10^{-5}$ | $10^{-4}$ | $10^{-3}$ | $10^{-2}$ |
|---|---|---|---|---|
| Blueprint | 17,837 | 2,358 | 362 | 63 |
| Measured updates to $J \ge 0.1$ | 17,842 | 2,358 | 362 | 63 |

The log-log slope is **−0.817**, inside −1.0 to −0.7. `--quick` omits $10^{-5}$ and gets −0.787, so quote the full run.

**Arm B, competing routes, at lr=2 for 6,000 steps:**

| Bonuses (A, B) | $q_{\pi_0}(B)$ | Peak | Final | Crossed $10 \times q_{\pi_0}$ |
|---|---|---|---|---|
| (0, 0) | 0.00391 | 0.49977 | 0.5 | Update 54 |
| (1, 0) | 0.00273 | 0.00273 | $4.1 \times 10^{-7}$ | Censored |
| (2, 0) | 0.00150 | 0.00150 | $3.7 \times 10^{-7}$ | Censored |
| (3, 0) | 0.00068 | 0.00068 | $3.6 \times 10^{-7}$ | Censored |
| (3, 1) | 0.01177 | 0.01177 | $4.3 \times 10^{-6}$ | Censored |
| (3, 2) | 0.09019 | 0.09019 | $3.9 \times 10^{-5}$ | Censored |
| (2, 1.5) | 0.06945 | 0.07045 | $2.5 \times 10^{-5}$ | Censored |

Only the symmetric start crossed. In every uneven start, the rarer route lost nearly all its mass.

**Dynamics, 4×4 trap base, $N=20$:**

| Update | Exact PG `delta_rel` | Exact PG `delta_cap` | REINFORCE `delta_rel` | REINFORCE `delta_cap` |
|---|---|---|---|---|
| 10 | +0.2323 | +0.0035 | +0.1068 | +0.0017 |
| 50 | +0.6839 | +0.0032 | +0.5412 | +0.0051 |
| 400 | +0.8656 | −0.0011 | +0.8588 | −0.0010 |

REINFORCE values are the mean of 5 seeds. `delta_rel` passed $c/N = 0.05$ by update 4 for exact PG and update 6 for the REINFORCE mean, while `delta_cap` never did.

**Against RQ2's prediction:** early gain was mostly in `delta_rel`, with `delta_cap` absent, as predicted. This is one small world and base with 5 seeds, so it's consistent with the prediction but doesn't establish it. `excess` also sticks at +0.0582 from about update 25, so its size says little after that.

### 5. Estimator

| Pair | Exact means [reach $\pi_0$, unreach $\pi_0$, reach $\pi_1$, unreach $\pi_1$] | $M = 38{,}064$ (Chernoff) | $M = 380{,}640$ |
|---|---|---|---|
| Reference | 0.25019, 0.01281, 0.09489, 0.49799 | 200/200 | 200/200 |
| Trained (REINFORCE on the trap base) | 0.11062, 0.00198, 0.96445, 0.00109 | 200/200 | 200/200 |

Settings are $\mu = 0.01$, $\eta = 0.2$, $\delta = 0.05$. Hoeffding at $\varepsilon = \eta\mu$ would need 634,397 samples, 16.7 times the Chernoff size. That target only matches for means at or below $\mu$, so the ratio isn't a general saving.

**The check can fail.** On a separate check outside the pilot run, coverage on the trained pair fell to 83.5% at $M=1{,}000$ and 37% at $M=200$. So the Chernoff size looks conservative here.

### 6. Proofs

`docs/proofs.md` covers all seven results, with numeric checks that match the blueprint's reference values. **No person has checked it, and no teammate has re-derived it, so Gate 1 is still open.**

Writing the proofs turned up three points:

1. The corollary needs $q_{\pi_0}(s) > 0$, which the blueprint leaves out.
2. The plan's "relative error η on all four means" is false below $\mu$. D11 already fixes this.
3. Proposition 3's "same curve means same multiset" uses every $k$ from 1 to $P$. RQ4's grid skips $k = 3, 4$ at $P = 4$, so RQ4 can't cite that part of the proposition for its grid.

### 7. Sequence task and cost

Results at $V=6, L=5, N=50$, with 5 seeds per level, 600 updates and one prompt (task seed 0):

| $k$ | $J_0$ | $J_1$ | Rare $q_{\pi_0}$ | Rare $q_{\pi_1}$ | `delta_rel` | `delta_cap` | Label |
|---|---|---|---|---|---|---|---|
| 0 | 0.258–0.259 | 0.998–0.999 | 0.00388–0.00412 | $7 \times 10^{-8}$ to $8 \times 10^{-5}$ | About +0.752 | −0.011 to −0.012 | `pure_sharpening`, 5/5 |
| 2 | 0.258–0.259 | 0.999 | 0.00391–0.00410 | $3 \times 10^{-6}$ to $2 \times 10^{-5}$ | About +0.752 | About −0.012 | `pure_sharpening`, 5/5 |
| 4 | 0.258 | 0.999 | 0.00392–0.00395 | $4 \times 10^{-6}$ to $4 \times 10^{-4}$ | About +0.752 | −0.011 to −0.012 | `pure_sharpening`, 5/5 |

- **Against the prototype:** $J_0$, rare $q_{\pi_0}$ and $J_1$ match. Rare mass fell in all 15 runs. However, the prototype fell below $10^{-4}$ in all 6 of its runs, and 3 of these 15 ended above that, all at $k=4$.
- **Tabular control:** it's `pure_sharpening` at every level, but its rare $q_{\pi_0}$ is 0.00107, 0.00128 and 0.00412, so only $k=4$ is comparable with the network runs.
- **Every result here is `pure_sharpening` by construction.** The pre-training weights make the most likely sequence correct, so sequence tempering alone reaches $J = 0.9994$. At $k = 0, 2, 4$ (policy seed 0), the best $\beta$ is 8, the top of the range, so `boundary_hit` is set. **This leaves RL at most 0.0006 of $J$ to beat the frontier, so a beyond-temperature result is close to impossible here**, which is the main finding for Phase 2.

**Measured cost at $V=8, L=6, G=16$:**

| Prompts | ms per update | With `evaluate` each update | Node logits | Frontier | Compile |
|---|---|---|---|---|---|
| 1 | 1.1 | 23.8 | 0.012 s | 0.84 s | 0.009 s |
| 4 | 4.3 | 94.9 | 0.048 s | 3.38 s | 0.036 s |

The update time was checked by hand: the weights move, $J$ rises from 0.25 to 0.96 within 50 updates, and the rate holds over 500 more. The prototype's 43 ms isn't reproduced, and we don't know what it included. Each prompt here has its own policy, run one after another, because the interface has no way to condition one network on a prompt.

### 8. Recommendation

See [the recommendation](#recommendation-for-the-phase-2-sweep) below.

## RQ4: pass@k rule

$k_{\text{large}} = 10$ was committed before the first run. The rule's $k$ grids stop at the largest valid $N$: 40 on the grids and 77 on the sequence task.

| Cell | Runs | Primary agreement | Secondary | Undetermined |
|---|---|---|---|---|
| `capability` | 10 | 1.000 | 1.000 | 0 |
| `compounded_sharpening` | 10 | 0.000 | 0.000 | 0 |
| `reweighting` | 10 | 0.000 | 0.000 | 0 |
| `pure_sharpening` | 9 | 0.000 | 0.000 | 0 |
| **All** | **39** | **0.256** | **0.256** | **0** |

**Don't quote 25.6% as a misdiagnosis rate.** The rule said `capability` in all 39 runs, because training raised $J$ on every prompt in every run. With no prompt falling, the curves can't drop below the base, so the rule never says `sharpening`. The agreement figure is therefore just the share of `capability` runs, which comes from how the runs were mixed.

What this does show is that all 29 runs in the other three cells were called `capability` by this rule, and those cells are the ones where RQ4 predicted disagreement. It's a real illustration, but it isn't an estimate of the rate.

**A related gap:** `evaluate_benchmark` labels from $E_N$, which came out 0.0000 in all 10 `grid weak` runs and all 9 sequence runs. In the `grid weak` runs, pass@20 saturates at $J_1 \approx 0.97$. `BenchmarkReport` has no `saturated` flag, although D2 asks for one, so these labels rest on a value that may have rounded away.

## Cost of the full sweep

These are extrapolations for 400 runs of 2,000 updates, or 800,000 updates in all, using the measured rates above. The 400-run sweep itself wasn't run.

| | 1 prompt | 4 prompts |
|---|---|---|
| Blueprint estimate | 9.6 CPU-hours | About 38 |
| Training only | **About 0.24** | **About 1** |
| With `evaluate` at every update | **About 5.3** | **About 21** |

These figures leave out calibration, frontier computation and any multi-prompt network. They come from one laptop running one job. **Evaluating less often is the largest saving available.** Evaluating at checkpoints would keep the cost close to the training-only row.

## Provisional decisions

Each recommendation below is a proposal for the team to ratify.

| Decision | Pilot evidence | Recommendation |
|---|---|---|
| **D1:** sequence tempering on trajectories | Changes the trap base's best $J$ from 0.1147 to 0.1325, and its $E_{20}$ for the boost-rare case from +0.0875 to +0.0582 | **Keep.** It matches Definition 3 |
| **D2:** reach threshold $c/N$ | `b_param` lands 10/50, 0/20 and 0/8 at threshold 0, and 50/50, 20/20 and 3/8 at $c/N$. The 6×6 misses show the threshold shrinking too far at large $N$ | **Keep for now, and discuss.** It clearly beats the sign rule, but the 6×6 misses are unresolved. Don't retune it without declaring the change, and keep reporting the sensitivity table |
| **D3:** $\beta$ grid plus refinement | Ran throughout. `boundary_hit` is set on the sequence task, at $\beta = 8$ | **Keep,** and report boundary hits in results, since `pilot_seq` doesn't save them yet |
| **D4:** loop-erased routes | Every pilot world uses the shortest $H$, where the maps agree. Only unit tests exercise the difference | **Keep, but untested.** Longer $H$ runs would be the real test |
| **D5:** benchmarks with a shared $\beta$ | Ran on 39 benchmarks. `_best_mean_pass` searches the grid only, without refinement | **Keep,** and add a `saturated` flag |
| **D6:** isolated and competing arms | Slope −0.817, and only the symmetric start crossed | **Keep** |
| **D7:** invalid $(H, N)$ flag | Every pilot pair is valid, so only tests exercise it | **Keep** |
| **D8:** frozen generators | The frozen sweep reproduced the prototype, and its misses were reported rather than tuned away | **Keep** |
| **D10:** pass@k rule and scoring | The rule said `capability` in 39 of 39 runs | **Keep the rule, and change the run mix** (see below). Changing the rule now would be tuning it to these results |
| **D11:** bound $\eta \max(m, \mu)$ | 100% coverage on both pairs, including means below $\mu$. Coverage falls as expected at small $M$ | **Keep** |

**Outside the provisional list:** $q_{\pi_0}$ stayed matched across $k$ (0.0039 to 0.0041), which supports D13's calibration approach.

**Not done in the pilot:** the blueprint's section 6.2 asks to check Proposition 2 on grid trajectories during RQ1. The sweep doesn't do this. It was only checked in a throwaway script for the proofs, and it passed.

## Recommendation for the Phase 2 sweep

**Proposed: change, then run.** Cost is no longer the obstacle, since even evaluating every update needs about 21 CPU-hours. The problem is that the current design would mostly confirm what's built into it.

**Change these before the sweep:**

1. **Fix the sequence base, which matters most.** Its most likely sequence is correct, so sharpening alone reaches $J \approx 1$, which leaves almost no room to beat the frontier. Phase 2 needs a base where sharpening doesn't solve the task, as the grid's trap base does, or it will almost always report `pure_sharpening`.
2. **Widen RQ4's run mix.** Include runs where some prompts get worse, such as early-stopped runs or training that concentrates on a subset of prompts. Without these the rule has no chance to say `sharpening`. Freeze the mix before running.
3. **Add a `saturated` flag to `BenchmarkReport`,** as D2 asks.
4. **Evaluate at checkpoints** instead of at every update.
5. **Run the Phase 2 tests in CI,** at least on a schedule or on `main`.

**Required regardless of the sweep:**

1. A person checks `docs/proofs.md` step by step, and a second teammate re-derives Propositions 1 to 3. Gate 1 stays open until then.
2. A person checks the SRR/NDR note against the paper, and the team settles its four open points before RQ4 reports those scores.
3. The team reviews this code, or at least the metric and label code, since no teammate has.
4. The team decides on D2's 6×6 behavior before the labels are frozen.

**Why not go:** the sweep as designed could barely contradict the sequence-level predictions, and D8 exists to prevent exactly that kind of design.

**Why not cut:** the exact pipeline works, the planted cases land, the estimator holds and the cost is affordable.

**Limits of this evidence:** the pilot uses small worlds, few seeds and one prompt per sequence run outside RQ4. It tests the design and measures cost, and it doesn't produce the project's final results.
