# capsplit

**Separating learned capability from pre-existing reliability in reinforcement learning.**
DSAN 6650 final project.

## Abstract

When a trained policy beats its starting policy, the gain can have two sources:
the policy may produce more reliably what it already produced occasionally, or it
may reach what it effectively could not reach before. Mean return, success rate and
pass@k rise under both, yet the two have opposite implications, because reliability
gain may be reproducible by sampling the original policy more carefully, while
capability gain means training added something new.

This project defines a measurable split of the change in success probability,
J(π₁) − J(π₀), into a reliability term (`Δ_rel`) on correct solutions the starting
policy could reach within a stated sampling budget N, and a capability term
(`Δ_cap`) on those it could not. The two terms sum exactly to the total change. We
pair the split with a budget-matched excess score, `E_N`, which compares the trained
policy with the best re-tempered version of the starting policy across per-step and
sequence-level tempering. Together they place each improvement in one of four cells:
pure sharpening, compounded sharpening, reweighting among reachable solutions, or
capability that no sampling baseline can reproduce.

We validate the measure in small grid worlds where trajectories can be enumerated
exactly and the true answer can be planted, then apply it to a verifiable-reward
sequence task with a controlled amount of structure shared between common and rare
solutions. We also measure how often a pass@k decision rule misdiagnoses a run. The
project does not claim to settle whether RL expands capability in large language
models; it supplies a measurement with proven properties, validated where the answer
is known.

## Status

As of 2026-11-24. The final report is not written yet.

**Done**

- **Proofs.** [docs/proofs.md](docs/proofs.md) is frozen at tag `proofs-frozen`. A teammate
  re-derived sections 1 to 6; section 7 (time to cross) was checked and signed off by Bilal.
- **RQ2, time to cross.** Pre-registered and run; results in
  [docs/rq2-time-to-cross.md](docs/rq2-time-to-cross.md).
- **RQ3, Phase 2 (sequence task).** Pre-registered and run with the experiment harness
  (`src/capsplit/harness/`), 240 runs at 10 seeds; results in [docs/report.md](docs/report.md).
- **RQ4, pass@k misdiagnosis.** Pre-registered and run in both phases, with SRR/NDR scores;
  results in [docs/rq4-passk.md](docs/rq4-passk.md).
- **Label rule decided.** H12/H39 is the primary 2×2 label and D2 is reported alongside it
  (`docs/DECISIONS.md`, F-1). RQ3 Phase 2 stays at 10 seeds, a disclosed deviation from the
  blueprint's 20 (F-2).
- **Code pre-review.** [docs/notes/code-review-metrics.md](docs/notes/code-review-metrics.md)
  found no bug that changes a reported number.

**Still to do**

- **RQ1 final sweep.** Only the pilot run exists ([docs/pilot-report.md](docs/pilot-report.md)).
  `experiments/pilot_rq1.py` must record the H12 label as well as D2 first (review finding M1).
- **RQ2 training dynamics and RQ3 in the tabular grid worlds (Phase 1).** Not run.
- **One command that regenerates every report figure** from a fresh clone.
- **The final report**, covering RQ1 to RQ4.

The pilot report is kept as the record of the pilot and is not updated.

## Setup

Requires Python 3.10 or later. The `deep` extra adds `torch`, which the sequence task
training and the pilot runner need.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,deep]"
pytest
python experiments/run_pilot.py --quick    # about 2 minutes; drop --quick for every trial
```

Raw runs go under `results/`, which is git-ignored. Outputs the docs cite are
copied to the tracked `results/published/`.

## Layout

| Path | Contents |
|---|---|
| `src/capsplit/core.py` | Original split, `E_N`, pass@N and the 2×2 label |
| `src/capsplit/engine.py` | Compiled trajectory tables and exact probabilities |
| `src/capsplit/tempering.py` | Sequence tempering and the refined frontier |
| `src/capsplit/evaluate.py` | `evaluate`, `evaluate_benchmark` and the D2 label |
| `src/capsplit/worlds.py` | Grid worlds with exact trajectory enumeration |
| `src/capsplit/planted.py` | Frozen RQ1 planted-case generators |
| `src/capsplit/train/` | Tabular policy gradient and sequence RLOO |
| `src/capsplit/seqtask/` | Sequence task, policy network and pre-training |
| `src/capsplit/estimator.py` | Sampled split and sample-size bounds |
| `src/capsplit/diagnostics.py` | The pass@k rule and its scoring |
| `src/capsplit/runlog.py` | Run directories and manifests |
| `src/capsplit/crossing.py` | Time to cross, log-log slopes, and the isolated-route scalar model (RQ2) |
| `src/capsplit/harness/` | Plug-in experiment harness: tasks, policies, trainers, backends, runner, report |
| `configs/` | Harness experiment configs (`main.json` is the RQ3 Phase 2 run) |
| `experiments/` | Scripts that reproduce results with fixed configurations |
| `experiments/rq2_time_to_cross.py` | RQ2 time-to-cross runs (`arm-a`, `robust`, `arm-b`, `sampled`) and `report` |
| `tests/` | Unit and regression tests |
| `docs/project-plan.pdf` | Research plan: definitions, propositions and hypotheses |
| `docs/blueprint.md` | Technical blueprint: design, schedule and decisions to ratify |
| `docs/pilot-blueprint.md` | Pilot plan and task specifications |
| `docs/pilot-report.md` | Pilot results and recommendation |
| `docs/proofs.md` | Proofs, checked and independently re-derived |
| `docs/report.md` | RQ3 Phase 2 results (not the final report) |
| `docs/preregistration.md` | Frozen RQ3 Phase 2 hypotheses and decision rules |
| `docs/harness.md` | Harness developer guide |
| `docs/rq2-time-to-cross-design.md` | RQ2 time-to-cross design |
| `docs/rq2-preregistration.md` | Frozen RQ2 time-to-cross predictions and decision rules |
| `docs/rq2-time-to-cross.md` | RQ2 time-to-cross results |
| `experiments/rq4_passk.py` | RQ4: score the pass@k rule against the 2×2 label (`labels`, `score`, `srr-ndr`, `srr-ndr-report`, `repro`) |
| `configs/rq4_grid.json` | RQ4 Phase 1 grid sweep with the shared tabular policy (`tabular_grid_shared`) |
| `docs/rq4-passk-design.md` | RQ4 design |
| `docs/rq4-preregistration.md` | Frozen RQ4 rule settings and decision rules |
| `docs/rq4-passk.md` | RQ4 results |

## Workflow

Branch per issue (`feat/...`, `exp/...`), open a PR, one teammate reviews, then merge
to `main`. Commit small and often; the commit history is graded.
