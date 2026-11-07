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

The pilot is built: exact engine and metrics, planted cases, tabular and sequence
training, the sample estimator and the pass@k rule. Results and open issues are in
[docs/pilot-report.md](docs/pilot-report.md), which is a draft for team review. The
proofs in [docs/proofs.md](docs/proofs.md) have been checked once but still need a
teammate to re-derive them. The
metric definitions may change as the team ratifies the proposals in the blueprint.

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

Results go under `results/`, which is git-ignored.

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
| `experiments/` | Scripts that reproduce results with fixed configurations |
| `tests/` | Unit and regression tests |
| `docs/project-plan.pdf` | Research plan: definitions, propositions and hypotheses |
| `docs/blueprint.md` | Technical blueprint: design, schedule and decisions to ratify |
| `docs/pilot-blueprint.md` | Pilot plan and task specifications |
| `docs/pilot-report.md` | Pilot results and recommendation |
| `docs/proofs.md` | Proofs, checked once and awaiting re-derivation |

## Workflow

Branch per issue (`feat/...`, `exp/...`), open a PR, one teammate reviews, then merge
to `main`. Commit small and often; the commit history is graded.
