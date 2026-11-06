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

Early Phase 1. The core split, `E_N`, the 2×2 label and exact 4×4 grid enumeration
are implemented. Training, the sequence task and the sample estimator are not built
yet. The metric definitions may change as the team ratifies the proposals in the
blueprint.

## Setup

Requires Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
python experiments/phase1_rq1.py
```

## Layout

| Path | Contents |
|---|---|
| `src/capsplit/core.py` | Split, `E_N`, pass@N and the 2×2 label |
| `src/capsplit/worlds.py` | Grid worlds with exact trajectory enumeration |
| `experiments/` | Scripts that reproduce results with fixed configurations |
| `tests/` | Unit and regression tests |
| `docs/project-plan.pdf` | Research plan: definitions, propositions and hypotheses |
| `docs/blueprint.md` | Technical blueprint: design, schedule and decisions to ratify |

## Workflow

Branch per issue (`feat/...`, `exp/...`), open a PR, one teammate reviews, then merge
to `main`. Commit small and often; the commit history is graded.
