# capsplit: Learned Improvement vs. Pre-Existing Capability in RL

DSAN 6650 final project. When a trained policy beats its starting policy, how
much of the gain is **reliability** (more often producing what it could already
do) versus **capability** (reaching what it essentially could not)?

`capsplit` computes, for two policies, a verifier and a sampling budget N:

- `delta_rel`, `delta_cap`: exact additive split of J(pi1) - J(pi0)
- `E_N`: budget-matched excess over the best re-tempered base policy
- a 2x2 label: pure sharpening / compounded sharpening / reweighting / capability

Full plan: see `docs/` (project plan PDF).

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"
pytest
```

## Layout
- `src/capsplit/`  library (exact computation, worlds, training)
- `tests/`         unit tests (Propositions 1-3 are checked numerically here)
- `experiments/`   scripts that regenerate each figure (fixed seeds)
- `notebooks/`     per-phase analysis
- `docs/`          plan, report sources, notes

## Workflow
Branch per issue (`feat/...`, `exp/...`), open a PR, one teammate reviews, then
merge to `main`. Commit small and often; the commit history is graded.
