# Harness task list

Status values: todo, in progress, verifying, done, failed. A task is **done** only when its definition of done is met, its tests pass, and a different subagent has verified it.

| Task | Depends on | Owner files | Definition of done | Status |
|---|---|---|---|---|
| T0 Interfaces | — | `harness/interfaces.py`, `registry.py`, `sequence.py`, `dummy.py`, `policies/tabular.py`, `trainers/none.py`, `tests/test_harness_interfaces.py` | Documented, type-checked (mypy clean) interfaces; trivial end-to-end test passes with dummy plug-ins | done (verified; 2 registry bugs found and fixed) |
| T1 Harness core | T0 | `harness/config.py`, `runner.py`, `cli.py`, `__main__.py`, `configs/dummy.json`, `tests/test_harness_runner.py` | Dummy config runs end to end from the CLI and regenerates identical outputs on a second run | done (verified: outputs byte-identical, J re-derived to 2e-16; 3 minor bugs fixed) |
| T2 Exact backend | T0 | `harness/backends/exact.py`, `tests/test_harness_exact.py` | Reproduces `evaluate` on the existing planted cases; Proposition 1 holds to float tolerance in a test | verifying |
| T3 Base model | T0 | `harness/tasks/rule_task.py`, `harness/policies/prompt_gru.py`, `harness/bases/pretrain.py`, `experiments/verify_base.py`, `docs/base_verification.md`, `tests/test_harness_base.py` | Committed table shows criteria a–e hold at every shared-structure level and across seeds | in progress |
| T4A RL arms (RLOO, GRPO, entropy bonus) | T0 | `harness/trainers/rl.py`, `tests/test_harness_rl.py` | Unit tests incl. zero gradient when every sample fails | done (verified: updates re-derived by hand, params match to 1e-7) |
| T4B SFT arm | T0 | `harness/trainers/sft.py`, `tests/test_harness_sft.py` | Unit tests; trains on hard train prompts only | done (verified: updates re-derived by hand, params match to 1e-7) |
| T4C Hint arm | T0 | `harness/trainers/hints.py`, `tests/test_harness_hints.py` | Unit tests; ≥2 hint lengths; no hints at evaluation | done (verified: updates re-derived by hand, params match to 1e-7) |
| T4D SFT then RL | T4A, T4B | `harness/trainers/sft_rl.py`, `tests/test_harness_sft_rl.py` | Unit tests | verifying |
| T4E Controls | T0, T4A | `trainers/none.py` (T0), entropy variant in `rl.py` | Test: no-post-training control gives exactly zero gain | verifying |
| Gate 1 | T1–T4 | `docs/preregistration.md`, `tests/test_harness_planted.py` | Full suite passes; base frozen; planted sharpening + capability cases labelled correctly; preregistration committed | todo |
| T5 Experiment | Gate 1 | `configs/main.json`, results | Every run has a manifest; grid regenerates from one command | todo |
| T6 Report generator | T1, T2 | `harness/report.py`, `tests/test_harness_report.py` | Report builds from T5 results with no manual steps | verifying |
| Gate 2 | T5, T6 | `docs/gate2.md` | Independent re-derivation of 3 headline numbers matches; H1–H4 verdicts | todo |
| T7 Generality | T1 | `harness/tasks/grid.py`, `harness/trainers/tabular.py`, `harness/backends/sampled.py`, `configs/grid_example.json`, tests | Second config and sampled backend run from the CLI; sampled matches exact within CIs | in progress |
| T8 Docs | Gate 2, T7 | `docs/report.md`, `docs/harness.md`, `docs/explainer.md`, `docs/slides.md` | A fresh subagent adds a trivial trainer by following harness.md | todo |
