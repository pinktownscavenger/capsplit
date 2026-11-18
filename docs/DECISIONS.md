# Harness decisions log

Each entry is one choice, with a one-line reason. Newest at the bottom. "T*" names the task that made it.

| # | Task | Decision | Reason |
|---|---|---|---|
| H1 | T0 | The harness lives in a new package `capsplit.harness`; no existing module changes meaning | The brief forbids changing teammates' functions |
| H2 | T0 | A prompt id is its index (0..P-1), and `Prompt.tags` (`difficulty`, `split`) defines measurement groups | Gives one simple, uniform way to group prompts for any task |
| H3 | T0 | Sequence state numbering follows the existing prefix tree (`node_offsets`), and `compile_prefix_tree` reproduces `compile_sequence` exactly (tested) | Reuses the exact engine unchanged |
| H4 | T0 | `Rollout.log_prob`/`entropy` count only generated steps; forced hint prefixes are excluded | The hint arm must not get gradient for tokens it didn't choose |
| H5 | T0 | A trainer's `budget` = number of optimizer updates, the same for every arm | "Same update budget" needs one unit; updates are what each arm controls |
| H6 | T0 | Plug-in modules self-register on import; the registry imports a fixed module list lazily and skips modules that don't exist yet | Lets tasks T1–T4 land in parallel without editing a shared file |
| H7 | T0 | Configs are JSON | No new dependency (Python 3.10 has no `tomllib`; YAML needs PyYAML) |
| H8 | T3 | Pre-train the base on **all** prompts (train and held-out); post-training only ever sees train prompts | The base is a model that "has seen" every prompt; held-out measures transfer of *post-training*, not of pre-training |
| H9 | T3 | Task design: prompts are (type, a, b). Easy prompts follow an "easy rule"; hard prompts follow the easy rule for the first k positions and a different "hard rule" after. k is the shared-structure dial | Shared structure means part of a hard solution is produced by a skill the base already has on easy prompts |
| H10 | T3 | Each prompt's most likely base sequence is an incorrect "trap" (the easy rule applied everywhere for hard prompts; a memorised wrong sequence for easy prompts) | Fixes the pilot's flaw: when the mode is correct, tempering alone reaches J ≈ 1 |
| H11 | T3 | Hard-solution base mass is calibrated per (k, seed) to a common target below c/N, following D13 | Keeps shared-structure levels comparable on base probability |
| H12 | T2 | Primary label rule: **beyond reach** iff at least one correct solution unreachable under the base becomes reachable under the trained policy (q1(s) ≥ c/N); **beyond temperature** iff E_N (both families) > 1e-9. The D2 rule (delta_cap ≥ c/N) is reported alongside | Leakage of small mass onto many unreachable solutions can pass D2 at large N without any solution becoming reachable (pilot 6×6 misses). Crossing is what "reachable at budget N" means |
| H13 | T2 | Group E_N uses one β per group and family, shared by the group's prompts, as in `evaluate_benchmark` (D5) | A single re-tempered base policy is the fair baseline; per-prompt β would be an oracle |
