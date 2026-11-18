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
| H14 | T4B | Trainer params are validated: an unknown param raises | A typo in a config must fail loudly, not run a different experiment |
| H15 | T4B | SFT minibatches are drawn without replacement from one generator seeded by the run seed; default is full batch | Reproducible runs; full batch is the simplest same-budget comparison |
| H16 | T4C | A hinted group uses one demonstration prefix (seeded choice), and the leave-one-out baseline is computed within that hinted group | Episodes compared by the baseline share the same hint |
| H17 | T4C | Forced hint tokens never receive gradient, so a hint of length h ≥ 1 cannot train the first h tokens on that prompt | Training on tokens the policy didn't choose would be supervised learning, not RL. The report must state this limit |
| H18 | T4A | RL arms have no KL penalty to the base; GRPO-style arm takes one on-policy step per batch, so the PPO clip ratio is 1 | Keeps the arm comparison to the training signal alone; KL can be added as a param |
| H19 | T4A | RL losses average over all P·G episodes; one generator seeded by the run seed drives sampling and minibatching | Matches the existing train/sequence.py; one seed controls all randomness |
| H20 | T4 | Arms are matched on optimizer updates, not on samples: the hint arm draws an extra group of rollouts for each all-fail prompt, and SFT draws none. The report states samples per arm | The brief fixes the update budget; sample cost differs by design and is disclosed, not hidden |
| H21 | T1 | Cached bases are whole pickled Policy objects (torch.save, weights_only=False) keyed by sha256 of task+base params and seed | Restores any Policy class without knowing its constructor; base built once per (factors, seed), shared by every arm |
| H22 | T1 | torch runs single-threaded during a run, and torch.manual_seed(seed) is set before each base build and each arm | Serial and parallel runs, and cached vs fresh bases, agree to the bit |
| H23 | T1 | A run's metrics file is written last and atomically; its existence marks the run complete (re-runs skip it unless --force) | Interrupted sweeps resume safely |
| H24 | T1 | Training prompts = the `split=train` prompts; unknown config keys and trainer checkpoint mismatches raise | Held-out prompts are never trained on; typos and contract violations fail loudly |
