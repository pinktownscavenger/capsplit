# Experiment harness: developer guide

`capsplit.harness` runs post-training experiments and measures each trained policy against its
base: the success rate J, the split of the gain into **reliability** (`delta_rel`) and
**capability** (`delta_cap`), the excess over the best tempered base (E_N) and a 2x2 label.
This guide covers how the pieces fit together and how to add a trainer, a task or a config.
Design choices are logged in `docs/DECISIONS.md` (entries H1–H39 are referred to below; the
addendum features in sections 7–9 are H48–H51, and the language-model experiment in section 10
is L1–L19).

Every example below was run against this branch: the plug-ins in a scratch copy of `src/`
(with `PLUGIN_MODULES` edited as shown) and also as out-of-tree modules.

## 1. Overview

### The interfaces (`src/capsplit/harness/interfaces.py`)

| Interface | Role | Must implement |
|---|---|---|
| `Task` | Prompts, dynamics and a 0/1 verifier | `prompts`, `verify`, `initial_state`, `step`; optional `demonstrations`, `table` |
| `Policy` | pi(a \| s), holding torch parameters | `sample`, `log_prob`, `state_logits`, `parameters`, `state_dict`, `load_state_dict` |
| `Trainer` | A post-training arm (an update rule) | `train(policy, task, budget, seed, prompt_ids, checkpoints, on_checkpoint)` |
| `Backend` | Measures a base/trained pair | `prepare(task, base, spec)` (base-only cache), `measure(...) -> list[GroupMetrics]` |
| `BaseRecipe` | Builds the base policy pi_0 ("pre-training") | `build(task, seed) -> (policy, info_dict)` |

RL framing, which every implementation keeps:

- **State**: the prompt plus the actions so far (the prefix for sequence tasks; the cell for grid tasks).
- **Action**: one of `task.n_actions` choices.
- **Reward**: terminal and binary, `1` iff `task.verify(prompt_id, actions)`. So the **return** equals
  the reward, and a policy's expected return on a prompt is its success probability J.
- **Policy**: a distribution over actions per state (`sample`, `log_prob`, `state_logits`).
- **Update rule**: a `Trainer`; it changes the policy in place for exactly `budget` optimizer updates.

Each interface has a registry in `registry.py` (`TASKS`, `POLICIES`, `TRAINERS`, `BACKENDS`,
`BASES`). A config names plug-ins by their registered names.

### Data flow

```
config.json --load_config/parse_config--> ExperimentConfig
  runner: for each (factor combination, seed):              # a "unit"
      task    = TASKS.create(...)
      base    = cached bases/<key>/policy.pt, or BASES.create(...).build(task, seed)
      cache   = backend.prepare(task, base, spec)          # once per unit
      for each arm: policy = base.clone()
                    trainer.train(..., on_checkpoint=hook)  # hook calls backend.measure
                    write <cell>/seed-<s>.metrics.json (+ policies.pt)
  write index.json
report   (python -m capsplit.harness report)   -> <results>/report/{report.md,summary.json,tables/,figures/}
verdicts (python -m capsplit.harness verdicts) -> <results>/report/verdicts.{md,json}
```

Training prompts are the prompts tagged `split=train`; if no prompt has a `split` tag, all
prompts are training prompts (H24). The base is pre-trained on all prompts (H8).

### Results folder

```
<out>/<name>/
  config.json                     the input config (sorted keys)
  index.json                      every run; the report reads this
  bases/<key>/policy.pt           pickled base Policy (torch.save)
  bases/<key>/info.json           {"task", "base", "seed", "info"}
  <cell>/manifest.json            git commit, dirty flag, the cell config
  <cell>/seed-<s>.metrics.json    measurements of one run
  <cell>/seed-<s>.policies.pt     {step: state_dict} (absent if save_policies is "none")
```

A **cell** is one (factor combination, arm); its folder is the first 12 hex digits of the
SHA-256 of the cell config (name, factors, resolved task/base/backend, arm, budget,
checkpoints, measure). Change any of those and you get a new folder; adding seeds does not.
A base `<key>` hashes the resolved task and base components plus the seed, so all arms of a
unit share one base.

`index.json` (all paths relative to `<out>/<name>/`):

```
{"name", "config": "config.json", "factors": {key: [values]}, "arms": [{"name","trainer","params"}],
 "seeds", "budget", "checkpoints", "measure", "save_policies",
 "runs": [   # ordered by factor combination, arm, seed
   {"factors": {key: value}, "arm", "trainer", "seed", "run_dir": "<cell>", "base": "bases/<key>",
    "files": {"manifest": ..., "metrics": ..., "policies": ... | null}}]}
```

`seed-<s>.metrics.json` holds `arm, trainer, seed, factors, base, groups ({group: [prompt
ids]}), train_prompt_ids` and `records`: one `GroupMetrics.to_dict()` plus `"step"` per
(checkpoint, group, N, c), in checkpoint order.

## 2. Adding a trainer

### The contract

1. Call `on_checkpoint(0, policy)` **before any update**, then `on_checkpoint(k, policy)` right
   **after update k** for each k in `checkpoints` (use `call_checkpoints`). The runner raises if
   the steps called differ from the config's checkpoints. The hook must not modify the policy.
2. `budget` counts **optimizer updates**, not samples (H5, H20). Every arm gets the same budget.
3. Train only on `prompt_ids` (the train split). Using a subset is fine (e.g. hard prompts via
   `trainers.rl.select_prompts`); using any other prompt is not. Every learning signal must come
   from episodes or demonstrations on those prompts. Prompt-independent changes to the
   parameters (e.g. adding noise to every weight) are not allowed: with a tabular policy they
   would change held-out prompts directly. With a shared network, held-out prompts do change, but
   only through generalisation, which is what the held-out groups measure.
4. All randomness comes from `seed` (a `torch.Generator().manual_seed(seed)`), so a rerun is
   bit-identical.
5. Validate params: an unknown param raises (H14), so a config typo fails loudly.

### Example: `sgd_on_reward`

Plain REINFORCE (no baseline) with SGD. Save as
`src/capsplit/harness/trainers/sgd_on_reward.py`:

```python
"""`sgd_on_reward`: plain REINFORCE (no baseline) with SGD.

Each update samples `batch` episodes per training prompt, sets r = 1[task.verify], and takes
one SGD step on  loss = -mean_j r_j * log pi(tau_j).
Params (unknown ones raise): batch (8), lr (0.5).
"""
from __future__ import annotations

import torch

from capsplit.harness import TRAINERS, Trainer, call_checkpoints
from capsplit.harness.trainers.rl import episode_rewards

_PARAMS = {"batch", "lr"}


@TRAINERS.register("sgd_on_reward")
class SGDOnReward(Trainer):
    def __init__(self, **params):
        unknown = set(params) - _PARAMS
        if unknown:
            raise ValueError(f"sgd_on_reward: unknown parameters {sorted(unknown)}")
        super().__init__(**params)
        self.batch = int(params.get("batch", 8))
        self.lr = float(params.get("lr", 0.5))
        if self.batch < 1:
            raise ValueError("sgd_on_reward: batch must be >= 1")

    def train(self, policy, task, budget, seed, prompt_ids, checkpoints, on_checkpoint):
        call_checkpoints(0, checkpoints, policy, on_checkpoint)    # step 0: before any update
        if budget <= 0:
            return
        ids = [int(p) for p in prompt_ids]                         # never train outside these
        generator = torch.Generator().manual_seed(int(seed))       # all randomness from seed
        optimizer = torch.optim.SGD(list(policy.parameters()), lr=self.lr)
        for step in range(1, budget + 1):                          # budget = optimizer updates
            rollout = policy.sample(task, [p for p in ids for _ in range(self.batch)], generator)
            rewards = episode_rewards(task, rollout)
            loss = -(rewards * rollout.log_prob).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            call_checkpoints(step, checkpoints, policy, on_checkpoint)  # after update `step`
```

`Rollout.log_prob` and `entropy` carry gradients and cover only generated steps (forced hint
prefixes are excluded, H4). Reusable helpers: `trainers.rl` (`episode_rewards`,
`group_advantages`, `rl_loss`, `select_prompts`) and `trainers.sft` (`demonstration_pairs`,
`sft_step`, `minibatch`).

### Registering a plug-in

`@TRAINERS.register(...)` only runs when the module is imported. The registries import the
modules listed in `registry.PLUGIN_MODULES` lazily, on first use; nothing else is imported.

**Recommended (in-repo plug-ins): add the module path to `PLUGIN_MODULES`** in
`src/capsplit/harness/registry.py`:

```python
PLUGIN_MODULES = (
    ...
    "capsplit.harness.bases.pretrain",
    "capsplit.harness.trainers.sgd_on_reward",
)
```

This is the only route that works everywhere: the CLI, config validation, and `--workers`
child processes (which start fresh and re-run the lazy import).

```
$ python -m capsplit.harness list
tasks: grid, pattern, rule_task, tiny_prompt
...
trainers: grpo, hints, none, rl_entropy, rloo, sft, sft_then_rl, sgd_on_reward, tabular_reinforce
```

Without registration the CLI fails at validation:
`error: task: unknown task 'pattern'; registered: grid, rule_task, tiny_prompt` (exit code 2).

**Alternatives.**

- *Side import from a listed module.* `tasks/grid.py` ends with
  `from ..bases import grid_base` and `from ..policies import tabular_grid`, which registers
  the grid base and policy whenever the grid task loads. Use this for helpers that belong to
  a listed module, not for unrelated plug-ins.
- *Out-of-tree (prototyping).* Import your modules yourself and call the Python API. Keep the
  run under `if __name__ == "__main__":`: with `workers > 1` the spawned workers re-import the
  script (which re-registers the plug-ins), but must not re-run the experiment.

  ```python
  """Run a config that uses plug-ins outside PLUGIN_MODULES: import them first."""
  import sys

  import plugins.pattern_task    # noqa: F401  registers "pattern"
  import plugins.sgd_on_reward   # noqa: F401  registers "sgd_on_reward"
  from capsplit.harness.runner import run_experiment

  if __name__ == "__main__":  # required: --workers > 1 spawns processes that re-import this file
      print(run_experiment(sys.argv[1], out_root="results", workers=int(sys.argv[2]) if len(sys.argv) > 2 else 1))
  ```

  `python run_pattern.py pattern.json 2` gives the same JSON output as the CLI with the
  in-repo registration.

### Unit test

`tests/test_harness_sgd_on_reward.py`. The tabular policy has one logit table per prompt and
shares nothing between them, so "prompts outside `prompt_ids` are untouched" is an exact check.
The dummy task `tiny_prompt` (`harness/dummy.py`) has V = 3, L = 3 and four prompts, tagged
(easy, train), (hard, train), (easy, heldout) and (hard, heldout) for ids 0 to 3. So its train ids are
[0, 1], and a matching tabular policy is `TabularSequencePolicy(4, 3, 3)`
(prompts, vocabulary, length).

```python
import pytest

torch = pytest.importorskip("torch")

from capsplit.harness import TRAINERS  # noqa: E402
from capsplit.harness.dummy import TinyPromptTask  # noqa: E402
from capsplit.harness.policies.tabular import TabularSequencePolicy  # noqa: E402


def _run(seed, ids=(0, 1)):
    task = TinyPromptTask()
    policy = TabularSequencePolicy(4, task.n_actions, task.horizon, init_scale=0.5, seed=0)
    calls = []
    hook = lambda step, p: calls.append((step, p.state_dict()["logits"].clone()))
    trainer = TRAINERS.create("sgd_on_reward", batch=4, lr=0.5)
    trainer.train(policy, task, budget=4, seed=seed, prompt_ids=list(ids), checkpoints=[0, 2, 4], on_checkpoint=hook)
    return calls


def test_checkpoints_in_order_and_step0_is_untrained():
    calls = _run(seed=0)
    assert [s for s, _ in calls] == [0, 2, 4]
    assert torch.equal(calls[0][1], TabularSequencePolicy(4, 3, 3, init_scale=0.5, seed=0).logits.detach())


def test_reproducible_from_seed():
    a, b = _run(seed=1), _run(seed=1)
    assert all(torch.equal(x, y) for (_, x), (_, y) in zip(a, b))


def test_only_trains_on_given_prompts():
    calls = _run(seed=0)
    first, last = calls[0][1], calls[-1][1]
    assert torch.equal(first[2:], last[2:])      # prompts 2, 3: not in prompt_ids, untouched
    assert not torch.equal(first[:2], last[:2])  # prompts 0, 1: trained


def test_unknown_param_raises():
    with pytest.raises(ValueError, match="unknown"):
        TRAINERS.create("sgd_on_reward", batchsize=4)
```

### Config arm

```json
{"name": "sgd", "trainer": "sgd_on_reward", "params": {"batch": 8, "lr": 0.5}}
```

The full config using it is in section 4.

## 3. Adding a task

For fixed-length token tasks, subclass `SequenceTask` (`sequence.py`). You supply
`n_actions`, `horizon`, `prompts()` and `correct_solutions(prompt_id)`; verification,
dynamics, demonstrations (the correct solutions) and the exact table follow. Prompt ids must
be `0..P-1` in order. Tags define training prompts (`split`) and measurement groups.

`src/capsplit/harness/tasks/pattern.py` (and add `"capsplit.harness.tasks.pattern"` to
`PLUGIN_MODULES`):

```python
"""`pattern`: binary strings of length L; prompt i accepts a small set of patterns."""
from __future__ import annotations

from capsplit.harness import TASKS, Prompt
from capsplit.harness.sequence import SequenceTask


@TASKS.register("pattern")
class PatternTask(SequenceTask):
    """Prompt 0/2 (easy): all-zeros or all-ones. Prompt 1/3 (hard): one alternating string."""

    def __init__(self, length: int = 4):
        self.n_actions, self.horizon = 2, int(length)
        tags = [("easy", "train"), ("hard", "train"), ("easy", "heldout"), ("hard", "heldout")]
        self._prompts = tuple(Prompt(i, (i,), {"difficulty": d, "split": s}) for i, (d, s) in enumerate(tags))

    def prompts(self):
        return self._prompts

    def correct_solutions(self, prompt_id):
        L = self.horizon
        if self._prompts[prompt_id].tags["difficulty"] == "easy":
            return ((0,) * L, (1,) * L)
        start = prompt_id // 2  # 0 for prompt 1, 1 for prompt 3
        return (tuple((start + t) % 2 for t in range(L)),)
```

Quick check:

```python
from capsplit.harness import TASKS

task = TASKS.create("pattern", length=4)
print(task.prompt_ids(difficulty="hard"))       # [1, 3]
print(task.verify(1, (0, 1, 0, 1)), task.verify(1, (1, 0, 1, 0)))  # True False
table = task.table(1)                            # exact enumerator from SequenceTask
print(table.index.shape, table.solutions)        # (16, 4) ((0, 1, 0, 1),)
```

Any `SequenceTask` works with the existing `uniform_tabular` base and `tabular_sequence`
policy (the `prompt_gru` policy uses the same prefix-tree state numbering). Real experiments
also need a `BaseRecipe` that gives the base a meaningful shape (see `bases/pretrain.py`).

**What each backend needs.**

- **exact**: `task.table(pid)` returning a `TrajectoryTable` (every trajectory, its state-action
  slots and solution id), and a policy whose `state_logits(task, pid)` uses the same state
  numbering (or overrides `trajectory_probs`). `SequenceTask` provides the table (V^L rows, so
  keep V^L small: the `rule_task` main experiment uses 6^5). A non-sequence task writes its own
  (the grid task uses `worlds.compile_grid`) plus a matching policy (`policies/tabular_grid.py`).
  For per-step tempering the policy's `tempering_logits` (default: `state_logits`) is used.
- **sampled**: only `Policy.sample` and `Task.verify`. Solution identity comes from
  `task.solution_of(pid, actions)` if defined (grid: the loop-erased route), else the table
  row, else the action sequence itself. Per-step E_N needs a policy with `tempered(beta)` or a
  torch `logits` Parameter; otherwise E_N is NaN.

## 4. Writing and running a config

```json
{
  "name": "pattern_demo",
  "task": {"type": "pattern", "params": {"length": 4}},
  "base": {"type": "uniform_tabular", "params": {"init_scale": 1.0}},
  "backend": {"type": "exact", "params": {}},
  "factors": {"base.init_scale": [0.5, 1.0]},
  "arms": [
    {"name": "none", "trainer": "none"},
    {"name": "sgd", "trainer": "sgd_on_reward", "params": {"batch": 8, "lr": 0.5}},
    {"name": "rloo", "trainer": "rloo", "params": {"group": 8, "lr": 0.05}}
  ],
  "seeds": [0, 1],
  "budget": 20,
  "checkpoints": [0, 10, 20],
  "measure": {"ns": [4, 16], "cs": [1], "group_by": ["difficulty", "split"],
              "primary": {"n": 16, "c": 1}},
  "save_policies": "final"
}
```

| Key | Meaning |
|---|---|
| `name` | Experiment folder name (letters, digits, `_ - .`) |
| `task`, `base`, `backend` | `{"type": registered name, "params": {...}}`; params go to the constructor |
| `factors` (optional) | `{"<section>.<param>": [values]}` with section `task`, `base` or `backend`. A longer dotted key sets a nested dict entry (`base.cfg.lr`). The grid is the cartesian product, keys sorted. Arm params are not factors: write separate arms |
| `arms` | `[{"name", "trainer", "params"}]`; names unique |
| `seeds` | Distinct non-negative ints; each seed builds its own base |
| `budget` | Optimizer updates per arm |
| `checkpoints` | Steps to measure; must include `0` and `budget` |
| `measure.ns`, `measure.cs` | Budgets N and multipliers c; reachability line is c/N |
| `measure.group_by` | Tag names. Groups: `all`, one per tag value, and one per full combination joined by `_`. The config above gives `all, easy, easy_heldout, easy_train, hard, hard_heldout, hard_train, heldout, train` |
| `measure.primary` (optional) | `{"n", "c"}` for headline numbers; must be in `ns`/`cs`. Default: largest N, c = 1 |
| `save_policies` (optional) | `"checkpoints"` (default), `"final"` or `"none"` |

Validation errors are raised as `ConfigError` (the CLI prints them and exits with code 2).
Examples from this config:

```
checkpoints [0, 10]               -> checkpoints must include 0 and budget=20
factors {"arm.lr": [0.1]}         -> factor key 'arm.lr' must look like task.<param>, base.<param> or backend.<param>
primary {"n": 64, "c": 1}         -> measure.primary must use an N in measure.ns and a c in measure.cs
trainer "sgd"                     -> arms[1] (sgd): unknown trainer 'sgd'; registered: grpo, hints, ...
"seed": [0]                       -> unknown config keys ['seed']
```

Trainer, task and backend params are checked by their constructors when the run starts
(not at config load).

### Commands

```
python -m capsplit.harness list                                  # registered plug-ins
python -m capsplit.harness run configs/x.json [--out results] [--workers 4] [--force]
python -m capsplit.harness report results/<name>                 # -> results/<name>/report/
python -m capsplit.harness verdicts results/<name>               # -> report/verdicts.{md,json}
python -m capsplit.harness budget-curve results/<name> [...]     # section 8
python -m capsplit.harness discovery {predict,analyze} ...       # section 9
```

- `run` prints the results folder. A run whose metrics file (and policies file, if required)
  exists is **skipped**, so an interrupted sweep resumes where it stopped (H23). `--force` reruns
  everything, rebuilds cached bases and rewrites manifests.
- `--workers N` runs (factor combination, seed) units in a spawn process pool. Results are
  identical to `--workers 1` (checked: byte-identical JSON).
- Run from the git checkout: manifests record the commit and dirty flag.
- `report` writes `report.md`, `summary.json`, `tables/*.csv` and `figures/*.png`
  (matplotlib), including validity checks: Proposition 1 residual, zero gain at step 0 and in
  the `none` arm, `saturated`, `boundary_hit`, `e_n_valid` (warns when V^L < 100·N, as in
  the toy config above) and sensitivity of conclusions to (N, c).
- `verdicts` applies **one specific pre-registered plan** (`docs/preregistration.md`,
  `prereg_v1`) for the main experiment (`configs/main.json`): it hard-codes the factor
  `task.shared_prefix` in {0, 2, 4}, the arms `rloo`, `grpo`, `sft`, `hints_h1`, `hints_h3`,
  `sft_then_rl` and tau0 = 0.02. On an experiment missing any of these it refuses with exit
  code 2 ("the pre-registered plan prereg_v1 does not apply"). Write a new plan module for a
  new experiment.

## 5. Measurement backends

Three backends are registered:
- `exact` enumerates every trajectory.
- `sampled` estimates from samples, with confidence intervals.
- `j_only` is a dummy for interface tests: it reports exact J and puts the whole change in
  delta_rel, with no split, no E_N and the label `"unmeasured"`. `configs/dummy.json` uses it, so
  its report shows "unmeasured" labels and n/a for E_N. Switch to `exact` for real numbers.

`exact` and `sampled` return one `GroupMetrics` per (group, N, c):

| Field | Meaning |
|---|---|
| `group`, `n`, `c`, `n_prompts` | Where it was measured |
| `j0`, `j1` | Mean success rate of base and trained policy over the group |
| `delta_rel`, `delta_cap` | Gain on solutions reachable / unreachable under the base: s reachable iff q0(s) ≥ c/N. `delta_rel + delta_cap = j1 - j0` exactly (Proposition 1) |
| `delta_total` (in `to_dict`) | `j1 - j0` |
| `excess` | E_N per tempering family `per_step`, `sequence`, `both`: mean pass@N(J1) minus the best mean pass@N of the tempered base, one β shared by the group (H13) |
| `label` | The 2x2 cell (below) |
| `extra` | Backend-specific details |

**Label rule (H12, H39).** *Beyond reach* iff some correct solution on some prompt of the
group crosses the line: q0(s) < c/N ≤ q1(s). *Beyond temperature* iff E_1 (`extra["e1"]["both"]`,
the same comparison at N = 1, i.e. on J) > 1e-9. Both → `capability`; reach only →
`compounded_sharpening`; temperature only → `reweighting`; neither → `pure_sharpening`.

**`exact`** (`backends/exact.py`; params `betas`, `refine`) enumerates `task.table`, so
numbers carry no sampling error. `prepare` computes the base's q0 and the tempering frontier
(β grid plus golden-section refinement) once. `extra` keys:
`e1` (per family), `label_en` (the label with E_N instead of E_1), `d2_label` (older D2 rule:
`delta_cap ≥ c/N` with the J gap), `beyond_reach`, `crossed_count` (solutions that crossed),
`crossed_mass` (mean q1 on them), `best_beta`, `best_family`, `boundary_hit`, `saturated`
(best tempered pass@N > 0.95), `frontier` (per family), `e_n_valid` (V^L ≥ 100·N),
`n_correct_solutions`, and per-prompt lists `j0`, `j1`, `delta_rel`, `delta_cap`.

**`sampled`** (`backends/sampled.py`; params `m`, `m_classify`, `m_temper`, `betas`, `seed`,
`n_boot`, `alpha`, `chunk`) is for tasks too large to enumerate. An independent sample of
`m_classify` base episodes decides reachability (keep `m_classify >> N/c`); fresh samples of
`m` episodes estimate the masses, with common random numbers so step 0 gives exactly zero.
Only per-step E_N is estimated; `sequence` and `both` are NaN (written as `NaN` in the JSON,
which Python's `json` reads), and the label uses per-step E_1 (`extra["label_basis"]`).
`extra` adds `ci` (two-level bootstrap over prompts and episodes, for j0, j1, delta_rel,
delta_cap), `ci_episodes` (episodes only, matching what `exact` computes), `crossed_count`,
`e1`, `label_en`, and the per-prompt lists. Example (pattern task, `sgd` arm, `hard_train`,
N = 16, with `"backend": {"type": "sampled", "params": {"m": 2048, "m_classify": 2048,
"n_boot": 200, "betas": [0.5, 1, 2]}}`):

```python
import json
import plugins.pattern_task, plugins.sgd_on_reward  # noqa: F401
from capsplit.harness.runner import run_experiment

out = run_experiment("pattern_sampled.json", out_root="results")
index = json.loads((out / "index.json").read_text())
run = next(r for r in index["runs"] if r["arm"] == "sgd")
metrics = json.loads((out / run["files"]["metrics"]).read_text())
rec = next(r for r in metrics["records"] if r["step"] == 20 and r["group"] == "hard_train" and r["n"] == 16)
print(rec["j0"], rec["j1"], rec["delta_rel"], rec["delta_cap"], rec["label"])
print(rec["excess"], rec["extra"]["ci"]["delta_rel"], rec["extra"]["crossed_count"])
# 0.2485... 0.4863... 0.2378... 0.0 reweighting
# {'both': nan, 'per_step': 0.000116..., 'sequence': nan} [0.2099..., 0.2626...] 0
```

## 6. Testing and reproducibility

- Tests live in `tests/test_harness_*.py`; run `python -m pytest tests -q` (or
  `-m "not slow"` to skip the `@pytest.mark.slow` ones). Start torch-dependent test files with
  `torch = pytest.importorskip("torch")`.
- Test plug-ins on the dummy pieces in `dummy.py` (`tiny_prompt` task, `uniform_tabular`
  base, `j_only` backend; `configs/dummy.json`) and the tabular policy, which makes
  "untouched prompt" and exact-J checks easy.
- For a trainer, test: checkpoint order and an untouched step-0 policy, same seed → identical
  parameters, no change on prompts outside `prompt_ids`, unknown params raise.
  `tests/test_harness_rl.py` and `test_harness_sft.py` are fuller models.
- For a task, test `verify` against hand-written cases and, if you write your own `table`,
  that it agrees with `verify` and `step`.
- Determinism the runner guarantees, and your plug-ins must not break: no timestamps; JSON
  with sorted keys and round-trip floats (reruns are byte-identical); `torch.manual_seed(seed)`
  before each base build and each arm; single-threaded torch during runs (H22), so serial and
  parallel runs agree. Don't use global RNGs (`random`, `np.random`) without seeding them
  from `seed`; prefer a local `torch.Generator`.
- A base is cached by (task, base, seed). If you change a base recipe's code without changing
  its params, rerun with `--force`, or the old pickled base is reused.

## 7. Separate seed streams and training logs (addendum B1, B2)

### `base_seed` and `train_seed` (H49)

By default one run seed `s` from `seeds` drives two things: the base build
(`torch.manual_seed(s)` before `BaseRecipe.build`) and the trainer (`seed=s`). Two optional
top-level config keys fix one of the two streams, so `seeds` varies only the other:

| Config | Base built with | Trainer seed | What varies across `seeds` |
|---|---|---|---|
| neither key (default) | s | s | both (original behaviour) |
| `"base_seed": 0` | 0 | s | training sampling only; every run shares one cached base |
| `"train_seed": 0` | s | 0 | the base only |

`configs/seedstudy_base.json` sets `"base_seed": 0` and `configs/seedstudy_train.json` sets
`"train_seed": 0` (30 run seeds each, `rloo`, k ∈ {0, 2, 4}). Values must be non-negative ints.

What changes in the outputs, and what does not:

- With either key set, each `index.json` run entry, each metrics file and each training log
  also records `base_seed`, `train_seed` (the effective seeds) and `measure_seed` (the backend's
  `seed` param; `null` for `exact`, which has no measurement randomness).
- The keys enter the cell config, and so the cell hash, **only when set**. A config without them
  produces byte-identical files to the code before the keys existed (checked on the frozen toy
  configs), and seed-separated results never share a folder with seed-coupled ones.
- The base cache key hashes the *effective* base seed, so with `base_seed` fixed all runs of a
  factor combination load one `bases/<key>/policy.pt`.

### Per-prompt training logs (H50, H51)

A trainer that defines `training_log() -> dict | None` gets its log written by the runner as
`<cell>/seed-<s>.training.json`, and `index.json` gains `files["training"]` for that run (the
key is absent when there is no log). The built-in `rloo`, `grpo`, `rl_entropy`, `hints`,
`sft` (an all-zero log: SFT samples nothing) and `sft_then_rl` provide one through
`trainers.rl.SampleLog`; `none` and `tabular_reinforce` do not. The file is
`{"arm", "trainer", "seed"}` (plus the effective seeds when the config separates them) merged
with the log:

```
{"updates": int,                          # optimizer updates the log covers (numbered 1..budget)
 "checkpoints": [int],
 "prompts": {"<prompt id>": {
     "first_success_update": int | null,  # first update whose sampled group held a correct episode
     "correct_by_checkpoint": {"<c>": int},   # correct sampled episodes in updates 1..c
     "episodes_sampled": int,                 # whole run
     "episodes_by_checkpoint": {"<c>": int},  # episodes sampled in updates 1..c (H51)
     "tokens_sampled": int}},                 # generated actions; forced hint tokens excluded
 "totals": {"episodes_sampled", "tokens_sampled", "correct_by_checkpoint"}}
```

The `hints` trainer adds `first_hinted_success_update`, `hinted_episodes_sampled` and
`hinted_correct_by_checkpoint`; its `first_success_update` counts un-hinted episodes only.
`sft_then_rl` merges both phases on global update numbers. To log from your own trainer, create
a `SampleLog(prompt_ids, checkpoints)` in `train`, call `log.record(update, rollout, rewards)`
after each sampling step, and return `log.to_dict()` from `training_log()`.

The frozen main toy run predates these logs, so its results have no `training.json` files.

## 8. `budget-curve`: the split as a function of N (addendum A, H48, H51)

```
python -m capsplit.harness budget-curve results/<name> [--out DIR] [--step K] [--c C] [--n-train ARM=N ...]
```

**What it computes** (`harness/budget.py`, no retraining). For every run in `index.json` it
loads the cached base and the saved checkpoint (`--step`, default the final saved one; it needs
`save_policies` other than `"none"` and a task with an exact table), computes every correct
solution's q0(s) and q1(s) exactly, and evaluates, on a grid of N (default 1, 2, 4, … up to the
first power of two ≥ max(2^20, 64 × the largest N_train)):

```
rel(N) = mean_i sum_{s: q0(s) >= c/N} (q1(s) - q0(s))
cap(N) = mean_i sum_{s: q0(s) <  c/N} (q1(s) - q0(s))     rel(N) + cap(N) = J1 - J0 (checked to 1e-12)
```

It also reports the crossover N\* (the smallest grid N with rel(N) ≥ half the total gain; null
when the total is ≤ 0 or the grid never gets there) and **cap(N_train)**, the capability part
measured at the number of episodes the arm actually sampled per training prompt.

**Where N_train comes from.**

1. The run's training log: the mean over the prompts the arm actually sampled of
   `episodes_by_checkpoint[step]` (or `episodes_sampled` for the whole run), with min/max kept.
   Source `"training_log"`. A log without counts for the requested step gives *unknown*, not 0.
2. Otherwise `--n-train ARM=N` (matched by arm name, then trainer name), marked
   `"from_config (not logged)"`. This exists for the frozen main run, which has no logs:
   `--n-train rloo=16000` is 1,000 updates × G = 16.
3. Otherwise `"unknown"`, and cap(N_train) is null.

**Output folder.** Default `<results_dir>/budget_curve/`; `--out` picks another. The command
**refuses a folder that exists and is not empty** (`FileExistsError`, exit code 2), so an earlier
curve is never overwritten or mixed with a new one. Files: `budget_curve.csv` (arm, factors,
seed, group, N, rel, cap, total), `summary.json`, `budget_curve.md`, `figures/budget_<arm>.png`.
The committed outputs are `results/published/budget_curve_main/` and `results/published/seedstudy/budget_curve/`.

**Sampled-backend truncation.** In a run measured with the `sampled` backend, reachability was
decided from `m_classify` base samples, and a solution with zero successes in m samples can only
be bounded by u0 = 1 − (α/2)^(1/m) (the 95% Clopper–Pearson upper bound). Solutions with
q0 < u0 cannot be told apart from 0, so the grid is cut at N ≤ floor(c/u0)
(`budget.resolvable_n_max`), and cap(N_train) is null with a note when N_train lies beyond it.

## 9. `discovery predict | analyze`: first success as a waiting time (addendum B4)

This is the code for the pre-registered prediction in `docs/prediction_discovery.md`
(`harness/discovery.py`). Without drift, a hard training prompt with base success
probability p, trained with G samples per update, first produces a correct sample at a geometric
update T: P(T > t) = (1 − p)^(G·t).

```
python -m capsplit.harness discovery predict CONFIG [--out DIR] [--results-root results] [--workers N]
python -m capsplit.harness discovery analyze RESULTS_DIR --predictions P/predictions.json [--out DIR]
                                             [--compare OTHER_RESULTS_DIR OTHER_PREDICTIONS ...]
```

- **`predict`** builds (or loads from `<results-root>/<name>/bases/`, the same cache the runs
  use) each distinct effective base of the config and computes p exactly for every hard-train
  prompt, then E[T], median T and P(undiscovered by the budget). No training. Default output:
  `results/published/seedstudy/predictions/<config name>/` (`predictions.json`, `predictions.md`). These were
  committed before the seed-study runs.
- **`analyze`** reads the runs' `first_success_update` from the training logs and compares them
  with the predictions per factor combination: observed vs predicted medians, never-discovered
  fraction, a KS-style distance, a randomised PIT z-score, the rank AUC of base p, and drift
  (J at the last checkpoint before T over p). It prints a verdict per k with the pre-registered
  thresholds (constants at the top of `discovery.py`). With `--compare` (the other seed study and
  its predictions) it adds the variance decomposition of final per-prompt Δcap into a
  training-seed part V_train (the `base_seed` study) and a base-seed part V_base (the `train_seed`
  study): share = V_train / V_base ≤ 1/3 → "outcome determined by the base", ≥ 2/3 → "outcome
  depends on sampling luck", else "mixed". Default output `<results_dir>/discovery/`
  (`analysis.json`, `analysis.md`, `figures/`).

The full seed study:

```
python -m capsplit.harness discovery predict configs/seedstudy_base.json
python -m capsplit.harness discovery predict configs/seedstudy_train.json
python -m capsplit.harness run configs/seedstudy_base.json  --workers 6
python -m capsplit.harness run configs/seedstudy_train.json --workers 6
python -m capsplit.harness discovery analyze results/seedstudy_base \
    --predictions results/published/seedstudy/predictions/seedstudy_base/predictions.json \
    --compare results/seedstudy_train results/published/seedstudy/predictions/seedstudy_train/predictions.json
```

## 10. The language-model experiment (`capsplit.harness.lm`)

The LM experiment post-trains a small local model (`Qwen/Qwen2.5-0.5B-Instruct` at a pinned
revision, L1) with LoRA on generated arithmetic and splits its gain with *sampled* measurement.
Design: `docs/lm_design.md`; pre-registration: `docs/preregistration_lm.md`; operations:
`docs/RUNBOOK.md`. It needs the `lm` extra (torch 2.8.0, transformers, peft; L2) and reuses the
harness interfaces (`Task`, `Policy`, `Trainer`, `GroupMetrics`) but **not** the toy runner: it
runs as separate, resumable stages (L5).

| RL object | Here |
|---|---|
| Episode | One generated problem, e.g. `347 * 29` |
| State | The chat-formatted prompt plus the tokens generated so far |
| Action | The next token (at most `model.max_new_tokens` = 12) |
| Reward | 1 if the first integer in the completion equals the answer, else 0 |
| Policy | π0 = the model with the LoRA adapter disabled; π1 = the same model with a trained adapter |

### Why the LM plug-ins are not in `PLUGIN_MODULES` (L8)

`lm/policy.py`, `lm/tasks.py` and `lm/trainers.py` register with the ordinary registries
(`POLICIES`, `TASKS`, `TRAINERS`), but they import transformers and peft. The registry raises on
a listed module whose dependencies are missing, so listing them would break `python -m
capsplit.harness list`, the toy runner and the toy tests on any machine without the `lm` extra.
Instead the LM stages import them explicitly (`stages._plugins()`), and
`python -m capsplit.harness list` does not show them. In your own script, import the modules
first (`import capsplit.harness.lm.trainers`) before calling `TRAINERS.create("lm_rl", ...)`.

### `LocalLMPolicy` (`lm/policy.py`, registered as `local_lm`)

`LocalLMPolicy(config: LMConfig, device=None, use_fallback=False, seed=0)` loads the pinned
checkpoint once and attaches a LoRA adapter (`lora.r`, `alpha`, `dropout`, `target_modules`)
whose initialisation is seeded by `seed` (L9). One loaded model serves both π0 (adapter
disabled) and π1.

- `sample(task, prompt_ids, generator)` generates with the repository's own key-value-cache loop
  and a CPU generator (not `model.generate`, which fails on MPS; L3). Actions are token-id
  tuples; the decoded text is what `task.verify` receives.
- `log_prob(task, prompt_ids, actions)` is a teacher-forced pass; it also accepts answer
  *strings* (used by SFT, which appends the stop token).
- The policy's distribution is softmax(logits / `sampling.reference_temperature`).
- `parameters()` are the LoRA weights only; `state_dict()` is the adapter only;
  `save_adapter(path)` / `load_adapter(path)` write and read `adapter.safetensors`.
- `state_logits` raises (no exact backend), forced hint prefixes raise, and `clone()` deep-copies
  the model, because peft's active adapter is shared module state.

### `LMTask` (`lm/tasks.py`, registered as `lm_arithmetic`)

`LMTask(problems, prompt_template)` wraps the rows of `data/problems.jsonl`
(`{"id", "generator", "question", "answer", "split", "difficulty"}`, ids 0..P−1). Each prompt is
`Prompt(id, tokens=(prompt_template.format(question=...),), tags={"split", "generator",
"difficulty"})`; the default template is `"Compute {question}. Reply with only the final
number."`. Differences from a toy task:

- `verify(pid, actions)` takes the decoded completion **text** (a `str`; anything else raises
  `TypeError`). `parse(pid, text)` returns `(valid, correct)`: valid iff the first number is an
  integer, correct iff it equals the answer (L6: "5.0" reads as 5, "2.5" is invalid, and an
  echoed "347 * 29 = 10063" reads 347 and is wrong).
- `initial_state` and `step` raise: the policy generates the episode, not the task.
  `n_actions` and `horizon` are 0 (the vocabulary and length belong to the policy).
- `demonstrations(pid)` is `(answer,)`, the string an SFT arm imitates.

### Problem generators and how to add one

The `data` stage calls `tasks.build_problems(config.task)`, which runs every entry of
`task.generators`. Generator `g` gets a numpy Generator seeded from
`SeedSequence([task.seed, g])`; its problems are split train/held-out with the same share in
every `difficulty` cell, and a question produced earlier (by any generator) is excluded, so no
question appears twice (L7). Two generators are registered: `arithmetic` (params `digits`,
`ops` ⊆ {"+", "-", "*"}, `negative`) and `gsm8k` (params `path` to a local GSM8K jsonl; rows get
`"possibly_contaminated": true`).

A generator is a class with one method,
`generate(rng, count, params, exclude) -> list[Problem]`, registered with
`@register_generator(name)` (keyword arguments become extra fields on every row it produces). It
must return exactly `count` problems with distinct questions, none in `exclude`; take all
randomness from `rng`; and raise `ValueError` on an unknown param. `Problem(question, answer,
difficulty, generator)` holds the answer as an exact integer string (the checker compares
integers) and a `difficulty` string, which becomes a tag, a stratum for the split and a
measurement group (`train:<difficulty>` etc.). The question is inserted into the prompt
template, so it should read naturally after "Compute".

Worked example: squares of d-digit numbers. Add it to `src/capsplit/harness/lm/tasks.py`
(below `ArithmeticGenerator`), or put it in its own module and add that module to the import in
`stages._plugins()`. A generator is found only if its module was imported before the `data`
stage runs.

```python
@register_generator("square")
class SquareGenerator:
    """`a * a` with a exactly d digits. Params: `digits` (list of ints). Difficulty e.g. "d2_sq"."""

    def generate(self, rng: np.random.Generator, count: int, params: dict,
                 exclude: Collection[str] = frozenset()) -> list[Problem]:
        _check_params("square", params, ("digits",), required=("digits",))
        digits = [int(d) for d in params["digits"]]
        seen, out = set(exclude), []
        for i in range(count):
            d = digits[i % len(digits)]                  # spread problems evenly over digit counts
            for _ in range(1000):
                a = int(rng.integers(10 ** (d - 1), 10 ** d))
                question = f"{a} * {a}"
                if question not in seen:
                    break
            else:
                raise ValueError(f"square: no new {d}-digit problem left; ask for fewer")
            seen.add(question)
            out.append(Problem(question, str(a * a), f"d{d}_sq", "square"))
        return out
```

Config snippet (`task` section of an LM config), mixing it with the arithmetic generator:

```json
"task": {
  "generators": [
    {"type": "arithmetic", "params": {"digits": [2, 3], "ops": ["+", "*"]}, "n_train": 6, "n_heldout": 4},
    {"type": "square", "params": {"digits": [2, 3]}, "n_train": 4, "n_heldout": 2}
  ],
  "seed": 0
}
```

With `lm_demo.json`'s other sections this gives 16 problems; the square rows look like
`{"id": 11, "generator": "square", "question": "76 * 76", "answer": "5776", "split": "train",
"difficulty": "d2_sq"}`. A misspelt param fails at the `data` stage with
`square: unknown params ['digit']; allowed params are ['digits']`, and an unregistered type with
`unknown problem generator 'sqaure'; registered: arithmetic, gsm8k, square` (both exit code 2).
Changing `task` after the `data` stage has run trips the stale-output guard (below).

### LM trainers (`lm/trainers.py`)

All four follow the harness `Trainer` contract (checkpoint 0 before any update, exactly `budget`
optimizer updates, only `prompt_ids`, all randomness from `seed`, unknown params raise) and
update only the LoRA parameters with Adam and gradient clipping (`max_grad_norm`, default 1.0).
They write no files: the `train` stage saves the adapter from `on_checkpoint` and writes
`training_log()` as `train/<arm>/seed<s>/training.json`.

| Trainer | Update rule | Params (defaults) |
|---|---|---|
| `lm_none` | none (control); `training_log()` is None | none |
| `lm_rl` | GRPO-style: each update takes `problems_per_update` train problems from a seeded shuffle cycle, samples `group` completions each without gradients, A = (r − mean) / (std + eps) per group, loss −mean(A · log π), log-probs recomputed in `micro_batch` chunks with gradient accumulation (L12, L14) | `group` (8), `problems_per_update` (8), `lr` (1e-4), `betas`, `weight_decay` (0), `max_grad_norm` (1.0), `eps` (1e-6), `micro_batch` (16) |
| `lm_sft` | maximum likelihood of `answer + stop` on train problems in the `target` base class(es); samples nothing | `target` ("unreachable"; a class or list of classes), `batch` (8), `lr` (1e-4), `betas`, `weight_decay`, `max_grad_norm` |
| `lm_sft_then_rl` | `lm_sft` for round(`sft_fraction` · budget) updates, then `lm_rl` for the rest, phases seeded separately and a fresh optimizer per phase (as the toy H25) | `sft_fraction` (0.5), `sft` (dict of `lm_sft` params), `rl` (dict of `lm_rl` params) |

**How `lm_sft` gets the base classification.** Config params are static and do not know the
output folder, so `lm_sft` and `lm_sft_then_rl` set `needs_base_classes = True`. Before `train`,
the stage reads `base/classes.json` and calls `trainer.set_base_classes({problem_id: "reachable"
| "unreachable" | "ambiguous"})` (int or str keys). `train` then uses only the train problems
whose class is in `target`; if none are, the `train` stage stops before any training starts and
says so. The log records `sft_target` and `sft_problem_ids`.

**Training log.** `LMSampleLog` extends the toy `SampleLog` (section 7) with per-prompt and total
`episodes_by_checkpoint` and `success_rate_by_checkpoint`, plus
`window_success_rate_by_checkpoint` in the totals. N_train for the LM budget curve is the mean
of `episodes_by_checkpoint[step]` over the train problems the arm sampled (H51); for the main
config's `rl` arm it is about 200 · 64 / 120 ≈ 107 (L16).

### Sampled measurement (`lm/measure.py`)

Pure numpy functions on per-problem success counts; nothing in it loads a model. The samples
come from fixed sets (`docs/lm_design.md`): **A** (base, classification only), **B** (base,
p̂0), **C** (base, one file per temperature, for the temperature baseline), **null** (a fresh
base sample) and **eval** (each trained checkpoint, p̂1).

- `clopper_pearson(successes, trials, alpha)` → (lower, upper), exact by bisection (L10).
- `classify(successes_A, trials_A, n, alpha)`: **reachable** iff lower ≥ 1/N, **unreachable** iff
  upper < 1/N, otherwise **ambiguous**.
- `split(classes, p0, p1)`: each class's summed gain divided by *all* problems, so the three
  contributions add up to mean(p̂1 − p̂0) (checked to 1e-12).
- `temperature_baseline(...)`: split-half choice of T* on set C, E_1 and E_N against it.
- `bootstrap(...)`: problems resampled, and each resampled problem's set-A count redrawn and
  reclassified, so classification noise is in the interval.
- `null_check(...)`: the pipeline with π1 := the null set; passes iff the total's interval
  contains 0.
- `budget_curve(successes_A, trials_A, p0, p1, n_grid, alpha)`: the three-way split at each N,
  cut at floor(1/u0) for the smallest set-A size (108 for 400 samples at α = 0.05).
- `group_metrics(...)` adapts all of this to `GroupMetrics`: `delta_cap` = the unreachable
  contribution, `delta_rel` = reachable + ambiguous (the three-way split is in
  `extra["split"]`); *beyond reach* needs an unreachable problem whose trained lower bound is
  ≥ 1/N, and *beyond temperature* needs E_1's bootstrap lower bound > 0 (L11).

```python
import numpy as np
from capsplit.harness.lm.measure import classify, split

classes = classify(np.array([0, 30, 6, 48]), 400, n=16, alpha=0.05)
# ['unreachable' 'ambiguous' 'unreachable' 'reachable']
split(classes, p0=np.array([0, 5, 1, 8]) / 64, p1=np.array([3, 9, 1, 20]) / 64)
# {'reachable': 0.046875, 'unreachable': 0.01171875, 'ambiguous': 0.015625, 'total': 0.07421875, ...}
```

### The staged CLI

```
python -m capsplit.harness.lm <stage> CONFIG [--out results] [--fallback] [--all-checkpoints] [--prereg P]
```

| Stage | Does | Writes under `<out>/<name>/` |
|---|---|---|
| `doctor` | reports versions, device, memory and disk | `doctor.json` |
| `fetch`, `speedtest`, `setup` | downloads the pinned model; measures throughput; `setup` = both | their own JSON records |
| `data` | generates the problems (CPU only) | `data/problems.jsonl`, `data/summary.json` |
| `base` | samples sets A, B, C and null from π0; classifies | `samples/base/*.jsonl`, `base/classes.json`, `base/summary.json` |
| `train` | trains every arm × seed | `train/<arm>/seed<s>/step<k>/adapter.safetensors`, `training.json`, `timing.json` |
| `measure` | samples each run's final checkpoint (`--all-checkpoints`: every checkpoint); split, null check, budget curve | `samples/<arm>/seed<s>/step<k>/eval_T*.jsonl`, `measure/{results,null_check,budget_curve}.json` |
| `report` | report from files only, byte-identical on rebuild (L18) | `report/report.md`, `summary.json`, `tables/*.csv`, `figures/*.png` |
| `verdicts` | the pre-registered L1–L3 tests, thresholds from `--prereg` (default `configs/lm_prereg.json`) | `report/verdicts.{md,json}` |

Each stage prints what it wrote and the next command; errors the user can fix exit with code 2
and say what to do. Every stage checks free memory first, and on MPS caps the allocator at
`memory.max_process_gb` (L14). Sampling resumes to the row (each problem × set × temperature
chunk has its own seed); an interrupted training run restarts from update 0 (L15). Run long
stages under `caffeinate -i nice -n 10` (L13). `--fallback` switches to the fallback model; give
it to every stage or to none.

```
python -m capsplit.harness.lm data    configs/lm_demo.json
python -m capsplit.harness.lm base    configs/lm_demo.json
python -m capsplit.harness.lm train   configs/lm_demo.json
python -m capsplit.harness.lm measure configs/lm_demo.json
python -m capsplit.harness.lm report  configs/lm_demo.json
```

The config schema is `lm/config.py` (its own, separate from the toy schema; unknown keys raise
`LMConfigError`). The seed streams are `task.seed` (data), the run seed from `seeds` (training
sampling and LoRA init) and `sampling.seed` (measurement sampling); the base is a pretrained
checkpoint, so `base_seed` / `train_seed` do not apply. All three are recorded in every manifest.

### Stale-output guard (L19)

Outputs live under the experiment name, so without a check a changed config could silently
reuse old samples or adapters. Every stage manifest stores a `fingerprint`: one hash per config
section its outputs depend on.

| Stage | Sections hashed |
|---|---|
| data | `task` |
| base | data's + `model` (id, revision, max_new_tokens, system prompt, dtype, and whether `--fallback` was given) + `sampling` |
| train (per arm × seed) | base's + `lora` + `arms.<that arm>` + `budget` + `checkpoints` + `seeds` |
| measure | train's + `measure` |

Before reusing anything, a stage compares the current fingerprint with the earlier stages'
manifests and its own, and exits with code 2 naming the differing sections, e.g. `outputs under
results/lm_main/data were produced with a config that differs from the current config/model in:
task; use a new experiment name, or delete results/lm_main/data to recompute`. Arms are hashed
separately, so adding an arm does not invalidate the others. Manifests from older code without a
fingerprint are checked by recomputing it from their stored config; outputs with no manifest at
all are reused with a warning.
