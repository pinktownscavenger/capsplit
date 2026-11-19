# Experiment harness: developer guide

`capsplit.harness` runs post-training experiments and measures each trained policy against its
base: the success rate J, the split of the gain into **reliability** (`delta_rel`) and
**capability** (`delta_cap`), the excess over the best tempered base (E_N) and a 2x2 label.
This guide covers how the pieces fit together and how to add a trainer, a task or a config.
Design choices are logged in `docs/DECISIONS.md` (entries H1–H39 are referred to below).

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
