"""Run an experiment grid: factor combination x seed x arm.

Result folder layout (`<out>` is `out_root`, `<name>` the experiment name)::

    <out>/<name>/
      config.json                      the input config (sorted keys)
      index.json                       every run; the report generator reads this
      bases/<key>/policy.pt            pickled base Policy object (torch.save)
      bases/<key>/info.json            {"task", "base", "seed", "info"}: what built it and the recipe's info
      <cell>/manifest.json             runlog manifest: commit, dirty flag, the cell config
      <cell>/seed-<s>.metrics.json     measurements of one run
      <cell>/seed-<s>.policies.pt      {step: policy.state_dict()} (absent if save_policies is "none")

A *cell* is one (factor combination, arm); its folder name is the first 12 hex digits of the
SHA-256 of the cell config (`capsplit.runlog.run_dir`). A base `<key>` is the first 12 hex
digits of the SHA-256 of {"task", "base", "seed"} (types and params after factor overrides).

index.json schema::

    {
     "name": str,
     "config": "config.json",
     "factors": {key: [values]},             # the grid axes (sorted keys)
     "arms": [{"name", "trainer", "params"}],
     "seeds": [int], "budget": int, "checkpoints": [int],
     "measure": {"ns", "cs", "group_by"},
     "save_policies": str,
     "runs": [                               # ordered by factor combination, arm, seed
       {"factors": {key: value}, "arm": str, "trainer": str, "seed": int,
        "run_dir": "<cell>", "base": "bases/<key>",
        "files": {"manifest": "<cell>/manifest.json",
                  "metrics": "<cell>/seed-<s>.metrics.json",
                  "policies": "<cell>/seed-<s>.policies.pt" | null}}
     ]
    }

All paths in index.json are relative to `<out>/<name>/`.

seed-<s>.metrics.json schema::

    {"arm": str, "trainer": str, "seed": int, "factors": {key: value}, "base": "bases/<key>",
     "groups": {group: [prompt ids]}, "train_prompt_ids": [int],
     "records": [GroupMetrics.to_dict() + {"step": int}, ...]}   # in checkpoint order

Reproducibility: no timestamps are written; JSON is written with sorted keys and Python's
shortest round-trip float repr, so a re-run gives byte-identical JSON. torch is seeded with the
run seed before the base is built and before each arm trains, and torch runs single-threaded
during the experiment so `workers=1` and `workers>1` produce identical numbers.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from capsplit.runlog import run_dir

from .config import Arm, Component, ExperimentConfig, load_config, measure_spec
from .interfaces import Policy, Task
from .registry import BACKENDS, BASES, TASKS, TRAINERS


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, (tuple, set, frozenset)):
        return list(value)
    raise TypeError(f"cannot write {type(value).__name__} to JSON")


def dumps(data: Any) -> str:
    """Deterministic JSON: sorted keys, 2-space indent, trailing newline."""
    return json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n"


def _atomic_write(path: Path, write) -> None:
    """Write via a temp file in the same folder, then rename (safe under parallel workers)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _write_text(path: Path, text: str) -> None:
    def write(tmp: str) -> None:
        with open(tmp, "w") as handle:
            handle.write(text)

    _atomic_write(path, write)


def _hash(data: Any) -> str:
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _log(message: str) -> None:
    print(f"[capsplit.harness] {message}", file=sys.stderr, flush=True)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, weights_only=False)
    except TypeError:  # torch without the weights_only argument
        return torch.load(path)


def train_prompt_ids(task: Task) -> list[int]:
    """The train split if any prompt has a "split" tag, else every prompt."""
    if any("split" in p.tags for p in task.prompts()):
        return task.prompt_ids(split="train")
    return [p.id for p in task.prompts()]


# ---------------------------------------------------------------------- bases


def base_key(task: Component, base: Component, seed: int) -> str:
    return _hash({"task": task.to_dict(), "base": base.to_dict(), "seed": seed})


def load_or_build_base(
    exp_dir: Path, task: Task, task_cfg: Component, base_cfg: Component, seed: int, rebuild: bool = False
) -> tuple[Policy, str]:
    """The base policy for (task, recipe, seed), from the on-disk cache or freshly built.

    The whole Policy object is pickled with torch.save, so any Policy class restores without
    knowing its constructor arguments.
    """
    key = base_key(task_cfg, base_cfg, seed)
    folder = exp_dir / "bases" / key
    policy_path, info_path = folder / "policy.pt", folder / "info.json"
    if not rebuild and policy_path.exists() and info_path.exists():
        policy = _torch_load(policy_path)
        if not isinstance(policy, Policy):
            raise TypeError(f"{policy_path} does not hold a Policy")
        return policy, key
    _log(f"building base {key} (seed {seed})")
    torch.manual_seed(seed)
    recipe = BASES.create(base_cfg.type, **base_cfg.params)
    policy, info = recipe.build(task, seed)
    _atomic_write(policy_path, lambda tmp: torch.save(policy, tmp))
    _write_text(info_path, dumps({"task": task_cfg.to_dict(), "base": base_cfg.to_dict(), "seed": seed, "info": info}))
    return policy, key


# ---------------------------------------------------------------------- one unit


def run_dir_path(experiment: str, config: dict, root: str) -> Path:
    """Where `runlog.run_dir` puts this cell, without creating it (same hashing rule)."""
    import hashlib
    import json

    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return Path(root) / experiment / hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _run_files(config: ExperimentConfig, cell: str, seed: int) -> dict:
    return {
        "manifest": f"{cell}/manifest.json",
        "metrics": f"{cell}/seed-{seed}.metrics.json",
        "policies": None if config.save_policies == "none" else f"{cell}/seed-{seed}.policies.pt",
    }


def run_unit(config: ExperimentConfig, out_root: str, combo_index: int, seed: int, force: bool = False) -> list[dict]:
    """Run every arm for one (factor combination, seed); return their index.json entries."""
    combo = config.factor_combos()[combo_index]
    parts = config.resolved(combo)
    exp_dir = Path(out_root) / config.name
    key = base_key(parts["task"], parts["base"], seed)

    entries: list[dict] = []
    pending: list[tuple[Arm, Path, dict]] = []
    for arm in config.arms:
        if force:
            # Rewrite the manifest so it records the code that produced the new results.
            stale = run_dir_path(config.name, config.cell_config(combo, arm), out_root) / "manifest.json"
            stale.unlink(missing_ok=True)
        directory = run_dir(config.name, config.cell_config(combo, arm), root=out_root)
        files = _run_files(config, directory.name, seed)
        entries.append(
            {
                "factors": dict(combo),
                "arm": arm.name,
                "trainer": arm.trainer,
                "seed": seed,
                "run_dir": directory.name,
                "base": f"bases/{key}",
                "files": files,
            }
        )
        complete = (exp_dir / files["metrics"]).exists() and (
            files["policies"] is None or (exp_dir / files["policies"]).exists()
        )
        if complete and not force:
            _log(f"skip {combo} arm={arm.name} seed={seed}: metrics exist")
            continue
        pending.append((arm, directory, files))
    if not pending:
        return entries

    task = TASKS.create(parts["task"].type, **parts["task"].params)
    base, built_key = load_or_build_base(exp_dir, task, parts["task"], parts["base"], seed, rebuild=force)
    assert built_key == key
    spec = measure_spec(task, config.measure)
    backend = BACKENDS.create(parts["backend"].type, **parts["backend"].params)
    cache = backend.prepare(task, base, spec)
    train_ids = train_prompt_ids(task)

    for arm, directory, files in pending:
        _log(f"run {combo} arm={arm.name} seed={seed}")
        policy = base.clone()
        records: list[dict] = []
        saved: dict[int, dict] = {}
        steps: list[int] = []

        def hook(step: int, trained: Policy) -> None:
            steps.append(step)
            for metric in backend.measure(task, base, trained, spec, cache):
                records.append({**metric.to_dict(), "step": step})
            if config.save_policies == "checkpoints" or (config.save_policies == "final" and step == config.budget):
                saved[step] = trained.state_dict()

        torch.manual_seed(seed)
        trainer = TRAINERS.create(arm.trainer, **arm.params)
        trainer.train(policy, task, config.budget, seed, train_ids, list(config.checkpoints), hook)
        if steps != list(config.checkpoints):
            raise RuntimeError(f"trainer {arm.trainer!r} called checkpoints {steps}, expected {list(config.checkpoints)}")

        if files["policies"] is not None:
            _atomic_write(exp_dir / files["policies"], lambda tmp: torch.save(saved, tmp))
        metrics = {
            "arm": arm.name,
            "trainer": arm.trainer,
            "seed": seed,
            "factors": dict(combo),
            "base": f"bases/{key}",
            "groups": {name: list(ids) for name, ids in spec.groups.items()},
            "train_prompt_ids": list(train_ids),
            "records": records,
        }
        # The metrics file is written last: its existence marks the run complete.
        _write_text(exp_dir / files["metrics"], dumps(metrics))
    return entries


def _worker(args: tuple) -> list[dict]:
    torch.set_num_threads(1)
    return run_unit(*args)


# ---------------------------------------------------------------------- experiment


def run_experiment(
    config: ExperimentConfig | str | Path, out_root: str | Path = "results", workers: int = 1, force: bool = False
) -> Path:
    """Run the whole grid and return `<out_root>/<name>`.

    Units of work are (factor combination, seed); each builds (or loads) its base once and runs
    every arm on a clone. Runs whose metrics file exists are skipped unless `force`, which also
    rebuilds the cached bases. `workers > 1` runs units in a spawn-context process pool.
    """
    if not isinstance(config, ExperimentConfig):
        config = load_config(config)
    out = str(out_root)
    exp_dir = Path(out) / config.name
    from capsplit.runlog import _git

    if _git("rev-parse", "HEAD") is None:
        _log("warning: not inside a git checkout, so manifests will record commit = null; run from the repo")
    exp_dir.mkdir(parents=True, exist_ok=True)
    _write_text(exp_dir / "config.json", dumps(config.raw))

    units = [(config, out, i, seed, force) for i in range(len(config.factor_combos())) for seed in config.seeds]
    threads = torch.get_num_threads()
    try:
        if workers > 1 and len(units) > 1:
            context = multiprocessing.get_context("spawn")
            with context.Pool(min(workers, len(units))) as pool:
                results = pool.map(_worker, units, chunksize=1)
        else:
            torch.set_num_threads(1)
            results = [run_unit(*unit) for unit in units]
    finally:
        torch.set_num_threads(threads)

    arm_order = {arm.name: i for i, arm in enumerate(config.arms)}
    runs: list[tuple[int, dict]] = []
    for (_, _, combo_index, _, _), entries in zip(units, results):
        runs.extend((combo_index, entry) for entry in entries)
    runs.sort(key=lambda item: (item[0], arm_order[item[1]["arm"]], item[1]["seed"]))

    index = {
        "name": config.name,
        "config": "config.json",
        "factors": config.factors,
        "arms": [arm.to_dict() for arm in config.arms],
        "seeds": list(config.seeds),
        "budget": config.budget,
        "checkpoints": list(config.checkpoints),
        "measure": config.measure.to_dict(),
        "save_policies": config.save_policies,
        "runs": [entry for _, entry in runs],
    }
    _write_text(exp_dir / "index.json", dumps(index))
    _log(f"done: {len(runs)} runs in {exp_dir}")
    return exp_dir
