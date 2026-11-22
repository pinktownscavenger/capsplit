"""The LM experiment's data, base, train, measure and report stages (docs/lm_design.md, C5).

Every stage: applies `config.runtime` (niceness, threads, the MPS hard cap), checks free memory
before it starts, turns torch out-of-memory errors into a `MemoryGuardError` naming what to lower,
skips work that is already on disk, writes whole files atomically (`storage.write_json`) and
sample rows as complete appended lines, and writes a manifest recording the three seed streams.

Outputs under `<out>/<name>/`:

* `data/problems.jsonl`, `data/summary.json`
* `samples/<tag>/<set>_T<T>.jsonl` (tag `base` with sets A, B, C, null; tag
  `<arm>/seed<s>/step<k>` with set eval), `base/classes.json`, `base/summary.json`
* `train/<arm>/seed<s>/step<k>/adapter.safetensors` (+ `manifest.step.json`),
  `train/<arm>/seed<s>/timing.json`, `train/<arm>/seed<s>/training.json`
* `measure/results.json`, `measure/null_check.json`, `measure/budget_curve.json`

Resuming:

* **Samples.** One chunk = (tag, set, T, problem): its `n` samples come from one torch generator
  seeded by `SeedSequence([measure_seed, crc32(tag), crc32(set), round(T*1000), problem])`, so a
  chunk is reproducible on its own. A re-run regenerates only chunks with missing indices, and of
  those writes only the missing rows; on CPU the result is identical to an uninterrupted run (the
  regenerated chunk replays the same stream with the same batching). Changing `sampling.batch_size`
  (or the n of a set) between runs changes the streams of chunks still to do; rows already on disk
  are kept. Reading uses indices < n only, first row per (problem, index).
* **Training.** An (arm, seed) is finished when its final adapter and `training.json` exist
  (`training.json` is written last). An interrupted run restarts from update 0: the trainers keep
  optimizer and sampler state in memory only, so a mid-run restart would not reproduce the run.
* **Measurement** is resumable at sample level; the statistics are recomputed (cheap) every run.

Stale outputs: every manifest records a `fingerprint` (one hash per config section that
determines the stage's outputs, plus the model actually used, i.e. whether `--fallback` was given):

* data: `task`
* base: data's + `model` (id, revision, max_new_tokens, system_prompt, dtype) + `sampling`
* train (per arm x seed): base's + `lora` + `arms.<name>` (each arm separately) + `budget` +
  `checkpoints` + `seeds`
* measure: train's + `measure`; the eval samples of one run carry their own manifest
  (`samples/<arm>/seed<s>/manifest.eval.json`, the run's train sections).

Before a stage reuses anything it compares the current fingerprint with the earlier stages'
manifests and its own (`check_manifest`) and raises `StageError` naming the differing sections.
A manifest without a fingerprint (written by older code) is checked by recomputing the
fingerprint from its stored full `config` and `model`. Each stage writes its manifest before it
starts work, so an interrupted run is guarded too; outputs with no manifest at all (older code,
interrupted) are reused with a warning.

`measure_seed` is `getattr(config.sampling, "seed", 0)`: `SamplingConfig` has no `seed` field yet,
so it is 0 and recorded as such. Likewise `measure.n_splits` defaults to `DEFAULT_N_SPLITS`.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import zlib
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import LMConfig
from .device import MemoryGuard, apply_runtime, explain_oom
from .storage import Progress, append_jsonl, read_json, read_jsonl, write_json, write_manifest

DEFAULT_N_SPLITS = 20  # temperature-baseline random half splits, until MeasureConfig has n_splits
SETS_BASE = ("A", "B", "C", "null")


class StageError(RuntimeError):
    """A stage cannot run (an earlier stage is missing, or outputs conflict). The message says what to do."""


def _plugins() -> None:
    """Register the LM plug-ins (they are deliberately not in registry.PLUGIN_MODULES, DECISIONS L8)."""
    from . import policy, tasks, trainers  # noqa: F401


# --------------------------------------------------------------------------- small helpers


def measure_seed(config: LMConfig) -> int:
    return int(getattr(config.sampling, "seed", 0))


def n_splits(config: LMConfig) -> int:
    return int(getattr(config.measure, "n_splits", DEFAULT_N_SPLITS))


def seed_streams(config: LMConfig, training_seed: int | None = None) -> dict:
    return {"data_seed": config.task.seed, "training_seed": training_seed, "measure_seed": measure_seed(config)}


def temp_str(temperature: float) -> str:
    return repr(float(temperature))


def sample_path(out: Path, tag: str, set_name: str, temperature: float) -> Path:
    return out / "samples" / tag / f"{set_name}_T{temp_str(temperature)}.jsonl"


def run_tag(arm: str, seed: int, step: int) -> str:
    return f"{arm}/seed{seed}/step{step}"


def run_dir(out: Path, arm: str, seed: int) -> Path:
    return out / "train" / arm / f"seed{seed}"


def adapter_path(out: Path, arm: str, seed: int, step: int) -> Path:
    return run_dir(out, arm, seed) / f"step{step}" / "adapter.safetensors"


def chunk_seed(seed: int, tag: str, set_name: str, temperature: float, problem: int) -> int:
    """The torch seed of one sample chunk (docs/lm_design.md, "Seeds")."""
    seq = np.random.SeedSequence([seed, zlib.crc32(tag.encode()), zlib.crc32(set_name.encode()),
                                  round(temperature * 1000), int(problem)])
    return int(seq.generate_state(1, dtype=np.uint64)[0])


def repair_tail(path: Path) -> None:
    """Drop a truncated last line (an interrupted append), so the next append starts on a fresh line."""
    if not path.exists():
        return
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        cut = data.rfind(b"\n") + 1
        with path.open("r+b") as handle:
            handle.truncate(cut)


def read_samples(path: Path, n: int) -> dict[tuple[int, int], dict]:
    """{(problem, index): row} for indices < n; the first row wins if a pair repeats."""
    rows: dict[tuple[int, int], dict] = {}
    for row in read_jsonl(path):
        key = (int(row["problem"]), int(row["index"]))
        if key[1] < n and key not in rows:
            rows[key] = row
    return rows


def load_counts(path: Path, n: int, n_problems: int) -> Any:
    """`measure.Counts` from a sample file; every problem must have all n samples."""
    from .measure import counts_from_samples

    rows = read_samples(path, n)
    counts = counts_from_samples(rows.values(), n_problems)
    short = np.flatnonzero(counts.trials < n)
    if len(short):
        raise StageError(f"{path} is incomplete ({len(short)} problems have fewer than {n} samples, first ids "
                         f"{short[:5].tolist()}); re-run the stage that writes it")
    return counts


@contextmanager
def _stage(config: LMConfig, stage: str, device: Any) -> Any:
    """Runtime settings, start-of-stage memory check, and out-of-memory errors explained."""
    apply_runtime(config.runtime, memory=config.memory, device=device)
    MemoryGuard(config.memory).check_start(stage)
    try:
        yield
    except RuntimeError as error:
        explained = explain_oom(error, config.memory, stage)
        if explained is None:
            raise
        raise explained from error


def _require(path: Path, stage: str) -> Path:
    if not path.exists():
        raise StageError(f"{path} does not exist; run the `{stage}` stage first")
    return path


def load_problems(out: Path) -> list[dict]:
    return list(read_jsonl(_require(out / "data" / "problems.jsonl", "data")))


def load_task(config: LMConfig, out: Path) -> Any:
    from .tasks import LMTask

    return LMTask(load_problems(out), config.task.prompt_template)


# --------------------------------------------------------------------------- stale-output guard

RUN_KEYS = ("task", "model", "sampling", "lora", "budget", "checkpoints", "seeds")  # + arms.<name>


def model_choice(config: LMConfig, use_fallback: bool = False) -> dict:
    """{"id", "revision"} of the checkpoint a stage loads (the primary, or the fallback with --fallback)."""
    m = config.model
    if not use_fallback:
        return {"id": m.id, "revision": m.revision}
    if not m.fallback_id or not m.fallback_revision:
        raise StageError("--fallback was given but model.fallback_id / model.fallback_revision are not set")
    return {"id": m.fallback_id, "revision": m.fallback_revision}


def plain_config(config: LMConfig) -> dict:
    """`config.to_dict()` as it reads back from a manifest (tuples become lists)."""
    return dict(json.loads(json.dumps(config.to_dict())))


def config_sections(cfg: dict, model: dict | None, stage: str) -> dict[str, Any]:
    """The config sections (JSON values) that determine `stage`'s outputs; `cfg` is a `plain_config`."""
    sections: dict[str, Any] = {"task": cfg["task"]}
    if stage == "data":
        return sections
    if model is None:
        raise ValueError(f"the {stage} fingerprint needs the model choice")
    mc = cfg["model"]
    sections["model"] = {"id": model["id"], "revision": model["revision"], "max_new_tokens": mc["max_new_tokens"],
                         "system_prompt": mc["system_prompt"], "dtype": mc["dtype"]}
    sections["sampling"] = cfg["sampling"]
    if stage == "base":
        return sections
    sections["lora"] = cfg["lora"]
    for arm in cfg["arms"]:
        sections[f"arms.{arm['name']}"] = arm
    for key in ("budget", "checkpoints", "seeds"):
        sections[key] = cfg[key]
    if stage in ("train", "step", "eval"):
        return sections
    if stage != "measure":
        raise ValueError(f"no fingerprint is defined for stage {stage!r}")
    sections["measure"] = cfg["measure"]
    return sections


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def fingerprint(cfg: dict, model: dict | None, stage: str) -> dict[str, str]:
    """{section: hash} for `stage` (see the module docstring for the sections)."""
    return {key: _digest(value) for key, value in config_sections(cfg, model, stage).items()}


def stored_fingerprint(manifest: dict, stage: str) -> dict[str, str] | None:
    """A manifest's fingerprint; for an older manifest without one, recomputed from its stored config
    and model (no recorded model = the config's primary model). None if it cannot be recovered."""
    stored = manifest.get("fingerprint")
    if isinstance(stored, dict):
        return {str(k): str(v) for k, v in stored.items()}
    cfg = manifest.get("config")
    if not isinstance(cfg, dict):
        return None
    try:
        model = manifest.get("model") or {"id": cfg["model"]["id"], "revision": cfg["model"]["revision"]}
        return fingerprint(cfg, model, stage)
    except (KeyError, TypeError, ValueError):
        return None


def _delete_hint(paths: Sequence[Path]) -> str:
    return " and ".join(str(p) for p in paths)


def check_manifest(path: Path, stage: str, current: dict[str, str], keys: Iterable[str],
                   delete: Sequence[Path]) -> bool:
    """Compare the manifest at `path` with `current` on `keys`; StageError on any difference.

    Returns False if there is no manifest (nothing to compare), True if it matches.
    """
    if not path.exists():
        return False
    advice = f"use a new experiment name, or delete {_delete_hint(delete)} to recompute"
    try:
        stored = stored_fingerprint(read_json(path), stage)
    except (OSError, ValueError):
        stored = None
    if stored is None:
        raise StageError(f"{path} records no usable config, so its outputs cannot be checked against the current "
                         f"config; {advice}")
    differing = [k for k in keys if stored.get(k) != current[k]]
    if differing:
        hint = (" (the model section includes the checkpoint: give --fallback to every stage or to none)"
                if "model" in differing else "")
        raise StageError(f"outputs under {path.parent} were produced with a config that differs from the current "
                         f"config/model in: {', '.join(differing)}{hint}; {advice}")
    return True


def _has_files(*paths: Path) -> bool:
    for path in paths:
        if path.is_file() or (path.is_dir() and any(p.is_file() for p in path.rglob("*"))):
            return True
    return False


def _warn_unverified(what: str, where: Path) -> None:
    print(f"WARNING: {what} in {where} has no manifest (written by older code, or interrupted before it wrote "
          "one), so it cannot be checked against the current config; reusing it. Delete it to recompute.",
          file=sys.stderr)


def run_keys(arm: str) -> list[str]:
    """Sections one (arm, seed) training run (and its eval samples) depend on."""
    return [*RUN_KEYS, f"arms.{arm}"]


def eval_dir(out: Path, arm: str, seed: int) -> Path:
    return out / "samples" / arm / f"seed{seed}"


def _check_upstream(out: Path, current: dict[str, str], base: bool) -> None:
    """Compare the data (and with `base`, base) manifests with the current config."""
    check_manifest(out / "data" / "manifest.data.json", "data", current, ["task"], [out / "data"])
    if base:
        keys = ["task", "model", "sampling"]
        if not check_manifest(out / "base" / "manifest.base.json", "base", current, keys,
                              [out / "base", out / "samples" / "base"]) and _has_files(out / "base"):
            _warn_unverified("base output", out / "base")


# --------------------------------------------------------------------------- policies

PolicyFactory = Callable[[int], Any]  # seed -> LocalLMPolicy (fresh LoRA seeded by `seed`)


def default_policy_factory(config: LMConfig, device: Any, use_fallback: bool = False) -> PolicyFactory:
    from .policy import LocalLMPolicy

    def build(seed: int) -> Any:
        return LocalLMPolicy(config, device=device, use_fallback=use_fallback, seed=seed)

    return build


def resolve_device(config: LMConfig) -> Any:
    from .device import pick_device

    return pick_device(config.model.device)


def reseed_policy(policy: Any, seed: int) -> Any:
    """The same loaded base with a fresh LoRA adapter seeded by `seed`.

    peft's `unload()` removes the LoRA layers and returns the untouched base modules, so attaching a
    new adapter gives exactly the adapter `LocalLMPolicy(seed=seed)` would create (tested), without
    re-reading the checkpoint.
    """
    from .policy import LocalLMPolicy

    base = policy.model.unload()
    return LocalLMPolicy.from_loaded(base, policy.tokenizer, policy.config, seed=seed, choice=policy.choice)


def _model_info(policy: Any) -> dict | None:
    choice = getattr(policy, "choice", None)
    return None if choice is None else {"id": choice.id, "revision": choice.revision}


# --------------------------------------------------------------------------- sampling


@dataclass(frozen=True)
class SampleJob:
    tag: str
    set_name: str
    temperature: float
    n: int
    use_adapter: bool


def sample_jobs(policy: Any, task: Any, jobs: Sequence[SampleJob], out: Path, seed: int, label: str) -> list[Path]:
    """Fill every job's sample file for every problem, skipping (problem, index) pairs on disk."""
    n_problems = len(task.prompts())
    todo: list[tuple[SampleJob, Path, int, set[int]]] = []
    total = done = 0
    for job in jobs:
        path = sample_path(out, job.tag, job.set_name, job.temperature)
        repair_tail(path)
        have = read_samples(path, job.n)
        for pid in range(n_problems):
            total += 1
            missing = {i for i in range(job.n) if (pid, i) not in have}
            if missing:
                todo.append((job, path, pid, missing))
            else:
                done += 1
    progress = Progress(label, total, done)
    try:
        import torch

        for job, path, pid, missing in todo:
            generator = torch.Generator().manual_seed(chunk_seed(seed, job.tag, job.set_name, job.temperature, pid))
            drawn = policy.sample_texts(task, [pid] * job.n, job.temperature, generator, use_adapter=job.use_adapter)
            rows = []
            for index, (text, tokens) in enumerate(drawn):
                if index in missing:
                    valid, correct = task.parse(pid, text)
                    rows.append({"problem": pid, "index": index, "text": text, "valid": bool(valid),
                                 "correct": bool(correct), "tokens": int(tokens)})
            append_jsonl(path, rows)
            done += 1
            progress.update(done)
    finally:
        progress.close()
    return sorted({sample_path(out, j.tag, j.set_name, j.temperature) for j in jobs})


# --------------------------------------------------------------------------- data


def _tally(rows: Iterable[dict], keys: Sequence[str]) -> dict:
    out: dict[str, int] = {}
    for row in rows:
        k = "/".join(str(row[key]) for key in keys)
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def run_data(config: LMConfig, out: Path, device: Any = "cpu") -> list[Path]:
    """problems.jsonl (from tasks.build_problems) and a summary of counts per split x difficulty."""
    _plugins()
    from .tasks import build_problems

    with _stage(config, "data", device):
        fp = fingerprint(plain_config(config), None, "data")
        check_manifest(out / "data" / "manifest.data.json", "data", fp, fp, [out / "data"])
        rows = build_problems(config.task)
        path = out / "data" / "problems.jsonl"
        text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        if path.exists() and path.read_text() != text:
            raise StageError(f"{path} exists and differs from what this config generates; later stages depend on it. "
                             "Use a new config name, or delete the experiment folder to start over")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(text)
            tmp.replace(path)
        summary = {
            "n_problems": len(rows),
            "by_split": _tally(rows, ["split"]),
            "by_difficulty": _tally(rows, ["difficulty"]),
            "by_split_difficulty": _tally(rows, ["split", "difficulty"]),
        }
        summary_path = out / "data" / "summary.json"
        write_json(summary_path, summary)
        manifest = write_manifest(out / "data", "data", config.to_dict(), config.task.seed, str(device),
                                  extra={"seeds": seed_streams(config), "fingerprint": fp})
    return [path, summary_path, manifest]


# --------------------------------------------------------------------------- base


def base_jobs(config: LMConfig) -> list[SampleJob]:
    s = config.sampling
    ref = float(s.reference_temperature)
    jobs = [SampleJob("base", "A", ref, s.n_classify, False), SampleJob("base", "B", ref, s.n_measure, False)]
    jobs += [SampleJob("base", "C", float(t), s.n_temperature, False) for t in s.temperatures]
    jobs.append(SampleJob("base", "null", ref, s.n_measure, False))
    return jobs


def run_base(config: LMConfig, out: Path, device: Any = "cpu", factory: PolicyFactory | None = None,
             use_fallback: bool = False) -> list[Path]:
    """Sample the base (adapter disabled) for sets A, B, C, null; classify on A; summarise."""
    _plugins()
    from .measure import classify, clopper_pearson

    chosen = model_choice(config, use_fallback)
    fp = fingerprint(plain_config(config), chosen, "base")
    with _stage(config, "base", device):
        task = load_task(config, out)
        _check_upstream(out, fp, base=False)
        own = [out / "base", out / "samples" / "base"]
        if not check_manifest(out / "base" / "manifest.base.json", "base", fp, fp, own) and _has_files(*own):
            _warn_unverified("base output", out / "samples" / "base")
        write_manifest(out / "base", "base", config.to_dict(), measure_seed(config), str(device),
                       extra={"seeds": seed_streams(config), "model": chosen, "fingerprint": fp, "status": "started"})
        jobs = base_jobs(config)
        written: list[Path] = []
        if any(_incomplete(out, job, len(task.prompts())) for job in jobs):
            policy = (factory or default_policy_factory(config, device))(0)
            written += sample_jobs(policy, task, jobs, out, measure_seed(config), "base samples")
            model = _model_info(policy) or chosen
            del policy
        else:
            written += [sample_path(out, j.tag, j.set_name, j.temperature) for j in jobs]
            model = chosen
        rows = task.rows
        n_problems = len(rows)
        s, m = config.sampling, config.measure
        ref = float(s.reference_temperature)
        counts_a = load_counts(sample_path(out, "base", "A", ref), s.n_classify, n_problems)
        counts_b = load_counts(sample_path(out, "base", "B", ref), s.n_measure, n_problems)
        classes = classify(counts_a.successes, counts_a.trials, m.n, m.alpha)
        lower, upper = clopper_pearson(counts_a.successes, counts_a.trials, m.alpha)
        by_id = {str(i): str(c) for i, c in enumerate(classes)}
        classes_path = out / "base" / "classes.json"
        write_json(classes_path, {
            "n": m.n, "alpha": m.alpha, "set": "A", "temperature": ref, "classes": by_id,
            "successes_A": counts_a.successes.tolist(), "trials_A": counts_a.trials.tolist(),
            "cp_lower": lower.tolist(), "cp_upper": upper.tolist(),
        })
        labelled = [{**r, "class": by_id[str(r["id"])]} for r in rows]
        rate_b = counts_b.successes / counts_b.trials

        def rates(key: str | None) -> dict:
            groups: dict[str, list[int]] = {}
            for r in rows:
                groups.setdefault("all" if key is None else str(r[key]), []).append(int(r["id"]))
            return {g: float(rate_b[ids].mean()) for g, ids in sorted(groups.items())}

        class_counts = _tally(labelled, ["class"])
        overall = {c: class_counts.get(c, 0) for c in ("reachable", "unreachable", "ambiguous")}
        summary = {
            "n": m.n, "alpha": m.alpha,
            "classes": overall,
            "classes_by_split": _tally(labelled, ["split", "class"]),
            "classes_by_difficulty": _tally(labelled, ["difficulty", "class"]),
            "classes_by_split_difficulty": _tally(labelled, ["split", "difficulty", "class"]),
            "success_rate_B": {"all": rates(None)["all"], "by_split": rates("split"), "by_difficulty": rates("difficulty")},
        }
        summary_path = out / "base" / "summary.json"
        write_json(summary_path, summary)
        empty = [c for c, k in overall.items() if k == 0]
        if empty:
            bar = "!" * 78
            print(f"{bar}\nWARNING: base classes {empty} are EMPTY at N = {m.n} (alpha = {m.alpha}). The split "
                  f"cannot attribute gains to an empty class; consider other difficulties, measure.n or "
                  f"sampling.n_classify.\n{bar}", file=sys.stderr)
        manifest = write_manifest(out / "base", "base", config.to_dict(), measure_seed(config), str(device),
                                  extra={"seeds": seed_streams(config), "model": model, "fingerprint": fp,
                                         "status": "finished"})
    return written + [classes_path, summary_path, manifest]


def _incomplete(out: Path, job: SampleJob, n_problems: int) -> bool:
    return len(read_samples(sample_path(out, job.tag, job.set_name, job.temperature), job.n)) < job.n * n_problems


# --------------------------------------------------------------------------- train


def train_done(out: Path, config: LMConfig, arm: str, seed: int) -> bool:
    return adapter_path(out, arm, seed, config.budget).exists() and (run_dir(out, arm, seed) / "training.json").exists()


def _empty_log(trainer: Any, config: LMConfig) -> dict:
    """training.json for an arm that logs nothing (lm_none): no episodes sampled."""
    return {"trainer": getattr(trainer, "name", type(trainer).__name__), "updates": config.budget, "prompts": {},
            "totals": {"episodes_sampled": 0, "tokens_sampled": 0}, "note": "this arm samples nothing"}


def sft_updates(trainer: Any, budget: int) -> int:
    """Updates the trainer spends in SFT: all of them for lm_sft, round(sft_fraction * budget) for lm_sft_then_rl."""
    if hasattr(trainer, "sft_fraction"):
        from ..trainers.sft_rl import split_budget

        return int(split_budget(budget, trainer.sft_fraction)[0])
    return budget


def run_train(config: LMConfig, out: Path, device: Any = "cpu", factory: PolicyFactory | None = None,
              use_fallback: bool = False) -> list[Path]:
    """Train every arm x seed on the train split, saving adapters at every checkpoint."""
    _plugins()
    from ..registry import TRAINERS

    chosen = model_choice(config, use_fallback)
    fp = fingerprint(plain_config(config), chosen, "train")
    written: list[Path] = []
    with _stage(config, "train", device):
        task = load_task(config, out)
        _check_upstream(out, fp, base=True)
        train_ids = [int(r["id"]) for r in task.rows if r["split"] == "train"]
        if not train_ids:
            raise StageError("no train-split problems in data/problems.jsonl; raise task.generators[*].n_train")
        classes_file = out / "base" / "classes.json"
        for arm in config.arms:  # validate every arm before any training starts
            check = TRAINERS.create(arm.trainer, **dict(arm.params))
            if getattr(check, "needs_base_classes", False):
                check.set_base_classes(read_json(_require(classes_file, "base"))["classes"])
                sft = getattr(check, "sft_trainer", check)
                if sft_updates(check, config.budget) > 0 and hasattr(sft, "target_ids") and not sft.target_ids(train_ids):
                    raise StageError(f"arm {arm.name}: no train problem is in its SFT target class(es) "
                                     f"{list(sft.targets)} (see base/summary.json); change the arm's `target` "
                                     "or the problem difficulties")
        runs = [(arm, seed) for arm in config.arms for seed in config.seeds]
        todo = [(arm, seed) for arm, seed in runs if not train_done(out, config, arm.name, seed)]
        for arm, seed in runs:  # every finished run must match the current config before anything trains
            if (arm, seed) not in todo:
                directory = run_dir(out, arm.name, seed)
                if not check_manifest(directory / "manifest.train.json", "train", fp, run_keys(arm.name), [directory]):
                    _warn_unverified(f"training run {arm.name} seed {seed}", directory)
        for arm, seed in runs:
            if (arm, seed) not in todo:
                print(f"[train] {arm.name} seed {seed}: finished earlier, skipped", file=sys.stderr)
                written += [adapter_path(out, arm.name, seed, config.budget), run_dir(out, arm.name, seed) / "training.json"]
        policy = None
        for arm, seed in todo:
            trainer = TRAINERS.create(arm.trainer, **dict(arm.params))
            if getattr(trainer, "needs_base_classes", False):
                trainer.set_base_classes(read_json(_require(classes_file, "base"))["classes"])
            policy = (factory or default_policy_factory(config, device))(seed) if policy is None \
                else reseed_policy(policy, seed)
            directory = run_dir(out, arm.name, seed)
            model = _model_info(policy) or chosen
            run_extra = {"arm": arm.name, "trainer": arm.trainer, "seeds": seed_streams(config, seed),
                         "model": model, "train_ids": train_ids, "fingerprint": fp}
            write_manifest(directory, "train", config.to_dict(), seed, str(device), extra={**run_extra, "status": "started"})
            progress = Progress(f"train {arm.name} seed {seed}", config.budget)

            def on_checkpoint(step: int, p: Any, arm_name: str = arm.name, seed_: int = seed,
                              progress_: Progress = progress, model: dict | None = model) -> None:
                path = p.save_adapter(adapter_path(out, arm_name, seed_, step))
                write_manifest(path.parent, "step", config.to_dict(), seed_, str(device),
                               extra={"arm": arm_name, "step": step, "seeds": seed_streams(config, seed_),
                                      "model": model, "fingerprint": fp})
                progress_.update(step)

            t0 = time.monotonic()
            try:
                trainer.train(policy, task, config.budget, seed, train_ids, list(config.checkpoints), on_checkpoint)
            finally:
                progress.close()
            seconds = time.monotonic() - t0
            write_json(directory / "timing.json", {"wall_seconds": seconds, "budget": config.budget,
                                                   "seconds_per_update": seconds / config.budget if config.budget else None})
            log = trainer.training_log() or _empty_log(trainer, config)
            manifest = write_manifest(directory, "train", config.to_dict(), seed, str(device),
                                      extra={**run_extra, "status": "finished"})
            write_json(directory / "training.json", log)  # last: marks the run finished
            written += [adapter_path(out, arm.name, seed, c) for c in config.checkpoints]
            written += [directory / "timing.json", directory / "training.json", manifest]
    return written


# --------------------------------------------------------------------------- measure


def measure_groups(rows: Sequence[dict]) -> dict[str, list[int]]:
    """all / train / heldout, then <split>:<difficulty>; only groups with >= 2 problems (the
    temperature baseline's split-half needs two)."""
    groups: dict[str, list[int]] = {"all": [int(r["id"]) for r in rows]}
    for split in ("train", "heldout"):
        groups[split] = [int(r["id"]) for r in rows if r["split"] == split]
    for d in sorted({str(r["difficulty"]) for r in rows}):
        for split in ("all", "train", "heldout"):
            groups[f"{split}:{d}"] = [int(r["id"]) for r in rows
                                      if r["difficulty"] == d and split in ("all", r["split"])]
    kept = {g: ids for g, ids in groups.items() if len(ids) >= 2}
    skipped = sorted(set(groups) - set(kept))
    if skipped:
        print(f"[measure] groups with fewer than 2 problems are skipped: {skipped}", file=sys.stderr)
    return kept


def n_train_from_log(log: dict, step: int) -> dict:
    """Mean episodes per training problem up to `step`, over problems the arm sampled (H51)."""
    prompts = log.get("prompts") or {}
    if not prompts:
        return {"value": None, "source": "training_log", "note": "the arm sampled no problems"}
    per = []
    for entry in prompts.values():
        by_cp = entry.get("episodes_by_checkpoint") or {}
        if str(step) not in by_cp:
            return {"value": None, "source": "training_log", "note": f"no episode counts at step {step}"}
        per.append(float(by_cp[str(step)]))
    sampled = [v for v in per if v > 0] or per
    return {"value": float(np.mean(sampled)), "min": float(min(sampled)), "max": float(max(sampled)),
            "source": "training_log", "note": f"averaged over {len(sampled)} of {len(per)} problems"}


def run_measure(config: LMConfig, out: Path, device: Any = "cpu", factory: PolicyFactory | None = None,
                all_checkpoints: bool = False, use_fallback: bool = False) -> list[Path]:
    """Sample every run's checkpoint(s) on all problems, then split, null check and budget curve."""
    _plugins()
    from ..budget import default_grid, resolvable_n_max
    from .measure import budget_curve, group_metrics, null_check

    s, m = config.sampling, config.measure
    ref = float(s.reference_temperature)
    steps = sorted(config.checkpoints) if all_checkpoints else [config.budget]
    chosen = model_choice(config, use_fallback)
    plain = plain_config(config)
    fp = fingerprint(plain, chosen, "measure")
    run_fp = fingerprint(plain, chosen, "eval")
    with _stage(config, "measure", device):
        task = load_task(config, out)
        rows = task.rows
        n_problems = len(rows)
        _require(out / "base" / "classes.json", "base")
        _check_upstream(out, fp, base=True)
        runs = [(arm.name, seed) for arm in config.arms for seed in config.seeds]
        for arm, seed in runs:
            if not train_done(out, config, arm, seed):
                raise StageError(f"run {arm} seed {seed} has not finished training; run the `train` stage first")
            directory = run_dir(out, arm, seed)
            if not check_manifest(directory / "manifest.train.json", "train", fp, run_keys(arm), [directory]):
                _warn_unverified(f"training run {arm} seed {seed}", directory)
            samples = eval_dir(out, arm, seed)
            if not check_manifest(samples / "manifest.eval.json", "eval", run_fp, run_keys(arm), [samples]) \
                    and _has_files(samples):
                _warn_unverified(f"eval samples of {arm} seed {seed}", samples)
        measure_keys = [k for k in fp if not k.startswith("arms.")]  # runs were checked one by one above
        check_manifest(out / "measure" / "manifest.measure.json", "measure", fp, measure_keys, [out / "measure"])
        for arm, seed in runs:
            write_manifest(eval_dir(out, arm, seed), "eval", config.to_dict(), measure_seed(config), str(device),
                           extra={"arm": arm, "seeds": seed_streams(config, seed), "model": chosen,
                                  "fingerprint": run_fp})
        jobs = [(arm, seed, step, SampleJob(run_tag(arm, seed, step), "eval", ref, s.n_trained, True))
                for arm, seed in runs for step in steps]
        policy = None
        written: list[Path] = []
        for arm, seed, step, job in jobs:
            if _incomplete(out, job, n_problems):
                if policy is None:
                    policy = (factory or default_policy_factory(config, device))(0)
                policy.load_adapter(_require(adapter_path(out, arm, seed, step), "train"))
                sample_jobs(policy, task, [job], out, measure_seed(config), f"measure {job.tag}")
            written.append(sample_path(out, job.tag, "eval", ref))
        del policy

        counts_a = load_counts(sample_path(out, "base", "A", ref), s.n_classify, n_problems)
        counts_b = load_counts(sample_path(out, "base", "B", ref), s.n_measure, n_problems)
        counts_null = load_counts(sample_path(out, "base", "null", ref), s.n_measure, n_problems)
        counts_t = {float(t): load_counts(sample_path(out, "base", "C", float(t)), s.n_temperature, n_problems)
                    for t in s.temperatures}
        groups = measure_groups(rows)
        top_groups = {g: ids for g, ids in groups.items() if g in ("all", "train", "heldout")}
        n_max = resolvable_n_max(s.n_classify, 1.0, m.alpha)

        records, curves = [], []
        progress = Progress("measure statistics", len(jobs))
        for k, (arm, seed, step, job) in enumerate(jobs):
            counts_1 = load_counts(sample_path(out, job.tag, "eval", ref), s.n_trained, n_problems)
            metrics = group_metrics(groups, counts_a, counts_b, counts_1, counts_t, m.n, m.alpha, m.n_boot,
                                    m.boot_seed, n_splits(config))
            records += [{"arm": arm, "seed": seed, "step": step, **gm.to_dict()}
                        for gm in metrics]
            log = read_json(run_dir(out, arm, seed) / "training.json")
            info = n_train_from_log(log, step)
            grid = sorted(set(default_grid(info["value"], floor=2 * n_max)) | {m.n})
            for group, raw in top_groups.items():
                ids = np.asarray(raw)
                a, b, t1 = counts_a.subset(ids), counts_b.subset(ids), counts_1.subset(ids)
                curve = budget_curve(a.successes, a.trials, b.rates(), t1.rates(), grid, m.alpha)
                curves.append({"arm": arm, "seed": seed, "step": step, "group": group, "n_train": info, "curve": curve})
            progress.update(k + 1)
        progress.close()

        nulls = []
        for group, raw in top_groups.items():
            ids = np.asarray(raw)
            a, b, nb = counts_a.subset(ids), counts_b.subset(ids), counts_null.subset(ids)
            result = null_check(a.successes, a.trials, b.rates(), nb.rates(), m.n, m.alpha, m.n_boot, m.boot_seed)
            nulls.append({"group": group, **result})
            if not result["passed"]:
                print(f"WARNING: null check FAILED for group {group}: the base-vs-base total gain interval "
                      f"[{result['total']['lo']:.4f}, {result['total']['hi']:.4f}] excludes 0", file=sys.stderr)
        meta = {"n": m.n, "alpha": m.alpha, "n_boot": m.n_boot, "boot_seed": m.boot_seed, "n_splits": n_splits(config),
                "reference_temperature": ref, "steps": steps, "groups": {g: len(i) for g, i in groups.items()}}
        results_path = out / "measure" / "results.json"
        null_path = out / "measure" / "null_check.json"
        curve_path = out / "measure" / "budget_curve.json"
        write_json(results_path, {**meta, "records": records})
        write_json(null_path, {**meta, "set": "null", "records": nulls})
        write_json(curve_path, {**meta, "n_max_resolvable": n_max, "records": curves})
        manifest = write_manifest(out / "measure", "measure", config.to_dict(), measure_seed(config), str(device),
                                  extra={"seeds": seed_streams(config), "all_checkpoints": all_checkpoints,
                                         "model": chosen, "fingerprint": fp})
    return written + [results_path, null_path, curve_path, manifest]


# --------------------------------------------------------------------------- report


def run_report(config: LMConfig, out: Path) -> list[Path]:
    try:
        from . import report  # type: ignore[attr-defined]
    except ImportError as error:
        if "report" not in str(error):
            raise
        raise StageError("the report module (capsplit.harness.lm.report) is not written yet; "
                         "the measure outputs are in measure/") from error
    result = report.build_report(out)
    if result is None:
        return []
    if isinstance(result, (str, Path)):
        return [Path(result)]
    return [Path(p) for p in result]
