"""Configuration for the language-model experiment: one JSON file drives every stage.

Why a separate schema from `capsplit.harness.config`: the LM experiment is staged (data, base,
train, measure, report run as separate, resumable commands) and needs model, LoRA, memory and
sampling settings the toy runner does not have. The validation style is the same: every section
is a frozen dataclass, and unknown keys raise `LMConfigError` so a typo fails loudly.
"""
from __future__ import annotations

import json
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


class LMConfigError(ValueError):
    """The config file is invalid; the message says which key and what is expected."""


def _take(section: str, raw: Any, allowed: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    """Validate one section: it must be an object, use only `allowed` keys, and have `required` keys.

    Returns the allowed defaults overlaid with the given values.
    """
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise LMConfigError(f"{section} must be a JSON object, got {type(raw).__name__}")
    unknown = sorted(set(raw) - set(allowed))
    if unknown:
        raise LMConfigError(f"{section}: unknown keys {unknown}; allowed keys are {sorted(allowed)}")
    missing = [k for k in required if k not in raw]
    if missing:
        raise LMConfigError(f"{section}: missing required keys {missing}")
    return {**allowed, **raw}


@dataclass(frozen=True)
class ModelConfig:
    """Which checkpoint to load and how. `revision` pins an exact commit on the Hugging Face hub."""

    id: str
    revision: str
    fallback_id: str | None = None
    fallback_revision: str | None = None
    dtype: str = "float16"  # weights in memory: float16 | bfloat16 | float32
    device: str = "auto"  # auto (mps if available, else cpu) | mps | cpu
    cache_dir: str | None = None  # None = the Hugging Face default cache
    max_new_tokens: int = 12
    system_prompt: str = "You are a precise calculator."


@dataclass(frozen=True)
class LoraConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: tuple = ("q_proj", "v_proj")


@dataclass(frozen=True)
class MemoryConfig:
    """Guards checked before every stage, so the run fails early instead of swapping the machine."""

    min_free_gb: float = 4.0  # refuse to start a stage with less system memory available
    max_process_gb: float = 9.0  # abort a stage if this process's memory (RSS + MPS) exceeds it
    # On MPS, also cap the Metal allocator at max_process_gb (torch.mps.set_per_process_memory_fraction),
    # so a transient peak inside one update raises an out-of-memory error instead of swapping the machine.
    mps_hard_cap: bool = True


@dataclass(frozen=True)
class RuntimeConfig:
    """Keep long stages from making the machine unusable."""

    nice: int = 10  # added to the process's niceness (lower CPU priority)
    threads: int = 4  # torch CPU threads


@dataclass(frozen=True)
class GeneratorConfig:
    """One family of generated problems. `type` names a registered generator (see tasks.py)."""

    type: str
    params: dict = field(default_factory=dict)
    n_train: int = 0
    n_heldout: int = 0


@dataclass(frozen=True)
class TaskConfig:
    generators: tuple
    seed: int = 0
    prompt_template: str = "Compute {question}. Reply with only the final number."


@dataclass(frozen=True)
class SamplingConfig:
    """How the base and trained policies are sampled for measurement."""

    n_classify: int = 64  # set A: per problem, only to classify reachability
    n_measure: int = 64  # set B: per problem, only to measure the base success rate
    n_trained: int = 64  # per problem, for each trained policy
    temperatures: tuple = (0.5, 0.7, 1.0, 1.3)  # base temperature sweep (the sharpening baseline)
    n_temperature: int = 32  # per problem and temperature in the sweep
    reference_temperature: float = 1.0  # sets A and B and every trained policy are sampled at this
    top_p: float = 1.0
    batch_size: int = 64  # sequences generated at once
    seed: int = 0  # measurement-sampling seed stream (docs/lm_design.md)


@dataclass(frozen=True)
class MeasureConfig:
    n: int = 16  # budget N: reachable iff success probability >= 1/N
    alpha: float = 0.05  # confidence level for classification intervals and bootstrap
    n_boot: int = 2000
    boot_seed: int = 0
    n_splits: int = 20  # random half-splits for the temperature baseline


@dataclass(frozen=True)
class ArmConfig:
    name: str
    trainer: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class LMConfig:
    name: str
    model: ModelConfig
    lora: LoraConfig
    memory: MemoryConfig
    runtime: RuntimeConfig
    task: TaskConfig
    sampling: SamplingConfig
    measure: MeasureConfig
    arms: tuple
    budget: int
    checkpoints: tuple
    seeds: tuple
    raw: dict = field(compare=False, repr=False, default_factory=dict)

    def to_dict(self) -> dict:
        out = asdict(self)
        out.pop("raw")
        return out


_TOP = {
    "name": None,
    "model": None,
    "lora": None,
    "memory": None,
    "runtime": None,
    "task": None,
    "sampling": None,
    "measure": None,
    "arms": None,
    "budget": None,
    "checkpoints": None,
    "seeds": None,
}


def _section(cls: type, name: str, raw: Any, required: tuple[str, ...] = ()) -> Any:
    """Build dataclass `cls` from `raw`, using the dataclass's own defaults; JSON lists become tuples."""
    allowed: dict[str, Any] = {}
    for f in fields(cls):
        if f.default is not MISSING:
            allowed[f.name] = f.default
        elif f.default_factory is not MISSING:
            allowed[f.name] = f.default_factory()
        else:
            allowed[f.name] = None
    values = _take(name, raw, allowed, required)
    return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in values.items()})


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise LMConfigError(f"{where} must be a positive integer, got {value!r}")
    return value


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def parse_config(data: dict) -> LMConfig:
    """Validate a parsed JSON object and return the typed config."""
    top = _take("config", data, _TOP, required=tuple(_TOP))
    if not isinstance(top["name"], str) or not top["name"]:
        raise LMConfigError("name must be a non-empty string")
    model = _section(ModelConfig, "model", top["model"], required=("id", "revision"))
    if model.dtype not in ("float16", "bfloat16", "float32"):
        raise LMConfigError(f"model.dtype must be float16, bfloat16 or float32, got {model.dtype!r}")
    if model.device not in ("auto", "mps", "cpu"):
        raise LMConfigError(f"model.device must be auto, mps or cpu, got {model.device!r}")
    lora = _section(LoraConfig, "lora", top["lora"])
    memory = _section(MemoryConfig, "memory", top["memory"])
    if not isinstance(memory.mps_hard_cap, bool):
        raise LMConfigError(f"memory.mps_hard_cap must be true or false, got {memory.mps_hard_cap!r}")
    if isinstance(memory.max_process_gb, bool) or not isinstance(memory.max_process_gb, (int, float)) \
            or memory.max_process_gb <= 0:
        raise LMConfigError(f"memory.max_process_gb must be a positive number, got {memory.max_process_gb!r}")
    runtime = _section(RuntimeConfig, "runtime", top["runtime"])
    task_raw = _take("task", top["task"], {"generators": None, "seed": 0, "prompt_template": TaskConfig.prompt_template},
                     required=("generators",))
    if not isinstance(task_raw["generators"], list) or not task_raw["generators"]:
        raise LMConfigError("task.generators must be a non-empty list")
    generators = tuple(
        _section(GeneratorConfig, f"task.generators[{i}]", g, required=("type",)) for i, g in enumerate(task_raw["generators"])
    )
    task = TaskConfig(generators=generators, seed=task_raw["seed"], prompt_template=task_raw["prompt_template"])
    sampling = _section(SamplingConfig, "sampling", top["sampling"])
    if not isinstance(sampling.temperatures, tuple) or not sampling.temperatures or \
            not all(_is_number(t) and t > 0 for t in sampling.temperatures):
        raise LMConfigError(f"sampling.temperatures must be a non-empty list of positive numbers, "
                            f"got {sampling.temperatures!r}")
    if not _is_number(sampling.top_p) or not 0 < sampling.top_p <= 1:
        raise LMConfigError(f"sampling.top_p must be a number in (0, 1], got {sampling.top_p!r}")
    if sampling.reference_temperature not in sampling.temperatures:
        raise LMConfigError("sampling.reference_temperature must be one of sampling.temperatures")
    measure = _section(MeasureConfig, "measure", top["measure"])
    if not isinstance(top["arms"], list) or not top["arms"]:
        raise LMConfigError("arms must be a non-empty list")
    arms = tuple(_section(ArmConfig, f"arms[{i}]", a, required=("name", "trainer")) for i, a in enumerate(top["arms"]))
    names = [a.name for a in arms]
    if len(set(names)) != len(names):
        raise LMConfigError(f"arm names must be unique, got {names}")
    budget = top["budget"]
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise LMConfigError(f"budget must be a non-negative integer, got {budget!r}")
    checkpoints = tuple(top["checkpoints"]) if isinstance(top["checkpoints"], list) else ()
    if not checkpoints or 0 not in checkpoints or budget not in checkpoints or any(
        not isinstance(c, int) or c < 0 or c > budget for c in checkpoints
    ):
        raise LMConfigError("checkpoints must be a list of update counts within [0, budget] containing 0 and budget")
    seeds = tuple(top["seeds"]) if isinstance(top["seeds"], list) else ()
    if not seeds or any(isinstance(s, bool) or not isinstance(s, int) for s in seeds):
        raise LMConfigError("seeds must be a non-empty list of integers")
    for where, value in (("sampling.n_classify", sampling.n_classify), ("sampling.n_measure", sampling.n_measure),
                         ("sampling.n_trained", sampling.n_trained), ("sampling.n_temperature", sampling.n_temperature),
                         ("sampling.batch_size", sampling.batch_size), ("measure.n", measure.n),
                         ("measure.n_splits", measure.n_splits), ("measure.n_boot", measure.n_boot)):
        _positive_int(value, where)
    return LMConfig(top["name"], model, lora, memory, runtime, task, sampling, measure, arms, budget,
                    tuple(sorted(set(checkpoints))), seeds, raw=data)


def load_config(path: str | Path) -> LMConfig:
    """Read and validate a config file."""
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise LMConfigError(f"config file {path} does not exist") from error
    except json.JSONDecodeError as error:
        raise LMConfigError(f"{path} is not valid JSON: {error}") from error
    return parse_config(data)
