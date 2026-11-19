"""Experiment configs: load and validate a JSON file into an `ExperimentConfig`.

Schema (every key except `factors`, `save_policies`, `base_seed` and `train_seed` is required)::

    {
     "name": "dummy",                                   # experiment name (a folder name)
     "task":    {"type": "tiny_prompt", "params": {}},  # registry name + constructor params
     "base":    {"type": "uniform_tabular", "params": {}},
     "backend": {"type": "j_only", "params": {}},
     "factors": {"task.length": [3]},                   # optional grid; see below
     "arms": [{"name": "none", "trainer": "none", "params": {}}],
     "seeds": [0, 1],                                   # run seeds
     "budget": 10,                                      # optimizer updates per arm
     "checkpoints": [0, 5, 10],                         # steps to measure; must include 0 and budget
     "measure": {"ns": [16, 64], "cs": [0.5, 1, 2], "group_by": ["difficulty", "split"]},
     "save_policies": "checkpoints",                    # "checkpoints" | "final" | "none"
     "base_seed": 0,                                    # optional: every run builds its base with this seed
     "train_seed": 0                                    # optional: every trainer uses this seed
    }

Seed streams. Without `base_seed`/`train_seed`, each run seed s seeds both the base build and
the trainer (the original behaviour). `base_seed` fixes the base for every run (runs then share
one cached base), and `train_seed` fixes training sampling; the run seeds in `seeds` still
index runs and supply whichever stream is not fixed. When either key is present it is part of
every cell config (so such results never share folders with seed-coupled ones); when both are
absent the cell config, and so every hash and file, is exactly as before.

A factor key is `<section>.<param>` with section one of task, base or backend; each value list
overrides that section's `params[<param>]` (a longer dotted path sets a nested dict entry).
The grid is the cartesian product of the factor lists, with keys in sorted order.

Measurement groups come from prompt tags (`group_by`): see `build_groups`.
"""
from __future__ import annotations

import copy
import itertools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .interfaces import MeasureSpec, Task
from .registry import BACKENDS, BASES, TASKS, TRAINERS, Registry

SECTIONS = ("task", "base", "backend")
SAVE_POLICIES = ("checkpoints", "final", "none")
_TOP_KEYS = {
    "name", "task", "base", "backend", "factors", "arms", "seeds", "budget", "checkpoints", "measure", "save_policies",
    "base_seed", "train_seed",
}
_REQUIRED = {"name", "task", "base", "backend", "arms", "seeds", "budget", "checkpoints", "measure"}
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ConfigError(ValueError):
    """The experiment config is invalid."""


@dataclass(frozen=True)
class Component:
    """A registry name and its constructor params."""

    type: str
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"type": self.type, "params": copy.deepcopy(self.params)}


@dataclass(frozen=True)
class Arm:
    """One post-training arm: a unique name, a registered trainer and its params."""

    name: str
    trainer: str
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "trainer": self.trainer, "params": copy.deepcopy(self.params)}


@dataclass(frozen=True)
class Measure:
    ns: tuple
    cs: tuple
    group_by: tuple
    primary: tuple | None = None  # (N, c) for headline numbers; None = largest N, c = 1

    def to_dict(self) -> dict:
        out: dict = {"ns": list(self.ns), "cs": list(self.cs), "group_by": list(self.group_by)}
        if self.primary is not None:  # omitted when unset, so older cell hashes don't change
            out["primary"] = {"n": self.primary[0], "c": self.primary[1]}
        return out


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    task: Component
    base: Component
    backend: Component
    factors: dict  # key -> list of values, keys sorted
    arms: tuple
    seeds: tuple
    budget: int
    checkpoints: tuple
    measure: Measure
    save_policies: str
    raw: dict  # the input config, unchanged, for config.json
    base_seed: int | None = None  # fixed base-build seed; None = the run seed
    train_seed: int | None = None  # fixed training seed; None = the run seed

    # ------------------------------------------------------------------ seed streams

    @property
    def separate_seeds(self) -> bool:
        """True when the config sets base_seed or train_seed (then seeds are recorded per run)."""
        return self.base_seed is not None or self.train_seed is not None

    def effective_base_seed(self, run_seed: int) -> int:
        return run_seed if self.base_seed is None else self.base_seed

    def effective_train_seed(self, run_seed: int) -> int:
        return run_seed if self.train_seed is None else self.train_seed

    # ------------------------------------------------------------------ grid

    def factor_combos(self) -> list[dict]:
        """Every factor combination as {key: value}, in a fixed order ([{}] without factors)."""
        keys = list(self.factors)
        return [dict(zip(keys, values)) for values in itertools.product(*(self.factors[k] for k in keys))]

    def resolved(self, combo: Mapping[str, Any]) -> dict[str, Component]:
        """task/base/backend components with the combination's overrides applied."""
        params = {s: copy.deepcopy(getattr(self, s).params) for s in SECTIONS}
        for key, value in combo.items():
            section, *path = key.split(".")
            target = params[section]
            for part in path[:-1]:
                target = target.setdefault(part, {})
                if not isinstance(target, dict):
                    raise ConfigError(f"factor {key!r} goes through a non-dict param")
            target[path[-1]] = copy.deepcopy(value)
        return {s: Component(getattr(self, s).type, params[s]) for s in SECTIONS}

    def cell_config(self, combo: Mapping[str, Any], arm: Arm) -> dict:
        """The config of one (factor combination, arm) cell: what its run directory hashes.

        base_seed / train_seed are included only when set, so older cell hashes don't change.
        """
        parts = self.resolved(combo)
        out = {
            "name": self.name,
            "factors": dict(combo),
            "task": parts["task"].to_dict(),
            "base": parts["base"].to_dict(),
            "backend": parts["backend"].to_dict(),
            "arm": arm.to_dict(),
            "budget": self.budget,
            "checkpoints": list(self.checkpoints),
            "measure": self.measure.to_dict(),
        }
        if self.base_seed is not None:
            out["base_seed"] = self.base_seed
        if self.train_seed is not None:
            out["train_seed"] = self.train_seed
        return out


# ---------------------------------------------------------------------- loading


def load_config(path: str | Path) -> ExperimentConfig:
    """Read and validate a JSON config file."""
    try:
        data = json.loads(Path(path).read_text())
    except json.JSONDecodeError as error:
        raise ConfigError(f"{path}: not valid JSON: {error}") from error
    return parse_config(data)


def _component(data: Any, section: str, registry: Registry) -> Component:
    if not isinstance(data, dict) or "type" not in data:
        raise ConfigError(f"{section} must be an object with a 'type'")
    extra = set(data) - {"type", "params"}
    if extra:
        raise ConfigError(f"{section}: unknown keys {sorted(extra)}")
    params = data.get("params", {})
    if not isinstance(params, dict):
        raise ConfigError(f"{section}.params must be an object")
    _check_registered(registry, data["type"], section)
    return Component(str(data["type"]), copy.deepcopy(params))


def _check_registered(registry: Registry, name: Any, where: str) -> None:
    if not isinstance(name, str) or name not in registry:
        known = ", ".join(registry.names()) or "none"
        raise ConfigError(f"{where}: unknown {registry.kind} {name!r}; registered: {known}")


def _int_list(value: Any, where: str, minimum: int | None = None) -> tuple:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{where} must be a non-empty list")
    for v in value:
        if not isinstance(v, int) or isinstance(v, bool) or (minimum is not None and v < minimum):
            raise ConfigError(f"{where} must hold integers" + (f" >= {minimum}" if minimum is not None else ""))
    if len(set(value)) != len(value):
        raise ConfigError(f"{where} has duplicates")
    return tuple(value)


def parse_config(data: Any) -> ExperimentConfig:
    """Validate a config dict (see the module docstring) and return an `ExperimentConfig`."""
    if not isinstance(data, dict):
        raise ConfigError("config must be a JSON object")
    unknown = set(data) - _TOP_KEYS
    if unknown:
        raise ConfigError(f"unknown config keys {sorted(unknown)}")
    missing = _REQUIRED - set(data)
    if missing:
        raise ConfigError(f"missing config keys {sorted(missing)}")

    name = data["name"]
    if not isinstance(name, str) or not _NAME.match(name):
        raise ConfigError(f"name {name!r} must be a plain folder name (letters, digits, '_', '-', '.')")

    task = _component(data["task"], "task", TASKS)
    base = _component(data["base"], "base", BASES)
    backend = _component(data["backend"], "backend", BACKENDS)

    factors_in = data.get("factors", {})
    if not isinstance(factors_in, dict):
        raise ConfigError("factors must be an object")
    factors: dict = {}
    for key in sorted(factors_in):
        section, _, rest = key.partition(".")
        if section not in SECTIONS or not rest or "" in rest.split("."):
            raise ConfigError(f"factor key {key!r} must look like task.<param>, base.<param> or backend.<param>")
        values = factors_in[key]
        if not isinstance(values, list) or not values:
            raise ConfigError(f"factor {key!r} must be a non-empty list of values")
        factors[key] = copy.deepcopy(values)

    arms_in = data["arms"]
    if not isinstance(arms_in, list) or not arms_in:
        raise ConfigError("arms must be a non-empty list")
    arms = []
    for i, arm in enumerate(arms_in):
        if not isinstance(arm, dict) or "name" not in arm or "trainer" not in arm:
            raise ConfigError(f"arms[{i}] must be an object with 'name' and 'trainer'")
        extra = set(arm) - {"name", "trainer", "params"}
        if extra:
            raise ConfigError(f"arms[{i}]: unknown keys {sorted(extra)}")
        if not isinstance(arm["name"], str) or not _NAME.match(arm["name"]):
            raise ConfigError(f"arms[{i}].name {arm['name']!r} must be a plain name")
        _check_registered(TRAINERS, arm["trainer"], f"arms[{i}] ({arm['name']})")
        params = arm.get("params", {})
        if not isinstance(params, dict):
            raise ConfigError(f"arms[{i}].params must be an object")
        arms.append(Arm(arm["name"], arm["trainer"], copy.deepcopy(params)))
    names = [a.name for a in arms]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ConfigError(f"duplicate arm names {duplicates}")

    seeds = _int_list(data["seeds"], "seeds", minimum=0)
    budget = data["budget"]
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
        raise ConfigError("budget must be a non-negative integer")
    checkpoints = tuple(sorted(_int_list(data["checkpoints"], "checkpoints")))
    if checkpoints[0] < 0 or checkpoints[-1] > budget:
        raise ConfigError(f"checkpoints must lie in [0, budget={budget}]")
    if 0 not in checkpoints or budget not in checkpoints:
        raise ConfigError(f"checkpoints must include 0 and budget={budget}")

    m = data["measure"]
    if not isinstance(m, dict) or "ns" not in m or "cs" not in m:
        raise ConfigError("measure must be an object with 'ns' and 'cs'")
    extra = set(m) - {"ns", "cs", "group_by", "primary"}
    if extra:
        raise ConfigError(f"measure: unknown keys {sorted(extra)}")
    ns = _int_list(m["ns"], "measure.ns", minimum=1)
    cs_in = m["cs"]
    if not isinstance(cs_in, list) or not cs_in:
        raise ConfigError("measure.cs must be a non-empty list")
    for c in cs_in:
        if isinstance(c, bool) or not isinstance(c, (int, float)) or c <= 0:
            raise ConfigError("measure.cs must hold positive numbers")
    cs = tuple(float(c) for c in cs_in)
    if len(set(cs)) != len(cs):
        raise ConfigError("measure.cs has duplicates")
    group_by = m.get("group_by", [])
    if not isinstance(group_by, list) or not all(isinstance(g, str) for g in group_by) or len(set(group_by)) != len(group_by):
        raise ConfigError("measure.group_by must be a list of distinct tag names")
    primary = None
    if "primary" in m:
        p = m["primary"]
        if not isinstance(p, dict) or set(p) != {"n", "c"}:
            raise ConfigError('measure.primary must be {"n": N, "c": c}')
        if p["n"] not in ns or float(p["c"]) not in cs:
            raise ConfigError("measure.primary must use an N in measure.ns and a c in measure.cs")
        primary = (int(p["n"]), float(p["c"]))

    save = data.get("save_policies", "checkpoints")
    if save not in SAVE_POLICIES:
        raise ConfigError(f"save_policies must be one of {SAVE_POLICIES}")

    fixed: dict[str, int | None] = {}
    for key in ("base_seed", "train_seed"):
        value = data.get(key)
        if key in data and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            raise ConfigError(f"{key} must be a non-negative integer")
        fixed[key] = value

    return ExperimentConfig(
        name=name,
        task=task,
        base=base,
        backend=backend,
        factors=factors,
        arms=tuple(arms),
        seeds=seeds,
        budget=budget,
        checkpoints=checkpoints,
        measure=Measure(ns, cs, tuple(group_by), primary),
        save_policies=save,
        raw=copy.deepcopy(data),
        base_seed=fixed["base_seed"],
        train_seed=fixed["train_seed"],
    )


# ---------------------------------------------------------------------- groups


def build_groups(task: Task, group_by: Sequence[str]) -> dict[str, tuple]:
    """Measurement groups from prompt tags, in a fixed order.

    "all"; then, for each tag in `group_by` order, one group per tag value (sorted); then one
    group per full combination of values, named by joining the values with "_" in `group_by`
    order (for example "hard_train"). Empty groups are skipped. With a single tag the
    combinations equal the single-value groups and are not repeated.
    """
    prompts = list(task.prompts())
    groups: dict[str, tuple] = {"all": tuple(p.id for p in prompts)}
    values = []
    for tag in group_by:
        seen = sorted({p.tags[tag] for p in prompts if tag in p.tags})
        if not seen:
            raise ConfigError(f"no prompt has the tag {tag!r} named in measure.group_by")
        values.append(seen)

    def add(name: str, ids: tuple) -> None:
        if not ids:
            return
        if name in groups:
            if groups[name] == ids:
                return
            raise ConfigError(f"group name {name!r} is ambiguous (tag values collide)")
        groups[name] = ids

    for tag, vals in zip(group_by, values):
        for v in vals:
            add(v, tuple(p.id for p in prompts if p.tags.get(tag) == v))
    if len(group_by) > 1:
        for combo in itertools.product(*values):
            ids = tuple(p.id for p in prompts if all(p.tags.get(t) == v for t, v in zip(group_by, combo)))
            add("_".join(combo), ids)
    return groups


def measure_spec(task: Task, measure: Measure) -> MeasureSpec:
    return MeasureSpec(ns=measure.ns, cs=measure.cs, groups=build_groups(task, measure.group_by))
