"""Registries: name -> class, so a config can say {"type": "rloo"} and a new plug-in is one decorated class."""
from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._classes: dict[str, type] = {}

    def register(self, name: str) -> Callable[[type], type]:
        """Class decorator: `@TRAINERS.register("rloo")`."""

        def decorate(cls: type) -> type:
            if name in self._classes and self._classes[name] is not cls:
                raise ValueError(f"{self.kind} {name!r} is already registered to {self._classes[name].__name__}")
            self._classes[name] = cls
            setattr(cls, "registered_name", name)
            return cls

        return decorate

    def get(self, name: str) -> type:
        _load_plugins()
        if name not in self._classes:
            known = ", ".join(sorted(self._classes)) or "none"
            raise KeyError(f"unknown {self.kind} {name!r}; registered: {known}")
        return self._classes[name]

    def create(self, name: str, **params: Any) -> T:
        return self.get(name)(**params)

    def names(self) -> list[str]:
        _load_plugins()
        return sorted(self._classes)

    def __contains__(self, name: str) -> bool:
        _load_plugins()
        return name in self._classes


TASKS: Registry = Registry("task")
POLICIES: Registry = Registry("policy")
TRAINERS: Registry = Registry("trainer")
BACKENDS: Registry = Registry("backend")
BASES: Registry = Registry("base recipe")

# Modules that register plug-ins. A module that doesn't exist yet is skipped, so the registry
# works while plug-ins are being added. Any other import error (a bug, a missing dependency,
# a missing sibling module) is raised, every time.
PLUGIN_MODULES = (
    "capsplit.harness.dummy",
    "capsplit.harness.policies.tabular",
    "capsplit.harness.trainers.none",
    "capsplit.harness.trainers.rl",
    "capsplit.harness.trainers.sft",
    "capsplit.harness.trainers.hints",
    "capsplit.harness.trainers.sft_rl",
    "capsplit.harness.trainers.tabular",
    "capsplit.harness.backends.exact",
    "capsplit.harness.backends.sampled",
    "capsplit.harness.tasks.rule_task",
    "capsplit.harness.tasks.grid",
    "capsplit.harness.policies.prompt_gru",
    "capsplit.harness.bases.pretrain",
)

_loaded = False


def _load_plugins() -> None:
    global _loaded
    if _loaded:
        return
    import importlib

    for module in PLUGIN_MODULES:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as error:
            if not _is_missing(module, error.name):
                raise
    _loaded = True  # only after every plug-in imported, so a failure raises again next time


def _is_missing(module: str, missing: str | None) -> bool:
    """True if the error means `module` itself (or a package containing it) doesn't exist."""
    return missing is not None and (module == missing or module.startswith(missing + "."))
