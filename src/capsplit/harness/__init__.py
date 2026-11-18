"""Experiment harness: plug-in tasks, policies, post-training arms and measurement backends.

See docs/harness.md. Importing this package does not import any plug-in; the registries load
them on first use.
"""
from .interfaces import (
    Backend,
    BaseRecipe,
    CheckpointHook,
    GroupMetrics,
    MeasureSpec,
    Policy,
    Prompt,
    Rollout,
    Task,
    Trainer,
    call_checkpoints,
)
from .registry import BACKENDS, BASES, POLICIES, TASKS, TRAINERS, Registry

__all__ = [
    "BACKENDS",
    "BASES",
    "Backend",
    "BaseRecipe",
    "CheckpointHook",
    "GroupMetrics",
    "MeasureSpec",
    "POLICIES",
    "Policy",
    "Prompt",
    "Registry",
    "Rollout",
    "TASKS",
    "TRAINERS",
    "Task",
    "Trainer",
    "call_checkpoints",
]
