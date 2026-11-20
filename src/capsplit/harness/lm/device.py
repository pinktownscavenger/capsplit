"""Device choice, memory guards and process niceness for running a model on a laptop.

Why: this runs on the user's working Mac (18 GB, shared with other apps). Every stage checks
free memory before it starts and aborts with a clear message if the process grows past its cap,
instead of pushing the machine into swap.

The between-batch guard (`MemoryGuard.check_running`) cannot see a transient peak inside one
batch or update. On MPS, `limit_mps_memory` therefore also sets a hard cap on the Metal allocator
(`torch.mps.set_per_process_memory_fraction`) at `memory.max_process_gb`, so an allocation past it
raises torch's out-of-memory RuntimeError at once; `explain_oom` turns that error into a
`MemoryGuardError` naming the config values to lower. `memory.mps_hard_cap = false` disables it.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import psutil  # type: ignore[import-untyped]  # no stubs in the lm venv
import torch

from .config import MemoryConfig, RuntimeConfig

GB = 1024**3


class MemoryGuardError(RuntimeError):
    """Not enough memory to start or continue a stage. The message says what to change."""


def pick_device(requested: str) -> torch.device:
    """`auto` -> mps if available, else cpu (with a message saying why); `mps`/`cpu` as asked."""
    mps_ok = torch.backends.mps.is_available()
    if requested == "mps" and not mps_ok:
        raise RuntimeError("model.device is 'mps' but PyTorch reports MPS unavailable; set model.device to 'auto' or 'cpu'")
    if requested == "cpu" or (requested == "auto" and not mps_ok):
        if requested == "auto":
            print("[capsplit.lm] MPS is not available, falling back to CPU (much slower).", file=sys.stderr)
        return torch.device("cpu")
    return torch.device("mps")


def dtype_of(name: str, device: torch.device) -> torch.dtype:
    """float16 on CPU is slow and poorly supported, so CPU always uses float32."""
    if device.type == "cpu":
        return torch.float32
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def available_gb() -> float:
    return psutil.virtual_memory().available / GB


def process_gb() -> float:
    """Resident memory of this process plus memory held by the MPS allocator."""
    used = psutil.Process().memory_info().rss
    if torch.backends.mps.is_available():
        used += torch.mps.driver_allocated_memory()
    return used / GB


@dataclass
class MemoryGuard:
    config: MemoryConfig

    def check_start(self, stage: str) -> None:
        """Refuse to start a stage when the system has less than `min_free_gb` available."""
        free = available_gb()
        if free < self.config.min_free_gb:
            raise MemoryGuardError(
                f"{stage}: only {free:.1f} GB of memory is available, below memory.min_free_gb = "
                f"{self.config.min_free_gb}. Close other apps, or lower sampling.batch_size / use the "
                "fallback model, then re-run; finished work is kept."
            )

    def check_running(self, stage: str) -> None:
        """Abort if this process has grown past `max_process_gb` (call between batches)."""
        used = process_gb()
        if used > self.config.max_process_gb:
            raise MemoryGuardError(
                f"{stage}: this process uses {used:.1f} GB, above memory.max_process_gb = "
                f"{self.config.max_process_gb}. Lower sampling.batch_size, the trainer's micro_batch or group size, and "
                "re-run; completed batches are kept and will be skipped."
            )


def mps_memory_fraction(max_process_gb: float, recommended_bytes: int) -> float:
    """The `set_per_process_memory_fraction` value capping MPS at `max_process_gb`, clamped to (0, 1].

    torch's limit is fraction * `torch.mps.recommended_max_memory()`; a fraction of 0 would mean "no
    limit", so the result is always > 0, and a cap above the recommended maximum is clamped to 1.
    """
    if max_process_gb <= 0:
        raise ValueError(f"memory.max_process_gb must be positive, got {max_process_gb}")
    if recommended_bytes <= 0:
        raise ValueError(f"recommended MPS memory must be positive, got {recommended_bytes}")
    return min(1.0, max_process_gb * GB / recommended_bytes)


def limit_mps_memory(memory: MemoryConfig, device: torch.device | str | None = None) -> float | None:
    """Hard-cap the MPS allocator at `memory.max_process_gb`; returns the fraction set, or None.

    Does nothing (None) when `memory.mps_hard_cap` is false, MPS is unavailable, or `device` is
    given and is not MPS. Idempotent: calling it again sets the same fraction.
    """
    if not memory.mps_hard_cap or not torch.backends.mps.is_available():
        return None
    if device is not None and torch.device(device).type != "mps":
        return None
    fraction = mps_memory_fraction(memory.max_process_gb, torch.mps.recommended_max_memory())
    torch.mps.set_per_process_memory_fraction(fraction)
    return fraction


_OOM_MARKERS = ("out of memory", "can't allocate memory", "cannot allocate memory")


def explain_oom(error: BaseException, memory: MemoryConfig | None = None,
                stage: str = "this stage") -> MemoryGuardError | None:
    """A `MemoryGuardError` explaining a torch out-of-memory error (MPS or CPU), or None for other errors.

    Use as `except RuntimeError as e: raise (explain_oom(e, cfg.memory, stage) or e) from e`.
    """
    if isinstance(error, MemoryGuardError) or not isinstance(error, RuntimeError):
        return None
    text = str(error)
    if not any(marker in text.lower() for marker in _OOM_MARKERS):
        return None
    where = "MPS" if "mps" in text.lower() else "memory"
    cap = ""
    if memory is not None:
        cap = (f" (memory.max_process_gb = {memory.max_process_gb}"
               + (", enforced on MPS by memory.mps_hard_cap = true" if memory.mps_hard_cap else "") + ")")
    first = text.strip().splitlines()[0] if text.strip() else type(error).__name__
    return MemoryGuardError(
        f"{stage}: ran out of {where}{cap}. Lower, in this order: the lm_rl / lm_sft_then_rl.rl param "
        "`micro_batch` (sequences per backward pass), sampling.batch_size (sequences per sampling or "
        "scoring forward), lm_rl `group` x `problems_per_update` (completions per update), lm_sft "
        "`batch`, or model.max_new_tokens; or use the fallback model. Raise memory.max_process_gb only if "
        "the machine has that much free. Completed work is kept. Torch said: " + first
    )


def apply_runtime(config: RuntimeConfig, memory: MemoryConfig | None = None,
                  device: torch.device | str | None = None) -> float | None:
    """Lower this process's CPU priority and cap torch threads so the machine stays usable.

    With `memory`, also apply the MPS hard cap (`limit_mps_memory`) and return its fraction.
    """
    if config.nice:
        try:
            os.nice(config.nice)
        except OSError:  # not permitted on some systems; running at normal priority is still correct
            print("[capsplit.lm] could not lower process priority; continuing", file=sys.stderr)
    torch.set_num_threads(max(1, config.threads))
    return None if memory is None else limit_mps_memory(memory, device)
