"""Device choice, memory guards and process niceness for running a model on a laptop.

Why: this runs on the user's working Mac (18 GB, shared with other apps). Every stage checks
free memory before it starts and aborts with a clear message if the process grows past its cap,
instead of pushing the machine into swap.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import psutil
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
                f"{self.config.max_process_gb}. Lower sampling.batch_size or the trainer's group size and "
                "re-run; completed batches are kept and will be skipped."
            )


def apply_runtime(config: RuntimeConfig) -> None:
    """Lower this process's CPU priority and cap torch threads so the machine stays usable."""
    if config.nice:
        try:
            os.nice(config.nice)
        except OSError:  # not permitted on some systems; running at normal priority is still correct
            print("[capsplit.lm] could not lower process priority; continuing", file=sys.stderr)
    torch.set_num_threads(max(1, config.threads))
