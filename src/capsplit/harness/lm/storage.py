"""Atomic files, resumable JSONL logs, run manifests and progress output for every stage.

Why: every stage must be safe to stop and restart. Whole files are written to a temporary name
and renamed (atomic on one filesystem); append-only logs are written one complete line at a
time, and a reader ignores a truncated last line, so an interrupted stage resumes cleanly.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

from capsplit.runlog import _git


def write_json(path: Path, data: Any) -> None:
    """Write `data` as sorted, indented JSON, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Append complete lines and flush to disk, so a crash loses at most the batch in progress."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> Iterator[dict]:
    """Every complete line; a truncated final line (from an interrupted write) is skipped."""
    if not path.exists():
        return
    with path.open() as handle:
        for line in handle:
            if not line.endswith("\n"):
                break
            yield json.loads(line)


def write_manifest(directory: Path, stage: str, config: dict, seed: int | None, device: str, extra: dict | None = None) -> Path:
    """Provenance for one stage output: commit, dirty flag, config, seed, device, versions."""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain") if commit is not None else None
    import torch
    import transformers

    manifest = {
        "stage": stage,
        "config": config,
        "seed": seed,
        "device": device,
        "commit": commit,
        "dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        **(extra or {}),
    }
    path = directory / f"manifest.{stage}.json"
    write_json(path, manifest)
    return path


class Progress:
    """One-line progress with elapsed time and an estimate of time remaining, on stderr."""

    def __init__(self, label: str, total: int, done: int = 0):
        self.label, self.total, self.start_done = label, max(total, 1), done
        self.done = done
        self.t0 = time.monotonic()

    def update(self, done: int) -> None:
        self.done = done
        elapsed = time.monotonic() - self.t0
        rate = (done - self.start_done) / elapsed if elapsed > 0 and done > self.start_done else 0.0
        remaining = (self.total - done) / rate if rate > 0 else float("nan")
        eta = "?" if remaining != remaining else _fmt(remaining)
        print(f"\r[{self.label}] {done}/{self.total} ({100 * done / self.total:.0f}%) elapsed {_fmt(elapsed)} "
              f"remaining ~{eta}   ", end="", file=sys.stderr, flush=True)

    def close(self) -> None:
        print(file=sys.stderr, flush=True)


def _fmt(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"
