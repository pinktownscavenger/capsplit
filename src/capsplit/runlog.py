"""Run directories: one per experiment config, with a manifest and one file per seed."""
import hashlib
import json
import pathlib
import platform
import subprocess

import numpy as np


def _git(*args: str) -> str | None:
    """Output of a git command in the working directory, or None if git fails for any reason."""
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _manifest(config: dict) -> dict:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain") if commit is not None else None
    return {
        "config": config,
        "commit": commit,
        "dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def run_dir(experiment: str, config: dict, root: str = "results") -> pathlib.Path:
    """Return results/<experiment>/<first 12 chars of the config hash>/, writing manifest.json once."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    directory = pathlib.Path(root) / experiment / digest
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "manifest.json"
    if not manifest.exists():
        manifest.write_text(json.dumps(_manifest(config), indent=2, sort_keys=True) + "\n")
    return directory


def save_seed(directory: pathlib.Path, seed: int, **arrays) -> pathlib.Path:
    path = pathlib.Path(directory) / f"seed-{seed}.npz"
    np.savez(path, seed=seed, **arrays)
    return path


def load_seeds(directory: pathlib.Path) -> list[dict]:
    """Every saved seed in the directory as {"seed": int, <name>: array}, sorted by seed."""
    records = []
    for path in pathlib.Path(directory).glob("seed-*.npz"):
        with np.load(path) as data:
            record = {name: data[name] for name in data.files}
        record["seed"] = int(record["seed"])
        records.append(record)
    return sorted(records, key=lambda record: record["seed"])
