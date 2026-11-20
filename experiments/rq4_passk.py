"""RQ4: how often does the pass@k rule (D10) misdiagnose a run?

Design: docs/rq4-passk-design.md. Plan: docs/rq4-passk-plan.md. Pre-registration:
docs/rq4-preregistration.md (git tag rq4-frozen).

A *unit* is one (run, checkpoint, prompt group) from a harness results folder. Its per-prompt
success rates J0 and J1 come straight from the metrics records (`extra.j0`, `extra.j1`), and its
label is the record's 2x2 label at the primary (N, c).

Run: python experiments/rq4_passk.py {labels,score,srr-ndr,repro} ...
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PRIMARY_N, PRIMARY_C = 64, 1.0


@dataclass(frozen=True)
class Unit:
    arm: str
    trainer: str
    factors: tuple  # sorted (key, value) items
    seed: int
    step: int
    group: str
    j0: tuple
    j1: tuple
    label: str
    label_en: str


def load_units(
    results_dir: str | Path, n: int = PRIMARY_N, c: float = PRIMARY_C, groups: Sequence[str] | None = None
) -> list[Unit]:
    """Every unit listed by `<results_dir>/index.json`, skipping update 0 and `none`-trainer arms."""
    results_dir = Path(results_dir)
    index_path = results_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"no index.json at {index_path}; is this a harness results folder?")
    index = json.loads(index_path.read_text())
    wanted = set(groups) if groups else None
    units = []
    for run in index["runs"]:
        if run["trainer"] == "none":
            continue
        metrics = json.loads((results_dir / run["files"]["metrics"]).read_text())
        for rec in metrics["records"]:
            if rec["step"] == 0 or rec["n"] != n or rec["c"] != c:
                continue
            if wanted is not None and rec["group"] not in wanted:
                continue
            units.append(
                Unit(
                    arm=run["arm"],
                    trainer=run["trainer"],
                    factors=tuple(sorted(run["factors"].items())),
                    seed=int(run["seed"]),
                    step=int(rec["step"]),
                    group=rec["group"],
                    j0=tuple(float(x) for x in rec["extra"]["j0"]),
                    j1=tuple(float(x) for x in rec["extra"]["j1"]),
                    label=rec["label"],
                    label_en=rec["extra"]["label_en"],
                )
            )
    return units


def label_distribution(units: Sequence[Unit]) -> dict[str, dict[str, int]]:
    """group -> label -> count. Uses labels only; it never calls the pass@k rule."""
    out: dict[str, Counter] = {}
    for u in units:
        out.setdefault(u.group, Counter())[u.label] += 1
    return {g: dict(c) for g, c in out.items()}


# ----------------------------------------------------------------------------- command line


def _cmd_labels(args) -> None:
    dist = label_distribution(load_units(args.results_dir, groups=[args.group] if args.group else None))
    for group, counts in sorted(dist.items()):
        total = sum(counts.values())
        shares = ", ".join(f"{label} {count} ({count / total:.0%})" for label, count in sorted(counts.items()))
        print(f"{group}: {total} units: {shares}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    labels = sub.add_parser("labels", help="label distribution only (allowed before the freeze)")
    labels.add_argument("results_dir")
    labels.add_argument("--group")
    labels.set_defaults(func=_cmd_labels)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
