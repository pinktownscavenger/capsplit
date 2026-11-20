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
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from pilot_rq4 import k_grid

from capsplit.diagnostics import passk_rule, score_rule
from capsplit.harness.report import bootstrap_ci

PRIMARY_N, PRIMARY_C = 64, 1.0
K_LARGE = 10  # frozen before the pilot; re-frozen by docs/rq4-preregistration.md
TOL = 1e-9
PRIMARY_GROUP = "heldout"
MIN_CELL_UNITS = 10
AGREEMENT_LINE = 0.95
CELLS = ("capability", "compounded_sharpening", "reweighting", "pure_sharpening")
FREEZE_TAG = "rq4-frozen"


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


# ----------------------------------------------------------------------------- scoring


def rule_of(unit: Unit, ks: Sequence[int], k_large: int = K_LARGE) -> str:
    """The D10 pass@k rule's call on one unit: sharpening, capability or undetermined."""
    return passk_rule(list(unit.j0), list(unit.j1), list(ks), k_large, tol=TOL)


def _primary_agreement(rules: Sequence[str], labels: Sequence[str]) -> float:
    return score_rule(list(rules), list(labels))["primary"]


def score_units(units: Sequence[Unit], ks: Sequence[int], k_large: int = K_LARGE) -> dict:
    """D10 scoring (score_rule) plus per-seed primary agreement and its bootstrap interval.

    Each seed's agreement pools that seed's units over arms, factors and checkpoints given, so the
    bootstrap resamples seeds jointly, as RQ3 did.
    """
    rules = [rule_of(u, ks, k_large) for u in units]
    labels = [u.label for u in units]
    out = score_rule(rules, labels)
    per_seed: dict[int, float] = {}
    for seed in sorted({u.seed for u in units}):
        picked = [i for i, u in enumerate(units) if u.seed == seed]
        per_seed[seed] = _primary_agreement([rules[i] for i in picked], [labels[i] for i in picked])
    out.update(per_seed_primary=per_seed, ci=bootstrap_ci([per_seed[s] for s in sorted(per_seed)]),
               n_units=len(units), rules=rules)
    return out


def q1_verdict(ci: tuple[float, float]) -> str:
    """Q1 and Q2: supported if the CI's upper end < 0.95; contradicted if its lower end > 0.95."""
    lo, hi = ci
    if hi < AGREEMENT_LINE:
        return "supported"
    if lo > AGREEMENT_LINE:
        return "contradicted"
    return "inconclusive"


def q3_verdict(per_cell: dict) -> dict:
    """Q3: disagreement (1 - primary agreement) is highest in compounded sharpening and reweighting.

    Supported if each of those two cells' disagreement exceeds both other cells'; contradicted if
    either of them has the lowest of the four. Inconclusive otherwise, or if any cell is missing
    or has fewer than MIN_CELL_UNITS units.
    """
    small = [c for c in CELLS if c not in per_cell or per_cell[c]["runs"] < MIN_CELL_UNITS]
    if small:
        return {"verdict": "inconclusive", "reason": f"cells with fewer than {MIN_CELL_UNITS} units: {small}"}
    dis = {c: 1.0 - per_cell[c]["primary"] for c in CELLS}
    focus, others = ("compounded_sharpening", "reweighting"), ("capability", "pure_sharpening")
    lowest = min(CELLS, key=lambda c: dis[c])
    if any(dis[c] == dis[lowest] for c in focus):
        verdict = "contradicted"
    elif all(dis[f] > dis[o] for f in focus for o in others):
        verdict = "supported"
    else:
        verdict = "inconclusive"
    return {"verdict": verdict, "disagreement": dis}


def _tag_is_ancestor() -> bool:
    result = subprocess.run(["git", "merge-base", "--is-ancestor", FREEZE_TAG, "HEAD"], capture_output=True)
    return result.returncode == 0


def require_frozen() -> None:
    """Refuse to score real units unless the pre-registration tag is an ancestor of HEAD."""
    if not _tag_is_ancestor():
        print(f"error: the tag {FREEZE_TAG} is not an ancestor of HEAD. Freeze docs/rq4-preregistration.md "
              f"and tag it before scoring (use --allow-unfrozen only in tests).", file=sys.stderr)
        raise SystemExit(2)


def phase_ks(results_dir: str | Path) -> tuple:
    """The rule's k grid for a harness results folder, from its task (D7's valid-N limit)."""
    config = json.loads((Path(results_dir) / "config.json").read_text())
    task = config["task"]
    if task["type"] == "rule_task":
        return k_grid(int(task["params"]["vocab"]), int(task["params"]["length"]))
    if task["type"] == "grid":
        return k_grid(4, int(task["params"].get("H", 8)))
    raise ValueError(f"no k grid rule for task type {task['type']!r}")


def write_report(phase2: dict, phase1: dict, out: Path, frozen: bool) -> dict:
    raise NotImplementedError("written in Task 4")


def _score_phase(results_dir: str) -> dict:
    ks = phase_ks(results_dir)
    units = load_units(results_dir)
    return {"results_dir": str(results_dir), "ks": ks, "units": units}


def _cmd_score(args) -> None:
    if not args.allow_unfrozen:
        require_frozen()
    phase2, phase1 = _score_phase(args.phase2), _score_phase(args.phase1)
    write_report(phase2, phase1, Path(args.out), frozen=not args.allow_unfrozen)


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
    score = sub.add_parser("score", help="apply the frozen rule and write the RQ4 report")
    score.add_argument("--phase2", required=True, help="RQ3 harness results folder (results/main)")
    score.add_argument("--phase1", required=True, help="grid sweep results folder (results/rq4_grid)")
    score.add_argument("--out", default="results/rq4_report")
    score.add_argument("--allow-unfrozen", action="store_true", help="tests only; stamps outputs UNFROZEN TEST")
    score.set_defaults(func=_cmd_score)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
