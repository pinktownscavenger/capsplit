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
import csv
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from pilot_rq4 import k_grid

import numpy as np

from capsplit.diagnostics import passk_curve, passk_rule, score_rule, srr_ndr
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


SENSITIVITY_K_LARGE = (5, 10, 20, 50)
REPORT_GROUPS = ("heldout", "all", "easy_train", "easy_heldout", "hard_train", "hard_heldout")
SRR_PENDING = "SRR/NDR pending a person's check of docs/notes/invisible-leash-scores.md."


def _score_phase(results_dir: str) -> dict:
    """Everything `write_report` needs about one phase's results folder."""
    index = json.loads((Path(results_dir) / "index.json").read_text())
    trained_arms = [a for a in index["arms"] if a["trainer"] != "none"]
    combos = {json.dumps(r["factors"], sort_keys=True) for r in index["runs"]}
    return {
        "results_dir": str(results_dir),
        "name": index["name"],
        "ks": phase_ks(results_dir),
        "final_step": int(index["budget"]),
        "expected_primary": len(trained_arms) * len(combos) * len(index["seeds"]),
        "units": load_units(results_dir),
    }


def _all_cells(per_cell: dict) -> dict:
    """score_rule's per-cell rates with every one of the four cells present (empty ones as runs = 0)."""
    empty = {"runs": 0, "primary": None, "secondary": None, "undetermined_rate": None}
    return {c: per_cell.get(c, empty) for c in CELLS}


def _primary_units(phase: dict) -> list[Unit]:
    return [u for u in phase["units"] if u.group == PRIMARY_GROUP and u.step == phase["final_step"]]


def _summary(score: dict) -> dict:
    return {k: score[k] for k in ("primary", "secondary", "undetermined_rate", "ci", "n_units")} | {
        "per_seed_primary": {str(k): v for k, v in score["per_seed_primary"].items()},
        "per_cell": _all_cells(score["per_cell"]),
    }


def write_report(phase2: dict, phase1: dict, out: Path, frozen: bool) -> dict:
    """Tables, figures and verdicts (design section 7). SRR/NDR never appear here."""
    out = Path(out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    phases = {"phase2": phase2, "phase1": phase1}
    primary = {name: _primary_units(ph) for name, ph in phases.items()}
    scores = {name: score_units(primary[name], ph["ks"]) for name, ph in phases.items()}

    # Q3 pools both phases' primary units, each scored with its own k grid.
    pooled_rules = scores["phase2"]["rules"] + scores["phase1"]["rules"]
    pooled_labels = [u.label for u in primary["phase2"] + primary["phase1"]]
    pooled = score_rule(pooled_rules, pooled_labels) if pooled_rules else {"per_cell": {}}

    rows = []
    for name, ph in phases.items():
        steps = sorted({u.step for u in ph["units"]})
        for group in REPORT_GROUPS:
            for step in steps:
                picked = [u for u in ph["units"] if u.group == group and u.step == step]
                if not picked:
                    continue
                sc = score_units(picked, ph["ks"])
                for cell, rates in _all_cells(sc["per_cell"]).items():
                    rows.append([name, group, step, cell, rates["runs"], rates["primary"], rates["secondary"],
                                 rates["undetermined_rate"]])
                rows.append([name, group, step, "ALL", sc["n_units"], sc["primary"], sc["secondary"],
                             sc["undetermined_rate"]])
    _csv(out / "tables/agreement.csv", ["phase", "group", "step", "cell", "units", "primary", "secondary",
                                         "undetermined_rate"], rows)

    sens = []
    for name, ph in phases.items():
        for k_large in SENSITIVITY_K_LARGE:
            if k_large <= max(ph["ks"]) and primary[name]:
                sc = score_units(primary[name], ph["ks"], k_large)
                sens.append([name, k_large, sc["primary"], sc["ci"][0], sc["ci"][1], sc["undetermined_rate"]])
    _csv(out / "tables/k_large_sensitivity.csv", ["phase", "k_large", "primary", "ci_low", "ci_high",
                                                  "undetermined_rate"], sens)

    unit_rows = [[name, u.arm, json.dumps(dict(u.factors), sort_keys=True), u.seed, u.step, u.group, len(u.j0),
                  u.label, u.label_en, rule]
                 for name in phases for u, rule in zip(primary[name], scores[name]["rules"])]
    _csv(out / "tables/units.csv", ["phase", "arm", "factors", "seed", "step", "group", "prompts", "label",
                                    "label_en", "rule"], unit_rows)

    validity = {
        "j_lists_consistent": all(len(u.j0) == len(u.j1) > 0 for ph in phases.values() for u in ph["units"]),
        "primary_sizes": {name: {"expected": ph["expected_primary"], "actual": len(primary[name])}
                          for name, ph in phases.items()},
    }
    validity["primary_sizes_match"] = all(v["expected"] == v["actual"] for v in validity["primary_sizes"].values())
    verdicts = {
        "frozen": frozen,
        "settings": {"k_large": K_LARGE, "tol": TOL, "group": PRIMARY_GROUP, "n": PRIMARY_N, "c": PRIMARY_C,
                     "ks": {name: list(ph["ks"]) for name, ph in phases.items()},
                     "final_step": {name: ph["final_step"] for name, ph in phases.items()}},
        "Q1": _summary(scores["phase2"]) | {"verdict": q1_verdict(scores["phase2"]["ci"])},
        "Q2": _summary(scores["phase1"]) | {"verdict": q1_verdict(scores["phase1"]["ci"])},
        "Q3": q3_verdict(_all_cells(pooled["per_cell"])) | {"per_cell": _all_cells(pooled["per_cell"])},
        "validity": validity,
        "overall": "valid" if validity["j_lists_consistent"] and validity["primary_sizes_match"] else "invalid",
    }
    (out / "verdicts.json").write_text(json.dumps(verdicts, indent=2, default=float) + "\n")
    (out / "verdicts.md").write_text(_verdicts_md(verdicts))
    _figures(out, phases, primary, scores)
    return verdicts


def _csv(path: Path, header: list, rows: list) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _verdicts_md(v: dict) -> str:
    lines = ["# RQ4 pass@k misdiagnosis verdicts", ""]
    if not v["frozen"]:
        lines += ["**UNFROZEN TEST: not the pre-registered result.**", ""]
    lines += [f"**Validity:** {v['overall']}", "",
              f"Primary units: group `{v['settings']['group']}`, final checkpoint, N = {v['settings']['n']}, "
              f"c = {v['settings']['c']}, k_large = {v['settings']['k_large']}.", ""]
    for key, title in (("Q1", "Phase 2 (RQ3 runs)"), ("Q2", "Phase 1 (grid sweep)")):
        q = v[key]
        lines += [f"## {key}: {title}: **{q['verdict']}**", "",
                  f"Primary agreement {_fmt(q['primary'])}, 95% CI [{_fmt(q['ci'][0])}, {_fmt(q['ci'][1])}], "
                  f"{q['n_units']} units; undetermined {_fmt(q['undetermined_rate'])}; secondary {_fmt(q['secondary'])}.",
                  "", "| Cell | Units | Primary | Secondary | Undetermined |", "|---|---|---|---|---|"]
        for cell, r in q["per_cell"].items():
            lines.append(f"| {cell} | {r['runs']} | {_fmt(r['primary'])} | {_fmt(r['secondary'])} | "
                         f"{_fmt(r['undetermined_rate'])} |")
        lines.append("")
    q3 = v["Q3"]
    lines += [f"## Q3: disagreement concentrates in compounded sharpening and reweighting: **{q3['verdict']}**", ""]
    if "disagreement" in q3:
        lines += [", ".join(f"{c} {_fmt(d)}" for c, d in q3["disagreement"].items()), ""]
    if "reason" in q3:
        lines += [q3["reason"], ""]
    lines += [SRR_PENDING, ""]
    return "\n".join(lines)


def _figures(out: Path, phases: dict, primary: dict, scores: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.5))
    width = 0.38
    for i, name in enumerate(phases):
        cells = _all_cells(scores[name]["per_cell"])
        values = [cells[c]["primary"] if cells[c]["primary"] is not None else np.nan for c in CELLS]
        ax.bar(np.arange(len(CELLS)) + (i - 0.5) * width, values, width,
               label=f"{name} ({scores[name]['n_units']} units)")
    ax.axhline(AGREEMENT_LINE, ls="--", color="grey", lw=0.8)
    ax.set(xticks=range(len(CELLS)), xticklabels=[c.replace("_", "\n") for c in CELLS], ylim=(0, 1.05),
           ylabel="primary agreement", title="pass@k rule against the 2x2 label (primary units)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "figures/agreement_by_cell.png", dpi=150)
    plt.close(fig)

    examples = []
    for name, ph in phases.items():
        for u, rule in zip(primary[name], scores[name]["rules"]):
            agree = (rule == "capability" and u.label == "capability") or (rule == "sharpening" and u.label != "capability")
            if not agree:
                examples.append((name, ph["ks"], u, rule))
    examples = examples[:3]
    fig, axes = plt.subplots(1, max(len(examples), 1), figsize=(4 * max(len(examples), 1), 3.2), squeeze=False)
    for ax, (name, ks, u, rule) in zip(axes[0], examples):
        ax.plot(ks, passk_curve(u.j0, ks), "o-", label="base")
        ax.plot(ks, passk_curve(u.j1, ks), "s-", label="trained")
        ax.axvline(K_LARGE, ls=":", color="grey")
        ax.set(xscale="log", xlabel="k", ylabel="mean pass@k", title=f"{name} {u.arm} seed {u.seed}\nrule {rule}, label {u.label}")
        ax.title.set_fontsize(8)
        ax.legend(fontsize=7)
    if not examples:
        axes[0][0].text(0.5, 0.5, "no disagreeing units", ha="center", va="center")
        axes[0][0].set_axis_off()
    fig.tight_layout()
    fig.savefig(out / "figures/disagreement_examples.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- SRR / NDR (unverified)


def srr_ndr_rows(results_dir: str | Path, n: int = PRIMARY_N, c: float = PRIMARY_C) -> list[list]:
    """Per-prompt SRR/NDR for every saved checkpoint after update 0, at eps = c / n. UNVERIFIED."""
    import torch

    from capsplit.engine import solution_probs
    from capsplit.harness.config import load_config
    from capsplit.harness.registry import TASKS
    from capsplit.harness.runner import _torch_load

    results_dir = Path(results_dir)
    index = json.loads((results_dir / "index.json").read_text())
    config = load_config(results_dir / "config.json")
    rows = []
    for run in index["runs"]:
        if run["trainer"] == "none" or not run["files"].get("policies"):
            continue
        part = config.resolved(run["factors"])["task"]
        task = TASKS.create(part.type, **part.params)
        base = _torch_load(results_dir / run["base"] / "policy.pt")
        saved = torch.load(results_dir / run["files"]["policies"], weights_only=False)
        q0 = {p.id: solution_probs(task.table(p.id), base.trajectory_probs(task, p.id)) for p in task.prompts()}
        for step, state in sorted(saved.items()):
            if int(step) == 0:
                continue
            trained = base.clone()
            trained.load_state_dict(state)
            for prompt in task.prompts():
                q1 = solution_probs(task.table(prompt.id), trained.trajectory_probs(task, prompt.id))
                r = srr_ndr(q0[prompt.id], q1, c / n)
                rows.append([index["name"], run["arm"], run["seed"], int(step), json.dumps(run["factors"], sort_keys=True),
                             prompt.id, prompt.tags.get("split", ""), r["P"], r["S"], r["E"], r["srr"], r["ndr"]])
    return rows


def _cmd_srr_ndr(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _csv(out / "srr_ndr_UNVERIFIED.csv", ["phase", "arm", "seed", "step", "factors", "prompt", "split", "P", "S", "E",
                                          "srr", "ndr"], srr_ndr_rows(args.results_dir))


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
    srr = sub.add_parser("srr-ndr", help="SRR/NDR per prompt into an UNVERIFIED file only")
    srr.add_argument("results_dir")
    srr.add_argument("--out", default="results/rq4_report")
    srr.set_defaults(func=_cmd_srr_ndr)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
