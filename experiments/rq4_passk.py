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

from capsplit.diagnostics import pool_srr_ndr, passk_curve, passk_rule, score_rule, srr_ndr
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


def srr_ndr_rows(results_dir: str | Path, n: int = PRIMARY_N, c: float = PRIMARY_C,
                 steps: Sequence[int] | None = None) -> list[list]:
    """Per-prompt SRR/NDR for every saved checkpoint after update 0 (or only `steps`), at eps = c / n."""
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
            if int(step) == 0 or (steps is not None and int(step) not in steps):
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


SRR_START, SRR_END = "<!-- srr-ndr:start -->", "<!-- srr-ndr:end -->"


def srr_ndr_units(rows: Sequence[Sequence], units: Sequence[Unit], group: str = PRIMARY_GROUP) -> list[dict]:
    """Pool `srr_ndr_rows` output over each unit's prompts in `group` (prompt split tag), with the unit's label.

    Raises KeyError if a pooled unit has no matching label in `units`.
    """
    labels = {(u.arm, json.dumps(dict(u.factors), sort_keys=True), u.seed, u.step): u.label
              for u in units if u.group == group}
    pooled: dict[tuple, list[dict]] = {}
    for _phase, arm, seed, step, factors, _prompt, split, p, s, e, *_ in rows:
        if split != group:
            continue
        pooled.setdefault((arm, factors, int(seed), int(step)), []).append({"P": p, "S": s, "E": e})
    out = []
    for key in sorted(pooled, key=lambda k: (k[0], k[1], k[2], k[3])):
        arm, factors, seed, step = key
        out.append({"arm": arm, "factors": factors, "seed": seed, "step": step, "group": group,
                    **pool_srr_ndr(pooled[key]), "label": labels[key]})
    return out


def srr_ndr_summary(unit_rows: Sequence[dict]) -> dict:
    """Counts pooled over all units and per 2x2 cell (every cell listed), plus how many units have a 0 denominator."""
    def pool(rows):
        return pool_srr_ndr(rows) | {"units": len(rows), "prompts": sum(int(r["prompts"]) for r in rows)}
    return {
        "overall": pool(unit_rows),
        "per_cell": {cell: pool([u for u in unit_rows if u["label"] == cell]) for cell in CELLS},
        "undefined_units": {"srr": sum(u["P"] + u["S"] == 0 for u in unit_rows),
                            "ndr": sum(u["P"] + u["E"] == 0 for u in unit_rows)},
    }


def _fmt(x) -> str:
    return "undefined" if x is None else f"{x:.3f}"


def srr_ndr_markdown(summaries: dict, settings: dict, missing: dict | None = None) -> list[str]:
    """The SRR/NDR section of verdicts.md (one table per phase; `missing` maps a skipped phase to the reason)."""
    lines = ["## SRR and NDR (The Invisible Leash, Def. 3.3)", "",
             f"Group `{settings['group']}`, final checkpoint, eps = c/N = {settings['eps']:.6g}, solution counted when "
             "its exact probability is > eps. P, S and E are summed over each unit's prompts and then over units "
             "before dividing; \"undefined\" means a zero denominator. Cells use each record's primary label "
             f"({settings['label']}).", ""]
    for name, title in (("phase2", "Phase 2 (RQ3 runs)"), ("phase1", "Phase 1 (grid sweep)")):
        if name not in summaries:
            if missing and name in missing:
                lines += [f"### {title}: not computed. {missing[name]}", ""]
            continue
        sm = summaries[name]
        o = sm["overall"]
        lines += [f"### {title}", "",
                  f"SRR {_fmt(o['srr'])}, NDR {_fmt(o['ndr'])} over {o['units']} units "
                  f"(P {o['P']}, S {o['S']}, E {o['E']}); units with undefined SRR {sm['undefined_units']['srr']}, "
                  f"NDR {sm['undefined_units']['ndr']}.", "",
                  "| Cell | Units | P | S | E | SRR | NDR |", "|---|---|---|---|---|---|---|"]
        for cell, r in sm["per_cell"].items():
            lines.append(f"| {cell} | {r['units']} | {r['P']} | {r['S']} | {r['E']} | {_fmt(r['srr'])} | {_fmt(r['ndr'])} |")
        lines.append("")
    return lines[:-1] if lines[-1] == "" else lines


def insert_srr_section(verdicts_md: str | Path, section: Sequence[str]) -> None:
    """Replace the SRR/NDR pending line (or an earlier generated section) in verdicts.md with `section`."""
    path = Path(verdicts_md)
    text = path.read_text()
    block = "\n".join([SRR_START, *section, SRR_END])
    if SRR_START in text and SRR_END in text:
        head, rest = text.split(SRR_START, 1)
        text = head + block + rest.split(SRR_END, 1)[1]
    elif SRR_PENDING in text:
        text = text.replace(SRR_PENDING, block)
    else:
        raise ValueError(f"{path} has neither the SRR/NDR pending line nor a generated section")
    path.write_text(text)


def _cmd_srr_ndr_report(args) -> None:
    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    eps = PRIMARY_C / PRIMARY_N
    summaries, table, missing = {}, [], {}
    for name, results_dir in (("phase2", args.phase2), ("phase1", args.phase1)):
        if results_dir is None:
            missing[name] = args.missing_reason
            continue
        final = int(json.loads((Path(results_dir) / "index.json").read_text())["budget"])
        rows = srr_ndr_rows(results_dir, steps=[final])
        units = [u for u in load_units(results_dir, groups=[PRIMARY_GROUP]) if u.step == final]
        unit_rows = srr_ndr_units(rows, units)
        summaries[name] = srr_ndr_summary(unit_rows)
        table += [[name, u["arm"], u["factors"], u["seed"], u["step"], u["group"], u["prompts"], u["P"], u["S"], u["E"],
                   u["srr"], u["ndr"], u["label"]] for u in unit_rows]
    settings = {"group": PRIMARY_GROUP, "n": PRIMARY_N, "c": PRIMARY_C, "eps": eps, "boundary": "strict > eps",
                "aggregation": "sum P, S, E over prompts and units, then divide", "zero_denominator": None,
                "label": "H12/H39, unratified"}
    _csv(out / "tables/srr_ndr_units.csv", ["phase", "arm", "factors", "seed", "step", "group", "prompts", "P", "S", "E",
                                            "srr", "ndr", "label"], table)
    (out / "srr_ndr.json").write_text(json.dumps({"settings": settings, "phases": summaries, "missing": missing},
                                                 indent=2) + "\n")
    insert_srr_section(out / "verdicts.md", srr_ndr_markdown(summaries, settings, missing))


def _cmd_score(args) -> None:
    if not args.allow_unfrozen:
        require_frozen()
    phase2, phase1 = _score_phase(args.phase2), _score_phase(args.phase1)
    write_report(phase2, phase1, Path(args.out), frozen=not args.allow_unfrozen)


# ----------------------------------------------------------------------------- RQ3 reproduction

REPRO_FILES = ("summary.json", "tables/main.csv", "verdicts.json")
REPRO_TOL = 1e-6


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _diff(a, b, path: str = "") -> tuple[float, str | None]:
    """(largest numeric difference, first non-numeric difference's path or None)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return 0.0, f"{path} keys {sorted(set(a) ^ set(b))[:3]}"
        worst = 0.0
        for k in sorted(a):
            d, where = _diff(a[k], b[k], f"{path}.{k}")
            if where:
                return d, where
            worst = max(worst, d)
        return worst, None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return 0.0, f"{path} length {len(a)} vs {len(b)}"
        worst = 0.0
        for i, (x, y) in enumerate(zip(a, b)):
            d, where = _diff(x, y, f"{path}[{i}]")
            if where:
                return d, where
            worst = max(worst, d)
        return worst, None
    x, y = _num(a), _num(b)
    if x is not None and y is not None and not isinstance(a, bool) and not isinstance(b, bool):
        if np.isnan(x) and np.isnan(y):
            return 0.0, None
        return abs(x - y), None
    return (0.0, None) if a == b else (0.0, f"{path}: {a!r} vs {b!r}")


def _parse(path: Path):
    if path.suffix == ".json":
        return json.loads(path.read_text())
    with path.open(newline="") as f:
        return list(csv.reader(f))


def repro_compare(new_report_dir: str | Path, committed_dir: str | Path = Path("results/published/experiment_main")) -> dict:
    """Compare a rerun RQ3 report with the committed one: identical bytes, numeric-only differences, or different."""
    new_report_dir, committed_dir = Path(new_report_dir), Path(committed_dir)
    files = {}
    for name in REPRO_FILES:
        new, old = new_report_dir / name, committed_dir / name
        if not new.exists() or not old.exists():
            files[name] = {"status": "missing", "where": str(new if not new.exists() else old)}
        elif new.read_bytes() == old.read_bytes():
            files[name] = {"status": "identical"}
        else:
            worst, where = _diff(_parse(old), _parse(new))
            files[name] = {"status": "different", "where": where} if where else {"status": "numeric", "max_abs_diff": worst}
    match = all(f["status"] == "identical" or (f["status"] == "numeric" and f["max_abs_diff"] <= REPRO_TOL)
                for f in files.values())
    return {"match": match, "tolerance": REPRO_TOL, "files": files}


def _cmd_repro(args) -> None:
    result = repro_compare(args.new_report_dir)
    print(json.dumps(result, indent=2))
    if not result["match"]:
        raise SystemExit(1)


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
    srr_report = sub.add_parser("srr-ndr-report", help="SRR/NDR per unit and per cell; fills verdicts.md's SRR section")
    srr_report.add_argument("--phase2", help="RQ3 results folder with saved policies; omit to mark Phase 2 not computed")
    srr_report.add_argument("--phase1", help="grid sweep results folder with saved policies")
    srr_report.add_argument("--missing-reason", default="Its results folder was not given.",
                            help="text shown for an omitted phase")
    srr_report.add_argument("--out", default="results/rq4_report", help="folder holding verdicts.md from `score`")
    srr_report.set_defaults(func=_cmd_srr_ndr_report)
    repro = sub.add_parser("repro", help="compare a rerun RQ3 report with results/published/experiment_main")
    repro.add_argument("new_report_dir")
    repro.set_defaults(func=_cmd_repro)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
