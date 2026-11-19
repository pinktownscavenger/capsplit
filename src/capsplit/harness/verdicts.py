"""Pre-registered hypothesis tests (docs/preregistration.md, plan "prereg_v1").

This module applies the frozen analysis plan to a results folder. It reads only the seed
metrics files, through `report.load_experiment`, and uses the same bootstrap as the report
(`report.bootstrap_ci`: 95% percentile interval of the seed mean, 10,000 resamples, RNG 0).

Decision rule for a claim "X > tau": supported if the interval's lower bound > tau,
not supported if its upper bound <= tau, inconclusive otherwise. "X <= tau" is the mirror
image. tau0 = 0.02.

Run: python -m capsplit.harness verdicts RESULTS_DIR
Writes RESULTS_DIR/report/verdicts.json and verdicts.md.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .report import Experiment, _fkey, bootstrap_ci, load_experiment, metric

TAU0 = 0.02
FACTOR = "task.shared_prefix"
LEVELS = (0, 2, 4)
SUPPORTED, NOT_SUPPORTED, INCONCLUSIVE = "supported", "not supported", "inconclusive"


def _vals(exp: Experiment, arm: str, k: int, group: str, name: str, n: int, c: float) -> np.ndarray:
    return np.array(exp.values(arm, _fkey({FACTOR: k}), exp.final, group, n, c, name), dtype=float)


def _stat(values: np.ndarray) -> dict:
    lo, hi = bootstrap_ci([float(v) for v in values])
    return {"mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values)), "ci": [lo, hi]}


def decide_gt(values: np.ndarray, tau: float) -> tuple[str, dict]:
    s = _stat(values)
    lo, hi = s["ci"]
    return (SUPPORTED if lo > tau else NOT_SUPPORTED if hi <= tau else INCONCLUSIVE), s


def decide_le(values: np.ndarray, tau: float) -> tuple[str, dict]:
    s = _stat(values)
    lo, hi = s["ci"]
    return (SUPPORTED if hi <= tau else NOT_SUPPORTED if lo > tau else INCONCLUSIVE), s


def _combine(parts: list[str]) -> str:
    if all(p == SUPPORTED for p in parts):
        return SUPPORTED
    if any(p == NOT_SUPPORTED for p in parts):
        return NOT_SUPPORTED
    return INCONCLUSIVE


def h1(exp: Experiment, n: int, c: float, arm: str = "rloo") -> dict:
    rel = _vals(exp, arm, 0, "all", "delta_rel", n, c)
    cap = _vals(exp, arm, 0, "all", "delta_cap", n, c)
    holds = rel >= 0.9 * (rel + cap)
    a = SUPPORTED if int(holds.sum()) >= 9 else NOT_SUPPORTED
    b, sb = decide_le(_vals(exp, arm, 0, "hard_heldout", "delta_cap", n, c), TAU0)
    cc, sc = decide_le(_vals(exp, arm, 0, "hard_train", "delta_cap", n, c), TAU0)
    return {
        "verdict": _combine([a, b, cc]),
        "a": {"verdict": a, "seeds_holding": int(holds.sum()), "of": int(holds.size),
              "delta_rel_all": _stat(rel), "delta_cap_all": _stat(cap),
              "note": "pre-registered: supported iff it holds in >= 9/10 seeds; otherwise not supported"},
        "b": {"verdict": b, "delta_cap_hard_heldout": sb},
        "c": {"verdict": cc, "delta_cap_hard_train": sc},
    }


def _monotone(exp: Experiment, arm: str, group: str, n: int, c: float) -> tuple[bool, list[float]]:
    means = [float(np.mean(_vals(exp, arm, k, group, "delta_cap", n, c))) for k in LEVELS]
    return all(a <= b for a, b in zip(means, means[1:])), means


def h2(exp: Experiment, n: int, c: float) -> dict:
    out: dict[str, Any] = {}
    for role, group in (("primary", "hard_heldout"), ("secondary", "hard_train")):
        diff = _vals(exp, "rloo", 4, group, "delta_cap", n, c) - _vals(exp, "rloo", 0, group, "delta_cap", n, c)
        d, s = decide_gt(diff, TAU0)
        mono, means = _monotone(exp, "rloo", group, n, c)
        verdict = SUPPORTED if d == SUPPORTED and mono else NOT_SUPPORTED if d == NOT_SUPPORTED else INCONCLUSIVE
        out[role] = {"group": group, "verdict": verdict, "diff_k4_minus_k0": s, "diff_verdict": d,
                     "monotone": mono, "seed_mean_by_k": dict(zip(map(str, LEVELS), means))}
    out["verdict"] = out["primary"]["verdict"]
    return out


def h3(exp: Experiment, n: int, c: float) -> dict:
    arms = ("sft", "hints_h1", "hints_h3")
    part_a: dict[str, Any] = {}
    for arm in arms:
        per_k = {str(k): dict(zip(("verdict", "stat"), decide_gt(_vals(exp, arm, k, "hard_train", "delta_cap", n, c), TAU0)))
                 for k in LEVELS}
        part_a[arm] = {"holds": all(v["verdict"] == SUPPORTED for v in per_k.values()), "by_k": per_k}
    part_b: dict[str, Any] = {}
    for arm in arms:
        diff = _vals(exp, arm, 4, "hard_heldout", "delta_cap", n, c) - _vals(exp, arm, 0, "hard_heldout", "delta_cap", n, c)
        s = _stat(diff)
        lo, hi = s["ci"]
        held = {str(k): _stat(_vals(exp, arm, k, "hard_heldout", "delta_cap", n, c)) for k in LEVELS}
        part_b[arm] = {"excludes_zero": bool(lo > 0 or hi < 0), "direction": "+" if s["mean"] > 0 else "-",
                       "diff_k4_minus_k0": s, "delta_cap_hard_heldout_by_k": held}
    a_ok = part_a["sft"]["holds"] and (part_a["hints_h1"]["holds"] or part_a["hints_h3"]["holds"])
    b_ok = any(v["excludes_zero"] for v in part_b.values())
    verdict = SUPPORTED if a_ok and b_ok else NOT_SUPPORTED
    return {"verdict": verdict, "a": part_a, "a_holds": a_ok, "b": part_b, "b_supported": b_ok,
            "note": "pre-registered: supported iff (a) holds for sft and >= 1 hint arm and (b) holds; otherwise not supported"}


def h4(exp: Experiment, n: int, c: float) -> dict:
    per_k: dict[str, Any] = {}
    parts: list[str] = []
    for k in LEVELS:
        both = _vals(exp, "sft_then_rl", k, "hard_heldout", "j1", n, c)
        row: dict[str, Any] = {}
        for other in ("sft", "rloo"):
            s = _stat(both - _vals(exp, other, k, "hard_heldout", "j1", n, c))
            row[f"minus_{other}"] = s
        lows = [row[f"minus_{o}"]["ci"][0] for o in ("sft", "rloo")]
        highs = [row[f"minus_{o}"]["ci"][1] for o in ("sft", "rloo")]
        row["verdict"] = SUPPORTED if min(lows) > 0 else NOT_SUPPORTED if min(highs) <= 0 else INCONCLUSIVE
        per_k[str(k)] = row
        parts.append(row["verdict"])
    return {"verdict": _combine(parts), "by_k": per_k, "metric": "J1 on hard_heldout, paired by seed"}


TESTS: dict[str, Callable[[Experiment, int, float], dict]] = {"H1": h1, "H2": h2, "H3": h3, "H4": h4}


REQUIRED_ARMS = ("rloo", "grpo", "sft", "hints_h1", "hints_h3", "sft_then_rl")
REQUIRED_GROUPS = ("all", "hard_train", "hard_heldout")


def check_applicable(exp: Experiment) -> None:
    """The plan is specific to configs/main.json; refuse to produce verdicts for anything else."""
    problems = []
    arms = {a for a, _ in exp.arms}
    if missing := [a for a in REQUIRED_ARMS if a not in arms]:
        problems.append(f"arms {missing}")
    if missing := [g for g in REQUIRED_GROUPS if g not in exp.groups]:
        problems.append(f"groups {missing}")
    if missing_levels := [k for k in LEVELS if _fkey({FACTOR: k}) not in exp.fkeys]:
        problems.append(f"{FACTOR} levels {missing_levels}")
    if problems:
        raise ValueError(
            "the pre-registered plan prereg_v1 does not apply to this experiment; missing " + "; ".join(problems)
        )


def compute(exp: Experiment) -> dict:
    check_applicable(exp)
    n0, c0 = exp.primary
    primary = {name: test(exp, n0, c0) for name, test in TESTS.items()}
    primary["H1"]["grpo_robustness"] = h1(exp, n0, c0, arm="grpo")["verdict"]
    sensitivity = {}
    flips = []
    for n in exp.ns:
        for c in exp.cs:
            verdicts = {name: test(exp, n, c)["verdict"] for name, test in TESTS.items()}
            sensitivity[f"N={n},c={c:g}"] = verdicts
            for name, v in verdicts.items():
                if v != primary[name]["verdict"]:
                    flips.append(f"{name}: {primary[name]['verdict']} at primary -> {v} at N={n}, c={c:g}")
    return {"plan": "prereg_v1", "primary": {"n": n0, "c": c0}, "final_step": exp.final, "tau0": TAU0,
            "tests": primary, "sensitivity": sensitivity, "flips": flips}


def _num(x: float) -> str:
    return "nan" if isinstance(x, float) and math.isnan(x) else f"{x:+.4f}"


def _s(stat: dict) -> str:
    return f"{_num(stat['mean'])} [{_num(stat['ci'][0])}, {_num(stat['ci'][1])}] ({_num(stat['min'])}..{_num(stat['max'])})"


def render(result: dict) -> str:
    t = result["tests"]
    n, c = result["primary"]["n"], result["primary"]["c"]
    lines = [
        "# Pre-registered verdicts (generated)",
        "",
        f"Generated by `python -m capsplit.harness verdicts` from the seed metrics files, applying "
        f"`docs/preregistration.md` exactly. Primary N = {n}, c = {c:g}, final update {result['final_step']}, "
        f"tau0 = {result['tau0']}. Each number is: seed mean [bootstrap 95% interval over 10 seeds] (min..max). "
        "Results are exact; intervals describe seed-to-seed spread only.",
        "",
        "| Hypothesis | Verdict |",
        "|---|---|",
        *[f"| {k} | **{v['verdict']}** |" for k, v in t.items()],
        "",
        "## H1: RL only gives mostly reliability; capability near zero at k = 0 (arm rloo)",
        f"- (a) delta_rel(all) >= 0.9 x total in {t['H1']['a']['seeds_holding']}/{t['H1']['a']['of']} seeds "
        f"-> {t['H1']['a']['verdict']}. delta_rel(all) {_s(t['H1']['a']['delta_rel_all'])}; delta_cap(all) {_s(t['H1']['a']['delta_cap_all'])}",
        f"- (b) delta_cap(hard_heldout) <= {TAU0}: {_s(t['H1']['b']['delta_cap_hard_heldout'])} -> {t['H1']['b']['verdict']}",
        f"- (c) delta_cap(hard_train) <= {TAU0}: {_s(t['H1']['c']['delta_cap_hard_train'])} -> {t['H1']['c']['verdict']}",
        f"- Robustness, same test on grpo: {t['H1']['grpo_robustness']}",
        "",
        "## H2: capability under RL only rises with shared structure (arm rloo)",
    ]
    for role in ("primary", "secondary"):
        r = t["H2"][role]
        lines.append(
            f"- {role} ({r['group']}): delta_cap(k=4) - delta_cap(k=0) = {_s(r['diff_k4_minus_k0'])} -> {r['diff_verdict']}; "
            f"seed means by k {', '.join(f'k={k}: {_num(v)}' for k, v in r['seed_mean_by_k'].items())}; "
            f"monotone: {r['monotone']} -> **{r['verdict']}**"
        )
    lines += ["", "## H3: SFT and hints give capability on hard train prompts; transfer depends on shared structure",
              f"- (a) holds for sft and >= 1 hint arm: {t['H3']['a_holds']}"]
    for arm, r in t["H3"]["a"].items():
        lines.append(f"  - {arm}: " + "; ".join(f"k={k}: {_s(v['stat'])} -> {v['verdict']}" for k, v in r["by_k"].items()))
    lines.append(f"- (b) transfer to held-out hard depends on k (difference k=4 - k=0 excludes 0 for >= 1 arm): {t['H3']['b_supported']}")
    for arm, r in t["H3"]["b"].items():
        lines.append(
            f"  - {arm}: difference {_s(r['diff_k4_minus_k0'])}, excludes 0: {r['excludes_zero']}; delta_cap(hard_heldout) by k: "
            + "; ".join(f"k={k}: {_s(v)}" for k, v in r["delta_cap_hard_heldout_by_k"].items())
        )
    lines += ["", "## H4: SFT then RL beats either alone on held-out hard prompts (J1, paired by seed)"]
    for k, r in t["H4"]["by_k"].items():
        lines.append(f"- k={k}: minus sft {_s(r['minus_sft'])}; minus rloo {_s(r['minus_rloo'])} -> {r['verdict']}")
    lines += ["", "## Sensitivity over (N, c)", "", "| (N, c) | " + " | ".join(TESTS) + " |", "|---|" + "---|" * len(TESTS)]
    for key, v in result["sensitivity"].items():
        lines.append(f"| {key} | " + " | ".join(v[name] for name in TESTS) + " |")
    lines += ["", "Flips relative to the primary (N, c):" if result["flips"] else "No verdict flips across (N, c)."]
    lines += [f"- WARNING: {f}" for f in result["flips"]]
    return "\n".join(lines) + "\n"


def build_verdicts(results_dir: str | Path) -> Path:
    results_dir = Path(results_dir)
    exp = load_experiment(results_dir)
    result = compute(exp)
    out = results_dir / "report"
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdicts.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (out / "verdicts.md").write_text(render(result))
    return out / "verdicts.md"
