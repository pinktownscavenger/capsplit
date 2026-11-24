"""Pre-registered verdicts L1-L3 for the LM experiment (C6).

`build_verdicts(out, prereg_path)` reads the stage outputs under `out` (as `report.load_run`) and
the thresholds from the pre-registration JSON (`configs/lm_prereg.json`; unknown keys raise), and
writes `out/report/verdicts.json` and `out/report/verdicts.md`.

Every verdict is one of "supported", "not supported", "inconclusive", stated with the rule
applied and the numbers it used (all read from files). All three outcomes are printed the same
way.

Intervals. Per-seed contributions on measurement groups are read from `measure/results.json`
(the measure stage's bootstrap). Intervals the measure stage does not compute (subsets of train
problems for L1, contributions pooled over seeds for L2/L3) use `report.contribution_bootstrap`,
the same procedure (problems resampled, set-A counts redrawn and reclassified) with the measure
stage's alpha and n_boot, and stream SeedSequence([boot_seed, crc32(label)]).

Arms. Unless the pre-registration names them: L1 uses every `lm_rl` arm, plus every
`lm_sft_then_rl` arm for which eval samples at its SFT/RL boundary step exist (its RL-phase gain
is then p̂1(final) - p̂1(boundary)); L2 uses every arm whose trainer is not `lm_none`; L3 uses
every `lm_sft` arm.
"""
from __future__ import annotations

import json
import math
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from .report import (
    ReportError,
    Run,
    contribution_bootstrap,
    load_run,
    md_num,
    md_table,
    write_json_det,
    write_text,
)

SUPPORTED, NOT_SUPPORTED, INCONCLUSIVE = "supported", "not supported", "inconclusive"
BASES = ("pooled", "each_seed")
PREREG_KEYS = {
    "schema": str, "status": str, "notes": str,
    "tau_l1": float, "l1_min_problems": int, "l1_arms": list,
    "tau_l2": float, "l2_basis": str, "l2_arms": list,
    "tau_l3_effect": float, "l3_basis": str, "l3_arms": list,
}
REQUIRED = ("tau_l1", "tau_l2", "tau_l3_effect")
DEFAULTS: dict[str, Any] = {"l1_min_problems": 1, "l1_arms": None, "l2_basis": "pooled", "l2_arms": None,
                            "l3_basis": "pooled", "l3_arms": None}
CLAIMS = {
    "L1": "RL gains on training problems classified unreachable occur only where training itself sampled a "
          "correct answer.",
    "L2": "No arm produces capability gain on held-out unreachable problems beyond a stated near-zero threshold.",
    "L3": "SFT on unreachable training problems lowers success on reachable problems.",
}


class PreregError(ValueError):
    """The pre-registration file is invalid; the message names the key."""


def load_prereg(path: str | Path) -> dict:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise PreregError("the pre-registration must be a JSON object")
    unknown = sorted(set(raw) - set(PREREG_KEYS))
    if unknown:
        raise PreregError(f"unknown pre-registration keys {unknown}; allowed keys are {sorted(PREREG_KEYS)}")
    missing = [k for k in REQUIRED if k not in raw]
    if missing:
        raise PreregError(f"missing pre-registration keys {missing}")
    out = {**DEFAULTS, **raw}
    for key, kind in PREREG_KEYS.items():
        value = out.get(key)
        if value is None:
            continue
        if kind is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise PreregError(f"{key} must be a finite number, got {value!r}")
            out[key] = float(value)
        elif kind is int:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PreregError(f"{key} must be a non-negative integer, got {value!r}")
        elif kind is list:
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise PreregError(f"{key} must be null or a list of arm names, got {value!r}")
        elif not isinstance(value, kind):
            raise PreregError(f"{key} must be a {kind.__name__}, got {value!r}")
    for key in ("l2_basis", "l3_basis"):
        if out[key] not in BASES:
            raise PreregError(f"{key} must be one of {BASES}, got {out[key]!r}")
    if out["tau_l3_effect"] < 0:
        raise PreregError("tau_l3_effect is a minimum effect size and must be >= 0")
    return out


# --------------------------------------------------------------------------- helpers


def _seed(run: Run, label: str) -> list[int]:
    return [int(run.measure_param("boot_seed")), zlib.crc32(label.encode())]


def _boot_args(run: Run) -> dict:
    return {"n": int(run.measure_param("n")), "alpha": float(run.measure_param("alpha")),
            "n_boot": int(run.measure_param("n_boot"))}


def _arms(run: Run, named: list | None, trainers: tuple[str, ...], exclude: bool = False) -> list[str]:
    known = [a["name"] for a in run.arms]
    if named is not None:
        bad = [a for a in named if a not in known]
        if bad:
            raise PreregError(f"pre-registration names arms {bad} that are not in the config (arms: {known})")
        return list(named)
    return [a["name"] for a in run.arms if (a["trainer"] in trainers) != exclude]


def _ci(c: dict | None) -> dict:
    c = c or {}
    return {"estimate": c.get("estimate"), "lo": c.get("lo"), "hi": c.get("hi")}


def _ci_md(c: dict | None) -> str:
    c = c or {}
    return f"{md_num(c.get('estimate'))} [{md_num(c.get('lo'))}, {md_num(c.get('hi'))}]"


def _combine(parts: list[str]) -> str:
    """not supported if any part is; supported if all are; otherwise inconclusive."""
    if not parts:
        return INCONCLUSIVE
    if NOT_SUPPORTED in parts:
        return NOT_SUPPORTED
    if all(p == SUPPORTED for p in parts):
        return SUPPORTED
    return INCONCLUSIVE


def _nan(x: Any) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


# --------------------------------------------------------------------------- L1


def _l1_unit(run: Run, arm: str, seed: int, prereg: dict) -> dict:
    """One (arm, seed): gains of unreachable train problems split by whether training sampled a correct answer."""
    log = run.training.get((arm, seed))
    trainer = run.trainer(arm)
    unit: dict[str, Any] = {"arm": arm, "seed": seed, "trainer": trainer}
    if not log:
        return {**unit, "verdict": INCONCLUSIVE, "reason": "training.json is missing"}
    p0_label = "base set B"
    p1 = run.eval_rates(arm, seed)
    if trainer == "lm_sft_then_rl":
        n_sft = int(log.get("n_sft", 0))
        boundary = run.sample_file(f"{arm}/seed{seed}/step{n_sft}", "eval")
        if not boundary.exists():
            return {**unit, "verdict": INCONCLUSIVE, "evaluated": False,
                    "reason": f"no eval samples at the SFT/RL boundary step {n_sft} ({boundary.name} under "
                              f"samples/{arm}/seed{seed}/step{n_sft}); run measure with --all-checkpoints "
                              "and that step among the checkpoints"}
        p0 = run.eval_rates(arm, seed, n_sft)
        p0_label = f"{arm} seed {seed} at step {n_sft} (end of SFT)"
    else:
        p0 = run.base_rates()
    ids = np.asarray(run.ids("train"), dtype=np.int64)
    k_a, m_a = run.set_a()
    prompts = log.get("prompts", {})
    first = [prompts.get(str(int(i)), {}).get("first_success_update") for i in ids]
    ever = np.asarray([f is not None for f in first], dtype=bool)
    gain = p1[ids] - p0[ids]
    args = _boot_args(run)
    never_ci = contribution_bootstrap(k_a[ids], m_a[ids], gain, ~ever, "unreachable", **args,
                                      seed=_seed(run, f"L1/{arm}/{seed}/never"))
    ever_ci = contribution_bootstrap(k_a[ids], m_a[ids], gain, ever, "unreachable", **args,
                                     seed=_seed(run, f"L1/{arm}/{seed}/ever"))
    cls = run.class_array()
    per_problem = [{"problem": int(i), "class": str(cls[i]), "first_success_update": f,
                    "episodes_sampled": prompts.get(str(int(i)), {}).get("episodes_sampled"),
                    "p0": float(p0[i]), "p1": float(p1[i]), "gain": float(p1[i] - p0[i])}
                   for i, f in zip(ids, first) if cls[i] == "unreachable"]
    tau, k_min = prereg["tau_l1"], prereg["l1_min_problems"]
    n_never = never_ci["n_in_subset"]
    if never_ci["hi"] is not None and not _nan(never_ci["hi"]) and never_ci["hi"] <= tau and n_never >= k_min:
        verdict = SUPPORTED
    elif not _nan(never_ci["lo"]) and never_ci["lo"] > tau:
        verdict = NOT_SUPPORTED
    else:
        verdict = INCONCLUSIVE
    reason = ""
    if verdict == INCONCLUSIVE and n_never < k_min:
        reason = f"only {n_never} unreachable train problem(s) were never sampled correctly (< {k_min} required)"
    return {**unit, "evaluated": True, "p0_source": p0_label, "verdict": verdict, "reason": reason,
            "n_train_problems": len(ids), "n_unreachable_never_correct": n_never,
            "n_unreachable_ever_correct": ever_ci["n_in_subset"],
            "never_correct": never_ci, "ever_correct": ever_ci, "per_problem": per_problem}


def verdict_l1(run: Run, prereg: dict) -> dict:
    named = prereg["l1_arms"]
    arms = _arms(run, named, ("lm_rl", "lm_sft_then_rl"))
    units = [_l1_unit(run, a, s, prereg) for a in arms for s in run.seeds]
    # an sft_then_rl arm without boundary samples is listed but does not enter the rule unless named
    counted = [u for u in units if u.get("evaluated", True) or named is not None]
    tau = prereg["tau_l1"]
    rule = (f"Per rl arm and seed, over train problems classified unreachable, the contribution to the train group's "
            f"mean gain of those training NEVER sampled correctly (training.json first_success_update null), with a "
            f"bootstrap interval. A unit is supported if its upper bound <= tau_l1 = {tau:g} and that subset has >= "
            f"{prereg['l1_min_problems']} problem(s); not supported if its lower bound > tau_l1; else inconclusive. "
            "Overall: not supported if any unit is not supported, supported if all are supported, else inconclusive "
            "(no correction for multiple seeds).")
    if not counted:
        return {"claim": CLAIMS["L1"], "rule": rule, "verdict": INCONCLUSIVE, "units": units,
                "reason": "no rl arm with a usable training log and eval samples"}
    return {"claim": CLAIMS["L1"], "rule": rule, "verdict": _combine([u["verdict"] for u in counted]),
            "units": units, "arms": arms}


# --------------------------------------------------------------------------- L2 / L3 shared


def _pooled(run: Run, arm: str, group: str, target: str, label: str) -> dict:
    ids = np.asarray(run.ids(group), dtype=np.int64)
    if not len(ids):
        return {"estimate": None, "lo": None, "hi": None, "n_problems": 0, "note": f"group {group} is empty"}
    p0 = run.base_rates()
    p1 = np.mean([run.eval_rates(arm, s) for s in run.seeds], axis=0)
    k_a, m_a = run.set_a()
    return contribution_bootstrap(k_a[ids], m_a[ids], p1[ids] - p0[ids], np.ones(len(ids), dtype=bool), target,
                                  **_boot_args(run), seed=_seed(run, label))


def _per_seed(run: Run, arm: str, group: str, target: str) -> list[dict]:
    out = []
    for s in run.seeds:
        rec = run.record(arm, s, group)
        out.append({"seed": s, "group": group,
                    **(_ci(rec["extra"]["ci"][target]) if rec else {"estimate": None, "lo": None, "hi": None}),
                    "source": "measure/results.json" if rec else "missing"})
    return out


def _intervals(run: Run, arm: str, group: str, target: str, basis: str, label: str) -> tuple[list[dict], dict, list]:
    seeds = _per_seed(run, arm, group, target)
    pooled = _pooled(run, arm, group, target, label)
    used = [pooled] if basis == "pooled" else seeds
    return seeds, pooled, used


# --------------------------------------------------------------------------- L2


def verdict_l2(run: Run, prereg: dict) -> dict:
    tau, basis = prereg["tau_l2"], prereg["l2_basis"]
    arms = _arms(run, prereg["l2_arms"], ("lm_none",), exclude=True)
    rule = (f"For every trained arm, the unreachable-class contribution on the heldout group "
            f"({'pooled over seeds' if basis == 'pooled' else 'each seed'}), with a bootstrap interval. Supported if "
            f"every interval's upper bound <= tau_l2 = {tau:g}; not supported if any lower bound > tau_l2; else "
            "inconclusive.")
    units, his, los = [], [], []
    for arm in arms:
        seeds, pooled, used = _intervals(run, arm, "heldout", "unreachable", basis, f"L2/{arm}")
        units.append({"arm": arm, "per_seed": seeds, "pooled": pooled})
        his += [u["hi"] for u in used]
        los += [u["lo"] for u in used]
    if not arms:
        verdict, reason = INCONCLUSIVE, "no trained arm"
    elif any(_nan(x) for x in his + los):
        verdict, reason = INCONCLUSIVE, "an interval is missing (no heldout measurement)"
    elif all(h <= tau for h in his):
        verdict, reason = SUPPORTED, ""
    elif any(lo > tau for lo in los):
        verdict, reason = NOT_SUPPORTED, ""
    else:
        verdict, reason = INCONCLUSIVE, ""
    n_unreach = int(np.sum(run.class_array()[np.asarray(run.ids("heldout"), dtype=np.int64)] == "unreachable")) \
        if run.ids("heldout") else 0
    out = {"claim": CLAIMS["L2"], "rule": rule, "verdict": verdict, "units": units, "arms": arms,
           "n_heldout_unreachable": n_unreach}
    if n_unreach == 0:
        out["reason"] = ("no heldout problem is classified unreachable, so every contribution is 0 by construction "
                         "and a 'supported' verdict is vacuous")
        if verdict == SUPPORTED:
            out["verdict"] = INCONCLUSIVE
    elif reason:
        out["reason"] = reason
    return out


# --------------------------------------------------------------------------- L3


def verdict_l3(run: Run, prereg: dict) -> dict:
    eff, basis = prereg["tau_l3_effect"], prereg["l3_basis"]
    arms = _arms(run, prereg["l3_arms"], ("lm_sft",))
    rule = (f"For the sft arm(s), the reachable-class contribution on all problems "
            f"({'pooled over seeds' if basis == 'pooled' else 'each seed'}), with a bootstrap interval. Supported if "
            f"the upper bound < -tau_l3_effect = {-eff:g}; not supported if the lower bound >= 0; else inconclusive. "
            "Several sft arms (or seeds): not supported if any is, supported if all are, else inconclusive. Train "
            "and heldout groups are reported but do not enter the rule.")
    units, parts = [], []
    for arm in arms:
        entry: dict[str, Any] = {"arm": arm}
        for group in ("all", "train", "heldout"):
            seeds, pooled, used = _intervals(run, arm, group, "reachable", basis, f"L3/{arm}/{group}")
            entry[group] = {"per_seed": seeds, "pooled": pooled}
            if group == "all":
                for u in used:
                    if _nan(u["hi"]) or _nan(u["lo"]):
                        parts.append(INCONCLUSIVE)
                    elif u["hi"] < -eff:
                        parts.append(SUPPORTED)
                    elif u["lo"] >= 0:
                        parts.append(NOT_SUPPORTED)
                    else:
                        parts.append(INCONCLUSIVE)
        units.append(entry)
    out = {"claim": CLAIMS["L3"], "rule": rule, "verdict": _combine(parts), "units": units, "arms": arms}
    if not arms:
        out["reason"] = "no lm_sft arm in the config"
    return out


# --------------------------------------------------------------------------- output


def compute(run: Run, prereg: dict) -> dict:
    return {
        "verdicts": {"L1": verdict_l1(run, prereg), "L2": verdict_l2(run, prereg), "L3": verdict_l3(run, prereg)},
        "prereg": prereg,
        "measure": {k: run.measure_param(k) for k in ("n", "alpha", "n_boot", "boot_seed")},
        "null_check_passed": all(r.get("passed", False) for r in run.null.get("records", [])),
        "inputs": ["base/classes.json", "measure/results.json", "measure/null_check.json",
                   "samples/base/B_*.jsonl", "samples/<arm>/seed<s>/step<k>/eval_*.jsonl",
                   "train/<arm>/seed<s>/training.json"],
    }


def _verdict_word(v: str) -> str:
    return f"**{v.upper()}**"


def render(result: dict, name: str) -> str:
    v = result["verdicts"]
    md = [f"# Pre-registered verdicts: {name}", "",
          ("Generated by `capsplit.harness.lm.verdicts`; thresholds from the pre-registration file, every number "
          "from the experiment's files. Supported, not supported and inconclusive outcomes are reported alike."), "",
          ("The thresholds were pre-registered for `configs/lm_main.json` only (`docs/preregistration_lm.md`). For "
           "any other config this file applies the same rules as a pipeline check, not as a pre-registered test."), ""]
    if not result["null_check_passed"]:
        md += [("**WARNING: the null check FAILED for at least one group (see report.md). Every verdict below "
               "rests on a measurement pipeline that reported a gain where there was none.**"), ""]
    md += md_table(["claim", "verdict", "statement"], [[k, _verdict_word(v[k]["verdict"]), v[k]["claim"]] for k in v])
    md.append("")
    m = result["measure"]
    md += [(f"Measurement: N = {m['n']}, alpha = {m['alpha']}, {m['n_boot']} bootstrap resamples, boot seed "
           f"{m['boot_seed']}."), ""]

    l1 = v["L1"]
    md += ["## L1", "", f"Claim: {l1['claim']}", "", f"Rule: {l1['rule']}", "", f"Verdict: {_verdict_word(l1['verdict'])}"]
    if l1.get("reason"):
        md.append(f" ({l1['reason']})")
    md.append("")
    md += md_table(["arm", "seed", "unit verdict", "n never correct", "never-correct contribution [lo, hi]",
                    "n ever correct", "ever-correct contribution [lo, hi]", "p0 from", "note"],
                   [[u["arm"], u["seed"], u["verdict"], u.get("n_unreachable_never_correct", "n/a"),
                     _ci_md(u.get("never_correct")), u.get("n_unreachable_ever_correct", "n/a"),
                     _ci_md(u.get("ever_correct")), u.get("p0_source", "n/a"), u.get("reason", "")]
                    for u in l1["units"]])
    rows = [[u["arm"], u["seed"], p["problem"], "never" if p["first_success_update"] is None
             else p["first_success_update"], p["episodes_sampled"], md_num(p["p0"], sign=False),
             md_num(p["p1"], sign=False), md_num(p["gain"])]
            for u in l1["units"] for p in u.get("per_problem", [])]
    md += ["", "Per problem (train, classified unreachable):", ""]
    md += md_table(["arm", "seed", "problem", "first success update", "episodes", "p̂0", "p̂1", "gain"], rows) \
        if rows else ["(none)"]
    md.append("")

    l2 = v["L2"]
    md += ["## L2", "", f"Claim: {l2['claim']}", "", f"Rule: {l2['rule']}", "", f"Verdict: {_verdict_word(l2['verdict'])}"]
    if l2.get("reason"):
        md.append(f" ({l2['reason']})")
    md += ["", f"Heldout problems classified unreachable: {l2['n_heldout_unreachable']}.", ""]
    md += md_table(["arm", "seed", "unreachable contribution, heldout [lo, hi]"],
                   [[u["arm"], s["seed"], _ci_md(s)] for u in l2["units"] for s in u["per_seed"]]
                   + [[u["arm"], "pooled", _ci_md(u["pooled"])] for u in l2["units"]])
    md.append("")

    l3 = v["L3"]
    md += ["## L3", "", f"Claim: {l3['claim']}", "", f"Rule: {l3['rule']}", "", f"Verdict: {_verdict_word(l3['verdict'])}"]
    if l3.get("reason"):
        md.append(f" ({l3['reason']})")
    md.append("")
    md += md_table(["arm", "group", "seed", "reachable contribution [lo, hi]"],
                   [[u["arm"], g, s["seed"], _ci_md(s)] for u in l3["units"] for g in ("all", "train", "heldout")
                    for s in u[g]["per_seed"]]
                   + [[u["arm"], g, "pooled", _ci_md(u[g]["pooled"])] for u in l3["units"]
                      for g in ("all", "train", "heldout")])
    md += ["", "## Pre-registration used", "", "```json", json.dumps(result["prereg"], indent=2, sort_keys=True),
           "```", ""]
    return "\n".join(md)


def build_verdicts(out_dir: str | Path, prereg_path: str | Path) -> Path:
    """Write `<out_dir>/report/verdicts.{json,md}`; returns the path of verdicts.json."""
    prereg = load_prereg(prereg_path)
    run = load_run(out_dir)
    try:
        result = compute(run, prereg)
    except KeyError as error:
        raise ReportError(f"an input file lacks the field {error.args[0]!r}; re-run the stage that writes it") from error
    rep = run.out / "report"
    path = write_json_det(rep / "verdicts.json", result)
    write_text(rep / "verdicts.md", render(result, run.name))
    return path
