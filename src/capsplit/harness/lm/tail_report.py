"""Analysis and report of the tail layer (`lm/tail.py`; pre-registration `results/published/lm_main/tail/preregistration_tail.md`).

Reads `<out>/<name>_tail/` (D, brackets, checks, importance samples, brute force), the run itself and,
if present, the KL layer; writes `<out>/<name>_tail/report/{tail.json, report.md, tables/, figures/}`.
Every number in the report comes from files; nothing is typed.

Order of the analysis follows the pre-registration: Step 1 (D and its check, Gate 1), Step 2 (the
bracket and its check), Step 3 (controls, then the stop gate, then estimates under the trust rule),
Step 4 (discovery, budget curve past N = 108, d/KL per arm) and the predictions.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .config import LMConfig
from .report import _plt, _save, load_run, md_table, write_csv, write_json_det, write_text
from .storage import read_jsonl
from .tail import COMPONENTS, TailConfig, base_reference, estimate, mixture_logq, tail_paths, trust

Z95 = 1.959963984540054
LABELS = ("reachable", "unreachable", "ambiguous")


def _rows(path: Path) -> list[dict]:
    return list(read_jsonl(path)) if path.exists() else []


def _sci(x: Any, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    v = float(x)
    if math.isnan(v):
        return "n/a"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return f"{v:.{digits}g}"


def label_of(lo: float, hi: float, n: float) -> str:
    """reachable iff the lower end >= 1/N, unreachable iff the upper end < 1/N, else ambiguous."""
    if lo >= 1.0 / n:
        return "reachable"
    if hi < 1.0 / n:
        return "unreachable"
    return "ambiguous"


def is_summary(rows: list[dict], won: float, d_value: float, weights: Any, cfg: TailConfig) -> dict[str, Any]:
    """Plain estimate of p0 (won + remainder), its trust, the known-answer control and the per-token gap."""
    lp = np.array([[r[f"lp_{c}"] for c in COMPONENTS] for r in rows], dtype=float)
    log_w = lp[:, 0] - mixture_logq(lp, weights)
    success = np.array([r["correct"] for r in rows], dtype=bool)
    rem = np.array([r["region"] == "rem" for r in rows], dtype=bool)
    in_d = np.array([r["in_d"] for r in rows], dtype=bool)
    ind = success & rem
    w = np.exp(log_w)
    est = estimate(ind, log_w, offset=won)
    tr = trust(w[ind], cfg)
    known = estimate(in_d, log_w)
    known_trust = trust(w[in_d], cfg)
    z_known = (known["estimate"] - d_value) / known["se"] if known["se"] and known["se"] > 0 else math.nan
    regions = {k: int(sum(r["region"] == k for r in rows)) for k in ("won", "dead", "rem")}
    bad_region = int(sum(r["region"] == "won" and not r["correct"] for r in rows)
                     + sum(r["region"] == "dead" and r["correct"] for r in rows))
    by_comp = {c: int(sum(r["component"] == c for r in rows)) for c in COMPONENTS}
    success_by_comp = {c: int(sum(r["component"] == c and r["correct"] for r in rows)) for c in COMPONENTS}
    return {"estimate": est["estimate"], "remainder": est["remainder"], "se": est["se"], "n": est["n"],
            "trust": tr, "known": {"estimate": known["estimate"], "se": known["se"], "z": z_known,
                                   "trust": known_trust},
            "regions": regions, "region_contradictions": bad_region, "by_component": by_comp,
            "success_by_component": success_by_comp, "n_success": int(success.sum()),
            "gap_max": float(max(r["gap_max"] for r in rows)),
            "gap_mean": float(np.mean([r["gap_mean"] for r in rows]))}


def _eval_rates(out: Path, config: LMConfig, arm: str, seed: int, step: int, n: int) -> np.ndarray | None:
    from . import stages as st

    path = st.sample_path(out, f"{arm}/seed{seed}/step{step}", "eval", float(config.sampling.reference_temperature))
    if not path.exists():
        return None
    k = np.zeros(n)
    m = np.zeros(n)
    for r in read_jsonl(path):
        k[int(r["problem"])] += bool(r["correct"])
        m[int(r["problem"])] += 1
    return k / np.maximum(m, 1)


def _set_rates(out: Path, config: LMConfig, set_name: str, n: int) -> np.ndarray:
    from . import stages as st

    path = st.sample_path(out, "base", set_name, float(config.sampling.reference_temperature))
    k = np.zeros(n)
    m = np.zeros(n)
    for r in read_jsonl(path):
        k[int(r["problem"])] += bool(r["correct"])
        m[int(r["problem"])] += 1
    return k / np.maximum(m, 1)


def budget_curve_ext(lo: np.ndarray, hi: np.ndarray, gains: dict[str, list[np.ndarray]], grid: list[int]) -> dict:
    """Label counts and gain contributions (mean over problems of gain x 1[label], averaged over seeds) per N."""
    counts, contrib = [], {arm: [] for arm in gains}
    for n in grid:
        labels = np.array([label_of(a, b, n) for a, b in zip(lo, hi)])
        counts.append({lab: int((labels == lab).sum()) for lab in LABELS})
        for arm, per_seed in gains.items():
            vals = {lab: float(np.mean([g[labels == lab].sum() / len(g) for g in per_seed])) for lab in LABELS}
            contrib[arm].append(vals)
    return {"n": grid, "counts": counts, "contributions": contrib}


def compute_tail(config: LMConfig, out: Path, tail_out: Path, cfg: TailConfig, kl_out: Path | None) -> dict[str, Any]:
    from .discovery import _rl_arms, compute_discovery, observations
    from .kl import bernoulli_kl

    paths = tail_paths(tail_out)
    run = load_run(out)
    n_all = run.n_problems
    d_rows = {int(r["problem"]): r for r in _rows(paths["d"])}
    b_rows = {int(r["problem"]): r for r in _rows(paths["bracket"])}
    problems = sorted(d_rows)
    checks = json.loads(paths["checks"].read_text())
    selection = json.loads(paths["selection"].read_text())
    k_ref, m_ref, used = base_reference(out, config, n_all)
    k_a, _ = run.set_a()
    cls = run.class_array()
    rows_meta = {int(r["id"]): r for r in run.problems}
    if not rows_meta:
        rows_meta = {int(r["id"]): r for r in read_jsonl(out / "data" / "problems.jsonl")}

    per: dict[int, dict] = {}
    for p in problems:
        d, b = d_rows[p], b_rows.get(p)
        won = float(b["won"]) if b else 0.0
        lo = max(float(d["D"]), won)
        hi = won + float(b["open"]) if b else 1.0
        per[p] = {"problem": p, "split": rows_meta[p]["split"], "difficulty": rows_meta[p]["difficulty"],
                  "answer": d["answer"], "class": str(cls[p]), "k_ref": int(k_ref[p]), "m_ref": int(m_ref[p]),
                  "p_hat": float(k_ref[p] / max(m_ref[p], 1)), "k_a": int(k_a[p]), "D": float(d["D"]),
                  "first_digit": float(d["first_digit"]), "won": won, "open": hi - won, "lo": lo, "hi": hi,
                  "ratio": hi / lo if lo > 0 else math.inf, "expansions": int(b["expansions"]) if b else 0,
                  "total_mass": float(b["total"]) if b else math.nan}

    # Step 1: Gate 1
    resolved = [p for p in problems if per[p]["p_hat"] >= cfg.gate1_min_p]
    ratios = {p: per[p]["D"] / per[p]["p_hat"] for p in resolved}
    gate1_fail = sorted(p for p, r in ratios.items() if r < cfg.gate1_ratio)
    gate1 = {"n_resolved": len(resolved), "fail": gate1_fail, "passed": not gate1_fail,
             "ratio_quantiles": {q: float(np.quantile(list(ratios.values()), q)) for q in (0.0, 0.1, 0.5, 0.9, 1.0)}
             if ratios else {}}

    # Step 3: importance sampling, controls, stop gate
    is_set = sorted(set(selection["target"]) | set(selection["bf"]))
    is_res: dict[int, dict] = {}
    for p in is_set:
        rows = _rows(paths["is"] / f"{p}.jsonl")
        if rows:
            is_res[p] = is_summary(rows, per[p]["won"], per[p]["D"], cfg.weights, cfg)
    known_trusted = [p for p in is_res if is_res[p]["known"]["trust"]["trusted"]]
    known_bad = sorted(p for p in known_trusted if abs(is_res[p]["known"]["z"]) > cfg.z_known)
    known_ok = len(known_trusted) >= cfg.known_min_trusted and not known_bad
    bf_rows = []
    for p in selection["bf"]:
        f = paths["bf"] / f"{p}.json"
        if not f.exists() or p not in is_res:
            continue
        b = json.loads(f.read_text())
        p_bf = b["correct"] / b["n"]
        se_bf = math.sqrt(max(p_bf * (1 - p_bf), 0.0) / b["n"])
        r = is_res[p]
        z = (r["estimate"] - p_bf) / math.hypot(r["se"], se_bf) if math.hypot(r["se"], se_bf) > 0 else math.nan
        bf_rows.append({"problem": p, "p_hat_560": per[p]["p_hat"], "bf_n": b["n"], "bf_correct": b["correct"],
                        "p_bf": p_bf, "se_bf": se_bf, "is_estimate": r["estimate"], "is_se": r["se"],
                        "trusted": r["trust"]["trusted"], "z": z, "lo": per[p]["lo"], "hi": per[p]["hi"]})
    bf_trusted = [r for r in bf_rows if r["trusted"]]
    bf_bad = sorted(r["problem"] for r in bf_trusted if abs(r["z"]) > cfg.z_bf)
    bf_ok = len(bf_trusted) >= cfg.bf_min_trusted and not bf_bad
    gate_ok = known_ok and bf_ok
    controls = {"known": {"n_trusted": len(known_trusted), "fail": known_bad, "passed": known_ok,
                          "z_threshold": cfg.z_known, "min_trusted": cfg.known_min_trusted},
                "brute_force": {"rows": bf_rows, "n_trusted": len(bf_trusted), "fail": bf_bad, "passed": bf_ok,
                                "z_threshold": cfg.z_bf, "min_trusted": cfg.bf_min_trusted},
                "stop_gate_passed": gate_ok}
    for p, r in is_res.items():
        usable = gate_ok and r["trust"]["trusted"]
        per[p].update({"is_estimate": r["estimate"], "is_se": r["se"], "is_trusted": r["trust"]["trusted"],
                       "is_usable": usable, "is_n_success_rem": r["trust"]["n_success"],
                       "is_ess": r["trust"]["ess"], "is_max_share": r["trust"]["max_share"],
                       "is_k_hat": r["trust"]["k_hat"], "is_outside_bracket":
                           bool(r["estimate"] < per[p]["won"] - 1e-15 or r["estimate"] > per[p]["hi"] + 1e-15),
                       "known_z": r["known"]["z"], "region_contradictions": r["region_contradictions"]})
        if usable:
            per[p]["lo_is"] = max(per[p]["lo"], r["estimate"] - Z95 * r["se"])
            per[p]["hi_is"] = min(per[p]["hi"], r["estimate"] + Z95 * r["se"])
    for p in problems:
        per[p].setdefault("lo_is", per[p]["lo"])
        per[p].setdefault("hi_is", per[p]["hi"])

    target = [p for p in selection["target"] if p in per]
    never = [p for p in target if per[p]["k_ref"] == 0]
    a_zero = [p for p in target if per[p]["k_a"] == 0]

    # Step 4: discovery with bracketed p0 for p-hat_A = 0
    disc_base = compute_discovery(run)
    discovery = []
    for arm in _rl_arms(run):
        group = int(arm.get("params", {}).get("group", 8))
        obs = [r for s in run.seeds for r in observations(run, arm, s)["rows"]
               if r["class"] == "unreachable" and r["problem"] in per]
        zero = [r for r in obs if r["p"] == 0]
        pos = [r for r in obs if r["p"] > 0]

        def pred(rows: list[dict], key: str, group: int = group) -> float:
            return float(sum(1.0 - (1.0 - per[r["problem"]][key]) ** (group * r["draws"]) for r in rows))

        base_unr = next(u for u in disc_base["arms"] if u["arm"] == arm["name"])["by_class"]["unreachable"]
        pos_pred = float(sum(1.0 - (1.0 - r["p"]) ** (group * r["draws"]) for r in pos))
        discovery.append({
            "arm": arm["name"], "group": group, "observations_unreachable": len(obs),
            "earlier_predicted": base_unr["predicted_discovered"], "earlier_observed": base_unr["observed_discovered"],
            "p_a_zero_observations": len(zero), "p_a_zero_discovered": int(sum(r["first_success_draw"] is not None
                                                                              for r in zero)),
            "p_a_zero_problems": len({r["problem"] for r in zero}),
            "p_a_zero_predicted_lo": pred(zero, "lo"), "p_a_zero_predicted_hi": pred(zero, "hi"),
            "p_a_zero_predicted_lo_is": pred(zero, "lo_is"), "p_a_zero_predicted_hi_is": pred(zero, "hi_is"),
            "p_a_zero_usable_estimates": int(sum(per[r["problem"]].get("is_usable", False) for r in zero)),
            "total_predicted_lo": pos_pred + pred(zero, "lo"), "total_predicted_hi": pos_pred + pred(zero, "hi"),
            "positive_predicted": pos_pred})

    # Step 4: budget curve past N = 108
    grid = [2**k for k in range(cfg.grid_max_exp + 1)]
    lo_all = np.array([per[p]["lo"] for p in problems])
    hi_all = np.array([per[p]["hi"] for p in problems])
    lo_is = np.array([per[p]["lo_is"] for p in problems])
    hi_is = np.array([per[p]["hi_is"] for p in problems])
    p0_b = _set_rates(out, config, "B", n_all)
    gains: dict[str, list[np.ndarray]] = {}
    for arm in run.arms:
        if arm["name"] == "none":
            continue
        seeds = [_eval_rates(out, config, arm["name"], s, int(run.config["budget"]), n_all) for s in run.seeds]
        gains[arm["name"]] = [(g - p0_b)[problems] for g in seeds if g is not None]
    curve = budget_curve_ext(lo_all, hi_all, gains, grid)
    curve_is = budget_curve_ext(lo_is, hi_is, gains, grid)

    # Step 4: d(p1 || p0_hi) / KL per arm, over the target problems
    kl_ratio = []
    per_csv = None if kl_out is None else kl_out / "report" / "tables" / "per_problem.csv"
    if per_csv is not None and per_csv.exists():
        with per_csv.open() as handle:
            kl_rows = [r for r in csv.DictReader(handle) if int(r["problem"]) in set(target)]
        for arm in sorted({r["arm"] for r in kl_rows}):
            sel = [r for r in kl_rows if r["arm"] == arm]
            p1 = np.array([float(r["p1_hat"]) for r in sel])
            p0 = np.array([per[int(r["problem"])]["hi"] for r in sel])
            kl = np.array([float(r["kl"]) for r in sel])
            undefined = int(np.sum((p0 == 0) & (p1 > 0)))
            ok = ~((p0 == 0) & (p1 > 0))
            d = bernoulli_kl(p1[ok], p0[ok])
            kl_ratio.append({"arm": arm, "cells": len(sel), "undefined": undefined,
                             "ratio": float(d.sum() / kl[ok].sum()) if kl[ok].sum() > 0 else math.nan})

    # predictions
    mul = [p for p in target if per[p]["difficulty"] in ("d3_mul", "d4_mul")]
    tight = [p for p in mul if per[p]["ratio"] <= 10]
    loose = [p for p in mul if per[p]["ratio"] > 10]
    tight_all = [p for p in target if per[p]["ratio"] <= 10]
    cert = [p for p in never if per[p]["hi"] < 1e-3]
    est_known = [p for p in never if per[p].get("is_usable")]
    est_low = [p for p in est_known if per[p]["is_estimate"] < 1e-3]
    d_low = [p for p in never if per[p]["D"] < 1e-3]
    rl = next((d for d in discovery if d["arm"] == "rl"), None)
    predictions = {
        "P1": {"mul_targets": len(mul), "tight": len(tight), "loose": len(loose),
               "first_digit_tight": float(np.mean([per[p]["first_digit"] for p in tight])) if tight else math.nan,
               "first_digit_loose": float(np.mean([per[p]["first_digit"] for p in loose])) if loose else math.nan},
        "P1_mine": {"targets": len(target), "tight": len(tight_all),
                    "supported": len(tight_all) < 0.5 * len(target)},
        "P2": {"never_correct": len(never), "certified_below_1e-3": len(cert), "usable_estimates": len(est_known),
               "estimate_below_1e-3": len(est_low), "D_below_1e-3": len(d_low),
               "supported_certified": len(cert) > 0.5 * len(never) if never else False,
               "supported_estimated": len(est_low) > 0.5 * len(never) if never else False},
        "P3": {"predicted_hi": rl["p_a_zero_predicted_hi"] if rl else math.nan,
               "predicted_lo": rl["p_a_zero_predicted_lo"] if rl else math.nan,
               "supported": bool(rl and rl["p_a_zero_predicted_hi"] <= 2)},
    }
    gaps = {"max": max((r["gap_max"] for r in is_res.values()), default=math.nan),
            "mean": float(np.mean([r["gap_mean"] for r in is_res.values()])) if is_res else math.nan}
    return {"reference_sets": used, "checks": checks, "gate1": gate1, "controls": controls,
            "per_problem": [per[p] for p in problems], "target": target, "never_correct": never, "a_zero": a_zero,
            "is_set": is_set, "is": {str(p): r for p, r in is_res.items()}, "discovery": discovery,
            "budget_curve": curve, "budget_curve_is": curve_is, "kl_ratio": kl_ratio, "predictions": predictions,
            "gaps": gaps, "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}}


# --------------------------------------------------------------------------- tables, figures, markdown

PER_HEADER = ["problem", "split", "difficulty", "class", "answer", "k_ref", "m_ref", "p_hat", "k_a", "D",
              "first_digit", "won", "open", "lo", "hi", "ratio", "is_estimate", "is_se", "is_trusted", "is_usable",
              "is_n_success_rem", "is_ess", "is_max_share", "is_k_hat", "is_outside_bracket", "known_z",
              "region_contradictions", "lo_is", "hi_is", "total_mass", "expansions"]


def _tables(res: dict, tab: Path) -> None:
    write_csv(tab / "per_problem.csv", PER_HEADER, [[r.get(h) for h in PER_HEADER] for r in res["per_problem"]])
    bf = res["controls"]["brute_force"]["rows"]
    if bf:
        write_csv(tab / "bf_control.csv", list(bf[0]), [list(r.values()) for r in bf])
    write_csv(tab / "discovery.csv", list(res["discovery"][0]) if res["discovery"] else ["arm"],
              [list(r.values()) for r in res["discovery"]])
    for name, curve in (("budget_curve.csv", res["budget_curve"]), ("budget_curve_is.csv", res["budget_curve_is"])):
        header = ["N", *[f"n_{lab}" for lab in LABELS],
                  *[f"{arm}_{lab}" for arm in curve["contributions"] for lab in LABELS]]
        rows = []
        for i, n in enumerate(curve["n"]):
            rows.append([n, *[curve["counts"][i][lab] for lab in LABELS],
                         *[curve["contributions"][arm][i][lab] for arm in curve["contributions"] for lab in LABELS]])
        write_csv(tab / name, header, rows)
    if res["kl_ratio"]:
        write_csv(tab / "kl_ratio.csv", list(res["kl_ratio"][0]), [list(r.values()) for r in res["kl_ratio"]])


def _figures(res: dict, fig: Path) -> None:
    plt = _plt()
    if plt is None:
        return
    per = {r["problem"]: r for r in res["per_problem"]}
    target = sorted(res["target"], key=lambda p: (per[p]["hi"], per[p]["lo"]))
    f, ax = plt.subplots(figsize=(9, 4.5))
    floor = 1e-14
    for i, p in enumerate(target):
        r = per[p]
        ax.plot([i, i], [max(r["lo"], floor), max(r["hi"], floor)], color="#4c72b0", lw=2)
        ax.plot(i, max(r["D"], floor), "_", color="black", ms=6)
        if r.get("is_usable"):
            ax.errorbar(i, r["is_estimate"], yerr=Z95 * r["is_se"], fmt="o", color="#dd8452", ms=3, lw=0.8)
    for n, ls in ((16, "--"), (108, ":")):
        ax.axhline(1 / n, color="grey", ls=ls, lw=0.8, label=f"1/{n}")
    ax.set_yscale("log")
    ax.set_xlabel("target problems (unreachable + ambiguous), sorted by upper end")
    ax.set_ylabel("p0")
    ax.set_title("Certified bracket [max(D, won), won + open] (blue), D (black), usable IS estimate (orange)")
    ax.legend(loc="lower right", fontsize=8)
    _save(plt, f, fig / "brackets.png")
    curve = res["budget_curve"]
    f, ax = plt.subplots(figsize=(7, 4))
    for lab, color in zip(LABELS, ("#55a868", "#c44e52", "#8172b2")):
        ax.plot(curve["n"], [c[lab] for c in curve["counts"]], marker="o", ms=3, label=lab, color=color)
    ax.axvline(108, color="grey", ls=":", lw=0.8, label="N = 108 (set A limit)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("budget N")
    ax.set_ylabel("problems")
    ax.set_title("Labels from the certified bracket (all 180 problems)")
    ax.legend(fontsize=8)
    _save(plt, f, fig / "budget_labels.png")
    bf = res["controls"]["brute_force"]["rows"]
    if bf:
        f, ax = plt.subplots(figsize=(5, 5))
        for r in bf:
            ax.errorbar(r["p_bf"], r["is_estimate"], xerr=Z95 * r["se_bf"], yerr=Z95 * r["is_se"], fmt="o",
                        color="#4c72b0" if r["trusted"] else "#c44e52", ms=4)
        top = max(max(r["p_bf"], r["is_estimate"]) for r in bf) * 1.2
        ax.plot([0, top], [0, top], color="black", lw=0.8)
        ax.set_xlabel("brute force (4,000 base samples)")
        ax.set_ylabel("importance-sampling estimate")
        ax.set_title("Brute-force control (red: untrusted)")
        _save(plt, f, fig / "bf_control.png")


def render(res: dict, name: str) -> str:
    per = {r["problem"]: r for r in res["per_problem"]}
    target = res["target"]
    c = res["checks"]
    g1 = res["gate1"]
    ctl = res["controls"]
    md = [f"# Tail success probability of the base model: {name}", "",
          ("Generated by `capsplit.harness.lm.tail_report` from `results/<name>_tail/`; no number is typed by hand. "
          "Pre-registered in `preregistration_tail.md` (tag `lm-tail-prereg`). Method credit: sequence-level "
          "importance sampling for rare LM outputs follows Angell et al., *Estimating Tail Risks in Language Model "
          "Output Distributions* (ICML 2026, arXiv:2604.22167); here it is applied to per-problem correctness."), "",
          f"Reference base samples (temperature 1 draws from pi0): {res['reference_sets']}.", ""]
    md += ["## Step 1: exact lower bound D", "",
           (f"- D above the one-sided CP upper bound at level 1 - 0.05/180 (a bug if any): {c['d_fail'] or 'none'}; "
           f"at the unadjusted one-sided 95% level: {c['d_fail_unadjusted'] or 'none'}."),
           f"- Gate 1 (D / p-hat >= {res['config']['gate1_ratio']} on the {g1['n_resolved']} problems with p-hat >= "
           f"{res['config']['gate1_min_p']}): **{'passed' if g1['passed'] else 'FAILED'}**; failing problems: "
           f"{g1['fail'] or 'none'}; quantiles of D / p-hat: "
           + ", ".join(f"{q}: {_sci(v)}" for q, v in g1["ratio_quantiles"].items()) + ".", ""]
    md += ["## Step 2: exact bracket", "",
           (f"- Bracket outside the two-sided CP interval at level 1 - 0.05/180 (a bug if any): "
           f"{c['bracket_fail'] or 'none'}; at the unadjusted 95% level: {c['bracket_fail_unadjusted'] or 'none'}."),
           (f"- Largest |1 - total mass| over problems: "
           f"{_sci(max(abs(1 - r['total_mass']) for r in res['per_problem']))}."), ""]
    ratios = np.array([per[p]["ratio"] for p in target])
    widths = np.array([per[p]["hi"] - per[p]["lo"] for p in target])
    md += [(f"Target problems (unreachable + ambiguous): {len(target)}. Upper / lower ratio: "
           f"<= 2 for {int((ratios <= 2).sum())}, <= 10 for {int((ratios <= 10).sum())}, infinite (lower end 0) for "
           f"{int(np.isinf(ratios).sum())}. Median width {_sci(np.median(widths))}; median upper end "
           f"{_sci(np.median([per[p]['hi'] for p in target]))}; median lower end "
           f"{_sci(np.median([per[p]['lo'] for p in target]))}."), "",
           "![brackets](figures/brackets.png)", ""]
    md += ["## Step 3: importance sampling of the open remainder", "",
           (f"Per-token gap between sampling-time and teacher-forced log q of the sampled component: max "
           f"{_sci(res['gaps']['max'])}, mean {_sci(res['gaps']['mean'])}."), "",
           (f"- Known-answer control (estimate D from the same samples): {ctl['known']['n_trusted']} problems pass the "
           f"trust rule; |z| > {ctl['known']['z_threshold']} for {ctl['known']['fail'] or 'none'} -> "
           f"**{'passed' if ctl['known']['passed'] else 'FAILED'}**."),
           (f"- Brute-force control ({len(ctl['brute_force']['rows'])} problems, 4,000 base samples each): "
           f"{ctl['brute_force']['n_trusted']} trusted; |z| > {ctl['brute_force']['z_threshold']} for "
           f"{ctl['brute_force']['fail'] or 'none'} -> **{'passed' if ctl['brute_force']['passed'] else 'FAILED'}**."),
           f"- **Stop gate: {'passed' if ctl['stop_gate_passed'] else 'FAILED: only the exact brackets are used'}.**",
           ""]
    if ctl["brute_force"]["rows"]:
        md += md_table(["problem", "p-hat (560)", "brute force", "IS estimate [95%]", "trusted", "z", "bracket"],
                       [[r["problem"], _sci(r["p_hat_560"]), f"{_sci(r['p_bf'])} ({r['bf_correct']}/{r['bf_n']})",
                         f"{_sci(r['is_estimate'])} ± {_sci(Z95 * r['is_se'])}", r["trusted"], _sci(r["z"], 2),
                         f"[{_sci(r['lo'])}, {_sci(r['hi'])}]"] for r in ctl["brute_force"]["rows"]])
        md += ["", "![bf control](figures/bf_control.png)", ""]
    usable = [p for p in target if per[p].get("is_usable")]
    outside = [p for p in target if per[p].get("is_outside_bracket")]
    contra = sum(per[p].get("region_contradictions", 0) for p in res["is_set"] if p in per)
    md += [(f"Target problems with a usable estimate (trust rule passed and stop gate passed): {len(usable)} of "
           f"{len(target)}; the rest are bracket-only. Estimates outside their bracket: {outside or 'none'}. "
           f"Samples whose tree region contradicts the checker: {contra}."), ""]
    md += md_table(["problem", "difficulty", "split", "class", "k/560", "D", "bracket", "IS estimate", "trust"],
                   [[p, per[p]["difficulty"], per[p]["split"], per[p]["class"], per[p]["k_ref"], _sci(per[p]["D"]),
                     f"[{_sci(per[p]['lo'])}, {_sci(per[p]['hi'])}]",
                     f"{_sci(per[p].get('is_estimate'))} ± {_sci(Z95 * per[p]['is_se'])}" if "is_se" in per[p]
                     else "n/a", "usable" if per[p].get("is_usable") else "bracket only"]
                    for p in sorted(target, key=lambda q: per[q]["hi"])])
    md += ["", "## Step 4: discovery with bracketed p0", "",
           ("Waiting-time model as in `discovery_lm.md`: predicted discoveries sum_i [1 - (1 - p_i)^(G D_i)]. For the "
           "unreachable observations with p-hat_A = 0, p_i runs over the certified bracket (and the bracket narrowed "
           "by usable IS intervals)."), ""]
    md += md_table(["arm", "earlier: predicted / observed (all unreachable)", "p-hat_A = 0: observations (problems)",
                    "observed", "predicted, bracket", "predicted, bracket narrowed by IS", "all unreachable, bracket"],
                   [[d["arm"], f"{d['earlier_predicted']:.2f} / {d['earlier_observed']}",
                     f"{d['p_a_zero_observations']} ({d['p_a_zero_problems']})", d["p_a_zero_discovered"],
                     f"[{d['p_a_zero_predicted_lo']:.3g}, {d['p_a_zero_predicted_hi']:.3g}]",
                     f"[{d['p_a_zero_predicted_lo_is']:.3g}, {d['p_a_zero_predicted_hi_is']:.3g}]",
                     f"[{d['total_predicted_lo']:.2f}, {d['total_predicted_hi']:.2f}] vs {d['earlier_observed']}"]
                    for d in res["discovery"]])
    curve = res["budget_curve"]
    md += ["", "## Step 4: budget curve past N = 108 (labels from the certified bracket)", "",
           ("reachable iff lower end >= 1/N; unreachable iff upper end < 1/N; ambiguous otherwise. Contributions: "
           "mean over problems of (p-hat1 eval - p-hat0 set B) x 1[label], averaged over seeds."), ""]
    arms = list(curve["contributions"])
    md += md_table(["N", "reachable", "unreachable", "ambiguous", *[f"{a}: unreachable contrib." for a in arms],
                    *[f"{a}: ambiguous contrib." for a in arms]],
                   [[n, curve["counts"][i]["reachable"], curve["counts"][i]["unreachable"],
                     curve["counts"][i]["ambiguous"],
                     *[f"{curve['contributions'][a][i]['unreachable']:+.4f}" for a in arms],
                     *[f"{curve['contributions'][a][i]['ambiguous']:+.4f}" for a in arms]]
                    for i, n in enumerate(curve["n"])])
    md += ["", "Secondary version with brackets narrowed by usable IS intervals: `tables/budget_curve_is.csv`.", "",
           "![budget labels](figures/budget_labels.png)", ""]
    if res["kl_ratio"]:
        md += ["## d(p1 || p0) / KL per arm (target problems; p0 = bracket upper end)", "",
               ("d is the Bernoulli KL with the raw eval p-hat1; it is undefined (infinite) when p0 = 0 and p-hat1 > 0, "
               "and those cells are left out and counted."), ""]
        md += md_table(["arm", "cells", "undefined cells", "sum d / sum KL"],
                       [[r["arm"], r["cells"], r["undefined"], _sci(r["ratio"])] for r in res["kl_ratio"]])
        md += [""]
    pr = res["predictions"]
    md += ["## Predictions", "",
           (f"- P1 (user): 3- and 4-digit multiplication targets: {pr['P1']['tight']} tight (upper/lower <= 10), "
           f"{pr['P1']['loose']} loose; mean first-token digit probability tight {_sci(pr['P1']['first_digit_tight'])}"
           f" vs loose {_sci(pr['P1']['first_digit_loose'])}."),
           (f"- P1' (mine): fewer than half of the {pr['P1_mine']['targets']} targets tight: {pr['P1_mine']['tight']} "
           f"tight -> {'supported' if pr['P1_mine']['supported'] else 'not supported'}."),
           (f"- P2 (user): of {pr['P2']['never_correct']} never-correct problems, {pr['P2']['certified_below_1e-3']} "
           f"have a certified upper end < 1e-3 ({'supported' if pr['P2']['supported_certified'] else 'not supported'} "
           f"as certified); {pr['P2']['usable_estimates']} have usable estimates, of which "
           f"{pr['P2']['estimate_below_1e-3']} are < 1e-3 "
           f"({'supported' if pr['P2']['supported_estimated'] else 'not supported'} as estimated); D < 1e-3 for "
           f"{pr['P2']['D_below_1e-3']}."),
           (f"- P3 (user): predicted discoveries among p-hat_A = 0 (rl, pooled seeds) at the bracket's upper end "
           f"{_sci(pr['P3']['predicted_hi'])} (lower end {_sci(pr['P3']['predicted_lo'])}) -> "
           f"{'supported (<= 2)' if pr['P3']['supported'] else 'not supported (> 2)'}."), "",
           "## Caveats", "",
           "- One 0.5B model, 12-token arithmetic completions; p0 is that of the float16 model on MPS.",
           "- The main run's null-check failure (z = -3.2) is unresolved.",
           ("- Importance sampling can silently miss success strings its proposal never visits; the bracket cannot, "
           "but it is only as tight as the expansion budget allows."),
           "- 'Certified' bounds are exact up to the numerics of the forward pass."]
    return "\n".join(md) + "\n"


def build_tail_report(config: LMConfig, out: str | Path, tail_out: str | Path, cfg: TailConfig,
                      kl_out: str | Path | None = None) -> Path:
    out, tail_out = Path(out), Path(tail_out)
    res = compute_tail(config, out, tail_out, cfg, None if kl_out is None else Path(kl_out))
    rep = tail_out / "report"
    _tables(res, rep / "tables")
    _figures(res, rep / "figures")
    write_text(rep / "report.md", render(res, config.name))
    return write_json_det(rep / "tail.json", res)
