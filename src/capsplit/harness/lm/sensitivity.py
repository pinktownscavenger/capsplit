"""Post-hoc sensitivity: do the L1-L3 verdicts depend on which base sample gives p̂0? (DECISIONS L21)

Why: the pre-registered verdicts take the base success rate p̂0 from base set B. The null check
compares B with a second, independently seeded base sample (set `null`) drawn by the identical
job. If the null check fails, either the pipeline is biased or B (or null) is an unlucky draw.
This stage does not change any registered verdict. It adds two things, both labelled post-hoc:

1. A diagnostic of the null check from the saved samples: per-set mean success (A, B, null),
   the null - B mean difference with an exact-style sign-flip permutation p-value (under the
   null hypothesis the two sets are exchangeable within each problem), and the ratio of the
   observed variance of per-problem differences to the binomial variance expected for
   independent samples (a ratio far from 1 would point to dependent sampling).
2. The L1-L3 rules re-applied, unchanged, with p̂0 from B (as registered), from `null`, and from
   B and `null` pooled. Only `Run.base_rates` changes; classes still come from set A.

Writes `<out>/report/sensitivity_p0.{json,md}`. Every number is computed here from files.
"""
from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any

import numpy as np

from .report import Run, load_run, md_ci, md_num, md_table, write_json_det, write_text
from .verdicts import compute, load_prereg

SOURCES: dict[str, tuple[str, ...]] = {"B": ("B",), "null": ("null",), "pooled B+null": ("B", "null")}
N_PERM = 20000
BUCKETS = ((0.0, 0.0), (0.0, 0.3), (0.3, 0.7), (0.7, 1.0), (1.0, 1.0))


def _rates(run: Run, set_name: str, n: int) -> np.ndarray:
    return run.rates("base", set_name, n)


def p0_rates(run: Run, source: str) -> np.ndarray:
    """p̂0 per problem from one or more base sets (equal sizes, so pooling is the mean of rates)."""
    n = int(run.config["sampling"]["n_measure"])
    return np.mean([_rates(run, s, n) for s in SOURCES[source]], axis=0)


def _bucket_label(lo: float, hi: float) -> str:
    if lo == hi:
        return f"p̂A = {lo:g}"
    return f"{lo:g} < p̂A < {hi:g}"


def null_diagnostic(run: Run) -> dict[str, Any]:
    n_b = int(run.config["sampling"]["n_measure"])
    k_a, m_a = run.set_a()
    pa = k_a / m_a
    pb, pn = _rates(run, "B", n_b), _rates(run, "null", n_b)
    d = pn - pb
    n = len(d)
    se = float(np.std(d, ddof=1) / np.sqrt(n))
    seed = np.random.SeedSequence([int(run.measure_param("boot_seed")), zlib.crc32(b"null-diagnostic")])
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(N_PERM, n))
    perm = (signs * d).mean(axis=1)
    p_perm = float((np.sum(np.abs(perm) >= abs(d.mean()) - 1e-15) + 1) / (N_PERM + 1))
    pool = (pb + pn) / 2
    expected = float(np.sum(2 * pool * (1 - pool) / n_b) / n**2)
    observed = float(np.var(d, ddof=1) / n)
    buckets = []
    for lo, hi in BUCKETS:
        member = (pa == lo) if lo == hi else (pa > lo) & (pa < hi)
        buckets.append({"bucket": _bucket_label(lo, hi), "n_problems": int(member.sum()),
                        "contribution": float(d[member].sum() / n)})
    untrained = []
    for arm in (a for a in run.arms if a["trainer"] == "lm_none"):
        for seed in run.seeds:
            if not run.sample_file(f"{arm['name']}/seed{seed}/step{run.final_step}", "eval").exists():
                continue
            p1 = run.eval_rates(arm["name"], seed)
            untrained.append({"arm": arm["name"], "seed": seed, "mean_success": float(p1.mean()),
                              "minus_B": float((p1 - pb).mean()), "minus_A": float((p1 - pa).mean())})
    return {
        "untrained_arms": untrained,
        "n_problems": n, "n_per_problem": {"A": int(m_a.min()), "B": n_b, "null": n_b},
        "mean_success": {"A": float(pa.mean()), "B": float(pb.mean()), "null": float(pn.mean())},
        "null_minus_B": {"mean": float(d.mean()), "se": se, "z": float(d.mean() / se) if se > 0 else None,
                         "sign_flip_p": p_perm, "n_perm": N_PERM},
        "B_minus_A": float((pb - pa).mean()), "null_minus_A": float((pn - pa).mean()),
        "variance_of_mean_difference": {"observed": observed, "binomial_expected": expected,
                                        "ratio": observed / expected if expected > 0 else None},
        "by_set_A_rate": buckets,
    }


def _rule_inputs(result: dict) -> dict[str, Any]:
    v = result["verdicts"]
    return {
        "L1": {"verdict": v["L1"]["verdict"],
               "units": [{"arm": u["arm"], "seed": u["seed"], "verdict": u.get("verdict"),
                          "never_correct": u.get("never_correct")} for u in v["L1"]["units"]]},
        "L2": {"verdict": v["L2"]["verdict"],
               "units": [{"arm": u["arm"], "pooled": u["pooled"]} for u in v["L2"]["units"]]},
        "L3": {"verdict": v["L3"]["verdict"],
               "units": [{"arm": u["arm"], "pooled": u["all"]["pooled"]} for u in v["L3"]["units"]]},
    }


def compute_sensitivity(run: Run, prereg: dict) -> dict[str, Any]:
    by_source = {}
    for source in SOURCES:
        rates = p0_rates(run, source)
        run.base_rates = lambda rates=rates: rates  # type: ignore[method-assign]
        try:
            by_source[source] = _rule_inputs(compute(run, prereg))
        finally:
            del run.base_rates
    verdicts = {claim: {s: by_source[s][claim]["verdict"] for s in SOURCES} for claim in ("L1", "L2", "L3")}
    return {
        "status": "post-hoc (added after the null check failed; DECISIONS L21); the registered verdicts use p̂0 "
                  "from B and are unchanged",
        "null_check_passed": all(r.get("passed", False) for r in run.null.get("records", [])),
        "null_diagnostic": null_diagnostic(run),
        "verdicts_by_p0_source": verdicts,
        "verdicts_stable": {c: len(set(v.values())) == 1 for c, v in verdicts.items()},
        "rule_inputs": by_source,
        "prereg": prereg,
    }


def render(res: dict, name: str) -> str:
    dg = res["null_diagnostic"]
    nb = dg["null_minus_B"]
    var = dg["variance_of_mean_difference"]
    md = [f"# Sensitivity of L1-L3 to the base sample: {name}", "",
          ("Generated by `capsplit.harness.lm.sensitivity`. **Post-hoc**: this stage was added after the main run's "
           "pre-registered null check failed (DECISIONS L21); it runs on any config. The registered verdicts (p̂0 "
           "from base set B) are in `verdicts.md` and are not changed by this file."), "",
          f"Null check passed: **{res['null_check_passed']}**.", "",
          "## Null-check diagnostic", "",
          *md_table(["quantity", "value"], [
              ["problems", dg["n_problems"]],
              ["samples per problem (A / B / null)",
               f"{dg['n_per_problem']['A']} / {dg['n_per_problem']['B']} / {dg['n_per_problem']['null']}"],
              ["mean success, set A", md_num(dg["mean_success"]["A"], sign=False)],
              ["mean success, set B", md_num(dg["mean_success"]["B"], sign=False)],
              ["mean success, set null", md_num(dg["mean_success"]["null"], sign=False)],
              ["null − B, mean per-problem difference", md_num(nb["mean"])],
              ["standard error", md_num(nb["se"], sign=False)],
              ["z", md_num(nb["z"], 2)],
              [f"sign-flip permutation p (two-sided, {nb['n_perm']} permutations)", md_num(nb["sign_flip_p"], sign=False)],
              ["B − A, mean", md_num(dg["B_minus_A"])],
              ["null − A, mean", md_num(dg["null_minus_A"])],
              ["variance of the mean difference: observed / binomial expected", md_num(var["ratio"], 3, sign=False)],
          ]), "",
          ("B and null are produced by the same sampling job with different seed streams. A variance ratio near 1 "
          "means the per-problem differences scatter as independent binomial samples should; set A (larger, "
          "independent) is a third estimate of the same base rates."), "",
          ("Untrained arms (`lm_none`: the adapter is never updated, so each is one more independent sample of the "
           "base policy, drawn by the measure stage):"), "",
          *md_table(["arm", "seed", "mean success", "minus B", "minus A"],
                    [[u["arm"], u["seed"], md_num(u["mean_success"], sign=False), md_num(u["minus_B"]),
                      md_num(u["minus_A"])] for u in dg["untrained_arms"]]), "",
          "Contribution of each set-A success band to the null − B total (summed difference / number of problems):", "",
          *md_table(["set-A band", "problems", "contribution"],
                    [[b["bucket"], b["n_problems"], md_num(b["contribution"])] for b in dg["by_set_A_rate"]]), "",
          "## Verdicts with p̂0 from each base sample", "",
          "The same rules and thresholds as `verdicts.md`; only the source of p̂0 changes.", "",
          *md_table(["claim", *SOURCES, "same verdict?"],
                    [[c, *[f"**{v[s].upper()}**" for s in SOURCES], "yes" if res["verdicts_stable"][c] else "**no**"]
                     for c, v in res["verdicts_by_p0_source"].items()]), "",
          "### Numbers entering each rule", ""]
    rows = []
    for s in SOURCES:
        ri = res["rule_inputs"][s]
        rows += [[s, "L1", f"{u['arm']} seed {u['seed']}", "never-correct contribution", md_ci(u["never_correct"])]
                 for u in ri["L1"]["units"] if u.get("never_correct")]
        rows += [[s, "L2", f"{u['arm']} pooled", "heldout unreachable contribution", md_ci(u["pooled"])]
                 for u in ri["L2"]["units"]]
        rows += [[s, "L3", f"{u['arm']} pooled", "all-problems reachable contribution", md_ci(u["pooled"])]
                 for u in ri["L3"]["units"]]
    md += md_table(["p̂0 from", "claim", "unit", "statistic", "estimate [lo, hi]"], rows)
    md.append("")
    return "\n".join(md)


def build_sensitivity(out_dir: str | Path, prereg_path: str | Path) -> Path:
    """Write `<out_dir>/report/sensitivity_p0.{json,md}`; returns the JSON path."""
    run = load_run(out_dir)
    res = compute_sensitivity(run, load_prereg(prereg_path))
    rep = run.out / "report"
    path = write_json_det(rep / "sensitivity_p0.json", res)
    write_text(rep / "sensitivity_p0.md", render(res, run.name))
    return path
