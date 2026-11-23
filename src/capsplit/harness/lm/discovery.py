"""Discovery in the LM experiment compared with the waiting-time model (descriptive; prereg "Also reported").

The pre-registration (`docs/preregistration_lm.md`) promises a descriptive comparison of each
training problem's first-success update with the no-drift waiting-time model of
`docs/prediction_discovery.md`: P(no correct sample in the first k draws of a problem) =
(1 - p)^(G k), where G is the group size (samples per draw) and p the base success rate. With one
model and 3 seeds this is not a hypothesis test. It was implemented after the results were seen
(DECISIONS L22); nothing in it feeds a verdict.

Time unit: draws of the problem, not updates. `lm_rl` draws `problems_per_update` problems per
update from a seeded epoch cycle over the train problems (`trainers.ProblemCycle`). When the number
of train problems is a multiple of `problems_per_update`, every block of n_train / ppu updates
draws each problem exactly once, so the draw index of update u (updates are numbered from 1) is
exactly ceil(u * ppu / n_train). The code checks this against the logged per-checkpoint episode
counts and reports any mismatch; when the divisibility fails the index is marked approximate.

p per problem: set A (the classification sample, `n_classify` per problem). For problems with
0 successes in set A the point estimate predicts "never"; the Clopper-Pearson upper bound is
also reported as the largest p consistent with set A, giving an upper reference for how many
such problems could be discovered by chance.

Statistics per (arm, class) pooled over seeds, and per seed:
* observed number discovered vs predicted sum_i [1 - (1 - p_i)^(G D_i)] (D_i = draws in the run);
* randomised PIT over problems with p > 0 (as in `harness.discovery`): U = F(k - 1) + V (F(k) -
  F(k - 1)) for a first success at draw k, U = F(D) + V (1 - F(D)) if never; Uniform(0, 1) under
  the model; mean U > 1/2 means later than predicted; z treats observations as independent;
* discoveries among problems with p = 0 in set A (the model with p = 0 predicts none).

Writes `<out>/report/discovery_lm.{json,md}`. Every number comes from files.
"""
from __future__ import annotations

import math
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from ..discovery import geometric_cdf, pit_summary
from .measure import clopper_pearson
from .report import Run, load_run, md_num, md_table, write_json_det, write_text

CLASSES = ("unreachable", "ambiguous", "reachable")


def _rl_arms(run: Run) -> list[dict]:
    return [a for a in run.arms if a["trainer"] == "lm_rl"]


def draw_index(update: int, ppu: int, n_train: int) -> int:
    """Draw number (1-based) of a problem drawn at `update` under the exact epoch cycle."""
    return math.ceil(update * ppu / n_train)


def _check_cycle(prompts: dict, group: int, ppu: int, n_train: int) -> int:
    """Number of problems whose logged draws at some checkpoint fall outside the epoch-cycle range."""
    per_epoch = n_train // ppu
    bad = 0
    for entry in prompts.values():
        by_c = entry.get("episodes_by_checkpoint") or {}
        for key, episodes in by_c.items():
            c = int(key)
            draws = episodes / group
            if not (c // per_epoch <= draws <= -(-c // per_epoch)):
                bad += 1
                break
    return bad


def observations(run: Run, arm: dict, seed: int) -> dict[str, Any]:
    params = arm.get("params", {})
    group, ppu = int(params.get("group", 8)), int(params.get("problems_per_update", 8))
    ids = [int(i) for i in run.ids("train")]
    n_train = len(ids)
    log = run.training.get((arm["name"], seed)) or {}
    prompts = log.get("prompts", {})
    exact = n_train % ppu == 0
    k_a, m_a = run.set_a()
    cls = run.class_array()
    alpha = float(run.measure_param("alpha"))
    rows = []
    for pid in ids:
        entry = prompts.get(str(pid))
        if entry is None:
            continue
        first = entry.get("first_success_update")
        draws = int(entry.get("episodes_sampled", 0)) // group
        p = float(k_a[pid] / m_a[pid])
        rows.append({"problem": pid, "class": str(cls[pid]), "p": p,
                     "p_upper": float(clopper_pearson(int(k_a[pid]), int(m_a[pid]), alpha)[1]),
                     "draws": draws, "first_success_update": first,
                     "first_success_draw": None if first is None else draw_index(int(first), ppu, n_train)})
    mismatch = _check_cycle(prompts, group, ppu, n_train) if exact else None
    return {"arm": arm["name"], "seed": seed, "group": group, "problems_per_update": ppu, "n_train": n_train,
            "draw_index_exact": exact, "cycle_mismatches": mismatch, "rows": rows}


def summarise(rows: list[dict], group: int, rng: np.random.Generator) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    disc = [r["first_success_draw"] is not None for r in rows]
    pred = [1.0 - (1.0 - r["p"]) ** (group * r["draws"]) for r in rows]
    pred_upper = [1.0 - (1.0 - r["p_upper"]) ** (group * r["draws"]) for r in rows]
    pos = [r for r in rows if r["p"] > 0]
    v = rng.random(len(pos))
    u = np.empty(len(pos))
    for i, r in enumerate(pos):
        if r["p"] >= 1.0:  # discovered at the first draw with certainty; avoids log(0) in the CDF
            u[i] = v[i]
            continue
        k = r["first_success_draw"]
        if k is None:
            lo, hi = float(geometric_cdf(r["draws"], r["p"], group)), 1.0
        else:
            lo, hi = float(geometric_cdf(k - 1, r["p"], group)), float(geometric_cdf(k, r["p"], group))
        u[i] = lo + v[i] * (hi - lo)
    zero = [r for r in rows if r["p"] == 0]
    zero_disc = sum(r["first_success_draw"] is not None for r in zero)
    return {
        "n": len(rows), "observed_discovered": int(sum(disc)), "predicted_discovered": float(sum(pred)),
        "predicted_discovered_at_p_upper": float(sum(pred_upper)),
        "pit_p_positive": pit_summary(u),
        "n_p_zero": len(zero), "discovered_p_zero": int(zero_disc),
        "predicted_discovered_p_zero_at_p_upper": float(sum(1.0 - (1.0 - r["p_upper"]) ** (group * r["draws"])
                                                            for r in zero)),
    }


def compute_discovery(run: Run) -> dict[str, Any]:
    units = []
    for arm in _rl_arms(run):
        per_seed = [observations(run, arm, s) for s in run.seeds]
        group = per_seed[0]["group"]
        out: dict[str, Any] = {"arm": arm["name"], "group": group,
                               "problems_per_update": per_seed[0]["problems_per_update"],
                               "draw_index_exact": all(o["draw_index_exact"] for o in per_seed),
                               "cycle_mismatches": {str(o["seed"]): o["cycle_mismatches"] for o in per_seed},
                               "by_class": {}, "by_seed": {}}
        for c in CLASSES:
            rng = np.random.default_rng([int(run.measure_param("boot_seed")), zlib.crc32(f"{arm['name']}/{c}".encode())])
            pooled = [r for o in per_seed for r in o["rows"] if r["class"] == c]
            out["by_class"][c] = summarise(pooled, group, rng)
            out["by_seed"][c] = {str(o["seed"]): summarise([r for r in o["rows"] if r["class"] == c], group,
                                                           np.random.default_rng([int(run.measure_param("boot_seed")),
                                                                                  zlib.crc32(f"{arm['name']}/{c}/{o['seed']}".encode())]))
                                 for o in per_seed}
        out["per_problem"] = [{"seed": o["seed"], **r} for o in per_seed for r in o["rows"]]
        units.append(out)
    return {"status": "descriptive (pre-registered as 'also reported'; implemented after results were seen, "
                      "DECISIONS L22); not a hypothesis test",
            "model": "P(no correct sample in the first k draws) = (1 - p)^(G k), p from set A",
            "arms": units,
            "excluded": [a["name"] for a in run.arms if a["trainer"] == "lm_sft_then_rl"]}


def _fmt(x: Any, digits: int = 2) -> str:
    return "n/a" if x is None else md_num(x, digits, sign=False)


def render(res: dict, name: str) -> str:
    md = [f"# Discovery vs the waiting-time model: {name}", "",
          "Generated by `capsplit.harness.lm.discovery`. Descriptive only (pre-registered under 'Also reported'; the "
          "code was written after the results were seen, DECISIONS L22). Model: " + res["model"] + ", G samples per "
          "draw; time is counted in draws of the problem (see the module docstring).", ""]
    if res["excluded"]:
        md += [(f"Arms not included (their RL phase starts from an SFT-trained policy, not the base): "
               f"{', '.join(res['excluded'])}."), ""]
    for u in res["arms"]:
        md += [f"## Arm `{u['arm']}` (G = {u['group']}, {u['problems_per_update']} problems per update)", "",
               (f"Draw index exact: {u['draw_index_exact']}; problems whose logged episode counts disagree with the "
               f"epoch cycle, per seed: {u['cycle_mismatches']}."), "",
               "Pooled over seeds (one observation = one problem in one seed):", "",
               *md_table(["class", "observations", "discovered (observed)", "discovered (predicted, p̂A)",
                          "discovered (predicted, CP upper p)", "PIT mean (p̂A > 0)", "PIT z", "n (p̂A > 0)",
                          "p̂A = 0: observations", "p̂A = 0: discovered", "p̂A = 0: predicted at CP upper p"],
                         [[c, s["n"], s.get("observed_discovered"), _fmt(s.get("predicted_discovered")),
                           _fmt(s.get("predicted_discovered_at_p_upper")),
                           _fmt((s.get("pit_p_positive") or {}).get("mean"), 3),
                           _fmt((s.get("pit_p_positive") or {}).get("z")),
                           (s.get("pit_p_positive") or {}).get("n"), s.get("n_p_zero"), s.get("discovered_p_zero"),
                           _fmt(s.get("predicted_discovered_p_zero_at_p_upper"))]
                          for c, s in u["by_class"].items() if s["n"]]), "",
               "Per seed (observed / predicted discovered, p̂A):", "",
               *md_table(["class", *u["by_seed"][CLASSES[0]].keys()],
                         [[c, *[f"{s.get('observed_discovered', 0)} / {_fmt(s.get('predicted_discovered'))}"
                                for s in u["by_seed"][c].values()]] for c in CLASSES if u["by_class"][c]["n"]]), "",
               ("Unreachable-class problems discovered in at least one seed (first-success draw per seed; '-' = "
               "never):"), ""]
        rows: dict[int, dict] = {}
        for r in u["per_problem"]:
            if r["class"] == "unreachable":
                rows.setdefault(r["problem"], {"p": r["p"], "p_upper": r["p_upper"], "draws": r["draws"], "k": {}})
                rows[r["problem"]]["k"][r["seed"]] = r["first_success_draw"]
        seeds = sorted({r["seed"] for r in u["per_problem"]})
        md += md_table(["problem", "p̂A", "CP upper p", *[f"seed {s}" for s in seeds]],
                       [[pid, _fmt(v["p"], 4), _fmt(v["p_upper"], 4),
                         *["-" if v["k"].get(s) is None else v["k"][s] for s in seeds]]
                        for pid, v in sorted(rows.items()) if any(k is not None for k in v["k"].values())])
        md.append("")
    md += [("Reading: under the no-drift model the PIT mean is 1/2; above 1/2 means first successes came later than "
           "the base rate predicts (or never), below 1/2 earlier. For p̂A = 0 the point estimate predicts no "
           "discovery; the CP-upper column is the most discoveries set A cannot rule out."), ""]
    return "\n".join(md)


def build_discovery(out_dir: str | Path) -> Path:
    """Write `<out_dir>/report/discovery_lm.{json,md}`; returns the JSON path."""
    run = load_run(out_dir)
    res = compute_discovery(run)
    rep = run.out / "report"
    path = write_json_det(rep / "discovery_lm.json", res)
    write_text(rep / "discovery_lm.md", render(res, run.name))
    return path
