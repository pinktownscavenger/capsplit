"""Verify that a base policy on a `rule_task` meets the design criteria (a)-(e), by exact enumeration.

Everything is computed here from `capsplit.engine` and `capsplit.tempering`, not from a
harness backend, so this check is independent of the measurement code it is meant to feed.

Criteria (primary budget N = 64, c = 1; thresholds at other (N, c) are reported):

a) one prompt-conditioned network serves every prompt, and each difficulty has train and
   held-out prompts.
b) hard prompts: every correct solution has q_base < 0.5/N (the loosest threshold, c = 0.5);
   the most likely sequence is incorrect; and for both tempering families and every beta in
   `DEFAULT_BETAS`, every hard correct solution's tempered q stays < 0.5/N.
c) easy prompts: every correct solution has q_base >= 1/N; mean easy J <= 0.6; max easy J <= 0.8.
d) sup over beta and both families of the mean J over all prompts (one shared beta) <= 0.6.
e) each hard correct solution shares exactly its first k tokens with the prompt's easy-rule
   sequence and differs at position k; median hard q_base is within a factor 2 of the target.

(b), (c) and (d) are checked on all prompts and on the train and held-out splits separately.

CLI::

    python -m capsplit.harness.bases.verify configs/base_v1.json --seeds 0-9 --out docs/base_verification.md
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from capsplit.engine import TrajectoryTable, solution_probs, table_success_prob, trajectory_probs
from capsplit.tempering import DEFAULT_BETAS, sequence_probs

from ..interfaces import Policy, Task

PRIMARY_N = 64
NS = (64, 16)
CS = (0.5, 1.0, 2.0)
EASY_MEAN_J_MAX, EASY_MAX_J_MAX, FRONTIER_MAX = 0.6, 0.8, 0.6
CALIBRATION_FACTOR = 2.0
_GOLDEN = (math.sqrt(5) - 1) / 2


def _betas() -> np.ndarray:
    return np.unique(np.r_[np.asarray(DEFAULT_BETAS, dtype=float), 1.0])


def _table(task: Task, prompt_id: int) -> TrajectoryTable:
    table = task.table(prompt_id)
    if table is None:
        raise ValueError(f"{type(task).__name__} has no exact table")
    return table


def _all_logits(task: Task, policy: Policy, ids: Sequence[int]) -> list:
    batched = getattr(policy, "all_state_logits", None)
    if batched is not None:
        return list(batched(task, list(ids)))
    return [policy.state_logits(task, pid) for pid in ids]


def _family_probs(table, logits, family: str, beta: float) -> np.ndarray:
    if family == "per_step":
        return trajectory_probs(table, logits, beta)
    return sequence_probs(table, logits, beta)


def _shared_beta_sup(j_of: Callable[[str, float], float], betas: np.ndarray) -> dict:
    """sup over beta (grid, then golden-section refinement around the best grid points) and family."""
    best: dict[str, Any] = {"value": -1.0, "beta": 1.0, "family": "per_step"}
    for family in ("per_step", "sequence"):
        values = np.array([j_of(family, b) for b in betas])
        for i in np.argsort(values)[::-1][:2]:
            value, beta = float(values[i]), float(betas[i])
            lo, hi = math.log(betas[max(i - 1, 0)]), math.log(betas[min(i + 1, len(betas) - 1)])
            c, d = hi - _GOLDEN * (hi - lo), lo + _GOLDEN * (hi - lo)
            fc, fd = j_of(family, math.exp(c)), j_of(family, math.exp(d))
            for _ in range(20):
                if fc > fd:
                    hi, d, fd = d, c, fc
                    c = hi - _GOLDEN * (hi - lo)
                    fc = j_of(family, math.exp(c))
                else:
                    lo, c, fc = c, d, fd
                    d = lo + _GOLDEN * (hi - lo)
                    fd = j_of(family, math.exp(d))
            for x, fx in ((c, fc), (d, fd)):
                if fx > value:
                    value, beta = fx, math.exp(x)
            if value > best["value"]:
                best = {"value": value, "beta": beta, "family": family}
    return best


def verify_base(task: Task, policy: Policy, target_q: float | None = None, n: int = PRIMARY_N) -> dict:
    """Compute criteria (a)-(e) for a base policy on a rule task. Returns values and PASS flags.

    `target_q` is the calibration target for criterion (e)'s matching check (default 0.2 / n).
    """
    target_q = 0.2 / n if target_q is None else target_q
    betas = _betas()
    prompts = list(task.prompts())
    ids = [p.id for p in prompts]
    logits = dict(zip(ids, _all_logits(task, policy, ids)))
    tables = {pid: _table(task, pid) for pid in ids}
    probs = {pid: trajectory_probs(tables[pid], logits[pid]) for pid in ids}
    q = {pid: solution_probs(tables[pid], probs[pid]) for pid in ids}
    j = {pid: table_success_prob(tables[pid], probs[pid]) for pid in ids}
    groups = {
        "all": ids,
        "train": [p.id for p in prompts if p.tags.get("split") == "train"],
        "heldout": [p.id for p in prompts if p.tags.get("split") == "heldout"],
    }
    hard_all = {p.id for p in prompts if p.tags.get("difficulty") == "hard"}
    easy_all = {p.id for p in prompts if p.tags.get("difficulty") == "easy"}
    out: dict[str, Any] = {"n": n, "target_q": target_q}

    # (a) one prompt-conditioned network; split exists for each difficulty
    split_counts = {
        f"{d}_{s}": len([p for p in prompts if p.tags.get("difficulty") == d and p.tags.get("split") == s])
        for d in ("easy", "hard")
        for s in ("train", "heldout")
    }
    shared = bool(getattr(policy, "prompt_conditioned", False))
    out["a"] = {"prompt_conditioned_network": shared, **split_counts, "pass": shared and min(split_counts.values()) > 0}

    # (b) hard prompts unreachable, mode incorrect, no tempering fixes them
    loose = 0.5 / n
    tempered_max: dict[str, Any] = {"value": 0.0, "beta": 1.0, "family": "per_step", "prompt": -1}
    for pid in sorted(hard_all):
        for family in ("per_step", "sequence"):
            for beta in betas:
                value = float(solution_probs(tables[pid], _family_probs(tables[pid], logits[pid], family, beta)).max())
                if value > tempered_max["value"]:
                    tempered_max = {"value": value, "beta": float(beta), "family": family, "prompt": pid}
    b: dict[str, Any] = {}
    for name, members in groups.items():
        hard = [pid for pid in members if pid in hard_all]
        b[name] = {
            "max_q": float(max(q[pid].max() for pid in hard)),
            "mode_correct": int(sum(tables[pid].solution_ids[int(np.argmax(probs[pid]))] >= 0 for pid in hard)),
        }
    b["tempered_max_q"] = tempered_max
    b["reachable_hard_solutions"] = {
        f"N{nn}_c{c:g}": int(sum((q[pid] >= c / nn).sum() for pid in hard_all)) for nn in NS for c in CS
    }
    b["pass"] = bool(
        all(b[g]["max_q"] < loose and b[g]["mode_correct"] == 0 for g in groups) and tempered_max["value"] < loose
    )
    out["b"] = b

    # (c) easy prompts reachable but not saturated
    c_out: dict[str, Any] = {}
    for name, members in groups.items():
        easy = [pid for pid in members if pid in easy_all]
        c_out[name] = {
            "min_q": float(min(q[pid].min() for pid in easy)),
            "mean_j": float(np.mean([j[pid] for pid in easy])),
            "max_j": float(max(j[pid] for pid in easy)),
        }
    c_out["unreachable_easy_solutions"] = {
        f"N{nn}_c{c:g}": int(sum((q[pid] < c / nn).sum() for pid in easy_all)) for nn in NS for c in CS
    }
    c_out["pass"] = bool(
        all(
            c_out[g]["min_q"] >= 1.0 / n and c_out[g]["mean_j"] <= EASY_MEAN_J_MAX and c_out[g]["max_j"] <= EASY_MAX_J_MAX
            for g in groups
        )
    )
    out["c"] = c_out

    # (d) tempering frontier headroom: one shared beta per group and family
    def mean_j(members: Sequence[int]) -> Callable[[str, float], float]:
        def j_of(family: str, beta: float) -> float:
            return float(
                np.mean([table_success_prob(tables[p], _family_probs(tables[p], logits[p], family, beta)) for p in members])
            )

        return j_of

    d: dict[str, Any] = {}
    for name, members in groups.items():
        d[name] = _shared_beta_sup(mean_j(members), betas)
    d["easy"] = _shared_beta_sup(mean_j(sorted(easy_all)), betas)
    d["pass"] = bool(all(d[g]["value"] <= FRONTIER_MAX for g in groups))
    out["d"] = d

    # (e) structural shared-prefix check and calibration matching
    k = int(getattr(task, "shared_prefix"))
    structural = True
    for pid in hard_all:
        easy_seq = tuple(getattr(task, "easy_rule_sequence")(pid))
        for sol in getattr(task, "correct_solutions")(pid):
            sol = tuple(sol)
            if sol[:k] != easy_seq[:k] or sol[k] == easy_seq[k]:
                structural = False
    hard_q = np.concatenate([q[pid] for pid in sorted(hard_all)])
    median = float(np.median(hard_q))
    ratio = median / target_q
    out["e"] = {
        "k": k,
        "structural": structural,
        "median_hard_q": median,
        "ratio_to_target": ratio,
        "hard_q_spread": float(hard_q.max() / hard_q.min()),
        "pass": bool(structural and 1 / CALIBRATION_FACTOR <= ratio <= CALIBRATION_FACTOR),
    }
    out["pass"] = all(out[x]["pass"] for x in "abcde")
    return out


# --------------------------------------------------------------------------- CLI


def _parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def _flag(ok: bool) -> str:
    return "PASS" if ok else "**FAIL**"


def _row(k: int, seed: int, r: dict, info: dict) -> str:
    n = r["n"]
    b, c, d, e = r["b"], r["c"], r["d"], r["e"]
    t = b["tempered_max_q"]
    cells = [
        str(k),
        str(seed),
        f"{info['hard_weight']:.2e}",
        f"{e['median_hard_q'] * n:.3f} ({e['ratio_to_target']:.2f}x)",
        f"{b['all']['max_q'] * n:.3f}",
        f"{t['value'] * n:.3f} ({t['family'][:3]}, {t['beta']:.2f})",
        str(b["all"]["mode_correct"]),
        f"{c['all']['min_q'] * n:.2f}",
        f"{c['all']['mean_j']:.3f}",
        f"{c['all']['max_j']:.3f}",
        f"{d['all']['value']:.3f}",
        f"{d['easy']['value']:.3f}",
        f"{info['seconds']:.1f}",
        _flag(r["a"]["pass"]),
        _flag(b["pass"]),
        _flag(c["pass"]),
        _flag(d["pass"]),
        _flag(e["pass"]),
    ]
    return "| " + " | ".join(cells) + " |"


def _summary(records: list) -> list[str]:
    def col(f):
        values = [f(r) for r in records]
        return f"{min(values):.4g} | {max(values):.4g}"

    lines = [
        "| Quantity | Min | Max | Requirement |",
        "|---|---|---|---|",
        f"| Median hard q_base × N / target | {col(lambda r: r['result']['e']['ratio_to_target'])} | in [0.5, 2] |",
        f"| Hard q spread (max/min over hard solutions) | {col(lambda r: r['result']['e']['hard_q_spread'])} | (reported) |",
        f"| Max hard q_base × N, all prompts | {col(lambda r: r['result']['b']['all']['max_q'] * r['result']['n'])} | < 0.5 |",
        f"| Max hard q_base × N, held-out | {col(lambda r: r['result']['b']['heldout']['max_q'] * r['result']['n'])} | < 0.5 |",
        f"| Max tempered hard q × N (both families, all β) | {col(lambda r: r['result']['b']['tempered_max_q']['value'] * r['result']['n'])} | < 0.5 |",
        f"| Hard prompts whose mode is correct | {col(lambda r: r['result']['b']['all']['mode_correct'])} | 0 |",
        f"| Min easy q_base × N, all prompts | {col(lambda r: r['result']['c']['all']['min_q'] * r['result']['n'])} | ≥ 1 |",
        f"| Min easy q_base × N, held-out | {col(lambda r: r['result']['c']['heldout']['min_q'] * r['result']['n'])} | ≥ 1 |",
        f"| Mean easy J (all / train / held-out, worst) | {col(lambda r: max(r['result']['c'][g]['mean_j'] for g in ('all', 'train', 'heldout')))} | ≤ 0.6 |",
        f"| Max easy J | {col(lambda r: r['result']['c']['all']['max_j'])} | ≤ 0.8 |",
        f"| Frontier: sup mean J, all prompts | {col(lambda r: r['result']['d']['all']['value'])} | ≤ 0.6 |",
        f"| Frontier: sup mean J, worst of train / held-out | {col(lambda r: max(r['result']['d'][g]['value'] for g in ('train', 'heldout')))} | ≤ 0.6 |",
        f"| Frontier: sup mean J, easy group | {col(lambda r: r['result']['d']['easy']['value'])} | (reported) |",
        f"| Build time (s) | {col(lambda r: r['info']['seconds'])} | well under 60 |",
    ]
    return lines


def _reach_table(records: list) -> list[str]:
    keys = list(records[0]["result"]["b"]["reachable_hard_solutions"])
    lines = [
        "| Threshold | Hard solutions reachable (max over runs) | Easy solutions unreachable (max over runs) |",
        "|---|---|---|",
    ]
    for key in keys:
        hard = max(r["result"]["b"]["reachable_hard_solutions"][key] for r in records)
        easy = max(r["result"]["c"]["unreachable_easy_solutions"][key] for r in records)
        lines.append(f"| {key.replace('_', ', ')} | {hard} | {easy} |")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    from ..registry import BASES, TASKS

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config")
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--levels", default=None, help="shared_prefix levels (default: config factors)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--json", default=None, help="also write the raw results as JSON")
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text())
    seeds = _parse_seeds(args.seeds)
    if args.levels is not None:
        levels = [int(x) for x in args.levels.split(",")]
    else:
        levels = list(config.get("factors", {}).get("task.shared_prefix", [config["task"]["params"]["shared_prefix"]]))
    base_params = dict(config["base"].get("params", {}))
    n = int(base_params.get("n", PRIMARY_N))
    target = float(base_params.get("target_ratio", 0.2)) / n

    records = []
    started = time.perf_counter()
    for k in levels:
        task = TASKS.create(config["task"]["type"], **{**config["task"].get("params", {}), "shared_prefix": k})
        for seed in seeds:
            policy, info = BASES.create(config["base"]["type"], **base_params).build(task, seed)
            result = verify_base(task, policy, target_q=target, n=n)
            records.append({"k": k, "seed": seed, "info": info, "result": result})
            print(
                f"k={k} seed={seed} pass={result['pass']} "
                f"median={result['e']['median_hard_q'] * n:.3f}/N tempered={result['b']['tempered_max_q']['value'] * n:.3f}/N "
                f"({info['seconds']:.1f}s)",
                flush=True,
            )
    total = time.perf_counter() - started
    all_pass = all(r["result"]["pass"] for r in records)

    header = (
        "| k | seed | w | median hard q·N (/target) | max hard q·N | max tempered hard q·N (family, β) | "
        "hard modes correct | min easy q·N | mean easy J | max easy J | frontier mean J (all) | "
        "frontier (easy) | build s | a | b | c | d | e |"
    )
    lines = [
        "# Base verification (generated)",
        "",
        "Generated by `python -m capsplit.harness.bases.verify "
        + " ".join([args.config, "--seeds", args.seeds] + (["--levels", args.levels] if args.levels else []) + ["--out", str(args.out)])
        + "`. Do not edit by hand.",
        "",
        f"- Config: `{args.config}`; task `{config['task']['type']}` {json.dumps(config['task'].get('params', {}))}; "
        f"base `{config['base']['type']}` {json.dumps(base_params)}",
        f"- Shared-structure levels k: {levels}; seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)})",
        f"- Primary budget N = {n}, c = 1; calibration target median hard q = {target * n:g}/N",
        f"- β grid: `tempering.DEFAULT_BETAS` ({len(_betas())} values, {_betas()[0]:.2f}..{_betas()[-1]:.2f}), "
        "both families (per-step β·logits; sequence-level P^β)",
        f"- Python {platform.python_version()}, numpy {np.__version__}; total wall time {total:.0f} s",
        "",
        f"**Overall: {'ALL CRITERIA PASS' if all_pass else 'SOME CRITERIA FAIL'}** "
        f"({sum(r['result']['pass'] for r in records)}/{len(records)} (k, seed) bases pass a–e).",
        "",
        "## Criteria",
        "",
        "- **a** one prompt-conditioned network serves all prompts; every difficulty has train and held-out prompts.",
        "- **b** hard prompts: every correct solution has q_base < 0.5/N; the most likely sequence is incorrect; "
        "for both tempering families and every β in the grid, every hard correct solution stays < 0.5/N.",
        "- **c** easy prompts: every correct solution has q_base ≥ 1/N; mean easy J ≤ 0.6; max easy J ≤ 0.8.",
        "- **d** sup over β and both families of mean J over all prompts (one shared β) ≤ 0.6.",
        "- **e** each hard correct solution shares exactly its first k tokens with the prompt's easy-rule sequence and "
        "differs at position k; median hard q_base within a factor 2 of the target.",
        "",
        "(b), (c) and (d) are evaluated on all prompts and separately on the train and held-out splits; "
        "a row passes only if every split passes. Probabilities are shown multiplied by N.",
        "",
        "## Summary across all (k, seed)",
        "",
        *_summary(records),
        "",
        "### Reachability at other budgets and thresholds",
        "",
        *_reach_table(records),
        "",
        "## Per-run table",
        "",
        header,
        "|" + "---|" * (header.count("|") - 1),
        *[_row(r["k"], r["seed"], r["result"], r["info"]) for r in records],
        "",
    ]
    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text)
    else:
        sys.stdout.write(text)
    if args.json:
        Path(args.json).write_text(json.dumps(records, indent=1, default=float))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
