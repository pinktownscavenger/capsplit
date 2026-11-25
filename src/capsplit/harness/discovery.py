"""Discovery as a waiting time: the pre-registered test of docs/prediction_discovery.md (addendum B4).

Model (no drift). A hard training prompt with base success probability p, trained by RLOO with
G samples of it per update, first gets a correct sample at update T, where

    P(T > t) = (1 - p)^(G t),   pi = 1 - (1 - p)^G,   E[T] = 1 / pi,
    median(T) = ceil(ln 2 / (-G ln(1 - p))),   P(undiscovered by B) = (1 - p)^(G B).

Two steps:

* `predict(config, out_dir)` computes p_i exactly from each frozen base (`task.table`,
  `base.trajectory_probs`, `engine.table_success_prob`) for every hard-train prompt, every factor
  combination and every distinct effective base seed of the config. Bases come from
  `runner.load_or_build_base`, so they are cached in `<results_root>/<name>/bases/` and are the very
  objects the runs will use. No training happens. Writes predictions.json and predictions.md.
* `analyze(results_dir, predictions, out_dir)` reads the runs' training logs
  (`first_success_update`, `trainers.rl.SampleLog`) and metrics, and compares them with the
  predictions per factor combination ("k" below is `task.shared_prefix`). Writes analysis.json,
  analysis.md and figures/.

Definitions used by `analyze` (all stated here so the markdown can quote them):

* An *observation* is one (run, hard-train prompt): observed T (null = never within B) and the
  prompt's predicted geometric (its own base's p).
* A *unit* is one (factor combination, base, prompt); its runs differ only in the training seed.
* Observed median of a set of T values = the smallest t with empirical P(T <= t) >= 1/2 (the
  same definition as the geometric median above); null when more than half are never discovered.
* KS-style distance: D = max over t = 1..B of |ECDF(t) - M(t)|, where M is the mean of the
  observations' predicted CDFs (the mixture). D+ = max(ECDF - M) (observed earlier), D- =
  max(M - ECDF) (observed later). Reported descriptively: observations are discrete, censored at
  B, from different prompts and correlated within a run, so no p-value is given.
* Randomised probability-integral transform (PIT): U = F(T - 1) + V (F(T) - F(T - 1)) for an
  observed T, and U = F(B) + V (1 - F(B)) for a never-discovered prompt, with F the prompt's
  predicted CDF and V ~ Uniform(0, 1) from numpy default_rng(0) (fresh per combination, in
  index order). Under the model U is exactly Uniform(0, 1); later-than-predicted first successes
  push U towards 1. z = (mean U - 1/2) / sqrt(1 / (12 n)) treats observations as independent.
* Rank comparison of base p: AUC = P(p of a never-discovered observation > p of a discovered
  one) + 1/2 P(tie) (Mann-Whitney U / (n1 n2)). Under the waiting-time picture AUC < 1/2.
* Drift: for a discovered observation, the last checkpoint c < T (the policy that sampled update
  T is the one after T - 1 updates) and ratio = J(c) / p with J(c) from metrics `extra.j1`. c = 0
  gives ratio 1 by construction, so only c > 0 is "informative". For a never-discovered prompt
  the final checkpoint is used and reported separately.

Verdict per combination (`verdict`), the doc's criteria read literally, with these thresholds
(constants below, fixed before any seed-study run was seen):

* "most prompts within a factor of 2": more than half of the units with >= MIN_REPLICATES runs
  have 1/2 <= observed median / predicted median <= 2. With no such unit (e.g. one run per base,
  as in seedstudy_base) the pooled ratio (observed pooled median / mixture median) must lie in
  [1/2, 2] instead, because a single geometric draw lies within a factor 2 of its median only
  ~46% of the time.
* "never discovered ~ 0": observed fraction <= NEVER_TOL. "non-zero": at least one.
* "systematically later": PIT z > PIT_Z and pooled ratio > 1 (or the pooled median is never
  reached). "much earlier": PIT z < -PIT_Z and pooled ratio < 1/2.
* "high-p prompts never discovered across most seeds while low-p prompts are discovered": some
  unit is discovered in fewer than half its runs (needs >= MIN_REPLICATES runs; otherwise the
  observation level, never vs discovered, is used) and the rank AUC of p (mostly-undiscovered vs
  the rest) exceeds 1/2.

Order: against the model ("much earlier" at k = 0, or the high-p anomaly at any k) > agreement >
later with never > 0 ("in the predicted direction" at k = 0; "negative drift" elsewhere) > much
earlier at k != 0 ("positive drift") > "none of the stated outcomes".

Variance decomposition (`variance_decomposition`): final per-prompt delta_cap of hard_train at the
primary (N, c) (`extra.delta_cap`), per combination. In a study with a fixed base (`base_seed`
set) the per-prompt variance across runs is the training-seed component V_train; with a fixed
training seed (`train_seed` set) it is V_base (which also contains sampling variation, since a
different base samples differently). share = V_train / V_base; share <= 1/3: "outcome
determined by the base", >= 2/3: "outcome depends on sampling luck", else "mixed". The same is
done for the run-level mean delta_cap(hard_train). Sample variances use ddof = 1; the
per-prompt value is the mean over prompts of each prompt's variance across runs.

Determinism: no timestamps or absolute paths in outputs, sorted-key JSON (`runner.dumps`),
matplotlib Agg with fixed metadata.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from capsplit.engine import table_success_prob

from .report import _SERIES, _fkey, _plt, _read_json, _save, _write
from .runner import dumps

TRAIN_GROUP = "hard_train"
RL_TRAINER = "rloo"
DEFAULT_GROUP = 16  # trainers.rl.RLOO default
WITHIN_FACTOR = 2.0
MOST = 0.5  # "most" = strictly more than this fraction
NEVER_TOL = 0.01
PIT_Z = 3.0
MIN_REPLICATES = 5
PIT_BINS = 10
PIT_SEED = 0
SHARE_BASE, SHARE_LUCK = 1.0 / 3.0, 2.0 / 3.0
K_FACTOR = "task.shared_prefix"

AGREEMENT = "agreement"
PREDICTED_DIRECTION = "disagreement in the predicted direction"
AGAINST = "disagreement against the model"
NEGATIVE_DRIFT = "later than predicted (negative drift)"
POSITIVE_DRIFT = "earlier than predicted (positive drift)"
NONE_MET = "none of the stated outcomes"

BY_BASE = "outcome determined by the base"
BY_LUCK = "outcome depends on sampling luck"
MIXED = "mixed"


# --------------------------------------------------------------------------- geometric model


def discovery_prob(p: float, group: int) -> float:
    """pi = 1 - (1 - p)^G: probability that one update's group holds a correct sample."""
    if p >= 1.0:
        return 1.0
    return float(-math.expm1(group * math.log1p(-p)))


def expected_first_success(p: float, group: int) -> float | None:
    """E[T] = 1 / pi (None when p = 0: never)."""
    pi = discovery_prob(p, group)
    return None if pi <= 0.0 else 1.0 / pi


def median_first_success(p: float, group: int) -> int | None:
    """median T = ceil(ln 2 / (-G ln(1 - p))), at least 1 (None when p = 0)."""
    if p <= 0.0:
        return None
    if p >= 1.0:
        return 1
    return max(1, math.ceil(math.log(2.0) / (-group * math.log1p(-p))))


def p_undiscovered(p: float, group: int, budget: int) -> float:
    """P(T > B) = (1 - p)^(G B)."""
    if p >= 1.0:
        return 0.0
    return float(math.exp(group * budget * math.log1p(-p)))


def geometric_cdf(t: Any, p: Any, group: int) -> np.ndarray:
    """P(T <= t) = 1 - (1 - p)^(G t), elementwise (t >= 0)."""
    t_arr = np.asarray(t, dtype=float)
    p_arr = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    with np.errstate(divide="ignore"):
        log_q = np.log1p(-p_arr)
    out = -np.expm1(group * t_arr * log_q)
    return np.where(t_arr <= 0, 0.0, np.nan_to_num(out, nan=1.0))


def prompt_prediction(pid: int, p: float, group: int, budget: int) -> dict:
    """The per-prompt prediction record written to predictions.json."""
    return {
        "id": int(pid),
        "p": float(p),
        "pi": discovery_prob(p, group),
        "expected_T": expected_first_success(p, group),
        "median_T": median_first_success(p, group),
        "p_undiscovered": p_undiscovered(p, group, budget),
    }


def empirical_median(times: Sequence[int | None], budget: int) -> int | None:
    """Smallest t with empirical P(T <= t) >= 1/2; None if never reached by `budget`."""
    if not times:
        return None
    seen = sorted(t for t in times if t is not None)
    need = math.ceil(len(times) / 2)
    if len(seen) < need:
        return None
    t = seen[need - 1]
    return int(t) if t <= budget else None


def mixture_median(ps: Sequence[float], group: int, budget: int) -> int | None:
    """Smallest t in 1..B with the mean predicted CDF >= 1/2; None if not reached by B."""
    cdf = mixture_cdf(ps, group, budget)
    hit = np.flatnonzero(cdf >= 0.5)
    return int(hit[0]) + 1 if hit.size else None


def mixture_cdf(ps: Sequence[float], group: int, budget: int) -> np.ndarray:
    """M(t) for t = 1..B: the mean over `ps` of each geometric CDF."""
    if not len(ps):
        return np.zeros(budget)
    t = np.arange(1, budget + 1, dtype=float)
    return geometric_cdf(t[None, :], np.asarray(ps, dtype=float)[:, None], group).mean(axis=0)


def empirical_cdf(times: Sequence[int | None], budget: int) -> np.ndarray:
    """ECDF(t) for t = 1..B; never-discovered entries never count."""
    counts = np.zeros(budget + 1)
    for t in times:
        if t is not None and 1 <= t <= budget:
            counts[t] += 1
    return np.cumsum(counts)[1:] / max(len(times), 1)


def ks_distance(times: Sequence[int | None], ps: Sequence[float], group: int, budget: int) -> dict:
    """D, D+ (observed earlier) and D- (observed later), the t of D, and the iid reference 1.36/sqrt(n)."""
    if not len(times):
        return {"D": None, "D_plus": None, "D_minus": None, "t_at_D": None, "n": 0, "iid_reference_5pct": None}
    diff = empirical_cdf(times, budget) - mixture_cdf(ps, group, budget)
    at = int(np.argmax(np.abs(diff)))
    return {
        "D": float(np.abs(diff).max()),
        "D_plus": float(max(diff.max(), 0.0)),
        "D_minus": float(max((-diff).max(), 0.0)),
        "t_at_D": at + 1,
        "n": len(times),
        "iid_reference_5pct": 1.36 / math.sqrt(len(times)),
    }


def randomized_pit(
    times: Sequence[int | None], ps: Sequence[float], group: int, budget: int, rng: np.random.Generator
) -> np.ndarray:
    """U_i = F_i(T - 1) + V (F_i(T) - F_i(T - 1)); censored: F_i(B) + V (1 - F_i(B))."""
    out = np.empty(len(times))
    v = rng.random(len(times))
    for i, (t, p) in enumerate(zip(times, ps)):
        if t is None:
            lo, hi = float(geometric_cdf(budget, p, group)), 1.0
        else:
            lo, hi = float(geometric_cdf(t - 1, p, group)), float(geometric_cdf(t, p, group))
        out[i] = lo + v[i] * (hi - lo)
    return out


def pit_summary(u: np.ndarray, bins: int = PIT_BINS) -> dict:
    n = len(u)
    if not n:
        return {"n": 0, "mean": None, "z": None, "histogram": [], "bins": bins}
    hist, _ = np.histogram(u, bins=bins, range=(0.0, 1.0))
    mean = float(u.mean())
    return {
        "n": n,
        "mean": mean,
        "z": (mean - 0.5) / math.sqrt(1.0 / (12.0 * n)),
        "histogram": [int(h) for h in hist],
        "histogram_fraction": [float(h) / n for h in hist],
        "bins": bins,
    }


def rank_auc(high: Sequence[float], low: Sequence[float]) -> float | None:
    """P(x > y) + 1/2 P(x = y) for x from `high`, y from `low` (Mann-Whitney U / (n1 n2))."""
    if not len(high) or not len(low):
        return None
    x, y = np.asarray(high, dtype=float), np.asarray(low, dtype=float)
    greater = (x[:, None] > y[None, :]).sum()
    ties = (x[:, None] == y[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(x) * len(y)))


def _ratio(observed: float | None, predicted: float | None) -> float | None:
    """observed / predicted; inf when observed is never reached; None when predicted is never."""
    if predicted is None:
        return None
    if observed is None:
        return math.inf
    return float(observed) / float(predicted)


def share_verdict(share: float | None) -> str | None:
    if share is None:
        return None
    if share <= SHARE_BASE:
        return BY_BASE
    if share >= SHARE_LUCK:
        return BY_LUCK
    return MIXED


def _var(values: Sequence[float]) -> float | None:
    return float(np.var(np.asarray(values, dtype=float), ddof=1)) if len(values) >= 2 else None


def per_prompt_variance(values: Mapping[int, Sequence[float]]) -> float | None:
    """Mean over prompts of each prompt's sample variance (ddof = 1) across runs."""
    vs = [v for v in (_var(vals) for vals in values.values()) if v is not None]
    return float(np.mean(vs)) if vs else None


def decompose(train_component: float | None, base_component: float | None) -> dict:
    """share = V_train / V_base and its verdict (see the module docstring)."""
    share = None
    if train_component is not None and base_component is not None and base_component > 0:
        share = train_component / base_component
    return {"V_train": train_component, "V_base": base_component, "share": share, "verdict": share_verdict(share)}


# --------------------------------------------------------------------------- predict


def _rl_arm(arms: Sequence[Any]) -> Any:
    matches = [a for a in arms if a.trainer == RL_TRAINER]
    if not matches:
        raise ValueError(f"the config has no {RL_TRAINER!r} arm")
    return matches[0]


def _base_unit(args: tuple) -> dict:
    """One (factor combination, effective base seed): load or build the base, return its record."""
    import torch

    from .config import build_groups
    from .registry import TASKS
    from .runner import load_or_build_base, train_prompt_ids

    config, exp_dir, combo, base_seed, run_seeds, group, budget = args
    threads = torch.get_num_threads()
    torch.set_num_threads(1)  # as runner.run_experiment, so a freshly built base is the same
    try:
        parts = config.resolved(combo)
        task = TASKS.create(parts["task"].type, **parts["task"].params)
        groups = build_groups(task, config.measure.group_by)
        if TRAIN_GROUP not in groups:
            raise ValueError(f"no {TRAIN_GROUP!r} group (measure.group_by = {list(config.measure.group_by)})")
        train_ids = set(train_prompt_ids(task))
        ids = [int(p) for p in groups[TRAIN_GROUP] if int(p) in train_ids]
        base, key = load_or_build_base(Path(exp_dir), task, parts["task"], parts["base"], base_seed)
        prompts = []
        for pid in ids:
            table = task.table(pid)
            if table is None:
                raise ValueError(f"task has no exact table for prompt {pid}")
            probs = np.asarray(base.trajectory_probs(task, pid), dtype=float)
            prompts.append(prompt_prediction(pid, table_success_prob(table, probs), group, budget))
    finally:
        torch.set_num_threads(threads)
    return base_record(combo, base_seed, key, run_seeds, prompts)


def predict(
    config_path: str | Path, out_dir: str | Path, results_root: str | Path = "results", workers: int = 1
) -> Path:
    """Write predictions.json / predictions.md for every combination x effective base seed.

    Bases are loaded from (or built into) `<results_root>/<name>/bases/` with
    `runner.load_or_build_base`; `workers > 1` builds them in a spawn-context process pool (each
    worker single-threaded, as the runner's). No training happens.
    """
    import multiprocessing

    from .config import load_config

    config = load_config(config_path)
    arm = _rl_arm(config.arms)
    group = int(arm.params.get("group", DEFAULT_GROUP))
    budget = int(config.budget)
    exp_dir = str(Path(results_root) / config.name)
    seeds_by_base: dict[int, list[int]] = {}
    for s in config.seeds:
        seeds_by_base.setdefault(config.effective_base_seed(s), []).append(int(s))
    units = [
        (config, exp_dir, combo, base_seed, seeds_by_base[base_seed], group, budget)
        for combo in config.factor_combos()
        for base_seed in sorted(seeds_by_base)
    ]
    if workers > 1 and len(units) > 1:
        with multiprocessing.get_context("spawn").Pool(min(workers, len(units))) as pool:
            bases = pool.map(_base_unit, units, chunksize=1)
    else:
        bases = [_base_unit(unit) for unit in units]
    data = predictions_document(config.name, arm.name, arm.trainer, group, budget, bases)
    out = Path(out_dir)
    _write(out / "predictions.json", dumps(data))
    _write(out / "predictions.md", predictions_markdown(data))
    return out


def base_record(combo: Mapping[str, Any], base_seed: int, key: str, run_seeds: Sequence[int], prompts: list[dict]) -> dict:
    ps = [p["p"] for p in prompts]
    return {
        "factors": dict(combo),
        "fkey": _fkey(dict(combo)),
        "base_seed": int(base_seed),
        "base_key": key,
        "run_seeds": [int(s) for s in run_seeds],
        "prompts": prompts,
        "never_fraction": float(np.mean([p["p_undiscovered"] for p in prompts])) if prompts else None,
        "median_p": float(np.median(ps)) if ps else None,
    }


def predictions_document(name: str, arm: str, trainer: str, group: int, budget: int, bases: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for fkey in dict.fromkeys(b["fkey"] for b in bases):
        prompts = [p for b in bases if b["fkey"] == fkey for p in b["prompts"]]
        ps = [p["p"] for p in prompts]
        medians = [p["median_T"] for p in prompts if p["median_T"] is not None]
        means = [p["expected_T"] for p in prompts if p["expected_T"] is not None]
        summary[fkey] = {
            "n_bases": sum(1 for b in bases if b["fkey"] == fkey),
            "n_prompts": len(prompts),
            "never_fraction": float(np.mean([p["p_undiscovered"] for p in prompts])) if prompts else None,
            "expected_never_count_per_base": float(np.sum([p["p_undiscovered"] for p in prompts]))
            / max(1, sum(1 for b in bases if b["fkey"] == fkey)),
            "p_min": float(min(ps)) if ps else None,
            "p_median": float(np.median(ps)) if ps else None,
            "p_max": float(max(ps)) if ps else None,
            "median_T_median": float(np.median(medians)) if medians else None,
            "median_T_max": int(max(medians)) if medians else None,
            "n_prompts_p_zero": sum(1 for p in ps if p <= 0.0),
            "expected_T_mean": float(np.mean(means)) if means else None,
            "mixture_median_T": mixture_median(ps, group, budget),
        }
    return {
        "config": name,
        "arm": arm,
        "trainer": trainer,
        "group": group,
        "budget": budget,
        "train_group": TRAIN_GROUP,
        "model": "geometric first-success time, no drift (docs/prediction_discovery.md)",
        "bases": bases,
        "summary": summary,
    }


def _g(x: float | None, digits: int = 3, none: str = "n/a") -> str:
    if x is None:
        return none
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return f"{x:.{digits}g}"


def _t(x: float | None, digits: int = 3) -> str:
    """A waiting time; None means it is never reached."""
    return _g(x, digits, none="never")


def predictions_markdown(data: dict) -> str:
    g, b = data["group"], data["budget"]
    lines = [
        f"# Discovery predictions: {data['config']}",
        "",
        "Generated by `python -m capsplit.harness discovery predict` from the frozen bases alone (no training).",
        "",
        (f"Model: arm `{data['arm']}` samples G = {g} episodes of each hard training prompt per update, for B = {b} "
        "updates. If a prompt's success probability stays at its base value p, the first update with a correct "
        "sample is geometric: P(T > t) = (1 - p)^(G t), E[T] = 1 / (1 - (1 - p)^G), median T = "
        "ceil(ln 2 / (-G ln(1 - p))), and P(never discovered by B) = (1 - p)^(G B)."),
        "",
        "## Summary per factor combination",
        "",
        ("| combination | bases | prompts | p min / median / max | median T (median, max over prompts) | "
        "mixture median T | mean E[T] | predicted never-discovered fraction |"),
        "|---|---|---|---|---|---|---|---|",
    ]
    for fkey, s in data["summary"].items():
        lines.append(
            f"| {fkey} | {s['n_bases']} | {s['n_prompts']} | {_g(s['p_min'])} / {_g(s['p_median'])} / "
            f"{_g(s['p_max'])} | {_t(s['median_T_median'])}, {_t(s['median_T_max'])} | {_t(s['mixture_median_T'])} | "
            f"{_t(s['expected_T_mean'])} | {_g(s['never_fraction'])} |"
        )
    lines += [""]
    for fkey, s in data["summary"].items():
        zero = f" {s['n_prompts_p_zero']} prompt(s) have p = 0 exactly and can never be discovered." if s[
            "n_prompts_p_zero"
        ] else ""
        lines.append(
            f"- **{fkey}**: without drift, half of the hard training prompts should be discovered by update "
            f"{_t(s['median_T_median'])} (the slowest prompt's median is {_t(s['median_T_max'])}), and the expected "
            f"number never discovered per base is {_g(s['expected_never_count_per_base'])}.{zero}"
        )
    lines += ["", "## Per base and prompt", ""]
    for base in data["bases"]:
        lines += [
            f"### {base['fkey']}, base seed {base['base_seed']} (`bases/{base['base_key']}`)",
            "",
            "| prompt | p | pi | E[T] | median T | P(undiscovered by B) |",
            "|---|---|---|---|---|---|",
        ]
        for p in base["prompts"]:
            lines.append(
                f"| {p['id']} | {_g(p['p'], 4)} | {_g(p['pi'], 4)} | {_t(p['expected_T'], 4)} | {_t(p['median_T'])} | "
                f"{_g(p['p_undiscovered'], 3)} |"
            )
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- analyze: loading


def _primary(measure: Mapping[str, Any]) -> tuple[int, float]:
    primary = measure.get("primary")
    if primary:
        return int(primary["n"]), float(primary["c"])
    return int(max(measure["ns"])), 1.0


def load_observations(results_dir: Path, predictions: dict) -> tuple[dict, list[dict]]:
    """index.json plus one observation per (run, hard-train prompt) of the predicted arm."""
    index = _read_json(results_dir / "index.json")
    n_primary, c_primary = _primary(index["measure"])
    budget = int(index["budget"])
    if budget != int(predictions["budget"]):
        raise ValueError(f"results budget {budget} != predictions budget {predictions['budget']}")
    pred: dict[tuple[str, int], dict] = {}
    for base in predictions["bases"]:
        for p in base["prompts"]:
            pred[(base["base_key"], int(p["id"]))] = p
    obs: list[dict] = []
    for run in index["runs"]:
        if run["arm"] != predictions["arm"]:
            continue
        files = run["files"]
        if "training" not in files:
            raise ValueError(f"run {run['run_dir']} seed {run['seed']} has no training log")
        metrics = _read_json(results_dir / files["metrics"])
        log = _read_json(results_dir / files["training"])
        key = run["base"].split("/")[-1]
        ids = [int(p) for p in metrics["groups"][TRAIN_GROUP]]
        recs = {
            int(r["step"]): r
            for r in metrics["records"]
            if r["group"] == TRAIN_GROUP and int(r["n"]) == n_primary and float(r["c"]) == c_primary
        }
        steps = sorted(recs)
        final = recs[steps[-1]]
        for i, pid in enumerate(ids):
            if (key, pid) not in pred:
                raise ValueError(f"no prediction for base {key} prompt {pid}: predictions are for another config or seed")
            p = float(pred[(key, pid)]["p"])
            entry = log["prompts"].get(str(pid))
            if entry is None:
                raise ValueError(f"training log of {run['run_dir']} seed {run['seed']} has no prompt {pid}")
            t = entry["first_success_update"]
            t = None if t is None else int(t)
            j1 = {s: float(recs[s]["extra"]["j1"][i]) for s in steps}
            before = [s for s in steps if (t is None or s < t)]
            c = before[-1] if before else 0
            obs.append({
                "fkey": _fkey(run["factors"]),
                "factors": dict(run["factors"]),
                "seed": int(run["seed"]),
                "base_key": key,
                "base_seed": run.get("base_seed"),
                "train_seed": run.get("train_seed"),
                "prompt": pid,
                "p": p,
                "j0_metrics": float(recs[steps[0]]["extra"]["j0"][i]),
                "T": t,
                "median_T": pred[(key, pid)]["median_T"],
                "p_undiscovered": float(pred[(key, pid)]["p_undiscovered"]),
                "checkpoint_before": c,
                "p_before": j1.get(c, None),
                "final_j1": j1[steps[-1]],
                "final_delta_cap": float(final["extra"]["delta_cap"][i]),
                "run_delta_cap": float(final["delta_cap"]),
            })
    if not obs:
        raise ValueError(f"no runs of arm {predictions['arm']!r} in {results_dir}")
    return index, obs


# --------------------------------------------------------------------------- analyze: statistics


def _units(obs: Sequence[dict]) -> dict[tuple[str, int], list[dict]]:
    units: dict[tuple[str, int], list[dict]] = {}
    for o in obs:
        units.setdefault((o["base_key"], o["prompt"]), []).append(o)
    return units


def combination_summary(obs: Sequence[dict], group: int, budget: int) -> dict:
    """Every per-combination statistic described in the module docstring."""
    times = [o["T"] for o in obs]
    ps = [o["p"] for o in obs]
    n = len(obs)
    never_obs = sum(1 for t in times if t is None)
    observed_median = empirical_median(times, budget)
    predicted_median = mixture_median(ps, group, budget)

    unit_rows = []
    for (key, pid), members in sorted(_units(obs).items(), key=lambda kv: (kv[0][0], kv[0][1])):
        ts = [m["T"] for m in members]
        om = empirical_median(ts, budget)
        pm = members[0]["median_T"]
        unit_rows.append({
            "base_key": key,
            "prompt": pid,
            "p": members[0]["p"],
            "runs": len(members),
            "discovered": sum(1 for t in ts if t is not None),
            "rate": sum(1 for t in ts if t is not None) / len(ts),
            "observed_median_T": om,
            "predicted_median_T": pm,
            "ratio": _ratio(om, pm),
        })
    replicated = [u for u in unit_rows if u["runs"] >= MIN_REPLICATES]
    within = [u for u in replicated if u["ratio"] is not None and 1 / WITHIN_FACTOR <= u["ratio"] <= WITHIN_FACTOR]
    later = [u for u in replicated if u["ratio"] is not None and u["ratio"] > 1]

    rng = np.random.default_rng(PIT_SEED)
    pit = pit_summary(randomized_pit(times, ps, group, budget, rng))

    never_p = [o["p"] for o in obs if o["T"] is None]
    ever_p = [o["p"] for o in obs if o["T"] is not None]
    if replicated:
        mostly_never = [u["p"] for u in replicated if u["rate"] < 0.5]
        mostly_found = [u["p"] for u in replicated if u["rate"] >= 0.5]
        anomaly_level = "unit"
    else:
        mostly_never, mostly_found, anomaly_level = never_p, ever_p, "observation"

    rates = {"all": 0, "none": 0, "some": 0}
    for u in unit_rows:
        if u["runs"] < 2:
            continue
        rates["all" if u["discovered"] == u["runs"] else "none" if u["discovered"] == 0 else "some"] += 1
    n_multi = sum(rates.values())

    informative = [o for o in obs if o["T"] is not None and o["checkpoint_before"] > 0 and o["p"] > 0]
    ratios = [o["p_before"] / o["p"] for o in informative]
    never_ratios = [o["final_j1"] / o["p"] for o in obs if o["T"] is None and o["p"] > 0]

    return {
        "n_runs": len({o["seed"] for o in obs}),
        "n_units": len(unit_rows),
        "n_observations": n,
        "median_T": {
            "observed_pooled": observed_median,
            "predicted_mixture": predicted_median,
            "ratio_pooled": _ratio(observed_median, predicted_median),
            "units_with_replicates": len(replicated),
            "fraction_within_factor_2": len(within) / len(replicated) if replicated else None,
            "fraction_later": len(later) / len(replicated) if replicated else None,
        },
        "never": {
            "observed_count": never_obs,
            "observed_fraction": never_obs / n if n else None,
            "predicted_fraction": float(np.mean([o["p_undiscovered"] for o in obs])) if obs else None,
        },
        "ks": ks_distance(times, ps, group, budget),
        "pit": pit,
        "base_p_never_vs_discovered": {
            "n_never": len(never_p),
            "n_discovered": len(ever_p),
            "median_p_never": float(np.median(never_p)) if never_p else None,
            "median_p_discovered": float(np.median(ever_p)) if ever_p else None,
            "auc_never_higher": rank_auc(never_p, ever_p),
        },
        "high_p_anomaly": {
            "level": anomaly_level,
            "n_mostly_undiscovered": len(mostly_never),
            "auc_mostly_undiscovered_higher": rank_auc(mostly_never, mostly_found),
        },
        "discovery_rate": {
            **rates,
            "units_with_multiple_runs": n_multi,
            "reading": None if not n_multi else (
                "determined by base (every prompt discovered in all or none of its runs)" if rates["some"] == 0
                else f"luck matters for {rates['some']} of {n_multi} prompt(s) (discovered in some runs only)"
            ),
        },
        "drift": {
            "n_informative": len(informative),
            "n_checkpoint_zero": sum(1 for o in obs if o["T"] is not None and o["checkpoint_before"] == 0),
            "median_ratio": float(np.median(ratios)) if ratios else None,
            "geometric_mean_ratio": float(np.exp(np.mean(np.log(np.maximum(ratios, 1e-300))))) if ratios else None,
            "fraction_below_1": float(np.mean([r < 1 for r in ratios])) if ratios else None,
            "never_discovered_final_over_base_median": float(np.median(never_ratios)) if never_ratios else None,
            "n_never_discovered": len(never_ratios),
        },
        "final_delta_cap": {
            "mean": float(np.mean([o["final_delta_cap"] for o in obs])),
            "mean_discovered": float(np.mean([o["final_delta_cap"] for o in obs if o["T"] is not None]))
            if ever_p else None,
            "mean_never": float(np.mean([o["final_delta_cap"] for o in obs if o["T"] is None])) if never_p else None,
        },
        "units": unit_rows,
    }


def verdict(summary: Mapping[str, Any], k: Any) -> dict:
    """The doc's agreement criteria, read literally (thresholds in the module docstring)."""
    med = summary["median_T"]
    pooled = med["ratio_pooled"]
    within = med["fraction_within_factor_2"]
    if within is not None:
        median_ok = within > MOST
        median_rule = f"{within:.0%} of {med['units_with_replicates']} replicated prompts within a factor 2 (need > 50%)"
    else:
        median_ok = pooled is not None and 1 / WITHIN_FACTOR <= pooled <= WITHIN_FACTOR
        median_rule = f"no prompt has >= {MIN_REPLICATES} runs; pooled median ratio {_g(pooled)} (need 0.5..2)"
    never = summary["never"]["observed_fraction"] or 0.0
    never_count = summary["never"]["observed_count"]
    never_zero = never <= NEVER_TOL
    z = summary["pit"]["z"]
    later = z is not None and z > PIT_Z and (pooled is None or pooled > 1)
    much_earlier = z is not None and z < -PIT_Z and pooled is not None and pooled < 1 / WITHIN_FACTOR
    anomaly = summary["high_p_anomaly"]
    high_p = anomaly["n_mostly_undiscovered"] > 0 and (anomaly["auc_mostly_undiscovered_higher"] or 0.0) > 0.5
    checks = {
        "median_within_factor_2": median_ok,
        "median_rule": median_rule,
        "never_approx_zero": never_zero,
        "never_nonzero": never_count > 0,
        "systematically_later": later,
        "much_earlier": much_earlier,
        "high_p_never_discovered": high_p,
        "pit_z": z,
        "pooled_ratio": pooled,
    }
    if (k == 0 and much_earlier) or high_p:
        reasons = []
        if k == 0 and much_earlier:
            reasons.append("first successes much earlier than predicted at k = 0")
        if high_p:
            reasons.append("prompts never discovered in most runs have higher base p than discovered ones")
        label, why = AGAINST, "; ".join(reasons)
    elif median_ok and never_zero:
        label, why = AGREEMENT, f"{median_rule}; never-discovered fraction {never:.3g} <= {NEVER_TOL}"
    elif later and never_count > 0:
        label = PREDICTED_DIRECTION if k == 0 else NEGATIVE_DRIFT
        why = f"PIT z = {_g(z)} > {PIT_Z} and pooled ratio {_g(pooled)} > 1; {never_count} never-discovered observation(s)"
    elif much_earlier:
        label, why = POSITIVE_DRIFT, f"PIT z = {_g(z)} < -{PIT_Z} and pooled ratio {_g(pooled)} < 0.5"
    else:
        parts = [median_rule, f"never-discovered fraction {never:.3g}", f"PIT z = {_g(z)}"]
        label, why = NONE_MET, "; ".join(parts)
    return {"k": k, "label": label, "why": why, "checks": checks}


# --------------------------------------------------------------------------- variance decomposition


def study_kind(index: Mapping[str, Any]) -> str | None:
    """"train" (base fixed, training seed varies), "base" (training seed fixed) or None."""
    runs = index["runs"]
    if not runs or "base_seed" not in runs[0]:
        return None
    base_seeds = {r["base_seed"] for r in runs}
    train_seeds = {r["train_seed"] for r in runs}
    if len(base_seeds) == 1 and len(train_seeds) > 1:
        return "train"
    if len(train_seeds) == 1 and len(base_seeds) > 1:
        return "base"
    return None


def components(obs: Sequence[dict]) -> dict[str, dict]:
    """Per combination: per-prompt and run-level variance of final delta_cap across runs."""
    out: dict[str, dict] = {}
    for fkey in dict.fromkeys(o["fkey"] for o in obs):
        mine = [o for o in obs if o["fkey"] == fkey]
        per_prompt: dict[int, list[float]] = {}
        run_level: dict[int, float] = {}
        for o in mine:
            per_prompt.setdefault(o["prompt"], []).append(o["final_delta_cap"])
            run_level[o["seed"]] = o["run_delta_cap"]
        out[fkey] = {
            "per_prompt": per_prompt_variance(per_prompt),
            "run_level": _var(list(run_level.values())),
            "run_level_mean": float(np.mean(list(run_level.values()))),
            "n_runs": len(run_level),
        }
    return out


def variance_decomposition(studies: Mapping[str, Mapping[str, dict]]) -> dict:
    """{"train": components, "base": components} -> per combination decomposition."""
    train, base = studies.get("train", {}), studies.get("base", {})
    out: dict[str, dict] = {}
    for fkey in dict.fromkeys(list(train) + list(base)):
        t, b = train.get(fkey, {}), base.get(fkey, {})
        out[fkey] = {
            "per_prompt": decompose(t.get("per_prompt"), b.get("per_prompt")),
            "run_level": decompose(t.get("run_level"), b.get("run_level")),
            "n_runs_train": t.get("n_runs"),
            "n_runs_base": b.get("n_runs"),
        }
    return out


# --------------------------------------------------------------------------- analyze


def analyze(
    results_dir: str | Path,
    predictions_path: str | Path,
    out_dir: str | Path,
    compare: Sequence[tuple[str | Path, str | Path]] = (),
) -> Path:
    """Compare observed first-success times with the predictions; write analysis.json/.md + figures.

    `compare` adds other (results_dir, predictions) pairs for the variance decomposition only
    (seedstudy_train with seedstudy_base); their kind comes from the recorded seeds.
    """
    results_dir, out = Path(results_dir), Path(out_dir)
    predictions = _read_json(Path(predictions_path))
    group, budget = int(predictions["group"]), int(predictions["budget"])
    index, obs = load_observations(results_dir, predictions)

    consistency = max(abs(o["p"] - o["j0_metrics"]) for o in obs)
    per_combo: dict[str, dict] = {}
    verdicts: dict[str, dict] = {}
    for fkey in dict.fromkeys(o["fkey"] for o in obs):
        mine = [o for o in obs if o["fkey"] == fkey]
        k = mine[0]["factors"].get(K_FACTOR)
        per_combo[fkey] = {"k": k, **combination_summary(mine, group, budget)}
        verdicts[fkey] = verdict(per_combo[fkey], k)

    studies: dict[str, dict] = {}
    kinds: dict[str, str | None] = {str(results_dir.name): study_kind(index)}
    if kinds[results_dir.name] is not None:
        studies[str(kinds[results_dir.name])] = components(obs)
    for other_dir, other_pred in compare:
        other_dir = Path(other_dir)
        other_index, other_obs = load_observations(other_dir, _read_json(Path(other_pred)))
        kind = study_kind(other_index)
        kinds[other_dir.name] = kind
        if kind is not None and kind not in studies:
            studies[kind] = components(other_obs)
    decomposition = variance_decomposition(studies) if studies else {}

    data = {
        "results": results_dir.name,
        "predictions_config": predictions["config"],
        "arm": predictions["arm"],
        "group": group,
        "budget": budget,
        "study_kinds": kinds,
        "max_abs_p_minus_metrics_j0": consistency,
        "thresholds": {
            "within_factor": WITHIN_FACTOR, "most": MOST, "never_tol": NEVER_TOL, "pit_z": PIT_Z,
            "min_replicates": MIN_REPLICATES, "share_base": SHARE_BASE, "share_luck": SHARE_LUCK,
        },
        "combinations": per_combo,
        "verdicts": verdicts,
        "variance_components": studies,
        "variance_decomposition": decomposition,
        "observations": [{k: v for k, v in o.items() if k != "factors"} for o in obs],
    }
    figures = _figures(obs, per_combo, group, budget, out / "figures")
    _write(out / "analysis.json", dumps(_finite(data)))
    _write(out / "analysis.md", analysis_markdown(data, [f.relative_to(out) for f in figures]))
    return out


def _finite(value: Any) -> Any:
    """JSON-safe: inf -> "inf" (ratios of never-reached medians)."""
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value


def _figures(obs: Sequence[dict], per_combo: Mapping[str, dict], group: int, budget: int, fig_dir: Path) -> list[Path]:
    try:
        plt = _plt()
    except ModuleNotFoundError:
        return []
    paths: list[Path] = []
    fkeys = list(per_combo)
    t = np.arange(1, budget + 1)

    fig, axes = plt.subplots(1, len(fkeys), figsize=(4 * len(fkeys), 3.2), squeeze=False, sharey=True)
    for ax, fkey in zip(axes[0], fkeys):
        mine = [o for o in obs if o["fkey"] == fkey]
        ax.step(t, empirical_cdf([o["T"] for o in mine], budget), where="post", color=_SERIES[0], label="observed")
        ax.plot(t, mixture_cdf([o["p"] for o in mine], group, budget), color=_SERIES[1], linestyle="--",
                label="predicted (no drift)")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_title(fkey)
        ax.set_xlabel("update t")
    axes[0][0].set_ylabel("P(first success <= t)")
    axes[0][0].legend(frameon=False, loc="upper left")
    path = fig_dir / "first_success_cdf.png"
    _save(fig, path)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(4.6, 4))
    every: list[float] = []
    for i, fkey in enumerate(fkeys):
        pts = [(o["p"], o["p_before"]) for o in obs if o["fkey"] == fkey and o["T"] is not None
               and o["checkpoint_before"] > 0 and o["p"] > 0 and o["p_before"] > 0]
        if pts:
            x, y = np.array(pts).T
            every += list(x) + list(y)
            ax.scatter(x, y, s=10, alpha=0.6, color=_SERIES[i % len(_SERIES)], label=fkey)
    if every:
        ax.plot([min(every), max(every)], [min(every), max(every)], color="#52514e", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("base p")
    ax.set_ylabel("p at last checkpoint before first success")
    ax.set_title("Drift before discovery (checkpoint > 0)")
    if any(ax.get_legend_handles_labels()[1]):
        ax.legend(frameon=False)
    path = fig_dir / "drift_before_discovery.png"
    _save(fig, path)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for i, fkey in enumerate(fkeys):
        units = per_combo[fkey]["units"]
        x = [u["p"] for u in units if u["p"] > 0]
        y = [u["rate"] for u in units if u["p"] > 0]
        ax.scatter(x, y, s=12, alpha=0.7, color=_SERIES[i % len(_SERIES)], label=fkey)
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("base p")
    ax.set_ylabel("fraction of runs that discovered the prompt")
    ax.legend(frameon=False)
    path = fig_dir / "discovery_rate_vs_p.png"
    _save(fig, path)
    plt.close(fig)
    paths.append(path)
    return paths


def analysis_markdown(data: dict, figures: Sequence[Path]) -> str:
    b, g = data["budget"], data["group"]
    lines = [
        f"# Discovery analysis: {data['results']}",
        "",
        ("Generated by `python -m capsplit.harness discovery analyze`. It tests the pre-registered prediction in "
        "`docs/prediction_discovery.md`: without drift, the first update with a correct sample on a hard training "
        f"prompt is geometric with P(T > t) = (1 - p)^({g} t), using each prompt's exact base p "
        f"(predictions from `{data['predictions_config']}`). Budget B = {b} updates; arm `{data['arm']}`."),
        "",
        (f"Consistency check: the predicted p and the base success rate recorded in the metrics differ by at most "
        f"{data['max_abs_p_minus_metrics_j0']:.2g}."),
        "",
        "## Verdicts",
        "",
        "| combination | verdict | evidence |",
        "|---|---|---|",
    ]
    for fkey, v in data["verdicts"].items():
        lines.append(f"| {fkey} | **{v['label']}** | {v['why']} |")
    lines += [
        "",
        ("The prediction doc tests agreement at k = 4 and expects disagreement in the predicted direction (later "
        "first successes, some prompts never discovered) at k = 0. Thresholds: within a factor of 2 for more than "
        f"half the prompts with >= {MIN_REPLICATES} runs; never-discovered fraction <= {NEVER_TOL} counts as ~0; "
        f"'systematically later' = PIT z > {PIT_Z} with pooled median ratio > 1; 'much earlier' = PIT z < -{PIT_Z} "
        "with pooled ratio < 0.5."),
        "",
        "## First-success times",
        "",
        ("| combination | runs | observations | median T observed / predicted (ratio) | prompts within x2 | "
        "never discovered observed / predicted | KS D (D+ earlier, D- later) | PIT mean (z) |"),
        "|---|---|---|---|---|---|---|---|",
    ]
    for fkey, s in data["combinations"].items():
        m, nv, ks, pit = s["median_T"], s["never"], s["ks"], s["pit"]
        within = "n/a" if m["fraction_within_factor_2"] is None else f"{m['fraction_within_factor_2']:.0%}"
        lines.append(
            f"| {fkey} | {s['n_runs']} | {s['n_observations']} | {_t(m['observed_pooled'])} / "
            f"{_t(m['predicted_mixture'])} ({_g(m['ratio_pooled'])}) | {within} | {_g(nv['observed_fraction'])} / "
            f"{_g(nv['predicted_fraction'])} | {_g(ks['D'])} ({_g(ks['D_plus'])}, {_g(ks['D_minus'])}) | "
            f"{_g(pit['mean'])} ({_g(pit['z'])}) |"
        )
    lines += [
        "",
        ("KS D is the largest gap between the observed and the predicted (mixture) first-success CDF over t = 1..B; it "
        "is descriptive only (discrete, censored, correlated observations; 1.36/sqrt(n) would be the 5% value for "
        "independent continuous data). The PIT values should be uniform on [0, 1] if the model holds; values near 1 "
        "mean later than predicted."),
        "",
        "PIT histograms (10 equal bins from 0 to 1, counts):",
        "",
    ]
    for fkey, s in data["combinations"].items():
        lines.append(f"- {fkey}: {s['pit']['histogram']}")
    lines += [
        "",
        "## Which prompts are discovered",
        "",
        ("| combination | prompts discovered in all / none / some runs | reading | median base p never / discovered | "
        "AUC (never-discovered p higher) |"),
        "|---|---|---|---|---|",
    ]
    for fkey, s in data["combinations"].items():
        r, pn = s["discovery_rate"], s["base_p_never_vs_discovered"]
        lines.append(
            f"| {fkey} | {r['all']} / {r['none']} / {r['some']} | {r['reading'] or 'one run per prompt'} | "
            f"{_g(pn['median_p_never'])} / {_g(pn['median_p_discovered'])} | {_g(pn['auc_never_higher'])} |"
        )
    lines += [
        "",
        "An AUC below 0.5 means never-discovered prompts had lower base p, as the waiting-time picture expects.",
        "",
        "## Drift before discovery",
        "",
        ("For each discovered prompt: its success probability at the last checkpoint before the first success, divided "
        "by its base p (only checkpoints after 0 are informative). The doc predicts ratios below 1 at k = 0 and not "
        "below 1 at k = 4."),
        "",
        ("| combination | informative | median ratio | geometric-mean ratio | fraction below 1 | "
        "never discovered: median final p / base p |"),
        "|---|---|---|---|---|---|",
    ]
    for fkey, s in data["combinations"].items():
        d = s["drift"]
        lines.append(
            f"| {fkey} | {d['n_informative']} | {_g(d['median_ratio'])} | {_g(d['geometric_mean_ratio'])} | "
            f"{_g(d['fraction_below_1'])} | {_g(d['never_discovered_final_over_base_median'])} "
            f"(n = {d['n_never_discovered']}) |"
        )
    lines += ["", "## Determined by the base, or by luck?", ""]
    comps = data["variance_components"]
    if not comps:
        lines.append("These results do not separate the seed streams, so no variance decomposition was made.")
    else:
        lines += [
            ("Variance of the final per-prompt capability gain (delta_cap, hard_train, primary N and c) across runs. "
            "V_train: base fixed, training seed varies. V_base: training seed fixed, base varies. "
            f"share = V_train / V_base; <= {SHARE_BASE:.2f} reads as '{BY_BASE}', >= {SHARE_LUCK:.2f} as "
            f"'{BY_LUCK}', otherwise '{MIXED}'."),
            "",
            ("| combination | per-prompt V_train | per-prompt V_base | share | verdict | run-level V_train | "
            "run-level V_base | share | verdict |"),
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for fkey, d in data["variance_decomposition"].items():
            pp, rl = d["per_prompt"], d["run_level"]
            lines.append(
                f"| {fkey} | {_g(pp['V_train'])} | {_g(pp['V_base'])} | {_g(pp['share'])} | {pp['verdict'] or 'n/a'} | "
                f"{_g(rl['V_train'])} | {_g(rl['V_base'])} | {_g(rl['share'])} | {rl['verdict'] or 'n/a'} |"
            )
        if len(comps) < 2:
            lines += ["", "Only one seed stream is available here; pass the other study to complete the comparison."]
    if figures:
        lines += ["", "## Figures", ""] + [f"![{f.stem}]({f.as_posix()})" for f in figures]
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- command line


def predict_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m capsplit.harness discovery predict")
    parser.add_argument("config", help="experiment config (JSON)")
    parser.add_argument("--out", help="output folder (default: results/published/seedstudy/predictions/<config name>)")
    parser.add_argument("--results-root", default="results", help="where the base cache lives (default: results)")
    parser.add_argument("--workers", type=int, default=1, help="processes for building bases (default: 1)")
    args = parser.parse_args(argv)
    out = args.out
    if out is None:
        from .config import load_config

        out = str(Path("results/published/seedstudy/predictions") / load_config(args.config).name)
    try:
        print(predict(args.config, out, args.results_root, workers=args.workers))
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


def analyze_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m capsplit.harness discovery analyze")
    parser.add_argument("results_dir", help="results folder of a run (holds index.json)")
    parser.add_argument("--predictions", required=True, help="predictions.json written by `discovery predict`")
    parser.add_argument("--out", help="output folder (default: <results_dir>/discovery)")
    parser.add_argument(
        "--compare", nargs=2, action="append", default=[], metavar=("RESULTS_DIR", "PREDICTIONS"),
        help="another seed study for the variance decomposition (repeatable)",
    )
    args = parser.parse_args(argv)
    out = args.out or str(Path(args.results_dir) / "discovery")
    try:
        print(analyze(args.results_dir, args.predictions, out, [tuple(c) for c in args.compare]))
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """`discovery predict ...` / `discovery analyze ...`."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("predict", "analyze"):
        print("usage: discovery {predict,analyze} ...", file=sys.stderr)
        return 2
    return (predict_main if argv[0] == "predict" else analyze_main)(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
