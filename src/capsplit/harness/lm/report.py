"""Generated report for the LM experiment (C6, docs/lm_design.md).

`build_report(out)` reads the stage outputs under `out` = `<results>/<name>/` (layout in
`stages.py`) and writes `out/report/`::

    report.md                 the human-readable report (warnings first)
    summary.json              headline numbers, warnings and inputs (machine-readable)
    tables/setup.csv          model, device and versions per stage manifest
    tables/data.csv           problems per split x difficulty
    tables/base_classes.csv   reachable / unreachable / ambiguous overall, per split, difficulty
    tables/split.csv          every arm x seed x step x group: gain, contributions, E_1, E_N, label
    tables/training.csv       per arm x seed: updates, episodes, tokens, N_train, timings
    tables/first_success.csv  per arm x seed x training problem: first success update, episodes
    tables/null_check.csv     base-vs-base null check per group
    tables/budget_curve.csv   contributions at every resolvable N
    tables/format_validity.csv valid / correct fraction of sampled completions per policy
    figures/*.png             contributions, first-success histogram, budget curves

Every number comes from a file; nothing is typed in. Missing optional inputs (a training log,
a sample file) are reported as missing rather than guessed.

Determinism: no timestamps or absolute paths; CSV/JSON floats use repr, Markdown a fixed
format; orderings come from the config (arms, seeds) or sorted keys; PNGs are written without
a Software/date tag. Same inputs give byte-identical files.
"""
from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .storage import read_jsonl

CLASSES = ("reachable", "unreachable", "ambiguous")
PARTS = CLASSES + ("total",)
TOP_GROUPS = ("all", "train", "heldout")
STAGES = ("data", "base", "measure")
# Categorical hues in fixed order (dataviz reference palette, light mode): reachable, unreachable,
# ambiguous; the total is drawn in neutral ink.
COLORS = {"reachable": "#2a78d6", "unreachable": "#eb6834", "ambiguous": "#1baf7a", "total": "#52514e"}
BAR = "!" * 78


class ReportError(RuntimeError):
    """A required stage output is missing; the message names the stage to run."""


# --------------------------------------------------------------------------- formatting


def _f(x: Any) -> float:
    if x is None or isinstance(x, bool):
        return math.nan
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def csv_num(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return str(x).lower()
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return "" if math.isnan(float(x)) else repr(float(x))
    return str(x)


def md_num(x: Any, digits: int = 4, sign: bool = True) -> str:
    v = _f(x)
    if math.isnan(v):
        return "n/a"
    return f"{v:+.{digits}f}" if sign else f"{v:.{digits}f}"


def md_ci(ci: dict | None, digits: int = 4) -> str:
    if not ci:
        return "n/a"
    return f"{md_num(ci.get('estimate'), digits)} [{md_num(ci.get('lo'), digits)}, {md_num(ci.get('hi'), digits)}]"


def md_table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> Path:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow([csv_num(c) for c in row])
    write_text(path, buf.getvalue())
    return path


def json_clean(x: Any) -> Any:
    """JSON-ready: nan/inf become null, numpy scalars become Python numbers, keys become strings."""
    if isinstance(x, dict):
        return {str(k): json_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return json_clean(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else v
    return x


def write_json_det(path: Path, data: Any) -> Path:
    write_text(path, json.dumps(json_clean(data), indent=2, sort_keys=True) + "\n")
    return path


# --------------------------------------------------------------------------- loading


def _read(path: Path, stage: str | None = None) -> Any:
    if not path.exists():
        if stage is None:
            return None
        raise ReportError(f"{path} does not exist; run the `{stage}` stage first")
    return json.loads(path.read_text())


def temp_str(t: float) -> str:
    return repr(float(t))


@dataclass
class Run:
    """Everything the report and the verdicts read, loaded once from `out`."""

    out: Path
    config: dict
    manifests: dict[str, dict]
    problems: list[dict]
    data_summary: dict
    classes: dict
    base_summary: dict
    results: dict
    null: dict
    curve: dict
    training: dict[tuple[str, int], dict | None] = field(default_factory=dict)
    timing: dict[tuple[str, int], dict | None] = field(default_factory=dict)
    train_manifests: dict[tuple[str, int], dict | None] = field(default_factory=dict)

    # -- config views
    @property
    def name(self) -> str:
        return str(self.config.get("name", self.out.name))

    @property
    def arms(self) -> list[dict]:
        return list(self.config["arms"])

    @property
    def seeds(self) -> list[int]:
        return [int(s) for s in self.config["seeds"]]

    def trainer(self, arm: str) -> str:
        return next(str(a["trainer"]) for a in self.arms if a["name"] == arm)

    @property
    def ref_temperature(self) -> float:
        return float(self.results.get("reference_temperature", self.config["sampling"]["reference_temperature"]))

    @property
    def final_step(self) -> int:
        return int(self.config["budget"])

    @property
    def n_problems(self) -> int:
        return len(self.problems)

    def ids(self, group: str) -> list[int]:
        """Problem ids of a measurement group (all / train / heldout / <split>:<difficulty>)."""
        if group == "all":
            return [int(r["id"]) for r in self.problems]
        if group in ("train", "heldout"):
            return [int(r["id"]) for r in self.problems if r["split"] == group]
        split, _, diff = group.partition(":")
        return [int(r["id"]) for r in self.problems
                if str(r["difficulty"]) == diff and split in ("all", r["split"])]

    def records(self, step: int | None = None) -> list[dict]:
        step = self.final_step if step is None else step
        return [r for r in self.results["records"] if int(r["step"]) == step]

    def record(self, arm: str, seed: int, group: str, step: int | None = None) -> dict | None:
        for r in self.records(step):
            if r["arm"] == arm and int(r["seed"]) == seed and r["group"] == group:
                return r
        return None

    # -- per-problem arrays from files
    def class_array(self) -> np.ndarray:
        by_id = self.classes["classes"]
        return np.asarray([by_id[str(i)] for i in range(self.n_problems)])

    def set_a(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.asarray(self.classes["successes_A"], dtype=np.int64),
                np.asarray(self.classes["trials_A"], dtype=np.int64))

    def sample_file(self, tag: str, set_name: str, temperature: float | None = None) -> Path:
        t = self.ref_temperature if temperature is None else temperature
        return self.out / "samples" / tag / f"{set_name}_T{temp_str(t)}.jsonl"

    def sample_rows(self, tag: str, set_name: str, n: int) -> dict[tuple[int, int], dict]:
        rows: dict[tuple[int, int], dict] = {}
        for row in read_jsonl(self.sample_file(tag, set_name)):
            key = (int(row["problem"]), int(row["index"]))
            if key[1] < n and key not in rows:
                rows[key] = row
        return rows

    def rates(self, tag: str, set_name: str, n: int) -> np.ndarray:
        """Per-problem success rate from a sample file (first n indices); every problem must be complete."""
        rows = self.sample_rows(tag, set_name, n)
        succ = np.zeros(self.n_problems)
        trials = np.zeros(self.n_problems)
        for (pid, _), row in rows.items():
            trials[pid] += 1
            succ[pid] += bool(row["correct"])
        if np.any(trials < n):
            short = np.flatnonzero(trials < n)[:5].tolist()
            raise ReportError(f"{self.sample_file(tag, set_name)} is incomplete (first short problems {short}); "
                              "re-run the stage that writes it")
        return succ / trials

    def base_rates(self) -> np.ndarray:
        return self.rates("base", "B", int(self.config["sampling"]["n_measure"]))

    def eval_rates(self, arm: str, seed: int, step: int | None = None) -> np.ndarray:
        step = self.final_step if step is None else step
        return self.rates(f"{arm}/seed{seed}/step{step}", "eval", int(self.config["sampling"]["n_trained"]))

    def measure_param(self, key: str) -> Any:
        return self.results.get(key, self.config["measure"].get(key))


def load_run(out: str | Path) -> Run:
    out = Path(out)
    manifests = {s: _read(out / s / f"manifest.{s}.json") for s in STAGES}
    present = {s: m for s, m in manifests.items() if m is not None}
    config = next((present[s]["config"] for s in ("measure", "base", "data") if s in present), None)
    if config is None:
        raise ReportError(f"no stage manifest under {out}; run the `data`, `base` and `measure` stages first")
    run = Run(
        out=out,
        config=config,
        manifests=present,
        problems=list(read_jsonl(out / "data" / "problems.jsonl")),
        data_summary=_read(out / "data" / "summary.json", "data"),
        classes=_read(out / "base" / "classes.json", "base"),
        base_summary=_read(out / "base" / "summary.json", "base"),
        results=_read(out / "measure" / "results.json", "measure"),
        null=_read(out / "measure" / "null_check.json", "measure"),
        curve=_read(out / "measure" / "budget_curve.json", "measure"),
    )
    if not run.problems:
        raise ReportError(f"{out / 'data' / 'problems.jsonl'} is missing or empty; run the `data` stage first")
    for arm in run.arms:
        for seed in run.seeds:
            d = out / "train" / arm["name"] / f"seed{seed}"
            run.training[(arm["name"], seed)] = _read(d / "training.json")
            run.timing[(arm["name"], seed)] = _read(d / "timing.json")
            run.train_manifests[(arm["name"], seed)] = _read(d / "manifest.train.json")
    return run


# --------------------------------------------------------------------------- bootstrap shared with verdicts


def contribution_bootstrap(successes_a: np.ndarray, trials_a: np.ndarray, gain: np.ndarray, member: np.ndarray,
                           target: str, n: int, alpha: float, n_boot: int, seed: Sequence[int]) -> dict:
    """Contribution sum(gain_i : class_i == target and member_i) / P, with a percentile interval.

    As `measure.bootstrap`: problems are resampled with replacement and each resampled problem's
    set-A count is redrawn from Binomial(trials_A, observed rate) and reclassified at budget N, so
    classification noise enters the interval. `member` is a fixed per-problem attribute (it travels
    with the problem). The stream is numpy default_rng(SeedSequence(seed)).
    """
    from .measure import CLASSES as M_CLASSES
    from .measure import classify

    k_a = np.asarray(successes_a, dtype=np.int64)
    m_a = np.asarray(trials_a, dtype=np.int64)
    gain = np.asarray(gain, dtype=float)
    member = np.asarray(member, dtype=bool)
    if target not in M_CLASSES:
        raise ValueError(f"unknown class {target!r}")
    size = len(gain)
    if size == 0:
        return {"estimate": math.nan, "lo": math.nan, "hi": math.nan, "n_problems": 0, "n_in_subset": 0,
                "mean_gain_in_subset": math.nan}
    cls = classify(k_a, m_a, n, alpha)
    sel = (cls == target) & member
    estimate = float(gain[sel].sum() / size)
    rng = np.random.default_rng(np.random.SeedSequence(list(seed)))
    idx = rng.integers(0, size, size=(n_boot, size))
    redrawn = rng.binomial(m_a[idx], k_a[idx] / m_a[idx])
    cls_b = classify(redrawn, m_a[idx], n, alpha)
    sel_b = (cls_b == target) & member[idx]
    samples = np.where(sel_b, gain[idx], 0.0).sum(axis=1) / size
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": estimate, "lo": float(lo), "hi": float(hi), "n_problems": size,
            "n_in_subset": int(sel.sum()),
            "mean_gain_in_subset": float(gain[sel].mean()) if sel.any() else math.nan}


# --------------------------------------------------------------------------- tables


def _modal_t(freq: dict | None) -> tuple[float, float]:
    if not freq:
        return math.nan, math.nan
    items = sorted(((float(k), float(v)) for k, v in freq.items()), key=lambda kv: (-kv[1], kv[0]))
    return items[0]


def split_rows(run: Run) -> list[list[Any]]:
    rows = []
    order = {a["name"]: i for i, a in enumerate(run.arms)}
    recs = sorted(run.results["records"], key=lambda r: (order.get(r["arm"], 99), int(r["seed"]), int(r["step"]),
                                                         _group_key(r["group"])))
    for r in recs:
        ex = r.get("extra", {})
        ci = ex.get("ci", {})
        tci = ex.get("temperature_ci", {})
        temp = ex.get("temperature", {})
        t1, f1 = _modal_t(temp.get("t_star_E_1"))
        tn, fn = _modal_t(temp.get("t_star_E_N"))
        row: list[Any] = [r["arm"], int(r["seed"]), int(r["step"]), r["group"], r["n_prompts"], r["j0"], r["j1"]]
        for part in PARTS:
            c = ci.get(part, {})
            row += [c.get("estimate"), c.get("lo"), c.get("hi")]
        for key, t, f in (("E_1", t1, f1), ("E_N", tn, fn)):
            c = tci.get(key, {})
            row += [temp.get(key), c.get("lo"), c.get("hi"), t, f]
        row += [r.get("label"), ex.get("label_en"), ex.get("beyond_reach"), ex.get("beyond_temperature"),
                len(ex.get("crossed", []))]
        rows.append(row)
    return rows


SPLIT_HEADER = (["arm", "seed", "step", "group", "n_problems", "j0", "j1"]
                + [f"{p}_{s}" for p in PARTS for s in ("est", "lo", "hi")]
                + [f"{k}_{s}" for k in ("E_1", "E_N") for s in ("est", "lo", "hi", "t_star", "t_star_freq")]
                + ["label", "label_en", "beyond_reach", "beyond_temperature", "n_crossed"])


def _group_key(group: str) -> tuple:
    return (TOP_GROUPS.index(group), "") if group in TOP_GROUPS else (len(TOP_GROUPS), group)


def class_rows(run: Run) -> list[list[Any]]:
    s = run.base_summary
    rows: list[list[Any]] = []
    overall = s.get("classes", {})
    rows.append(["overall", "all"] + [overall.get(c, 0) for c in CLASSES] + [sum(overall.get(c, 0) for c in CLASSES)])
    for scope, key in (("split", "classes_by_split"), ("difficulty", "classes_by_difficulty"),
                       ("split:difficulty", "classes_by_split_difficulty")):
        table: dict[str, dict[str, int]] = {}
        for k, v in s.get(key, {}).items():
            head, _, cls = k.rpartition("/")
            table.setdefault(head, {})[cls] = int(v)
        for head in sorted(table):
            counts = table[head]
            rows.append([scope, head.replace("/", ":")] + [counts.get(c, 0) for c in CLASSES]
                        + [sum(counts.values())])
    return rows


def _first_success_info(run: Run, arm: str, seed: int) -> list[dict]:
    log = run.training.get((arm, seed))
    if not log:
        return []
    cls = run.classes["classes"]
    split = {int(r["id"]): r["split"] for r in run.problems}
    diff = {int(r["id"]): r["difficulty"] for r in run.problems}
    out = []
    for key in sorted(log.get("prompts", {}), key=int):
        e = log["prompts"][key]
        pid = int(key)
        out.append({"problem": pid, "class": cls.get(str(pid)), "split": split.get(pid), "difficulty": diff.get(pid),
                    "first_success_update": e.get("first_success_update"),
                    "episodes_sampled": e.get("episodes_sampled"), "tokens_sampled": e.get("tokens_sampled")})
    return out


def _n_train(run: Run, arm: str, seed: int) -> dict:
    for rec in run.curve.get("records", []):
        if rec["arm"] == arm and int(rec["seed"]) == seed and int(rec["step"]) == run.final_step \
                and rec["group"] == "all":
            return dict(rec.get("n_train") or {})
    return {}


def training_rows(run: Run) -> list[list[Any]]:
    rows = []
    for arm in run.arms:
        for seed in run.seeds:
            name = arm["name"]
            log = run.training.get((name, seed))
            timing = run.timing.get((name, seed)) or {}
            totals = (log or {}).get("totals", {})
            info = _first_success_info(run, name, seed)
            sampled = [i for i in info if (i["episodes_sampled"] or 0) > 0]
            ever = [i for i in info if i["first_success_update"] is not None]
            nt = _n_train(run, name, seed)
            rows.append([name, arm["trainer"], seed, "present" if log else "MISSING", (log or {}).get("updates"),
                         totals.get("episodes_sampled"), totals.get("tokens_sampled"), len(info), len(sampled),
                         len(ever), nt.get("value"), timing.get("wall_seconds"), timing.get("seconds_per_update")])
    return rows


TRAINING_HEADER = ["arm", "trainer", "seed", "training_log", "updates", "episodes_sampled", "tokens_sampled",
                   "problems_logged", "problems_sampled", "problems_ever_correct", "n_train", "wall_seconds",
                   "seconds_per_update"]


def validity_rows(run: Run) -> list[list[Any]]:
    """Valid and correct fractions of sampled completions, overall and per split, per policy."""
    s = run.config["sampling"]
    split = {int(r["id"]): r["split"] for r in run.problems}
    policies = [("base", "base", "B", int(s["n_measure"]))]
    for arm in run.arms:
        for seed in run.seeds:
            policies.append((arm["name"], f"{arm['name']}/seed{seed}/step{run.final_step}", "eval",
                             int(s["n_trained"])))
    rows = []
    for arm_name, tag, set_name, n in policies:
        samples = run.sample_rows(tag, set_name, n)
        for scope in ("all", "train", "heldout"):
            sel = [r for (pid, _), r in sorted(samples.items()) if scope == "all" or split.get(pid) == scope]
            k = len(sel)
            valid = sum(bool(r["valid"]) for r in sel)
            correct = sum(bool(r["correct"]) for r in sel)
            rows.append([arm_name, tag, set_name, scope, k, valid / k if k else None, correct / k if k else None,
                         correct / valid if valid else None])
    return rows


VALIDITY_HEADER = ["arm", "tag", "set", "split", "n_samples", "valid_fraction", "correct_fraction",
                   "correct_given_valid"]


# --------------------------------------------------------------------------- figures


def _plt() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is in the dev environment
        return None
    plt.rcParams.update({"svg.hashsalt": "capsplit", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#999999", "grid.color": "#e6e6e6"})
    return plt


def _save(plt: Any, fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return path


def figure_contributions(run: Run, fig_dir: Path, plt: Any) -> Path:
    runs = [(a["name"], s) for a in run.arms for s in run.seeds]
    fig, axes = plt.subplots(1, len(TOP_GROUPS), figsize=(4.2 * len(TOP_GROUPS), 0.45 * len(runs) + 1.6),
                             sharex=True, sharey=True, squeeze=False)
    for ax, group in zip(axes[0], TOP_GROUPS):
        for i, (arm, seed) in enumerate(runs):
            rec = run.record(arm, seed, group)
            if rec is None:
                continue
            ci = rec["extra"]["ci"]
            for j, part in enumerate(PARTS):
                y = i + (j - 1.5) * 0.18
                c = ci[part]
                ax.errorbar([c["estimate"]], [y], xerr=[[c["estimate"] - c["lo"]], [c["hi"] - c["estimate"]]],
                            fmt="o", ms=4, lw=1.5, color=COLORS[part], label=part if i == 0 else None)
        ax.axvline(0, color="#999999", lw=1)
        ax.set_yticks(range(len(runs)))
        ax.set_yticklabels([f"{a} s{s}" for a, s in runs])
        ax.invert_yaxis()
        ax.set_title(f"group {group}")
        ax.set_xlabel("contribution to mean gain")
        ax.grid(axis="x")
    axes[0][0].legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=4, frameon=False)
    return _save(plt, fig, fig_dir / "contributions.png")


def figure_first_success(run: Run, fig_dir: Path, plt: Any) -> Path | None:
    arms = [a["name"] for a in run.arms if any(_first_success_info(run, a["name"], s) for s in run.seeds)]
    if not arms:
        return None
    fig, axes = plt.subplots(len(arms), 1, figsize=(6.4, 1.9 * len(arms) + 0.6), squeeze=False)
    budget = max(run.final_step, 1)
    bins = np.linspace(0, budget, min(budget, 25) + 1)
    for ax, arm in zip(axes[:, 0], arms):
        times, never, logged = [], 0, 0
        for seed in run.seeds:
            for i in _first_success_info(run, arm, seed):
                logged += 1
                if i["first_success_update"] is None:
                    never += 1
                else:
                    times.append(int(i["first_success_update"]))
        ax.hist(times, bins=bins, color=COLORS["reachable"], edgecolor="white", linewidth=1)
        ax.set_title(f"{arm}: first correct training sample (pooled over seeds); "
                     f"never correct: {never} of {logged} problem-runs", fontsize=9)
        ax.set_xlabel("update")
        ax.set_ylabel("problems")
    fig.tight_layout()
    return _save(plt, fig, fig_dir / "first_success.png")


def figure_budget_curves(run: Run, fig_dir: Path, plt: Any) -> list[Path]:
    paths = []
    n_max = run.curve.get("n_max_resolvable")
    arms = [a["name"] for a in run.arms]
    for group in TOP_GROUPS:
        recs = [r for r in run.curve.get("records", []) if r["group"] == group and int(r["step"]) == run.final_step]
        if not recs:
            continue
        fig, axes = plt.subplots(1, len(arms), figsize=(3.6 * len(arms), 3.0), sharey=True, squeeze=False)
        for ax, arm in zip(axes[0], arms):
            for k, rec in enumerate(r for r in recs if r["arm"] == arm):
                c = rec["curve"]
                style = ["-", "--", ":", "-."][k % 4]
                for part in PARTS:
                    ax.plot(c["n"], c[part], style, color=COLORS[part], lw=2,
                            label=part if k == 0 else None)
                nt = _f((rec.get("n_train") or {}).get("value"))
                if not math.isnan(nt):
                    ax.axvline(nt, color="#0b0b0b", lw=1, ls=style, label="N_train" if k == 0 else None)
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
            ax.axhline(0, color="#999999", lw=0.8)
            ax.set_title(f"{arm} ({group})")
            ax.set_xlabel("budget N (log)")
            ax.grid(True)
        axes[0][0].set_ylabel("contribution")
        axes[0][0].legend(fontsize=7, frameon=False)
        fig.suptitle(f"curves end at the largest resolvable N = {n_max}; line style = seed", fontsize=9)
        fig.tight_layout()
        paths.append(_save(plt, fig, fig_dir / f"budget_curve_{group}.png"))
    return paths


# --------------------------------------------------------------------------- warnings


def collect_warnings(run: Run) -> list[str]:
    warnings = []
    for rec in run.null.get("records", []):
        if not rec.get("passed", False):
            t = rec["total"]
            warnings.append(f"NULL CHECK FAILED for group `{rec['group']}`: a fresh base sample scored as a "
                            f"'trained' policy gives a total gain interval [{md_num(t['lo'])}, {md_num(t['hi'])}] "
                            "that excludes 0. The measurement pipeline reports gains where there are none; "
                            "treat every gain below with suspicion.")
    overall = run.base_summary.get("classes", {})
    empty = [c for c in CLASSES if overall.get(c, 0) == 0]
    if empty:
        warnings.append(f"EMPTY BASE CLASS(ES) {empty} at N = {run.classes.get('n')}: the split cannot attribute "
                        "any gain to an empty class.")
    missing = [f"{a} seed {s}" for (a, s), log in run.training.items()
               if log is None and run.trainer(a) != "lm_none"]
    if missing:
        warnings.append(f"Training logs missing for {missing}; their training statistics are absent.")
    expected = {(a["name"], s) for a in run.arms for s in run.seeds}
    measured = {(r["arm"], int(r["seed"])) for r in run.records()}
    unmeasured = sorted(expected - measured)
    if unmeasured:
        warnings.append(f"No measurement at the final step for {unmeasured}.")
    dirty = sorted(s for s, m in run.manifests.items() if m.get("dirty"))
    if dirty:
        warnings.append(f"Stages {dirty} ran from a working tree with uncommitted changes (manifest `dirty`).")
    return warnings


# --------------------------------------------------------------------------- report


def _setup_rows(run: Run) -> list[list[Any]]:
    rows = []
    entries: list[tuple[str, dict]] = [(s, run.manifests[s]) for s in STAGES if s in run.manifests]
    entries += [(f"train {a} seed{s}", m) for (a, s), m in sorted(run.train_manifests.items(),
                                                                  key=lambda kv: (kv[0][0], kv[0][1])) if m]
    for label, m in entries:
        model = m.get("model") or {}
        rows.append([label, model.get("id"), model.get("revision"), m.get("device"), m.get("commit"), m.get("dirty"),
                     m.get("python"), m.get("torch"), m.get("transformers"), json.dumps(m.get("seeds"), sort_keys=True)])
    return rows


SETUP_HEADER = ["stage", "model_id", "model_revision", "device", "commit", "dirty", "python", "torch",
                "transformers", "seeds"]


def _md_cell(x: Any) -> str:
    return "n/a" if x is None else str(x).replace("|", "/")


STAGES_TIMED = ("speedtest", "data", "base", "train", "measure", "report", "verdicts")


def stage_durations(out: Path) -> list[list[str]]:
    """[stage, invocations, total seconds, total hours] per stage, or "not recorded"."""
    rows = []
    for stage in STAGES_TIMED:
        entries = list(read_jsonl(out / "timing" / f"{stage}.jsonl"))
        if not entries:
            rows.append([stage, "0", "not recorded", "not recorded"])
            continue
        total = sum(float(e["wall_seconds"]) for e in entries)
        rows.append([stage, str(len(entries)), f"{total:.0f}", f"{total / 3600:.2f}"])
    return rows


def build_report(out_dir: str | Path) -> Path:
    """Write `<out_dir>/report/` from the stage outputs; returns the path of report.md."""
    run = load_run(out_dir)
    rep = run.out / "report"
    tables = rep / "tables"
    figs = rep / "figures"
    cfg = run.config
    s, m = cfg["sampling"], cfg["measure"]
    warnings = collect_warnings(run)

    setup = _setup_rows(run)
    write_csv(tables / "setup.csv", SETUP_HEADER, setup)
    data_rows = [[k.split("/")[0], k.split("/")[1], v] for k, v in
                 sorted(run.data_summary.get("by_split_difficulty", {}).items())]
    write_csv(tables / "data.csv", ["split", "difficulty", "n_problems"], data_rows)
    classes = class_rows(run)
    write_csv(tables / "base_classes.csv", ["scope", "key"] + list(CLASSES) + ["total"], classes)
    splits = split_rows(run)
    write_csv(tables / "split.csv", SPLIT_HEADER, splits)
    training = training_rows(run)
    write_csv(tables / "training.csv", TRAINING_HEADER, training)
    first = [[a["name"], seed, i["problem"], i["class"], i["split"], i["difficulty"], i["first_success_update"],
              i["episodes_sampled"], i["tokens_sampled"]]
             for a in run.arms for seed in run.seeds for i in _first_success_info(run, a["name"], seed)]
    write_csv(tables / "first_success.csv", ["arm", "seed", "problem", "class", "split", "difficulty",
                                             "first_success_update", "episodes_sampled", "tokens_sampled"], first)
    nulls = [[r["group"], r["total"]["estimate"], r["total"]["lo"], r["total"]["hi"], r["passed"]]
             for r in sorted(run.null.get("records", []), key=lambda r: _group_key(r["group"]))]
    write_csv(tables / "null_check.csv", ["group", "total_est", "total_lo", "total_hi", "passed"], nulls)
    curve_rows = []
    for rec in run.curve.get("records", []):
        c = rec["curve"]
        for k, n in enumerate(c["n"]):
            counts = c["counts"][k]
            curve_rows.append([rec["arm"], rec["seed"], rec["step"], rec["group"], n]
                              + [c[p][k] for p in PARTS] + [counts.get(cl) for cl in CLASSES]
                              + [(rec.get("n_train") or {}).get("value")])
    write_csv(tables / "budget_curve.csv", ["arm", "seed", "step", "group", "N"] + list(PARTS)
              + [f"n_{c}" for c in CLASSES] + ["n_train"], curve_rows)
    validity = validity_rows(run)
    write_csv(tables / "format_validity.csv", VALIDITY_HEADER, validity)

    figure_paths: list[Path] = []
    plt = _plt()
    if plt is not None:
        figure_paths.append(figure_contributions(run, figs, plt))
        fs = figure_first_success(run, figs, plt)
        if fs is not None:
            figure_paths.append(fs)
        figure_paths += figure_budget_curves(run, figs, plt)

    # ---------------------------------------------------------------- markdown
    md: list[str] = [f"# LM experiment report: {run.name}", ""]
    md.append("Generated by `capsplit.harness.lm.report` from the files under this experiment folder; "
              "no number below is typed by hand.")
    md.append("")
    if warnings:
        md += ["## WARNINGS", "", "```", BAR]
        md += [f"WARNING: {w}" for w in warnings]
        md += [BAR, "```", ""]
    else:
        md += [("No warnings: the null check passed for every group, no base class is empty, and every run has "
               "its training log and final measurement."), ""]

    md += ["## Setup", ""]
    md += md_table(["stage", "model", "revision", "device", "commit", "dirty", "python", "torch", "transformers"],
                   [[_md_cell(c) for c in r[:9]] for r in setup])
    loaded = sorted({(r[1], r[2]) for r in setup if r[1]})
    md += ["", "Model loaded, as recorded by the manifests: "
           + (", ".join(f"`{i}` @ `{r}`" for i, r in loaded) if loaded else
              "none recorded (every sample file already existed when the stages ran); the config's model is below")
           + "."]
    md += ["", (f"- Budget N = {m['n']}, alpha = {m['alpha']}, bootstrap {m['n_boot']} resamples "
           f"(seed {m['boot_seed']}), {run.results.get('n_splits')} temperature half-splits."),
           (f"- Samples per problem: set A {s['n_classify']} (classification only), set B {s['n_measure']} "
           f"(base rate), trained {s['n_trained']}, temperature sweep {s['n_temperature']} at "
           f"T in {list(s['temperatures'])}; reference T = {s['reference_temperature']}."),
           f"- Training budget {cfg['budget']} updates, checkpoints {list(cfg['checkpoints'])}, seeds {run.seeds}.",
           f"- Config model: `{cfg['model']['id']}` @ `{cfg['model']['revision']}`"
           + (f" (fallback `{cfg['model'].get('fallback_id')}`)" if cfg["model"].get("fallback_id") else "") + "."]
    md += ["", "Arms:", ""]
    md += md_table(["arm", "trainer", "params"],
                   [[a["name"], a["trainer"], f"`{json.dumps(a.get('params', {}), sort_keys=True)}`"] for a in run.arms])

    md += ["", "## Data", ""]
    md += md_table(["split", "difficulty", "problems"], data_rows)
    md += ["", f"Total problems: {run.data_summary.get('n_problems')}.", ""]

    md += ["## Base classification", "",
           (f"From set A at T = {run.classes.get('temperature')}: reachable iff the {100 * (1 - m['alpha']):g}% "
           f"Clopper-Pearson lower bound >= 1/N, unreachable iff the upper bound < 1/N (N = {run.classes.get('n')})."),
           ""]
    md += md_table(["scope", "key"] + list(CLASSES) + ["total"], classes)
    rate = run.base_summary.get("success_rate_B", {})
    md += ["", f"Base success rate (set B): all {md_num(rate.get('all'), sign=False)}; by split "
           + ", ".join(f"{k} {md_num(v, sign=False)}" for k, v in sorted(rate.get("by_split", {}).items())) + "."]
    if any(overall == 0 for overall in classes[0][2:5]):
        md += ["", f"**WARNING: base class(es) {[c for c, k in zip(CLASSES, classes[0][2:5]) if k == 0]} are EMPTY.**"]
    md.append("")

    md += ["## Split of the gain (final checkpoint)", "",
           ("Per arm and seed: total gain = mean(p̂1 - p̂0) and its three class contributions (each sums its "
           "class's gains and divides by all problems in the group), with bootstrap intervals "
           f"(level {1 - m['alpha']:g}; problems resampled, set-A counts redrawn and reclassified). "
           "E_1 / E_N: trained minus the base at the best temperature T* (chosen on the other half of a "
           "random split; T* is the modal choice and its frequency). Per-difficulty groups are in "
           "`tables/split.csv`."), ""]
    head = ["arm", "seed", "group", "total", "reachable", "unreachable", "ambiguous", "E_1 [lo, hi]", "T*(E_1)",
            "E_N [lo, hi]", "label"]
    body = []
    for row in splits:
        if row[2] != run.final_step or row[3] not in TOP_GROUPS:
            continue
        cis = [f"{md_num(row[7 + 3 * j])} [{md_num(row[8 + 3 * j])}, {md_num(row[9 + 3 * j])}]" for j in (3, 0, 1, 2)]
        e1 = f"{md_num(row[19])} [{md_num(row[20])}, {md_num(row[21])}]"
        en = f"{md_num(row[24])} [{md_num(row[25])}, {md_num(row[26])}]"
        tstar = "n/a" if math.isnan(_f(row[22])) else f"{row[22]:g} ({100 * row[23]:.0f}%)"
        body.append([row[0], row[1], row[3], *cis, e1, tstar, en, row[29]])
    md += md_table(head, body)
    md += ["", "![contributions](figures/contributions.png)", ""] if plt is not None else [""]

    md += ["## Format validity", "",
           ("Fraction of sampled completions containing an integer (valid) and the fraction correct, so that "
           "format effects stay separate from arithmetic ability (DECISIONS L6)."), ""]
    md += md_table(["arm", "tag", "split", "samples", "valid", "correct", "correct given valid"],
                   [[r[0], r[1], r[3], r[4], md_num(r[5], sign=False), md_num(r[6], sign=False),
                     md_num(r[7], sign=False)] for r in validity])
    md.append("")

    md += ["## Stage durations", "",
           ("Wall-clock time of each stage, measured by the command itself (`timing/<stage>.jsonl`, one line per "
           "invocation; a resumed stage has several). A stage with no file was run by a version that did not record "
           "its duration."), ""]
    md += md_table(["stage", "invocations", "total wall s", "total wall h"], stage_durations(run.out))
    md.append("")

    md += ["## Training", "",
           ("Arms are matched on optimizer updates, not on samples (H20): the episodes and tokens each arm sampled "
           "are listed. N_train = mean episodes per training problem the arm sampled (H51). Timings are measured "
           "wall-clock times from `timing.json`."), ""]
    md += md_table(["arm", "seed", "log", "updates", "episodes", "tokens", "problems sampled", "ever correct",
                    "N_train", "wall s", "s / update"],
                   [[r[0], r[2], r[3], _md_cell(r[4]), _md_cell(r[5]), _md_cell(r[6]), r[8], r[9],
                     md_num(r[10], 1, sign=False), md_num(r[11], 1, sign=False), md_num(r[12], 2, sign=False)]
                    for r in training])
    if plt is not None and (figs / "first_success.png").exists():
        md += ["", ("First update in which training sampled a correct answer, per training problem "
               "(`tables/first_success.csv`):"), "", "![first success](figures/first_success.png)"]
    md.append("")

    md += ["## Null check", "",
           ("The same pipeline with a fresh base sample (set `null`) in place of the trained policy. "
           "It passes iff the total gain's interval contains 0."), ""]
    md += md_table(["group", "total [lo, hi]", "result"],
                   [[r[0], f"{md_num(r[1])} [{md_num(r[2])}, {md_num(r[3])}]", "PASS" if r[4] else "**FAILED**"]
                    for r in nulls])
    md.append("")

    n_max = run.curve.get("n_max_resolvable")
    md += ["## Budget curve", "",
           (f"Contributions as a function of N (log axis), reclassifying at each N from the same counts. The curve "
           f"is truncated at the largest resolvable N = {n_max}: with {s['n_classify']} set-A samples per problem "
           f"no problem can be classified unreachable beyond it (the Clopper-Pearson upper bound for 0 successes "
           "exceeds 1/N). The vertical line marks N_train from the training log."), ""]
    md += [f"![budget curve {g}](figures/budget_curve_{g}.png)" for g in TOP_GROUPS
           if (figs / f"budget_curve_{g}.png").exists()]
    md += ["", "Full values: `tables/budget_curve.csv`.", ""]

    md += ["## What this cannot show", "",
           ("- **One small model.** Every result is for one ~0.5B instruct checkpoint at one revision; nothing "
           "here says how larger or differently trained models behave."),
           ("- **Generated problems.** The problems are synthetic arithmetic; reachability on them need not "
           "transfer to natural tasks."),
           ("- **Sampled, not exact.** Success probabilities and the reachable / unreachable classes are "
           f"estimated from finitely many samples; 'unreachable' means 'upper bound below 1/N at "
           f"{100 * (1 - m['alpha']):g}% confidence', and nothing beyond N = {n_max} can be resolved."),
           ("- **LoRA only.** Training updates a low-rank adapter on a few projection matrices; full fine-tuning "
           "could behave differently."),
           ("- **Finite budget and few seeds.** One update budget and the listed seeds; slower effects or rare "
           "runs are not covered."), ""]

    md += ["## Files", ""]
    md += [f"- `{p.relative_to(rep).as_posix()}`" for p in sorted(tables.glob("*.csv"))]
    md += [f"- `{p.relative_to(rep).as_posix()}`" for p in sorted(figure_paths)]
    md.append("")
    report_path = rep / "report.md"
    write_text(report_path, "\n".join(md))

    summary = {
        "name": run.name,
        "warnings": warnings,
        "null_check": {r["group"]: {"total": r["total"], "passed": r["passed"]} for r in run.null.get("records", [])},
        "null_check_passed": all(r.get("passed", False) for r in run.null.get("records", [])),
        "base_classes": run.base_summary.get("classes"),
        "measure": {k: run.results.get(k) for k in ("n", "alpha", "n_boot", "boot_seed", "n_splits",
                                                     "reference_temperature", "steps")},
        "n_max_resolvable": n_max,
        "split": [dict(zip(SPLIT_HEADER, r)) for r in splits if r[2] == run.final_step and r[3] in TOP_GROUPS],
        "training": [dict(zip(TRAINING_HEADER, r)) for r in training],
        "format_validity": [dict(zip(VALIDITY_HEADER, r)) for r in validity],
        "model": {s_: (m_.get("model") or None) for s_, m_ in run.manifests.items()},
        "config": cfg,
        "files": sorted([p.relative_to(rep).as_posix() for p in tables.glob("*.csv")]
                        + [p.relative_to(rep).as_posix() for p in figure_paths]),
    }
    write_json_det(rep / "summary.json", summary)
    return report_path
