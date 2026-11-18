"""Report generator: tables, validity checks, figures and a JSON summary for one results folder.

`build_report(results_dir)` reads `<results_dir>/index.json` and every run's metrics file (the
layout is documented in `runner.py`) and writes, under `<results_dir>/report/` by default::

    report.md                  the human-readable report
    summary.json               headline numbers, checks and sensitivity flips (machine-readable)
    tables/main.csv            primary (N, c), final checkpoint: seed mean/min/max/bootstrap CI
    tables/labels.csv          label counts over seeds (primary (N, c), final checkpoint)
    tables/trajectory.csv      seed mean/min/max of each metric at every checkpoint
    tables/frontier.csv        trained pass@N vs the best tempered-base pass@N
    tables/sensitivity.csv     headline conclusions at every (N, c), and flips vs the primary
    tables/sensitivity_flips.csv
    tables/checks.csv          validity checks (PASS / WARNING)
    figures/*.png              heatmap, training curves, tempering frontier (needs matplotlib)

Primary (N, c): `measure.primary` in the config ({"n": N, "c": c} or [N, c]) if present and
measured; otherwise the largest N and c = 1 (or the measured c closest to 1, smaller on a tie).

Intervals: the seed mean's 95% bootstrap percentile interval over seeds (10,000 resamples,
numpy default_rng(0), a fresh generator per cell). With an exact backend the numbers carry no
sampling error, so intervals describe seed-to-seed spread only.

Determinism: no timestamps or absolute paths; floats are written with repr (CSV/JSON) or a fixed
format (Markdown); every ordering comes from index.json. Same inputs give byte-identical
CSV/MD/JSON files.
"""
from __future__ import annotations

import csv
import io
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

HIGHLIGHT = ("hard_heldout", "hard_train", "easy_train", "easy_heldout")
METRICS = ("delta_rel", "delta_cap", "delta_total", "j1", "E_both", "E_per_step", "E_sequence")
TRAJ_METRICS = ("delta_rel", "delta_cap", "delta_total", "j1", "E_both")
N_BOOT = 10_000
BOOT_SEED = 0
PROP1_TOL = 1e-9
CAP_SIGN_TOL = 1e-12  # |seed-mean delta_cap| <= this counts as zero
E_SIGN_TOL = 1e-9  # same threshold as "beyond temperature" (H12)
EXACT_BACKENDS = ("exact", "j_only")
_MAX_EXAMPLES = 5

RecKey = tuple  # (arm, fkey, seed, step, group, n, c)


# --------------------------------------------------------------------------- small helpers


def _f(value: Any) -> float:
    """A float, or nan for anything missing or not numeric."""
    if value is None or isinstance(value, bool):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _csv_num(x: Any) -> str:
    if isinstance(x, bool) or not isinstance(x, (int, float, np.floating, np.integer)):
        return "" if x is None else str(x)
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    x = float(x)
    return "" if math.isnan(x) else repr(x)


def _md_num(x: float, digits: int = 4) -> str:
    return "n/a" if x is None or math.isnan(x) else f"{x:+.{digits}f}"


def _json_num(x: float) -> float | None:
    return None if x is None or math.isnan(x) else float(x)


def _c_str(c: float) -> str:
    return f"{c:g}"


def _sign(x: float, tol: float) -> str:
    if math.isnan(x):
        return "n/a"
    return "+" if x > tol else "-" if x < -tol else "0"


def _fkey(factors: dict) -> str:
    if not factors:
        return "(none)"
    return ", ".join(f"{k}={json.dumps(factors[k], sort_keys=True)}" for k in sorted(factors))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow([_csv_num(v) if not isinstance(v, str) else v for v in row])
    _write(path, buffer.getvalue())


def bootstrap_ci(values: Sequence[float], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean of `values` (resampling with replacement)."""
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


@dataclass(frozen=True)
class Stat:
    n_seeds: int
    mean: float
    min: float
    max: float
    ci_low: float = math.nan
    ci_high: float = math.nan

    def to_json(self) -> dict:
        return {
            "n_seeds": self.n_seeds,
            "mean": _json_num(self.mean),
            "min": _json_num(self.min),
            "max": _json_num(self.max),
            "ci_low": _json_num(self.ci_low),
            "ci_high": _json_num(self.ci_high),
        }


def summarize(values: Sequence[float], ci: bool = True) -> Stat:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return Stat(0, math.nan, math.nan, math.nan)
    lo, hi = bootstrap_ci(vals) if ci else (math.nan, math.nan)
    return Stat(len(vals), float(np.mean(vals)), float(min(vals)), float(max(vals)), lo, hi)


def metric(rec: dict | None, name: str) -> float:
    """One metric from a GroupMetrics record; nan if absent."""
    if rec is None:
        return math.nan
    if name.startswith("E_"):
        excess = rec.get("excess") or {}
        return _f(excess.get(name[2:])) if isinstance(excess, dict) else math.nan
    if name == "delta_total" and "delta_total" not in rec:
        return _f(rec.get("j1")) - _f(rec.get("j0"))
    return _f(rec.get(name))


def pass_and_frontier(rec: dict | None) -> tuple[float, float, str]:
    """(mean pass@N of the trained policy, best tempered-base pass@N, source) for a record."""
    if rec is None:
        return math.nan, math.nan, "missing"
    n = int(rec.get("n", 0))
    extra = rec.get("extra") or {}
    e_both = metric(rec, "E_both")
    frontier = extra.get("frontier") if isinstance(extra, dict) else None
    j1_list = extra.get("j1") if isinstance(extra, dict) else None
    if isinstance(frontier, dict) and frontier:
        best = max(_f(v.get("best_pass")) if isinstance(v, dict) else math.nan for v in frontier.values())
        if not math.isnan(best):
            trained = best + e_both if not math.isnan(e_both) else math.nan
            if isinstance(j1_list, list) and j1_list:
                trained = float(np.mean([1.0 - (1.0 - _f(j)) ** n for j in j1_list]))
            return trained, best, "extra.frontier"
    if isinstance(j1_list, list) and j1_list:
        trained = float(np.mean([1.0 - (1.0 - _f(j)) ** n for j in j1_list]))
        return trained, trained - e_both, "extra.j1 - E_N"
    trained = 1.0 - (1.0 - metric(rec, "j1")) ** n
    return trained, trained - e_both, "approx pass@N(mean J1) - E_N"


# --------------------------------------------------------------------------- loading


@dataclass
class Experiment:
    name: str
    backend: str
    factors: dict
    arms: list  # [(name, trainer)]
    seeds: list
    checkpoints: list
    ns: list
    cs: list
    fkeys: list  # factor-combination labels in index order
    combos: dict  # fkey -> {factor: value}
    groups: list
    records: dict  # RecKey -> record
    missing: list = field(default_factory=list)  # metrics files that do not exist
    primary: tuple[int, float] = (0, 1.0)
    primary_rule: str = ""
    primary_note: str = ""

    @property
    def final(self) -> int:
        return max(self.checkpoints) if self.checkpoints else 0

    def trainer(self, arm: str) -> str:
        return dict(self.arms).get(arm, "?")

    def rec(self, arm: str, fkey: str, seed: int, step: int, group: str, n: int, c: float) -> dict | None:
        return self.records.get((arm, fkey, seed, step, group, n, c))

    def values(self, arm: str, fkey: str, step: int, group: str, n: int, c: float, name: str) -> list[float]:
        return [metric(self.rec(arm, fkey, s, step, group, n, c), name) for s in self.seeds]

    def cell_records(self, arm: str, fkey: str, step: int, group: str, n: int, c: float) -> list[dict]:
        out = [self.rec(arm, fkey, s, step, group, n, c) for s in self.seeds]
        return [r for r in out if r is not None]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _choose_primary(measure: dict, ns: list, cs: list) -> tuple[tuple[int, float], str, str]:
    rule = "config measure.primary"
    note = ""
    primary = measure.get("primary") if isinstance(measure, dict) else None
    if primary is not None:
        n_p, c_p = (primary.get("n"), primary.get("c")) if isinstance(primary, dict) else tuple(primary)[:2]
        if int(n_p) in ns and float(c_p) in cs:
            return (int(n_p), float(c_p)), rule, note
        note = f"measure.primary {primary!r} was not measured; fell back to the default rule"
    n = max(ns) if ns else 0
    c = 1.0 if 1.0 in cs else (min(cs, key=lambda x: (abs(x - 1.0), x)) if cs else 1.0)
    rule = "default: largest N and c = 1" + ("" if c == 1.0 else " (c = 1 not measured; closest c used)")
    return (n, c), rule, note


def load_experiment(results_dir: Path) -> Experiment:
    index_path = results_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"{index_path} not found: is {results_dir} a results folder (<out>/<name>)?")
    index = _read_json(index_path)
    config: dict = {}
    config_path = results_dir / str(index.get("config", "config.json"))
    if config_path.exists():
        config = _read_json(config_path)
    measure = dict(index.get("measure") or {})
    for key, value in (config.get("measure") or {}).items():
        measure.setdefault(key, value)

    records: dict = {}
    fkeys: list = []
    combos: dict = {}
    groups: list = []
    missing: list = []
    seen_ns: list = []
    seen_cs: list = []
    for run in index.get("runs", []):
        factors = dict(run.get("factors") or {})
        fkey = _fkey(factors)
        if fkey not in combos:
            combos[fkey] = factors
            fkeys.append(fkey)
        metrics_rel = (run.get("files") or {}).get("metrics")
        path = results_dir / str(metrics_rel)
        if not metrics_rel or not path.exists():
            missing.append(str(metrics_rel))
            continue
        data = _read_json(path)
        for rec in data.get("records", []):
            group = str(rec.get("group", "all"))
            n, c = int(rec.get("n", 0)), float(rec.get("c", 1.0))
            if group not in groups:
                groups.append(group)
            if n not in seen_ns:
                seen_ns.append(n)
            if c not in seen_cs:
                seen_cs.append(c)
            key = (run.get("arm"), fkey, int(run.get("seed", 0)), int(rec.get("step", -1)), group, n, c)
            records[key] = rec

    ns = [int(n) for n in measure.get("ns", [])] or sorted(seen_ns)
    cs = [float(c) for c in measure.get("cs", [])] or sorted(seen_cs)
    primary, rule, note = _choose_primary(measure, ns, cs)
    arms = [(str(a.get("name")), str(a.get("trainer"))) for a in index.get("arms", [])]
    known = {a for a, _ in arms}
    for run in index.get("runs", []):  # arms present in runs but not in the arm list
        if run.get("arm") not in known:
            arms.append((str(run.get("arm")), str(run.get("trainer"))))
            known.add(run.get("arm"))
    return Experiment(
        name=str(index.get("name", results_dir.name)),
        backend=str((config.get("backend") or {}).get("type", "unknown")),
        factors=dict(index.get("factors") or {}),
        arms=arms,
        seeds=[int(s) for s in index.get("seeds", [])],
        checkpoints=sorted(int(s) for s in index.get("checkpoints", [])),
        ns=ns,
        cs=cs,
        fkeys=fkeys or ["(none)"],
        combos=combos or {"(none)": {}},
        groups=groups,
        records=records,
        missing=missing,
        primary=primary,
        primary_rule=rule,
        primary_note=note,
    )


# --------------------------------------------------------------------------- analysis


def groups_of_interest(exp: Experiment) -> list[str]:
    """Highlighted groups first (when present), then every other group in record order."""
    first = [g for g in HIGHLIGHT if g in exp.groups]
    return first + [g for g in exp.groups if g not in first]


def focus_group(exp: Experiment) -> str:
    for g in ("hard_heldout", "all"):
        if g in exp.groups:
            return g
    return exp.groups[0] if exp.groups else "all"


def _majority(counts: Counter) -> str:
    if not counts:
        return "n/a"
    top = max(counts.values())
    winners = sorted(k for k, v in counts.items() if v == top)
    return winners[0] if len(winners) == 1 else "tie(" + "|".join(winners) + ")"


def label_counts(exp: Experiment, arm: str, fkey: str, group: str, n: int, c: float, key: str = "label") -> Counter:
    counts: Counter = Counter()
    for rec in exp.cell_records(arm, fkey, exp.final, group, n, c):
        value = rec.get(key) if key == "label" else (rec.get("extra") or {}).get(key)
        if value is not None:
            counts[str(value)] += 1
    return counts


def conclusions(exp: Experiment, n: int, c: float) -> dict:
    """Headline conclusions at (N, c), final checkpoint: {(kind, arm, fkey, group): value}."""
    out: dict = {}
    for arm, _ in exp.arms:
        for fkey in exp.fkeys:
            for group in exp.groups:
                caps = [v for v in exp.values(arm, fkey, exp.final, group, n, c, "delta_cap") if not math.isnan(v)]
                if caps:
                    out[("delta_cap sign", arm, fkey, group)] = _sign(float(np.mean(caps)), CAP_SIGN_TOL)
                counts = label_counts(exp, arm, fkey, group, n, c)
                if counts:
                    out[("label majority", arm, fkey, group)] = _majority(counts)
                es = [v for v in exp.values(arm, fkey, exp.final, group, n, c, "E_both") if not math.isnan(v)]
                if es:
                    out[("E_N both sign", arm, fkey, group)] = _sign(float(np.mean(es)), E_SIGN_TOL)
    return out


def sensitivity(exp: Experiment) -> tuple[list, list]:
    """(summary rows, flip rows) comparing every measured (N, c) with the primary."""
    base = conclusions(exp, *exp.primary)
    rows, flips = [], []
    for n in exp.ns:
        for c in exp.cs:
            here = conclusions(exp, n, c)
            n_flip = 0
            for key in base:
                if key in here and here[key] != base[key]:
                    n_flip += 1
                    flips.append((n, c, *key, base[key], here[key]))
            rows.append((n, c, (n, c) == exp.primary, len(here), n_flip))
    return rows, flips


@dataclass(frozen=True)
class Check:
    status: str  # PASS or WARNING
    name: str
    message: str

    def line(self) -> str:
        return f"{self.status}: [{self.name}] {self.message}"


def _describe(key: RecKey) -> str:
    arm, fkey, seed, step, group, n, c = key
    return f"arm={arm} factors={fkey} seed={seed} step={step} group={group} N={n} c={_c_str(c)}"


def _examples(keys: list) -> str:
    shown = "; ".join(_describe(k) for k in keys[:_MAX_EXAMPLES])
    return shown + (f"; ... ({len(keys) - _MAX_EXAMPLES} more)" if len(keys) > _MAX_EXAMPLES else "")


def _zero_gain(rec: dict) -> bool:
    values = [metric(rec, "delta_total"), metric(rec, "delta_rel"), metric(rec, "delta_cap")]
    values.append(_f(rec.get("j1")) - _f(rec.get("j0")))
    return all(v == 0.0 for v in values)


def validity_checks(exp: Experiment, flips: list) -> list[Check]:
    checks: list[Check] = []
    keys = sorted(exp.records, key=lambda k: tuple(str(x) for x in k))

    if exp.missing:
        checks.append(Check("WARNING", "completeness", f"{len(exp.missing)} metrics file(s) missing: " + ", ".join(exp.missing[:_MAX_EXAMPLES])))
    else:
        checks.append(Check("PASS", "completeness", f"all {len(exp.records)} records loaded; no metrics file missing"))
    expected = {(a, f, s) for a, _ in exp.arms for f in exp.fkeys for s in exp.seeds}
    have = {(k[0], k[1], k[2]) for k in exp.records}
    lacking: list[tuple] = sorted(expected - have, key=str)
    for arm, fkey, seed in expected & have:
        for step in exp.checkpoints:
            for group in exp.groups:
                for n in exp.ns:
                    for c in exp.cs:
                        if (arm, fkey, seed, step, group, n, c) not in exp.records:
                            lacking.append((arm, fkey, seed, step, group, n, c))
    if lacking:
        shown = "; ".join(str(x) for x in lacking[:_MAX_EXAMPLES])
        checks.append(Check("WARNING", "coverage", f"{len(lacking)} expected (run or record) entries absent, e.g. {shown}"))
    else:
        checks.append(Check("PASS", "coverage", "every arm x factor x seed x checkpoint x group x (N, c) has a record"))

    bad, unverifiable, worst = [], [], 0.0
    for k in keys:
        rec = exp.records[k]
        dr, dc, j0, j1 = metric(rec, "delta_rel"), metric(rec, "delta_cap"), _f(rec.get("j0")), _f(rec.get("j1"))
        resid = abs(dr + dc - (j1 - j0))
        if math.isnan(resid):
            unverifiable.append(k)
            continue
        worst = max(worst, resid)
        if resid > PROP1_TOL:
            bad.append(k)
    if bad or unverifiable:
        msg = f"{len(bad)} record(s) violate |delta_rel + delta_cap - (j1 - j0)| <= {PROP1_TOL:g} (worst {worst:.3g})"
        if bad:
            msg += ": " + _examples(bad)
        if unverifiable:
            msg += f"; {len(unverifiable)} record(s) lack fields to verify: " + _examples(unverifiable)
        checks.append(Check("WARNING", "proposition 1", msg))
    else:
        checks.append(Check("PASS", "proposition 1", f"all {len(keys)} records satisfy it (worst residual {worst:.3g})"))

    none_arms = [a for a, t in exp.arms if t == "none"]
    if not none_arms:
        checks.append(Check("PASS", "zero-gain control", "not applicable: no arm has trainer 'none'"))
    else:
        bad = [k for k in keys if k[0] in none_arms and not _zero_gain(exp.records[k])]
        if bad:
            checks.append(Check("WARNING", "zero-gain control", f"{len(bad)} record(s) of trainer-'none' arm(s) have nonzero gain: " + _examples(bad)))
        else:
            checks.append(Check("PASS", "zero-gain control", f"arm(s) {', '.join(none_arms)}: delta_total = delta_rel = delta_cap = 0 exactly in every record"))

    step0 = [k for k in keys if k[3] == 0]
    bad = [k for k in step0 if not _zero_gain(exp.records[k])]
    if not step0:
        checks.append(Check("WARNING", "step 0", "no step-0 records found"))
    elif bad:
        checks.append(Check("WARNING", "step 0", f"{len(bad)} step-0 record(s) have nonzero gain: " + _examples(bad)))
    else:
        checks.append(Check("PASS", "step 0", f"all {len(step0)} step-0 records have zero gain exactly"))

    n_p, c_p = exp.primary
    if flips:
        by_nc = Counter((f[0], f[1]) for f in flips)
        for (n, c), count in sorted(by_nc.items()):
            examples = [f for f in flips if (f[0], f[1]) == (n, c)][:_MAX_EXAMPLES]
            shown = "; ".join(f"{f[2]} arm={f[3]} factors={f[4]} group={f[5]}: {f[6]} -> {f[7]}" for f in examples)
            checks.append(Check("WARNING", "sensitivity", f"N={n} c={_c_str(c)}: {count} conclusion(s) flip vs primary N={n_p} c={_c_str(c_p)}: {shown}"))
    else:
        checks.append(Check("PASS", "sensitivity", f"no headline conclusion flips across N in {exp.ns}, c in {[_c_str(c) for c in exp.cs]} (vs primary N={n_p} c={_c_str(c_p)})"))

    for flag, bad_value, meaning in (
        ("saturated", True, "best tempered base already near pass@N = 1"),
        ("boundary_hit", True, "best beta at the edge of the beta grid"),
        ("e_n_valid", False, "N too large for the trajectory space"),
    ):
        flagged = [k for k in keys if isinstance(exp.records[k].get("extra"), dict) and flag in exp.records[k]["extra"]]
        if not flagged:
            checks.append(Check("PASS", flag, "not recorded by this backend"))
            continue
        hits = [k for k in flagged if exp.records[k]["extra"][flag] == bad_value]
        primary_hits = [k for k in hits if k[3] == exp.final and k[5] == n_p and k[6] == c_p]
        if primary_hits:
            checks.append(Check("WARNING", flag, f"{len(primary_hits)} record(s) at the primary (N, c) and final checkpoint ({meaning}), {len(hits)} over all records: " + _examples(primary_hits)))
        else:
            checks.append(Check("PASS", flag, f"none at the primary (N, c) and final checkpoint; {len(hits)} of {len(flagged)} over all records"))
    return checks


# --------------------------------------------------------------------------- figures

_SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
_INK, _MUTED, _GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "axes.edgecolor": _MUTED, "axes.labelcolor": _INK, "xtick.color": _MUTED, "ytick.color": _MUTED,
        "axes.grid": True, "grid.color": _GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 9, "lines.linewidth": 2,
    })
    return plt


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight", metadata={"Software": None})


def _seed_mean(exp: Experiment, arm: str, fkey: str, step: int, group: str, name: str) -> float:
    vals = [v for v in exp.values(arm, fkey, step, group, *exp.primary, name) if not math.isnan(v)]
    return float(np.mean(vals)) if vals else math.nan


def figure_heatmaps(exp: Experiment, fig_dir: Path) -> list[Path]:
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    group = "hard_heldout" if "hard_heldout" in exp.groups else "all"
    factor_names = list(exp.factors)
    first = factor_names[0] if factor_names else None
    columns = [json.dumps(v, sort_keys=True) for v in exp.factors[first]] if first else ["(none)"]
    rest_keys: dict = {}  # rest-of-factors label -> {column label: fkey}
    for fkey in exp.fkeys:
        combo = exp.combos[fkey]
        col = json.dumps(combo.get(first), sort_keys=True) if first else "(none)"
        rest = _fkey({k: v for k, v in combo.items() if k != first})
        rest_keys.setdefault(rest, {})[col] = fkey
    paths = []
    for i, (rest, cols) in enumerate(rest_keys.items()):
        arms = [a for a, _ in exp.arms]
        grid = np.full((len(arms), len(columns)), np.nan)
        for r, arm in enumerate(arms):
            for j, col in enumerate(columns):
                if col in cols:
                    grid[r, j] = _seed_mean(exp, arm, cols[col], exp.final, group, "delta_cap")
        finite = np.abs(grid[np.isfinite(grid)])
        vmax = float(finite.max()) if finite.size and finite.max() > 0 else 1e-12
        fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(columns), 1.0 + 0.5 * len(arms)))
        ax.grid(False)
        ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        for r in range(len(arms)):
            for j in range(len(columns)):
                text = "n/a" if math.isnan(grid[r, j]) else f"{grid[r, j]:+.3g}"
                ax.text(j, r, text, ha="center", va="center", color=_INK, fontsize=8)
        ax.set_xticks(range(len(columns)), [f"{first}={c}" if first else c for c in columns], rotation=30, ha="right")
        ax.set_yticks(range(len(arms)), arms)
        title = f"seed-mean delta_cap on {group}, N={exp.primary[0]} c={_c_str(exp.primary[1])}, step {exp.final}"
        ax.set_title(title + ("" if rest == "(none)" else f"\n{rest}"), fontsize=9)
        name = "main_grid.png" if len(rest_keys) == 1 else f"main_grid_{i}.png"
        _save(fig, fig_dir / name)
        plt.close(fig)
        paths.append(fig_dir / name)
    return paths


def figure_training(exp: Experiment, fig_dir: Path) -> list[Path]:
    plt = _plt()
    group = focus_group(exp)
    steps = exp.checkpoints
    paths = []
    for a, (arm, _) in enumerate(exp.arms):
        fig, axes = plt.subplots(1, len(exp.fkeys), figsize=(3.6 * len(exp.fkeys), 2.8), squeeze=False, sharey=True)
        for ax, fkey in zip(axes[0], exp.fkeys):
            for s, name in enumerate(("delta_rel", "delta_cap")):
                rows = [[v for v in exp.values(arm, fkey, st, group, *exp.primary, name) if not math.isnan(v)] for st in steps]
                ok = [i for i, r in enumerate(rows) if r]
                if not ok:
                    continue
                xs = [steps[i] for i in ok]
                ax.fill_between(xs, [min(rows[i]) for i in ok], [max(rows[i]) for i in ok], color=_SERIES[s], alpha=0.18, linewidth=0)
                ax.plot(xs, [float(np.mean(rows[i])) for i in ok], color=_SERIES[s], marker="o", markersize=4, label=name)
            ax.axhline(0, color=_MUTED, linewidth=0.8)
            ax.set_title(fkey, fontsize=9)
            ax.set_xlabel("update step")
        axes[0][0].set_ylabel(f"{group} (seed mean, min-max band)")
        axes[0][0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{arm}: N={exp.primary[0]} c={_c_str(exp.primary[1])}", fontsize=10)
        path = fig_dir / f"training_{a:02d}_{arm}.png"
        _save(fig, path)
        plt.close(fig)
        paths.append(path)
    return paths


def frontier_rows(exp: Experiment) -> list[tuple]:
    rows = []
    for arm, _ in exp.arms:
        for fkey in exp.fkeys:
            for group in exp.groups:
                pts = [pass_and_frontier(r) for r in exp.cell_records(arm, fkey, exp.final, group, *exp.primary)]
                trained = [p[0] for p in pts if not math.isnan(p[0])]
                front = [p[1] for p in pts if not math.isnan(p[1])]
                sources = sorted({p[2] for p in pts})
                rows.append((
                    arm, fkey, group,
                    float(np.mean(trained)) if trained else math.nan,
                    float(np.mean(front)) if front else math.nan,
                    _seed_mean(exp, arm, fkey, exp.final, group, "E_both"),
                    "|".join(sources),
                ))
    return rows


def figure_frontier(exp: Experiment, fig_dir: Path, rows: list[tuple]) -> list[Path]:
    plt = _plt()
    panels = [g for g in ("all", *HIGHLIGHT) if g in exp.groups] or exp.groups[:1]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.4), squeeze=False)
    arms = [a for a, _ in exp.arms]
    for ax, group in zip(axes[0], panels):
        ax.plot([0, 1], [0, 1], color=_MUTED, linestyle="--", linewidth=1)
        for arm, fkey, g, trained, front, _, _ in rows:
            if g != group or math.isnan(trained) or math.isnan(front):
                continue
            a, f = arms.index(arm), exp.fkeys.index(fkey)
            # Earlier arms are drawn larger so identical points stay visible as rings.
            size = max(110 - 30 * a, 20)
            ax.scatter([front], [trained], s=size, color=_SERIES[a % len(_SERIES)], marker=_MARKERS[f % len(_MARKERS)],
                       edgecolors="white", linewidths=1.5, zorder=3 + a)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_title(group, fontsize=9)
        ax.set_xlabel("best tempered-base pass@N")
    axes[0][0].set_ylabel("trained pass@N")
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=_SERIES[i % len(_SERIES)], marker="o", linestyle="", label=a) for i, a in enumerate(arms)]
    if len(exp.fkeys) > 1:
        handles += [Line2D([], [], color=_MUTED, marker=_MARKERS[i % len(_MARKERS)], linestyle="", label=f)
                    for i, f in enumerate(exp.fkeys)]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.10 - 0.05 * ((len(handles) - 1) // 4)))
    fig.suptitle(f"Trained policy vs tempering frontier (N={exp.primary[0]}, step {exp.final}); above the diagonal beats every tempered base", fontsize=9)
    path = fig_dir / "frontier.png"
    _save(fig, path)
    plt.close(fig)
    return [path]


# --------------------------------------------------------------------------- report


def _cell_md(stat: Stat) -> str:
    if stat.n_seeds == 0:
        return "n/a"
    return f"{_md_num(stat.mean)} [{_md_num(stat.ci_low)}, {_md_num(stat.ci_high)}] ({_md_num(stat.min, 3)}..{_md_num(stat.max, 3)})"


def _counts_str(counts: Counter) -> str:
    return ", ".join(f"{k}:{counts[k]}" for k in sorted(counts)) or "n/a"


def _md_table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |" for row in rows]
    return lines + [""]


def build_report(results_dir: str | Path, out_dir: str | Path | None = None) -> Path:
    """Build the report for `<out>/<name>` and return the report folder."""
    results = Path(results_dir)
    out = Path(out_dir) if out_dir is not None else results / "report"
    tables, figs = out / "tables", out / "figures"
    exp = load_experiment(results)
    n_p, c_p = exp.primary
    final = exp.final
    groups = groups_of_interest(exp)

    # Main table.
    main: dict = {}
    for arm, _ in exp.arms:
        for fkey in exp.fkeys:
            for group in groups:
                for name in METRICS:
                    main[(arm, fkey, group, name)] = summarize(exp.values(arm, fkey, final, group, n_p, c_p, name))
    _write_csv(
        tables / "main.csv",
        ("arm", "trainer", "factors", "group", "metric", "n_seeds", "mean", "min", "max", "ci_low", "ci_high"),
        ((k[0], exp.trainer(k[0]), k[1], k[2], k[3], s.n_seeds, s.mean, s.min, s.max, s.ci_low, s.ci_high) for k, s in main.items()),
    )
    label_rows = []
    for arm, _ in exp.arms:
        for fkey in exp.fkeys:
            for group in groups:
                for kind in ("label", "d2_label"):
                    counts = label_counts(exp, arm, fkey, group, n_p, c_p, kind)
                    label_rows += [(arm, fkey, group, kind, lab, counts[lab]) for lab in sorted(counts)]
    _write_csv(tables / "labels.csv", ("arm", "factors", "group", "kind", "label", "count"), label_rows)

    # Trajectory.
    traj_rows = []
    for arm, _ in exp.arms:
        for fkey in exp.fkeys:
            for group in groups:
                for step in exp.checkpoints:
                    for name in TRAJ_METRICS:
                        s = summarize(exp.values(arm, fkey, step, group, n_p, c_p, name), ci=False)
                        traj_rows.append((arm, fkey, group, step, name, s.n_seeds, s.mean, s.min, s.max))
    _write_csv(tables / "trajectory.csv", ("arm", "factors", "group", "step", "metric", "n_seeds", "mean", "min", "max"), traj_rows)

    # Sensitivity and checks.
    sens_rows, flips = sensitivity(exp)
    _write_csv(tables / "sensitivity.csv", ("n", "c", "is_primary", "n_conclusions", "n_flips"),
               ((n, c, "yes" if p else "no", k, f) for n, c, p, k, f in sens_rows))
    _write_csv(tables / "sensitivity_flips.csv", ("n", "c", "conclusion", "arm", "factors", "group", "primary_value", "value"), flips)
    checks = validity_checks(exp, flips)
    _write_csv(tables / "checks.csv", ("status", "check", "message"), ((c.status, c.name, c.message) for c in checks))
    for check in checks:
        print(check.line(), file=sys.stderr)

    fr_rows = frontier_rows(exp)
    _write_csv(tables / "frontier.csv", ("arm", "factors", "group", "trained_pass_at_n", "best_tempered_pass_at_n", "E_N_both", "source"), fr_rows)

    # Figures.
    figure_paths: list[Path] = []
    figure_note = ""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        figure_note = "matplotlib is not installed (`pip install capsplit[viz]`); figures were skipped."
    else:
        if exp.records:
            figure_paths += figure_heatmaps(exp, figs)
            figure_paths += figure_training(exp, figs)
            figure_paths += figure_frontier(exp, figs, fr_rows)

    # summary.json
    cells = []
    for arm, trainer in exp.arms:
        for fkey in exp.fkeys:
            for group in groups:
                counts = label_counts(exp, arm, fkey, group, n_p, c_p)
                d2 = label_counts(exp, arm, fkey, group, n_p, c_p, "d2_label")
                cells.append({
                    "arm": arm, "trainer": trainer, "factors": exp.combos.get(fkey, {}), "factors_label": fkey, "group": group,
                    "metrics": {m: main[(arm, fkey, group, m)].to_json() for m in METRICS},
                    "labels": dict(sorted(counts.items())), "label_majority": _majority(counts),
                    "d2_labels": dict(sorted(d2.items())),
                })
    summary = {
        "experiment": exp.name,
        "backend": exp.backend,
        "primary": {"n": n_p, "c": c_p, "rule": exp.primary_rule, "note": exp.primary_note},
        "final_step": final,
        "seeds": exp.seeds,
        "groups": groups,
        "bootstrap": {"resamples": N_BOOT, "rng_seed": BOOT_SEED, "method": "percentile", "level": 0.95},
        "cells": cells,
        "checks": [{"status": c.status, "check": c.name, "message": c.message} for c in checks],
        "n_warnings": sum(c.status == "WARNING" for c in checks),
        "sensitivity_flips": [
            {"n": f[0], "c": f[1], "conclusion": f[2], "arm": f[3], "factors": f[4], "group": f[5], "primary_value": f[6], "value": f[7]}
            for f in flips
        ],
    }
    _write(out / "summary.json", json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")

    # report.md
    md: list[str] = [f"# Report: {exp.name}", ""]
    if exp.backend in EXACT_BACKENDS:
        md.append(f"Backend `{exp.backend}`: every number is **exact** (computed by enumeration), so there is no sampling error. "
                  "The intervals below describe **seed-to-seed spread** only (different bases and training runs), not measurement noise.")
    else:
        md.append(f"Backend `{exp.backend}`: the intervals below describe **seed-to-seed spread** only; they do not include any "
                  "within-run sampling error the backend may have.")
    md += ["", f"Cell format: `seed mean [bootstrap 95% interval] (min..max)`. Interval: percentile bootstrap of the seed mean, "
           f"{N_BOOT:,} resamples, numpy default_rng({BOOT_SEED}).", ""]
    md += ["## Setup", ""]
    md.append(f"- Arms: " + ", ".join(f"`{a}` ({t})" for a, t in exp.arms))
    md.append(f"- Factors: " + (", ".join(f"`{k}` in {json.dumps(v)}" for k, v in exp.factors.items()) or "none"))
    md.append(f"- Seeds: {exp.seeds}; checkpoints: {exp.checkpoints}; N in {exp.ns}; c in {[_c_str(c) for c in exp.cs]}")
    md.append(f"- **Primary (N, c) = ({n_p}, {_c_str(c_p)})**, rule: {exp.primary_rule}." + (f" Note: {exp.primary_note}." if exp.primary_note else ""))
    md.append(f"- Main tables use the final checkpoint (step {final}).")
    md += ["", "## Validity checks", ""]
    md += [f"- {'**WARNING**' if c.status == 'WARNING' else 'PASS'}: [{c.name}] {c.message}" for c in checks]
    md += ["", f"## Main table (N={n_p}, c={_c_str(c_p)}, step {final})", ""]
    md.append("Highlighted groups (marked *) come first. E_N columns: excess pass@N over the best tempered base, per family.")
    md.append("")
    for group in groups:
        mark = " *" if group in HIGHLIGHT else ""
        md += [f"### Group `{group}`{mark}", ""]
        header = ["arm", "factors", *METRICS, "labels", "d2 labels"]
        rows = []
        for arm, _ in exp.arms:
            for fkey in exp.fkeys:
                rows.append([arm, fkey, *(_cell_md(main[(arm, fkey, group, m)]) for m in METRICS),
                             _counts_str(label_counts(exp, arm, fkey, group, n_p, c_p)),
                             _counts_str(label_counts(exp, arm, fkey, group, n_p, c_p, "d2_label"))])
        md += _md_table(header, rows)
    traj_groups = [focus_group(exp)] + (["all"] if focus_group(exp) != "all" and "all" in exp.groups else [])
    md += [f"## Trajectories (seed means, N={n_p}, c={_c_str(c_p)})", "",
           f"Groups shown: {', '.join(traj_groups)}; every group is in `tables/trajectory.csv`.", ""]
    for arm, _ in exp.arms:
        md += [f"### Arm `{arm}`", ""]
        rows = []
        for fkey in exp.fkeys:
            for group in traj_groups:
                for step in exp.checkpoints:
                    rows.append([fkey, group, str(step), *(_md_num(_seed_mean(exp, arm, fkey, step, group, m)) for m in TRAJ_METRICS)])
        md += _md_table(["factors", "group", "step", *TRAJ_METRICS], rows)
    md += ["## Sensitivity to N and c", "",
           "Headline conclusions at the final checkpoint: sign of seed-mean delta_cap (zero if |x| <= "
           f"{CAP_SIGN_TOL:g}), majority label, sign of seed-mean E_N both (zero if |x| <= {E_SIGN_TOL:g}), "
           "for every arm x factors x group. A flip is a conclusion that differs from the primary (N, c).", ""]
    md += _md_table(["N", "c", "primary", "conclusions", "flips"],
                    ([str(n), _c_str(c), "yes" if p else "", str(k), str(f)] for n, c, p, k, f in sens_rows))
    if flips:
        md += ["Flips (all rows in `tables/sensitivity_flips.csv`):", ""]
        md += _md_table(["N", "c", "conclusion", "arm", "factors", "group", "primary", "here"],
                        ([str(f[0]), _c_str(f[1]), *map(str, f[2:])] for f in flips[:50]))
    md += [f"## Tempering frontier (N={n_p}, step {final})", "",
           "Trained pass@N vs the best tempered-base pass@N (seed means). Positive E_N both = beats every tempered base.", ""]
    md += _md_table(["arm", "factors", "group", "trained pass@N", "best tempered pass@N", "E_N both", "source"],
                    ([r[0], r[1], r[2], _md_num(r[3]), _md_num(r[4]), _md_num(r[5]), r[6]] for r in fr_rows if r[2] in ("all", *HIGHLIGHT)))
    md += ["## Figures", ""]
    if figure_note:
        md.append(figure_note)
    for path in figure_paths:
        rel = path.relative_to(out).as_posix()
        md.append(f"- [{rel}]({rel})")
    md += ["", "## Files", "", "`tables/*.csv` (main, labels, trajectory, frontier, sensitivity, sensitivity_flips, checks), `summary.json`.", ""]
    _write(out / "report.md", "\n".join(md))
    return out
