"""Budget curve: how the reliability/capability split of a trained policy moves with the budget N.

For prompt i, q0(s) and q1(s) are the base and trained probabilities of correct solution s
(`engine.solution_probs` on exact trajectory probabilities). At budget N and multiplier c,
s is *reachable* iff q0(s) >= c/N (the exact backend's rule). For a group of prompts::

    rel(N) = mean_i sum_{s reachable}   (q1(s) - q0(s))
    cap(N) = mean_i sum_{s unreachable} (q1(s) - q0(s))
    rel(N) + cap(N) = J1 - J0 = total, for every N          (checked to 1e-12)

The curve is computed from the saved policies, with no training: for every run in
`index.json`, the base policy (`bases/<key>/policy.pt`, cached per base) and a clone with the
checkpoint's state_dict (`<cell>/seed-<s>.policies.pt`). The task is rebuilt through the
registry from the cell's resolved config (`<cell>/manifest.json`), and the groups are the
experiment's (`config.build_groups` with `measure.group_by`).

Quantities reported:

* The curve on a grid of integer budgets, by default 1, 2, 4, ... up to the first power of two
  >= max(2^20, 64 x the largest N_train) (`default_grid`).
* Crossover N* = the smallest grid N with rel(N) >= 0.5 x total; defined only when
  total > 0 (null otherwise, or when the grid never reaches it). rel(N) is non-decreasing in N
  when every per-solution change q1 - q0 is >= 0, but not in general, so N* is read literally
  off the grid.
* cap(N_train), evaluated exactly at N_train (a mean, so possibly not an integer).

N_train is the number of episodes sampled per training prompt during training. It is read from
`<cell>/seed-<s>.training.json` (`trainers.rl.SampleLog`: "prompts" -> {pid: {"episodes_sampled"}};
the mean over the run's
training prompts, with min/max recorded). Without a log it comes from the caller's `n_train`
mapping (by arm name, then trainer name) and is marked "from_config (not logged)"; otherwise
it is unknown and cap(N_train) is null.

Sampled-backend runs: reachability there is decided from `m_classify` base samples, so the
experiment cannot resolve a solution whose base probability is below the 95% Clopper-Pearson
upper bound for zero successes, u0 = 1 - (alpha/2)^(1/m). Only N <= floor(c/u0) is resolvable
(`resolvable_n_max`); the grid is truncated there and the output says so.

Outputs (in `out_dir`, default `<results_dir>/budget_curve/`)::

    budget_curve.csv     arm, factors, seed, group, N, rel, cap, total
    summary.json         per arm x factors x group: seed-mean N*, cap at N_train (min/max over seeds), N_train + source
    budget_curve.md      plain-language summary
    figures/budget_<arm>.png
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from capsplit.engine import solution_probs, table_success_prob

from .config import build_groups
from .registry import TASKS
from .report import _SERIES, _fkey, _plt, _save, _write, _write_csv
from .runner import _torch_load, dumps

IDENTITY_TOL = 1e-12
DEFAULT_ALPHA = 0.05
DEFAULT_M_CLASSIFY = 4096  # the sampled backend's default
SOURCE_LOG = "training_log"
SOURCE_CONFIG = "from_config (not logged)"
SOURCE_UNKNOWN = "unknown"


# --------------------------------------------------------------------------- pure functions


def default_grid(n_train_max: float | None = None, factor: int = 64, floor: int = 2**20) -> list[int]:
    """1, 2, 4, ... up to the first power of two >= max(floor, factor * n_train_max)."""
    top = max(float(floor), factor * float(n_train_max or 0))
    grid, n = [1], 1
    while n < top:
        n *= 2
        grid.append(n)
    return grid


def clopper_pearson_zero_upper(m: int, alpha: float = DEFAULT_ALPHA) -> float:
    """Two-sided (1 - alpha) Clopper-Pearson upper bound on p after 0 successes in m trials."""
    if m < 1:
        raise ValueError("m must be >= 1")
    return 1.0 - (alpha / 2.0) ** (1.0 / m)


def resolvable_n_max(m_classify: int, c: float = 1.0, alpha: float = DEFAULT_ALPHA) -> int:
    """Largest N a sampled classification with `m_classify` base samples can resolve: floor(c/u0)."""
    return int(math.floor(c / clopper_pearson_zero_upper(m_classify, alpha)))


def split_curve(
    q0: Sequence[np.ndarray], q1: Sequence[np.ndarray], ns: Sequence[float], c: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """(rel, cap) at each N for a group: per-prompt sums over reachable / unreachable solutions,
    averaged over prompts. `q0[i]`, `q1[i]` are prompt i's per-solution probabilities."""
    if len(q0) != len(q1) or not len(q0):
        raise ValueError("need the same, non-zero number of prompts in q0 and q1")
    rel = np.zeros(len(ns))
    cap = np.zeros(len(ns))
    for a, b in zip(q0, q1):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        change = b - a
        for k, n in enumerate(ns):
            reachable = a >= c / n
            rel[k] += change[reachable].sum()
            cap[k] += change[~reachable].sum()
    return rel / len(q0), cap / len(q0)


def crossover(ns: Sequence[int], rel: Sequence[float], total: float, fraction: float = 0.5) -> int | None:
    """Smallest N in `ns` (in order) with rel(N) >= fraction * total; None if total <= 0 or never."""
    if not total > 0:
        return None
    for n, r in zip(ns, rel):
        if r >= fraction * total:
            return int(n)
    return None


# --------------------------------------------------------------------------- N_train


def _episodes_per_prompt(log: Any, train_ids: Sequence[int]) -> dict[int, float]:
    """{prompt id: episodes sampled} from a training log; accepts the shapes listed below.

    * {"per_prompt" | "prompts": {pid: {"episodes_sampled": n} | n}}
    * {"per_prompt" | "prompts": [{"prompt_id" | "id": pid, "episodes_sampled": n}, ...]}
    * {"episodes_sampled": {pid: n}} or {"episodes_sampled": [n, ...]} aligned with
      "prompt_ids" (else with the run's training prompts)
    """
    if not isinstance(log, dict):
        raise ValueError("training log is not a JSON object")

    def value(v: Any) -> float:
        return float(v["episodes_sampled"] if isinstance(v, dict) else v)

    for key in ("per_prompt", "prompts"):
        if key in log:
            table = log[key]
            if isinstance(table, dict):
                return {int(k): value(v) for k, v in table.items()}
            if isinstance(table, list):
                out = {}
                for row in table:
                    pid = row.get("prompt_id", row.get("id"))
                    if pid is None:
                        raise ValueError(f"training log {key!r} row has no prompt_id")
                    out[int(pid)] = value(row)
                return out
    if "episodes_sampled" in log:
        table = log["episodes_sampled"]
        if isinstance(table, dict):
            return {int(k): float(v) for k, v in table.items()}
        if isinstance(table, list):
            ids = list(log.get("prompt_ids") or log.get("train_prompt_ids") or train_ids)
            if len(ids) != len(table):
                raise ValueError("training log episodes_sampled list does not match its prompt ids")
            return {int(p): float(v) for p, v in zip(ids, table)}
    raise ValueError("training log has no per-prompt episodes_sampled")


def n_train_for_run(
    exp_dir: Path, run: dict, train_ids: Sequence[int], n_train: Mapping[str, float] | None
) -> dict:
    """{"value", "min", "max", "source", "note"} for one run (value None when unknown)."""
    rel = run.get("files", {}).get("training") or f"{run['run_dir']}/seed-{run['seed']}.training.json"
    log_path = exp_dir / rel
    if log_path.exists():
        per = _episodes_per_prompt(json.loads(log_path.read_text()), train_ids)
        values = [per.get(int(p), 0.0) for p in train_ids]
        missing = [int(p) for p in train_ids if int(p) not in per]
        note = f"{len(missing)} training prompts absent from the log, counted as 0" if missing else ""
        return {
            "value": float(np.mean(values)) if values else 0.0,
            "min": float(min(values)) if values else 0.0,
            "max": float(max(values)) if values else 0.0,
            "source": SOURCE_LOG,
            "note": note,
        }
    mapping = n_train or {}
    for key in (run["arm"], run.get("trainer")):
        if key is not None and key in mapping:
            v = float(mapping[key])
            return {"value": v, "min": v, "max": v, "source": SOURCE_CONFIG, "note": f"n_train[{key!r}]"}
    return {"value": None, "min": None, "max": None, "source": SOURCE_UNKNOWN, "note": "no training log and not in n_train"}


# --------------------------------------------------------------------------- per run


class _Context:
    """Caches tasks, tables, groups, base policies and base q0 across runs."""

    def __init__(self, exp_dir: Path):
        self.exp_dir = exp_dir
        self.tasks: dict[str, Any] = {}
        self.tables: dict[str, dict] = {}
        self.bases: dict[str, Any] = {}
        self.q0: dict[str, dict] = {}

    def task(self, cfg: dict) -> tuple[Any, str]:
        key = json.dumps(cfg, sort_keys=True)
        if key not in self.tasks:
            self.tasks[key] = TASKS.create(cfg["type"], **cfg.get("params", {}))
            self.tables[key] = {}
        return self.tasks[key], key

    def table(self, task_key: str, pid: int):
        tables = self.tables[task_key]
        if pid not in tables:
            tables[pid] = self.tasks[task_key].table(pid)
        return tables[pid]

    def base(self, rel_path: str):
        if rel_path not in self.bases:
            self.bases[rel_path] = _torch_load(self.exp_dir / rel_path / "policy.pt")
        return self.bases[rel_path]


def _probs(task, table, policy, pid) -> tuple[np.ndarray, float]:
    probs = np.asarray(policy.trajectory_probs(task, pid), dtype=float)
    return solution_probs(table, probs), table_success_prob(table, probs)


def _run_curves(ctx: _Context, run: dict, step: int | None, ns_full: list[int], c: float, n_train_info: dict) -> dict:
    """Per-group curves for one run, or {"skip": reason}."""
    exp_dir = ctx.exp_dir
    if not run["files"].get("policies"):
        return {"skip": "no saved policies"}
    manifest = json.loads((exp_dir / run["files"]["manifest"]).read_text())
    cfg = manifest["config"]
    task, task_key = ctx.task(cfg["task"])
    pids = [p.id for p in task.prompts()]
    if any(ctx.table(task_key, pid) is None for pid in pids[:1]):
        return {"skip": "task has no exact table"}

    backend = cfg.get("backend", {})
    resolvable = None
    ns = list(ns_full)
    if backend.get("type") == "sampled":
        params = backend.get("params", {})
        resolvable = resolvable_n_max(
            int(params.get("m_classify", DEFAULT_M_CLASSIFY)), c, float(params.get("alpha", DEFAULT_ALPHA))
        )
        ns = [n for n in ns if n <= resolvable]

    saved = _torch_load(exp_dir / run["files"]["policies"])
    use = max(saved) if step is None else step
    if use not in saved:
        return {"skip": f"checkpoint {use} not saved (have {sorted(saved)})"}
    base = ctx.base(run["base"])
    trained = base.clone()
    trained.load_state_dict(saved[use])

    q0_cache = ctx.q0.setdefault(run["base"], {})
    q0, q1, j0, j1 = {}, {}, {}, {}
    with torch.no_grad():
        for pid in pids:
            table = ctx.table(task_key, pid)
            if pid not in q0_cache:
                q0_cache[pid] = _probs(task, table, base, pid)
            q0[pid], j0[pid] = q0_cache[pid]
            q1[pid], j1[pid] = _probs(task, table, trained, pid)

    groups = build_groups(task, cfg["measure"].get("group_by", []))
    nt = n_train_info["value"]
    out: dict = {"step": use, "ns": ns, "resolvable_n_max": resolvable, "groups": {}}
    for name, ids in groups.items():
        total = float(np.mean([j1[p] - j0[p] for p in ids]))
        rel, cap = split_curve([q0[p] for p in ids], [q1[p] for p in ids], ns, c)
        worst = float(np.max(np.abs(rel + cap - total))) if len(ns) else 0.0
        if worst > IDENTITY_TOL:
            raise AssertionError(f"rel + cap != total for {run['run_dir']} seed {run['seed']} {name}: off by {worst}")
        cap_nt, cap_note = None, ""
        if nt is None:
            cap_note = "N_train unknown"
        elif nt < 1:
            cap_note = "N_train < 1 (no episodes sampled)"
        elif resolvable is not None and nt > resolvable:
            cap_note = f"N_train {nt:g} > largest resolvable N {resolvable}"
        else:
            r_nt, c_nt = split_curve([q0[p] for p in ids], [q1[p] for p in ids], [nt], c)
            assert abs(r_nt[0] + c_nt[0] - total) <= IDENTITY_TOL
            cap_nt = float(c_nt[0])
        out["groups"][name] = {
            "rel": rel.tolist(),
            "cap": cap.tolist(),
            "total": total,
            "n_star": crossover(ns, [float(x) for x in rel], total),
            "cap_at_n_train": cap_nt,
            "cap_note": cap_note,
            "n_prompts": len(ids),
        }
    return out


# --------------------------------------------------------------------------- build


def _stat(values: list) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "min": None, "max": None, "n": 0}
    return {"mean": float(np.mean(vals)), "min": float(min(vals)), "max": float(max(vals)), "n": len(vals)}


def _train_heldout_panels(groups: list[str]) -> list[str]:
    """Groups for the figure columns: *_train / *_heldout (then train / heldout), else all."""
    split = [g for g in groups if g.endswith("_train") or g.endswith("_heldout")]
    if not split:
        split = [g for g in groups if g in ("train", "heldout")]
    return sorted(split, key=lambda g: (g.endswith("heldout"), g)) or ["all"]


def build_budget_curve(
    results_dir: str | Path,
    out_dir: str | Path | None = None,
    n_grid: Sequence[int] | None = None,
    c: float = 1.0,
    step: int | None = None,
    n_train: Mapping[str, float] | None = None,
    arms: Sequence[str] | None = None,
    log=None,
) -> Path:
    """Compute the budget curve for every run in `results_dir/index.json`; return `out_dir`.

    `n_grid`: budgets to evaluate (default `default_grid`); `c`: threshold multiplier; `step`:
    checkpoint (default the final saved one); `n_train`: {arm or trainer name: episodes per
    training prompt} used only for runs without a training log; `arms`: restrict to these arms.
    Never writes outside `out_dir`, and refuses an `out_dir` that already holds outputs.
    """
    exp_dir = Path(results_dir)
    out = Path(out_dir) if out_dir is not None else exp_dir / "budget_curve"
    for name in ("budget_curve.csv", "summary.json", "budget_curve.md"):
        if (out / name).exists():
            raise FileExistsError(f"{out / name} exists; choose a new out_dir")
    index = json.loads((exp_dir / "index.json").read_text())
    runs = [r for r in index["runs"] if arms is None or r["arm"] in arms]
    say = log or (lambda msg: None)
    ctx = _Context(exp_dir)

    # N_train first: the default grid depends on the largest one.
    infos = []
    for run in runs:
        metrics_path = exp_dir / run["files"]["metrics"]
        train_ids = json.loads(metrics_path.read_text())["train_prompt_ids"]
        infos.append(n_train_for_run(exp_dir, run, train_ids, n_train))
    known = [i["value"] for i in infos if i["value"] is not None]
    ns = sorted({int(n) for n in n_grid}) if n_grid is not None else default_grid(max(known) if known else None)
    if not ns or ns[0] < 1:
        raise ValueError("n_grid must hold integers >= 1")

    results = []
    for k, (run, info) in enumerate(zip(runs, infos)):
        say(f"[{k + 1}/{len(runs)}] {run['arm']} {_fkey(run['factors'])} seed {run['seed']}")
        results.append(_run_curves(ctx, run, step, ns, c, info))

    # ---------------------------------------------------------------- CSV
    rows = []
    for run, res in zip(runs, results):
        if "skip" in res:
            continue
        for g, cur in res["groups"].items():
            for n, r, cp in zip(res["ns"], cur["rel"], cur["cap"]):
                rows.append((run["arm"], _fkey(run["factors"]), run["seed"], g, n, r, cp, cur["total"]))
    _write_csv(out / "budget_curve.csv", ("arm", "factors", "seed", "group", "N", "rel", "cap", "total"), rows)

    # ---------------------------------------------------------------- summary
    cells: dict = defaultdict(list)
    for run, res, info in zip(runs, results, infos):
        cells[(run["arm"], _fkey(run["factors"]))].append((run, res, info))
    entries, skipped = [], []
    for (arm, fkey), items in cells.items():
        done = [(run, res, info) for run, res, info in items if "skip" not in res]
        for run, res, _ in items:
            if "skip" in res:
                skipped.append({"arm": arm, "factors": fkey, "seed": run["seed"], "reason": res["skip"]})
        if not done:
            continue
        sources = sorted({info["source"] for _, _, info in done})
        nt_vals = [info["value"] for _, _, info in done]
        resolvable = sorted({res["resolvable_n_max"] for _, res, _ in done if res["resolvable_n_max"] is not None})
        for g in done[0][1]["groups"]:
            per = [(run["seed"], res["groups"][g]) for run, res, _ in done]
            n_stars = [cur["n_star"] for _, cur in per]
            totals = [cur["total"] for _, cur in per]
            entries.append({
                "arm": arm,
                "factors": fkey,
                "group": g,
                "seeds": [s for s, _ in per],
                "step": sorted({res["step"] for _, res, _ in done}),
                "total": _stat(totals),
                "n_star": {
                    "seed_mean": _stat(n_stars)["mean"],
                    "seed_geomean": float(np.exp(np.mean(np.log([n for n in n_stars if n is not None]))))
                    if any(n is not None for n in n_stars) else None,
                    "per_seed": n_stars,
                    "n_defined": sum(n is not None for n in n_stars),
                    "note": "mean over seeds where N* is defined (total > 0 and reached within the grid)",
                },
                "cap_at_n_train": {**_stat([cur["cap_at_n_train"] for _, cur in per]),
                                   "notes": sorted({cur["cap_note"] for _, cur in per if cur["cap_note"]})},
                "n_train": {
                    **{k: v for k, v in _stat(nt_vals).items() if k != "n"},
                    "per_prompt_min": _stat([i["min"] for _, _, i in done])["min"],
                    "per_prompt_max": _stat([i["max"] for _, _, i in done])["max"],
                    "source": sources[0] if len(sources) == 1 else "mixed: " + ", ".join(sources),
                },
                "resolvable_n_max": resolvable[0] if len(resolvable) == 1 else (resolvable or None),
            })
    summary = {
        "results_dir": str(exp_dir),
        "c": c,
        "step": "final" if step is None else step,
        "grid": ns,
        "identity_tolerance": IDENTITY_TOL,
        "definitions": {
            "reachable": "q0(s) >= c/N",
            "rel": "mean over prompts of sum over reachable s of q1(s) - q0(s)",
            "cap": "mean over prompts of sum over unreachable s of q1(s) - q0(s)",
            "n_star": "smallest grid N with rel(N) >= 0.5 * total; null if total <= 0 or never reached",
            "cap_at_n_train": "cap evaluated exactly at N = N_train (mean episodes per training prompt)",
        },
        "entries": entries,
        "skipped": skipped,
    }
    _write(out / "summary.json", dumps(summary))

    # ---------------------------------------------------------------- figures + markdown
    figures = _figures(runs, results, infos, out / "figures")
    _write(out / "budget_curve.md", _markdown(summary, figures, out))
    return out


# --------------------------------------------------------------------------- figures


def _figures(runs: list, results: list, infos: list, fig_dir: Path) -> list[Path]:
    plt = _plt()
    by_arm: dict = defaultdict(lambda: defaultdict(list))
    for run, res, info in zip(runs, results, infos):
        if "skip" not in res:
            by_arm[run["arm"]][_fkey(run["factors"])].append((res, info))
    paths = []
    for arm, levels in by_arm.items():
        first = next(iter(levels.values()))[0][0]
        panels = _train_heldout_panels(list(first["groups"]))
        fig, axes = plt.subplots(
            len(levels), len(panels), figsize=(3.2 * len(panels), 2.6 * len(levels)), squeeze=False, sharex=True
        )
        for r, (fkey, items) in enumerate(levels.items()):
            ns = items[0][0]["ns"]
            nts = [i["value"] for _, i in items if i["value"] is not None]
            for col, g in enumerate(panels):
                ax = axes[r][col]
                for name, color in (("rel", _SERIES[0]), ("cap", _SERIES[1])):
                    vals = np.array([res["groups"][g][name] for res, _ in items])
                    ax.fill_between(ns, vals.min(axis=0), vals.max(axis=0), color=color, alpha=0.2, linewidth=0)
                    ax.plot(ns, vals.mean(axis=0), color=color, label=name)
                total = float(np.mean([res["groups"][g]["total"] for res, _ in items]))
                ax.axhline(total, color="#52514e", linestyle=":", linewidth=1, label="total")
                if nts:
                    ax.axvline(float(np.mean(nts)), color="#0b0b0b", linestyle="--", linewidth=1, label="N_train")
                ax.set_xscale("log", base=2)
                ax.set_title(f"{g}\n{fkey}", fontsize=8)
                if r == len(levels) - 1:
                    ax.set_xlabel("budget N")
                if col == 0:
                    ax.set_ylabel("contribution to J1 - J0")
        axes[0][0].legend(fontsize=7, loc="best")
        fig.subplots_adjust(hspace=0.5, wspace=0.3)
        fig.suptitle(f"Budget curve: {arm} (seed mean, band = min-max over seeds)", fontsize=10)
        path = fig_dir / f"budget_{arm}.png"
        _save(fig, path)
        plt.close(fig)
        paths.append(path)
    return paths


# --------------------------------------------------------------------------- markdown


def _num(x: float | None, fmt: str = "+.4f") -> str:
    return "n/a" if x is None else format(x, fmt)


def _markdown(summary: dict, figures: list[Path], out: Path) -> str:
    c = summary["c"]
    lines = [
        "# Budget curve",
        "",
        "Generated by `capsplit.harness.budget.build_budget_curve`; do not edit by hand.",
        "",
        f"Source: `{summary['results_dir']}`, checkpoint: {summary['step']}, c = {c:g}.",
        "",
        "## What this measures",
        "",
        f"A correct solution counts as *reachable* at budget N when the base model already finds it with "
        f"probability at least {c:g}/N, so N tries would usually find it. Training changes each solution's "
        "probability; **rel(N)** adds up the change on reachable solutions (better reliability on what the base "
        "could already do) and **cap(N)** the change on unreachable ones (new capability at that budget). "
        "The two always add up to the total gain J1 - J0 (checked to 1e-12 at every N).",
        "",
        "As N grows more solutions become reachable, so gain moves from cap to rel. **N\\*** is the smallest "
        "budget on the grid at which rel holds at least half of the gain: above N\\*, most of what training "
        "added was already within reach of the base given that many tries. rel(N) only rises with N when no "
        "solution lost probability; otherwise it can move either way, so N\\* is read off the grid literally.",
        "",
        "**cap at N_train** is cap evaluated at the number of episodes the arm sampled per training prompt: "
        "the part of the gain that the base could not have found with the same sampling the arm used.",
        "",
        f"Grid: {len(summary['grid'])} budgets from {summary['grid'][0]} to {summary['grid'][-1]} "
        "(powers of two by default).",
        "",
    ]
    entries = summary["entries"]
    if any(e["resolvable_n_max"] for e in entries):
        lines += [
            "**Sampled-backend runs**: reachability there was decided from a finite base sample, so the grid is "
            "cut at the largest resolvable N (95% Clopper-Pearson bound for zero successes); nothing is "
            "extrapolated beyond it.",
            "",
        ]
    lines += ["## Results", "", "N\\* is the seed mean over seeds where it is defined (k/n = defined/seeds). "
              "cap at N_train: seed mean [min, max].", ""]
    for arm in dict.fromkeys(e["arm"] for e in entries):
        lines += [f"### {arm}", ""]
        lines += ["| factors | group | total gain | N* (k/n) | cap at N_train [min, max] | N_train | N_train source |",
                  "|---|---|---|---|---|---|---|"]
        for e in (e for e in entries if e["arm"] == arm):
            ns, cap, nt = e["n_star"], e["cap_at_n_train"], e["n_train"]
            n_star = "n/a" if ns["seed_mean"] is None else f"{ns['seed_mean']:.4g}"
            cap_s = "n/a" if cap["mean"] is None else f"{cap['mean']:+.4f} [{cap['min']:+.4f}, {cap['max']:+.4f}]"
            if cap["notes"] and cap["mean"] is None:
                cap_s += " (" + "; ".join(cap["notes"]) + ")"
            nt_s = "unknown" if nt["mean"] is None else f"{nt['mean']:.6g}"
            if e["resolvable_n_max"]:
                nt_s += f" (grid cut at N <= {e['resolvable_n_max']})"
            lines.append(f"| {e['factors']} | {e['group']} | {_num(e['total']['mean'])} | {n_star} "
                         f"({ns['n_defined']}/{len(ns['per_seed'])}) | {cap_s} | {nt_s} | {nt['source']} |")
        lines.append("")
    if summary["skipped"]:
        lines += ["## Skipped runs", ""]
        lines += [f"- {s['arm']} {s['factors']} seed {s['seed']}: {s['reason']}" for s in summary["skipped"]]
        lines.append("")
    if figures:
        lines += ["## Figures", ""]
        lines += [f"![{p.stem}]({p.relative_to(out).as_posix()})" for p in figures]
        lines.append("")
    return "\n".join(lines)
