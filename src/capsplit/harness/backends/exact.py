"""Exact measurement backend: the split, E_N and the 2x2 label from enumerated trajectory tables.

Definitions (docs/proofs.md section 1, docs/DECISIONS.md H12 and H13):

* For prompt i, q0(s) and q1(s) are the base and trained probabilities of correct solution s
  (`engine.solution_probs` on exact trajectory probabilities).
* s is reachable at budget N and multiplier c iff q0(s) >= c/N. delta_rel_i sums q1 - q0 over
  reachable s and delta_cap_i over unreachable s. Group values are means over the group's
  prompts, so delta_rel + delta_cap = J1 - J0 exactly (Proposition 1).
* E_N[family] = mean_i pass@N(J1_i) - sup_beta mean_i pass@N(J_i under the family at beta),
  with one beta shared by every prompt in the group (H13). Families: "per_step" (multiply the
  base's tempering logits by beta), "sequence" (P0(tau)^beta renormalised over all trajectories)
  and "both" (sup over the two). The sup is a grid search over `tempering.DEFAULT_BETAS`, then
  golden-section refinement in log beta around the best grid points, as `tempering.frontier`.
* E_1 (`extra["e1"]`) is the same comparison at budget 1, i.e. on success rate J:
  mean J1 - sup_beta mean J(tempered base), one shared beta. It is computed for every group
  whether or not 1 is among the measured budgets.
* Label (H12, H39): beyond reach iff some correct s has q0(s) < c/N <= q1(s) on some prompt of
  the group; beyond temperature iff E_1["both"] > 1e-9. Using E_1 rather than E_N matches
  `evaluate.evaluate` (whose single-task label uses the J gap) and avoids a blind spot: once the
  base's pass@N saturates on reachable prompts, E_N cannot see any change there. The E_N-based
  label is kept in `extra["label_en"]` and the D2 label (`evaluate.label_cell` with the J gap)
  in `extra["d2_label"]`.

`prepare` enumerates the base once (tables, q0, frontier per (group, N, family)); `measure`
only needs the trained policy's trajectory probabilities, so it is cheap enough per checkpoint.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from capsplit.engine import TrajectoryTable, solution_probs, table_success_prob
from capsplit.evaluate import label_cell
from capsplit.tempering import _REFINE_POINTS, DEFAULT_BETAS, _golden_section

from ..interfaces import Backend, GroupMetrics, MeasureSpec, Policy, Task
from ..registry import BACKENDS

FAMILIES = ("per_step", "sequence")
TEMPERATURE_EPS = 1e-9  # beyond temperature iff E_N["both"] > this
SATURATED_BELOW = 0.05  # saturated iff 1 - best tempered pass@N < this
_MAX_GATHER = 20_000_000  # elements per batched (prompts, T, H) gather


# --------------------------------------------------------------------------- pure functions


def pass_at_n(j, n: int):
    """pass@N = 1 - (1 - J)^N, elementwise on arrays."""
    return 1.0 - (1.0 - np.asarray(j, dtype=float)) ** n


def sequence_probs_from_probs(probs: np.ndarray, beta: float) -> np.ndarray:
    """Sequence-level tempering from trajectory probabilities: P(tau)^beta / sum P^beta.

    Zero-probability trajectories stay at zero (the sum runs over the support). Equals
    `tempering.sequence_probs(table, logits, beta)` when probs = trajectory_probs(table, logits).
    """
    probs = np.asarray(probs, dtype=float)
    with np.errstate(divide="ignore"):
        log_p = beta * np.log(probs)
    out = np.exp(log_p - log_p.max(axis=-1, keepdims=True))
    return out / out.sum(axis=-1, keepdims=True)


@dataclass(frozen=True)
class Split:
    """Per-prompt split at one (N, c). `crossed` are the correct solutions (by index into
    `table.solutions`) with q0 < c/N <= q1; `crossed_mass` is their total q1."""

    j0: float
    j1: float
    delta_rel: float
    delta_cap: float
    crossed: tuple
    crossed_mass: float


def split_from_probs(table: TrajectoryTable, probs0: np.ndarray, probs1: np.ndarray, n: int, c: float = 1.0) -> Split:
    """The reliability/capability split of one prompt from two trajectory distributions."""
    q0 = solution_probs(table, np.asarray(probs0, dtype=float))
    q1 = solution_probs(table, np.asarray(probs1, dtype=float))
    line = c / n
    reachable = q0 >= line
    change = q1 - q0
    crossed = (q0 < line) & (q1 >= line)
    return Split(
        j0=table_success_prob(table, np.asarray(probs0, dtype=float)),
        j1=table_success_prob(table, np.asarray(probs1, dtype=float)),
        delta_rel=float(change[reachable].sum()),
        delta_cap=float(change[~reachable].sum()),
        crossed=tuple(int(i) for i in np.flatnonzero(crossed)),
        crossed_mass=float(q1[crossed].sum()),
    )


def h12_label(beyond_reach: bool, e_n_both: float, eps: float = TEMPERATURE_EPS) -> str:
    """The 2x2 cell under decision H12."""
    beyond_temperature = e_n_both > eps
    if beyond_reach and beyond_temperature:
        return "capability"
    if beyond_reach:
        return "compounded_sharpening"
    if beyond_temperature:
        return "reweighting"
    return "pure_sharpening"


@dataclass(frozen=True)
class FamilyFrontier:
    """Best mean pass@N over one shared beta for one (group, N, family); `best_j` is the mean J there."""

    best_pass: float
    best_j: float
    best_beta: float
    boundary_hit: bool


def shared_beta_frontier(
    objective: Callable[[float], tuple], betas=DEFAULT_BETAS, refine: bool = True
) -> FamilyFrontier:
    """Maximise `objective(beta) = (mean pass@N, mean J)` over beta, comparing lexicographically
    (mean J only breaks ties, which happen when pass@N rounds to 1.0). Grid first, then
    golden-section refinement in log beta between the neighbours of the best `_REFINE_POINTS`
    grid points, as `tempering.frontier`."""
    betas = np.unique(np.r_[np.asarray(betas, dtype=float), 1.0])
    values = [tuple(objective(float(b))) for b in betas]
    order = sorted(range(len(betas)), key=lambda i: values[i], reverse=True)
    best_value: tuple = (-1.0, -1.0)
    best_beta = 1.0
    for i in order[: _REFINE_POINTS if refine else 1]:
        beta: float = float(betas[i])
        value: tuple = values[i]
        if refine:
            low, high = betas[max(i - 1, 0)], betas[min(i + 1, len(betas) - 1)]
            # _golden_section only compares values with >, so tuples work (its hint says float).
            found: Any = _golden_section(lambda x: tuple(objective(math.exp(x))), math.log(low), math.log(high))  # type: ignore[arg-type,return-value]
            log_beta, refined = found
            if refined > value:
                beta, value = math.exp(log_beta), refined
        if value > best_value:
            best_value, best_beta = value, beta
    boundary_hit = bool(best_beta <= betas[0] * 1.001 or best_beta >= betas[-1] * 0.999)
    return FamilyFrontier(float(best_value[0]), float(best_value[1]), best_beta, boundary_hit)


# --------------------------------------------------------------------------- batched base tempering


def _stacked_log_policy(logits: np.ndarray, beta: float) -> np.ndarray:
    """(P, S, A) logits -> (P, S*A + 1) flattened per-step log pi^beta, with a 0 for padding."""
    z = beta * logits
    z = z - z.max(axis=2, keepdims=True)
    z = z - np.log(np.exp(z).sum(axis=2, keepdims=True))
    flat = z.reshape(z.shape[0], -1)
    return np.concatenate([flat, np.zeros((flat.shape[0], 1))], axis=1)


@dataclass
class _Batch:
    """Prompts whose tables share one trajectory index, so they can be tempered together."""

    index: np.ndarray  # (T, H)
    pids: list = field(default_factory=list)
    correct: list = field(default_factory=list)  # (T,) bool per prompt
    logits: list = field(default_factory=list)  # (S, A) per prompt, or None
    log_p0: list = field(default_factory=list)  # (T,) per prompt

    def finalise(self) -> None:
        self.correct_arr = np.stack(self.correct)
        self.log_p0_arr = np.stack(self.log_p0)
        self.logits_arr = None if any(l is None for l in self.logits) else np.stack(self.logits).astype(float)
        self.row = {pid: k for k, pid in enumerate(self.pids)}


class _Tempering:
    """J_i(beta) for many prompts at once, under either family."""

    def __init__(self, batches: list):
        self.batches = batches
        self.where = {pid: (b, batch.row[pid]) for b, batch in enumerate(batches) for pid in batch.pids}

    def has_per_step(self, pids: Sequence[int]) -> bool:
        return all(self.batches[self.where[p][0]].logits_arr is not None for p in pids)

    def js(self, family: str, beta: float, pids: Sequence[int]) -> np.ndarray:
        out = np.empty(len(pids))
        by_batch: dict = {}
        for k, pid in enumerate(pids):
            b, row = self.where[pid]
            by_batch.setdefault(b, ([], []))
            by_batch[b][0].append(k)
            by_batch[b][1].append(row)
        for b, (positions, rows) in by_batch.items():
            batch = self.batches[b]
            rows_arr = np.asarray(rows)
            if family == "sequence":
                log_p = beta * batch.log_p0_arr[rows_arr]
                w = np.exp(log_p - log_p.max(axis=1, keepdims=True))
                j = (w * batch.correct_arr[rows_arr]).sum(axis=1) / w.sum(axis=1)
            elif family == "per_step":
                j = np.empty(len(rows_arr))
                chunk = max(1, _MAX_GATHER // max(1, batch.index.size))
                for start in range(0, len(rows_arr), chunk):
                    sel = rows_arr[start : start + chunk]
                    flat = _stacked_log_policy(batch.logits_arr[sel], beta)  # type: ignore[index]
                    probs = np.exp(flat[:, batch.index].sum(axis=2))
                    j[start : start + chunk] = (probs * batch.correct_arr[sel]).sum(axis=1)
            else:
                raise ValueError(f"unknown tempering family {family!r}")
            out[np.asarray(positions)] = j
        return out


# --------------------------------------------------------------------------- the backend


@dataclass
class ExactCache:
    """What `prepare` computes from the base: per-prompt arrays and the frontier per (group, N, family)."""

    pids: tuple  # union of the spec's groups, sorted
    tables: dict  # pid -> TrajectoryTable
    probs0: dict  # pid -> (T,) base trajectory probabilities
    q0: dict  # pid -> (n_solutions,) base solution probabilities
    j0: dict  # pid -> float
    frontiers: dict  # (group, n) -> {family: FamilyFrontier}
    tempering: Any = None


def _table(task: Task, pid: int) -> TrajectoryTable:
    table = task.table(pid)
    if table is None:
        raise ValueError(f"{type(task).__name__} has no exact table for prompt {pid}")
    return table


@BACKENDS.register("exact")
class ExactBackend(Backend):
    """Exact split, E_N and label for tasks with `Task.table`. Params: `betas` (default grid),
    `refine` (golden-section refinement, default True)."""

    def prepare(self, task: Task, base: Policy, spec: MeasureSpec) -> ExactCache:
        betas = self.params.get("betas", DEFAULT_BETAS)
        refine = bool(self.params.get("refine", True))
        pids = tuple(sorted({int(p) for ids in spec.groups.values() for p in ids}))
        tables, probs0, q0, j0 = {}, {}, {}, {}
        batches: list = []
        for pid in pids:
            table = _table(task, pid)
            probs = np.asarray(base.trajectory_probs(task, pid), dtype=float)
            _check_probs(table, probs, "base")
            tables[pid], probs0[pid] = table, probs
            q0[pid] = solution_probs(table, probs)
            j0[pid] = table_success_prob(table, probs)
            logits = base.tempering_logits(task, pid)
            batch = next((b for b in batches if b.index is table.index or (
                b.index.shape == table.index.shape and np.array_equal(b.index, table.index))), None)
            if batch is None:
                batch = _Batch(table.index)
                batches.append(batch)
            batch.pids.append(pid)
            batch.correct.append(table.solution_ids >= 0)
            batch.logits.append(None if logits is None else np.asarray(logits, dtype=float))
            with np.errstate(divide="ignore"):
                batch.log_p0.append(np.log(probs))
        for batch in batches:
            batch.finalise()
        tempering = _Tempering(batches)

        frontiers: dict = {}
        for group, group_ids in spec.groups.items():
            ids = [int(p) for p in group_ids]
            families = [f for f in FAMILIES if f != "per_step" or tempering.has_per_step(ids)]
            grid_js = {f: {float(b): tempering.js(f, float(b), ids) for b in betas} for f in families}
            for n in sorted(set(spec.ns) | {1}):
                result = {}
                for family in families:

                    def objective(beta: float, family=family, n=n) -> tuple:
                        js = grid_js[family].get(beta)
                        if js is None:
                            js = tempering.js(family, beta, ids)
                        return float(np.mean(pass_at_n(js, n))), float(np.mean(js))

                    result[family] = shared_beta_frontier(objective, betas, refine)
                frontiers[(group, n)] = result
        return ExactCache(pids, tables, probs0, q0, j0, frontiers, tempering)

    def measure(
        self, task: Task, base: Policy, trained: Policy, spec: MeasureSpec, cache: ExactCache
    ) -> list[GroupMetrics]:
        q1, j1 = {}, {}
        for pid in cache.pids:
            table = cache.tables[pid]
            probs = np.asarray(trained.trajectory_probs(task, pid), dtype=float)
            _check_probs(table, probs, "trained")
            q1[pid] = solution_probs(table, probs)
            j1[pid] = table_success_prob(table, probs)
        e_n_valid_base = task.n_actions**task.horizon

        out = []
        for group, group_ids in spec.groups.items():
            ids = [int(p) for p in group_ids]
            # Flatten every correct solution of the group, with its owning prompt.
            owner = np.concatenate([np.full(len(cache.q0[p]), k) for k, p in enumerate(ids)]).astype(int)
            flat_q0 = np.concatenate([cache.q0[p] for p in ids])
            flat_q1 = np.concatenate([q1[p] for p in ids])
            change = flat_q1 - flat_q0
            j0_list = [cache.j0[p] for p in ids]
            j1_list = [j1[p] for p in ids]
            j0_mean, j1_mean = float(np.mean(j0_list)), float(np.mean(j1_list))
            m = len(ids)
            if (group, 1) not in cache.frontiers:
                raise KeyError(f"cache has no budget-1 frontier for group {group!r}; re-run prepare")
            front1 = cache.frontiers[(group, 1)]
            e1 = {f: j1_mean - fr.best_pass for f, fr in front1.items()}
            e1["both"] = j1_mean - max(fr.best_pass for fr in front1.values())
            for n in spec.ns:
                if (group, n) not in cache.frontiers:
                    raise KeyError(f"cache has no frontier for group {group!r} at N={n}; re-run prepare")
                frontier = cache.frontiers[(group, n)]
                trained_pass = float(np.mean(pass_at_n(j1_list, n)))
                excess = {f: trained_pass - fr.best_pass for f, fr in frontier.items()}
                # per_step first on exact ties, as tempering.frontier
                best_family = max(frontier, key=lambda f: (frontier[f].best_pass, frontier[f].best_j))
                best = frontier[best_family]
                excess["both"] = trained_pass - best.best_pass
                for c in spec.cs:
                    line = c / n
                    reachable = flat_q0 >= line
                    d_rel = np.bincount(owner, weights=np.where(reachable, change, 0.0), minlength=m)
                    d_cap = np.bincount(owner, weights=np.where(reachable, 0.0, change), minlength=m)
                    crossed = (flat_q0 < line) & (flat_q1 >= line)
                    crossed_mass = np.bincount(owner, weights=np.where(crossed, flat_q1, 0.0), minlength=m)
                    delta_rel, delta_cap = float(d_rel.mean()), float(d_cap.mean())
                    beyond_reach = bool(crossed.any())
                    extra = {
                        "best_beta": best.best_beta,
                        "best_family": best_family,
                        "boundary_hit": best.boundary_hit,
                        "saturated": bool(1.0 - best.best_pass < SATURATED_BELOW),
                        "frontier": {
                            f: {
                                "best_pass": fr.best_pass,
                                "best_j": fr.best_j,
                                "best_beta": fr.best_beta,
                                "boundary_hit": fr.boundary_hit,
                            }
                            for f, fr in frontier.items()
                        },
                        "beyond_reach": beyond_reach,
                        "e1": e1,
                        "label_en": h12_label(beyond_reach, excess["both"]),
                        "d2_label": label_cell(delta_cap, e1["both"], line),
                        "crossed_count": int(crossed.sum()),
                        "crossed_mass": float(crossed_mass.mean()),  # mean over prompts of q1 on crossed solutions
                        "e_n_valid": bool(e_n_valid_base >= 100 * n),
                        "n_correct_solutions": int(len(flat_q0)),
                        "j0": j0_list,
                        "j1": j1_list,
                        "delta_rel": [float(x) for x in d_rel],
                        "delta_cap": [float(x) for x in d_cap],
                    }
                    out.append(
                        GroupMetrics(
                            group=group,
                            n=n,
                            c=c,
                            n_prompts=m,
                            j0=j0_mean,
                            j1=j1_mean,
                            delta_rel=delta_rel,
                            delta_cap=delta_cap,
                            excess=excess,
                            label=h12_label(beyond_reach, e1["both"]),
                            extra=extra,
                        )
                    )
        return out


def _check_probs(table: TrajectoryTable, probs: np.ndarray, who: str) -> None:
    if probs.shape != (table.index.shape[0],):
        raise ValueError(f"{who} trajectory probs shape {probs.shape} != {(table.index.shape[0],)}")
    if (probs < 0).any() or abs(probs.sum() - 1.0) > 1e-6:
        raise ValueError(f"{who} trajectory probs are not a probability distribution")
