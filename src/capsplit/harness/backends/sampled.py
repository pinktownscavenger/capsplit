"""Sampled measurement backend: the reliability/capability split from Monte Carlo episodes.

For tasks too large to enumerate. Only `Policy.sample` and `Task.verify` are required.

Per prompt i (all draws use torch generators seeded from `seed`, the prompt id and a role):

1. **Classify** (prepare): draw `m_classify` base episodes. q̂0(s) = fraction of them whose
   solution is s. A correct solution is *reachable* at (N, c) iff q̂0(s) >= c/N; a solution
   never seen in this sample is unreachable. This sample is used for nothing else, so the
   classification is independent of the mass estimates below. Its resolution is ~1/m_classify:
   keep m_classify >> N/c.
2. **Masses**: draw a fresh `m` base episodes (prepare) and `m` trained episodes (measure).
   With R the reachable set, a = fraction of episodes correct with solution in R, b = fraction
   correct with solution not in R:

       j0, j1 = fractions correct;  delta_rel = a1 - a0;  delta_cap = b1 - b0

   so delta_rel + delta_cap = j1 - j0 exactly. The trained sample reuses the base sample's
   generator seed (common random numbers): at step 0 (trained == base) every gain is exactly
   0, and positive coupling only makes the independent bootstrap below conservative.
3. Group values are means over the group's prompts, as the exact backend.

**Solution identity**: `task.solution_of(pid, actions)` if the task has it (the grid task:
the loop-erased route); else the row of `task.table(pid)` with those actions; else the action
sequence itself. Incorrect episodes (verify False) have no solution.

**Confidence intervals** (percentile bootstrap, `n_boot` replicates, level 1 - alpha) for j0,
j1, delta_rel and delta_cap. Each prompt's three samples are summarised as counts over
categories (each solution seen, plus "incorrect"); a replicate redraws all three count vectors
from multinomials with the observed frequencies (= resampling episodes with replacement within
the prompt), *including the classification sample*, so reachability is re-decided in every
replicate and classification noise is part of the interval. Two schemes:

* ``extra["ci"]`` (two-level): each replicate also resamples the group's prompts with
  replacement, then averages the resampled prompts' replicate values. It targets the group
  mean over a population of prompts like these, and is conservative for the fixed prompts.
* ``extra["ci_episodes"]`` (episodes only): the fixed prompts' replicate values are averaged.
  This targets exactly what the exact backend computes (the group's own prompts).

**E_N**: not computable from samples in general. ``per_step`` is estimated when the base can be
tempered: the policy has a `tempered(beta)` method, or a torch `logits` Parameter (then a clone
with logits * beta). For each beta in `betas` we draw `m_temper` tempered-base episodes per
prompt, estimate J_i(beta), and take E_N = mean pass@N(ĵ1_i) - max_beta mean pass@N(ĵ_i(beta))
(one shared beta per group, H13; grid only, no refinement). The max over noisy estimates is
biased upward (E_N biased low by O(1/sqrt(m_temper))); the coarse grid misses the sup (E_N
biased high, unlike the exact backend's golden-section refinement). ``sequence`` (and so
``both``) need the normaliser over all trajectories and are reported as NaN. The label uses
the per-step E_N only (``extra["label_basis"]``), which can overstate "beyond temperature".
Cost: len(betas) * m_temper extra base episodes per prompt, once in prepare.

Params: m (4096), m_classify (4096), m_temper (= m), betas (11 log-spaced values in [0.25, 8],
including 1; [] skips E_N), seed (0), n_boot (1000), alpha (0.05), chunk (16384 episodes per sample call).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Sequence

import numpy as np
import torch

from ..interfaces import Backend, GroupMetrics, MeasureSpec, Policy, Task
from ..registry import BACKENDS

_PARAMS = {"m", "m_classify", "m_temper", "betas", "seed", "n_boot", "alpha", "chunk"}
DEFAULT_BETAS = tuple(round(float(b), 12) for b in np.geomspace(0.25, 8.0, 11))  # includes 1.0
TEMPERATURE_EPS = 1e-9
METRICS = ("j0", "j1", "delta_rel", "delta_cap")
_ROLE_CLASSIFY, _ROLE_MASS, _ROLE_TEMPER = 1, 2, 3


def _generator(seed: int, pid: int, role: int, k: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(int((seed * 1_000_003 + pid * 7_919 + role * 104_729 + k * 15_485_863) % 2**63))


def solution_keyer(task: Task, pid: int) -> Callable[[Sequence[int]], Hashable | None]:
    """actions -> solution key, or None if the episode is incorrect (see the module docstring)."""
    solution_of = getattr(task, "solution_of", None)
    if callable(solution_of):
        return lambda actions: solution_of(pid, actions)
    table = task.table(pid)
    if table is not None:
        pad = table.n_states * table.n_actions
        lookup = {}
        for row, sid in zip(table.index, table.solution_ids):
            if sid >= 0:
                lookup[tuple(int(e) % table.n_actions for e in row if e != pad)] = int(sid)

        def from_table(actions: Sequence[int]) -> Hashable | None:
            sid = lookup.get(tuple(int(a) for a in actions))
            return None if sid is None or not task.verify(pid, actions) else table.solutions[sid]

        return from_table
    return lambda actions: tuple(int(a) for a in actions) if task.verify(pid, actions) else None


def tempered_policy(policy: Policy, beta: float) -> Policy | None:
    """pi^beta per step, or None if the policy exposes no way to temper it."""
    tempered = getattr(policy, "tempered", None)
    if callable(tempered):
        return tempered(beta)
    logits = getattr(policy, "logits", None)
    if isinstance(logits, torch.nn.Parameter):
        clone = policy.clone()
        with torch.no_grad():
            clone.logits.mul_(float(beta))  # type: ignore[attr-defined]
        return clone
    return None


def pass_at_n(j, n: int):
    return 1.0 - (1.0 - np.asarray(j, dtype=float)) ** n


def _label(beyond_reach: bool, e_n: float) -> str:
    if math.isnan(e_n):
        return "unmeasured"
    beyond_temperature = e_n > TEMPERATURE_EPS
    if beyond_reach and beyond_temperature:
        return "capability"
    if beyond_reach:
        return "compounded_sharpening"
    if beyond_temperature:
        return "reweighting"
    return "pure_sharpening"


@dataclass
class SampledCache:
    pids: tuple
    keyers: dict  # pid -> keyer
    classify: dict  # pid -> Counter of solution keys (None = incorrect)
    base: dict  # pid -> Counter
    temper_js: dict  # pid -> {beta: ĵ}; empty if E_N is not estimated
    e_n_note: str


@BACKENDS.register("sampled")
class SampledBackend(Backend):
    def __init__(self, **params: Any):
        unknown = set(params) - _PARAMS
        if unknown:
            raise ValueError(f"sampled backend: unknown params {sorted(unknown)}")
        super().__init__(**params)
        self.m = int(params.get("m", 4096))
        self.m_classify = int(params.get("m_classify", 4096))
        self.m_temper = int(params.get("m_temper", self.m))
        self.betas = tuple(float(b) for b in params.get("betas", DEFAULT_BETAS))
        self.seed = int(params.get("seed", 0))
        self.n_boot = int(params.get("n_boot", 1000))
        self.alpha = float(params.get("alpha", 0.05))
        self.chunk = int(params.get("chunk", 16384))
        if min(self.m, self.m_classify, self.m_temper, self.n_boot, self.chunk) < 1:
            raise ValueError("m, m_classify, m_temper, n_boot and chunk must be positive")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")

    # ------------------------------------------------------------------ sampling

    def _counts(self, task: Task, policy: Policy, pid: int, m: int, generator: torch.Generator, keyer) -> Counter:
        counts: Counter = Counter()
        with torch.no_grad():
            for start in range(0, m, self.chunk):
                rollout = policy.sample(task, [pid] * min(self.chunk, m - start), generator)
                counts.update(keyer(actions) for actions in rollout.actions)
        return counts

    # ------------------------------------------------------------------ Backend API

    def prepare(self, task: Task, base: Policy, spec: MeasureSpec) -> SampledCache:
        pids = tuple(sorted({int(p) for ids in spec.groups.values() for p in ids}))
        keyers = {pid: solution_keyer(task, pid) for pid in pids}
        classify: dict = {}
        fresh: dict = {}
        temper_js: dict = {}
        for pid in pids:
            classify[pid] = self._counts(task, base, pid, self.m_classify, _generator(self.seed, pid, _ROLE_CLASSIFY), keyers[pid])
            fresh[pid] = self._counts(task, base, pid, self.m, _generator(self.seed, pid, _ROLE_MASS), keyers[pid])
        note = ""
        if not self.betas:
            note = "E_N not estimated (betas is empty)"
        else:
            tempered = {beta: tempered_policy(base, beta) for beta in self.betas}
            if any(p is None for p in tempered.values()):
                note = f"E_N not estimated: {type(base).__name__} exposes no per-step tempering"
            else:
                for pid in pids:
                    temper_js[pid] = {}
                    for k, (beta, policy) in enumerate(tempered.items()):
                        counts = self._counts(task, policy, pid, self.m_temper, _generator(self.seed, pid, _ROLE_TEMPER, k), keyers[pid])  # type: ignore[arg-type]
                        temper_js[pid][beta] = 1.0 - counts[None] / self.m_temper
                note = (
                    f"per_step E_N from {self.m_temper} tempered-base episodes per prompt at betas {list(self.betas)} "
                    "(max over noisy estimates: biased low); sequence/both not estimable from samples (NaN)"
                )
        return SampledCache(pids, keyers, classify, fresh, temper_js, note)

    def measure(self, task: Task, base: Policy, trained: Policy, spec: MeasureSpec, cache: SampledCache) -> list[GroupMetrics]:
        rng = np.random.default_rng([self.seed, 2027])
        points: dict = {}  # pid -> per-(n, c) point values
        boots: dict = {}  # pid -> per-(n, c) (n_boot, 4) replicate values
        crossed: dict = {}  # (pid, n, c) -> number of crossed solutions
        j1s: dict = {}
        lines = sorted({(n, c) for n in spec.ns for c in spec.cs})
        for pid in cache.pids:
            trained_counts = self._counts(task, trained, pid, self.m, _generator(self.seed, pid, _ROLE_MASS), cache.keyers[pid])
            keys = sorted(
                {k for counter in (cache.classify[pid], cache.base[pid], trained_counts) for k in counter if k is not None},
                key=repr,
            )
            cc = np.array([cache.classify[pid][k] for k in keys], dtype=float)
            bb = np.array([cache.base[pid][k] for k in keys] + [cache.base[pid][None]], dtype=float)
            tt = np.array([trained_counts[k] for k in keys] + [trained_counts[None]], dtype=float)
            c_class = self.m_classify - cc.sum()  # classification episodes that were incorrect
            cc_boot = rng.multinomial(self.m_classify, np.r_[cc, c_class] / self.m_classify, size=self.n_boot)[:, :-1]
            bb_boot = rng.multinomial(self.m, bb / self.m, size=self.n_boot)
            tt_boot = rng.multinomial(self.m, tt / self.m, size=self.n_boot)
            j1s[pid] = 1.0 - tt[-1] / self.m
            points[pid], boots[pid] = {}, {}
            for n, c in lines:
                line = c / n
                reach = cc / self.m_classify >= line
                points[pid][(n, c)] = _split(reach, bb, tt, self.m)
                crossed[(pid, n, c)] = int(((cc / self.m_classify < line) & (tt[:-1] / self.m >= line)).sum())
                reach_boot = cc_boot / self.m_classify >= line
                boots[pid][(n, c)] = _split_boot(reach_boot, bb_boot, tt_boot, self.m)

        out = []
        q = [100 * self.alpha / 2, 100 * (1 - self.alpha / 2)]
        for g_index, (group, group_ids) in enumerate(spec.groups.items()):
            ids = [int(p) for p in group_ids]
            resample = np.random.default_rng([self.seed, 2028, g_index]).integers(0, len(ids), size=(self.n_boot, len(ids)))
            j1_list = [j1s[p] for p in ids]
            # E_1 (decision H39): success-rate gap against the best shared-beta tempered base.
            if cache.temper_js:
                j_by_beta = {b: float(np.mean([cache.temper_js[p][b] for p in ids])) for b in self.betas}
                e1_step = float(np.mean(j1_list)) - max(j_by_beta.values())
            else:
                e1_step = float("nan")
            for n in spec.ns:
                trained_pass = float(np.mean(pass_at_n(j1_list, n)))
                frontier: dict = {}
                if cache.temper_js:
                    by_beta = {b: float(np.mean(pass_at_n([cache.temper_js[p][b] for p in ids], n))) for b in self.betas}
                    best_beta = max(by_beta, key=lambda b: by_beta[b])
                    frontier = {"best_pass": by_beta[best_beta], "best_beta": best_beta}
                    e_step = trained_pass - by_beta[best_beta]
                else:
                    e_step = float("nan")
                excess = {"per_step": e_step, "sequence": float("nan"), "both": float("nan")}
                for c in spec.cs:
                    per_prompt = np.array([points[p][(n, c)] for p in ids])  # (P, 4)
                    reps = np.stack([boots[p][(n, c)] for p in ids])  # (P, n_boot, 4)
                    episodes = reps.mean(axis=0)  # (n_boot, 4)
                    two_level = reps[resample, np.arange(self.n_boot)[:, None]].mean(axis=1)  # (n_boot, 4)
                    means = per_prompt.mean(axis=0)
                    n_crossed = sum(crossed[(p, n, c)] for p in ids)
                    extra = {
                        "backend": "sampled",
                        "ci": {k: [float(x) for x in np.percentile(two_level[:, i], q)] for i, k in enumerate(METRICS)},
                        "ci_episodes": {k: [float(x) for x in np.percentile(episodes[:, i], q)] for i, k in enumerate(METRICS)},
                        "ci_scheme": "percentile bootstrap; ci: prompts and episodes (incl. classification); ci_episodes: episodes only",
                        "alpha": self.alpha,
                        "m": self.m,
                        "m_classify": self.m_classify,
                        "beyond_reach": n_crossed > 0,
                        "crossed_count": n_crossed,
                        "frontier": {"per_step": frontier} if frontier else {},
                        "e_n_note": cache.e_n_note,
                        "label_basis": "per_step E_1 only (H39); label_en uses per_step E_N",
                        "e1": {"per_step": e1_step, "sequence": float("nan"), "both": float("nan")},
                        "label_en": _label(n_crossed > 0, e_step),
                        "j0": [float(x) for x in per_prompt[:, 0]],
                        "j1": [float(x) for x in per_prompt[:, 1]],
                        "delta_rel": [float(x) for x in per_prompt[:, 2]],
                        "delta_cap": [float(x) for x in per_prompt[:, 3]],
                    }
                    out.append(
                        GroupMetrics(
                            group=group,
                            n=n,
                            c=c,
                            n_prompts=len(ids),
                            j0=float(means[0]),
                            j1=float(means[1]),
                            delta_rel=float(means[2]),
                            delta_cap=float(means[3]),
                            excess=excess,
                            label=_label(n_crossed > 0, e1_step),
                            extra=extra,
                        )
                    )
        return out


def _split(reach: np.ndarray, base: np.ndarray, trained: np.ndarray, m: int) -> tuple:
    """(j0, j1, delta_rel, delta_cap) from count vectors whose last entry is 'incorrect'."""
    j0, j1 = 1.0 - base[-1] / m, 1.0 - trained[-1] / m
    rel = (trained[:-1][reach].sum() - base[:-1][reach].sum()) / m
    return (j0, j1, rel, (j1 - j0) - rel)


def _split_boot(reach: np.ndarray, base: np.ndarray, trained: np.ndarray, m: int) -> np.ndarray:
    """(n_boot, 4) replicate values; `reach` is (n_boot, K), counts are (n_boot, K + 1)."""
    j0, j1 = 1.0 - base[:, -1] / m, 1.0 - trained[:, -1] / m
    rel = ((trained[:, :-1] - base[:, :-1]) * reach).sum(axis=1) / m
    return np.stack([j0, j1, rel, (j1 - j0) - rel], axis=1)
