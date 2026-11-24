"""Sampled measurement for the LM experiment (C4, docs/lm_design.md "Measurement (C4)").

Pure functions on per-problem success counts (numpy arrays). Nothing here touches a model;
only `counts_from_samples` reads sample rows. All randomness comes from explicit numpy
Generators (or seeds turned into them).

Pipeline for one group of P problems, budget N, level 1 - alpha:

* **Classification** (set A only): two-sided Clopper-Pearson interval [lo, hi] on each
  problem's success rate. reachable iff lo >= 1/N; unreachable iff hi < 1/N; else ambiguous.
* **Split**: gain_i = p̂1_i - p̂0_i (p̂0 from set B, p̂1 from `eval`). A class's contribution is
  the sum of its problems' gains divided by P, so the three contributions sum to mean(gain)
  (checked to 1e-12).
* **Temperature baseline** (set C): random half split; on half 1 pick T* maximising the base's
  mean success (for E_1) and, separately, mean pass@N (for E_N); evaluate trained - base(T*) on
  half 2; swap and average; repeat `n_splits` times.
  pass@N: the unbiased estimator 1 - C(m-c, N)/C(m, N) when trial counts are given and every
  problem (trained and every T) has m >= N trials; otherwise the plug-in 1 - (1 - p̂)^N, which
  is biased upward for the noisier side. The result's `pass_at_n_estimator` says which.
* **Bootstrap**: resample problems with replacement; in each resample redraw every resampled
  problem's set-A count ~ Binomial(m_A, observed rate_A) and reclassify, so classification
  noise enters the interval. Percentile intervals at level 1 - alpha.
* **Null check**: the same bootstrap with p1 := the base `null` set; passes iff the total's
  interval contains 0.
* **Budget curve**: the split at each N of a grid, reclassifying per N, only for
  N <= floor(1/u0) (`budget.resolvable_n_max`, u0 = the zero-success upper bound for the
  smallest set-A size); larger N cannot be resolved by set A.
* **GroupMetrics adapter** (label rule H12 + H39, adapted to samples):
  beyond_reach iff some problem classified *unreachable* (set A) has a trained Clopper-Pearson
  lower bound >= 1/N (from its own eval counts), i.e. it is confidently reachable after
  training; beyond_temperature iff the bootstrap interval of E_1 has lower bound > 0.
  GroupMetrics has a two-way split: delta_cap = the unreachable contribution and
  delta_rel = reachable + ambiguous contributions (a capability claim needs evidence of
  unreachability); the three-way split is in extra["split"].
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

from capsplit.harness.backends.exact import h12_label
from capsplit.harness.budget import clopper_pearson_zero_upper, resolvable_n_max
from capsplit.harness.interfaces import GroupMetrics

CLASSES = ("reachable", "unreachable", "ambiguous")
REACHABLE, UNREACHABLE, AMBIGUOUS = 0, 1, 2
IDENTITY_TOL = 1e-12
_BISECT_ITERS = 200  # far more than needed for double precision; the loop stops early once converged
_CHUNK = 1 << 21  # matrix entries per CDF evaluation chunk


# --------------------------------------------------------------------------- counts


@dataclass(frozen=True)
class Counts:
    """Per-problem successes and trials (int arrays of equal length)."""

    successes: NDArray[np.int64]
    trials: NDArray[np.int64]

    def __post_init__(self) -> None:
        if self.successes.shape != self.trials.shape or self.successes.ndim != 1:
            raise ValueError("successes and trials must be 1-d arrays of the same length")
        if np.any(self.successes < 0) or np.any(self.successes > self.trials):
            raise ValueError("need 0 <= successes <= trials for every problem")

    def rates(self) -> NDArray[np.float64]:
        """successes / trials; every problem must have been sampled."""
        if np.any(self.trials < 1):
            missing = np.flatnonzero(self.trials < 1)[:10].tolist()
            raise ValueError(f"problems without samples (first ids: {missing})")
        return self.successes / self.trials

    def subset(self, ids: Sequence[int] | NDArray[np.int64]) -> Counts:
        ids = np.asarray(ids, dtype=np.int64)
        return Counts(self.successes[ids], self.trials[ids])


def counts_from_samples(rows: Iterable[Mapping[str, Any]], n_problems: int) -> Counts:
    """Successes and trials per problem from sample rows {"problem": id, "correct": bool, ...}."""
    successes = np.zeros(n_problems, dtype=np.int64)
    trials = np.zeros(n_problems, dtype=np.int64)
    for row in rows:
        pid = row["problem"]
        if isinstance(pid, bool) or not isinstance(pid, (int, np.integer)) or not 0 <= pid < n_problems:
            raise ValueError(f"sample row has problem id {pid!r}, expected an int in [0, {n_problems})")
        correct = row["correct"]
        if not isinstance(correct, (bool, np.bool_)):
            raise TypeError(f"sample row for problem {pid} has correct={correct!r}, expected a bool")
        trials[pid] += 1
        successes[pid] += int(correct)
    return Counts(successes, trials)


# --------------------------------------------------------------------------- Clopper-Pearson


@cache
def _log_factorials(m: int) -> NDArray[np.float64]:
    out = np.zeros(m + 1)
    if m:
        out[1:] = np.cumsum(np.log(np.arange(1, m + 1, dtype=float)))
    return out


def _bisect_tail(m: int, ks: NDArray[np.int64], target: float, upper_tail: bool) -> NDArray[np.float64]:
    """For each k, the p in [0, 1] where P(X <= k) = target (upper_tail False; decreasing in p)
    or P(X >= k) = target (upper_tail True; increasing in p), X ~ Binomial(m, p)."""
    lf = _log_factorials(m)
    j = np.arange(m + 1)
    log_binom = lf[m] - lf[j] - lf[m - j]
    lo = np.zeros(len(ks))
    hi = np.ones(len(ks))
    rows = max(1, _CHUNK // (m + 1))
    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        tail = np.empty(len(ks))
        for start in range(0, len(ks), rows):
            p = mid[start:start + rows, None]
            k = ks[start:start + rows, None]
            with np.errstate(divide="ignore"):
                log_pmf = log_binom + j * np.log(p) + (m - j) * np.log1p(-p)
            pmf = np.exp(log_pmf)
            mask = (j >= k) if upper_tail else (j <= k)
            tail[start:start + rows] = np.where(mask, pmf, 0.0).sum(axis=1)
        # upper tail grows with p; lower tail shrinks with p
        go_up = (tail < target) if upper_tail else (tail > target)
        new_lo = np.where(go_up, mid, lo)
        new_hi = np.where(go_up, hi, mid)
        if np.array_equal(new_lo, lo) and np.array_equal(new_hi, hi):
            break
        lo, hi = new_lo, new_hi
    return 0.5 * (lo + hi)


@cache
def clopper_pearson_table(m: int, alpha: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(lower, upper) two-sided level 1 - alpha Clopper-Pearson bounds for k = 0..m successes in m trials.

    lower(k) solves P(X >= k; p) = alpha/2, upper(k) solves P(X <= k; p) = alpha/2, by bisection
    on the exact binomial tail (computed in log space). k = 0 and k = m use the closed forms.
    """
    if m < 1:
        raise ValueError("m must be >= 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    ks = np.arange(m + 1)
    lower = np.zeros(m + 1)
    upper = np.ones(m + 1)
    if m > 1:
        lower[1:m] = _bisect_tail(m, ks[1:m], alpha / 2, upper_tail=True)
        upper[1:m] = _bisect_tail(m, ks[1:m], alpha / 2, upper_tail=False)
    u0 = clopper_pearson_zero_upper(m, alpha)
    upper[0] = u0
    lower[m] = 1.0 - u0
    lower.setflags(write=False)
    upper.setflags(write=False)
    return lower, upper


def clopper_pearson(successes: Any, trials: Any, alpha: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Elementwise (lower, upper) bounds; arrays of any shape."""
    k = np.asarray(successes, dtype=np.int64)
    m = np.broadcast_to(np.asarray(trials, dtype=np.int64), k.shape)
    if np.any(k < 0) or np.any(k > m):
        raise ValueError("need 0 <= successes <= trials")
    lower = np.empty(k.shape)
    upper = np.empty(k.shape)
    for size in np.unique(m):
        sel = m == size
        lo_tab, hi_tab = clopper_pearson_table(int(size), float(alpha))
        lower[sel] = lo_tab[k[sel]]
        upper[sel] = hi_tab[k[sel]]
    return lower, upper


def _classify_codes(successes: Any, trials: Any, n: float, alpha: float) -> NDArray[np.int64]:
    lower, upper = clopper_pearson(successes, trials, alpha)
    codes = np.full(lower.shape, AMBIGUOUS, dtype=np.int64)
    codes[lower >= 1.0 / n] = REACHABLE
    codes[upper < 1.0 / n] = UNREACHABLE
    return codes


def classify(successes_A: Any, trials_A: Any, n: float, alpha: float) -> NDArray[np.str_]:
    """"reachable" / "unreachable" / "ambiguous" per problem from its set-A counts."""
    return np.asarray(CLASSES)[_classify_codes(successes_A, trials_A, n, alpha)]


# --------------------------------------------------------------------------- split


def _codes_of(classes: Any) -> NDArray[np.int64]:
    arr = np.asarray(classes)
    if arr.dtype.kind in "iu":
        return arr.astype(np.int64)
    lookup = {name: i for i, name in enumerate(CLASSES)}
    try:
        return np.array([lookup[str(c)] for c in arr], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"unknown class {error.args[0]!r}; expected one of {CLASSES}") from error


def _split_rows(codes: NDArray[np.int64], gain: NDArray[np.float64]) -> NDArray[np.float64]:
    """(..., 4) array: reachable, unreachable, ambiguous contributions and total, over the last axis."""
    p = gain.shape[-1]
    parts = [np.where(codes == c, gain, 0.0).sum(axis=-1) / p for c in range(3)]
    total = gain.mean(axis=-1)
    return np.stack(parts + [total], axis=-1)


def split(classes: Any, p0: Any, p1: Any) -> dict:
    """Three-way split of the mean gain mean(p1 - p0) by class."""
    codes = _codes_of(classes)
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    if not (codes.shape == p0.shape == p1.shape) or codes.ndim != 1 or not len(codes):
        raise ValueError("classes, p0 and p1 must be non-empty 1-d arrays of the same length")
    gain = p1 - p0
    row = _split_rows(codes, gain)
    residual = row[:3].sum() - row[3]
    if abs(residual) > IDENTITY_TOL:
        raise AssertionError(f"split identity violated by {residual:.3e}")
    counts = {name: int((codes == c).sum()) for c, name in enumerate(CLASSES)}
    conditional = {
        name: (float(gain[codes == c].mean()) if counts[name] else None) for c, name in enumerate(CLASSES)
    }
    return {
        **{name: float(row[c]) for c, name in enumerate(CLASSES)},
        "total": float(row[3]),
        "n_problems": len(codes),
        "counts": counts,
        "conditional_means": conditional,
    }


# --------------------------------------------------------------------------- temperature baseline


def pass_at_n_plugin(p: Any, n: int) -> NDArray[np.float64]:
    """1 - (1 - p)^N from per-problem rates."""
    return 1.0 - (1.0 - np.asarray(p, dtype=float)) ** n


def pass_at_n_unbiased(successes: Any, trials: Any, n: int) -> NDArray[np.float64]:
    """1 - C(m - c, N) / C(m, N): unbiased for pass@N when m >= N."""
    c = np.asarray(successes, dtype=np.int64)
    m = np.broadcast_to(np.asarray(trials, dtype=np.int64), c.shape)
    if np.any(m < n):
        raise ValueError("the unbiased pass@N estimator needs trials >= N")
    lf = _log_factorials(int(m.max()))
    fails = m - c
    out = np.ones(c.shape)
    ok = fails >= n
    out[ok] = 1.0 - np.exp(lf[fails[ok]] - lf[fails[ok] - n] - lf[m[ok]] + lf[m[ok] - n])
    return out


def _successes_from(p: NDArray[np.float64], trials: NDArray[np.int64]) -> NDArray[np.int64]:
    s = np.rint(p * trials)
    if np.max(np.abs(s - p * trials), initial=0.0) > 1e-6:
        raise ValueError("rates are not successes / trials for the given trial counts")
    return s.astype(np.int64)


def _pass_metrics(
    p1: NDArray[np.float64],
    base: NDArray[np.float64],
    n: int,
    trials1: Any,
    trials_base: NDArray[np.int64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], str]:
    """Per-problem pass@N for the trained policy (P,) and each temperature (T, P), and the estimator name."""
    if trials1 is not None and trials_base is not None:
        t1 = np.broadcast_to(np.asarray(trials1, dtype=np.int64), p1.shape)
        tb = np.broadcast_to(trials_base, base.shape)
        if t1.min() >= n and tb.min() >= n:
            return (
                pass_at_n_unbiased(_successes_from(p1, t1), t1, n),
                pass_at_n_unbiased(_successes_from(base, tb), tb, n),
                "unbiased",
            )
    return pass_at_n_plugin(p1, n), pass_at_n_plugin(base, n), "plug_in"


def split_half_value(
    trained: NDArray[np.float64], base: NDArray[np.float64], h1: NDArray[np.int64], h2: NDArray[np.int64]
) -> tuple[float, int, int]:
    """Choose T on h1, evaluate trained - base(T) on h2; swap; average. Returns (value, T index from h1,
    T index from h2). Ties go to the first temperature. Symmetric in (h1, h2) by construction."""
    t_a = int(np.argmax(base[:, h1].mean(axis=1)))
    t_b = int(np.argmax(base[:, h2].mean(axis=1)))
    v_a = float(np.mean(trained[h2] - base[t_a, h2]))
    v_b = float(np.mean(trained[h1] - base[t_b, h1]))
    return 0.5 * (v_a + v_b), t_a, t_b


def temperature_baseline(
    p1: Any,
    base_by_T: Mapping[float, Any],
    n: int,
    n_splits: int,
    rng: np.random.Generator,
    *,
    trials1: Any = None,
    trials_by_T: Mapping[float, Any] | None = None,
) -> dict:
    """Split-half E_1 (success rate) and E_N (pass@N) of the trained policy over the best base temperature."""
    p1 = np.asarray(p1, dtype=float)
    temps = sorted(base_by_T)
    if not temps:
        raise ValueError("base_by_T is empty")
    base = np.stack([np.asarray(base_by_T[t], dtype=float) for t in temps])
    if base.shape[1:] != p1.shape or p1.ndim != 1 or len(p1) < 2:
        raise ValueError("need >= 2 problems and one rate per problem for the trained policy and every temperature")
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    trials_base = (
        np.stack([np.broadcast_to(np.asarray(trials_by_T[t], dtype=np.int64), p1.shape) for t in temps])
        if trials_by_T is not None
        else None
    )
    pass1, pass_base, estimator = _pass_metrics(p1, base, n, trials1, trials_base)
    half = len(p1) // 2
    values: dict[str, list[float]] = {"E_1": [], "E_N": []}
    chosen: dict[str, list[float]] = {"E_1": [], "E_N": []}
    for _ in range(n_splits):
        perm = rng.permutation(len(p1))
        h1, h2 = perm[:half], perm[half:]
        for key, trained, base_metric in (("E_1", p1, base), ("E_N", pass1, pass_base)):
            value, t_a, t_b = split_half_value(trained, base_metric, h1, h2)
            values[key].append(value)
            chosen[key] += [temps[t_a], temps[t_b]]
    freq = {
        key: {float(t): chosen[key].count(t) / len(chosen[key]) for t in temps} for key in chosen
    }
    return {
        "E_1": float(np.mean(values["E_1"])),
        "E_N": float(np.mean(values["E_N"])),
        "t_star_E_1": freq["E_1"],
        "t_star_E_N": freq["E_N"],
        "per_split": values,
        "pass_at_n_estimator": estimator,
        "n": n,
        "n_splits": n_splits,
    }


# --------------------------------------------------------------------------- bootstrap


def _interval(estimate: float, samples: NDArray[np.float64], alpha: float) -> dict:
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": float(estimate), "lo": float(lo), "hi": float(hi)}


def bootstrap(
    successes_A: Any,
    trials_A: Any,
    p0: Any,
    p1: Any,
    n: int,
    alpha: float,
    n_boot: int,
    boot_seed: int,
    *,
    base_by_T: Mapping[float, Any] | None = None,
    trials1: Any = None,
    trials_by_T: Mapping[float, Any] | None = None,
    n_splits_boot: int = 1,
) -> dict:
    """Percentile intervals (level 1 - alpha) for the total and each contribution, and, when
    `base_by_T` is given, for the temperature baseline's E_1 and E_N.

    Each replicate resamples problems with replacement and redraws every resampled problem's
    set-A count from Binomial(trials_A, observed rate_A) before reclassifying. The temperature
    baseline is recomputed on the same resampled problems with `n_splits_boot` fresh random
    half splits (split noise is then part of the interval, which makes it conservative).
    Streams: SeedSequence(boot_seed) spawns one stream for the split and one for temperature.
    """
    k_a = np.asarray(successes_A, dtype=np.int64)
    m_a = np.broadcast_to(np.asarray(trials_A, dtype=np.int64), k_a.shape)
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    if not (k_a.shape == p0.shape == p1.shape) or k_a.ndim != 1 or not len(k_a):
        raise ValueError("set-A counts, p0 and p1 must be non-empty 1-d arrays of the same length")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    split_ss, temp_ss = np.random.SeedSequence(boot_seed).spawn(2)
    rng = np.random.default_rng(split_ss)
    size = len(k_a)
    gain = p1 - p0
    point = _split_rows(_classify_codes(k_a, m_a, n, alpha), gain)
    idx = rng.integers(0, size, size=(n_boot, size))
    redrawn = rng.binomial(m_a[idx], k_a[idx] / m_a[idx])
    rows = _split_rows(_classify_codes(redrawn, m_a[idx], n, alpha), gain[idx])
    out: dict[str, Any] = {
        name: _interval(point[c], rows[:, c], alpha) for c, name in enumerate(CLASSES + ("total",))
    }
    out.update({"alpha": alpha, "n_boot": n_boot, "boot_seed": boot_seed, "samples": rows})
    if base_by_T is not None:
        temps = sorted(base_by_T)
        base = {t: np.asarray(base_by_T[t], dtype=float) for t in temps}
        t1 = None if trials1 is None else np.broadcast_to(np.asarray(trials1, dtype=np.int64), p1.shape)
        tt = (
            None
            if trials_by_T is None
            else {t: np.broadcast_to(np.asarray(trials_by_T[t], dtype=np.int64), p1.shape) for t in temps}
        )
        point_t = temperature_baseline(p1, base, n, n_splits_boot, np.random.default_rng(temp_ss.spawn(1)[0]),
                                       trials1=t1, trials_by_T=tt)
        rng_t = np.random.default_rng(temp_ss)
        e1 = np.empty(n_boot)
        en = np.empty(n_boot)
        for b in range(n_boot):
            ii = idx[b]
            res = temperature_baseline(
                p1[ii], {t: base[t][ii] for t in temps}, n, n_splits_boot, rng_t,
                trials1=None if t1 is None else t1[ii],
                trials_by_T=None if tt is None else {t: tt[t][ii] for t in temps},
            )
            e1[b], en[b] = res["E_1"], res["E_N"]
        out["temperature"] = {
            "E_1": _interval(point_t["E_1"], e1, alpha),
            "E_N": _interval(point_t["E_N"], en, alpha),
            "pass_at_n_estimator": point_t["pass_at_n_estimator"],
            "note": "estimate here uses n_splits_boot splits; report temperature_baseline's mean over n_splits",
        }
    return out


def null_check(
    successes_A: Any, trials_A: Any, p0: Any, p_null: Any, n: int, alpha: float, n_boot: int, boot_seed: int
) -> dict:
    """The split pipeline with p1 := a second base sample. Passes iff the total's interval contains 0."""
    result = bootstrap(successes_A, trials_A, p0, p_null, n, alpha, n_boot, boot_seed)
    total = result["total"]
    return {"total": total, "passed": bool(total["lo"] <= 0.0 <= total["hi"])}


# --------------------------------------------------------------------------- budget curve


def budget_curve(successes_A: Any, trials_A: Any, p0: Any, p1: Any, n_grid: Sequence[int], alpha: float) -> dict:
    """The three-way split at each N of `n_grid`, reclassifying per N, cut at the largest
    resolvable N = floor(1/u0) for the smallest set-A size (`budget.resolvable_n_max`, c = 1)."""
    k_a = np.asarray(successes_A, dtype=np.int64)
    m_a = np.broadcast_to(np.asarray(trials_A, dtype=np.int64), k_a.shape)
    gain = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    m_min = int(m_a.min())
    n_max = resolvable_n_max(m_min, 1.0, alpha)
    kept = [int(x) for x in n_grid if x <= n_max]
    dropped = [int(x) for x in n_grid if x > n_max]
    curve: dict[str, list] = {name: [] for name in CLASSES + ("total",)}
    curve["counts"] = []
    for n in kept:
        codes = _classify_codes(k_a, m_a, n, alpha)
        row = _split_rows(codes, gain)
        for c, name in enumerate(CLASSES + ("total",)):
            curve[name].append(float(row[c]))
        curve["counts"].append({name: int((codes == c).sum()) for c, name in enumerate(CLASSES)})
    reason = (
        f"set A has >= {m_min} samples per problem; the {100 * (1 - alpha):g}% Clopper-Pearson upper bound for "
        f"0 successes is u0 = {clopper_pearson_zero_upper(m_min, alpha):.4g}, so no problem can be classified "
        f"unreachable for N > floor(1/u0) = {n_max}"
    )
    return {"n": kept, **curve, "n_max_resolvable": n_max, "truncated": bool(dropped), "dropped": dropped,
            "reason": reason}


# --------------------------------------------------------------------------- GroupMetrics adapter


def group_metrics(
    groups: Mapping[str, Sequence[int]],
    counts_A: Counts,
    counts_B: Counts,
    counts_1: Counts,
    counts_by_T: Mapping[float, Counts],
    n: int,
    alpha: float,
    n_boot: int,
    boot_seed: int,
    n_splits: int,
    *,
    counts_null: Counts | None = None,
) -> list[GroupMetrics]:
    """One GroupMetrics per group at budget N (c = 1). Group g uses streams SeedSequence([boot_seed, g])."""
    out = []
    for g_index, (group, raw_ids) in enumerate(groups.items()):
        ids = np.asarray(raw_ids, dtype=np.int64)
        a, b, t = counts_A.subset(ids), counts_B.subset(ids), counts_1.subset(ids)
        temp_counts = {temp: c.subset(ids) for temp, c in counts_by_T.items()}
        p0, p1 = b.rates(), t.rates()
        base_by_T = {temp: c.rates() for temp, c in temp_counts.items()}
        trials_by_T = {temp: c.trials for temp, c in temp_counts.items()}
        boot_ss, temp_ss = np.random.SeedSequence([boot_seed, g_index]).spawn(2)
        classes = classify(a.successes, a.trials, n, alpha)
        three = split(classes, p0, p1)
        temperature = temperature_baseline(p1, base_by_T, n, n_splits, np.random.default_rng(temp_ss),
                                           trials1=t.trials, trials_by_T=trials_by_T)
        boot_seed_g = int(boot_ss.generate_state(1)[0])
        ci = bootstrap(a.successes, a.trials, p0, p1, n, alpha, n_boot, boot_seed_g,
                       base_by_T=base_by_T, trials1=t.trials, trials_by_T=trials_by_T)
        trained_lower, _ = clopper_pearson(t.successes, t.trials, alpha)
        crossed = (classes == "unreachable") & (trained_lower >= 1.0 / n)
        beyond_reach = bool(crossed.any())
        e1_lo = ci["temperature"]["E_1"]["lo"]
        en_lo = ci["temperature"]["E_N"]["lo"]
        extra: dict[str, Any] = {
            "backend": "lm_sampled",
            "split": three,
            "ci": {k: ci[k] for k in CLASSES + ("total",)},
            "temperature": temperature,
            "temperature_ci": {k: ci["temperature"][k] for k in ("E_1", "E_N")},
            "e1": {"per_step": temperature["E_1"], "sequence": float("nan"), "both": float("nan")},
            "beyond_reach": beyond_reach,
            "beyond_temperature": bool(e1_lo > 0),
            "crossed": [int(i) for i in ids[crossed]],
            "label_basis": "beyond_temperature iff E_1 bootstrap lower bound > 0 (H39); label_en uses E_N's",
            "label_en": h12_label(beyond_reach, en_lo, eps=0.0),
            "delta_rel_includes": ["reachable", "ambiguous"],
            "alpha": alpha,
            "n_boot": n_boot,
            "boot_seed": boot_seed,
            "group_seed_index": g_index,
        }
        if counts_null is not None:
            nb = counts_null.subset(ids)
            extra["null_check"] = null_check(a.successes, a.trials, p0, nb.rates(), n, alpha, n_boot, boot_seed_g)
        out.append(
            GroupMetrics(
                group=group,
                n=n,
                c=1.0,
                n_prompts=len(ids),
                j0=float(p0.mean()),
                j1=float(p1.mean()),
                delta_rel=three["reachable"] + three["ambiguous"],
                delta_cap=three["unreachable"],
                excess={"per_step": temperature["E_N"], "sequence": float("nan"), "both": float("nan")},
                label=h12_label(beyond_reach, e1_lo, eps=0.0),
                extra=extra,
            )
        )
    return out

