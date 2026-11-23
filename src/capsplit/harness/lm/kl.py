"""Function-space layer: per-problem KL(pi1 || pi0) of each trained policy from the base (DECISIONS L23-L31).

The output metric asks whether a completion is correct. This layer asks how far the whole
completion distribution moved. For a problem x, pi0 is the base (LoRA adapter disabled) and pi1 the
trained policy (adapter on), both at the reference temperature; a completion is the generated token
sequence including its stop token, truncated at `model.max_new_tokens`. With S = "the completion
is correct" and p0, p1 its probabilities, the data-processing inequality gives

    KL(pi1 || pi0) >= d(p1 || p0) = p1 ln(p1 / p0) + (1 - p1) ln((1 - p1) / (1 - p0)),

so moving a problem from p0 < 1/N to p1 ~ 1 costs at least ln(1 / p0) > ln N nats.

Estimator: draw y_1..y_m ~ pi1(. | x) and average ln pi1(y_i | x) - ln pi0(y_i | x). It is unbiased
for KL(pi1 || pi0); single terms can be negative and are not clipped. Both log-probs come from the
same code path, `LocalLMPolicy.log_prob` (teacher-forced), once with the adapter on and once inside
peft's `disable_adapter()`, so the two differ only by the adapter.

Stages (CLI): `kl` samples and scores on the device; `kl-report` computes everything else on CPU.
Inputs are read-only: the completed run's adapters, base sets A and B, the trained arms' eval
samples and training logs under `<out>/<name>/`. Outputs go to a separate folder
`<out>/<name>_kl/` (DECISIONS L24):

* `samples/<arm>/seed<s>/step<k>.jsonl`: one row per completion (problem, index, token ids, text,
  valid, correct, logp1, logp0), resumable per problem chunk;
* `manifest.kl.json`: configs, model, device, commit, sha256 of every adapter read;
* `report/`: report.md, kl.json, tables/*.csv, figures/*.png.

Weight descriptors (secondary, descriptive): DeltaW = (alpha / r) B A per LoRA module. They are
coordinates: many weight settings give the same behaviour, so they do not define capability.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import zlib
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import LMConfig
from .report import _plt, _save, load_run, md_num, md_table, write_csv, write_json_det, write_text
from .storage import Progress, append_jsonl, read_jsonl, write_manifest

SMOOTHINGS = ("jeffreys", "laplace")
GROUPS = (
    ("train_reachable", "train, reachable"),
    ("train_unreachable_discovered", "train, unreachable, sampled correct in training"),
    ("train_unreachable_never", "train, unreachable, never sampled correct in training"),
    ("heldout_reachable", "held-out, reachable"),
    ("heldout_unreachable", "held-out, unreachable"),
    ("heldout_ambiguous", "held-out, ambiguous (not one of the five pre-registered groups)"),
)
_KEY = re.compile(r"layers\.(\d+)\.self_attn\.(\w+)\.lora_([AB])\.weight$")


class KLConfigError(ValueError):
    """configs/lm_kl.json is invalid; the message names the key."""


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class KLConfig:
    """What the `kl` and `kl-report` stages do (configs/lm_kl.json; every key required, none unknown)."""

    arms: tuple[str, ...]
    m_final: int
    checkpoints: tuple[int, ...]
    m_checkpoint: int
    m_zero: int
    max_problems: int | None
    seed: int
    smoothing: str
    alt_smoothing: str
    violation_se: float
    n_boot: int
    boot_seed: int
    extra: dict = field(default_factory=dict)


def parse_kl_config(raw: dict) -> KLConfig:
    names = {f for f in KLConfig.__dataclass_fields__ if f != "extra"}
    unknown, missing = sorted(set(raw) - names), sorted(names - set(raw))
    if unknown:
        raise KLConfigError(f"unknown KL config key(s): {', '.join(unknown)}")
    if missing:
        raise KLConfigError(f"missing KL config key(s): {', '.join(missing)}")
    for key in ("m_final", "m_checkpoint", "m_zero", "n_boot"):
        if not isinstance(raw[key], int) or raw[key] < 1:
            raise KLConfigError(f"{key} must be a positive integer, got {raw[key]!r}")
    if raw["max_problems"] is not None and (not isinstance(raw["max_problems"], int) or raw["max_problems"] < 1):
        raise KLConfigError(f"max_problems must be null or a positive integer, got {raw['max_problems']!r}")
    for key in ("smoothing", "alt_smoothing"):
        if raw[key] not in SMOOTHINGS:
            raise KLConfigError(f"{key} must be one of {SMOOTHINGS}, got {raw[key]!r}")
    if not isinstance(raw["arms"], list) or not raw["arms"]:
        raise KLConfigError("arms must be a non-empty list of arm names")
    if not isinstance(raw["checkpoints"], list):
        raise KLConfigError("checkpoints must be a list of update counts")
    return KLConfig(arms=tuple(str(a) for a in raw["arms"]), m_final=raw["m_final"],
                    checkpoints=tuple(int(c) for c in raw["checkpoints"]), m_checkpoint=raw["m_checkpoint"],
                    m_zero=raw["m_zero"], max_problems=raw["max_problems"], seed=int(raw["seed"]),
                    smoothing=raw["smoothing"], alt_smoothing=raw["alt_smoothing"],
                    violation_se=float(raw["violation_se"]), n_boot=raw["n_boot"], boot_seed=int(raw["boot_seed"]))


def load_kl_config(path: str | Path) -> KLConfig:
    return parse_kl_config(json.loads(Path(path).read_text()))


def _kl_dict(kcfg: KLConfig) -> dict:
    d = asdict(kcfg)
    d.pop("extra")
    return json.loads(json.dumps(d))


# --------------------------------------------------------------------------- math


def bernoulli_kl(p1: np.ndarray, p0: np.ndarray) -> np.ndarray:
    """d(p1 || p0) elementwise, with 0 ln 0 = 0; +inf where p1 puts mass where p0 has none."""
    p1, p0 = np.asarray(p1, dtype=float), np.asarray(p0, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p1 > 0, p1 * np.log(p1 / p0), 0.0)
        b = np.where(p1 < 1, (1 - p1) * np.log((1 - p1) / (1 - p0)), 0.0)
    return a + b


def smooth(k: np.ndarray, n: np.ndarray, kind: str) -> np.ndarray:
    """Posterior mean of a success rate: Beta(1/2, 1/2) prior ('jeffreys') or Beta(1, 1) ('laplace')."""
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    if kind == "jeffreys":
        return (k + 0.5) / (n + 1.0)
    if kind == "laplace":
        return (k + 1.0) / (n + 2.0)
    raise ValueError(f"unknown smoothing {kind!r}; use one of {SMOOTHINGS}")


def kl_estimate(log_ratios: np.ndarray) -> tuple[float, float]:
    """(mean, standard error) of sampled log-ratios ln pi1(y) - ln pi0(y), y ~ pi1. No clipping."""
    x = np.asarray(log_ratios, dtype=float)
    mean = float(x.mean())
    se = float(np.std(x, ddof=1) / math.sqrt(len(x))) if len(x) > 1 else math.nan
    return mean, se


def cluster_bootstrap(values: np.ndarray, clusters: np.ndarray, n_boot: int, rng: np.random.Generator,
                      level: float = 0.95) -> dict:
    """Mean of `values` with a percentile interval, resampling whole clusters (problems) with replacement.

    Each problem may carry several observations (one per seed); they are kept together.
    """
    values, clusters = np.asarray(values, dtype=float), np.asarray(clusters)
    if len(values) == 0:
        return {"estimate": None, "lo": None, "hi": None, "n": 0, "n_clusters": 0}
    ids, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(ids))
    counts = np.bincount(inverse, minlength=len(ids)).astype(float)
    draws = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    stats = sums[draws].sum(1) / counts[draws].sum(1)
    tail = (1 - level) / 2
    return {"estimate": float(values.mean()), "lo": float(np.quantile(stats, tail)),
            "hi": float(np.quantile(stats, 1 - tail)), "n": len(values), "n_clusters": len(ids)}


def ratio_bootstrap(num: np.ndarray, den: np.ndarray, clusters: np.ndarray, n_boot: int,
                    rng: np.random.Generator, level: float = 0.95) -> dict:
    """mean(num) / mean(den) with a cluster-resampled percentile interval (efficiency = gain per nat)."""
    num, den, clusters = np.asarray(num, float), np.asarray(den, float), np.asarray(clusters)
    if len(num) == 0 or float(den.mean()) <= 0:
        return {"estimate": None, "lo": None, "hi": None, "n": len(num)}
    ids, inverse = np.unique(clusters, return_inverse=True)
    sn = np.bincount(inverse, weights=num, minlength=len(ids))
    sd = np.bincount(inverse, weights=den, minlength=len(ids))
    draws = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    with np.errstate(divide="ignore", invalid="ignore"):
        stats = sn[draws].sum(1) / sd[draws].sum(1)
    stats = stats[np.isfinite(stats)]
    tail = (1 - level) / 2
    return {"estimate": float(num.mean() / den.mean()), "lo": float(np.quantile(stats, tail)),
            "hi": float(np.quantile(stats, 1 - tail)), "n": len(num)}


# --------------------------------------------------------------------------- weights


def lora_modules(state: dict[str, np.ndarray]) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]:
    """{(layer, module): (A, B)} from a saved LoRA state dict (A is r x in, B is out x r)."""
    parts: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for key, value in state.items():
        match = _KEY.search(key)
        if match is None:
            raise ValueError(f"unexpected adapter key {key!r}")
        layer, module, which = int(match.group(1)), match.group(2), match.group(3)
        parts.setdefault((layer, module), {})[which] = np.asarray(value, dtype=np.float64)
    out = {}
    for k, v in parts.items():
        if set(v) != {"A", "B"}:
            raise ValueError(f"adapter module {k} lacks lora_A or lora_B")
        out[k] = (v["A"], v["B"])
    return out


def delta_w_singular_values(a: np.ndarray, b: np.ndarray, scale: float) -> np.ndarray:
    """Singular values of DeltaW = scale * B A (at most r of them), via QR of B and A^T (r x r SVD)."""
    qb, rb = np.linalg.qr(b)
    qa, ra = np.linalg.qr(a.T)
    del qb, qa
    return scale * np.linalg.svd(rb @ ra.T, compute_uv=False)


def delta_w_inner(a1: np.ndarray, b1: np.ndarray, a2: np.ndarray, b2: np.ndarray, scale: float) -> float:
    """Frobenius inner product <DeltaW1, DeltaW2> = scale^2 tr((B1^T B2)(A2 A1^T)), without forming DeltaW."""
    return float(scale ** 2 * np.trace((b1.T @ b2) @ (a2 @ a1.T)))


def participation_ratio(sv: np.ndarray) -> float:
    """Effective rank (sum s^2)^2 / sum s^4 of singular values s: 1 for rank one, k for k equal values."""
    s2 = np.asarray(sv, dtype=float) ** 2
    denom = float(np.sum(s2 ** 2))
    return math.nan if denom == 0 else float(np.sum(s2) ** 2 / denom)


def load_adapter_np(path: Path) -> dict[str, np.ndarray]:
    from safetensors.numpy import load_file

    return dict(load_file(str(path)))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- sampling and scoring


def score_pair(policy: Any, task: Any, ids: Sequence[int], actions: Sequence[tuple]) -> tuple[np.ndarray, np.ndarray]:
    """(ln pi1, ln pi0) of each completion, both from `policy.log_prob` (adapter on, then disabled)."""
    import torch

    with torch.no_grad():
        lp1 = policy.log_prob(task, list(ids), list(actions))
        with policy.model.disable_adapter():
            lp0 = policy.log_prob(task, list(ids), list(actions))
    return lp1.float().cpu().double().numpy(), lp0.float().cpu().double().numpy()  # MPS has no float64


def sample_completions(policy: Any, task: Any, pid: int, m: int, generator: Any) -> list[tuple[int, ...]]:
    """m completions (token ids incl. the stop token when generated) from pi1 at the reference temperature.

    The same draws as `LocalLMPolicy.sample` (same `sample_batch` loop, batches, temperature, top_p
    and generator), without its extra scoring forward: scores come from `score_pair` only (L27).
    """
    from .model import encode_prompts, sample_batch

    question = str(task.prompts()[pid].tokens[0])
    size = int(policy.config.sampling.batch_size)
    out: list[tuple[int, ...]] = []
    for lo in range(0, m, size):
        n = min(m, lo + size) - lo
        batch = encode_prompts(policy.tokenizer, policy.config.model.system_prompt, [question] * n, policy.device)
        drawn = sample_batch(policy.model, batch, policy.config.model.max_new_tokens, policy.temperature,
                             policy.config.sampling.top_p, policy.stop_ids, policy.pad_id, generator)
        out += [tuple(int(t) for t in drawn.tokens[i, : int(k)].tolist()) for i, k in enumerate(drawn.lengths)]
        policy.guard.check_running("kl sample")
    return out


def kl_chunk(policy: Any, task: Any, pid: int, m: int, generator: Any) -> list[dict]:
    """m completions of problem `pid` sampled from pi1, each scored under pi1 and pi0."""
    ids = [pid] * m
    actions = sample_completions(policy, task, pid, m, generator)
    lp1, lp0 = score_pair(policy, task, ids, actions)
    texts = policy.tokenizer.batch_decode([list(a) for a in actions], skip_special_tokens=True)
    rows = []
    for i, (text, act) in enumerate(zip(texts, actions, strict=True)):
        valid, correct = task.parse(pid, text) if hasattr(task, "parse") else (True, bool(task.verify(pid, text)))
        rows.append({"problem": int(pid), "index": i, "ids": list(act), "text": text, "valid": bool(valid),
                     "correct": bool(correct), "logp1": float(lp1[i]), "logp0": float(lp0[i])})
    return rows


@dataclass(frozen=True)
class KLJob:
    arm: str
    seed: int
    step: int
    m: int
    role: str  # final | checkpoint | zero

    @property
    def tag(self) -> str:
        return f"{self.arm}/seed{self.seed}/step{self.step}"


def kl_jobs(config: LMConfig, kcfg: KLConfig) -> list[KLJob]:
    """Final checkpoint first (the main measurement), then the exact-zero checks, then intermediate steps."""
    arms = {a.name for a in config.arms}
    bad = [a for a in kcfg.arms if a not in arms]
    if bad:
        raise KLConfigError(f"arm(s) {bad} are not in the experiment config (arms: {sorted(arms)})")
    steps = set(config.checkpoints)
    bad_steps = [c for c in kcfg.checkpoints if c not in steps or c in (0, config.budget)]
    if bad_steps:
        raise KLConfigError(f"checkpoints {bad_steps} must be saved intermediate checkpoints of the run "
                            f"(saved: {sorted(steps)}; 0 and the final step are covered separately)")
    final, seeds = config.budget, list(config.seeds)
    jobs = [KLJob(a, s, final, kcfg.m_final, "final") for a in kcfg.arms for s in seeds]
    zero_arms = [a.name for a in config.arms if a.trainer == "lm_none"]
    jobs += [KLJob(a, s, final, kcfg.m_zero, "zero") for a in zero_arms for s in seeds]
    jobs += [KLJob(a, s, 0, kcfg.m_zero, "zero") for a in kcfg.arms for s in seeds] if 0 in steps else []
    jobs += [KLJob(a, s, c, kcfg.m_checkpoint, "checkpoint") for c in kcfg.checkpoints for a in kcfg.arms
             for s in seeds]
    return jobs


def kl_sample_path(kl_out: Path, job: KLJob) -> Path:
    return kl_out / "samples" / job.arm / f"seed{job.seed}" / f"step{job.step}.jsonl"


def _have(path: Path, m: int) -> dict[int, set[int]]:
    from .stages import repair_tail

    repair_tail(path)
    have: dict[int, set[int]] = {}
    for row in read_jsonl(path):
        if int(row["index"]) < m:
            have.setdefault(int(row["problem"]), set()).add(int(row["index"]))
    return have


def run_kl(config: LMConfig, kcfg: KLConfig, out: Path, kl_out: Path, device: Any = "cpu",
           factory: Callable[[int], Any] | None = None, use_fallback: bool = False) -> list[Path]:
    """Sample and score every job; resumable per (job, problem). Reads `out`, writes only `kl_out`."""
    import torch

    from . import stages as st

    st._plugins()
    chosen = st.model_choice(config, use_fallback)
    fp = st.fingerprint(st.plain_config(config), chosen, "train")
    jobs = kl_jobs(config, kcfg)
    with st._stage(config, "kl", device):
        st._check_upstream(out, fp, base=True)
        task = st.load_task(config, out)
        n_problems = len(task.prompts())
        problems = list(range(n_problems if kcfg.max_problems is None else min(n_problems, kcfg.max_problems)))
        missing_adapters, present = [], []
        for job in jobs:
            path = st.adapter_path(out, job.arm, job.seed, job.step)
            if not path.exists():
                missing_adapters.append(job.tag)
                print(f"WARNING: adapter {path} is missing; skipping {job.tag} (not retrained)", file=sys.stderr)
            else:
                present.append(job)
        for arm, seed in sorted({(j.arm, j.seed) for j in present}):
            d = st.run_dir(out, arm, seed)
            st.check_manifest(d / "manifest.train.json", "train", fp, st.run_keys(arm), [d])
        hashes = {j.tag: file_sha256(st.adapter_path(out, j.arm, j.seed, j.step)) for j in present}
        manifest = write_manifest(kl_out, "kl", config.to_dict(), kcfg.seed, str(device),
                                  extra={"kl_config": _kl_dict(kcfg), "model": chosen, "source": str(out),
                                         "adapter_sha256": hashes, "missing_adapters": missing_adapters,
                                         "jobs": [asdict(j) for j in present], "problems": len(problems)})
        todo = []
        for job in present:
            have = _have(kl_sample_path(kl_out, job), job.m)
            todo += [(job, pid, set(range(job.m)) - have.get(pid, set())) for pid in problems
                     if len(have.get(pid, set())) < job.m]
        total = len(present) * len(problems)
        progress = Progress("kl", total, total - len(todo))
        policy, loaded = None, None
        ref = float(config.sampling.reference_temperature)
        try:
            for k, (job, pid, want) in enumerate(todo):
                if policy is None:
                    policy = (factory or st.default_policy_factory(config, device, use_fallback))(0)
                if loaded != job.tag:
                    policy.load_adapter(st.adapter_path(out, job.arm, job.seed, job.step))
                    loaded = job.tag
                gen = torch.Generator().manual_seed(st.chunk_seed(kcfg.seed, "kl/" + job.tag, "kl", ref, pid))
                rows = [r for r in kl_chunk(policy, task, pid, job.m, gen) if r["index"] in want]
                append_jsonl(kl_sample_path(kl_out, job), rows)
                progress.update(total - len(todo) + k + 1)
        finally:
            progress.close()
    return sorted({kl_sample_path(kl_out, j) for j in present}) + [manifest]


# --------------------------------------------------------------------------- analysis


def read_kl_samples(path: Path, m: int, problems: Sequence[int]) -> dict[int, list[dict]]:
    by: dict[int, dict[int, dict]] = {}
    for row in read_jsonl(path):
        if int(row["index"]) < m:
            by.setdefault(int(row["problem"]), {}).setdefault(int(row["index"]), row)
    short = [p for p in problems if len(by.get(p, {})) < m]
    if short:
        raise ValueError(f"{path} is incomplete ({len(short)} problems lack {m} samples, first {short[:5]}); "
                         "re-run the `kl` stage")
    return {p: [by[p][i] for i in range(m)] for p in problems}


def _per_problem(rows: dict[int, list[dict]]) -> dict[int, dict]:
    out = {}
    for pid, rs in rows.items():
        lr = np.array([r["logp1"] - r["logp0"] for r in rs])
        mean, se = kl_estimate(lr)
        out[pid] = {"kl": mean, "se": se, "m": len(rs), "p1_kl": float(np.mean([r["correct"] for r in rs])),
                    "max_abs_log_ratio": float(np.max(np.abs(lr))),
                    "mean_tokens": float(np.mean([len(r["ids"]) for r in rs]))}
    return out


def _counts(run: Any, tag: str, set_name: str, n: int) -> np.ndarray:
    return np.rint(run.rates(tag, set_name, n) * n).astype(np.int64)


def _group_of(split: str, cls: str, ever: bool | None) -> str | None:
    if split == "heldout":
        return f"heldout_{cls}"
    if cls == "reachable":
        return "train_reachable"
    if cls == "unreachable":
        return "train_unreachable_discovered" if ever else "train_unreachable_never"
    return None  # train ambiguous: not a pre-registered group


def _rng(kcfg: KLConfig, label: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([kcfg.boot_seed, zlib.crc32(label.encode())]))


def compute_kl(out: Path, kl_out: Path, kcfg: KLConfig) -> dict[str, Any]:
    run = load_run(out)
    manifest = json.loads((kl_out / "manifest.kl.json").read_text())
    jobs = [KLJob(**j) for j in manifest["jobs"]]
    n_problems = int(manifest["problems"])
    problems = list(range(n_problems))
    sp = run.config["sampling"]
    n_a, n_b, n_t = int(sp["n_classify"]), int(sp["n_measure"]), int(sp["n_trained"])
    k_a, m_a = run.set_a()
    k_b = _counts(run, "base", "B", n_b)
    k0, n0 = (k_a + k_b)[:n_problems], (m_a + n_b)[:n_problems]
    p0_raw = k0 / n0
    cls = run.class_array()[:n_problems]
    split = np.array([r["split"] for r in run.problems])[:n_problems]
    final = run.final_step

    per: dict[str, dict[int, dict]] = {}
    for job in jobs:
        per[job.tag] = _per_problem(read_kl_samples(kl_sample_path(kl_out, job), job.m, problems))

    # zero checks: exact equality is the expectation, so report the largest |log ratio| seen
    zero = [{"arm": j.arm, "seed": j.seed, "step": j.step, "m": j.m,
             "max_abs_log_ratio": max(v["max_abs_log_ratio"] for v in per[j.tag].values()),
             "mean_kl": float(np.mean([v["kl"] for v in per[j.tag].values()]))}
            for j in jobs if j.role == "zero"]

    # per-problem table at the final checkpoint
    rows = []
    for job in (j for j in jobs if j.role == "final"):
        k1 = _counts(run, f"{job.arm}/seed{job.seed}/step{final}", "eval", n_t)[:n_problems]
        log = run.training.get((job.arm, job.seed)) or {}
        prompts = log.get("prompts", {})
        sampled = int((log.get("totals") or {}).get("episodes_sampled", 0)) > 0
        bounds = {kind: bernoulli_kl(smooth(k1, np.full(n_problems, n_t), kind), smooth(k0, n0, kind))
                  for kind in (kcfg.smoothing, kcfg.alt_smoothing)}
        for pid in problems:
            v = per[job.tag][pid]
            entry = prompts.get(str(pid)) or {}
            ever = (entry.get("first_success_update") is not None) if sampled and split[pid] == "train" else None
            d, d_alt = float(bounds[kcfg.smoothing][pid]), float(bounds[kcfg.alt_smoothing][pid])
            se = v["se"] if not math.isnan(v["se"]) else 0.0
            rows.append({"arm": job.arm, "seed": job.seed, "problem": pid, "split": str(split[pid]),
                         "class": str(cls[pid]), "ever_correct_in_training": ever,
                         "group": _group_of(str(split[pid]), str(cls[pid]), ever),
                         "kl": v["kl"], "se": v["se"], "m": v["m"], "mean_tokens": v["mean_tokens"],
                         "p0_hat": float(p0_raw[pid]), "p1_hat": float(k1[pid] / n_t), "p1_kl_samples": v["p1_kl"],
                         "delta_p": float(k1[pid] / n_t - p0_raw[pid]), "bound": d, "bound_alt": d_alt,
                         "violation": bool(v["kl"] + kcfg.violation_se * se < d),
                         "violation_alt": bool(v["kl"] + kcfg.violation_se * se < d_alt)})

    def k1_row(arm: str, sel: list[dict]) -> dict:
        n = len(sel)
        return {"arm": arm, "n": n, "violations": sum(r["violation"] for r in sel),
                "fraction": sum(r["violation"] for r in sel) / n if n else None,
                "violations_alt": sum(r["violation_alt"] for r in sel),
                "fraction_alt": sum(r["violation_alt"] for r in sel) / n if n else None,
                "kl_minus_bound_min_in_se": min(((r["kl"] - r["bound"]) / r["se"] for r in sel
                                                 if r["se"] and not math.isnan(r["se"])), default=None)}

    arms = list(dict.fromkeys(r["arm"] for r in rows))
    k1 = [k1_row(a, [r for r in rows if r["arm"] == a]) for a in arms] + [k1_row("pooled", rows)]

    # K2 / K3: group means pooled over seeds, bootstrap over problems
    k2 = []
    for arm in arms:
        for key, label in GROUPS:
            sel = [r for r in rows if r["arm"] == arm and r["group"] == key]
            kl_v = np.array([r["kl"] for r in sel])
            dp = np.array([r["delta_p"] for r in sel])
            clusters = np.array([r["problem"] for r in sel])
            k2.append({"arm": arm, "group": key, "label": label,
                       "kl": cluster_bootstrap(kl_v, clusters, kcfg.n_boot, _rng(kcfg, f"k2/{arm}/{key}")),
                       "delta_p": cluster_bootstrap(dp, clusters, kcfg.n_boot, _rng(kcfg, f"dp/{arm}/{key}")),
                       "efficiency": ratio_bootstrap(dp, kl_v, clusters, kcfg.n_boot, _rng(kcfg, f"eff/{arm}/{key}")),
                       "bound": float(np.mean([r["bound"] for r in sel])) if sel else None,
                       "per_seed_kl": {str(s): float(np.mean([r["kl"] for r in sel if r["seed"] == s]))
                                       for s in sorted({r["seed"] for r in sel})}})

    # trajectory over updates: mean KL and gain (p1 from the KL samples) per arm x seed x step
    traj = []
    for job in jobs:
        if job.role == "zero" and job.arm not in kcfg.arms:
            continue
        v = per[job.tag]
        kls = np.array([v[p]["kl"] for p in problems])
        p1 = np.array([v[p]["p1_kl"] for p in problems])
        traj.append({"arm": job.arm, "seed": job.seed, "step": job.step, "m": job.m, "mean_kl": float(kls.mean()),
                     "gain": float(np.mean(p1 - p0_raw)), "mean_kl_train": float(kls[split == "train"].mean()),
                     "mean_kl_heldout": float(kls[split == "heldout"].mean())})
    traj.sort(key=lambda r: (r["arm"], r["seed"], r["step"]))
    early = []
    for arm in kcfg.arms:
        for seed in run.seeds:
            by = {r["step"]: r["mean_kl"] for r in traj if r["arm"] == arm and r["seed"] == seed}
            if final in by and by[final] > 0:
                early.append({"arm": arm, "seed": seed, **{f"fraction_at_{s}": by[s] / by[final]
                                                            for s in sorted(by) if s not in (0, final)}})
    return {"status": "exploratory function-space layer on the completed lm_main run; pre-registered in "
                      "docs/lm_main/kl/preregistration_kl.md (DECISIONS L23-L31); no verdict on L1-L3",
            "kl_config": _kl_dict(kcfg), "source_run": str(out), "n_problems": n_problems,
            "p0_source": f"base sets A and B pooled ({n_a} + {n_b} samples per problem)",
            "p1_source": f"the trained arm's saved eval samples ({n_t} per problem)",
            "zero_checks": zero, "k1": k1, "k2": k2, "trajectory": traj, "early_fraction": early,
            "per_problem": rows, "weights": weight_descriptors(out, run, per, jobs, final)}


def weight_descriptors(out: Path, run: Any, per: dict, jobs: Sequence[KLJob], final: int) -> dict:
    """Frobenius norm and effective rank of DeltaW per adapter and module; cosines between final adapters."""
    lora = run.config["lora"]
    scale = float(lora["alpha"]) / float(lora["r"])
    adapters, modules = [], []
    finals: dict[str, dict] = {}
    for arm in (a["name"] for a in run.arms):
        for seed in run.seeds:
            for step in run.config["checkpoints"]:
                path = out / "train" / arm / f"seed{seed}" / f"step{step}" / "adapter.safetensors"
                if not path.exists():
                    continue
                mods = lora_modules(load_adapter_np(path))
                total2, prs = 0.0, []
                for (layer, module), (a, b) in sorted(mods.items()):
                    sv = delta_w_singular_values(a, b, scale)
                    fro = float(np.sqrt(np.sum(sv ** 2)))
                    pr = participation_ratio(sv)
                    total2 += fro ** 2
                    prs.append(pr)
                    if step == final:
                        modules.append({"arm": arm, "seed": seed, "step": step, "layer": layer, "module": module,
                                        "fro": fro, "effective_rank": pr})
                tag = f"{arm}/seed{seed}/step{step}"
                mean_kl = float(np.mean([v["kl"] for v in per[tag].values()])) if tag in per else None
                adapters.append({"arm": arm, "seed": seed, "step": step, "fro_total": math.sqrt(total2),
                                 "effective_rank_mean": float(np.nanmean(prs)) if not all(map(math.isnan, prs)) else None,
                                 "mean_kl": mean_kl})
                if step == final and arm in {j.arm for j in jobs if j.role == "final"}:
                    finals[f"{arm}/seed{seed}"] = mods
    names = sorted(finals)
    norms = {n: math.sqrt(sum(delta_w_inner(a, b, a, b, scale) for a, b in finals[n].values())) for n in names}
    cos = []
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            inner = sum(delta_w_inner(*finals[x][k], *finals[y][k], scale) for k in finals[x])
            cos.append({"a": x, "b": y, "cosine": inner / (norms[x] * norms[y]) if norms[x] and norms[y] else None})
    return {"scale": scale, "definition": {"delta_w": "(alpha / r) * B @ A per LoRA module",
                                           "effective_rank": "participation ratio (sum s^2)^2 / sum s^4 of the "
                                                             "singular values s of DeltaW",
                                           "cosine": "<vec DeltaW_a, vec DeltaW_b> / (|DeltaW_a| |DeltaW_b|) over "
                                                     "all modules concatenated (final checkpoint)"},
            "adapters": adapters, "modules": modules, "cosines": cos}


# --------------------------------------------------------------------------- report


def _ci(d: dict | None, digits: int = 3) -> str:
    if not d or d.get("estimate") is None:
        return "n/a"
    return f"{md_num(d['estimate'], digits, sign=False)} [{md_num(d['lo'], digits, sign=False)}, " \
           f"{md_num(d['hi'], digits, sign=False)}]"


def _sci(d: dict | None, digits: int = 3) -> str:
    if not d or d.get("estimate") is None:
        return "n/a"
    return f"{md_num(d['estimate'], digits)} [{md_num(d['lo'], digits)}, {md_num(d['hi'], digits)}]"


def _figures(res: dict, fig_dir: Path) -> list[Path]:
    plt = _plt()
    if plt is None:
        return []
    paths = []
    arms = list(dict.fromkeys(r["arm"] for r in res["k2"]))
    keys = [k for k, _ in GROUPS[:5]]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    width = 0.8 / max(1, len(arms))
    for i, arm in enumerate(arms):
        xs, ys, lo, hi = [], [], [], []
        for j, key in enumerate(keys):
            g = next(r for r in res["k2"] if r["arm"] == arm and r["group"] == key)["kl"]
            if g["estimate"] is None:
                continue
            xs.append(j + (i - (len(arms) - 1) / 2) * width)
            ys.append(g["estimate"])
            lo.append(g["estimate"] - g["lo"])
            hi.append(g["hi"] - g["estimate"])
        ax.bar(xs, ys, width=width * 0.9, label=arm, yerr=[lo, hi], capsize=2)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.replace("_", "\n") for k in keys], fontsize=7)
    ax.set_ylabel("mean KL(pi1 || pi0), nats")
    ax.set_title("Per-problem KL by group (final checkpoint; 95% bootstrap over problems)")
    ax.legend(frameon=False)
    paths.append(_save(plt, fig, fig_dir / "kl_by_group.png"))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for arm in arms:
        steps = sorted({r["step"] for r in res["trajectory"] if r["arm"] == arm})
        xs = [float(np.mean([r["mean_kl"] for r in res["trajectory"] if r["arm"] == arm and r["step"] == s])) for s in steps]
        ys = [float(np.mean([r["gain"] for r in res["trajectory"] if r["arm"] == arm and r["step"] == s])) for s in steps]
        ax.plot(xs, ys, marker="o", label=arm)
        for x, y, s in zip(xs, ys, steps, strict=True):
            ax.annotate(str(s), (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("mean KL(pi1 || pi0) over all problems, nats")
    ax.set_ylabel("total gain (p1 from KL samples) minus p0 (A+B)")
    ax.set_title("Gain vs KL over updates (labels: update count; mean over seeds)")
    ax.legend(frameon=False)
    paths.append(_save(plt, fig, fig_dir / "gain_vs_kl.png"))

    fig, ax = plt.subplots(figsize=(5, 5))
    for arm in arms:
        sel = [r for r in res["per_problem"] if r["arm"] == arm]
        ax.scatter([r["bound"] for r in sel], [r["kl"] for r in sel], s=6, alpha=0.6, label=arm)
    top = max([max(r["bound"], r["kl"]) for r in res["per_problem"]] + [1e-6])
    ax.plot([0, top], [0, top], color="#555555", lw=1, label="y = x")
    ax.set_xlabel("lower bound d(p1_hat || p0_hat), nats")
    ax.set_ylabel("KL_hat, nats")
    ax.set_title("K1: sampled KL vs the data-processing bound")
    ax.legend(frameon=False)
    paths.append(_save(plt, fig, fig_dir / "k1_scatter.png"))
    return paths


def _tables(res: dict, tab: Path) -> list[Path]:
    paths = []
    pp = res["per_problem"]
    cols = ["arm", "seed", "problem", "split", "class", "group", "ever_correct_in_training", "kl", "se", "m",
            "mean_tokens", "p0_hat", "p1_hat", "p1_kl_samples", "delta_p", "bound", "bound_alt", "violation",
            "violation_alt"]
    paths.append(write_csv(tab / "per_problem.csv", cols, [[r[c] for c in cols] for r in pp]))
    k1c = ["arm", "n", "violations", "fraction", "violations_alt", "fraction_alt", "kl_minus_bound_min_in_se"]
    paths.append(write_csv(tab / "k1.csv", k1c, [[r[c] for c in k1c] for r in res["k1"]]))
    paths.append(write_csv(tab / "k2_groups.csv",
                           ["arm", "group", "n_obs", "n_problems", "kl", "kl_lo", "kl_hi", "delta_p", "delta_p_lo",
                            "delta_p_hi", "efficiency", "efficiency_lo", "efficiency_hi", "mean_bound"],
                           [[g["arm"], g["group"], g["kl"]["n"], g["kl"].get("n_clusters"), g["kl"]["estimate"],
                             g["kl"]["lo"], g["kl"]["hi"], g["delta_p"]["estimate"], g["delta_p"]["lo"],
                             g["delta_p"]["hi"], g["efficiency"]["estimate"], g["efficiency"]["lo"],
                             g["efficiency"]["hi"], g["bound"]] for g in res["k2"]]))
    tc = ["arm", "seed", "step", "m", "mean_kl", "mean_kl_train", "mean_kl_heldout", "gain"]
    paths.append(write_csv(tab / "trajectory.csv", tc, [[r[c] for c in tc] for r in res["trajectory"]]))
    zc = ["arm", "seed", "step", "m", "max_abs_log_ratio", "mean_kl"]
    paths.append(write_csv(tab / "zero_checks.csv", zc, [[r[c] for c in zc] for r in res["zero_checks"]]))
    w = res["weights"]
    ac = ["arm", "seed", "step", "fro_total", "effective_rank_mean", "mean_kl"]
    paths.append(write_csv(tab / "weights_adapters.csv", ac, [[r[c] for c in ac] for r in w["adapters"]]))
    mc = ["arm", "seed", "step", "layer", "module", "fro", "effective_rank"]
    paths.append(write_csv(tab / "weights_modules.csv", mc, [[r[c] for c in mc] for r in w["modules"]]))
    paths.append(write_csv(tab / "weights_cosine.csv", ["a", "b", "cosine"],
                           [[r["a"], r["b"], r["cosine"]] for r in w["cosines"]]))
    return paths


def _cos_summary(cos: list[dict]) -> list[list[Any]]:
    pairs: dict[tuple[str, str], list[float]] = {}
    for r in cos:
        if r["cosine"] is None:
            continue
        a, b = r["a"].split("/")[0], r["b"].split("/")[0]
        pairs.setdefault(tuple(sorted((a, b))), []).append(r["cosine"])
    return [[f"{a} vs {b}" if a != b else f"{a}, across seeds", len(v), md_num(np.mean(v), 3),
             md_num(min(v), 3), md_num(max(v), 3)] for (a, b), v in sorted(pairs.items())]


def render(res: dict, name: str) -> str:
    kc = res["kl_config"]
    md = [f"# Function-space layer (KL to the base): {name}", "",
          ("Generated by `capsplit.harness.lm.kl` from the completed run's saved adapters and samples; no number "
           "below is typed by hand. Exploratory: pre-registered in `preregistration_kl.md` after the main results "
           "were known; it gives no verdict on L1-L3."), "",
          f"- p̂0: {res['p0_source']}; p̂1: {res['p1_source']}.",
          (f"- KL_hat: mean of ln pi1(y) - ln pi0(y) over m completions y ~ pi1 at the reference temperature "
          f"(m = {kc['m_final']} at the final checkpoint, {kc['m_checkpoint']} at checkpoints {kc['checkpoints']}, "
          f"{kc['m_zero']} for the exact-zero checks); standard error = sd / sqrt(m)."),
          (f"- Bound d(p̂1 || p̂0) with {kc['smoothing']} smoothing (alternative: {kc['alt_smoothing']}); a violation "
          f"is KL_hat + {kc['violation_se']:g} SE < d."), "",
          "## Exact-zero checks", "",
          "A fresh adapter (lora_B = 0) is the `none` arm at every step and every arm at step 0; KL must be exactly 0.",
          "", *md_table(["arm", "seed", "step", "m", "max |ln pi1 - ln pi0|", "mean KL_hat"],
                        [[z["arm"], z["seed"], z["step"], z["m"], md_num(z["max_abs_log_ratio"], 6, sign=False),
                          md_num(z["mean_kl"], 6, sign=False)] for z in res["zero_checks"]]), "",
          "## K1: bound check", "",
          *md_table(["arm", "problems x seeds", "violations", "fraction", f"violations ({kc['alt_smoothing']})",
                     f"fraction ({kc['alt_smoothing']})", "min (KL_hat - d) / SE"],
                    [[r["arm"], r["n"], r["violations"], md_num(r["fraction"], 4, sign=False), r["violations_alt"],
                      md_num(r["fraction_alt"], 4, sign=False), md_num(r["kl_minus_bound_min_in_se"], 2)]
                     for r in res["k1"]]), "",
          "![K1 scatter](figures/k1_scatter.png)", "",
          "## K2 and K3: KL, gain and efficiency by group (final checkpoint)", "",
          ("Seeds pooled; one observation = one problem in one seed; intervals resample problems (all seeds of a "
           "problem together). 'Sampled correct in training' comes from the training log (first_success_update); "
           "arms that never sample (sft) have no such split, so all their train unreachable problems are in the "
           "'never' row. Efficiency = mean Δp / mean KL_hat (Δp = p̂1 - p̂0, gain per nat)."), "",
          *md_table(["arm", "group", "obs", "problems", "KL_hat [95%]", "Δp [95%]", "Δp / KL [95%]", "mean bound d",
                     "KL_hat per seed"],
                    [[g["arm"], g["label"], g["kl"]["n"], g["kl"].get("n_clusters", 0), _ci(g["kl"]),
                      _sci(g["delta_p"]), _sci(g["efficiency"]), md_num(g["bound"], 3, sign=False),
                      ", ".join(f"{s}: {md_num(v, 3, sign=False)}" for s, v in g["per_seed_kl"].items())]
                     for g in res["k2"] if g["kl"]["n"]]), "",
          "![KL by group](figures/kl_by_group.png)", "",
          "## Over updates", "",
          ("Mean KL_hat over all problems and the total gain at each saved checkpoint. Here p̂1 comes from the KL "
           "samples themselves (the run measured only the final checkpoint), so the step-200 gain differs slightly "
           "from the main report's."), "",
          *md_table(["arm", "seed", "step", "m", "mean KL_hat", "train", "held-out", "gain"],
                    [[r["arm"], r["seed"], r["step"], r["m"], md_num(r["mean_kl"], 4, sign=False),
                      md_num(r["mean_kl_train"], 4, sign=False), md_num(r["mean_kl_heldout"], 4, sign=False),
                      md_num(r["gain"], 4)] for r in res["trajectory"]]), "",
          "Fraction of the final mean KL reached at each intermediate checkpoint:", "",
          *md_table(["arm", "seed", *sorted({k for r in res["early_fraction"] for k in r if k.startswith("fraction")},
                                            key=lambda s: int(s.rsplit("_", 1)[1]))],
                    [[r["arm"], r["seed"], *[md_num(r[k], 3, sign=False) for k in sorted(
                        (k for k in r if k.startswith("fraction")), key=lambda s: int(s.rsplit("_", 1)[1]))]]
                     for r in res["early_fraction"]]), "",
          "![gain vs KL](figures/gain_vs_kl.png)", "",
          "## Weight descriptors (descriptive only)", "",
          ("Weights are coordinates: many weight settings give the same behaviour, and these numbers do not define "
           "capability. " + "; ".join(f"{k}: {v}" for k, v in res["weights"]["definition"].items()) + "."), "",
          *md_table(["arm", "seed", "step", "‖ΔW‖_F (all modules)", "mean effective rank", "mean KL_hat"],
                    [[a["arm"], a["seed"], a["step"], md_num(a["fro_total"], 4, sign=False),
                      md_num(a["effective_rank_mean"], 3, sign=False), md_num(a["mean_kl"], 4, sign=False)]
                     for a in res["weights"]["adapters"] if a["step"] == max(x["step"] for x in res["weights"]["adapters"])
                     and a["arm"] in kc["arms"]]), "",
          "Cosine between flattened ΔW (final checkpoint):", "",
          *md_table(["pair", "pairs", "mean cosine", "min", "max"], _cos_summary(res["weights"]["cosines"])), "",
          ("Per-layer and per-module norms and effective ranks: `tables/weights_modules.csv`; every checkpoint: "
          "`tables/weights_adapters.csv`."), "",
          "## What this cannot show", "",
          "- One 0.5B model with LoRA, generated arithmetic, 3 seeds; nothing here generalises beyond that.",
          "- The held-out unreachable group is small (see the K2 table's problem count).",
          ("- KL_hat is a sampled estimate from m completions per problem; heavy-tailed log-ratios can make the "
          "standard error itself unreliable for a few problems."),
          "- Completions are at most `max_new_tokens` tokens; KL is over these truncated sequences only.",
          ("- KL measures how far the output distribution moved, not in which direction or why; a large KL is not "
          "evidence of new capability, and the weight descriptors are coordinates, not capability."), ""]
    return "\n".join(md)


def build_kl_report(out: str | Path, kl_out: str | Path, kcfg: KLConfig) -> Path:
    """Write `<kl_out>/report/{kl.json, report.md, tables/*.csv, figures/*.png}`; returns kl.json."""
    out, kl_out = Path(out), Path(kl_out)
    res = compute_kl(out, kl_out, kcfg)
    rep = kl_out / "report"
    path = write_json_det(rep / "kl.json", res)
    _tables(res, rep / "tables")
    _figures(res, rep / "figures")
    name = json.loads((kl_out / "manifest.kl.json").read_text())["config"]["name"]
    write_text(rep / "report.md", render(res, name))
    return path
