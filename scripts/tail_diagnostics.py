"""Diagnostics for the tail layer's failed stop gate (DECISIONS L43-L44).

Run after the `tail` and `tail-report` stages. Writes plain text to stdout. Nothing here changes any
stage output. Three parts:

1. Known-answer control: the z distribution over trusted problems, then candidate causes. (a) The
   proposal: teacher-forced minus sampling-time log q of the sampled component, by component. (b) The
   numerator: the batched teacher-forced pi0 of each D-set sample (as stored) minus the single-path
   forward that produced D. (c) Chance in the component draw: z against the deviation of the SFT
   component's episode count from its expectation (SFT draws most D-set hits). (d) What is left: the
   mean of log(D-hat / D) minus the numerator gap.
2. Numerics: one D-set completion per chosen problem, scored batched (its stored chunk), alone, and as a
   single-path forward, on MPS float16 (the run) and on CPU float32.
3. Brute force: re-draws one control problem's stored brute-force sample (same seed), and a second
   sample with a different seed, then pools every base sample for it against its bracket.

Usage: python scripts/tail_diagnostics.py [--numerics 30,172,138] [--bf 178] [--alt-seed 12345]
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np

RUN = Path("results/lm_main_tail")
OUT = Path("results/lm_main")
COMPONENTS = ("base", "hot", "sft", "rl_half")


def _rows(pid: int) -> list[dict]:
    with open(RUN / "is" / f"{pid}.jsonl") as handle:
        return [json.loads(line) for line in handle]


def _d_rows() -> dict[int, dict]:
    with open(RUN / "d.jsonl") as handle:
        return {r["problem"]: r for r in map(json.loads, handle)}


def _binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), summed in log space."""
    logs = [math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * math.log(p)
            + (n - j) * math.log1p(-p) for j in range(k, n + 1)]
    top = max(logs)
    return math.exp(top) * sum(math.exp(v - top) for v in logs)


def known_answer_part() -> None:
    """z over trusted problems and the decomposition described in the module docstring."""
    tail = json.loads((RUN / "report" / "tail.json").read_text())
    weights = dict(zip(COMPONENTS, json.loads((RUN / "manifest.tail.json").read_text())["tail_config"]["weights"]))
    d = _d_rows()
    z, gap, log_ratio, sft_dev, q_diff = [], [], [], [], {c: [] for c in COMPONENTS}
    for key, entry in tail["is"].items():
        rows = _rows(int(key))
        for r in rows:
            q_diff[r["component"]].append(r[f"lp_{r['component']}"] - sum(r["logq_sample"]))
        known = entry["known"]
        if not known["trust"]["trusted"]:
            continue
        hits = [r for r in rows if r["in_d"]]
        log_digits = d[int(key)]["log_digits"]
        z.append(known["z"])
        gap.append(float(np.median([r["lp_base"] - log_digits for r in hits])) - math.log(d[int(key)]["end_mass"]))
        log_ratio.append(math.log(known["estimate"] / d[int(key)]["D"]))
        n_sft = sum(r["component"] == "sft" for r in rows)
        sft_dev.append((n_sft - len(rows) * weights["sft"]) / math.sqrt(len(rows) * weights["sft"] * (1 - weights["sft"])))
    z_a, gap_a, lr_a, dev_a = map(np.array, (z, gap, log_ratio, sft_dev))
    print("== 1. known-answer control")
    print(f"trusted problems {len(z_a)}; z mean {z_a.mean():.3f} (SE of the mean if calibrated "
          f"{1 / math.sqrt(len(z_a)):.3f}); z sd {z_a.std(ddof=1):.3f}; negative {(z_a < 0).sum()}; "
          f"|z| > 3.5: {(np.abs(z_a) > 3.5).sum()}")
    print("(a) teacher-forced minus sampling-time log q, all IS rows, by component:")
    for c in COMPONENTS:
        v = np.array(q_diff[c])
        print(f"    {c}: rows {len(v)}, mean {v.mean():.5f}, sd {v.std():.4f}")
    print(f"(b) batched minus single-path log pi0 of D-set samples (per problem, end factor removed): "
          f"mean {gap_a.mean():.4f}, min {gap_a.min():.4f}, max {gap_a.max():.4f}, median |.| {np.median(np.abs(gap_a)):.4f}")
    print(f"    correlation of z with (b): {np.corrcoef(z_a, gap_a)[0, 1]:.3f}")
    print(f"(c) SFT episode-count deviation (z units): mean {dev_a.mean():.3f}; correlation with control z "
          f"{np.corrcoef(z_a, dev_a)[0, 1]:.3f}")
    resid = lr_a - gap_a
    print(f"(d) log(D-hat / D): mean {lr_a.mean():.4f}; minus (b): mean {resid.mean():.4f}, "
          f"SE {resid.std(ddof=1) / math.sqrt(len(resid)):.4f}")
    worst = int(np.argmin(z_a))
    keys = [k for k, e in tail["is"].items() if e["known"]["trust"]["trusted"]]
    print(f"most extreme: problem {keys[worst]}, z {z_a[worst]:.3f}, (b) {gap_a[worst]:.4f}, "
          f"SFT episode deviation {dev_a[worst]:.3f}, log(D-hat / D) {lr_a[worst]:.4f}")


def numerics_part(problems: list[int]) -> None:
    """One D-set completion per problem: batched vs alone vs single-path, MPS float16 and CPU float32."""
    import torch

    from capsplit.harness.lm import tail as tl
    from capsplit.harness.lm.config import load_config
    from capsplit.harness.lm.policy import LocalLMPolicy
    from capsplit.harness.lm.tasks import load_task

    cfg = load_config("configs/lm_main.json")
    task = load_task(str(OUT / "data" / "problems.jsonl"), cfg.task.prompt_template)
    size = int(cfg.sampling.batch_size)
    print("== 2. numerics: log pi0 of one D-set completion (sum over tokens)")
    devices = (["mps"] if torch.backends.mps.is_available() else []) + ["cpu"]
    for device in devices:
        policy = LocalLMPolicy(cfg, device=device)
        dtype = next(policy.model.parameters()).dtype
        for pid in problems:
            rows = _rows(pid)
            target = next(i for i, r in enumerate(rows) if r["in_d"])
            lo = (target // size) * size
            chunk = [r["ids"] for r in rows[lo: lo + size]]
            y = rows[target]["ids"]
            prompt = tl.prompt_ids(policy, task, pid)
            batched = tl.score_completions(policy, prompt, chunk, (1.0,), use_adapter=False)[0][target - lo]
            alone = tl.score_completions(policy, prompt, [y], (1.0,), use_adapter=False)[0][0]
            lp = tl.forward_logprobs(policy, prompt, [y[:-1]], keep=len(y))[0][0]
            single = sum(lp[i][y[i]] for i in range(len(y)))
            print(f"{device} {dtype} problem {pid}: batched {np.nansum(batched):.4f}  alone {np.nansum(alone):.4f}  "
                  f"single-path {single:.4f}  (stored lp_base {rows[target]['lp_base']:.4f})")
        del policy
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()


def bf_part(pid: int, alt_seed: int) -> None:
    """Re-draw the stored brute-force sample (same seed) and one with another seed; pool against the bracket."""
    import torch

    from capsplit.harness.lm import stages as st
    from capsplit.harness.lm import tail as tl
    from capsplit.harness.lm.config import load_config
    from capsplit.harness.lm.tasks import check

    config = load_config("configs/lm_main.json")
    tcfg = tl.load_tail_config("configs/lm_tail.json")
    st._plugins()
    policy = st.default_policy_factory(config, st.resolve_device(config), False)(0)
    task = st.load_task(config, OUT)
    stop = {int(s) for s in policy.stop_ids}
    vocab = tl.build_vocab(policy.tokenizer, int(policy.model.get_output_embeddings().weight.shape[0]), stop)
    decode = tl.decoder(policy.tokenizer, vocab.n_tokens)
    ref = float(config.sampling.reference_temperature)
    max_new = int(config.model.max_new_tokens)
    prompt, answer = tl.prompt_ids(policy, task, pid), task.answer(pid)

    def draw(seed: int) -> int:
        gen = torch.Generator().manual_seed(seed)
        size, correct = int(config.sampling.batch_size), 0
        for lo in range(0, tcfg.n_bf, size):
            count = min(tcfg.n_bf, lo + size) - lo
            ids = torch.tensor([prompt] * count, dtype=torch.long, device=policy.device)
            with policy.model.disable_adapter():
                toks, _ = tl.sample_with_logq(policy.model, {"input_ids": ids, "attention_mask": torch.ones_like(ids)},
                                              max_new, ref, policy.stop_ids, policy.pad_id, gen)
            correct += sum(bool(check(decode(t), answer)[1]) for t in toks)
        return correct

    stored = json.loads((RUN / "bf" / f"{pid}.json").read_text())
    again = draw(st.chunk_seed(tcfg.seed, "tail/bf", "bf", ref, pid))
    alt = draw(alt_seed)
    with open(RUN / "bracket.jsonl") as handle:
        b = next(r for r in map(json.loads, handle) if r["problem"] == pid)
    k_ref, m_ref, _ = tl.base_reference(OUT, config, len(task.prompts()))
    k = stored["correct"] + alt + int(k_ref[pid])
    n = stored["n"] + tcfg.n_bf + int(m_ref[pid])
    upper = b["won"] + b["open"]
    print(f"== 3. brute force, problem {pid} (answer {answer})")
    print(f"stored {stored['correct']}/{stored['n']}; same seed re-drawn {again}/{tcfg.n_bf}; "
          f"seed {alt_seed} {alt}/{tcfg.n_bf}; reference {int(k_ref[pid])}/{int(m_ref[pid])}")
    print(f"bracket [{b['won']:.5g}, {upper:.5g}]; pooled {k}/{n} = {k / n:.5g}; "
          f"one-sided P(X >= {k} | p = upper end) = {_binom_sf(k, n, upper):.3g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--numerics", default="30,172,138")
    parser.add_argument("--bf", type=int, default=178)
    parser.add_argument("--alt-seed", type=int, default=12345)
    args = parser.parse_args()
    known_answer_part()
    numerics_part([int(p) for p in args.numerics.split(",") if p])
    bf_part(args.bf, args.alt_seed)


if __name__ == "__main__":
    main()
