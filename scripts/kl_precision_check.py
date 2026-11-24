"""Precision check for the KL layer's re-derivation (DECISIONS L35).

Re-scores the saved KL samples of one problem with the stage's own scoring path
(`capsplit.harness.lm.kl.score_pair`) twice: on CPU in float32 and on MPS in float16 (the run's
setting). It prints each mean log-ratio next to the file's, so a gap between the independent float32
re-derivation and the stage can be attributed to precision or to code.

Usage: python scripts/kl_precision_check.py [arm] [seed] [problem]
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

from capsplit.harness.lm.config import load_config
from capsplit.harness.lm.kl import score_pair
from capsplit.harness.lm.policy import LocalLMPolicy
from capsplit.harness.lm.tasks import load_task


def main(arm: str = "rl", seed: int = 0, problem: int = 8) -> None:
    cfg = load_config("configs/lm_main.json")
    task = load_task("results/lm_main/data/problems.jsonl", cfg.task.prompt_template)
    path = f"results/lm_main_kl/samples/{arm}/seed{seed}/step200.jsonl"
    with open(path) as handle:
        rows = [r for r in map(json.loads, handle) if r["problem"] == problem]
    file = np.array([r["logp1"] - r["logp0"] for r in rows])
    print(f"{arm} seed {seed} problem {problem}: {len(rows)} samples; file mean log-ratio {file.mean()!r}")
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for device in devices:
        policy = LocalLMPolicy(cfg, device=device)
        policy.load_adapter(f"results/lm_main/train/{arm}/seed{seed}/step200/adapter.safetensors")
        lp1, lp0 = score_pair(policy, task, [problem] * len(rows), [tuple(r["ids"]) for r in rows])
        ratio = lp1 - lp0
        dtype = next(policy.model.parameters()).dtype
        print(f"{device} {dtype}: mean log-ratio {ratio.mean()!r}; minus file {ratio.mean() - file.mean()!r}; "
              f"max |row diff| {np.abs(ratio - file).max()!r}")
        del policy


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*(args[:1]), *(int(a) for a in args[1:3]))
