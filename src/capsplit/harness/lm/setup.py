"""Stages doctor, fetch and speedtest: prove the model downloads and runs on this machine.

Why: the experiment's sizes (problems, samples, updates) are chosen from measured throughput,
so these stages run first and save what they measured. `setup` = fetch + speedtest, which is the
single command that repeats model setup from scratch.
"""
from __future__ import annotations

import platform
import shutil
import sys
import time
from pathlib import Path

import psutil
import torch

from .config import LMConfig
from .device import GB, MemoryGuard, apply_runtime, available_gb, pick_device, process_gb
from .model import ModelChoice, attach_lora, choose, encode_prompts, fetch, load, sample_batch, stop_token_ids
from .storage import write_json, write_manifest

SPEEDTEST_QUESTIONS = ("Compute 23 + 58. Reply with only the final number.",
                       "Compute 417 + 386. Reply with only the final number.",
                       "Compute 12 * 34. Reply with only the final number.",
                       "Compute 908 - 457. Reply with only the final number.")


def doctor(config: LMConfig, out: Path) -> Path:
    """Report the environment: versions, device, memory and disk. Writes doctor.json."""
    import peft
    import transformers

    device = pick_device(config.model.device)
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "device": device.type,
        "memory_available_gb": round(available_gb(), 2),
        "memory_total_gb": round(psutil.virtual_memory().total / GB, 2),
        "disk_free_gb": round(shutil.disk_usage(out.parent if out.parent.exists() else Path.cwd()).free / GB, 1),
        "min_free_gb_required": config.memory.min_free_gb,
    }
    path = out / "doctor.json"
    write_json(path, info)
    return path


def fetch_stage(config: LMConfig, out: Path) -> Path:
    """Download the primary and (if configured) fallback checkpoints. Writes fetch.json."""
    rows = []
    choices = [choose(config.model, False)] + ([choose(config.model, True)] if config.model.fallback_id else [])
    for choice in choices:
        t0 = time.monotonic()
        path = Path(fetch(config.model, choice))
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        rows.append({"id": choice.id, "revision": choice.revision, "path": str(path),
                     "size_gb": round(size / GB, 3), "seconds": round(time.monotonic() - t0, 1)})
        print(f"[fetch] {choice.id}@{choice.revision[:12]}: {size / GB:.2f} GB at {path}", file=sys.stderr)
    target = out / "fetch.json"
    write_json(target, {"models": rows})
    return target


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def speedtest(config: LMConfig, out: Path, use_fallback: bool = False) -> Path:
    """Load the model, generate arithmetic answers, time batched sampling, take one LoRA step.

    Writes speedtest.json with device, memory, generated answers, tokens per second for single
    and batched generation, and seconds per LoRA training step.
    """
    apply_runtime(config.runtime)
    guard = MemoryGuard(config.memory)
    guard.check_start("speedtest")
    device = pick_device(config.model.device)
    choice: ModelChoice = choose(config.model, use_fallback)
    torch.manual_seed(0)
    t0 = time.monotonic()
    model, tokenizer = load(config.model, choice, device)
    load_seconds = time.monotonic() - t0
    memory_after_load = process_gb()

    # Low-temperature answers to a few arithmetic prompts: does the model work at all?
    stops = stop_token_ids(tokenizer)
    generator = torch.Generator().manual_seed(0)
    batch = encode_prompts(tokenizer, config.model.system_prompt, list(SPEEDTEST_QUESTIONS), device)
    near_greedy = sample_batch(model, batch, config.model.max_new_tokens, 0.01, 1.0, stops, tokenizer.pad_token_id, generator)
    answers = tokenizer.batch_decode(near_greedy.tokens, skip_special_tokens=True)

    # Sampled throughput at the configured batch size (the shape the experiment uses).
    rates = {}
    for size in sorted({1, config.sampling.batch_size}):
        questions = [SPEEDTEST_QUESTIONS[i % len(SPEEDTEST_QUESTIONS)] for i in range(size)]
        batch = encode_prompts(tokenizer, config.model.system_prompt, questions, device)
        _sync(device)
        t = time.monotonic()
        sampled = sample_batch(model, batch, config.model.max_new_tokens, config.sampling.reference_temperature,
                               config.sampling.top_p, stops, tokenizer.pad_token_id, generator)
        _sync(device)
        seconds = time.monotonic() - t
        new_tokens = int(sampled.lengths.sum())
        rates[str(size)] = {"sequences": size, "seconds": round(seconds, 3), "new_tokens": new_tokens,
                            "tokens_per_second": round(new_tokens / seconds, 1),
                            "sequences_per_second": round(size / seconds, 2)}
        guard.check_running("speedtest")

    # One LoRA training step on a supervised example.
    model = attach_lora(model, config.lora)
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    batch = encode_prompts(tokenizer, config.model.system_prompt, [SPEEDTEST_QUESTIONS[0]], device)
    target = tokenizer("81", return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    input_ids = torch.cat([batch["input_ids"], target], dim=1)
    labels = torch.full_like(input_ids, -100)
    labels[:, -target.shape[1]:] = target
    _sync(device)
    t = time.monotonic()
    loss = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=labels).loss
    loss.backward()
    optimizer.step()
    _sync(device)
    lora_step = time.monotonic() - t
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    result = {
        "model": {"id": choice.id, "revision": choice.revision},
        "device": device.type,
        "dtype": str(next(model.parameters()).dtype),
        "load_seconds": round(load_seconds, 2),
        "memory_gb": {"after_load": round(memory_after_load, 2), "peak_process": round(process_gb(), 2),
                      "available_at_end": round(available_gb(), 2)},
        "low_temperature_answers": dict(zip(SPEEDTEST_QUESTIONS, answers)),
        "sampling": rates,
        "lora": {"trainable_parameters": trainable, "step_seconds": round(lora_step, 3), "loss": round(float(loss), 4)},
    }
    target_path = out / ("speedtest_fallback.json" if use_fallback else "speedtest.json")
    write_json(target_path, result)
    write_manifest(out, "speedtest", config.to_dict(), seed=0, device=device.type,
                   extra={"model": result["model"]})
    return target_path
