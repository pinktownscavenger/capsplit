"""Downloading and loading the pinned Hugging Face checkpoint, with LoRA attached.

Why one module: the fetch, speedtest, base, train and measure stages all need the same model
loaded the same way (same revision, dtype, device), so loading lives in exactly one place.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import LoraConfig, ModelConfig
from .device import dtype_of


@dataclass(frozen=True)
class ModelChoice:
    """The checkpoint actually used (primary or fallback), recorded in every manifest."""

    id: str
    revision: str


def choose(config: ModelConfig, use_fallback: bool) -> ModelChoice:
    if not use_fallback:
        return ModelChoice(config.id, config.revision)
    if not config.fallback_id or not config.fallback_revision:
        raise ValueError("use_fallback was requested but model.fallback_id / fallback_revision are not set")
    return ModelChoice(config.fallback_id, config.fallback_revision)


def fetch(config: ModelConfig, choice: ModelChoice, local_only: bool = False) -> str:
    """Download (or, with local_only, locate) the pinned snapshot; returns its local path.

    Only weights, config and tokenizer files are fetched. Models that need a login or licence
    acceptance are not used (docs/DECISIONS.md), so no token is required.
    """
    return snapshot_download(
        repo_id=choice.id,
        revision=choice.revision,
        cache_dir=config.cache_dir,
        local_files_only=local_only,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
    )


def load(config: ModelConfig, choice: ModelChoice, device: torch.device) -> tuple[torch.nn.Module, object]:
    """Load model and tokenizer from the local cache (never the network) onto `device`."""
    path = fetch(config, choice, local_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.padding_side = "left"  # batched generation appends to the right
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype_of(config.dtype, device))
    model.to(device)
    model.eval()
    return model, tokenizer


def attach_lora(model: torch.nn.Module, config: LoraConfig) -> torch.nn.Module:
    """Wrap `model` with fresh LoRA adapters; only adapter weights are trainable."""
    peft_config = PeftLoraConfig(
        r=config.r,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(config.target_modules),
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, peft_config)


def encode_prompts(tokenizer: object, system_prompt: str, questions: list[str], device: torch.device) -> dict:
    """Chat-format each question (system + user turn + assistant start) and left-pad into a batch."""
    texts = [
        tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in questions
    ]
    batch = tokenizer(texts, return_tensors="pt", padding=True)  # type: ignore[operator]
    return {k: v.to(device) for k, v in batch.items()}


@dataclass
class SampledBatch:
    """Generated continuations for one batch. `tokens` is (B, T) and padded with `pad_id` after a
    sequence ends; `lengths[i]` counts generated tokens of sequence i including its end token."""

    tokens: torch.Tensor
    lengths: torch.Tensor


@torch.no_grad()
def sample_batch(
    model: torch.nn.Module,
    batch: dict,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop_ids: set[int],
    pad_id: int,
    generator: torch.Generator,
) -> SampledBatch:
    """Sample continuations with a key-value cache, one token at a time.

    Why not `model.generate`: on macOS 14 its MPS path trips a Metal limit (an internal array over
    4 GB), and this loop also lets every draw come from one explicit CPU generator, so sampling is
    reproducible given the seed. Sampling happens on CPU from float32 probabilities.
    """
    device = batch["input_ids"].device
    mask = batch["attention_mask"]
    positions = (mask.cumsum(-1) - 1).clamp(min=0)
    out = model(input_ids=batch["input_ids"], attention_mask=mask, position_ids=positions, use_cache=True, logits_to_keep=1)
    past = out.past_key_values
    size = batch["input_ids"].shape[0]
    finished = torch.zeros(size, dtype=torch.bool)
    lengths = torch.zeros(size, dtype=torch.long)
    steps = []
    for _ in range(max_new_tokens):
        logits = out.logits[:, -1].float().cpu()
        probs = _filtered_probs(logits, temperature, top_p)
        token = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        token = torch.where(finished, torch.full_like(token, pad_id), token)
        lengths += (~finished).long()
        finished |= torch.tensor([int(t) in stop_ids for t in token.tolist()])
        steps.append(token)
        if bool(finished.all()):
            break
        mask = torch.cat([mask, torch.ones((size, 1), dtype=mask.dtype, device=device)], dim=1)
        positions = positions[:, -1:] + 1
        out = model(input_ids=token.unsqueeze(1).to(device), attention_mask=mask, position_ids=positions,
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
    return SampledBatch(torch.stack(steps, dim=1), lengths)


def _filtered_probs(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """softmax(logits / T), restricted to the smallest top-p set when top_p < 1."""
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p >= 1.0:
        return probs
    sorted_probs, order = probs.sort(dim=-1, descending=True)
    keep = sorted_probs.cumsum(-1) - sorted_probs < top_p
    sorted_probs = sorted_probs * keep
    return torch.zeros_like(probs).scatter(-1, order, sorted_probs) / sorted_probs.sum(-1, keepdim=True)


def stop_token_ids(tokenizer: object) -> set[int]:
    """End-of-turn tokens: the tokenizer's eos plus the chat template's end marker if it has one."""
    ids = {int(tokenizer.eos_token_id)}  # type: ignore[attr-defined]
    for marker in ("<|im_end|>", "<|eot_id|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(marker)  # type: ignore[attr-defined]
        if isinstance(token_id, int) and token_id != tokenizer.unk_token_id:  # type: ignore[attr-defined]
            ids.add(token_id)
    return ids
