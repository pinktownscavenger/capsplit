"""`LocalLMPolicy`: a local causal language model with a LoRA adapter, as a harness `Policy`.

RL framing (docs/lm_design.md): the state is the chat-formatted prompt plus the tokens generated
so far, an action is the next token, and an episode ends at a stop token or after
`model.max_new_tokens` tokens. pi0 is the base model (adapter disabled), pi1 the model with its
trained adapter, so one loaded model serves both.

Design choices (each documented where it is implemented):

* **Sampling** reuses `model.sample_batch` (no gradients, one explicit CPU generator), batched by
  `sampling.batch_size`. Log-probabilities and entropies with gradients come from a separate
  teacher-forced forward over prompt + completion, so `sample` and `log_prob` share one scoring
  path and agree exactly.
* **The policy's distribution** is softmax(logits / T) with T = `sampling.reference_temperature`
  (the temperature every trained policy is measured at). With the default T = 1 this is the model
  itself. `sampling.top_p` < 1 truncates *sampling* only; `log_prob` and `entropy` are those of the
  untruncated tempered distribution.
* **LoRA init** is seeded from `seed` inside `torch.random.fork_rng`, so the global RNG state is
  untouched. peft creates the adapter weights on CPU and only then moves them to the model's
  device, so forking the CPU generator is enough on every device.
* **Mode**: the model stays in eval mode. LoRA dropout (`lora.dropout`) is therefore inactive
  unless a trainer calls `policy.model.train()`; the configured experiment uses dropout 0.
* **Forced prefixes** (hint arms) are not supported: the LM experiment has no hint arm, so passing
  `prefixes` raises NotImplementedError.
* **clone()** deep-copies the model. Sharing one base between several adapters is not safe here:
  peft's active adapter and `disable_adapter()` are state of the shared module, so two policies on
  one base would change each other's behaviour. Stages that need several arms should load the
  base once per arm or reuse one policy with `load_adapter`.
* **Determinism**: on CPU, the same `seed` and the same generator seed give identical samples
  (tested). On MPS the draws still come from the CPU generator, but Metal kernels are not
  guaranteed bit-identical across runs or batch shapes, so an occasional sample can differ.
"""
from __future__ import annotations

import copy
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file

from ..interfaces import Policy, Rollout, Task
from ..registry import POLICIES
from .config import LMConfig
from .device import MemoryGuard, pick_device
from .model import ModelChoice, attach_lora, choose, encode_prompts, load, sample_batch, stop_token_ids


@dataclass
class LMRollout(Rollout):
    """A `Rollout` plus the decoded completion texts and the raw completion token ids.

    `actions[i] == tuple(completion_ids[i])`; the ids include the stop token when one was
    generated. `texts` are decoded without special tokens (what `Task.verify` checks).
    """

    texts: list[str]
    completion_ids: list[list[int]]


@POLICIES.register("local_lm")
class LocalLMPolicy(Policy):
    """A pinned Hugging Face checkpoint with a trainable LoRA adapter."""

    def __init__(self, config: LMConfig, device: torch.device | str | None = None, use_fallback: bool = False,
                 seed: int = 0):
        guard = MemoryGuard(config.memory)
        guard.check_start("load model")
        if device is None:
            device = pick_device(config.model.device)
        choice = choose(config.model, use_fallback)
        model, tokenizer = load(config.model, choice, torch.device(device))
        self._setup(model, tokenizer, config, seed, choice)

    @classmethod
    def from_loaded(cls, model: torch.nn.Module, tokenizer: Any, config: LMConfig, seed: int = 0,
                    choice: ModelChoice | None = None) -> LocalLMPolicy:
        """Wrap an already-loaded (model, tokenizer) pair (tests, or reusing one load).

        A plain model gets a fresh seeded adapter; a `PeftModel` is used as is. The tokenizer must
        pad on the left with a pad token set, as `model.load` arranges.
        """
        policy = cls.__new__(cls)
        policy._setup(model, tokenizer, config, seed, choice)
        return policy

    def _setup(self, model: torch.nn.Module, tokenizer: Any, config: LMConfig, seed: int,
               choice: ModelChoice | None) -> None:
        if tokenizer.padding_side != "left" or tokenizer.pad_token_id is None:
            raise ValueError("the tokenizer must pad on the left and have a pad token (see lm/model.py: load)")
        if not isinstance(model, PeftModel):
            with torch.random.fork_rng(devices=[]):  # seeded adapter init without touching global RNG
                torch.manual_seed(seed)
                model = attach_lora(model, config.lora)
        model.eval()
        self.model: Any = model
        self.tokenizer: Any = tokenizer
        self.config = config
        self.seed = seed
        self.choice = choice
        self.guard = MemoryGuard(config.memory)
        self.device = next(model.parameters()).device
        self.stop_ids = stop_token_ids(tokenizer)
        self.end_id = int(tokenizer.eos_token_id)
        if self.end_id not in self.stop_ids:
            raise ValueError(f"eos token {self.end_id} is not a stop token {sorted(self.stop_ids)}")
        self.pad_id = int(tokenizer.pad_token_id)
        self.temperature = float(config.sampling.reference_temperature)

    # ------------------------------------------------------------------ helpers

    def _prompt_texts(self, task: Task, prompt_ids: Sequence[int]) -> list[str]:
        prompts = task.prompts()
        return [str(prompts[pid].tokens[0]) for pid in prompt_ids]

    def _chunks(self, n: int) -> Iterable[slice]:
        size = self.config.sampling.batch_size
        for lo in range(0, n, size):
            yield slice(lo, min(n, lo + size))

    def _encode(self, texts: list[str]) -> dict:
        return encode_prompts(self.tokenizer, self.config.model.system_prompt, texts, self.device)

    def _score(self, batch: dict, completions: Sequence[Sequence[int]],
               starts: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced (log_prob, entropy) sums over completion steps t >= start, with gradients.

        Prompts are left-padded and completions right-padded, so the completion occupies the same
        last columns in every row: one forward over prompt + completion[:-1] with
        `logits_to_keep = T` gives exactly the T next-token distributions that generated the
        completion, and prompt and padding positions never enter the sums.
        """
        size = len(completions)
        width = max((len(c) for c in completions), default=0)
        if width == 0:
            zeros = torch.zeros(size, device=self.device)
            return zeros, zeros.clone()
        tokens = torch.full((size, width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((size, width), dtype=torch.long)
        for i, c in enumerate(completions):
            tokens[i, : len(c)] = torch.tensor(list(c), dtype=torch.long)
            mask[i, : len(c)] = 1
        tokens, mask = tokens.to(self.device), mask.to(self.device)
        prompt_mask = batch["attention_mask"].to(dtype=mask.dtype)
        input_ids = torch.cat([batch["input_ids"], tokens[:, :-1]], dim=1)
        attention = torch.cat([prompt_mask, mask[:, :-1]], dim=1)
        positions = (attention.cumsum(-1) - 1).clamp(min=0)
        out = self.model(input_ids=input_ids, attention_mask=attention, position_ids=positions,
                         use_cache=False, logits_to_keep=width)
        logp = torch.log_softmax(out.logits.float() / self.temperature, dim=-1)  # (B, T, V)
        token_logp = logp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        entropy = -(logp.exp() * logp).sum(-1)
        steps = torch.arange(width, device=self.device).unsqueeze(0)
        start = torch.tensor(list(starts), device=self.device).unsqueeze(1)
        keep = (mask.bool() & (steps >= start)).float()
        return (token_logp * keep).sum(1), (entropy * keep).sum(1)

    def _to_ids(self, action: tuple | list | str) -> list[int]:
        """A completion as token ids: strings are tokenized without special tokens, plus the stop token."""
        if isinstance(action, str):
            return list(self.tokenizer(action, add_special_tokens=False)["input_ids"]) + [self.end_id]
        return [int(a) for a in action]

    # ------------------------------------------------------------------ Policy API

    def sample(self, task: Task, prompt_ids: Sequence[int], generator: torch.Generator,
               prefixes: Sequence[tuple] | None = None) -> LMRollout:
        """One completion per prompt id at the reference temperature; log_prob/entropy carry gradients."""
        if prefixes is not None:
            raise NotImplementedError("LocalLMPolicy does not support forced prefixes (the LM experiment has no hint arm)")
        ids = list(prompt_ids)
        texts = self._prompt_texts(task, ids)
        completions: list[list[int]] = []
        log_probs, entropies = [], []
        for part in self._chunks(len(ids)):
            batch = self._encode(texts[part])
            sampled = sample_batch(self.model, batch, self.config.model.max_new_tokens, self.temperature,
                                   self.config.sampling.top_p, self.stop_ids, self.pad_id, generator)
            chunk = [sampled.tokens[i, : int(n)].tolist() for i, n in enumerate(sampled.lengths)]
            lp, ent = self._score(batch, chunk, [0] * len(chunk))
            completions += chunk
            log_probs.append(lp)
            entropies.append(ent)
            self.guard.check_running("sample")
        decoded = self.tokenizer.batch_decode(completions, skip_special_tokens=True) if completions else []
        empty = torch.zeros(0, device=self.device)
        return LMRollout(
            prompt_ids=ids,
            actions=[tuple(c) for c in completions],
            log_prob=torch.cat(log_probs) if log_probs else empty,
            entropy=torch.cat(entropies) if entropies else empty,
            n_forced=[0] * len(ids),
            texts=list(decoded),
            completion_ids=completions,
        )

    def log_prob(self, task: Task, prompt_ids: Sequence[int], actions: Sequence[Any],
                 start: int | Sequence[int] = 0) -> torch.Tensor:
        """(B,) teacher-forced log-prob of each completion (token-id tuples or strings), with gradients."""
        ids = list(prompt_ids)
        if len(actions) != len(ids):
            raise ValueError(f"{len(ids)} prompt ids but {len(actions)} actions")
        starts = [int(start)] * len(ids) if isinstance(start, int) else [int(s) for s in start]
        texts = self._prompt_texts(task, ids)
        completions = [self._to_ids(a) for a in actions]
        out = []
        for part in self._chunks(len(ids)):
            lp, _ = self._score(self._encode(texts[part]), completions[part], starts[part])
            out.append(lp)
            self.guard.check_running("log_prob")
        return torch.cat(out) if out else torch.zeros(0, device=self.device)

    @torch.no_grad()
    def sample_texts(self, task: Task, prompt_ids: Sequence[int], temperature: float, generator: torch.Generator,
                     use_adapter: bool = True) -> list[tuple[str, int]]:
        """(text, generated token count incl. stop) per prompt id, no gradients, for measurement.

        `use_adapter=False` samples the base model pi0 via peft's `disable_adapter()`.
        """
        ids = list(prompt_ids)
        texts = self._prompt_texts(task, ids)
        out: list[tuple[str, int]] = []
        for part in self._chunks(len(ids)):
            batch = self._encode(texts[part])
            if use_adapter:
                sampled = self._sample_batch(batch, temperature, generator)
            else:
                with self.model.disable_adapter():
                    sampled = self._sample_batch(batch, temperature, generator)
            for i, n in enumerate(sampled.lengths.tolist()):
                text = self.tokenizer.decode(sampled.tokens[i, :n].tolist(), skip_special_tokens=True)
                out.append((text, int(n)))
            self.guard.check_running("sample_texts")
        return out

    def _sample_batch(self, batch: dict, temperature: float, generator: torch.Generator) -> Any:
        return sample_batch(self.model, batch, self.config.model.max_new_tokens, temperature,
                            self.config.sampling.top_p, self.stop_ids, self.pad_id, generator)

    def state_logits(self, task: Task, prompt_id: int) -> np.ndarray:
        raise NotImplementedError("LM policies support sampled measurement only")

    def tempering_logits(self, task: Task, prompt_id: int) -> np.ndarray | None:
        return None

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        """The trainable LoRA parameters (the base weights are frozen)."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def state_dict(self) -> dict:
        """LoRA adapter weights only, as detached CPU tensors."""
        return {k: v.detach().to("cpu").clone().contiguous() for k, v in get_peft_model_state_dict(self.model).items()}

    def load_state_dict(self, state: dict) -> None:
        """Load adapter weights saved by `state_dict`; every adapter key must be present and known."""
        expected = set(get_peft_model_state_dict(self.model))
        if set(state) != expected:
            missing, extra = sorted(expected - set(state)), sorted(set(state) - expected)
            raise ValueError(f"adapter keys do not match: missing {missing[:3]}..., unexpected {extra[:3]}...")
        set_peft_model_state_dict(self.model, state)

    def save_adapter(self, path: str | Path) -> Path:
        """Write the adapter to a safetensors file atomically (temporary file, then rename)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        save_file(self.state_dict(), str(tmp))
        os.replace(tmp, path)
        return path

    def load_adapter(self, path: str | Path) -> None:
        self.load_state_dict(load_file(str(path)))

    def clone(self) -> LocalLMPolicy:
        """Independent copy: deep-copies the model (see module docstring), shares the tokenizer."""
        new = copy.copy(self)
        new.model = copy.deepcopy(self.model)
        return new
