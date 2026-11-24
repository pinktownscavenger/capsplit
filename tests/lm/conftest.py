"""Shared fixtures for the LM tests: the real tokenizer (from the local cache) with a tiny random
Qwen2 model, so policy tests run on CPU in seconds, and a stub task with fixed prompts.

Tests that need the real model weights use the `lm_model` marker and skip when the weights are
not in the local Hugging Face cache.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
import torch

from capsplit.harness.interfaces import Prompt, Task
from capsplit.harness.lm.config import LMConfig, parse_config
from capsplit.harness.lm.model import choose, fetch

ROOT = Path(__file__).resolve().parents[2]
DEMO_CONFIG = ROOT / "configs" / "lm_demo.json"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "lm_model: needs the real model weights in the local Hugging Face cache")


def make_config(**sampling: object) -> LMConfig:
    """The demo config with test-friendly memory limits and optional sampling overrides."""
    raw = json.loads(DEMO_CONFIG.read_text())
    raw["memory"] = {"min_free_gb": 0.0, "max_process_gb": 1000.0}
    raw["sampling"].update(sampling)
    return parse_config(raw)


class StubTask(Task):
    """Fixed arithmetic prompts; `verify` checks the decoded text, as LMTask does."""

    n_actions = 0
    horizon = 12

    def __init__(self, questions: Sequence[tuple[str, str]]):
        self._prompts = [Prompt(i, (q,), {"split": "train"}) for i, (q, _) in enumerate(questions)]
        self._answers = [a for _, a in questions]

    def prompts(self) -> Sequence[Prompt]:
        return self._prompts

    def verify(self, prompt_id: int, actions: Sequence[int]) -> bool:
        return str(actions).strip().startswith(self._answers[prompt_id])

    def initial_state(self, prompt_id: int) -> int:
        return 0

    def step(self, prompt_id: int, state: object, action: int) -> tuple[int, bool]:
        return 0, False


QUESTIONS = (
    ("Compute 23 + 58. Reply with only the final number.", "81"),
    ("Compute 417 + 386. Reply with only the final number.", "803"),
    ("Compute 12 * 34. Reply with only the final number.", "408"),
    ("What is 7 * 8? Answer with a single integer and nothing else, please.", "56"),
)


@pytest.fixture
def task() -> StubTask:
    return StubTask(QUESTIONS)


@pytest.fixture(scope="session")
def lm_config() -> LMConfig:
    return make_config()


@pytest.fixture(scope="session")
def tokenizer(lm_config: LMConfig) -> object:
    """The pinned model's real tokenizer, from the local cache only."""
    from transformers import AutoTokenizer

    try:
        path = fetch(lm_config.model, choose(lm_config.model, False), local_only=True)
        tok = AutoTokenizer.from_pretrained(path)
    except (OSError, ValueError) as error:  # not cached (LocalEntryNotFoundError is an OSError): skip
        pytest.skip(f"tokenizer for {lm_config.model.id} is not in the local Hugging Face cache ({error})")
    tok.padding_side = "left"  # as lm/model.py: load
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="session")
def tiny_model_factory(tokenizer: object) -> Callable[[], torch.nn.Module]:
    """Build a tiny random Qwen2 (fp32, CPU), the same weights every call.

    The end-of-turn token's output row is scaled up so sampled completions end at varied lengths
    (a plain random model almost never emits it among ~150k tokens).
    """
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,  # type: ignore[arg-type]
                      num_key_value_heads=1, intermediate_size=64, max_position_embeddings=512,
                      tie_word_embeddings=False)

    def build() -> torch.nn.Module:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1234)
            model = Qwen2ForCausalLM(cfg)
            with torch.no_grad():
                model.lm_head.weight[tokenizer.eos_token_id] = torch.randn(32) * 2.5  # type: ignore[attr-defined]
        return model.eval()

    return build
