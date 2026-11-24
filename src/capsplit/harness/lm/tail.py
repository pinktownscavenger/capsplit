"""The base model's tail success probability p0 (exploratory; `docs/lm_main/tail/preregistration_tail.md`).

For a problem x, p0 is the probability that the base model (LoRA adapter disabled), sampled at the
reference temperature for at most `model.max_new_tokens` tokens, writes a completion the strict
checker (`tasks.check`) accepts. This module brackets p0 exactly and estimates the rest by
importance sampling. Sequence-level importance sampling for rare LM outputs is not new (Angell et
al., "Estimating Tail Risks in Language Model Output Distributions", ICML 2026, arXiv:2604.22167);
here it is applied to per-problem correctness.

* **Exact lower bound D** (`d_bound`): pi0(answer digits) x pi0(next token ends the integer). The
  Qwen2.5 tokenizer writes every digit as one token and no other token contains a digit (checked by
  `build_vocab`), so the digit path is unique.
* **Exact bracket** (`expand_tree`): best-first partial expansion of the token tree. Every prefix is
  won (the checker accepts every continuation), dead (it accepts none) or open (`prefix_status`), so
  p0 lies in [won mass, won mass + open mass]. Tokens other than digits and stop tokens are grouped
  by the character classes of their first 6 decoded characters (`Vocab`), which is exact because the
  checker's regular expression only distinguishes digits, '.', ',', the two minus signs and
  "anything else", and looks at most 5 characters past a match.
* **Importance sampling of the open remainder** (`is_chunk`, `estimate`): a defensive mixture of
  proposals (base, hot base, SFT adapter, half-scaled RL adapter), the plain (not self-normalised)
  estimator, applied only to completions the tree did not resolve (`classify_path`).

All model work is under `torch.no_grad()`. The numbers are those of the run's float16 model on MPS
with float32 log-softmax (DECISIONS L37).
"""
from __future__ import annotations

import heapq
import itertools
import json
import math
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .config import LMConfig
from .storage import Progress, append_jsonl, read_jsonl, write_manifest
from .tasks import _NUMBER, check

# --------------------------------------------------------------------------- config


class TailConfigError(ValueError):
    """The tail config is invalid; the message says which key and what is expected."""


@dataclass(frozen=True)
class TailConfig:
    """Settings of the tail layer (`configs/lm_tail.json`); every key is required to be known."""

    budget: int = 4096  # node expansions per problem (bracket)
    children: int = 64  # explicit open children per expanded node (besides digits)
    batch: int = 64  # prefixes per forward (bracket)
    n_is: int = 1024  # importance samples per problem
    weights: tuple = (0.25, 0.25, 0.25, 0.25)  # mixture: base, hot base, sft, scaled rl
    hot_temperature: float = 1.3
    sft: tuple = ("sft", 0, 200)  # arm, seed, step of the SFT component's adapter
    rl: tuple = ("rl", 0, 200)  # arm, seed, step of the RL component's adapter
    rl_scale: float = 0.5  # lora_B multiplier of the RL component
    n_bf: int = 4000  # brute-force base samples per control problem
    n_bf_problems: int = 12
    bf_range: tuple = (0.005, 0.06)  # p-hat range (560 T = 1 base samples) for brute-force controls
    alpha: float = 0.05
    seed: int = 0
    grid_max_exp: int = 20  # budget curve N = 2^k up to 2^grid_max_exp
    trust_min_success: int = 20
    trust_min_ess: float = 10.0
    trust_max_share: float = 0.2
    trust_max_k: float = 0.7
    z_known: float = 3.5
    known_min_trusted: int = 5
    z_bf: float = 3.0
    bf_min_trusted: int = 6
    gate1_min_p: float = 0.01
    gate1_ratio: float = 0.1
    max_problems: int | None = None  # tests only: restrict to the first problems


def parse_tail_config(raw: dict) -> TailConfig:
    """Strict parse: unknown keys raise, lists become tuples, basic ranges are checked."""
    if not isinstance(raw, dict):
        raise TailConfigError("the tail config must be a JSON object")
    names = {f.name for f in fields(TailConfig)}
    unknown = sorted(set(raw) - names)
    if unknown:
        raise TailConfigError(f"unknown tail config keys {unknown}; allowed: {sorted(names)}")
    values = {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}
    cfg = TailConfig(**values)
    if len(cfg.weights) != 4 or any(w <= 0 for w in cfg.weights) or not math.isclose(sum(cfg.weights), 1.0):
        raise TailConfigError(f"weights must be 4 positive numbers summing to 1, got {cfg.weights!r}")
    for key in ("budget", "children", "batch", "n_is", "n_bf", "n_bf_problems"):
        value = getattr(cfg, key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise TailConfigError(f"{key} must be a positive integer, got {value!r}")
    if len(cfg.sft) != 3 or len(cfg.rl) != 3:
        raise TailConfigError("sft and rl must be [arm, seed, step]")
    return cfg


def load_tail_config(path: str | Path) -> TailConfig:
    try:
        return parse_tail_config(json.loads(Path(path).read_text()))
    except FileNotFoundError as error:
        raise TailConfigError(f"tail config {path} does not exist") from error


# --------------------------------------------------------------------------- the checker on prefixes

DIGITS = "0123456789"
_REP = {"d": "0", ".": ".", ",": ",", "-": "-", "o": "x"}
_PROBES = tuple("".join(p) for n in range(6) for p in itertools.product("0.,-x", repeat=n))
_LEAD = re.compile(r"[-−]?([0-9]+)")


def char_class(c: str) -> str:
    """The only distinctions the checker's regular expression makes: digit, '.', ',', minus, other."""
    if c in DIGITS:
        return "d"
    if c in ".,":
        return c
    if c in "-−":
        return "-"
    return "o"


def class_rep(text: str) -> str:
    """`text` with every character replaced by its class representative ('0', '.', ',', '-', 'x')."""
    return "".join(_REP[char_class(c)] for c in text)


@lru_cache(maxsize=200_000)
def _final_tail(tail: str) -> bool:
    """True iff the number matched at the start of `tail` is the same for every extension of `tail`.

    Probes every extension of up to 5 characters over the class representatives: the regular
    expression looks at most 5 characters past a match (",ddd" plus a one-character look-ahead).
    """
    base = _NUMBER.match(tail)
    if base is None:
        return False
    key = (base.end(), base.group(1))
    for s in _PROBES:
        m = _NUMBER.match(tail + s)
        if m is None or (m.end(), m.group(1)) != key:
            return False
    return True


def number_is_final(text: str) -> bool:
    """Whether the first number in `text` (if any) can no longer change, whatever is appended."""
    m = _NUMBER.search(text)
    if m is None:
        return False
    if len(text) - m.end() >= 6:  # every character the expression could still inspect is fixed
        return True
    return _final_tail(text[m.start():])


def prefix_status(text: str, answer: str) -> str:
    """'won', 'dead' or 'open' for a completion prefix whose generation has not ended.

    dead: the first number is negative, or its leading digits (leading zeros stripped) are not a
    prefix of the answer, or it is final and wrong; won: it is final and correct; open otherwise.
    """
    m = _NUMBER.search(text)
    if m is None:
        return "open"
    lead = _LEAD.match(text, m.start())
    assert lead is not None  # the expression starts with an optional minus and a digit
    if lead.group(0)[0] in "-−":
        return "dead"
    if not answer.startswith(lead.group(1).lstrip("0")):
        return "dead"
    if number_is_final(text):
        return "won" if check(text, answer)[1] else "dead"
    return "open"


def terminal_status(text: str, answer: str) -> str:
    """'won' or 'dead' for a finished completion (stop token drawn or token limit reached)."""
    return "won" if check(text, answer)[1] else "dead"


# --------------------------------------------------------------------------- vocabulary


@dataclass
class Vocab:
    """The model's output rows, partitioned for the bracket.

    `stop`: stop-token ids; `digits`: {id: digit}; `empty`: tokens that decode to "" with special
    tokens skipped, and padded logit rows beyond the tokenizer (their text adds nothing); every other
    id has a group, the class representative of its first 6 decoded characters (`reps[group]`).
    `ends[g]` is True when group g's first character ends an integer (not a digit, '.' or ',').
    """

    size: int
    n_tokens: int
    stop: np.ndarray
    digits: dict[int, str]
    empty: np.ndarray
    group_of: np.ndarray
    reps: list[str]
    ends: np.ndarray = field(repr=False)


def build_vocab(tokenizer: Any, size: int, stop_ids: set[int]) -> Vocab:
    """Partition `size` output rows. Raises if any token other than a single digit contains a digit."""
    n_tokens = len(tokenizer)
    group_of = np.full(size, -1, dtype=np.int64)
    digits: dict[int, str] = {}
    empty: list[int] = []
    reps: dict[str, int] = {}
    for i in range(size):
        if i in stop_ids:
            continue
        if i >= n_tokens:
            empty.append(i)
            continue
        piece = tokenizer.decode([i], skip_special_tokens=True)
        if piece == "":
            empty.append(i)
            continue
        if any(c in DIGITS for c in piece):
            if len(piece) == 1:
                digits[i] = piece
                continue
            raise ValueError(f"token {i} ({piece!r}) contains a digit but is not a single digit; "
                             "the bracket assumes digits are tokenized one per token")
        rep = class_rep(piece[:6])
        group_of[i] = reps.setdefault(rep, len(reps))
    rep_list = [""] * len(reps)
    for rep, g in reps.items():
        rep_list[g] = rep
    if sorted(digits.values()) != list(DIGITS):
        raise ValueError(f"expected one token per digit, found {sorted(digits.values())}")
    ends = np.array([r[0] in "x-" for r in rep_list], dtype=bool)
    return Vocab(size, n_tokens, np.array(sorted(stop_ids), dtype=np.int64), digits, np.array(empty, dtype=np.int64),
                 group_of, rep_list, ends)


def end_mask(vocab: Vocab) -> np.ndarray:
    """Boolean mask of the ids that certainly end an integer: stop tokens and groups in `ends`."""
    mask = np.zeros(vocab.size, dtype=bool)
    mask[vocab.stop] = True
    grouped = vocab.group_of >= 0
    mask[grouped] = vocab.ends[vocab.group_of[grouped]]
    return mask


def answer_ids(vocab: Vocab, answer: str) -> list[int]:
    by_digit = {d: i for i, d in vocab.digits.items()}
    return [by_digit[c] for c in answer]


# --------------------------------------------------------------------------- model access


def prompt_ids(policy: Any, task: Any, pid: int) -> list[int]:
    """The chat-formatted prompt of problem `pid` as token ids (as `LocalLMPolicy` encodes it)."""
    from .model import encode_prompts

    batch = encode_prompts(policy.tokenizer, policy.config.model.system_prompt,
                           [str(task.prompts()[pid].tokens[0])], policy.device)
    return [int(t) for t in batch["input_ids"][0].tolist()]


def forward_logprobs(policy: Any, prompt: Sequence[int], prefixes: Sequence[Sequence[int]], keep: int,
                     temperatures: Sequence[float] = (1.0,), use_adapter: bool = False) -> list[np.ndarray]:
    """Log-probs (float64, CPU) of the next token at the last `keep` positions of prompt + prefix.

    Returns one array (B, keep, V) per temperature: log softmax(logits / T) in float32, as
    `LocalLMPolicy` scores. Rows are left-padded; position ids count only real tokens. With
    `use_adapter=False` the adapter is disabled (pi0).
    """
    import torch

    seqs = [list(prompt) + [int(t) for t in p] for p in prefixes]
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), int(policy.pad_id), dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, width - len(s):] = torch.tensor(s, dtype=torch.long)
        mask[i, width - len(s):] = 1
    ids, mask = ids.to(policy.device), mask.to(policy.device)
    positions = (mask.cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        if use_adapter:
            out = policy.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False,
                               logits_to_keep=keep)
        else:
            with policy.model.disable_adapter():
                out = policy.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False,
                                   logits_to_keep=keep)
        logits = out.logits.float()
        return [torch.log_softmax(logits / float(t), dim=-1).cpu().double().numpy() for t in temperatures]


def decoder(tokenizer: Any, n_tokens: int) -> Callable[[Sequence[int]], str]:
    """Completion text as the run decodes it (special tokens skipped); padded rows decode to ""."""

    def decode(ids: Sequence[int]) -> str:
        return str(tokenizer.decode([int(i) for i in ids if int(i) < n_tokens], skip_special_tokens=True))

    return decode


# --------------------------------------------------------------------------- Step 1: D


def d_bound(logp_path: np.ndarray, digit_ids: Sequence[int], ends: np.ndarray, max_new: int) -> dict[str, float]:
    """D = prod_i pi0(d_i | d_<i) x pi0(E | digits), from next-token log-probs along the digit path.

    `logp_path` is (len(digits) + 1, V): row i is the distribution after the first i digits.
    When the digits already use all `max_new` tokens the end factor is 1. Also returns the
    first-token digit probability.
    """
    k = len(digit_ids)
    if k > max_new:
        return {"log_digits": -math.inf, "end_mass": 0.0, "D": 0.0, "log_D": -math.inf, "first_digit": math.nan}
    log_digits = float(sum(logp_path[i, d] for i, d in enumerate(digit_ids)))
    end = 1.0 if k == max_new else float(np.exp(logp_path[k][ends]).sum())
    log_d = log_digits + (math.log(end) if end > 0 else -math.inf)
    return {"log_digits": log_digits, "end_mass": end, "D": math.exp(log_d), "log_D": log_d,
            "first_digit": math.nan}


def in_d_set(y: Sequence[int], digit_ids: Sequence[int], ends: np.ndarray, max_new: int) -> bool:
    """Whether completion `y` starts with the answer digits followed by a token that ends the integer."""
    k = len(digit_ids)
    if list(y[:k]) != list(digit_ids):
        return False
    if k == max_new:
        return len(y) == k
    return len(y) > k and bool(ends[int(y[k])])


# --------------------------------------------------------------------------- Step 2: bracket


@dataclass
class Bracket:
    """Masses after expansion; p0 lies in [won, won + open_frontier + open_lumped]."""

    won: float = 0.0
    dead: float = 0.0
    open_frontier: float = 0.0
    open_lumped: float = 0.0
    expansions: int = 0
    expanded: list = field(default_factory=list)
    frontier_size: int = 0

    @property
    def open(self) -> float:
        return self.open_frontier + self.open_lumped

    @property
    def total(self) -> float:
        return self.won + self.dead + self.open


def _status(text: str, answer: str, terminal: bool) -> str:
    return terminal_status(text, answer) if terminal else prefix_status(text, answer)


def expand_tree(score: Callable[[list[tuple]], np.ndarray], answer: str, vocab: Vocab,
                decode: Callable[[Sequence[int]], str], max_new: int, budget: int, children: int,
                batch: int) -> Bracket:
    """Best-first partial expansion of the completion tree under pi0 (see the module docstring).

    `score(prefixes)` returns (B, V) next-token log-probs (float64) after each prefix. Digit and
    stop children are always explicit; other tokens are classified by group, and the `children`
    most probable open ones become explicit nodes, the rest of the open mass is kept as unexpanded
    ("lumped") open mass. Nodes are expanded in order of prefix probability until `budget`.
    """
    out = Bracket()
    counter = itertools.count()
    heap: list[tuple[float, int, tuple]] = [(-0.0, next(counter), ())]
    digit_items = sorted(vocab.digits.items())
    grouped = vocab.group_of >= 0
    gids = vocab.group_of[grouped]
    n_groups = len(vocab.reps)
    while heap and out.expansions < budget:
        take = min(batch, budget - out.expansions, len(heap))
        nodes = [heapq.heappop(heap) for _ in range(take)]
        rows = score([n[2] for n in nodes])
        for (neg, _, prefix), row in zip(nodes, rows):
            out.expansions += 1
            out.expanded.append(list(prefix))
            log_p = -neg
            mass = math.exp(log_p)
            probs = np.exp(row)
            text = decode(prefix)
            terminal = len(prefix) + 1 >= max_new
            # stop tokens end the completion here
            stop_mass = float(probs[vocab.stop].sum())
            if terminal_status(text, answer) == "won":
                out.won += mass * stop_mass
            else:
                out.dead += mass * stop_mass
            # tokens that add no text: same status as this (open) prefix, or terminal
            empty_mass = float(probs[vocab.empty].sum())
            if terminal:
                if terminal_status(text, answer) == "won":
                    out.won += mass * empty_mass
                else:
                    out.dead += mass * empty_mass
            else:
                out.open_lumped += mass * empty_mass
            for tid, ch in digit_items:
                s = _status(text + ch, answer, terminal)
                p = float(probs[tid])
                if s == "won":
                    out.won += mass * p
                elif s == "dead":
                    out.dead += mass * p
                elif p > 0:
                    heapq.heappush(heap, (-(log_p + float(row[tid])), next(counter), prefix + (tid,)))
            group_mass = np.bincount(gids, weights=probs[grouped], minlength=n_groups)
            if _NUMBER.search(text) is None and not terminal:
                statuses = ["open"] * n_groups  # no digits in these tokens: still no number
            elif _NUMBER.search(text) is None:
                statuses = ["dead"] * n_groups  # the completion ends without a number
            else:
                statuses = [_status(text + rep, answer, terminal) for rep in vocab.reps]
            st_arr = np.array(statuses)
            out.won += mass * float(group_mass[st_arr == "won"].sum())
            out.dead += mass * float(group_mass[st_arr == "dead"].sum())
            open_groups = st_arr == "open"
            open_total = float(group_mass[open_groups].sum())
            if open_total > 0:
                token_open = np.zeros(vocab.size, dtype=bool)
                token_open[grouped] = open_groups[gids]
                cand = np.flatnonzero(token_open)
                k = min(children, len(cand))
                top = cand[np.argpartition(-probs[cand], k - 1)[:k]] if k > 0 else cand[:0]
                explicit = 0.0
                for tid in top.tolist():
                    p = float(probs[tid])
                    if p <= 0:
                        continue
                    explicit += p
                    heapq.heappush(heap, (-(log_p + float(row[tid])), next(counter), prefix + (int(tid),)))
                out.open_lumped += mass * max(open_total - explicit, 0.0)
    out.frontier_size = len(heap)
    out.open_frontier = float(sum(math.exp(-n[0]) for n in heap))
    return out


def classify_path(y: Sequence[int], expanded: set[tuple], answer: str,
                  decode: Callable[[Sequence[int]], str], max_new: int, stop: set[int]) -> str:
    """Where completion `y` sits in an expanded tree: 'won' or 'dead' (resolved), or 'rem'.

    'rem' (remainder) means its path leaves the expanded nodes through an open child (an unexpanded
    explicit node or lumped open mass), so it is not counted in the tree's won mass. Unexpanded
    explicit children are exactly the open ones, so only the expanded set is needed.
    """
    y = tuple(int(t) for t in y)
    for i in range(len(y)):
        child = y[: i + 1]
        if child in expanded:
            continue
        if y[i] in stop:
            return terminal_status(decode(y[:i]), answer)
        text = decode(child)
        if len(child) >= max_new:
            return terminal_status(text, answer)
        s = prefix_status(text, answer)
        return "rem" if s == "open" else s
    return "rem"


# --------------------------------------------------------------------------- Step 3: importance sampling


def sample_with_logq(model: Any, batch: dict, max_new_tokens: int, temperature: float, stop_ids: set[int],
                     pad_id: int, generator: Any) -> tuple[list[list[int]], list[list[float]]]:
    """`model.sample_batch` (same draws for the same generator), also returning each chosen token's
    log q = log softmax(logits / T) in float32, as computed at sampling time (KV-cache path)."""
    import torch

    with torch.no_grad():
        device = batch["input_ids"].device
        mask = batch["attention_mask"]
        positions = (mask.cumsum(-1) - 1).clamp(min=0)
        out = model(input_ids=batch["input_ids"], attention_mask=mask, position_ids=positions, use_cache=True,
                    logits_to_keep=1)
        past = out.past_key_values
        size = batch["input_ids"].shape[0]
        finished = torch.zeros(size, dtype=torch.bool)
        tokens: list[list[int]] = [[] for _ in range(size)]
        logq: list[list[float]] = [[] for _ in range(size)]
        for _ in range(max_new_tokens):
            logits = out.logits[:, -1].float().cpu()
            probs = torch.softmax(logits / temperature, dim=-1)
            token = torch.multinomial(probs, 1, generator=generator).squeeze(1)
            lp = torch.log_softmax(logits / temperature, dim=-1).gather(1, token.unsqueeze(1)).squeeze(1)
            for i in range(size):
                if not bool(finished[i]):
                    tokens[i].append(int(token[i]))
                    logq[i].append(float(lp[i]))
            token = torch.where(finished, torch.full_like(token, pad_id), token)
            finished |= torch.tensor([int(t) in stop_ids for t in token.tolist()])
            if bool(finished.all()):
                break
            mask = torch.cat([mask, torch.ones((size, 1), dtype=mask.dtype, device=device)], dim=1)
            positions = positions[:, -1:] + 1
            out = model(input_ids=token.unsqueeze(1).to(device), attention_mask=mask, position_ids=positions,
                        past_key_values=past, use_cache=True)
            past = out.past_key_values
    return tokens, logq


def score_completions(policy: Any, prompt: Sequence[int], completions: Sequence[Sequence[int]],
                      temperatures: Sequence[float], use_adapter: bool) -> list[np.ndarray]:
    """Teacher-forced per-token log-probs (B, W) per temperature, NaN beyond each completion.

    One forward per chunk of `sampling.batch_size` over prompt + completion[:-1] (left-padded), the
    last W next-token distributions kept, as `LocalLMPolicy._score` does.
    """
    import torch

    width = max(len(c) for c in completions)
    size = int(policy.config.sampling.batch_size)
    outs = [np.full((len(completions), width), np.nan) for _ in temperatures]
    for lo in range(0, len(completions), size):
        part = completions[lo: lo + size]
        prefixes = [list(c)[:-1] + [int(policy.pad_id)] * (width - len(c)) for c in part]
        # right-pad inside the completion region so the last `width` positions align for every row
        seqs = [list(prompt) + p for p in prefixes]
        ids = torch.tensor(seqs, dtype=torch.long, device=policy.device)
        mask = torch.ones_like(ids)
        for i, c in enumerate(part):
            if len(c) < width:
                mask[i, len(prompt) + len(c) - 1:] = 0
        positions = (mask.cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            if use_adapter:
                out = policy.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False,
                                   logits_to_keep=width)
            else:
                with policy.model.disable_adapter():
                    out = policy.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False,
                                       logits_to_keep=width)
            logits = out.logits.float()
            for t_index, t in enumerate(temperatures):
                lp = torch.log_softmax(logits / float(t), dim=-1).cpu().double().numpy()
                for i, c in enumerate(part):
                    outs[t_index][lo + i, : len(c)] = lp[i, np.arange(len(c)), list(c)]
        policy.guard.check_running("tail score")
    return outs


def scaled_state(state: dict, scale: float) -> dict:
    """An adapter state with every lora_B multiplied by `scale` (Delta W = scale x the original)."""
    return {k: (v * scale if "lora_B" in k else v) for k, v in state.items()}


COMPONENTS = ("base", "hot", "sft", "rl_half")


def is_chunk(policy: Any, prompt: Sequence[int], n: int, weights: Sequence[float], hot_t: float,
             states: dict[str, dict], generator: Any) -> list[dict]:
    """n importance samples for one problem: component per episode, sampling, and scoring under all 4.

    Each row: index, component, ids, logq_sample (per token, at sampling time), and the
    teacher-forced log-probs `lp_base` (pi0), `lp_hot`, `lp_sft`, `lp_rl_half` (sums over tokens),
    plus the per-token gap between the sampled component's two log q paths.
    """
    import torch

    comp = torch.multinomial(torch.tensor(list(weights), dtype=torch.float64), n, replacement=True,
                             generator=generator).tolist()
    size = int(policy.config.sampling.batch_size)
    temps = {"base": 1.0, "hot": hot_t, "sft": 1.0, "rl_half": 1.0}
    seqs: dict[int, list[int]] = {}
    logq: dict[int, list[float]] = {}
    for c_index, name in enumerate(COMPONENTS):
        idx = [i for i, c in enumerate(comp) if c == c_index]
        if not idx:
            continue
        if name in states:
            policy.load_state_dict(states[name])
        for lo in range(0, len(idx), size):
            part = idx[lo: lo + size]
            ids = torch.tensor([list(prompt)] * len(part), dtype=torch.long, device=policy.device)
            batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            if name in states:
                toks, lq = sample_with_logq(policy.model, batch, policy.config.model.max_new_tokens, temps[name],
                                            policy.stop_ids, policy.pad_id, generator)
            else:
                with policy.model.disable_adapter():
                    toks, lq = sample_with_logq(policy.model, batch, policy.config.model.max_new_tokens,
                                                temps[name], policy.stop_ids, policy.pad_id, generator)
            for i, t, q in zip(part, toks, lq):
                seqs[i], logq[i] = t, q
            policy.guard.check_running("tail sample")
    completions = [seqs[i] for i in range(n)]
    base, hot = score_completions(policy, prompt, completions, (1.0, hot_t), use_adapter=False)
    scored = {"base": base, "hot": hot}
    for name in ("sft", "rl_half"):
        policy.load_state_dict(states[name])
        scored[name] = score_completions(policy, prompt, completions, (1.0,), use_adapter=True)[0]
    rows = []
    for i in range(n):
        name = COMPONENTS[comp[i]]
        k = len(completions[i])
        tf = scored[name][i, :k]
        gap = np.abs(tf - np.array(logq[i]))
        rows.append({"index": i, "component": name, "ids": completions[i], "logq_sample": logq[i],
                     **{f"lp_{c}": float(np.nansum(scored[c][i, :k])) for c in COMPONENTS},
                     "gap_max": float(gap.max()), "gap_mean": float(gap.mean())})
    return rows


def mixture_logq(rows_lp: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    """log sum_c w_c q_c(y) from an (n, 4) array of component log-probs (numerically stable)."""
    a = rows_lp + np.log(np.asarray(weights, dtype=float))[None, :]
    top = a.max(axis=1, keepdims=True)
    return (top + np.log(np.exp(a - top).sum(axis=1, keepdims=True)))[:, 0]


def gpd_khat(x: np.ndarray) -> float:
    """Zhang & Stephens (2009) generalised Pareto shape estimate for exceedances `x` > 0, with the
    weakly informative prior used in Pareto-smoothed importance sampling (Vehtari et al.)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n < 2 or x[-1] <= 0:
        return math.nan
    prior_bs, prior_k = 3.0, 10.0
    m_est = 30 + int(n ** 0.5)
    b_ary = 1 - np.sqrt(m_est / (np.arange(1, m_est + 1, dtype=float) - 0.5))
    b_ary /= prior_bs * x[max(int(n / 4 + 0.5) - 1, 0)]
    b_ary += 1 / x[-1]
    k_ary = np.log1p(-b_ary[:, None] * x).mean(axis=1)
    len_scale = n * (np.log(-(b_ary / k_ary)) - k_ary - 1)
    weights = 1 / np.exp(len_scale - len_scale[:, None]).sum(axis=1)
    keep = weights >= 10 * np.finfo(float).eps
    weights, b_ary = weights[keep], b_ary[keep]
    weights /= weights.sum()
    b_post = float(np.sum(b_ary * weights))
    k_post = float(np.log1p(-b_post * x).mean())
    return float((n * k_post + prior_k * 0.5) / (n + prior_k))


def pareto_k(w: np.ndarray) -> float:
    """k-hat from the largest ceil(min(0.2 n, 3 sqrt n)) of the weights `w`.

    NaN when the tail has fewer than 4 weights. Exceedances equal to 0 (ties at the threshold, e.g. the
    same completion drawn repeatedly) are dropped; with fewer than 4 positive ones left the tail is
    degenerate (a few distinct values) and k-hat is -inf, "no evidence of a heavy tail" (DECISIONS L38);
    single large weights are caught by the max-share rule instead.
    """
    w = np.sort(np.asarray(w, dtype=float))
    n = len(w)
    m = math.ceil(min(0.2 * n, 3 * math.sqrt(n)))
    if m < 4 or n <= m:
        return math.nan
    tail = w[-m:] - w[-m - 1]
    tail = tail[tail > 0]
    if len(tail) < 4:
        return -math.inf
    return gpd_khat(tail)


def trust(w: np.ndarray, cfg: TailConfig) -> dict[str, Any]:
    """The pre-registered trust rule on the weights of the contributing (successful) samples."""
    w = np.asarray(w, dtype=float)
    n_s = len(w)
    total = float(w.sum())
    ess = float(total ** 2 / np.sum(w ** 2)) if n_s and total > 0 else 0.0
    share = float(w.max() / total) if n_s and total > 0 else math.nan
    k = pareto_k(w) if n_s else math.nan
    checks = {"n_success": n_s >= cfg.trust_min_success, "ess": ess >= cfg.trust_min_ess,
              "max_share": bool(n_s) and share <= cfg.trust_max_share,
              "pareto_k": bool(n_s) and (k <= cfg.trust_max_k if not math.isnan(k) else False)}
    return {"n_success": n_s, "ess": ess, "max_share": share, "k_hat": k, "checks": checks,
            "trusted": all(checks.values())}


def estimate(indicator: np.ndarray, log_w: np.ndarray, offset: float = 0.0) -> dict[str, float]:
    """Plain importance-sampling estimate offset + mean(1 x w), with SE = sd / sqrt(n) of the summands."""
    w = np.exp(np.asarray(log_w, dtype=float))
    summand = np.where(np.asarray(indicator, dtype=bool), w, 0.0)
    n = len(summand)
    se = float(summand.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
    return {"estimate": offset + float(summand.mean()), "remainder": float(summand.mean()), "se": se, "n": n}


# --------------------------------------------------------------------------- reference samples and checks


def clopper_pearson_level(k: np.ndarray, n: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided Clopper-Pearson interval with total miss probability 1 - level."""
    from .measure import clopper_pearson

    return clopper_pearson(np.asarray(k), np.asarray(n), 1.0 - level)


def base_reference(out: Path, config: LMConfig, n_problems: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Successes and trials per problem over every base sample drawn at the reference temperature
    (sets A, B, C at T_ref, null): the draws from pi0 (DECISIONS L36)."""
    from . import stages as st

    ref = float(config.sampling.reference_temperature)
    k = np.zeros(n_problems, dtype=np.int64)
    m = np.zeros(n_problems, dtype=np.int64)
    used = {}
    for set_name in ("A", "B", "C", "null"):
        path = st.sample_path(out, "base", set_name, ref)
        if not path.exists():
            continue
        count = 0
        for row in read_jsonl(path):
            pid = int(row["problem"])
            if pid < n_problems:
                k[pid] += bool(row["correct"])
                m[pid] += 1
                count += 1
        used[set_name] = count
    return k, m, used


def select_bf_problems(p_hat: np.ndarray, cfg: TailConfig) -> list[int]:
    """Brute-force control problems: p-hat in `bf_range`, sorted by p-hat (ties by id), evenly spaced ranks."""
    lo, hi = cfg.bf_range
    eligible = sorted((float(p), int(i)) for i, p in enumerate(p_hat) if lo <= p <= hi)
    if len(eligible) <= cfg.n_bf_problems:
        return sorted(i for _, i in eligible)
    ranks = np.linspace(0, len(eligible) - 1, cfg.n_bf_problems).round().astype(int)
    return sorted(eligible[r][1] for r in sorted(set(ranks.tolist())))


# --------------------------------------------------------------------------- the stage


def tail_paths(tail_out: Path) -> dict[str, Path]:
    return {"d": tail_out / "d.jsonl", "bracket": tail_out / "bracket.jsonl", "trees": tail_out / "trees",
            "is": tail_out / "is", "bf": tail_out / "bf", "checks": tail_out / "checks.json",
            "selection": tail_out / "selection.json"}


def _done(path: Path) -> set[int]:
    return {int(r["problem"]) for r in read_jsonl(path)} if path.exists() else set()


def bracket_check(k: np.ndarray, m: np.ndarray, lo: np.ndarray, hi: np.ndarray, d: np.ndarray,
                  alpha: float) -> dict[str, Any]:
    """CP checks: D below the one-sided upper bound, and the bracket meeting the two-sided interval,
    both at level 1 - alpha / P (Bonferroni over the P problems); unadjusted counts too."""
    p = len(k)
    _, cp_hi_b = clopper_pearson_level(k, m, 1.0 - 2 * alpha / p)  # one-sided upper at alpha / P
    d_fail = np.flatnonzero(cp_hi_b < d)
    cp_lo2, cp_hi2 = clopper_pearson_level(k, m, 1.0 - alpha / p)
    meet_fail = np.flatnonzero((cp_hi2 < lo) | (cp_lo2 > hi))
    _, cp_hi_1 = clopper_pearson_level(k, m, 1.0 - 2 * alpha)  # one-sided 95% upper
    cp_lo_u, cp_hi_u = clopper_pearson_level(k, m, 1.0 - alpha)  # two-sided 95%
    return {"d_fail": d_fail.tolist(), "d_fail_unadjusted": np.flatnonzero(cp_hi_1 < d).tolist(),
            "bracket_fail": meet_fail.tolist(),
            "bracket_fail_unadjusted": np.flatnonzero((cp_hi_u < lo) | (cp_lo_u > hi)).tolist(),
            "level": 1.0 - alpha / p}


def run_tail(config: LMConfig, tcfg: TailConfig, out: Path, tail_out: Path, device: Any = "cpu",
             factory: Callable[[int], Any] | None = None, use_fallback: bool = False,
             force: bool = False) -> list[Path]:
    """Steps 1-3 for every problem (D, bracket) and for the IS set (target + brute-force controls).

    Resumable per problem. Reads `out`, writes only `tail_out`. Stops before importance sampling if
    the bracket check fails (a bug to investigate), unless `force`.
    """
    import torch
    from safetensors.torch import load_file

    from . import stages as st

    st._plugins()
    chosen = st.model_choice(config, use_fallback)
    fp = st.fingerprint(st.plain_config(config), chosen, "train")
    paths = tail_paths(tail_out)
    with st._stage(config, "tail", device):
        st._check_upstream(out, fp, base=True)
        task = st.load_task(config, out)
        n_problems = len(task.prompts())
        problems = list(range(n_problems if tcfg.max_problems is None else min(n_problems, tcfg.max_problems)))
        classes = json.loads((out / "base" / "classes.json").read_text())["classes"]
        k_ref, m_ref, used = base_reference(out, config, n_problems)
        manifest = write_manifest(tail_out, "tail", config.to_dict(), tcfg.seed, str(device),
                                  extra={"tail_config": asdict(tcfg), "model": chosen, "source": str(out),
                                         "reference_sets": used, "problems": len(problems)})
        policy = (factory or st.default_policy_factory(config, device, use_fallback))(0)
        stop = {int(s) for s in policy.stop_ids}
        size = int(policy.model.get_output_embeddings().weight.shape[0])
        vocab = build_vocab(policy.tokenizer, size, stop)
        ends = end_mask(vocab)
        decode = decoder(policy.tokenizer, vocab.n_tokens)
        max_new = int(config.model.max_new_tokens)
        answers = {pid: task.answer(pid) for pid in problems}
        prompts = {pid: prompt_ids(policy, task, pid) for pid in problems}

        # Step 1: D
        have = _done(paths["d"])
        for pid in problems:
            if pid in have:
                continue
            digits = answer_ids(vocab, answers[pid])
            path_ids = digits[:max_new]
            lp = forward_logprobs(policy, prompts[pid], [path_ids], keep=len(path_ids) + 1)[0][0]
            row = d_bound(lp, digits, ends, max_new)
            row["first_digit"] = float(np.exp(lp[0][list(vocab.digits)]).sum())
            append_jsonl(paths["d"], [{"problem": pid, "answer": answers[pid], **row}])

        # Step 2: bracket
        have = _done(paths["bracket"])
        todo = [pid for pid in problems if pid not in have]
        progress = Progress("tail bracket", len(problems), len(problems) - len(todo))
        try:
            for j, pid in enumerate(todo):
                def score(prefixes: list[tuple], pid: int = pid) -> np.ndarray:
                    return forward_logprobs(policy, prompts[pid], prefixes, keep=1)[0][:, 0, :]

                b = expand_tree(score, answers[pid], vocab, decode, max_new, tcfg.budget, tcfg.children, tcfg.batch)
                paths["trees"].mkdir(parents=True, exist_ok=True)
                (paths["trees"] / f"{pid}.json").write_text(json.dumps({"expanded": b.expanded}))
                append_jsonl(paths["bracket"], [{"problem": pid, "won": b.won, "dead": b.dead,
                                                 "open_frontier": b.open_frontier, "open_lumped": b.open_lumped,
                                                 "open": b.open, "total": b.total, "expansions": b.expansions,
                                                 "frontier_size": b.frontier_size}])
                policy.guard.check_running("tail bracket")
                progress.update(len(problems) - len(todo) + j + 1)
        finally:
            progress.close()

        d_rows = {int(r["problem"]): r for r in read_jsonl(paths["d"])}
        b_rows = {int(r["problem"]): r for r in read_jsonl(paths["bracket"])}
        ids = np.array(problems)
        d_arr = np.array([d_rows[p]["D"] for p in problems])
        lo = np.maximum(d_arr, np.array([b_rows[p]["won"] for p in problems]))
        hi = np.array([b_rows[p]["won"] + b_rows[p]["open"] for p in problems])
        checks = bracket_check(k_ref[ids], m_ref[ids], lo, hi, d_arr, tcfg.alpha)
        checks["d_fail"] = [problems[i] for i in checks["d_fail"]]
        checks["bracket_fail"] = [problems[i] for i in checks["bracket_fail"]]
        checks["d_fail_unadjusted"] = [problems[i] for i in checks["d_fail_unadjusted"]]
        checks["bracket_fail_unadjusted"] = [problems[i] for i in checks["bracket_fail_unadjusted"]]
        paths["checks"].write_text(json.dumps(checks, indent=2))
        if (checks["bracket_fail"] or checks["d_fail"]) and not force:
            raise st.StageError(f"tail: the exact bounds disagree with the saved base samples (D above the CP upper "
                                f"bound for {checks['d_fail']}, bracket outside the CP interval for "
                                f"{checks['bracket_fail']}): a bug to investigate before importance sampling. "
                                f"Details: {paths['checks']}")

        # Step 3: importance sampling (target set + brute-force controls), then brute force
        p_hat = k_ref / np.maximum(m_ref, 1)
        target = [p for p in problems if classes[str(p)] in ("unreachable", "ambiguous")]
        bf = [p for p in select_bf_problems(p_hat, tcfg) if p in problems]
        paths["selection"].write_text(json.dumps({"target": target, "bf": bf}, indent=2))
        is_set = sorted(set(target) | set(bf))
        states: dict[str, dict] = {}
        arm, seed, step = tcfg.sft
        states["sft"] = load_file(str(st.adapter_path(out, arm, int(seed), int(step))))
        arm, seed, step = tcfg.rl
        states["rl_half"] = scaled_state(load_file(str(st.adapter_path(out, arm, int(seed), int(step)))),
                                         tcfg.rl_scale)
        ref = float(config.sampling.reference_temperature)
        todo = [p for p in is_set if not (paths["is"] / f"{p}.jsonl").exists()]
        progress = Progress("tail is", len(is_set), len(is_set) - len(todo))
        try:
            for j, pid in enumerate(todo):
                gen = torch.Generator().manual_seed(st.chunk_seed(tcfg.seed, "tail/is", "is", ref, pid))
                rows = is_chunk(policy, prompts[pid], tcfg.n_is, tcfg.weights, tcfg.hot_temperature, states, gen)
                tree = json.loads((paths["trees"] / f"{pid}.json").read_text())
                expanded = {tuple(x) for x in tree["expanded"]}
                digits = answer_ids(vocab, answers[pid])
                for r in rows:
                    r["problem"] = pid
                    r["text"] = decode(r["ids"])
                    r["correct"] = bool(check(r["text"], answers[pid])[1])
                    r["region"] = classify_path(r["ids"], expanded, answers[pid], decode, max_new, stop)
                    r["in_d"] = in_d_set(r["ids"], digits, ends, max_new)
                tmp = paths["is"] / f"{pid}.jsonl.tmp"
                paths["is"].mkdir(parents=True, exist_ok=True)
                tmp.unlink(missing_ok=True)
                append_jsonl(tmp, rows)
                tmp.replace(paths["is"] / f"{pid}.jsonl")
                progress.update(len(is_set) - len(todo) + j + 1)
        finally:
            progress.close()
        todo = [p for p in bf if not (paths["bf"] / f"{p}.json").exists()]
        progress = Progress("tail brute force", len(bf), len(bf) - len(todo))
        try:
            for j, pid in enumerate(todo):
                gen = torch.Generator().manual_seed(st.chunk_seed(tcfg.seed, "tail/bf", "bf", ref, pid))
                size = int(policy.config.sampling.batch_size)
                correct = 0
                for lo_i in range(0, tcfg.n_bf, size):
                    count = min(tcfg.n_bf, lo_i + size) - lo_i
                    ids_t = torch.tensor([prompts[pid]] * count, dtype=torch.long, device=policy.device)
                    with policy.model.disable_adapter():
                        toks, _ = sample_with_logq(policy.model, {"input_ids": ids_t,
                                                                  "attention_mask": torch.ones_like(ids_t)},
                                                   max_new, ref, policy.stop_ids, policy.pad_id, gen)
                    correct += sum(check(decode(t), answers[pid])[1] for t in toks)
                    policy.guard.check_running("tail brute force")
                paths["bf"].mkdir(parents=True, exist_ok=True)
                (paths["bf"] / f"{pid}.json").write_text(json.dumps({"problem": pid, "n": tcfg.n_bf,
                                                                     "correct": int(correct)}))
                progress.update(len(bf) - len(todo) + j + 1)
        finally:
            progress.close()
    print(f"tail: wrote {tail_out}", file=sys.stderr)
    return [manifest, paths["d"], paths["bracket"], paths["checks"], paths["selection"]]
