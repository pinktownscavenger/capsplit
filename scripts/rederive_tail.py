"""Independent re-derivation of three lm_main_tail numbers from raw files (no capsplit import).

1. D for problem 138 (answer digits teacher-forced after the prompt, times the end-token mass),
   recomputed on CPU in float32 with an own prompt construction and an own end-token set E;
   also on MPS float16 (the stage's numerics) when MPS is available.
2. A best-first bracket [won, won + open] of p0 for problem 56 at a smaller budget (512 expansions),
   own design: right-padded batched forwards, every token above a probability threshold classified
   exactly from the full decoded prefix with an own (conservative) prefix-status rule, all other
   tokens (and open children whose prefix probability is below PUSH) lumped as open mass.
3. Importance-sampling estimates from the saved rows only: the plain remainder estimate for
   problem 172 and the known-answer (D-set) estimate for problem 138, with `correct` and `in_d`
   recomputed by an own checker / own E set and disagreements counted.

Usage: python scripts/rederive_tail.py > docs/lm_main/tail/rederivation.txt   (run from the repo root)
"""
import heapq
import json
import math
import os
import re
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

T0 = time.time()
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REV = "7ae557604adf67be50417f59c2c2f167def9a775"
SYSTEM = "You are a precise calculator."
TEMPLATE = "Compute {question}. Reply with only the final number."
MAX_NEW = 12
STOP = (151643, 151645)
TAIL = "results/lm_main_tail"
WEIGHTS = (0.25, 0.25, 0.25, 0.25)
COMPS = ("base", "hot", "sft", "rl_half")
BUDGET = int(os.environ.get("REDERIVE_TAIL_BUDGET", "512"))  # env override only for smoke tests
BATCH = 32
TAU = 1e-10  # children with probability above TAU (given the parent) are classified exactly
PUSH = 1e-9  # open children with absolute prefix probability below PUSH are lumped (open), not queued
D_PID, BR_PID, IS_PID, KNOWN_PID = 138, 56, 172, 138


def jl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def compare(name, stage, mine, method):
    ad = abs(mine - stage)
    rel = ad / abs(stage) if stage != 0 else (0.0 if ad == 0 else math.inf)
    print(f"  {name}: stage={stage!r} mine={mine!r} abs_diff={ad!r} rel_diff={rel!r}")
    print(f"    method: {method}")


torch.manual_seed(0)
torch.set_num_threads(4)
problems = {int(r["id"]): r for r in jl("results/lm_main/data/problems.jsonl")}
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, local_files_only=True)
N_TOK = len(tok)


def prompt_ids(pid):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": TEMPLATE.format(question=problems[pid]["question"])}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return [int(t) for t in tok(text)["input_ids"]]


def decode(ids):
    return tok.decode([int(i) for i in ids if int(i) < N_TOK], skip_special_tokens=True)


# ---------------------------------------------------------------- own checker
NUM = re.compile(r"[-−]?(?:[0-9]{1,3}(?:,[0-9]{3})+(?![0-9])|[0-9]+)(?:\.([0-9]+))?")


def first_int(text):
    m = NUM.search(text)
    if m is None:
        return None
    if m.group(1) is not None and set(m.group(1)) != {"0"}:
        return None
    return int(m.group(0).split(".")[0].replace(",", "").replace("−", "-"))


def correct(text, answer):
    v = first_int(text)
    return v is not None and v == int(answer)


def status(text, answer, terminal):
    """'won' / 'dead' / 'open' for a prefix (terminal: the completion ends with this text).

    Non-terminal rules (each sound; when unsure: open):
    * no match: open (later text may add a number).
    * the match start s is fixed (a match starting earlier would already exist). Every regex path
      consumes only a sign at s and then digits, ',' and '.', so a character outside [0-9.,] at a
      position > s stops every path: the match can no longer change -> won/dead by the checker.
    * otherwise the leading digit run from s (sign aside) is a prefix of the eventual integer's
      digits: a sign, or leading digits (zeros stripped) not a prefix of the answer -> dead.
    """
    if terminal:
        return "won" if correct(text, answer) else "dead"
    m = NUM.search(text)
    if m is None:
        return "open"
    s = m.start()
    if any(c not in "0123456789.," for c in text[s + 1:]):
        return "won" if correct(text, answer) else "dead"
    if text[s] in "-−":
        return "dead"
    run = re.match(r"[0-9]+", text[s:]).group(0).lstrip("0")
    return "open" if answer.startswith(run) else "dead"


# ---------------------------------------------------------------- own vocabulary classes
pieces = [tok.decode([i], skip_special_tokens=True) for i in range(N_TOK)]
digit_tok = {}
for i, p in enumerate(pieces):
    if any(c in "0123456789" for c in p):
        assert len(p) == 1, (i, p)
        digit_tok[p] = i
assert sorted(digit_tok) == list("0123456789")
special_ids = set(tok.all_special_ids)


def end_set(size):
    """E as a boolean mask over `size` logit rows."""
    e = np.zeros(size, dtype=bool)
    for i in range(min(size, N_TOK)):
        if i in STOP or (pieces[i] != "" and pieces[i][0] not in "0123456789.,"):
            e[i] = True
    return e


print("Independent re-derivation of lm_main_tail numbers (scripts/rederive_tail.py; no capsplit import)")
print(f"model {MODEL}@{REV}; torch {torch.__version__}; tokenizer len {N_TOK}; stop ids {STOP}")


# ---------------------------------------------------------------- 1. D for problem 138
def d_value(model, device, pid):
    p = prompt_ids(pid)
    ans = problems[pid]["answer"]
    dig = [digit_tok[c] for c in ans]
    x = torch.tensor([p + dig], device=device)
    with torch.no_grad():
        logits = model(input_ids=x).logits[0, len(p) - 1:].float().cpu().double()
    lp = torch.log_softmax(logits, dim=-1).numpy()
    log_digits = float(sum(lp[j, t] for j, t in enumerate(dig)))
    e = end_set(lp.shape[1])
    end = float(np.exp(lp[len(dig)][e]).sum())
    return {"log_digits": log_digits, "end_mass": end, "D": math.exp(log_digits) * end,
            "n_prompt": len(p), "n_E": int(e.sum()), "rows": lp.shape[1]}


t1 = time.time()
stage_d = {int(r["problem"]): r for r in jl(f"{TAIL}/d.jsonl")}
cpu = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, local_files_only=True, dtype=torch.float32).eval()
d32 = d_value(cpu, "cpu", D_PID)
del cpu
sd = stage_d[D_PID]
print(f"\n[1] D, problem {D_PID} ({problems[D_PID]['question']} = {problems[D_PID]['answer']}); "
      f"prompt tokens {d32['n_prompt']}, logit rows {d32['rows']}, |E| = {d32['n_E']}")
how = "CPU float32 weights/forward, log-softmax of float32 logits in float64; own chat prompt and own E set"
compare("D", sd["D"], d32["D"], how)
compare("log_digits", sd["log_digits"], d32["log_digits"], how)
compare("end_mass", sd["end_mass"], d32["end_mass"], how)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
dt = torch.float16 if dev == "mps" else torch.float32
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, local_files_only=True, dtype=dt).to(dev).eval()
d16 = d_value(model, dev, D_PID)
how16 = f"{dev} {str(dt).replace('torch.', '')} forward (the stage's numerics), log-softmax in float64"
compare("D (stage numerics)", sd["D"], d16["D"], how16)
compare("log_digits (stage numerics)", sd["log_digits"], d16["log_digits"], how16)
print(f"  time {time.time() - t1:.1f}s")


# ---------------------------------------------------------------- 2. bracket for problem 56
def next_logprobs(prompt, prefixes):
    """Right-padded batch; read each row at its last real position (causal: padding never seen)."""
    seqs = [prompt + list(q) for q in prefixes]
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), STOP[0], dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    with torch.no_grad():
        logits = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits
    last = torch.tensor([len(s) - 1 for s in seqs], device=dev)
    rows = logits[torch.arange(len(seqs), device=dev), last].float().cpu().double()
    return torch.log_softmax(rows, dim=-1).numpy()


def bracket(pid, budget):
    prompt = prompt_ids(pid)
    ans = problems[pid]["answer"]
    won = dead = lumped = 0.0
    heap = [(-0.0, 0, ())]
    count, expansions, n_explicit = 1, 0, 0
    while heap and expansions < budget:
        nodes = [heapq.heappop(heap) for _ in range(min(BATCH, budget - expansions, len(heap)))]
        rows = next_logprobs(prompt, [n[2] for n in nodes])
        for (neg, _, prefix), lp in zip(nodes, rows):
            expansions += 1
            mass = math.exp(-neg)
            probs = np.exp(lp)
            text = decode(prefix)
            terminal = len(prefix) + 1 >= MAX_NEW
            explicit = np.flatnonzero(probs > TAU)
            n_explicit += len(explicit)
            rest = np.ones(len(probs), dtype=bool)
            rest[explicit] = False
            lumped += mass * float(probs[rest].sum())
            for t in explicit.tolist():
                p = float(probs[t])
                if t in STOP:
                    s = status(text, ans, True)
                else:
                    s = status(decode(prefix + (t,)), ans, terminal)
                if s == "won":
                    won += mass * p
                elif s == "dead":
                    dead += mass * p
                elif mass * p < PUSH:
                    lumped += mass * p
                else:
                    heapq.heappush(heap, (neg - float(lp[t]), count, prefix + (t,)))
                    count += 1
    frontier = float(sum(math.exp(-n[0]) for n in heap))
    return {"won": won, "dead": dead, "open_frontier": frontier, "open_lumped": lumped,
            "open": frontier + lumped, "expansions": expansions, "frontier_size": len(heap),
            "mean_explicit": n_explicit / max(expansions, 1)}


t2 = time.time()
sb = {int(r["problem"]): r for r in jl(f"{TAIL}/bracket.jsonl")}[BR_PID]
mb = bracket(BR_PID, BUDGET)
s_lo, s_hi = sb["won"], sb["won"] + sb["open"]
m_lo, m_hi = mb["won"], mb["won"] + mb["open"]
print(f"\n[2] bracket, problem {BR_PID} ({problems[BR_PID]['question']} = {problems[BR_PID]['answer']})")
how2 = (f"own best-first expansion, {dev} {str(dt).replace('torch.', '')} forward, log-softmax in float64, "
        f"budget {BUDGET} (stage {sb['expansions']}), children with p > {TAU} classified exactly, rest and open "
        f"children with prefix prob < {PUSH} lumped open")
compare("lower (won)", s_lo, m_lo, how2)
compare("upper (won + open)", s_hi, m_hi, how2)
print(f"  mine: dead={mb['dead']!r} open_frontier={mb['open_frontier']!r} open_lumped={mb['open_lumped']!r} "
      f"total={mb['won'] + mb['dead'] + mb['open']!r} frontier_size={mb['frontier_size']} "
      f"mean explicit children/node={mb['mean_explicit']:.1f}")
print(f"  stage: dead={sb['dead']!r} open={sb['open']!r} total={sb['total']!r}")
print(f"  stage width={s_hi - s_lo!r} mine width={m_hi - m_lo!r} ratio mine/stage={(m_hi - m_lo) / (s_hi - s_lo)!r}")
print(f"  my lower <= stage lower: {m_lo <= s_lo} (margin {s_lo - m_lo!r}); "
      f"my upper >= stage upper: {m_hi >= s_hi} (margin {m_hi - s_hi!r})")
print(f"  stage bracket contained in mine: {m_lo <= s_lo and m_hi >= s_hi}")
print(f"  time {time.time() - t2:.1f}s")


# ---------------------------------------------------------------- 3. importance sampling from saved rows
def log_mix(row):
    a = np.array([row[f"lp_{c}"] for c in COMPS]) + np.log(np.array(WEIGHTS))
    top = a.max()
    return float(top + np.log(np.exp(a - top).sum()))


def is_est(ind, logw, offset=0.0):
    summ = np.where(np.asarray(ind, dtype=bool), np.exp(np.asarray(logw)), 0.0)
    return offset + float(summ.mean()), float(summ.std(ddof=1) / math.sqrt(len(summ))), int(np.sum(ind))


def recheck(rows, pid):
    ans = problems[pid]["answer"]
    has_text = all("text" in r for r in rows)
    dis_text = sum(correct(r["text"], ans) != bool(r["correct"]) for r in rows) if has_text else None
    dis_ids = sum(correct(decode(r["ids"]), ans) != bool(r["correct"]) for r in rows)
    dis_dec = sum(decode(r["ids"]) != r["text"] for r in rows) if has_text else None
    return has_text, dis_text, dis_ids, dis_dec


t3 = time.time()
with open(f"{TAIL}/report/tail.json") as fh:
    tail = json.load(fh)
how3 = "float64 numpy from stored teacher-forced lp_* sums; log-sum-exp mixture, weights 0.25 each; SE = sd(ddof=1)/sqrt(n)"
rows = jl(f"{TAIL}/is/{IS_PID}.jsonl")
st = tail["is"][str(IS_PID)]
won = {int(r["problem"]): r for r in jl(f"{TAIL}/bracket.jsonl")}[IS_PID]["won"]
logw = [r["lp_base"] - log_mix(r) for r in rows]
ind = [bool(r["correct"]) and r["region"] == "rem" for r in rows]
est, se, k = is_est(ind, logw, won)
has_text, dt_, di_, dd_ = recheck(rows, IS_PID)
ind_mine = [correct(r["text"], problems[IS_PID]["answer"]) and r["region"] == "rem" for r in rows]
est_m, se_m, k_m = is_est(ind_mine, logw, won)
print(f"\n[3a] plain remainder IS estimate, problem {IS_PID} ({problems[IS_PID]['question']} = "
      f"{problems[IS_PID]['answer']}); n={len(rows)}, won from bracket.jsonl={won!r}, "
      f"contributing (correct & rem)={k}")
compare("p_hat", st["estimate"], est, how3)
compare("SE", st["se"], se, how3)
print(f"  'text' field present: {has_text}; checker on stored text disagrees with stored correct: {dt_}; "
      f"checker on re-decoded ids disagrees: {di_}; re-decoded ids != stored text: {dd_}")
print(f"  with my own correct flags: p_hat={est_m!r} SE={se_m!r} contributing={k_m}")

rows = jl(f"{TAIL}/is/{KNOWN_PID}.jsonl")
st = tail["is"][str(KNOWN_PID)]["known"]
ans = problems[KNOWN_PID]["answer"]
dig = [digit_tok[c] for c in ans]
e = end_set(d32["rows"])
in_d_mine = [list(r["ids"][: len(dig)]) == dig and len(r["ids"]) > len(dig) and bool(e[r["ids"][len(dig)]])
             for r in rows]
dis_ind = sum(a != bool(r["in_d"]) for a, r in zip(in_d_mine, rows))
logw = [r["lp_base"] - log_mix(r) for r in rows]
est, se, k = is_est([bool(r["in_d"]) for r in rows], logw)
est_m, se_m, _ = is_est(in_d_mine, logw)
d_stage = stage_d[KNOWN_PID]["D"]
z = (est - d_stage) / se
has_text, dt_, di_, dd_ = recheck(rows, KNOWN_PID)
print(f"\n[3b] known-answer (D-set) IS estimate, problem {KNOWN_PID}; n={len(rows)}, in_d count={k}; "
      f"my in_d recomputed from ids (own E) disagrees with stored in_d: {dis_ind}")
compare("D_hat", st["estimate"], est, how3)
compare("SE", st["se"], se, how3)
compare("z = (D_hat - D_stage)/SE", st["z"], z, how3 + "; D from d.jsonl")
print(f"  z against my CPU fp32 D: {(est - d32['D']) / se!r}; with my in_d flags: D_hat={est_m!r} SE={se_m!r}")
print(f"  'text' field present: {has_text}; checker on stored text disagrees with stored correct: {dt_}; "
      f"checker on re-decoded ids disagrees: {di_}; re-decoded ids != stored text: {dd_}")
print(f"  time {time.time() - t3:.1f}s")
print(f"\ntotal run time {time.time() - T0:.1f}s")
