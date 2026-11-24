"""Independent re-derivation of three lm_main_kl numbers from raw files (no capsplit import).

1. Monte-Carlo KL(pi1 || pi0) for rl/seed0/step200, problem 8, recomputed on CPU float32.
2. Bernoulli data-processing bound d(p1 || p0) for rl/seed0/step200, problem 5.
3. Effective rank (participation ratio) and Frobenius norm of dense Delta W, rl/seed0/step200, layer 0 q_proj.

Adapter is loaded through peft (LoraConfig r=8, alpha=16, q_proj/v_proj + set_peft_model_state_dict) for (1);
Delta W for (3) is built manually from the lora_A/lora_B tensors.
Usage: python scripts/rederive_kl.py   (run from the repo root)
"""
import csv
import json
import math

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REV = "7ae557604adf67be50417f59c2c2f167def9a775"
ADAPTER = "results/lm_main/train/rl/seed0/step200/adapter.safetensors"
KL_SAMPLES = "results/lm_main_kl/samples/rl/seed0/step200.jsonl"
STAGE_KL, STAGE_SE = 2.9955591559410095, 0.20405460056895616
STAGE_BOUND = 3.5908907994792925
R, ALPHA = 8, 16


def jl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def triple(name, mine, stage):
    print(f"{name}: independent={mine!r} stage={stage!r} abs_diff={abs(mine - stage)!r}")


torch.manual_seed(0)
torch.set_num_threads(4)
problems = {int(r["id"]): r for r in jl("results/lm_main/data/problems.jsonl")}

# ---- 1. KL for problem 8 ----
PROB = 8
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, local_files_only=True, torch_dtype=torch.float32)
model.eval()
cfg = LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=0.0, target_modules=["q_proj", "v_proj"])
pm = get_peft_model(model, cfg)
res = set_peft_model_state_dict(pm, load_file(ADAPTER))
assert not res.unexpected_keys, res.unexpected_keys
assert not [k for k in res.missing_keys if "lora" in k], "lora weights missing"
pm.eval()

messages = [
    {"role": "system", "content": "You are a precise calculator."},
    {"role": "user", "content": f"Compute {problems[PROB]['question']}. Reply with only the final number."},
]
prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
prompt_ids = tok(prompt)["input_ids"]


@torch.no_grad()
def logp(comp, adapter_on):
    x = torch.tensor([prompt_ids + list(comp)])
    if adapter_on:
        logits = pm(input_ids=x).logits[0]
    else:
        with pm.disable_adapter():
            logits = pm(input_ids=x).logits[0]
    lp = torch.log_softmax(logits.double(), dim=-1)
    n = len(prompt_ids)
    return float(sum(lp[n - 1 + j, t] for j, t in enumerate(comp)))


rows = sorted([r for r in jl(KL_SAMPLES) if int(r["problem"]) == PROB], key=lambda r: int(r["index"]))
cache = {}
mine, theirs = [], []
for r in rows:
    key = tuple(r["ids"])
    if key not in cache:
        cache[key] = (logp(key, True), logp(key, False))
    lp1, lp0 = cache[key]
    mine.append(lp1 - lp0)
    theirs.append(r["logp1"] - r["logp0"])
mine, theirs = np.array(mine), np.array(theirs)
kl_hat = float(mine.mean())
se = float(mine.std(ddof=1) / math.sqrt(len(mine)))
print(f"[1] problem {PROB}: rows={len(rows)} distinct_completions={len(cache)} prompt_tokens={len(prompt_ids)}")
triple("KL_hat", kl_hat, STAGE_KL)
triple("KL_SE", se, STAGE_SE)
print(f"file mean(logp1-logp0)={float(theirs.mean())!r} file SE={float(theirs.std(ddof=1) / math.sqrt(len(theirs)))!r}")
print(f"max |mine - file| per-row log-ratio={float(np.abs(mine - theirs).max())!r}")
print(f"mean (mine - file) per-row log-ratio={float((mine - theirs).mean())!r}")

# ---- 2. Data-processing bound for problem 5 ----
PROB = 5


def counts(path):
    rs = [r for r in jl(path) if int(r["problem"]) == PROB]
    keys = {int(r["index"]) for r in rs}
    return sum(bool(r["correct"]) for r in rs), len(rs), len(keys)


kA, nA, uA = counts("results/lm_main/samples/base/A_T1.0.jsonl")
kB, nB, uB = counts("results/lm_main/samples/base/B_T1.0.jsonl")
k1, n1, u1 = counts("results/lm_main/samples/rl/seed0/step200/eval_T1.0.jsonl")
k0, n0 = kA + kB, nA + nB
p0 = (k0 + 0.5) / (n0 + 1)
p1 = (k1 + 0.5) / (n1 + 1)
bound = p1 * math.log(p1 / p0) + (1 - p1) * math.log((1 - p1) / (1 - p0))
print(f"\n[2] problem {PROB}: base A k/n={kA}/{nA} (unique idx {uA}), B k/n={kB}/{nB} (unique idx {uB}), "
      f"pooled {k0}/{n0}; trained k/n={k1}/{n1} (unique idx {u1})")
print(f"raw p0_hat={k0 / n0!r} raw p1_hat={k1 / n1!r} smoothed p0={p0!r} p1={p1!r}")
triple("bound_nats", bound, STAGE_BOUND)

# ---- 3. Effective rank, layer 0 q_proj (manual Delta W) ----
sd = load_file(ADAPTER)
pre = "base_model.model.model.layers.0.self_attn.q_proj."
A = sd[pre + "lora_A.weight"].double().numpy()
B = sd[pre + "lora_B.weight"].double().numpy()
dW = (ALPHA / R) * (B @ A)
s = np.linalg.svd(dW, compute_uv=False)
eff = float((s**2).sum() ** 2 / (s**4).sum())
fro = float(np.linalg.norm(dW, "fro"))
stage = None
with open("results/lm_main_kl/report/tables/weights_modules.csv") as f:
    for r in csv.DictReader(f):
        if (r["arm"], r["seed"], r["step"], r["layer"], r["module"]) == ("rl", "0", "200", "0", "q_proj"):
            stage = r
print(f"\n[3] layer 0 q_proj: Delta W shape={dW.shape} top singular values={s[:8].tolist()}")
triple("effective_rank", eff, float(stage["effective_rank"]))
triple("fro", fro, float(stage["fro"]))
