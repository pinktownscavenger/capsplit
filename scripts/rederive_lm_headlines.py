"""Independent re-derivation of four LM headline numbers from raw samples (Results gate).

Written by a separate checker agent that did not read capsplit/src, the report or the measure outputs, and
does not import capsplit. Usage: python scripts/rederive_lm_headlines.py [results/lm_main]
Compare its output with docs/lm_main/report/verdicts.md (L1 rl seed 0, L2 rl pooled, L3 sft pooled) and
the null check in report.md.
"""
import sys
import json, math, os, collections
import numpy as np
R = sys.argv[1] if len(sys.argv) > 1 else 'results/lm_main'
def jl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]
probs = jl(f'{R}/data/problems.jsonl')
split = {int(r['id']): r['split'] for r in probs}
ids = sorted(split); assert ids == list(range(180)), ids[:5]
issues = []
def load(path, n):
    seen = {}; dup = 0
    for r in jl(path):
        key = (int(r['problem']), int(r['index']))
        if key in seen: dup += 1; continue
        seen[key] = bool(r['correct'])
    k = collections.defaultdict(int); c = collections.defaultdict(int)
    for (p, i), v in seen.items():
        c[p] += 1; k[p] += v
    bad = [p for p in ids if c[p] != n or set(i for (q,i) in seen if q==p) != set(range(n))]
    extra = set(c) - set(ids)
    if dup or bad or extra:
        issues.append(f'{os.path.relpath(path,R)}: dups={dup} bad_count_problems={bad[:10]}(n={len(bad)}) extra={sorted(extra)[:5]}')
    return {p: k[p] for p in ids}, {p: c[p] for p in ids}
def rate(path, n):
    k, c = load(path, n); return {p: k[p]/c[p] for p in ids}, k
kA, cA = load(f'{R}/samples/base/A_T1.0.jsonl', 400)
p0, _ = rate(f'{R}/samples/base/B_T1.0.jsonl', 64)
pnull, _ = rate(f'{R}/samples/base/null_T1.0.jsonl', 64)

def logpmf(i, n, p):
    if p <= 0: return 0.0 if i == 0 else -math.inf
    if p >= 1: return 0.0 if i == n else -math.inf
    return math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)+i*math.log(p)+(n-i)*math.log1p(-p)
def sf_ge(k, n, p): return sum(math.exp(logpmf(i,n,p)) for i in range(k, n+1))
def cdf_le(k, n, p): return sum(math.exp(logpmf(i,n,p)) for i in range(0, k+1))
def bisect(f, target, increasing):
    a, b = 0.0, 1.0
    for _ in range(200):
        m = (a+b)/2
        if (f(m) < target) == increasing: a = m
        else: b = m
    return (a+b)/2
def cp(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else bisect(lambda p: sf_ge(k,n,p), alpha/2, True)   # P(X>=k) increasing in p
    hi = 1.0 if k == n else bisect(lambda p: cdf_le(k,n,p), alpha/2, False) # P(X<=k) decreasing in p
    return lo, hi
t = 1/16
cls = {}
for p in ids:
    lo, hi = cp(kA[p], cA[p])
    cls[p] = 'reachable' if lo >= t else ('unreachable' if hi < t else 'ambiguous')
groups = {'all': ids, 'train': [p for p in ids if split[p]=='train'], 'heldout': [p for p in ids if split[p]=='heldout']}
for g, G in groups.items():
    print('counts', g, len(G), dict(collections.Counter(cls[p] for p in G)))
# boundary diagnostics
for p in ids:
    lo, hi = cp(kA[p], cA[p])
    if abs(lo-t) < 2e-3 or abs(hi-t) < 2e-3: print('near-boundary', p, kA[p], round(lo,5), round(hi,5), cls[p])
def p1(arm, s): return rate(f'{R}/samples/{arm}/seed{s}/step200/eval_T1.0.jsonl', 64)[0]
def pooled(arm):
    ps = [p1(arm, s) for s in range(3)]; return {p: np.mean([x[p] for x in ps]) for p in ids}
def contrib(P1, G, c): return sum(P1[p]-p0[p] for p in G if cls[p]==c)/len(G)
sft = pooled('sft')
print('Q1 sft pooled reachable contrib all: %.4f (raw %r)' % (contrib(sft, ids, 'reachable'), contrib(sft, ids, 'reachable')))
rl0 = p1('rl', 0)
tr = json.load(open(f'{R}/train/rl/seed0/training.json'))['prompts']
trainG = groups['train']
print('training.json prompt keys:', len(tr), 'train ids covered:', all(str(p) in tr for p in trainG), 'non-train keys:', sorted(int(k) for k in tr if split.get(int(k))!='train')[:10])
un = [p for p in trainG if cls[p]=='unreachable']
nn = [p for p in un if tr.get(str(p),{}).get('first_success_update') is not None]
nl = [p for p in un if str(p) in tr and tr[str(p)].get('first_success_update') is None]
miss = [p for p in un if str(p) not in tr]
v1 = sum(rl0[p]-p0[p] for p in nn)/len(trainG); v2 = sum(rl0[p]-p0[p] for p in nl)/len(trainG)
print('Q2 rl seed0 train-unreachable first_success notnull: n=%d contrib=%.4f (raw %r)' % (len(nn), v1, v1))
print('Q2 rl seed0 train-unreachable first_success null:    n=%d contrib=%.4f (raw %r)  missing=%s' % (len(nl), v2, v2, miss))
rl = pooled('rl')
v3 = contrib(rl, groups['heldout'], 'unreachable')
print('Q3 rl pooled unreachable contrib heldout: %.4f (raw %r)' % (v3, v3))
v4 = np.mean([pnull[p]-p0[p] for p in ids])
print('Q4 null - B mean: %.4f (raw %r)' % (v4, v4))
for arm in ['rl','sft','sft_then_rl','none']:
    for s in range(3): p1(arm, s)
print('ISSUES:', issues if issues else 'none')
