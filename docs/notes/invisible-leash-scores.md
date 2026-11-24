# SRR and NDR in *The Invisible Leash*

> **Status: checked against v4 on 2026-11-24 by Claude (Bilal's request), not yet signed off by a person.** Every entry below was compared with the arXiv v4 HTML page text, including Appendix C.4. The definitions match. Section references are corrected and three of the four open points are answered from the paper (see the discrepancy list). The project rule still wants a person to confirm; once someone has, replace this notice with their name and the date. The choices in "Our convention" are proposed (DECISIONS RQ4-5) until the team ratifies them.

**Source:** Wu, Xuan, Lu, Liu, Dong, Harchaoui and Choi, *The Invisible Leash? Why RLVR May or May Not Escape Its Origin*, [arXiv:2507.14843v4](https://arxiv.org/html/2507.14843v4).

**Cite v4 or later.** The scores aren't in [v1](https://arxiv.org/html/2507.14843v1), and section numbers differ between versions. The page doesn't number its equations, so entries below cite sections, definitions and tables.

## Definitions

For a prompt $x$, base model $q$ and RL model $\pi_\theta$:

| Item | Definition | Where (v4) |
|---|---|---|
| Correct completions | $\mathcal{C} = \{y \in \mathcal{Y} \mid R(x, y) = 1\}$ | Section 3.1, Definition 3.1 |
| Empirical support | $\text{supp}_\epsilon(q) = \{y \in \mathcal{C} \mid q(y \mid x) > \epsilon\}$, strict | Section 3.1, "Empirical Support Relaxation" (unnumbered display) |
| Expansion, Shrinkage (as events) | $\text{supp}_\epsilon(\pi_\theta) \setminus \text{supp}_\epsilon(q) \ne \emptyset$; $\text{supp}_\epsilon(q) \setminus \text{supp}_\epsilon(\pi_\theta) \ne \emptyset$ | Section 3.2, Definition 3.2 |
| Preservation, $P$ | Correct completions with $q > \epsilon$ and $\pi_\theta > \epsilon$ | Figure 1 caption (Section 1) |
| Shrinkage, $S$ | Correct completions with $q > \epsilon$ and $\pi_\theta \le \epsilon$ | Definition 3.2; Figure 1 caption |
| Expansion, $E$ | Correct completions with $q \le \epsilon$ and $\pi_\theta > \epsilon$ | Definition 3.2; Figure 1 caption |
| Out of support, $O$ | Correct completions missed by both (not used by SRR or NDR) | Figure 1 caption; Definition 3.3 |
| **SRR** (Support Retention Rate) | $P / (P + S)$ | Section 3.2, Definition 3.3 |
| **NDR** (Net Discovery Rate) | $E / (P + E)$ | Section 3.2, Definition 3.3 |

$P$, $S$, $E$ and $O$ are counts ("the number of correct completions", Definition 3.3), not probability mass. SRR is the share of the base model's accessible correct solutions that RL keeps. NDR is the share of RL's accessible correct solutions that are new. Definition 3.3 also defines SDS and NSCR, which we don't use.

**Threshold.** Appendix C.4 bounds the probability of a correct completion never seen in $k$ samples: $\pi(y^* \mid x) \le -\log(\zeta) / k$ with confidence $1 - \zeta$, using $\log(1-p) \le -p$. Its worked value is $k = 8192$, $\zeta = 0.05$, giving about $3.66 \times 10^{-4}$. At $\zeta = 1/e$ this is $1/k$, which is our $c/N$ line at $c = 1$ (our remark, not the paper's).

**Equivalence.** Section 3.1 counts completions at the answer level: reasoning traces with the same verified outcome are one class.

## Check against v4: discrepancies

1. **Location of $\mathcal{C}$:** the draft said "Section 3". It is Definition 3.1 in Section 3.1.
2. **Location of P, S, E:** the draft put the Figure 1 caption in Section 3.2. The caption is in Section 1. Section 3.2's Definition 3.2 formalises only Expansion and Shrinkage, as set differences being non-empty; Preservation and Out of support are defined only informally (Figure 1 and Table 1 captions).
3. **Open point 2 (aggregation) is answered.** Appendix A says the aggregate rows are computed from raw counts, and Table 6's Overall row is the sum of its Math and Non-Math counts (P 2400 = 1355 + 1045, S 163 = 56 + 107, E 36 = 5 + 31), with SRR 0.936 = 2400/2563 and NDR 0.0148 = 36/2436. So scores over many prompts sum the counts first, then divide.
4. **Open point 1 (measurement) is partly answered.** The Table 1 caption says Preservation means "both base and RLVR find the solution", and C.4 derives $\epsilon$ from not observing a completion in $k$ samples. So the paper counts a solution as supported when it is seen in $k$ samples (budgets in Section 4.1: $k \in \{4096, 8192\}$ for math). The paper never states this as a rule in so many words, and it gives no $\epsilon$ value beyond C.4's example. In Tables 3 and 4 the four counts sum to each benchmark's problem count (e.g. AIME24: 15 + 4 + 4 + 7 = 30), so in practice there each problem contributes one answer class.
5. **Open point 3 (zero denominators):** not stated. Table 4 prints "-" for its step-0 rows; no row with a zero denominator appears.
6. **Open point 4 (boundary):** confirmed. The paper uses strict $q > \epsilon$ (Section 3.1). Our reachable set uses $q \ge c/N$, so the two differ only for a mass exactly on the line.
7. **Arithmetic note, not ours to fix:** Table 1 prints NDR 0.02 for ProRL-1.5B-v1 Overall, while its own counts give 0.0148 (printed as such in Table 6).

## Our convention (proposed, DECISIONS RQ4-5)

- **Measurement:** exact solution probabilities from the `TrajectoryTable` (`solution_probs`), no sampling. `table.solutions` plays the role of answer classes.
- **Threshold:** $\epsilon = c/N$ at the RQ4 primary $N = 64$, $c = 1$, so $\epsilon = 1/64$. This is C.4's bound at $\zeta = 1/e$ with $k = N$.
- **Boundary:** strict $> \epsilon$, as the paper.
- **Aggregation:** for a unit (run, final checkpoint, held-out prompts), sum P, S, E over its prompts, then divide; per cell and per phase, sum over units, then divide. This follows Appendix A.
- **Zero denominators:** undefined (`None`, printed "undefined"), counted separately.

Implemented in `capsplit.diagnostics.srr_ndr` and `pool_srr_ndr`, and `experiments/rq4_passk.py srr-ndr-report`.
