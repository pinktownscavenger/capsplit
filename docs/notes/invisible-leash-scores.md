# SRR and NDR in *The Invisible Leash*

> **Status: unverified draft.** These definitions were extracted from the arXiv HTML by an automated tool. N2 requires a person to read the paper, so check every entry against the source, then delete this notice. Nothing here is implemented in the pilot.

**Source:** Wu, Xuan, Lu, Liu, Dong, Harchaoui and Choi, *The Invisible Leash? Why RLVR May or May Not Escape Its Origin*, [arXiv:2507.14843v4](https://arxiv.org/html/2507.14843v4).

**Cite v4 or later.** The scores aren't in [v1](https://arxiv.org/html/2507.14843v1), and section numbers differ between versions. The page doesn't number its equations, so entries below cite sections and definitions.

## Definitions

For a prompt $x$, base model $q$ and RL model $\pi_\theta$:

| Item | Definition | Where |
|---|---|---|
| Correct completions | $\mathcal{C} = \{y \mid R(x, y) = 1\}$ | Section 3 |
| Empirical support | $\text{supp}_\epsilon(q) = \{y \in \mathcal{C} \mid q(y \mid x) > \epsilon\}$ | Section 3.1 |
| Preservation, $P$ | Correct completions with $q > \epsilon$ and $\pi_\theta > \epsilon$ | Section 3.2, Figure 1 caption |
| Shrinkage, $S$ | Correct completions with $q > \epsilon$ and $\pi_\theta \le \epsilon$ | Section 3.2, Figure 1 caption |
| Expansion, $E$ | Correct completions with $q \le \epsilon$ and $\pi_\theta > \epsilon$ | Section 3.2, Figure 1 caption |
| **SRR** | $P / (P + S)$ | Section 3.2, Definition 3.3 |
| **NDR** | $E / (P + E)$ | Section 3.2, Definition 3.3 |

$P$, $S$ and $E$ are counts of completions, not probability mass. SRR is the share of the base model's accessible correct solutions that RL keeps. NDR is the share of RL's accessible correct solutions that are new.

**Threshold.** Appendix C.4 bounds the probability of a correct completion never seen in $k$ samples: $\pi(y^* \mid x) \le -\log(\zeta) / k$ with confidence $1 - \zeta$. Its worked value is $k = 8192$, $\zeta = 0.05$, giving about $3.66 \times 10^{-4}$. At $\zeta = 1/e$ this is $1/k$, which is our $c/N$ line at $c = 1$.

**Equivalence.** Section 3.1 counts completions at the answer level: reasoning traces with the same verified outcome are one class.

## Inputs each score needs

Both scores need the same things:

- a correct set, with an equivalence map from traces to answers;
- $\epsilon$, or a budget $k$ and confidence $\zeta$;
- the base and RL probabilities of each correct class, or estimates of them.

## Can they be computed from a `TrajectoryTable`?

| Source | Computable? | Notes |
|---|---|---|
| `TrajectoryTable` with exact probabilities | **Yes** | `solution_probs` gives each solution's mass under both policies, and `table.solutions` plays the role of the answer classes. Apply $\epsilon$ and count. |
| Sample counts | **Not yet determined** | The sections read don't say how $q$ and $\pi_\theta$ are estimated from samples, so a counting rule such as "seen at least once in $k$ samples" would be our assumption, not the paper's. |

## Open points to settle before implementing

1. **Measurement:** how the paper gets $P$, $S$, $E$ from samples, and which $\epsilon$ produced its reported counts. The tool found neither in the text it read.
2. **Aggregation:** the definitions are per prompt, but the sections read don't say whether reported counts are per prompt or summed over a dataset. RQ4 runs are multi-prompt, so this changes the numbers.
3. **Zero denominators:** I saw no stated value for SRR when $P + S = 0$, or NDR when $P + E = 0$. A base policy with $J_0$ near 0 could hit this.
4. **Boundary convention:** the paper counts a solution as supported when $q > \epsilon$, while our split counts it as reachable when $q \ge c/N$. They differ only for a mass exactly on the line, but a comparison should use one convention and say which.
