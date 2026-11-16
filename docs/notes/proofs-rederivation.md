# Re-deriving the proofs (exit criterion 6, Gate 1)

> **Status: complete.** Munashe re-derived [docs/proofs.md](../proofs.md) by hand on 2026-11-16. A coding agent checked every step and added [tests/test_proofs.py](../../tests/test_proofs.py).

## How to use this

For each result, close `proofs.md`, rebuild the argument from the key steps below on paper, then compare. Tick the box only if you could rebuild it yourself.

## Steps to rebuild

### Proposition 1: exact split

- [ ] $\mathcal{R}_N$ and $\mathcal{U}_N$ partition $S^*$. Split $\sum_{s \in S^*}(q_1 - q_0)$ over the two parts.

### Proposition 2a: per-step family

- [ ] Write $P_\beta(\tau)/P_0(\tau) = \prod_i \pi_0(a_i \mid s_i)^{\beta-1}/Z(s_i)$, and pull out $P_0(\tau)^{\beta-1}$. This uses deterministic dynamics.
- [ ] Jensen's inequality with $x^\beta$ convex gives $Z(s) \ge |A|^{1-\beta}$, so $1/Z(s) \le |A|^{\beta-1}$.
- [ ] Multiply over the $t \le H$ steps, then use $|A|^t \le |A|^H$ because $\beta - 1 \ge 0$.

### Proposition 2b: sequence family

- [ ] Jensen over the $T$ support trajectories gives $Z \ge T^{1-\beta}$.
- [ ] $T \le |A|^H$: no trajectory is a proper prefix of another, because the goal is absorbing, so their extensions to length $H$ are disjoint.

### Corollary

- [ ] For $\beta > 1$, each trajectory with $0 < P_0 < U$ strictly falls, and a trajectory with $P_0 = 0$ stays at 0. Sum over $\phi^{-1}(s)$. You need $q_0(s) > 0$ for the fall to be strict.
- [ ] Recompute the example: 0.4 falls to 0.308 under per-step and to 0.185 under sequence tempering.

### Lemma

- [ ] $f(J) = 1 - (1 - J)^N$ is continuous and strictly increasing, so $\sup f = f(\sup)$ and the sign carries over.
- [ ] If $\mathcal{R}_N \ne \emptyset$, then $J^* \ge J_0 \ge 1/N$, so $E_N \le (1 - 1/N)^N \le e^{-1}$.

### Proposition 3

- [ ] $p_k = P(1 - C(k))$. Newton's identities take $p_1, \ldots, p_P$ to $e_1, \ldots, e_P$, which fix the polynomial and so its roots.
- [ ] Recompute the example table: $(+0.4, -0.003)$ and $(-0.5, +0.897)$, each summing to 0.397.

### Estimator bound

- [ ] Upper tail: the exponent $Mt^2/(2m + t) \ge M\eta^2 m'/3$, using $2m + t \le 3m'$.
- [ ] Lower tail: the probability is 0 if $t > m$. Otherwise the exponent $Mt^2/(2m) \ge M\eta^2 m'/2$.
- [ ] A union bound over 8 tail events gives $\delta$.
- [ ] Recompute $M = 1{,}522{,}553$ and $38{,}064$.

## What the agent check found

Every step holds. **No errors.** These are the suggested additions to `proofs.md`, none of which changes a result:

1. **Estimator, lower tail at $t = m$.** The usual multiplicative Chernoff bound is stated for $0 < \gamma < 1$, and `proofs.md` uses $\gamma \le 1$. The endpoint holds, because the event is $\hat{m} = 0$, with probability $(1 - m)^M \le e^{-mM} \le e^{-mM/2}$. Add this line.
2. **Lemma: the code computes a lower bound on $J^*$.** `tempering.frontier` searches a β grid with golden-section refinement, which can miss the true sup. So the computed $J$ gap and $E_N$ can only overstate, and a `reweighting` or `capability` label could be a false positive. Add this to the Lemma's limits, next to D3.
3. **Lemma, general $c$.** The same argument gives $E_N \le (1 - c/N)^N \le e^{-c}$ for $c \le N$. The experiments vary $c$, so this is worth stating.
4. **Proposition 3 can be stated more precisely.** The proof needs only $k = 1, \ldots, P$, so the statement can say "at $k = 1, \ldots, P$" instead of "at every integer $k \ge 1$". That makes the RQ4 limit exact: the grid must contain $1, \ldots, P$.

**Numerical checks** are now in `tests/test_proofs.py` and run in CI:

| Check | Result |
|---|---|
| Proposition 2 on compiled grid trajectories ($3{\times}3$ with $H = 4, 6$; $4{\times}4$ with $H = 6$), 40 policies, both families, $\beta$ from 1 to 8 | Holds. Largest excess in log ratio is $3.6 \times 10^{-15}$ |
| The same check with $U$ 1.5 times too large | Fails for per-step tempering, so the test is sensitive and the per-step bound is close to tight |
| Corollary on the same grids | No counterexample, and the condition fires, so the test isn't vacuous |
| Corollary example, Newton reconstruction, $(1 - 1/N)^N \le 1/e$ | Pass |
| Exact binomial tails at $M = 1219$ | Largest $8.2 \times 10^{-5}$, against a limit of $0.00625$ |
| The same at $M/2$ | Largest $4.1 \times 10^{-3}$, so the Chernoff size is conservative here, as the pilot report says |

The first row closes the pilot report's "not done" item: Proposition 2 is now checked on the grid trajectories the experiments use.

## Sign-off

- [x] Munashe re-derived Propositions 1 to 3, the corollary, the Lemma and the estimator bound by hand on: 2026-11-16
