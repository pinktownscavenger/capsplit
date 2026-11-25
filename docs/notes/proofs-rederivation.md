# Re-deriving the proofs (exit criterion 6, Gate 1)

> **Status: complete.** Munashe re-derived sections 1 to 6 of [docs/proofs.md](../proofs.md) by hand on 2026-11-16. A coding agent checked every step and added [tests/test_proofs.py](../../tests/test_proofs.py). Section 7 (time to cross), added on 2026-11-19 for RQ2, was checked by a coding agent on 2026-11-24 ([below](#section-7-time-to-cross)). Bilal accepted that check as the sign-off on 2026-11-24, without a separate hand re-derivation, and `proofs-frozen` was tagged.

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

## Section 7: time to cross

RQ2's P1 verdict rests on this section, so it gets its own pass. The four suggested additions above are now applied in `proofs.md` (Lemma limits and general $c$, Proposition 3 statement, estimator endpoint).

### Steps to rebuild

- [ ] $J = \prod_{i=1}^H \pi(a_i \mid s_i)$ because the route fills the horizon, its cells are distinct and dynamics are deterministic. Differentiate: $\partial J / \partial \theta(s_i, a) = J(\mathbb{1}[a = a_i] - \pi(a \mid s_i))$, and 0 off the route.
- [ ] Induction: if every route cell has gap $g$ and equal off-route logits, the route action gains $\eta J(1-p)$ and each other action loses $\eta J(1-p)/(A-1)$, the same at every cell. The gap grows by $\eta \frac{A}{A-1} J(1-p)$.
- [ ] $dp/dg = p(1-p)$, so $dp/dt = \eta \frac{A}{A-1} p^{H+1}(1-p)^2$. Separate variables to get the integral.
- [ ] Asymptotic: the leading term is $(q_0^{-1} - \ell^{-1})/H$; bound the remainder by $C\int p^{-H}\,dp$, which needs $H \ge 2$. Check the first-order coefficient $2H/(H-1)$.
- [ ] Part 3: with $G(g) = \int dx/\varphi(x)$ and $\varphi$ increasing below the line, one Euler step advances $G$ by at most 1 (so $T \le N$) and by at least $\varphi(g_s)/\varphi(g_{s+1}) \ge 1 - \ln(\varphi(g_{s+1})/\varphi(g_s))$. Telescope the logs.
- [ ] Check the peak condition: $p^H(1-p)$ increases for $p < H/(H+1)$, so the line must satisfy $\ell \le (H/(H+1))^H$.
- [ ] Part 4: differentiate $T$ at its lower limit and use $p_0^{H+1} = q_0 p_0$. Expand to get slope $= -1 + \frac{2}{H-1} p_0 + o(p_0)$.
- [ ] Recompute: $q_0^{1/8} = 0.178$ at $q_0 = 10^{-6}$; leading term $3/(4 \cdot 8 \cdot 10^{-6}) = 93{,}750$; $T = 148{,}403$; with $(1-p_0)^{-2}$ the leading term is 138,690, 6.5% low.

### What the agent check found (2026-11-24)

**No error in the recursion, the integral, or the RQ2 numbers.** The recursion (part 1) is exact, and P1's slopes ($-0.844$ fitted, $-0.928$ smallest pair) come from it, so P1 stands on part 1 alone. Changes made to `proofs.md`:

1. **The asymptotic was stated for all $H$, but it fails at $H = 1$.** There the next term integrates to a log, so the correction is $O(q_0 \ln(1/q_0))$, not $O(q_0)$. The statement now says $H \ge 2$. RQ2 uses $H = 8$, so no result changes.
2. **The remainder bound was a series expansion only.** The proof now bounds it explicitly ($r(p) \le Cp$ on $[0, \ell^{1/H}]$), and states the first-order coefficient $2H/(H-1)$.
3. **New part 3: the recursion and the integral were linked only numerically** (`test_integral_within_5pct_of_recursion`). It's now a theorem: $T \le N \le T + 1 + \ln(\ell(1-\ell^{1/H}) / (q_0(1-q_0^{1/H})))$ when $\ell \le (H/(H+1))^H$. At $q_0 = 10^{-6}$ that is $148{,}403 \le 148{,}409 \le 148{,}414$.
4. **New part 4: "the local slope tends to $-1$" was argued from the asymptotic of $T$**, which on its own says nothing about the derivative. Part 4 gives the slope in closed form and shows it approaches $-1$ from above as $-1 + \frac{2}{H-1} q_0^{1/H}$.

### Numerical checks (in `tests/test_proofs.py`, run in CI)

| Check | Result |
|---|---|
| Part 1 against full-table gradient ascent with finite-difference gradients (independent of the proof's closed form), $(A, H) = (2,3), (4,4), (5,6)$, 30 steps | Same $J$ to rel $10^{-6}$; all cells keep one gap |
| The same with one off-route logit changed | Cells drift apart, so the check is sensitive |
| Part 3 over 405 settings ($A$, $H$, $\eta$, $\ell$, $q_0$) | $T \le N \le$ bound in all of them |
| Part 3 with $\ell$ above the peak | Refused (`ValueError`) |
| Part 2 coefficient at $q_0 = 10^{-16}$ | Within 2% of $2H/(H-1)$ for $H = 2, 4, 8$, converging |
| Part 4 against a finite-difference slope | Equal to rel $10^{-6}$; $-0.9346$ at $q_0 = 10^{-6}$, $-0.9906$ at $10^{-12}$ |
| Numbers quoted in "What this means for RQ2" | 0.178, 93,750, 148,403, "within about 7%" (6.5%) all reproduce |

### Sign-off

- [ ] Hand re-derivation of section 7 by a teammate: waived. Bilal accepted the agent check as the sign-off on 2026-11-24.
- [x] `proofs-frozen` tagged on 2026-11-24.
