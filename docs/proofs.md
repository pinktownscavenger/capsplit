# capsplit proofs

> **Status: re-derived and closed on Nov 16, 2026.** Written by a coding agent, checked numerically and step by step by Bilal Malik on Nov 7, then independently re-derived by Munashe on Nov 16. Pilot exit criterion 6 is complete and Gate 1 is closed.

**Setup.** A verifier gives each trajectory $\tau$ a reward in $\{0, 1\}$. A solution map $\phi$ groups trajectories into solutions, $S^*$ is the set of correct solutions, and $q_\pi(s) = \sum_{\tau \in \phi^{-1}(s)} P_\pi(\tau)$. Success is $J(\pi) = \sum_{s \in S^*} q_\pi(s)$. Dynamics are deterministic, so a trajectory is a sequence of actions. It ends at an absorbing goal or after $H$ steps, so its length is $t(\tau) \le H$.

## 1. Proposition 1: exact split

**Statement.** For any $\pi_0, \pi_1, N, c, \phi$:

$$J(\pi_1) - J(\pi_0) = \Delta_{\text{rel}}(N) + \Delta_{\text{cap}}(N).$$

**Proof.** By definition, $\mathcal{R}_N = \{s \in S^* : q_{\pi_0}(s) \ge c/N\}$ and $\mathcal{U}_N = S^* \setminus \mathcal{R}_N$, so the two sets partition $S^*$. Therefore

$$J(\pi_1) - J(\pi_0) = \sum_{s \in S^*} \big(q_{\pi_1}(s) - q_{\pi_0}(s)\big) = \sum_{s \in \mathcal{R}_N} \big(q_{\pi_1}(s) - q_{\pi_0}(s)\big) + \sum_{s \in \mathcal{U}_N} \big(q_{\pi_1}(s) - q_{\pi_0}(s)\big). \qquad \blacksquare$$

**Limits.** This is bookkeeping and says nothing about what either term means. Both terms can be negative. On a benchmark the identity holds per prompt, so it also holds for the sum or mean over prompts.

## 2. Proposition 2: sharpening can't lift what the base disfavors

**Statement.** For $\beta \ge 1$ and any $\tau$ with $P_{\pi_0}(\tau) > 0$, in either family:

$$\frac{P_\beta(\tau)}{P_{\pi_0}(\tau)} \le \left(\frac{P_{\pi_0}(\tau)}{U}\right)^{\beta - 1},$$

with $U = |A|^{-H}$ for the per-step family and $U = 1/T$ for the sequence family, where $T$ is the number of trajectories in the support of $\pi_0$.

**Inequality used by both parts.** For $\beta \ge 1$ and $x \mapsto x^\beta$ convex on $[0, \infty)$, Jensen's inequality says that any $p_1, \ldots, p_n \ge 0$ summing to 1 satisfy

$$\frac{1}{n}\sum_i p_i^\beta \ge \left(\frac{1}{n}\sum_i p_i\right)^\beta = n^{-\beta}, \quad\text{so}\quad \sum_i p_i^\beta \ge n^{1 - \beta}. \tag{J}$$

### 2a. Per-step family

Let $\tau$ visit states $s_1, \ldots, s_t$ and take actions $a_1, \ldots, a_t$, with $t = t(\tau)$. A state may repeat. Write $Z(s) = \sum_a \pi_0(a \mid s)^\beta$, so that $\pi_0^\beta(a \mid s) = \pi_0(a \mid s)^\beta / Z(s)$. For softmax policies, this is the same as multiplying the logits by $\beta$.

Dynamics are deterministic, so $P_{\pi_0}(\tau) = \prod_{i=1}^t \pi_0(a_i \mid s_i)$ and

$$\frac{P_\beta(\tau)}{P_{\pi_0}(\tau)} = \prod_{i=1}^t \frac{\pi_0(a_i \mid s_i)^{\beta - 1}}{Z(s_i)} = P_{\pi_0}(\tau)^{\beta - 1} \prod_{i=1}^t \frac{1}{Z(s_i)}.$$

By (J) with $n = |A|$, each $1/Z(s_i) \le |A|^{\beta - 1}$, which gives

$$\frac{P_\beta(\tau)}{P_{\pi_0}(\tau)} \le \big(P_{\pi_0}(\tau)\, |A|^{t}\big)^{\beta - 1}.$$

**Early termination.** A goal-reaching trajectory may have $t < H$. Since $|A|^t \le |A|^H$ and $\beta - 1 \ge 0$, the right side is at most $(P_{\pi_0}(\tau) |A|^H)^{\beta - 1} = (P_{\pi_0}(\tau) / U)^{\beta - 1}$. The absorbing goal adds no factor, because no action is taken there. $\blacksquare$

The intermediate line is tighter for short trajectories, with bar $|A|^{-t}$. We state $|A|^{-H}$ because it's a single bar for all trajectories.

### 2b. Sequence family

Here $P_\beta(\tau) = P_{\pi_0}(\tau)^\beta / Z$ with $Z = \sum_{\tau'} P_{\pi_0}(\tau')^\beta$, summed over the $T$ support trajectories. Those probabilities sum to 1, so (J) with $n = T$ gives $Z \ge T^{1 - \beta}$. Therefore

$$\frac{P_\beta(\tau)}{P_{\pi_0}(\tau)} = \frac{P_{\pi_0}(\tau)^{\beta - 1}}{Z} \le \big(P_{\pi_0}(\tau)\, T\big)^{\beta - 1} = \left(\frac{P_{\pi_0}(\tau)}{U}\right)^{\beta - 1}. \qquad \blacksquare$$

**At least as tight.** Under deterministic dynamics, $T \le |A|^H$. To see this, extend each trajectory to length $H$ in every possible way. Trajectories end at the goal or at step $H$, so no trajectory is a prefix of another, and the sets of extensions are disjoint and non-empty. There are $|A|^H$ action sequences of length $H$, so $T \le |A|^H$. Hence $1/T \ge |A|^{-H}$, and for the same $P_{\pi_0}(\tau)$ the sequence bound $(P_{\pi_0}(\tau) T)^{\beta - 1}$ is no larger than the per-step bound $(P_{\pi_0}(\tau) |A|^H)^{\beta - 1}$.

This compares the two upper bounds. It doesn't say that one family's $P_\beta(\tau)$ is below the other's.

**Limits.**

- **$\beta < 1$:** the proposition says nothing useful here. For $0 < \beta < 1$, (J) reverses, which turns the bound into a lower bound, so flattening raises every trajectory below $U$. $E_N$ covers that side.
- **Deterministic dynamics:** the per-step proof cancels only policy factors. With stochastic transitions, those factors aren't tempered, so the bound holds only for the policy part of $P$. $T \le |A|^H$ can also fail, because one action sequence can produce several trajectories.
- **Trajectories, not solutions:** see the corollary.

## 3. Corollary: solution level

**Statement.** Fix a family and its $U$. Let $s$ have $q_{\pi_0}(s) > 0$, and let every $\tau \in \phi^{-1}(s)$ have $P_{\pi_0}(\tau) < U$. Then every $\beta > 1$ strictly lowers $q(s)$.

The blueprint leaves out $q_{\pi_0}(s) > 0$. It's needed for strictness, since otherwise both sides are 0.

**Proof.** Consider $\tau \in \phi^{-1}(s)$. If $P_{\pi_0}(\tau) = 0$, then $P_\beta(\tau) = 0$ as well. If $P_{\pi_0}(\tau) > 0$, then $P_{\pi_0}(\tau)/U < 1$ and $\beta - 1 > 0$, so Proposition 2 gives $P_\beta(\tau) < P_{\pi_0}(\tau)$. At least one $\tau$ has positive mass, so summing over $\phi^{-1}(s)$ gives $q_\beta(s) < q_{\pi_0}(s)$. $\blacksquare$

**The solution mass can exceed $U$.** The condition must be checked on trajectories, as this example shows. Take $|A| = 2$, $H = 2$ and three states, with no early termination:

| State | $\pi_0(a)$ | $\pi_0(b)$ |
|---|---|---|
| Start | 0.4 | 0.6 |
| After $a$ | 0.5 | 0.5 |
| After $b$ | 0.99 | 0.01 |

The trajectories have $P(aa) = P(ab) = 0.2$, $P(ba) = 0.594$ and $P(bb) = 0.006$. Here $T = 4$, so $U = 1/4$ in both families. Let $\phi^{-1}(s) = \{aa, ab\}$. Then $q_{\pi_0}(s) = 0.4 > U$, while both of its trajectories have $0.2 < U$. At $\beta = 2$, $q(s)$ falls in both families:

- **Per-step:** $0.4^2 / (0.4^2 + 0.6^2) \approx 0.308$.
- **Sequence:** $(0.04 + 0.04) / (0.04 + 0.04 + 0.352836 + 0.000036) \approx 0.185$.

**Limits.** The condition is sufficient, not necessary. If some trajectory is at or above $U$, the corollary says nothing about $q(s)$.

## 4. Lemma: single-task sign

**Statement.** Fix one task and $N \ge 1$, and let $J^* = \sup_{\beta > 0,\ \text{both families}} J(\pi_0^\beta)$. Then:

1. $\operatorname{sign}(E_N) = \operatorname{sign}(J(\pi_1) - J^*)$.
2. If $\mathcal{R}_N \neq \emptyset$ at $c = 1$, then $E_N \le (1 - 1/N)^N \le 1/e$.

**Proof of 1.** Let $f(J) = 1 - (1 - J)^N$. It's continuous and strictly increasing on $[0, 1]$ for $N \ge 1$. Since $f$ is continuous and increasing, $\sup_\beta f(J(\pi_0^\beta)) = f(J^*)$, even if the sup isn't attained. So $E_N = f(J(\pi_1)) - f(J^*)$, and strict monotonicity gives the sign.

**Proof of 2.** Some $s \in S^*$ has $q_{\pi_0}(s) \ge 1/N$, so $J(\pi_0) \ge 1/N$. The families contain $\pi_0$ at $\beta = 1$, so $J^* \ge 1/N$. Since $f(J(\pi_1)) \le 1$,

$$E_N \le 1 - f(J^*) = (1 - J^*)^N \le (1 - 1/N)^N.$$

Finally, $\ln(1 - x) \le -x$ for $x < 1$ gives $(1 - 1/N)^N \le e^{-1}$ for $N \ge 2$, and at $N = 1$ the value is 0. $\blacksquare$

**Limits.** This holds for one task only: averaged curves can cross, which is why Proposition 3 needs a benchmark. In floating point, a real gap can round to $E_N = 0$ when pass@$N$ saturates, so labels use the $J$ gap (D2).

## 5. Proposition 3: pass@k can't determine the split

**Statement.** Take a benchmark of $P$ prompts, with averaged curve $C(k) = \frac{1}{P} \sum_i [1 - (1 - J_i)^k]$.

1. Two policies share $C(k)$ at every integer $k \ge 1$ exactly when they share the multiset $\{J_i\}$.
2. The multiset doesn't determine the split, so neither does the curve.

**Proof of 1.** If the multisets match, the sums match term by term. For the converse, let $x_i = 1 - J_i$ and $p_k = \sum_i x_i^k$. Then $p_k = P(1 - C(k))$, so equal curves give equal power sums.

Let $e_k$ be the elementary symmetric polynomials of the $x_i$, with $e_0 = 1$. Newton's identities give, for $k = 1, \ldots, P$:

$$k\, e_k = \sum_{i=1}^k (-1)^{i-1} e_{k-i}\, p_i.$$

We can divide by $k$ over the reals, so by induction $p_1, \ldots, p_P$ determine $e_1, \ldots, e_P$. These are the coefficients of

$$\prod_{i=1}^P (t - x_i) = \sum_{k=0}^P (-1)^k e_k\, t^{P-k}.$$

Equal polynomials have equal roots with multiplicity, so the multisets $\{x_i\}$ and $\{J_i\}$ match. $\blacksquare$

**Proof of 2, by example.** Let $N = 100$ and $c = 1$, so the line is $0.01$. Both policies share this base:

| Prompt | Base on reachable solution | Base on rare solution |
|---|---|---|
| 1 | $a$: 0.2 | $b$: 0.001 |
| 2 | $a'$: 0.3 | $b'$: 0.002 |

So $a, a' \in \mathcal{R}_N$ and $b, b' \in \mathcal{U}_N$, and the base success is $0.201$ and $0.302$.

- $\pi_1$ puts 0.4 on $a$ and 0.5 on $a'$, with 0 on the rare solutions. Success is $\{0.4, 0.5\}$.
- $\pi_1'$ puts 0.5 on $b$ and 0.4 on $b'$, with 0 on the reachable solutions. Success is $\{0.5, 0.4\}$.

The multisets match, so by part 1 the curves match at every $k$. Summed over prompts:

| Policy | $\Delta_{\text{rel}}$ | $\Delta_{\text{cap}}$ | Total |
|---|---|---|---|
| $\pi_1$ | $0.2 + 0.2 = +0.4$ | $-0.001 - 0.002 = -0.003$ | $0.397$ |
| $\pi_1'$ | $-0.2 - 0.3 = -0.5$ | $0.499 + 0.398 = +0.897$ | $0.397$ |

The curves are the same and the splits differ, while both totals equal $0.9 - 0.503 = 0.397$, as Proposition 1 requires. $\blacksquare$

**Limits.**

- The example uses a per-prompt sum, as the plan does. The library's benchmark split is the mean over prompts, which halves these values.
- The converse uses $k = 1, \ldots, P$. A grid missing some of those values, such as $\{1, 2, 5, 10, \ldots\}$ with $P = 4$, isn't covered by this argument, so RQ4 can't cite part 1 for that grid.
- This shows that pass@k can't determine the split in principle. How often a practical rule gets a run wrong is RQ4's question.

## 6. Estimator bound

**Setting.** Draw $M$ i.i.d. trajectories from each policy. Each of the four means, mass on $\mathcal{R}_N$ and on $\mathcal{U}_N$ under each policy, is estimated by the sample mean $\hat{m}$ of $\{0, 1\}$ indicators with true mean $m$. Classifying a sample uses an exact lookup of $q_{\pi_0}$.

**Statement (D11).** Let $0 < \eta \le 1$, $\mu > 0$ and

$$M \ge \frac{3 \ln(8/\delta)}{\eta^2 \mu}.$$

With probability at least $1 - \delta$, every mean satisfies $|\hat{m} - m| \le \eta \max(m, \mu)$. On that event, each $\Delta = m_1 - m_0$ satisfies

$$|\hat{\Delta} - \Delta| \le \eta \big(\max(m_1, \mu) + \max(m_0, \mu)\big).$$

**Proof.** Fix one mean. If $m = 0$, then $\hat{m} = 0$ and there's no error, so let $m > 0$. Write $m' = \max(m, \mu)$ and $t = \eta m'$. The multiplicative Chernoff bounds for a sum of independent indicators are, with $\gamma = t/m$:

$$\Pr[\hat{m} \ge (1 + \gamma) m] \le \exp\!\left(-\frac{\gamma^2 m M}{2 + \gamma}\right) \quad (\gamma > 0), \qquad \Pr[\hat{m} \le (1 - \gamma) m] \le \exp\!\left(-\frac{\gamma^2 m M}{2}\right) \quad (0 < \gamma \le 1).$$

**Upper tail.** The exponent is $M t^2 / (2m + t)$. Since $m \le m'$ and $\eta \le 1$, we have $2m + t \le 3m'$, so

$$\frac{M t^2}{2m + t} \ge \frac{M \eta^2 m'^2}{3 m'} = \frac{M \eta^2 m'}{3} \ge \frac{M \eta^2 \mu}{3} \ge \ln(8/\delta).$$

**Lower tail.** If $t > m$, the event would need $\hat{m} < 0$, so its probability is 0. Otherwise the exponent is $M t^2 / (2m)$, and since $m \le m'$,

$$\frac{M t^2}{2m} \ge \frac{M \eta^2 m'}{2} \ge \frac{M \eta^2 \mu}{3} \ge \ln(8/\delta).$$

So each tail has probability at most $\delta/8$. There are four means with two tails each, so the union bound over 8 events gives total failure probability at most $\delta$. It doesn't need independence, so means estimated from the same samples are fine. The bound on each $\Delta$ follows from the triangle inequality. $\blacksquare$

**Hoeffding version.** Each tail is at most $\exp(-2M\varepsilon^2)$, which is at most $\delta/8$ when $M \ge \ln(8/\delta) / (2\varepsilon^2)$. Every mean is then within $\varepsilon$, and every $\Delta$ within $2\varepsilon$.

**Limits.**

- **A relative bound on a mean isn't one on a difference.** If $m_1 \approx m_0$, the bound on $\Delta$ stays of order $\eta \max(m, \mu)$ while $\Delta$ itself can be near 0, so its relative error can be arbitrarily large.
- **The plan's wording is too strong.** It claims relative error $\eta$ on all four means, which can't hold below $\mu$. For example, with $\mu = 10^{-3}$, $\eta = 0.1$ and $\delta = 0.05$, a mean of $m = 10^{-5}$ gets about 15 expected hits, for a relative standard deviation near 0.26. D11's form replaces that claim.
- **Exact lookup is assumed.** If $q_{\pi_0}$ is estimated instead, samples can land on the wrong side of $c/N$, which this bound doesn't cover.
- **Rare mass.** Below $\mu$ the bound is only absolute. Importance sampling is the fallback and isn't in the pilot.

## 7. Time to cross on an isolated route

> **Status: written 2026-11-19; not yet re-derived by a second teammate.** Sections 1 to 6 are closed; this section is new for RQ2 and needs its own check.

**Setting.** A grid world has exactly one correct route $\tau^* = (s_1, a_1, \ldots, s_H, a_H)$, whose length equals the horizon $H$, so any trajectory that leaves it fails. Its cells $s_1, \ldots, s_H$ are distinct. The policy is tabular softmax with logits $\theta(s, a)$ over $A$ actions, so $J(\theta) = \prod_{i=1}^{H} \pi(a_i \mid s_i)$. The base gives the route action $a_i$ the logit $g_0$ and every other action at $s_i$ the logit 0. Training is exact gradient ascent, $\theta \leftarrow \theta + \eta \nabla J$.

**Statement.** Every update keeps all route cells identical. With $p = e^g / (e^g + A - 1)$ and $q = J = p^H$, the logit gap $g$ (route action minus any other action at a route cell) evolves as

$$g_{t+1} = g_t + \eta \, \frac{A}{A-1} \, q_t (1 - p_t).$$

In continuous time, the time from $q_0$ to the line $\ell$ is

$$T(q_0) = \frac{A-1}{A \eta} \int_{q_0^{1/H}}^{\ell^{1/H}} \frac{dp}{p^{H+1} (1-p)^2}, \qquad T(q_0) = \frac{A-1}{A \eta H q_0} \big(1 + O(q_0^{1/H})\big) \text{ as } q_0 \to 0.$$

**Proof.** Since $\partial \log \pi(a_i \mid s_i) / \partial \theta(s_i, a) = \mathbb{1}[a = a_i] - \pi(a \mid s_i)$, and $J$ is a product over distinct cells,

$$\frac{\partial J}{\partial \theta(s_i, a)} = J \big(\mathbb{1}[a = a_i] - \pi(a \mid s_i)\big),$$

and the gradient is 0 at cells off the route. Suppose that at some step every route cell has gap $g$ and equal logits on its $A - 1$ other actions (true at step 0). Then $\pi(a_i \mid s_i) = p$, and $\pi(a \mid s_i) = (1-p)/(A-1)$ for $a \ne a_i$. The route action gains $\eta J (1 - p)$, and each other action loses $\eta J (1-p)/(A-1)$. These are the same at every route cell, so the cells stay identical and the other actions stay equal. The gap grows by the sum, $\eta J (1-p)(1 + 1/(A-1)) = \eta \frac{A}{A-1} J (1-p)$. Softmax depends only on logit differences, so $p$, and therefore $q = p^H$, depends only on $g$. That proves the recursion by induction.

For the continuous version, $dp/dg = p(1-p)$, so $dp/dt = \eta \frac{A}{A-1} p^{H+1}(1-p)^2$. Separating variables from $p_0 = q_0^{1/H}$ to $\ell^{1/H}$ gives the integral. For the asymptotic, expand $(1-p)^{-2} = 1 + 2p + 3p^2 + \cdots$. The first term integrates to $(q_0^{-1} - \ell^{-1})/H$, and the next to $2(q_0^{-1 + 1/H} - \ell^{-1+1/H})/(H-1)$, which is smaller by a factor of order $q_0^{1/H}$. $\blacksquare$

**What this means for RQ2.** The local log-log slope of $T$ against $q_0$ tends to $-1$, the plan's "roughly $1/q_0$". It gets there slowly, because the correction is of order $q_0^{1/H}$, which is 0.18 at $q_0 = 10^{-6}$ and $H = 8$. That is why the slope over $[10^{-6}, 10^{-2}]$ is $-0.844$ rather than $-1$, and why the local slope between the two smallest points is only $-0.928$. At larger $q_0$ the fixed cost of climbing to $\ell$ dominates and the curve flattens. Since $p_0$ is far from 0, the leading term alone underestimates $T$: at $q_0 = 10^{-6}$ it gives 93,750 updates against 148,403. Keeping the factor $(1 - p_0)^{-2}$ gets within about 7%.

**Limits.**

- **Symmetry is needed.** If the off-route logits differ between cells, or within one cell, the cells no longer move together and the recursion is only an approximation. RQ2's robustness draws (design §6.2, R1) measure how far it drifts.
- **The route must fill the horizon.** If $H$ is longer than the route, detours that loop back are erased onto the same route (`loop_erased`), so more trajectories count as correct and $J$ is no longer a product over $H$ fixed cells (R2).
- **Exact gradients only.** Under sampled REINFORCE, a rare route has to be sampled before it gets any gradient, so time to cross is random and can be censored (design §6.4).
- **The factor $A/(A-1)$ depends on the action count.** It is 4/3 for the grid's 4 moves. Leaving it out, as a first draft of the design did, overestimates $T$ by a third.

## Numerical checks

These were run on Nov 6 with a throwaway script that isn't in the repo, except the last three rows, which are tests in `tests/test_crossing.py` (added Nov 19). They support the proofs but don't replace checking them.

| Check | Expected | Got |
|---|---|---|
| Chernoff $M$ at $\mu = 10^{-3}, \eta = 0.1, \delta = 0.05$ | 1,522,553 | 1,522,553 |
| Hoeffding $M$ at $\varepsilon = 10^{-4}, \delta = 0.05$ | 253,758,691 | 253,758,691 |
| Proposition 3, $\pi_1$ | $(+0.4, -0.003)$, sum 0.397 | Match |
| Proposition 3, $\pi_1'$ | $(-0.5, +0.897)$, sum 0.397 | Match |
| Proposition 2 on no-wall grids ($3{\times}3$ with $H = 4, 6$ and $4{\times}4$ with $H = 6$), 40 random policies each, $\beta$ from 1 to 8, both families | Bound holds | Holds, largest excess in log ratio $7 \times 10^{-15}$ (rounding) |
| Corollary on the same runs | No solution rises | 0 rises in 11,660 cases |
| Corollary example | $q(s)$ falls at $\beta = 2$ | 0.308 and 0.185, from 0.4 |
| Newton reconstruction, 200 random multisets of size 5 | Recovers the multiset | Recovered |
| Exact binomial tails at $\mu = 0.05, \eta = 0.5, \delta = 0.05$ ($M = 1219$), means from 0.001 to 0.9 | Each tail at most 0.00625 | Largest $8.2 \times 10^{-5}$ |
| Section 7 recursion against the pilot's full-table exact PG, isolated world, $\eta = 1$, $q_0 = 10^{-5} \ldots 10^{-2}$ | 17,842, 2,358, 362, 63 updates | Match exactly |
| Section 7 integral against the blueprint's D6 table, $q_0 = 10^{-6} \ldots 10^{-2}$ | 148,403, 17,837, 2,354, 359, 61 | Match exactly (rounded) |
| Section 7 slopes over 9 log-spaced $q_0$ in $[10^{-6}, 10^{-2}]$ | Fitted near $-0.85$, local slope tending to $-1$ | Fitted $-0.844$; smallest-pair local $-0.928$ |
