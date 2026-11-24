# Base-model tail probability: the answer in plain words

**Question:** for the problems the base model almost never solves, how likely is it to solve one, and does knowing that change the reachability labels or the discovery test?

**Answer:**
- **Certified intervals.** Every problem has one: an exact bracket, exact up to the model's float16 arithmetic.
  - About two problems in five of the target set have a tight bracket. These are mostly addition and smaller multiplication.
  - The rest, mostly 4-digit multiplication, are loose. Their upper end is limited by the unexpanded mass, not by the model.
  - The counts are in the generated report below.
- **Importance-sampling estimates: none usable.** The registered stop gate failed, so this layer reports brackets only.
- **Labels:** they change once the budget goes past the old curve's end. Problems called unreachable at the main run's budgets become ambiguous or reachable, and well before the largest budget no problem remains certified unreachable.
- **Discovery test:** it gains nothing decisive. Where RL saw no correct base sample, the bracket allows anywhere from about one predicted discovery to many, against none observed. The bracket is too loose to say whether zero discoveries is surprising.

## Negative results, stated first

1. **The stop gate failed on both controls** (DECISIONS L43). Neither failure points to a bug in the bracket or the estimator.
   - **Known-answer control.** It failed on one problem, and across the trusted problems its z values lean negative. The diagnosis (`diagnostics.txt`) points to three things:
     - **A rare draw.** On the failing problem, the proposal component that produces almost all correct samples was drawn unusually rarely.
     - **Float16 numerics.** On the Mac GPU, a completion's log-probability depends on the batch it is scored in, by up to about a tenth of a nat. That was enough to push this problem over the line. The noise averages out across problems, but the estimator's standard errors do not include it. In float32 on CPU it disappears.
     - **An unexplained lean.** After both effects are accounted for, a small downward lean of the estimates against D remains.
     - The proposal densities themselves check out: the scored and sampled values agree.
   - **Brute-force control.** No problem met the trust rule.
     - This was a flaw in my design. The control problems were ones where the bracket had already resolved nearly all the mass, so importance sampling had almost nothing left to estimate.
     - The brute-force rates did agree with the brackets themselves. That comparison was not registered as a check.
2. **Your prediction P2 is not supported as certified.** Most never-correct problems do not have a certified upper end below one in a thousand. Their lower bound D is below that level for all of them, so the claim may well be true, but this budget cannot certify it. P2 as estimated cannot be assessed, because there are no usable estimates.
3. **Your prediction P3 is not supported.** The predicted discovery count at the bracket's upper end is far above the registered line. My counter-prediction P3' holds: the bracket is too loose to rule discoveries out.
4. **Your prediction P1 is not supported in the registered form.** Tight and loose multiplication brackets have practically the same first-token digit probability: the model almost always starts with a digit. What separates them is how many digit paths are left unexpanded, not the opening token.

## What held

- **Checks against the samples:** the lower bound D and every bracket agree with the base samples at the registered level, so no bug was flagged. Gate 1 passed: D captures nearly all of the measured success where that success is measurable.
- **P1' (mine), fewer than half of the targets tight:** supported.
- **Discovery, with all unreachable observations pooled:** the bracket's lower end already predicts at least as many discoveries as the earlier sampled estimate. The observed count sits at its low edge.
- **Problem 178:** its brute-force rate sat above its bracket's upper end. A re-draw and a second seed showed this was a high draw, not a bug (L44).

## Independent re-derivation

`scripts/rederive_tail.py` recomputes one D, one bracket and one importance-sampling estimate without importing the tail code. Its output and the differences from the stage are in `rederivation.txt`.
- **D:** in float16 on the GPU, the re-derivation reproduces the stage's value to rounding. In float32 on CPU it comes out a few percent lower.
  - Measured against the float32 D, the failing known-answer problem would no longer cross the line.
  - So whether the known-answer control fails depends on numeric precision.
- **Bracket:** the independent implementation, run at a smaller budget, is sound and contains the stage's bracket.
- **Importance-sampling estimates:** they match exactly.
- **Not re-derived:** the tree region of each sample, and the teacher-forced log-probabilities, which were taken from the stored rows.

## Caveats

- **Scope:** one 0.5B model and 12-token arithmetic completions.
- **Unresolved:** the main run's null-check failure (z = −3.2).
- **Precision:** "certified" means exact for the float16 model on MPS. The same strings differ by a noticeable fraction of a nat in float32, and between batch compositions in float16.
- **Silent misses:** importance sampling can miss success strings its proposals never visit. The bracket cannot, but it is only as tight as its budget.
- **Order:** the pre-registration was written after the main results were known. The diagnosis of the failed gate was done after seeing it.
