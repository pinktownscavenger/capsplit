# RQ4 pass@k misdiagnosis verdicts

**Validity:** valid

Primary units: group `heldout`, final checkpoint, N = 64, c = 1.0, k_large = 10.

## Q1: Phase 2 (RQ3 runs): **contradicted**

Primary agreement 0.986, 95% CI [0.962, 1.000], 210 units; undetermined 0.000; secondary 0.976.

| Cell | Units | Primary | Secondary | Undetermined |
|---|---|---|---|---|
| capability | 3 | 0.000 | 0.000 | 0.000 |
| compounded_sharpening | 82 | 1.000 | 1.000 | 0.000 |
| reweighting | 2 | 1.000 | 0.000 | 0.000 |
| pure_sharpening | 123 | 1.000 | 1.000 | 0.000 |

## Q2: Phase 1 (grid sweep): **supported**

Primary agreement 0.550, 95% CI [0.383, 0.717], 60 units; undetermined 0.000; secondary 0.467.

| Cell | Units | Primary | Secondary | Undetermined |
|---|---|---|---|---|
| capability | 16 | 0.250 | 0.250 | 0.000 |
| compounded_sharpening | 19 | 0.842 | 0.842 | 0.000 |
| reweighting | 15 | 0.333 | 0.000 | 0.000 |
| pure_sharpening | 10 | 0.800 | 0.800 | 0.000 |

## Q3: disagreement concentrates in compounded sharpening and reweighting: **inconclusive**

capability 0.789, compounded_sharpening 0.030, reweighting 0.588, pure_sharpening 0.015

<!-- srr-ndr:start -->
## SRR and NDR (The Invisible Leash, Def. 3.3)

Group `heldout`, final checkpoint, eps = c/N = 0.015625, solution counted when its exact probability is > eps. P, S and E are summed over each unit's prompts and then over units before dividing; "undefined" means a zero denominator. Cells use each record's primary label (H12/H39, unratified).

### Phase 2 (RQ3 runs): not computed. The frozen Phase 2 run folder (results/main from RQ4-3) is not in the repo, and a rerun at rq4-frozen in a different environment (Python 3.13, numpy 2.5.3, torch 2.15.0) reproduced only 151 of the 210 frozen primary labels (all 210 rule calls match), so its policies are not the scored ones. Needs the original results/main.

### Phase 1 (grid sweep)

SRR 0.459, NDR 0.259 over 60 units (P 183, S 216, E 64); units with undefined SRR 0, NDR 0.

| Cell | Units | P | S | E | SRR | NDR |
|---|---|---|---|---|---|---|
| capability | 16 | 41 | 43 | 35 | 0.488 | 0.461 |
| compounded_sharpening | 19 | 55 | 61 | 29 | 0.474 | 0.345 |
| reweighting | 15 | 48 | 74 | 0 | 0.393 | 0.000 |
| pure_sharpening | 10 | 39 | 38 | 0 | 0.506 | 0.000 |
<!-- srr-ndr:end -->
