# RQ2 time-to-cross verdicts

**Validity:** valid

- arm_a_matches_scalar: pass
- bases_within_1pct: pass
- arm_b_rule_achievable: pass

| Prediction | Verdict | Numbers |
|---|---|---|
| P1 | supported | `{"fitted": -0.8436288363452707, "local": [-0.9278652852764573, -0.9121681221738065, -0.8920212938024452, -0.865778174746139, -0.8327507405674144, -0.7949197198843949, -0.7594632464799931, -0.7592727956791752]}` |
| P2 | supported | `{"r1_at_most_-0.7": 20, "r1_draws": 20, "r1_median": -0.8493816868164308, "r2_slope": -0.8744904292374033}` |
| P3 | holds | `{"worst_spread": 0.031746031746031744, "spreads": {"1e-06": 5.390484447329352e-05, "3.162277660168379e-06": 0.00015687607443769732, "1e-05": 0.00036429872495446266, "3.1622776601683795e-05": 0.0010171878667744711, "0.0001": 0.002543810062182024, "0.00031622776601683794": 0.004424778761061947, "0.001": 0.009664058904739991, "0.0031622776601683794": 0.016574585635359115, "0.01": 0.031746031746031744}}` |
| P4 | inconclusive | `{"worst_fraction_at_ratio_le_0.5": 0.3}` |

P1 note: Settled by the derivation in docs/proofs.md section 7: the full-table run equals the scalar recursion.
