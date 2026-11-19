"""The pre-registered decision rules in capsplit.harness.verdicts."""
import numpy as np

from capsplit.harness.verdicts import (
    INCONCLUSIVE,
    NOT_SUPPORTED,
    SUPPORTED,
    _combine,
    decide_gt,
    decide_le,
)


def test_decide_gt_three_outcomes():
    assert decide_gt(np.full(10, 0.5), 0.02)[0] == SUPPORTED
    assert decide_gt(np.full(10, 0.0), 0.02)[0] == NOT_SUPPORTED
    assert decide_gt(np.array([0.0] * 5 + [0.06] * 5), 0.02)[0] == INCONCLUSIVE


def test_decide_le_mirrors_decide_gt():
    assert decide_le(np.full(10, 0.0), 0.02)[0] == SUPPORTED
    assert decide_le(np.full(10, 0.5), 0.02)[0] == NOT_SUPPORTED
    assert decide_le(np.array([0.0] * 5 + [0.06] * 5), 0.02)[0] == INCONCLUSIVE


def test_decision_uses_the_interval_not_the_mean():
    # Mean 0.025 is above tau, but the interval reaches below it: inconclusive, not supported.
    values = np.array([0.0, 0.05] * 5)
    verdict, stat = decide_gt(values, 0.02)
    assert stat["mean"] > 0.02 and stat["ci"][0] < 0.02 and verdict == INCONCLUSIVE


def test_combine():
    assert _combine([SUPPORTED, SUPPORTED]) == SUPPORTED
    assert _combine([SUPPORTED, NOT_SUPPORTED, INCONCLUSIVE]) == NOT_SUPPORTED
    assert _combine([SUPPORTED, INCONCLUSIVE]) == INCONCLUSIVE


def test_verdicts_refuse_an_experiment_the_plan_does_not_cover(tmp_path):
    import subprocess
    import sys

    from capsplit.harness.runner import run_experiment

    exp = run_experiment("configs/dummy.json", out_root=tmp_path)
    result = subprocess.run([sys.executable, "-m", "capsplit.harness", "verdicts", str(exp)], capture_output=True, text=True)
    assert result.returncode == 2 and "does not apply" in result.stderr
