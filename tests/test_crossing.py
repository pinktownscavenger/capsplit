import numpy as np
import pytest

from capsplit.crossing import crossing_step, loglog_slopes


def test_crossing_first_index_at_or_above():
    assert crossing_step(np.array([0.01, 0.05, 0.1, 0.2]), 0.1) == 2


def test_crossing_censored():
    assert crossing_step(np.array([0.01, 0.05]), 0.1) is None


def test_crossing_already_above_at_step_zero():
    assert crossing_step(np.array([0.3, 0.2]), 0.1) == 0


def test_slopes_recover_power_law():
    xs = np.logspace(-6, -2, 9)
    fitted, local = loglog_slopes(xs, 5.0 * xs**-0.9)
    assert fitted == pytest.approx(-0.9, abs=1e-12)
    assert local.shape == (8,) and np.allclose(local, -0.9)


def test_slopes_refuse_censored_point():
    with pytest.raises(ValueError, match="censored"):
        loglog_slopes([1e-3, 1e-2], [None, 10])


def test_slopes_refuse_nonpositive_and_too_few():
    with pytest.raises(ValueError):
        loglog_slopes([1e-3, 1e-2], [0, 10])
    with pytest.raises(ValueError):
        loglog_slopes([1e-3], [10])
