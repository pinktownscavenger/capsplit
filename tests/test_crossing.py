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


# --------------------------------------------------------------- isolated-route scalar model

from capsplit.crossing import integral_time, scalar_isolated_trace, scalar_isolated_updates  # noqa: E402


@pytest.mark.parametrize("q0,updates", [(1e-5, 17842), (1e-4, 2358), (1e-3, 362), (1e-2, 63)])
def test_scalar_recursion_reproduces_pilot_counts(q0, updates):
    assert scalar_isolated_updates(q0, 8, 1.0, 0.1, 30000) == updates


@pytest.mark.parametrize("q0,updates", [(1e-6, 148403), (1e-5, 17837), (1e-4, 2354), (1e-3, 359), (1e-2, 61)])
def test_integral_reproduces_blueprint_d6(q0, updates):
    assert round(integral_time(q0, 8, 1.0, 0.1)) == updates


def test_integral_within_5pct_of_recursion():
    for q0 in np.logspace(-6, -2, 9):
        assert integral_time(q0, 8, 1.0, 0.1) == pytest.approx(scalar_isolated_updates(q0, 8, 1.0, 0.1, 400_000), rel=0.05)


def test_scalar_fitted_and_local_slopes():
    qs = np.logspace(-6, -2, 9)
    fitted, local = loglog_slopes(qs, [scalar_isolated_updates(q, 8, 1.0, 0.1, 400_000) for q in qs])
    assert fitted == pytest.approx(-0.844, abs=0.001)
    assert local[0] == pytest.approx(-0.928, abs=0.001)


def test_scalar_time_scales_inversely_with_lr():
    t1 = scalar_isolated_updates(1e-4, 8, 1.0, 0.1, 100_000)
    assert abs(0.5 * scalar_isolated_updates(1e-4, 8, 0.5, 0.1, 100_000) - t1) / t1 < 0.05


def test_scalar_trace_starts_at_q0_and_stops_at_line():
    trace = scalar_isolated_trace(1e-3, 8, 1.0, 30000, stop_line=0.1)
    assert trace[0] == pytest.approx(1e-3, rel=1e-12)
    assert len(trace) - 1 == 362 and trace[-1] >= 0.1 > trace[-2]


def test_scalar_censored_returns_none():
    assert scalar_isolated_updates(1e-5, 8, 1.0, 0.1, 100) is None
