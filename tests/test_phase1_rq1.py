"""Regression: planted cases land in their predicted 2x2 cells (RQ1, trap base, N=20)."""
import contextlib
import importlib.util
import io
import pathlib

spec = importlib.util.spec_from_file_location(
    "phase1_rq1", pathlib.Path(__file__).parents[1] / "experiments" / "phase1_rq1.py"
)
rq1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rq1)

N = 20


def label(lg1, sharpen=None):
    base = rq1.trap(3.2, 3.2, 0.0)
    with contextlib.redirect_stdout(io.StringIO()):
        if sharpen is not None:
            return rq1.report("x", base, None, N, lambda q0: rq1.enumerate_paths(rq1.WORLD, base, sharpen))
        return rq1.report("x", base, lg1, N)


def test_pure_sharpening_cell():
    assert label(None, sharpen=3.0) == "pure_sharpening"


def test_reweighting_cell():
    assert label(rq1.lg(2.0, 5.0, 0.0)) == "reweighting"


def test_capability_cell():
    assert label(rq1.lg(3.2, 3.2, 4.0)) == "capability"
