import subprocess
import sys

import pytest


def test_k_large_is_frozen():
    pytest.importorskip("torch")
    from pilot_rq4 import K_LARGE

    assert K_LARGE == 10


@pytest.mark.parametrize(
    "n_actions,horizon,expected",
    [
        (4, 6, (1, 2, 5, 10, 20, 40)),  # 4096 trajectories give a largest valid N of 40
        (6, 5, (1, 2, 5, 10, 20, 50, 77)),  # 7776 give 77
        (2, 7, (1,)),  # 128 give 1
    ],
)
def test_k_grid_stops_at_the_largest_valid_budget(n_actions, horizon, expected):
    pytest.importorskip("torch")
    from pilot_rq4 import k_grid

    assert k_grid(n_actions, horizon) == expected


def test_run_pilot_covers_every_experiment():
    pytest.importorskip("torch")
    from run_pilot import experiments

    assert [name for name, _ in experiments(quick=True)] == [
        "pilot_smoke",
        "pilot_rq1",
        "pilot_rq2",
        "pilot_estimator",
        "pilot_seq",
        "pilot_rq4",
    ]


def test_package_exports_the_pilot_api():
    code = (
        "import sys, capsplit\n"
        "assert 'torch' not in sys.modules\n"
        "import types\n"
        "assert isinstance(capsplit.evaluate, types.ModuleType)\n"
        "for name in ('evaluate_benchmark', 'frontier', 'TrajectoryTable', 'compile_grid',\n"
        "             'passk_rule', 'score_rule', 'estimate_split', 'run_dir', 'split'):\n"
        "    assert hasattr(capsplit, name), name\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env={"PYTHONPATH": "src"})
