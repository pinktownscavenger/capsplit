"""The Report and 2x2 label, on reference values from the pilot blueprint."""
import numpy as np
import phase1_rq1 as rq1
import pytest

import capsplit.evaluate as evaluate_module
from capsplit import split
from capsplit.engine import solution_probs, table_success_prob, trajectory_probs
from capsplit.evaluate import evaluate, evaluate_benchmark, label_cell
from capsplit.tempering import frontier, sequence_probs
from capsplit.worlds import GridWorld, compile_grid, enumerate_paths, logits_to_array

N = 20


@pytest.fixture(scope="module")
def world():
    table, states = compile_grid(rq1.WORLD)
    bases = {
        "easy": logits_to_array(states, rq1.lg(2.5, 2.0, 0.0)),
        "weak": logits_to_array(states, rq1.lg(1.0, 0.0, 0.0)),
        "trap": logits_to_array(states, rq1.trap(3.2, 3.2, 0.0)),
    }
    return table, states, bases


def trained(table, states, bonuses):
    return trajectory_probs(table, logits_to_array(states, rq1.lg(*bonuses)))


CASES = [
    ("easy", "per", +0.7231, -0.0125, -0.0139, "pure_sharpening"),
    ("easy", "seq", +0.7287, -0.0128, -0.0083, "pure_sharpening"),
    ("weak", "per", +0.4261, +0.4261, -0.5524, "compounded_sharpening"),
    ("weak", "seq", +0.4597, +0.4597, -0.5189, "compounded_sharpening"),
    ("trap", (2.0, 5.0, 0.0), +0.7497, +0.0009, +0.7298, "reweighting"),
    ("trap", (3.2, 3.2, 4.0), +0.7173, +0.6073, +0.6973, "capability"),
]


def probs_for(table, states, logits0, pi1):
    if pi1 == "per":
        return trajectory_probs(table, logits0, 3.0)
    if pi1 == "seq":
        return sequence_probs(table, logits0, 3.0)
    return trained(table, states, pi1)


@pytest.mark.parametrize("base,pi1,delta_j,delta_cap,j_gap,label", CASES)
def test_reference_cases(world, base, pi1, delta_j, delta_cap, j_gap, label):
    table, states, bases = world
    report = evaluate(table, bases[base], probs_for(table, states, bases[base], pi1), N)
    assert report.delta_j == pytest.approx(delta_j, abs=0.001)
    assert report.delta_cap == pytest.approx(delta_cap, abs=0.001)
    assert report.j_gap == pytest.approx(j_gap, abs=0.001)
    assert report.label == label
    assert report.delta_rel + report.delta_cap == pytest.approx(report.delta_j, abs=1e-12)


def test_matches_existing_split_on_paths(world):
    table, states, bases = world
    q0 = enumerate_paths(rq1.WORLD, rq1.trap(3.2, 3.2, 0.0))
    q1 = enumerate_paths(rq1.WORLD, rq1.lg(3.2, 3.2, 4.0))
    correct = [s for s in q0 if s[-1] == rq1.WORLD.goal]
    old = split(q0, q1, correct, N)
    report = evaluate(table, bases["trap"], trained(table, states, (3.2, 3.2, 4.0)), N)
    assert report.delta_rel == pytest.approx(old["delta_rel"], abs=1e-12)
    assert report.delta_cap == pytest.approx(old["delta_cap"], abs=1e-12)


@pytest.mark.parametrize("base", ["easy", "weak"])
def test_boundary_reported(world, base):
    table, _, bases = world
    report = evaluate(table, bases[base], trajectory_probs(table, bases[base], 3.0), N)
    assert report.boundary_hit is True
    assert report.best_beta == pytest.approx(8.0, abs=0.01)


def test_improved_flag(world):
    table, states, bases = world
    assert evaluate(table, bases["trap"], trajectory_probs(table, bases["trap"], 3.0), N).improved is False
    assert evaluate(table, bases["trap"], trained(table, states, (2.0, 5.0, 0.0)), N).improved is True


def test_e_n_validity(world):
    table, _, bases = world
    probs = trajectory_probs(table, bases["easy"])
    assert evaluate(table, bases["easy"], probs, 40).e_n_valid is True
    assert evaluate(table, bases["easy"], probs, 100).e_n_valid is False


def test_label_comes_from_j_gap_when_excess_saturates(world):
    table, _, bases = world
    report = evaluate(table, bases["easy"], trajectory_probs(table, bases["easy"], 3.0), N)
    assert abs(report.excess) < 1e-6
    assert report.j_gap == pytest.approx(-0.014, abs=0.001)
    assert report.saturated is True
    assert report.label == "pure_sharpening"


def test_not_saturated_on_low_budget(world):
    table, _, bases = world
    assert evaluate(table, bases["trap"], trajectory_probs(table, bases["trap"]), 2).saturated is False


def test_crossed_lists_solutions_that_reach_the_line(world):
    table, _, bases = world
    report = evaluate(table, bases["weak"], trajectory_probs(table, bases["weak"], 3.0), N)
    assert tuple(rq1.R_COMMON) in report.crossed
    q0, q1 = report.crossed[tuple(rq1.R_COMMON)]
    assert q0 < 1 / N <= q1


def test_c_changes_the_line(world):
    table, states, bases = world
    probs = trained(table, states, (2.0, 5.0, 0.0))
    # Both trap routes sit at 0.0553: reachable at c=1 (line 0.05), not at c=3 (line 0.15).
    assert evaluate(table, bases["trap"], probs, N, c=1.0).delta_cap == pytest.approx(0.0009, abs=0.001)
    assert evaluate(table, bases["trap"], probs, N, c=3.0).delta_cap > 0.7


@pytest.mark.parametrize("total", [0.9, 1.1])
def test_probs_not_a_distribution_raises(world, total):
    table, _, bases = world
    with pytest.raises(ValueError):
        evaluate(table, bases["easy"], trajectory_probs(table, bases["easy"]) * total, N)


def test_negative_probs_raise(world):
    table, _, bases = world
    probs = np.zeros(table.index.shape[0])
    probs[0], probs[1] = 2.0, -1.0
    with pytest.raises(ValueError):
        evaluate(table, bases["easy"], probs, N)


def test_no_solutions_labels_without_crashing():
    table, _ = compile_grid(GridWorld(4, (0, 0), (3, 3), 3))
    logits = np.zeros((table.n_states, table.n_actions))
    report = evaluate(table, logits, trajectory_probs(table, logits), N)
    assert report.delta_j == 0.0
    assert report.label == "pure_sharpening"
    assert report.crossed == {}


def test_extreme_trained_policy(world):
    table, states, bases = world
    logits = logits_to_array(states, rq1.lg(3.2, 3.2, 4.0)) * 1000
    report = evaluate(table, bases["trap"], trajectory_probs(table, logits), N)
    assert np.isfinite([report.delta_cap, report.excess, report.j_gap]).all()


def test_frontier_result_is_reused(world, monkeypatch):
    table, _, bases = world
    best = frontier(table, bases["trap"])

    def fail(*args, **kwargs):
        raise AssertionError("frontier recomputed")

    monkeypatch.setattr(evaluate_module, "frontier", fail)
    evaluate(table, bases["trap"], trajectory_probs(table, bases["trap"]), N, frontier_result=best)


def test_reach_threshold_override(world):
    table, states, bases = world
    probs = trained(table, states, (2.0, 5.0, 0.0))
    assert evaluate(table, bases["trap"], probs, N, reach_threshold=0.0).label == "capability"
    assert evaluate(table, bases["trap"], probs, N).label == "reweighting"


@pytest.mark.parametrize(
    "delta_cap,j_gap,threshold,label",
    [
        (0.0, 0.0, 0.05, "pure_sharpening"),
        (0.05, 0.0, 0.05, "compounded_sharpening"),
        (0.04, 0.5, 0.05, "reweighting"),
        (0.5, 0.5, 0.05, "capability"),
        (0.0, 0.5, 0.0, "reweighting"),  # zero gain on U_N is never beyond reach
        (1e-15, 0.0, 0.0, "pure_sharpening"),  # nor is float noise, or a gap within eps
        (0.0, 1e-12, 0.05, "pure_sharpening"),
    ],
)
def test_label_cell(delta_cap, j_gap, threshold, label):
    assert label_cell(delta_cap, j_gap, threshold) == label


def test_one_prompt_benchmark_matches_evaluate(world):
    table, states, bases = world
    probs = trained(table, states, (3.2, 3.2, 4.0))
    single = evaluate(table, bases["trap"], probs, N)
    bench = evaluate_benchmark([table], [bases["trap"]], [probs], N)
    assert bench.delta_rel == pytest.approx(single.delta_rel)
    assert bench.delta_cap == pytest.approx(single.delta_cap)
    assert bench.delta_j == pytest.approx(single.delta_j)
    assert bench.per_prompt_j1 == pytest.approx((table_success_prob(table, probs),))


def test_benchmark_averages_prompts(world):
    table, states, bases = world
    p_a, p_b = trained(table, states, (2.0, 5.0, 0.0)), trained(table, states, (3.2, 3.2, 4.0))
    a, b = evaluate(table, bases["trap"], p_a, N), evaluate(table, bases["easy"], p_b, N)
    bench = evaluate_benchmark([table, table], [bases["trap"], bases["easy"]], [p_a, p_b], N)
    assert bench.delta_rel == pytest.approx((a.delta_rel + b.delta_rel) / 2)
    assert bench.delta_cap == pytest.approx((a.delta_cap + b.delta_cap) / 2)
    assert len(bench.per_prompt_j0) == 2


def test_benchmark_shared_beta_is_no_better_than_per_prompt_betas(world):
    # A shared beta can't beat each prompt choosing its own, so its excess is at least as high.
    table, states, bases = world
    probs = [trained(table, states, (2.0, 5.0, 0.0))] * 2
    bench = evaluate_benchmark([table, table], [bases["trap"], bases["easy"]], probs, 2)
    own = [1 - (1 - frontier(table, bases[b]).best_j) ** 2 for b in ("trap", "easy")]
    mean_pass = 1 - (1 - table_success_prob(table, probs[0])) ** 2
    assert bench.excess >= mean_pass - np.mean(own) - 1e-12


def test_benchmark_length_mismatch_raises(world):
    table, _, bases = world
    with pytest.raises(ValueError):
        evaluate_benchmark([table], [bases["trap"], bases["easy"]], [np.zeros(1)], N)


def test_solution_probs_sum_to_j(world):
    table, _, bases = world
    probs = trajectory_probs(table, bases["easy"])
    assert solution_probs(table, probs).sum() == pytest.approx(table_success_prob(table, probs))
