"""Discovery (waiting-time) analysis: geometric formulas, synthetic result folders, variance split."""
import json
import math
import pathlib

import numpy as np
import pytest

from capsplit.harness import discovery as d

G, B = 16, 1000
CHECKPOINTS = [0, 10, 25, 50, 100, 150, 200, 300, 500, 750, 1000]
K = "task.shared_prefix"


# --------------------------------------------------------------------------- formulas


def test_geometric_formulas_on_known_p():
    p = 0.006
    pi = 1 - (1 - p) ** G
    assert d.discovery_prob(p, G) == pytest.approx(pi, rel=1e-12)
    assert d.expected_first_success(p, G) == pytest.approx(1 / pi, rel=1e-12)
    assert d.median_first_success(p, G) == math.ceil(math.log(2) / (-G * math.log(1 - p)))
    assert d.median_first_success(p, G) == 8  # ln2 / (16 * 0.006018) = 7.2
    assert d.p_undiscovered(p, G, B) == pytest.approx((1 - p) ** (G * B), rel=1e-9)
    # The median is the smallest t with P(T <= t) >= 1/2.
    m = d.median_first_success(p, G)
    assert d.geometric_cdf(m, p, G) >= 0.5 > d.geometric_cdf(m - 1, p, G)
    # Edge cases.
    assert d.median_first_success(0.0, G) is None and d.expected_first_success(0.0, G) is None
    assert d.p_undiscovered(0.0, G, B) == 1.0
    assert d.median_first_success(1.0, G) == 1 and d.p_undiscovered(1.0, G, B) == 0.0
    assert d.median_first_success(0.5, 1) == 1  # P(T = 1) = 1/2
    record = d.prompt_prediction(3, p, G, B)
    assert record["id"] == 3 and record["median_T"] == 8


def test_geometric_median_matches_simulation():
    rng = np.random.default_rng(1)
    p = 0.002
    pi = d.discovery_prob(p, G)
    draws = rng.geometric(pi, size=200_000)
    assert abs(np.median(draws) - d.median_first_success(p, G)) <= 1
    assert draws.mean() == pytest.approx(d.expected_first_success(p, G), rel=0.02)


def test_empirical_and_mixture_medians():
    assert d.empirical_median([1, 2, 3, None], B) == 2
    assert d.empirical_median([None, None, 5], B) is None
    assert d.empirical_median([4], B) == 4
    ps = [0.01] * 5
    assert d.mixture_median(ps, G, B) == d.median_first_success(0.01, G)
    assert d.rank_auc([3, 4], [1, 2]) == 1.0 and d.rank_auc([1], [1]) == 0.5 and d.rank_auc([], [1]) is None


def test_pit_is_uniform_under_the_model():
    rng = np.random.default_rng(2)
    ps = rng.uniform(0.0005, 0.02, size=4000)
    times = [int(t) if t <= B else None for t in rng.geometric(1 - (1 - ps) ** G)]
    u = d.randomized_pit(times, list(ps), G, B, np.random.default_rng(0))
    summary = d.pit_summary(u)
    assert abs(summary["z"]) < 3.5
    assert max(summary["histogram_fraction"]) < 0.13 and min(summary["histogram_fraction"]) > 0.07
    ks = d.ks_distance(times, list(ps), G, B)
    assert ks["D"] < 2 * ks["iid_reference_5pct"]


# --------------------------------------------------------------------------- synthetic result folders


def _simulate(rng, p0, decay=None):
    """First success update and per-checkpoint success probability for one prompt in one run."""
    t = np.arange(1, B + 1)
    p_t = p0 * np.ones(B) if decay is None else p0 * np.exp(-(t - 1) / decay)
    hit = np.flatnonzero(rng.random(B) < 1 - (1 - p_t) ** G)
    first = int(hit[0]) + 1 if hit.size else None
    j1 = []
    for c in CHECKPOINTS:
        p_c = p0 if decay is None else p0 * math.exp(-c / decay)
        j1.append(0.9 if first is not None and c >= first else p_c)
    return first, j1


def _write(path: pathlib.Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _build(root: pathlib.Path, name: str, kind: str, base_ps: dict, n_seeds: int, rng, decay=None, gain=None):
    """A fake results folder and predictions for one study.

    base_ps: {k: [per-base list of p lists]}; kind "train" uses base 0 for every seed, "base"
    uses base s for seed s. gain(rng, k, prompt index, base index) gives the final delta_cap.
    """
    exp = root / name
    runs, bases = [], []
    for k, per_base in base_ps.items():
        for b_index, ps in enumerate(per_base):
            key = f"{name}k{k}b{b_index}"
            bases.append(d.base_record({K: k}, b_index, key, [], [d.prompt_prediction(i, p, G, B) for i, p in enumerate(ps)]))
        for seed in range(n_seeds):
            b_index = 0 if kind == "train" else seed
            ps = per_base[b_index]
            key = f"{name}k{k}b{b_index}"
            cell = f"cell{k}"
            ids = list(range(len(ps)))
            prompts, j1s, caps = {}, [], []
            for i, p in enumerate(ps):
                first, j1 = _simulate(rng, p, decay.get(k) if decay else None)
                prompts[str(i)] = {"first_success_update": first}
                j1s.append(j1)
                caps.append(gain(rng, k, i, b_index) if gain else (0.9 - p if first else 0.0))
            records = [
                {"group": "hard_train", "n": 64, "c": 1.0, "step": c, "delta_cap": float(np.mean(caps)) * (c == B),
                 "extra": {"j0": list(ps), "j1": [j[ci] for j in j1s], "delta_cap": [x * (c == B) for x in caps]}}
                for ci, c in enumerate(CHECKPOINTS)
            ]
            files = {"metrics": f"{cell}/seed-{seed}.metrics.json", "training": f"{cell}/seed-{seed}.training.json"}
            _write(exp / files["metrics"], {"groups": {"hard_train": ids}, "records": records})
            _write(exp / files["training"], {"prompts": prompts})
            runs.append({
                "arm": "rloo", "trainer": "rloo", "seed": seed, "factors": {K: k}, "run_dir": cell,
                "base": f"bases/{key}", "files": files,
                "base_seed": b_index, "train_seed": seed if kind == "train" else 0, "measure_seed": None,
            })
    _write(exp / "index.json", {
        "name": name, "budget": B, "checkpoints": CHECKPOINTS, "runs": runs,
        "measure": {"ns": [16, 64], "cs": [1], "group_by": ["difficulty", "split"], "primary": {"n": 64, "c": 1}},
    })
    pred = root / f"{name}.predictions.json"
    _write(pred, d.predictions_document(name, "rloo", "rloo", G, B, bases))
    return exp, pred


def test_analyze_agreement_on_geometric_data(tmp_path):
    rng = np.random.default_rng(0)
    ps = list(np.geomspace(0.0015, 0.02, 20))
    exp, pred = _build(tmp_path, "train", "train", {4: [ps]}, 30, rng)
    out = d.analyze(exp, pred, tmp_path / "out")
    data = json.loads((out / "analysis.json").read_text())
    (fkey,) = data["verdicts"]
    v = data["verdicts"][fkey]
    assert v["label"] == d.AGREEMENT, v
    s = data["combinations"][fkey]
    assert s["median_T"]["fraction_within_factor_2"] > 0.9
    assert abs(s["pit"]["mean"] - 0.5) < 0.05 and abs(s["pit"]["z"]) < 3
    assert s["never"]["observed_count"] == 0
    assert data["max_abs_p_minus_metrics_j0"] == 0.0
    assert data["study_kinds"] == {"train": "train"}
    assert s["discovery_rate"]["all"] == 20
    assert s["drift"]["fraction_below_1"] in (None, 0.0)  # no drift was simulated
    md = (out / "analysis.md").read_text()
    assert "**agreement**" in md
    assert all((out / "figures" / f).exists() for f in (
        "first_success_cdf.png", "drift_before_discovery.png", "discovery_rate_vs_p.png"))
    # Deterministic: a second run gives identical files.
    out2 = d.analyze(exp, pred, tmp_path / "out2")
    assert (out2 / "analysis.json").read_text() == (out / "analysis.json").read_text()
    assert (out2 / "analysis.md").read_text() == md


def test_analyze_drift_is_later_than_predicted(tmp_path):
    rng = np.random.default_rng(1)
    ps = list(np.geomspace(0.001, 0.006, 20))
    exp, pred = _build(tmp_path, "drift", "train", {0: [ps]}, 30, rng, decay={0: 15.0})
    data = json.loads((d.analyze(exp, pred, tmp_path / "out") / "analysis.json").read_text())
    (fkey,) = data["verdicts"]
    v, s = data["verdicts"][fkey], data["combinations"][fkey]
    assert v["label"] == d.PREDICTED_DIRECTION, v
    assert s["never"]["observed_fraction"] > 0.1 > s["never"]["predicted_fraction"]
    assert s["pit"]["z"] > 3 and s["ks"]["D_minus"] > s["ks"]["D_plus"]
    assert s["drift"]["median_ratio"] < 1
    assert s["discovery_rate"]["some"] > 0
    # Never-discovered observations had lower base p.
    assert s["base_p_never_vs_discovered"]["auc_never_higher"] < 0.5


def test_verdict_flags_high_p_never_discovered():
    summary = {
        "median_T": {"ratio_pooled": 1.0, "fraction_within_factor_2": 0.9, "units_with_replicates": 10},
        "never": {"observed_fraction": 0.0, "observed_count": 0},
        "pit": {"z": 0.0},
        "high_p_anomaly": {"n_mostly_undiscovered": 2, "auc_mostly_undiscovered_higher": 0.9},
    }
    assert d.verdict(summary, 4)["label"] == d.AGAINST
    summary["high_p_anomaly"]["auc_mostly_undiscovered_higher"] = 0.1
    assert d.verdict(summary, 4)["label"] == d.AGREEMENT
    summary["median_T"].update(ratio_pooled=0.2, fraction_within_factor_2=0.1)
    summary["pit"]["z"] = -10.0
    assert d.verdict(summary, 0)["label"] == d.AGAINST
    assert d.verdict(summary, 4)["label"] == d.POSITIVE_DRIFT


# --------------------------------------------------------------------------- variance decomposition


def test_variance_decomposition_constructed():
    assert d.per_prompt_variance({0: [1.0, 3.0], 1: [0.0, 0.0, 0.0]}) == pytest.approx(1.0)  # (2 + 0) / 2
    assert d.decompose(0.1, 1.0)["verdict"] == d.BY_BASE
    assert d.decompose(0.9, 1.0)["verdict"] == d.BY_LUCK
    assert d.decompose(0.5, 1.0)["verdict"] == d.MIXED
    assert d.decompose(None, 1.0)["verdict"] is None and d.decompose(0.1, 0.0)["share"] is None

    def obs(fkey, seed, prompt, value):
        return {"fkey": fkey, "seed": seed, "prompt": prompt, "final_delta_cap": value, "run_delta_cap": value}

    train = d.components([obs("k", s, i, 0.5 + 0.01 * (s % 2)) for s in range(4) for i in range(3)])
    base = d.components([obs("k", s, i, float(s % 2)) for s in range(4) for i in range(3)])
    assert train["k"]["per_prompt"] == pytest.approx(np.var([0.5, 0.51, 0.5, 0.51], ddof=1))
    assert base["k"]["per_prompt"] == pytest.approx(np.var([0, 1, 0, 1], ddof=1))
    out = d.variance_decomposition({"train": train, "base": base})["k"]
    assert out["per_prompt"]["verdict"] == d.BY_BASE and out["run_level"]["verdict"] == d.BY_BASE
    swapped = d.variance_decomposition({"train": base, "base": train})["k"]
    assert swapped["per_prompt"]["verdict"] == d.BY_LUCK


def test_analyze_with_both_studies(tmp_path):
    rng = np.random.default_rng(3)
    base_ps = [list(np.geomspace(0.002, 0.02, 8) * f) for f in np.linspace(0.6, 1.4, 12)]

    def gain(rng_, k, i, b):  # determined by the base: depends on (prompt, base) only
        return 0.3 + 0.04 * ((i * 7 + b * 3) % 11)

    train_exp, train_pred = _build(tmp_path, "train", "train", {2: base_ps[:1]}, 12, rng, gain=gain)
    base_exp, base_pred = _build(tmp_path, "base", "base", {2: base_ps}, 12, rng, gain=gain)
    out = d.analyze(train_exp, train_pred, tmp_path / "out", compare=[(base_exp, base_pred)])
    data = json.loads((out / "analysis.json").read_text())
    assert data["study_kinds"] == {"train": "train", "base": "base"}
    (dec,) = data["variance_decomposition"].values()
    assert dec["per_prompt"]["V_train"] < 1e-20 and dec["per_prompt"]["verdict"] == d.BY_BASE
    # The base study alone: one run per base, so the median criterion falls back to the pooled ratio.
    alone = json.loads((d.analyze(base_exp, base_pred, tmp_path / "out_b") / "analysis.json").read_text())
    (v,) = alone["verdicts"].values()
    assert v["checks"]["median_rule"].startswith("no prompt has")
    assert v["label"] == d.AGREEMENT


def test_analyze_rejects_mismatched_predictions(tmp_path):
    rng = np.random.default_rng(4)
    exp, _ = _build(tmp_path, "a", "train", {4: [[0.01, 0.02]]}, 2, rng)
    _, other = _build(tmp_path, "b", "train", {4: [[0.01, 0.02]]}, 2, rng)
    with pytest.raises(ValueError, match="no prediction"):
        d.analyze(exp, other, tmp_path / "out")


# --------------------------------------------------------------------------- predict on a tiny real config


def test_predict_uses_the_runner_base_cache(tmp_path):
    pytest.importorskip("torch")
    from capsplit.harness.runner import run_experiment

    config = {
        "name": "tiny_disc",
        "task": {"type": "tiny_prompt", "params": {"vocab": 3}},
        "base": {"type": "uniform_tabular", "params": {"init_scale": 0.5}},
        "backend": {"type": "exact", "params": {}},
        "arms": [{"name": "rloo", "trainer": "rloo", "params": {"group": 4, "lr": 0.1}}],
        "seeds": [0, 1],
        "base_seed": 0,
        "budget": 6,
        "checkpoints": [0, 3, 6],
        "measure": {"ns": [16], "cs": [1], "group_by": ["difficulty", "split"]},
        "save_policies": "none",
    }
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps(config))
    out = d.predict(path, tmp_path / "pred", results_root=tmp_path / "results")
    pred = json.loads((out / "predictions.json").read_text())
    assert pred["group"] == 4 and pred["budget"] == 6
    (base,) = pred["bases"]  # one effective base seed
    assert base["run_seeds"] == [0, 1]
    assert (out / "predictions.md").read_text().startswith("# Discovery predictions: tiny_disc")
    exp = run_experiment(path, out_root=tmp_path / "results")
    index = json.loads((exp / "index.json").read_text())
    assert {r["base"] for r in index["runs"]} == {f"bases/{base['base_key']}"}
    metrics = json.loads((exp / index["runs"][0]["files"]["metrics"]).read_text())
    rec = next(r for r in metrics["records"] if r["group"] == "hard_train" and r["step"] == 0)
    assert [p["p"] for p in base["prompts"]] == pytest.approx(rec["extra"]["j0"], abs=1e-15)
    assert [p["id"] for p in base["prompts"]] == metrics["groups"]["hard_train"]
    result = d.analyze(exp, out / "predictions.json", tmp_path / "analysis")
    assert (result / "analysis.md").exists()
