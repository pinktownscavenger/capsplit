"""Budget curve (capsplit.harness.budget): identity, planted values, N_train sources, CP bound, end to end."""
import csv
import json
import math
import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from capsplit.harness import TASKS  # noqa: E402
from capsplit.harness.budget import (  # noqa: E402
    SOURCE_CONFIG,
    SOURCE_LOG,
    SOURCE_UNKNOWN,
    build_budget_curve,
    clopper_pearson_zero_upper,
    crossover,
    default_grid,
    resolvable_n_max,
    split_curve,
)
from capsplit.harness.interfaces import Policy  # noqa: E402
from capsplit.harness.runner import run_experiment  # noqa: E402

pytest.importorskip("matplotlib")


# --------------------------------------------------------------------------- pure functions


def test_identity_rel_plus_cap_equals_total_at_every_n():
    rng = np.random.default_rng(0)
    q0 = [rng.dirichlet(np.ones(6))[:4] * rng.uniform(0.1, 1) for _ in range(5)]
    q1 = [rng.dirichlet(np.ones(6))[:4] * rng.uniform(0.1, 1) for _ in range(5)]
    total = float(np.mean([b.sum() - a.sum() for a, b in zip(q0, q1)]))
    ns = default_grid(1000)
    for c in (0.5, 1.0, 2.0):
        rel, cap = split_curve(q0, q1, ns, c)
        assert np.max(np.abs(rel + cap - total)) <= 1e-12


def test_default_grid_reaches_64_times_n_train():
    assert default_grid(None) == [2**k for k in range(21)]
    grid = default_grid(100_000)
    assert grid[0] == 1 and grid[-1] >= 6_400_000 and grid[-2] < 6_400_000
    assert all(isinstance(n, int) for n in grid)


def test_crossover_is_literal_and_needs_positive_total():
    ns = [1, 2, 4, 8]
    assert crossover(ns, [0.0, 0.1, 0.3, 0.2], 0.4) == 4
    assert crossover(ns, [0.0, 0.25, 0.1, 0.3], 0.4) == 2  # non-monotone: first crossing on the grid
    assert crossover(ns, [0.0, 0.0, 0.0, 0.1], 0.4) is None  # never reached
    assert crossover(ns, [0.0, 0.1, 0.2, 0.3], 0.0) is None
    assert crossover(ns, [0.0, -0.1, -0.2, -0.3], -0.3) is None


def test_clopper_pearson_resolvable_n():
    m = 4096
    u0 = clopper_pearson_zero_upper(m)
    assert u0 == pytest.approx(1 - 0.025 ** (1 / m))
    assert u0 == pytest.approx(-math.log(0.025) / m, rel=1e-3)  # ~ 3.69 / m
    assert resolvable_n_max(m) == math.floor(1 / u0) == 1110
    assert resolvable_n_max(m, c=2.0) == math.floor(2 / u0)
    assert resolvable_n_max(1) == 1  # u0 = 0.975
    assert resolvable_n_max(m, alpha=0.5) > resolvable_n_max(m)
    with pytest.raises(ValueError):
        clopper_pearson_zero_upper(0)


# --------------------------------------------------------------------------- planted results folder


class PlantedPolicy(Policy):
    """Solution probabilities q[pid, k] for the tiny_prompt task; the rest on one wrong row."""

    def __init__(self, q):
        self.q = torch.as_tensor(np.asarray(q, dtype=float))

    def trajectory_probs(self, task, pid):
        table = task.table(pid)
        probs = np.zeros(len(table.solution_ids))
        q = self.q[pid].numpy()
        for k in range(len(table.solutions)):
            probs[np.flatnonzero(table.solution_ids == k)[0]] = q[k]
        probs[np.flatnonzero(table.solution_ids < 0)[0]] = 1.0 - q.sum()
        return probs

    def sample(self, *a, **k):
        raise NotImplementedError

    def log_prob(self, *a, **k):
        raise NotImplementedError

    def state_logits(self, *a, **k):
        raise NotImplementedError

    def parameters(self):
        return []

    def state_dict(self):
        return {"q": self.q.clone()}

    def load_state_dict(self, state):
        self.q = state["q"].clone()


BASE_Q = [[0.5, 0.001], [0.2, 0.2], [0.3, 0.0], [0.1, 0.1]]
TRAINED_Q = [[0.6, 0.101], [0.2, 0.2], [0.3, 0.0], [0.1, 0.1]]  # only prompt 0 (easy_train) changes


def _planted_dir(tmp_path: pathlib.Path, backend: dict | None = None, log: dict | None = None) -> pathlib.Path:
    exp = tmp_path / "exp"
    task_cfg = {"type": "tiny_prompt", "params": {"vocab": 3, "length": 3}}
    cfg = {"task": task_cfg, "backend": backend or {"type": "exact", "params": {}},
           "measure": {"ns": [16], "cs": [1.0], "group_by": ["difficulty", "split"]}, "arm": {"name": "rloo"}}
    (exp / "bases" / "b0").mkdir(parents=True)
    (exp / "cell").mkdir()
    torch.save(PlantedPolicy(BASE_Q), exp / "bases" / "b0" / "policy.pt")
    torch.save({0: {"q": torch.tensor(BASE_Q, dtype=torch.float64)},
                10: {"q": torch.tensor(TRAINED_Q, dtype=torch.float64)}}, exp / "cell" / "seed-0.policies.pt")
    (exp / "cell" / "manifest.json").write_text(json.dumps({"config": cfg}))
    (exp / "cell" / "seed-0.metrics.json").write_text(json.dumps({"train_prompt_ids": [0, 1]}))
    if log is not None:
        (exp / "cell" / "seed-0.training.json").write_text(json.dumps(log))
    run = {"arm": "rloo", "trainer": "rloo", "seed": 0, "factors": {"task.length": 3}, "run_dir": "cell",
           "base": "bases/b0", "files": {"manifest": "cell/manifest.json", "metrics": "cell/seed-0.metrics.json",
                                         "policies": "cell/seed-0.policies.pt"}}
    (exp / "index.json").write_text(json.dumps({"runs": [run]}))
    return exp


def _csv(out: pathlib.Path) -> list[dict]:
    with open(out / "budget_curve.csv") as handle:
        return list(csv.DictReader(handle))


def _entry(out: pathlib.Path, group: str) -> dict:
    summary = json.loads((out / "summary.json").read_text())
    return next(e for e in summary["entries"] if e["group"] == group)


def test_planted_curve_and_n_star_with_training_log(tmp_path):
    exp = _planted_dir(tmp_path, log={"per_prompt": {"0": {"episodes_sampled": 1000}, "1": {"episodes_sampled": 2000}}})
    out = build_budget_curve(exp, out_dir=tmp_path / "bc", n_grid=[1, 2, 4, 512, 1024, 2048])
    rows = {(r["group"], int(r["N"])): r for r in _csv(out)}
    # easy_train is prompt 0: q0 = (0.5, 0.001), change = (+0.1, +0.1), total 0.2.
    expected = {1: (0.0, 0.2), 2: (0.1, 0.1), 4: (0.1, 0.1), 512: (0.1, 0.1), 1024: (0.2, 0.0), 2048: (0.2, 0.0)}
    for n, (rel, cap) in expected.items():
        row = rows[("easy_train", n)]
        assert float(row["rel"]) == pytest.approx(rel, abs=1e-12)
        assert float(row["cap"]) == pytest.approx(cap, abs=1e-12)
        assert float(row["total"]) == pytest.approx(0.2, abs=1e-12)
    # "all" averages over 4 prompts; unchanged prompts add nothing.
    assert float(rows[("all", 2)]["rel"]) == pytest.approx(0.025, abs=1e-12)
    for row in rows.values():
        assert abs(float(row["rel"]) + float(row["cap"]) - float(row["total"])) <= 1e-12

    e = _entry(out, "easy_train")
    assert e["n_star"]["per_seed"] == [2] and e["n_star"]["seed_mean"] == 2
    assert e["n_train"]["mean"] == 1500 and e["n_train"]["source"] == SOURCE_LOG
    assert (e["n_train"]["per_prompt_min"], e["n_train"]["per_prompt_max"]) == (1000, 2000)
    assert e["cap_at_n_train"]["mean"] == pytest.approx(0.0, abs=1e-12)  # 0.001 >= 1/1500: reachable
    assert e["step"] == [10]
    hard = _entry(out, "hard_train")
    assert hard["n_star"]["seed_mean"] is None and hard["total"]["mean"] == 0  # no gain: N* undefined
    assert (out / "budget_curve.md").exists() and (out / "figures" / "budget_rloo.png").exists()


def test_planted_n_train_fallback_is_marked(tmp_path):
    exp = _planted_dir(tmp_path)
    out = build_budget_curve(exp, out_dir=tmp_path / "bc", n_grid=[1, 2], n_train={"rloo": 500})
    e = _entry(out, "easy_train")
    assert e["n_train"]["source"] == SOURCE_CONFIG == "from_config (not logged)"
    assert e["cap_at_n_train"]["mean"] == pytest.approx(0.1, abs=1e-12)  # 0.001 < 1/500: unreachable
    assert "from_config (not logged)" in (out / "budget_curve.md").read_text()
    # Unknown arm: no silent assumption.
    out2 = build_budget_curve(exp, out_dir=tmp_path / "bc2", n_grid=[1, 2])
    e2 = _entry(out2, "easy_train")
    assert e2["n_train"]["source"] == SOURCE_UNKNOWN and e2["cap_at_n_train"]["mean"] is None
    # An earlier checkpoint: step 0 is the base, so nothing changes.
    out3 = build_budget_curve(exp, out_dir=tmp_path / "bc3", n_grid=[1, 2], step=0, n_train={"rloo": 500})
    assert _entry(out3, "easy_train")["total"]["mean"] == 0
    with pytest.raises(FileExistsError):
        build_budget_curve(exp, out_dir=tmp_path / "bc", n_grid=[1, 2])


def test_planted_sampled_backend_truncates_grid(tmp_path):
    exp = _planted_dir(tmp_path, backend={"type": "sampled", "params": {"m_classify": 64}})
    out = build_budget_curve(exp, out_dir=tmp_path / "bc", n_train={"rloo": 1000})
    limit = resolvable_n_max(64)
    assert limit == 17  # u0 = 1 - 0.025^(1/64) ~ 0.0560
    assert max(int(r["N"]) for r in _csv(out)) == 16
    e = _entry(out, "easy_train")
    assert e["resolvable_n_max"] == limit
    assert e["cap_at_n_train"]["mean"] is None and "resolvable" in e["cap_at_n_train"]["notes"][0]
    assert "largest resolvable N" in (out / "budget_curve.md").read_text()


# --------------------------------------------------------------------------- end to end


def test_end_to_end_on_runner_output_matches_exact_backend(tmp_path):
    config = {
        "name": "bc_e2e",
        "task": {"type": "tiny_prompt", "params": {"vocab": 3}},
        "base": {"type": "uniform_tabular", "params": {"init_scale": 0.5}},
        "backend": {"type": "exact", "params": {}},
        "factors": {"task.length": [3]},
        "arms": [{"name": "rloo", "trainer": "rloo", "params": {"group": 4, "lr": 0.05}}],
        "seeds": [0, 1],
        "budget": 6,
        "checkpoints": [0, 3, 6],
        "measure": {"ns": [4, 16], "cs": [1], "group_by": ["difficulty", "split"]},
        "save_policies": "checkpoints",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    exp = run_experiment(path, out_root=tmp_path / "results")
    out = build_budget_curve(exp, out_dir=tmp_path / "bc", n_grid=[1, 2, 4, 8, 16, 32], n_train={"rloo": 24})
    rows = _csv(out)
    for seed in (0, 1):
        metrics = json.loads((exp / json.loads((exp / "index.json").read_text())["runs"][seed]["files"]["metrics"]).read_text())
        for rec in metrics["records"]:
            if rec["step"] != 6:
                continue
            row = next(r for r in rows if int(r["seed"]) == seed and r["group"] == rec["group"] and int(r["N"]) == rec["n"])
            assert float(row["rel"]) == pytest.approx(rec["delta_rel"], abs=1e-9)
            assert float(row["cap"]) == pytest.approx(rec["delta_cap"], abs=1e-9)
    summary = json.loads((out / "summary.json").read_text())
    assert {e["group"] for e in summary["entries"]} >= {"all", "easy_train", "hard_train", "easy_heldout", "hard_heldout"}
    # The runner now writes training logs: N_train comes from them, not from the fallback mapping.
    for e in summary["entries"]:
        assert e["n_train"]["source"] == SOURCE_LOG
        assert e["n_train"]["mean"] == 6 * 4  # budget x group episodes per training prompt
    # Deterministic outputs.
    again = build_budget_curve(exp, out_dir=tmp_path / "bc_again", n_grid=[1, 2, 4, 8, 16, 32], n_train={"rloo": 24})
    for name in ("budget_curve.csv", "summary.json", "budget_curve.md", "figures/budget_rloo.png"):
        assert (out / name).read_bytes() == (again / name).read_bytes()
    assert TASKS.get("tiny_prompt")
