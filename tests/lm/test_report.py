"""Report and L1-L3 verdicts on a synthetic results folder (no model).

The folder is made by the real data / base / measure stages; only the sample files, adapters and
training logs are written by hand, so every per-problem success count (and so every class and
gain) is chosen by the test.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from capsplit.harness.lm import report as rp
from capsplit.harness.lm import stages as st
from capsplit.harness.lm import verdicts as vd
from capsplit.harness.lm.config import LMConfig, parse_config

from .conftest import DEMO_CONFIG

N_A = N_B = N_T = N_C = 16
REF = 1.0


def raw_config() -> dict:
    raw = json.loads(DEMO_CONFIG.read_text())
    raw["name"] = "synth"
    raw["model"]["device"] = "cpu"
    raw["memory"] = {"min_free_gb": 0.0, "max_process_gb": 1000.0}
    raw["runtime"] = {"nice": 0, "threads": 1}
    raw["task"]["generators"] = [{"type": "arithmetic", "params": {"digits": [2], "ops": ["+"]},
                                  "n_train": 12, "n_heldout": 8}]
    raw["sampling"] = {"n_classify": N_A, "n_measure": N_B, "n_trained": N_T, "temperatures": [0.7, 1.0],
                       "n_temperature": N_C, "reference_temperature": REF, "batch_size": 4}
    raw["measure"] = {"n": 4, "alpha": 0.05, "n_boot": 200, "boot_seed": 0, "n_splits": 4}
    raw["arms"] = [{"name": "none", "trainer": "lm_none", "params": {}},
                   {"name": "rl", "trainer": "lm_rl", "params": {"group": 2, "problems_per_update": 2}},
                   {"name": "sft", "trainer": "lm_sft", "params": {"batch": 2}}]
    raw["budget"] = 2
    raw["checkpoints"] = [0, 2]
    raw["seeds"] = [0, 1]
    return raw


def write_samples(path: Path, successes: list[int], n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for pid, k in enumerate(successes):
        for i in range(n):
            ok = i < k
            text = "42" if ok else ("7" if i % 2 else "no idea")
            lines.append(json.dumps({"problem": pid, "index": i, "text": text, "valid": ok or i % 2 == 1,
                                     "correct": ok, "tokens": 2}, sort_keys=True))
    path.write_text("\n".join(lines) + "\n")


class Synth:
    """Class plan: train ids [0:8] unreachable, [8:11] reachable, [11] ambiguous; heldout [0:4]
    unreachable, [4:7] reachable, [7] ambiguous (positions within each split)."""

    def __init__(self, out: Path, config: LMConfig):
        self.out, self.config = out, config
        problems = list(st.read_jsonl(out / "data" / "problems.jsonl"))
        self.n = len(problems)
        self.train = [int(r["id"]) for r in problems if r["split"] == "train"]
        self.heldout = [int(r["id"]) for r in problems if r["split"] == "heldout"]
        assert len(self.train) == 12 and len(self.heldout) == 8
        self.cls: dict[int, str] = {}
        for ids, n_u, n_r in ((self.train, 8, 3), (self.heldout, 4, 3)):
            for k, pid in enumerate(ids):
                self.cls[pid] = "unreachable" if k < n_u else "reachable" if k < n_u + n_r else "ambiguous"
        self.a = [{"unreachable": 0, "reachable": 12, "ambiguous": 2}[self.cls[i]] for i in range(self.n)]
        self.b = [{"unreachable": 0, "reachable": 12, "ambiguous": 2}[self.cls[i]] for i in range(self.n)]

    def ids(self, cls: str, split: str) -> list[int]:
        return [i for i in (self.train if split == "train" else self.heldout) if self.cls[i] == cls]

    def write_base(self, null_shift: int = 0) -> None:
        write_samples(st.sample_path(self.out, "base", "A", REF), self.a, N_A)
        write_samples(st.sample_path(self.out, "base", "B", REF), self.b, N_B)
        write_samples(st.sample_path(self.out, "base", "null", REF), [min(N_B, k + null_shift) for k in self.b], N_B)
        for t in (0.7, 1.0):
            write_samples(st.sample_path(self.out, "base", "C", t), self.b, N_C)

    def write_run(self, arm: str, seed: int, eval_k: dict[int, int] | None = None,
                  ever: list[int] | None = None) -> None:
        k = list(self.b)
        for pid, v in (eval_k or {}).items():
            k[pid] = v
        write_samples(st.sample_path(self.out, st.run_tag(arm, seed, 2), "eval", REF), k, N_T)
        st.adapter_path(self.out, arm, seed, 2).parent.mkdir(parents=True, exist_ok=True)
        st.adapter_path(self.out, arm, seed, 2).write_bytes(b"stub")
        d = st.run_dir(self.out, arm, seed)
        if arm == "none":
            log = {"trainer": "lm_none", "updates": 2, "prompts": {}, "totals": {"episodes_sampled": 0,
                                                                                    "tokens_sampled": 0}}
        else:
            ever = list(ever or [])
            sampled = arm == "rl"
            prompts: dict[str, dict[str, Any]] = {str(p): {"first_success_update": (1 if p in ever else None),
                                "episodes_sampled": 4 if sampled else 0, "tokens_sampled": 8 if sampled else 0,
                                "correct_by_checkpoint": {"0": 0, "2": 1 if p in ever else 0},
                                "episodes_by_checkpoint": {"0": 0, "2": 4 if sampled else 0}}
                       for p in (self.train if sampled else self.ids("unreachable", "train"))}
            log = {"trainer": "lm_" + arm, "updates": 2, "prompts": prompts,
                   "totals": {"episodes_sampled": sum(e["episodes_sampled"] for e in prompts.values()),
                              "tokens_sampled": sum(e["tokens_sampled"] for e in prompts.values())}}
        st.write_json(d / "timing.json", {"wall_seconds": 12.5 + seed, "budget": 2, "seconds_per_update": 6.25})
        st.write_json(d / "training.json", log)

    def measure(self) -> None:
        st.run_measure(self.config, self.out)


@pytest.fixture(scope="module")
def base_dir(tmp_path_factory) -> Path:
    config = parse_config(raw_config())
    out = tmp_path_factory.mktemp("base") / "synth"
    st.run_data(config, out)
    s = Synth(out, config)
    s.write_base()
    st.run_base(config, out)
    return out


def make(base_dir: Path, tmp_path: Path, null_shift: int = 0) -> Synth:
    out = tmp_path / "synth"
    shutil.copytree(base_dir, out)
    s = Synth(out, parse_config(raw_config()))
    if null_shift:
        s.write_base(null_shift)
    return s


def default_runs(s: Synth, rl_eval: dict | None = None, sft_eval: dict | None = None,
                 ever: list[int] | None = None) -> None:
    for seed in (0, 1):
        s.write_run("none", seed)
        s.write_run("rl", seed, rl_eval, ever if ever is not None else s.ids("unreachable", "train")[:2])
        s.write_run("sft", seed, sft_eval)
    s.measure()


def prereg(tmp_path: Path, **over: object) -> Path:
    data = {"tau_l1": 0.005, "l1_min_problems": 1, "tau_l2": 0.005, "tau_l3_effect": 0.0, **over}
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(data))
    return path


def verdicts(s: Synth, tmp_path: Path, **over: object) -> dict:
    path = vd.build_verdicts(s.out, prereg(tmp_path, **over))
    assert (s.out / "report" / "verdicts.md").exists()
    return json.loads(path.read_text())["verdicts"]


# --------------------------------------------------------------------------- report


def test_classes_as_planned(base_dir):
    classes = json.loads((base_dir / "base" / "classes.json").read_text())["classes"]
    s = Synth(base_dir, parse_config(raw_config()))
    assert {int(k): v for k, v in classes.items()} == s.cls


def test_report_files_and_determinism(base_dir, tmp_path):
    s = make(base_dir, tmp_path)
    default_runs(s)
    path = rp.build_report(s.out)
    rep = s.out / "report"
    for name in ("setup", "data", "base_classes", "split", "training", "first_success", "null_check",
                 "budget_curve", "format_validity"):
        assert (rep / "tables" / f"{name}.csv").stat().st_size > 0
    for name in ("contributions", "first_success", "budget_curve_all", "budget_curve_train", "budget_curve_heldout"):
        assert (rep / "figures" / f"{name}.png").stat().st_size > 0
    text = path.read_text()
    assert "NULL CHECK FAILED" not in text and "EMPTY BASE CLASS" not in text and "missing" not in text.lower()
    for section in ("## Setup", "## Base classification", "## Split of the gain", "## Format validity",
                    "## Training", "## Null check", "## Budget curve", "## What this cannot show"):
        assert section in text
    assert "largest resolvable N" in text
    summary = json.loads((rep / "summary.json").read_text())
    assert summary["null_check_passed"] is True
    assert summary["base_classes"] == {"reachable": 6, "unreachable": 12, "ambiguous": 2}
    # timings come from timing.json
    rows = {(r["arm"], r["seed"]): r for r in summary["training"]}
    assert rows[("rl", 1)]["wall_seconds"] == 13.5
    assert rows[("rl", 0)]["episodes_sampled"] == 48 and rows[("rl", 0)]["problems_ever_correct"] == 2
    # validity from sample files: base B, reachable problems valid on all 16, others on correct + odd indices
    base_all = next(r for r in summary["format_validity"] if r["tag"] == "base" and r["split"] == "all")
    assert base_all["n_samples"] == 20 * 16
    snapshot = {p: p.read_bytes() for p in sorted(rep.rglob("*")) if p.is_file()}
    rp.build_report(s.out)
    assert {p: p.read_bytes() for p in sorted(rep.rglob("*")) if p.is_file()} == snapshot


def test_null_failure_warning(base_dir, tmp_path):
    s = make(base_dir, tmp_path, null_shift=6)
    default_runs(s)
    text = rp.build_report(s.out).read_text()
    head = text.split("## Setup")[0]
    assert "## WARNINGS" in head and "NULL CHECK FAILED" in head
    assert "**FAILED**" in text
    assert json.loads((s.out / "report" / "summary.json").read_text())["null_check_passed"] is False
    vd.build_verdicts(s.out, prereg(tmp_path))
    assert "null check FAILED" in (s.out / "report" / "verdicts.md").read_text()


def test_missing_measure_raises(base_dir, tmp_path):
    out = tmp_path / "synth"
    shutil.copytree(base_dir, out)
    with pytest.raises(rp.ReportError, match="measure"):
        rp.build_report(out)


# --------------------------------------------------------------------------- L1


def test_l1_supported(base_dir, tmp_path):
    s = make(base_dir, tmp_path)
    ever = s.ids("unreachable", "train")[:2]
    default_runs(s, rl_eval={p: 8 for p in ever}, ever=ever)
    l1 = verdicts(s, tmp_path)["L1"]
    assert l1["verdict"] == "supported"
    unit = l1["units"][0]
    assert unit["n_unreachable_never_correct"] == 6 and unit["n_unreachable_ever_correct"] == 2
    assert unit["never_correct"]["hi"] == 0.0 and unit["ever_correct"]["estimate"] == pytest.approx(2 * 0.5 / 12)
    assert len(unit["per_problem"]) == 8


def test_l1_vacuous_is_inconclusive(base_dir, tmp_path):
    s = make(base_dir, tmp_path)
    ever = s.ids("unreachable", "train")[:2]
    default_runs(s, ever=ever)
    assert verdicts(s, tmp_path, l1_min_problems=7)["L1"]["verdict"] == "inconclusive"


def test_l1_not_supported(base_dir, tmp_path):
    s = make(base_dir, tmp_path)
    unreach = s.ids("unreachable", "train")
    default_runs(s, rl_eval={p: 8 for p in unreach[2:]}, ever=unreach[:2])
    l1 = verdicts(s, tmp_path)["L1"]
    assert l1["verdict"] == "not supported"
    assert l1["units"][0]["never_correct"]["lo"] > 0.005


def test_l1_inconclusive(base_dir, tmp_path):
    s = make(base_dir, tmp_path)
    unreach = s.ids("unreachable", "train")
    default_runs(s, rl_eval={unreach[-1]: 1}, ever=unreach[:2])
    l1 = verdicts(s, tmp_path)["L1"]
    assert l1["verdict"] == "inconclusive"
    nc = l1["units"][0]["never_correct"]
    assert nc["lo"] <= 0.005 < nc["hi"]


# --------------------------------------------------------------------------- L2


def test_l2_supported(base_dir, tmp_path):
    s = make(base_dir, tmp_path)
    default_runs(s)
    l2 = verdicts(s, tmp_path)["L2"]
    assert l2["verdict"] == "supported" and l2["n_heldout_unreachable"] == 4
    assert {u["arm"] for u in l2["units"]} == {"rl", "sft"}


def test_l2_not_supported(base_dir, tmp_path):
    s = make(base_dir, tmp_path)
    default_runs(s, rl_eval={p: 8 for p in s.ids("unreachable", "heldout")})
    l2 = verdicts(s, tmp_path)["L2"]
    assert l2["verdict"] == "not supported"
    rl = next(u for u in l2["units"] if u["arm"] == "rl")
    assert rl["pooled"]["lo"] > 0.005 and all(x["lo"] > 0 for x in rl["per_seed"])
    assert verdicts(s, tmp_path, l2_basis="each_seed")["L2"]["verdict"] == "not supported"


def test_l2_inconclusive(base_dir, tmp_path):
    s = make(base_dir, tmp_path)
    default_runs(s, rl_eval={s.ids("unreachable", "heldout")[0]: 1})
    assert verdicts(s, tmp_path)["L2"]["verdict"] == "inconclusive"


# --------------------------------------------------------------------------- L3


def test_l3_supported(base_dir, tmp_path):
    s = make(base_dir, tmp_path)
    default_runs(s, sft_eval={p: 4 for p in s.ids("reachable", "train") + s.ids("reachable", "heldout")})
    l3 = verdicts(s, tmp_path)["L3"]
    assert l3["verdict"] == "supported"
    assert l3["units"][0]["all"]["pooled"]["estimate"] == pytest.approx(-6 * 0.5 / 20)


def test_l3_not_supported(base_dir, tmp_path):
    s = make(base_dir, tmp_path)
    default_runs(s)
    assert verdicts(s, tmp_path)["L3"]["verdict"] == "not supported"


def test_l3_inconclusive(base_dir, tmp_path):
    s = make(base_dir, tmp_path)
    default_runs(s, sft_eval={s.ids("reachable", "train")[0]: 11})
    l3 = verdicts(s, tmp_path)["L3"]
    assert l3["verdict"] == "inconclusive"
    # a minimum effect size larger than the drop makes even a clear drop inconclusive
    s2 = make(base_dir, tmp_path / "b")
    default_runs(s2, sft_eval={p: 4 for p in s2.ids("reachable", "train") + s2.ids("reachable", "heldout")})
    assert verdicts(s2, tmp_path, tau_l3_effect=0.5)["L3"]["verdict"] == "inconclusive"


# --------------------------------------------------------------------------- prereg


def test_verdicts_deterministic(base_dir, tmp_path):
    s = make(base_dir, tmp_path)
    default_runs(s)
    path = vd.build_verdicts(s.out, prereg(tmp_path))
    first = (path.read_bytes(), (s.out / "report" / "verdicts.md").read_bytes())
    vd.build_verdicts(s.out, prereg(tmp_path))
    assert (path.read_bytes(), (s.out / "report" / "verdicts.md").read_bytes()) == first


def test_prereg_validation(tmp_path):
    with pytest.raises(vd.PreregError, match="unknown"):
        vd.load_prereg(prereg(tmp_path, tau_l4=0.1))
    bad = tmp_path / "missing.json"
    bad.write_text(json.dumps({"tau_l1": 0.0}))
    with pytest.raises(vd.PreregError, match="missing"):
        vd.load_prereg(bad)
    with pytest.raises(vd.PreregError, match="l2_basis"):
        vd.load_prereg(prereg(tmp_path, l2_basis="mean"))
    with pytest.raises(vd.PreregError, match="tau_l1"):
        vd.load_prereg(prereg(tmp_path, tau_l1="small"))


def test_shipped_prereg_loads():
    p = vd.load_prereg(Path(__file__).resolve().parents[2] / "configs" / "lm_prereg.json")
    assert set(p) == set(vd.PREREG_KEYS)


def test_prereg_unknown_arm(base_dir, tmp_path):
    s = make(base_dir, tmp_path)
    default_runs(s)
    with pytest.raises(vd.PreregError, match="not in the config"):
        vd.build_verdicts(s.out, prereg(tmp_path, l3_arms=["nope"]))


# --------------------------------------------------------------------------- post-hoc p0 sensitivity (L21)


def test_sensitivity_p0(base_dir, tmp_path):
    from capsplit.harness.lm import sensitivity as sn

    s = make(base_dir, tmp_path, null_shift=6)
    default_runs(s)
    path = sn.build_sensitivity(s.out, prereg(tmp_path))
    res = json.loads(path.read_text())
    assert res["null_check_passed"] is False
    dg = res["null_diagnostic"]
    # null = B + 6 successes on every non-saturated problem: the diagnostic sees a positive shift
    assert dg["null_minus_B"]["mean"] > 0 and dg["mean_success"]["null"] > dg["mean_success"]["B"]
    assert set(res["verdicts_by_p0_source"]["L1"]) == set(sn.SOURCES)
    # p0 from B reproduces the registered verdicts exactly
    registered = json.loads(vd.build_verdicts(s.out, prereg(tmp_path)).read_text())["verdicts"]
    for claim in ("L1", "L2", "L3"):
        assert res["verdicts_by_p0_source"][claim]["B"] == registered[claim]["verdict"]
    # base_rates is restored afterwards and the output is deterministic
    run = rp.load_run(s.out)
    assert "base_rates" not in vars(run)
    first = path.read_bytes()
    sn.build_sensitivity(s.out, prereg(tmp_path))
    assert path.read_bytes() == first
    assert "Post-hoc" in (s.out / "report" / "sensitivity_p0.md").read_text()


def test_sensitivity_identical_sets_pass(base_dir, tmp_path):
    from capsplit.harness.lm import sensitivity as sn

    s = make(base_dir, tmp_path)
    default_runs(s)
    res = sn.compute_sensitivity(rp.load_run(s.out), vd.load_prereg(prereg(tmp_path)))
    assert res["null_diagnostic"]["null_minus_B"]["mean"] == 0.0
    assert all(res["verdicts_stable"].values())


# --------------------------------------------------------------------------- discovery vs waiting time (L22)


def test_draw_index_epoch_cycle():
    from capsplit.harness.lm import discovery as dc

    # 120 problems, 8 per update: updates 1..15 are draw 1, 16..30 draw 2
    assert [dc.draw_index(u, 8, 120) for u in (1, 15, 16, 30, 31, 200)] == [1, 1, 2, 2, 3, 14]


def test_discovery_summary_known_values():
    import numpy as np

    from capsplit.harness.lm import discovery as dc

    rows = [{"p": 0.5, "p_upper": 0.6, "draws": 3, "first_success_draw": 1},
            {"p": 0.0, "p_upper": 0.01, "draws": 3, "first_success_draw": None},
            {"p": 1.0, "p_upper": 1.0, "draws": 3, "first_success_draw": 1}]
    s = dc.summarise(rows, 2, np.random.default_rng(0))
    assert s["n"] == 3 and s["observed_discovered"] == 2
    assert s["predicted_discovered"] == pytest.approx((1 - 0.5 ** 6) + 0.0 + 1.0)
    assert s["n_p_zero"] == 1 and s["discovered_p_zero"] == 0
    assert s["predicted_discovered_p_zero_at_p_upper"] == pytest.approx(1 - 0.99 ** 6)
    assert s["pit_p_positive"]["n"] == 2
    # p = 0.5, G = 2, first success at draw 1: U in [0, F(1)] = [0, 0.75]
    assert 0.0 <= s["pit_p_positive"]["mean"] <= 1.0


def test_discovery_report_on_synth(base_dir, tmp_path):
    from capsplit.harness.lm import discovery as dc

    s = make(base_dir, tmp_path)
    ever = s.ids("unreachable", "train")[:2]
    default_runs(s, rl_eval={p: 8 for p in ever}, ever=ever)
    path = dc.build_discovery(s.out)
    res = json.loads(path.read_text())
    (unit,) = res["arms"]
    assert unit["arm"] == "rl" and unit["draw_index_exact"] is True
    # the hand-written synth logs draw every problem twice in 2 updates, which the 6-update epoch cycle
    # cannot do: the consistency check must flag every problem
    assert unit["cycle_mismatches"] == {"0": 12, "1": 12}
    un = unit["by_class"]["unreachable"]
    assert un["n"] == 16 and un["observed_discovered"] == 4 and un["discovered_p_zero"] == 4
    first = path.read_bytes()
    dc.build_discovery(s.out)
    assert path.read_bytes() == first
