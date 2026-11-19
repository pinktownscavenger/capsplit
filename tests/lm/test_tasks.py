"""Problem generators, build_problems, the checker and LMTask (no model needed)."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from capsplit.harness.lm.config import GeneratorConfig, TaskConfig
from capsplit.harness.lm.tasks import (
    LMTask,
    build_problems,
    check,
    first_integer,
    get_generator,
    load_task,
)
from capsplit.harness.registry import TASKS


def _config(n_train=30, n_heldout=18, seed=0, digits=(1, 2, 3), ops=("+", "-", "*"), **extra) -> TaskConfig:
    params = {"digits": list(digits), "ops": list(ops), **extra}
    return TaskConfig(generators=(GeneratorConfig("arithmetic", params, n_train, n_heldout),), seed=seed)


def test_generation_is_deterministic_and_seed_dependent():
    assert build_problems(_config()) == build_problems(_config())
    assert build_problems(_config()) != build_problems(_config(seed=1))


def test_rows_follow_the_contract():
    rows = build_problems(_config())
    assert [r["id"] for r in rows] == list(range(48))
    for r in rows:
        assert set(r) == {"id", "generator", "question", "answer", "split", "difficulty"}
        assert r["generator"] == "arithmetic" and r["split"] in ("train", "heldout")
    assert Counter(r["split"] for r in rows) == {"train": 30, "heldout": 18}


def test_exact_digit_counts_and_correct_answers():
    rows = build_problems(_config(n_train=120, n_heldout=60, digits=(1, 2, 4, 7)))
    for r in rows:
        a, op, b = r["question"].split(" ")
        d = int(r["difficulty"].split("_")[0][1:])
        assert len(a) == len(b) == d and not a.startswith("0") and not b.startswith("0")
        assert {"add": "+", "sub": "-", "mul": "*"}[r["difficulty"].split("_")[1]] == op
        assert r["answer"] == str(eval(r["question"]))
        if op == "-":
            assert int(r["answer"]) >= 0


def test_negative_subtraction_allowed():
    rows = build_problems(_config(n_train=200, n_heldout=0, digits=(2,), ops=("-",), negative=True))
    assert any(int(r["answer"]) < 0 for r in rows)


def test_grid_cells_are_filled_evenly():
    counts = Counter(r["difficulty"] for r in build_problems(_config(n_train=30, n_heldout=15)))
    assert len(counts) == 9 and max(counts.values()) - min(counts.values()) <= 1


def test_split_proportions_per_difficulty():
    rows = build_problems(_config(n_train=80, n_heldout=40, digits=(2, 3), ops=("+", "*")))
    for difficulty in {r["difficulty"] for r in rows}:
        cell = Counter(r["split"] for r in rows if r["difficulty"] == difficulty)
        assert cell == {"train": 20, "heldout": 10}  # 120 problems / 4 cells, 2:1 split each


def test_split_totals_exact_when_cells_do_not_divide():
    rows = build_problems(_config(n_train=7, n_heldout=6, digits=(2, 3), ops=("+", "*")))
    assert Counter(r["split"] for r in rows) == {"train": 7, "heldout": 6}
    for difficulty in {r["difficulty"] for r in rows}:
        cell = [r["split"] for r in rows if r["difficulty"] == difficulty]
        assert abs(cell.count("train") - len(cell) * 7 / 13) < 1


def test_no_duplicate_questions_across_splits_or_generators():
    # 1-digit addition has only 45 distinct problems once "a + b" and "b + a" count as one,
    # so collisions are certain without the de-duplication.
    gens = (
        GeneratorConfig("arithmetic", {"digits": [1], "ops": ["+"]}, 20, 10),
        GeneratorConfig("arithmetic", {"digits": [1], "ops": ["+"]}, 5, 5),
    )
    rows = build_problems(TaskConfig(generators=gens, seed=3))
    questions = [r["question"] for r in rows]
    assert len(questions) == len(set(questions)) == 40
    assert not {r["question"] for r in rows if r["split"] == "train"} & {
        r["question"] for r in rows if r["split"] == "heldout"
    }


def test_exhausted_cell_raises():
    with pytest.raises(ValueError, match="cannot find a new distinct problem"):
        build_problems(_config(n_train=82, n_heldout=0, digits=(1,), ops=("+",)))


@pytest.mark.parametrize(
    "params",
    [
        {"digits": [2], "ops": ["+"], "bogus": 1},
        {"digits": [2]},
        {"digits": [], "ops": ["+"]},
        {"digits": [0], "ops": ["+"]},
        {"digits": [2], "ops": ["/"]},
        {"digits": [2], "ops": ["+"], "negative": "yes"},
    ],
)
def test_bad_params_raise(params):
    with pytest.raises((ValueError, TypeError)):
        get_generator("arithmetic").generate(np.random.default_rng(0), 3, params)


def test_unknown_generator_raises():
    with pytest.raises(KeyError, match="unknown problem generator"):
        build_problems(TaskConfig(generators=(GeneratorConfig("nope", {}, 1, 1),)))


@pytest.mark.parametrize(
    "text, value",
    [
        ("10063", 10063),
        ("The answer is 5", 5),
        ("5 apples", 5),
        ("1,234", 1234),
        ("-1,234", -1234),
        ("−7", -7),
        ("Sure!\n42", 42),
        ("3 then 4", 3),
        ("1,2345", 1),
        ("12,34", 12),
        ("5.", 5),
        ("5.0", 5),
        ("2.5", None),
        ("five", None),
        ("", None),
    ],
)
def test_first_integer(text, value):
    assert first_integer(text) == value


def test_check():
    assert check("The answer is -1,234.", "-1234") == (True, True)
    assert check("1233", "1234") == (True, False)
    assert check("no idea", "1234") == (False, False)
    assert check("347 * 29 = 10063", "10063") == (True, False)  # first integer wins


def test_lm_task(tmp_path):
    rows = build_problems(_config(n_train=4, n_heldout=2))
    path = tmp_path / "problems.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    task = load_task(path, "Compute {question}.")
    assert TASKS.get("lm_arithmetic") is LMTask
    assert task.n_actions == 0 and task.horizon == 0
    prompts = task.prompts()
    assert [p.id for p in prompts] == list(range(6))
    assert prompts[0].tokens == (f"Compute {rows[0]['question']}.",)
    assert dict(prompts[0].tags) == {k: rows[0][k] for k in ("split", "generator", "difficulty")}
    assert len(task.prompt_ids(split="train")) == 4
    answer = rows[1]["answer"]
    assert task.verify(1, f"It is {answer}") and not task.verify(1, "It is 999999999")
    assert task.parse(1, "nothing") == (False, False)
    assert task.demonstrations(1) == (answer,)
    with pytest.raises(TypeError):
        task.verify(1, [1, 2])
    with pytest.raises(NotImplementedError):
        task.initial_state(0)
    with pytest.raises(NotImplementedError):
        task.step(0, None, 0)
    with pytest.raises(ValueError, match="ids"):
        LMTask(rows[1:], "{question}")


def test_gsm8k(tmp_path):
    path = tmp_path / "gsm8k.jsonl"
    data = [
        {"question": "Ann has 3 apples and buys 4. How many?", "answer": "3+4=7\n#### 7"},
        {"question": "A farm sells 1,000 eggs twice. How many?", "answer": "2*1000\n#### 2,000"},
        {"question": "Bob owes 5 and pays 8. Balance?", "answer": "#### -3"},
    ]
    path.write_text("".join(json.dumps(d) + "\n" for d in data))
    config = TaskConfig(generators=(GeneratorConfig("gsm8k", {"path": str(path)}, 2, 1),), seed=0)
    rows = build_problems(config)
    assert {r["answer"] for r in rows} == {"7", "2000", "-3"}
    assert all(r["difficulty"] == "gsm8k" and r["possibly_contaminated"] is True for r in rows)
    assert Counter(r["split"] for r in rows) == {"train": 2, "heldout": 1}
    with pytest.raises(ValueError, match="only 3"):
        build_problems(TaskConfig(generators=(GeneratorConfig("gsm8k", {"path": str(path)}, 4, 0),)))


def test_gsm8k_missing_file_and_params():
    gen = get_generator("gsm8k")
    with pytest.raises(FileNotFoundError, match="nothing is downloaded"):
        gen.generate(np.random.default_rng(0), 1, {"path": "/nonexistent/gsm8k.jsonl"})
    with pytest.raises(ValueError, match="missing required"):
        gen.generate(np.random.default_rng(0), 1, {})


def test_commutative_twins_never_appear_twice():
    """Regression (review): "3 + 4" and "4 + 3" are one problem and must not be split across sets."""
    from capsplit.harness.lm.config import GeneratorConfig, TaskConfig
    from capsplit.harness.lm.tasks import build_problems

    config = TaskConfig(generators=(GeneratorConfig("arithmetic", {"digits": [1], "ops": ["+", "*"]}, 30, 30),), seed=3)
    keys = set()
    for row in build_problems(config):
        a, op, b = row["question"].split()
        key = (op, min(a, b), max(a, b)) if op in "+*" else (op, a, b)
        assert key not in keys, row["question"]
        keys.add(key)
