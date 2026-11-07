import json
import platform
import subprocess

import numpy as np

from capsplit.runlog import load_seeds, run_dir, save_seed


def _git(repo, *args):
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _repo_with_one_commit(path):
    _git(path, "init")
    (path / "tracked.txt").write_text("one")
    _git(path, "add", "tracked.txt")
    _git(path, "commit", "-m", "first")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    )
    return head.stdout.strip()


def _manifest(directory):
    return json.loads((directory / "manifest.json").read_text())


def test_same_config_same_dir(tmp_path):
    first = run_dir("exp", {"a": 1, "b": 2}, root=tmp_path)
    second = run_dir("exp", {"b": 2, "a": 1}, root=tmp_path)
    assert first == second


def test_directory_layout(tmp_path):
    directory = run_dir("exp", {"a": 1}, root=tmp_path)
    assert directory.is_dir()
    assert directory.parent == tmp_path / "exp"
    assert len(directory.name) == 12


def test_different_config_different_dir(tmp_path):
    assert run_dir("exp", {"a": 1}, root=tmp_path) != run_dir("exp", {"a": 2}, root=tmp_path)


def test_manifest_fields(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    head = _repo_with_one_commit(repo)
    monkeypatch.chdir(repo)

    manifest = _manifest(run_dir("exp", {"a": 1}, root=tmp_path / "out"))

    assert manifest["config"] == {"a": 1}
    assert manifest["commit"] == head
    assert manifest["dirty"] is False
    assert manifest["python"] == platform.python_version()
    assert manifest["numpy"] == np.__version__


def test_manifest_flags_dirty_tree(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _repo_with_one_commit(repo)
    (repo / "tracked.txt").write_text("two")
    monkeypatch.chdir(repo)

    assert _manifest(run_dir("exp", {"a": 1}, root=tmp_path / "out"))["dirty"] is True


def test_manifest_written_once(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _repo_with_one_commit(repo)
    monkeypatch.chdir(repo)
    directory = run_dir("exp", {"a": 1}, root=tmp_path / "out")

    (repo / "tracked.txt").write_text("two")
    run_dir("exp", {"a": 1}, root=tmp_path / "out")

    assert _manifest(directory)["dirty"] is False


def test_save_load_roundtrip(tmp_path):
    directory = run_dir("exp", {"a": 1}, root=tmp_path)
    assert save_seed(directory, 10, curve=np.array([1.0, 2.0])) == directory / "seed-10.npz"
    save_seed(directory, 2, curve=np.array([3.0]))

    loaded = load_seeds(directory)

    assert [record["seed"] for record in loaded] == [2, 10]  # numeric, not "10" before "2"
    np.testing.assert_array_equal(loaded[0]["curve"], [3.0])
    np.testing.assert_array_equal(loaded[1]["curve"], [1.0, 2.0])


def test_works_outside_git(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = _manifest(run_dir("exp", {"a": 1}, root=tmp_path / "out"))
    assert manifest["commit"] is None
