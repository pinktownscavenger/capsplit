"""Memory helpers in lm/device.py: MPS hard-cap fraction, OOM explanation, config flag (CPU)."""
from __future__ import annotations

import json

import pytest
import torch

from capsplit.harness.lm.config import LMConfigError, MemoryConfig, parse_config
from capsplit.harness.lm.device import GB, MemoryGuardError, explain_oom, limit_mps_memory, mps_memory_fraction

from .conftest import DEMO_CONFIG


def test_fraction_is_cap_over_recommended_clamped():
    assert mps_memory_fraction(6.0, 12 * GB) == pytest.approx(0.5)
    assert mps_memory_fraction(8.0, 12 * GB) == pytest.approx(8 / 12)
    assert mps_memory_fraction(20.0, 12 * GB) == 1.0
    assert 0 < mps_memory_fraction(1e-6, 12 * GB) < 1e-6
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            mps_memory_fraction(bad, 12 * GB)
    with pytest.raises(ValueError):
        mps_memory_fraction(4.0, 0)


def test_limit_noop_when_disabled_or_not_mps(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.mps, "set_per_process_memory_fraction", lambda f: calls.append(f))
    assert limit_mps_memory(MemoryConfig(mps_hard_cap=False)) is None
    assert limit_mps_memory(MemoryConfig(), torch.device("cpu")) is None
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert limit_mps_memory(MemoryConfig(), "mps") is None
    assert calls == []


def test_limit_sets_fraction_when_mps(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 12 * GB)
    monkeypatch.setattr(torch.mps, "set_per_process_memory_fraction", lambda f: calls.append(f))
    assert limit_mps_memory(MemoryConfig(max_process_gb=9.0), "mps") == pytest.approx(0.75)
    assert calls == [pytest.approx(0.75)]


def test_explain_oom():
    mps = RuntimeError("MPS backend out of memory (MPS allocated: 7.9 GB, other allocations: 0.1 GB, "
                       "max allowed: 8.0 GB). Tried to allocate 350 MB on private pool.")
    err = explain_oom(mps, MemoryConfig(max_process_gb=8.0), "lm_rl update 3")
    assert isinstance(err, MemoryGuardError)
    text = str(err)
    for needle in ("lm_rl update 3", "MPS", "memory.max_process_gb = 8.0", "micro_batch", "sampling.batch_size",
                   "group", "mps_hard_cap", "Tried to allocate"):
        assert needle in text
    assert isinstance(explain_oom(RuntimeError("DefaultCPUAllocator: can't allocate memory")), MemoryGuardError)
    assert explain_oom(RuntimeError("shape mismatch")) is None
    assert explain_oom(ValueError("out of memory")) is None
    assert explain_oom(MemoryGuardError("out of memory")) is None


def test_config_flag():
    raw = json.loads(DEMO_CONFIG.read_text())
    assert parse_config(raw).memory.mps_hard_cap is True
    raw["memory"]["mps_hard_cap"] = False
    assert parse_config(raw).memory.mps_hard_cap is False
    raw["memory"]["mps_hard_cap"] = "no"
    with pytest.raises(LMConfigError, match="mps_hard_cap"):
        parse_config(raw)
    raw["memory"] = {"max_process_gb": 0}
    with pytest.raises(LMConfigError, match="max_process_gb"):
        parse_config(raw)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS only")
def test_hard_cap_on_real_mps():
    """The cap makes an over-large MPS allocation raise instead of swapping (restores no limit after)."""
    try:
        fraction = limit_mps_memory(MemoryConfig(max_process_gb=0.25), "mps")
        assert fraction is not None and 0 < fraction < 1
        with pytest.raises(RuntimeError) as info:
            torch.empty(int(0.5 * GB), dtype=torch.uint8, device="mps")
        assert explain_oom(info.value) is not None
    finally:
        torch.mps.set_per_process_memory_fraction(0.0)  # 0 = no limit
        torch.mps.empty_cache()
