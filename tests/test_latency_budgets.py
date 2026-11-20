"""Latency budgets (ADR-0061): over-budget reporting, sample minimum, key validation, parts in sync with live."""
from __future__ import annotations

import pytest

from oiris import live
from oiris.config import Config, ConfigError
from oiris.power.budgets import (BUDGET_MAX_MS, DEFAULT_BUDGETS_MS, TIMING_PARTS, BudgetMonitor, split_key,
                                 validate_budgets)
from oiris.timing import StageTimer


def entry(count: int, p95: float) -> dict:
    return {"count": count, "p50_ms": p95 / 2, "p95_ms": p95, "max_ms": p95 * 2}


def test_parts_equal_live_timing_parts():
    assert TIMING_PARTS == live.TIMING_PARTS


def test_default_budgets():
    assert DEFAULT_BUDGETS_MS == {"engine.step": 50.0, "voice.queue_wait": 1000.0, "voice.decision": 1000.0,
                                  "output.chime_wait": 1500.0, "output.speech_wait": 1500.0}
    assert validate_budgets(DEFAULT_BUDGETS_MS) == []
    assert DEFAULT_BUDGETS_MS["engine.step"] < 1000.0 / live.MAX_HZ   # well inside the shortest live step


def test_over_budget_is_reported():
    snap = {"engine": {"step": entry(25, 80.0), "collect": entry(25, 999.0)},
            "output": {"speech_wait": entry(40, 1500.5)}}
    out = BudgetMonitor().check(snap)
    assert out == [{"part": "engine", "stage": "step", "p95_ms": 80.0, "budget_ms": 50.0, "count": 25},
                   {"part": "output", "stage": "speech_wait", "p95_ms": 1500.5, "budget_ms": 1500.0, "count": 40}]
    for row in out:   # numbers and code names only
        assert set(row) == {"part", "stage", "p95_ms", "budget_ms", "count"}


def test_under_budget_or_too_few_samples_is_not():
    m = BudgetMonitor(min_samples=20)
    assert m.check({"engine": {"step": entry(100, 50.0)}}) == []    # at the budget is not over
    assert m.check({"engine": {"step": entry(19, 500.0)}}) == []    # too few samples
    assert m.check({"engine": {"step": entry(20, 500.0)}}) != []
    assert m.check({}) == [] and m.check(None) == [] and m.check({"engine": {}}) == [] and m.check({"engine": 5}) == []
    assert m.check({"engine": {"step": {"count": "x", "p95_ms": 900}}}) == []


def test_configured_budgets_override_and_extend():
    m = BudgetMonitor({"engine.step": 200.0, "engine.guard": 1.0}, min_samples=1)
    snap = {"engine": {"step": entry(5, 150.0), "guard": entry(5, 2.0)}}
    assert [(r["stage"], r["budget_ms"]) for r in m.check(snap)] == [("guard", 1.0)]


def test_with_a_real_stage_timer():
    now = [0.0]
    t = StageTimer(clock=lambda: now[0])
    for _ in range(30):
        t.record("step", 0.120)
    out = BudgetMonitor().check({"engine": t.summary(), "voice": {}, "output": {}})
    assert out == [{"part": "engine", "stage": "step", "p95_ms": 120.0, "budget_ms": 50.0, "count": 30}]


@pytest.mark.parametrize("key", ["step", "camera.step", "engine.Step", "engine.", ".step", "engine.step.x",
                                 "engine.1step", 7, "engine.step name"])
def test_bad_keys(key):
    assert split_key(key) is None
    assert validate_budgets({key: 10.0})
    with pytest.raises(ConfigError):
        Config(budgets={key: 10.0}).validate()


@pytest.mark.parametrize("value", [0, -1.0, BUDGET_MAX_MS + 1, float("nan"), float("inf"), True, "50"])
def test_bad_values(value):
    assert validate_budgets({"engine.step": value})
    with pytest.raises(ConfigError):
        Config(budgets={"engine.step": value}).validate()
    with pytest.raises(ValueError):
        BudgetMonitor({"engine.step": value})


def test_good_keys_and_limits():
    assert validate_budgets({"engine.step": BUDGET_MAX_MS, "voice.embed": 0.5, "output.chime_play": 3000}) == []
    assert validate_budgets([]) == ["budgets must be an object"]
    with pytest.raises(ValueError):
        BudgetMonitor(min_samples=0)
