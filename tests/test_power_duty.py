"""Duty plans (ADR-0061): defaults, order rule, ranges, the critical minimum; FrameGate and ListenDuty."""
from __future__ import annotations

import pytest

from oiris.config import Config, ConfigError
from oiris.power.duty import (CRITICAL, DEFAULT_PLANS, HOT, LEVELS, NORMAL, PLAN_KEYS, SAVER, FrameGate, ListenDuty,
                              build_plans, validate_plans)


def test_default_plans_are_valid_and_complete():
    assert validate_plans({}) == [] and validate_plans(DEFAULT_PLANS) == []
    assert list(DEFAULT_PLANS) == list(LEVELS)
    for p in DEFAULT_PLANS.values():
        assert set(p) == set(PLAN_KEYS)
    plans = build_plans(None)
    crit = plans[CRITICAL]
    assert crit.camera_max_fps == 0 and not crit.voice_match and not crit.face_match and not crit.ir_illuminator
    assert crit.radio_scan_interval_s > plans[NORMAL].radio_scan_interval_s   # radio stays on, slower


@pytest.mark.parametrize("level,key,value", [
    (SAVER, "camera_max_fps", 31.0),            # higher fps than normal
    (HOT, "frame_stride", 1),                   # lower stride than saver
    (HOT, "listen_duty", 0.6),                  # higher duty than saver
    (CRITICAL, "radio_scan_interval_s", 20.0),  # shorter scan interval than hot
    (CRITICAL, "display_brightness_cap", 0.9),  # brighter than hot
])
def test_out_of_order_plan_is_a_config_error(level, key, value):
    bad = {level: {key: value}}
    errs = validate_plans(bad)
    assert errs and "restrictive" in errs[0]
    with pytest.raises(ConfigError):
        Config(power={"plans": bad}).validate()


def test_a_feature_off_stays_off():
    errs = validate_plans({SAVER: {"face_match": False}, HOT: {"face_match": True}})
    assert errs == ["power.plans.hot.face_match is off in saver, so it must stay off"]


@pytest.mark.parametrize("key,value", [
    ("camera_max_fps", -1.0), ("camera_max_fps", 61.0), ("camera_max_fps", float("nan")),
    ("frame_stride", 0), ("frame_stride", 1.5), ("frame_stride", 101),
    ("listen_duty", -0.1), ("listen_duty", 1.1), ("listen_period_s", 0.5), ("listen_period_s", 601.0),
    ("radio_scan_interval_s", 0.0), ("radio_scan_interval_s", 601.0),
    ("display_brightness_cap", 1.5), ("display_brightness_cap", -0.1),
    ("ir_illuminator", 1), ("voice_match", "yes"), ("face_match", None), ("camera_max_fps", True),
])
def test_each_range_error(key, value):
    errs = validate_plans({NORMAL: {key: value}})
    assert any(f"power.plans.normal.{key}" in e for e in errs)
    with pytest.raises(ConfigError):
        Config(power={"plans": {NORMAL: {key: value}}}).validate()


@pytest.mark.parametrize("change", [{"camera_max_fps": 5.0}, {"voice_match": True}, {"ir_illuminator": True},
                                    {"face_match": True}])
def test_critical_minimum(change):
    # Make every earlier level permissive so only the critical rule can fail.
    loose = {lv: {"camera_max_fps": 30.0, "voice_match": True, "ir_illuminator": True, "face_match": True}
             for lv in (NORMAL, SAVER, HOT)}
    errs = validate_plans({**loose, CRITICAL: change})
    assert any(e.startswith("power.plans.critical.") and ("must be 0" in e or "must be false" in e) for e in errs)


def test_unknown_levels_and_keys():
    assert validate_plans({"turbo": {}})
    assert validate_plans({NORMAL: {"warp": 1}})
    assert validate_plans({NORMAL: []})
    assert validate_plans("plans")


def test_frame_gate_rate_and_stride_exact():
    # A 30 fps camera, frames at k/30 s; cap 10 fps, every 2nd admitted frame processed.
    g = FrameGate(10.0, 2)
    admitted = [k for k in range(31) if g.admit(k / 30)]
    assert admitted == [0, 6, 12, 18, 24, 30]          # rate-capped at 0, 3, 6, ... then every second of those
    g = FrameGate(30.0, 1)
    assert all(g.admit(k / 30) for k in range(60))       # cap at the camera rate passes every frame
    g = FrameGate(0.0, 1)
    assert not any(g.admit(k / 30) for k in range(60))   # 0 = camera processing off
    g = FrameGate(1.0, 3)
    assert [t for t in range(10) if g.admit(float(t))] == [0, 3, 6, 9]


def test_frame_gate_time_backwards_restarts():
    g = FrameGate(1.0, 1)
    assert g.admit(10.0) and not g.admit(10.5)
    assert g.admit(2.0) and not g.admit(2.5) and g.admit(3.0)


def test_frame_gate_rejects_bad_arguments():
    for args in ((-1.0, 1), (float("nan"), 1), (10.0, 0), (10.0, 1.5), (10.0, True)):
        with pytest.raises(ValueError):
            FrameGate(*args)


def test_listen_duty_exact():
    d = ListenDuty(0.25, 10.0)
    assert [t for t in range(30) if d.active(float(t))] == [0, 1, 2, 10, 11, 12, 20, 21, 22]
    assert d.active(2.49) and not d.active(2.5)
    assert all(ListenDuty(1.0, 10.0).active(t / 7) for t in range(100))
    assert not any(ListenDuty(0.0, 10.0).active(t / 7) for t in range(100))
    half = ListenDuty(0.5, 4.0)
    assert [half.active(t / 2) for t in range(10)] == [True, True, True, True, False, False, False, False, True, True]


def test_listen_duty_rejects_bad_arguments():
    for args in ((1.5, 10.0), (-0.1, 10.0), (0.5, 0.0), (0.5, float("inf"))):
        with pytest.raises(ValueError):
            ListenDuty(*args)
