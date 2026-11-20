"""Power governor (ADR-0061): levels, hysteresis, dwell, staleness, charging, thermal status. Fake clocks only."""
from __future__ import annotations

import pytest

from oiris.config import Config, ConfigError
from oiris.power.governor import (CRITICAL, HOT, NORMAL, POWER_DEFAULTS, REASONS, SAVER, Governor, PowerReading,
                                  PowerSettings, thermal_status_level, validate_power)

DWELL = POWER_DEFAULTS["min_dwell_s"]
STALE = POWER_DEFAULTS["stale_after_s"]


def feed(g: Governor, t: float, **kw):
    return g.update(PowerReading(t=t, **kw), now=t)


def test_defaults_are_the_named_values():
    d = POWER_DEFAULTS
    assert (d["battery_saver_enter_pct"], d["battery_saver_exit_pct"]) == (20.0, 25.0)
    assert (d["battery_critical_enter_pct"], d["battery_critical_exit_pct"]) == (7.0, 12.0)
    assert (d["skin_hot_enter_c"], d["skin_hot_exit_c"]) == (41.0, 39.5)
    assert (d["skin_critical_enter_c"], d["skin_critical_exit_c"]) == (43.0, 41.5)
    assert (d["device_hot_enter_c"], d["device_hot_exit_c"]) == (75.0, 70.0)
    assert (d["device_critical_enter_c"], d["device_critical_exit_c"]) == (85.0, 80.0)
    assert (DWELL, STALE) == (30.0, 60.0)
    assert validate_power({}) == []


@pytest.mark.parametrize("pct,level", [(100, NORMAL), (20.01, NORMAL), (20, SAVER), (7.01, SAVER), (7, CRITICAL), (0, CRITICAL)])
def test_battery_enter(pct, level):
    assert feed(Governor(), 0.0, battery_pct=pct).level == level


@pytest.mark.parametrize("field,value,level", [
    ("skin_c", 40.99, NORMAL), ("skin_c", 41.0, HOT), ("skin_c", 42.99, HOT), ("skin_c", 43.0, CRITICAL),
    ("device_c", 74.9, NORMAL), ("device_c", 75.0, HOT), ("device_c", 85.0, CRITICAL)])
def test_heat_enter(field, value, level):
    assert feed(Governor(), 0.0, **{field: value}).level == level


def test_battery_exit_needs_the_exit_value_and_the_dwell():
    g = Governor()
    assert feed(g, 0, battery_pct=19).level == SAVER
    assert feed(g, 100, battery_pct=24.9).level == SAVER          # above enter, below exit: held (hysteresis)
    assert feed(g, 200, battery_pct=25).level == NORMAL           # at the exit value, dwell long passed
    g = Governor()
    feed(g, 0, battery_pct=19)
    assert feed(g, DWELL - 1, battery_pct=80).level == SAVER       # exit met, dwell not yet
    assert feed(g, DWELL, battery_pct=80).level == NORMAL


def test_skin_exit_values():
    g = Governor()
    feed(g, 0, skin_c=41.0)
    assert feed(g, 100, skin_c=39.6).level == HOT
    assert feed(g, 200, skin_c=39.5).level == SAVER                # one level at a time
    assert feed(g, 200 + DWELL, skin_c=39.5).level == NORMAL
    g = Governor()
    feed(g, 0, skin_c=43.0)
    assert feed(g, 100, skin_c=41.6).level == CRITICAL
    assert feed(g, 200, skin_c=41.5).level == HOT                  # critical exit met; still above hot exit


def test_tightening_is_immediate():
    g = Governor()
    feed(g, 0, battery_pct=50, skin_c=30)
    assert feed(g, 1, battery_pct=19, skin_c=30).level == SAVER
    assert feed(g, 2, battery_pct=19, skin_c=41.2).level == HOT
    assert feed(g, 3, battery_pct=19, skin_c=43.5).level == CRITICAL   # no dwell to go up
    g = Governor()
    assert feed(g, 0, skin_c=30, battery_pct=5).level == CRITICAL       # straight from normal to critical


def test_relaxing_is_one_level_at_a_time_after_the_dwell():
    g = Governor()
    assert feed(g, 0, skin_c=44, battery_pct=90).level == CRITICAL
    levels = []
    t = 1.0
    while t <= 4 * DWELL:
        levels.append((t, feed(g, t, skin_c=30, battery_pct=90).level))   # everything fine from t = 1
        t += 1.0
    first = {lv: next(t for t, x in levels if x == lv) for lv in (HOT, SAVER, NORMAL)}
    assert first[HOT] == DWELL and first[SAVER] == 2 * DWELL and first[NORMAL] == 3 * DWELL
    assert [lv for _, lv in levels if lv not in (CRITICAL, HOT, SAVER, NORMAL)] == []


def test_stale_reading_holds_the_level_and_says_so():
    g = Governor()
    feed(g, 0, battery_pct=10)
    d = g.update(PowerReading(t=10, battery_pct=90), now=10 + STALE + 1)
    assert d.level == SAVER and "stale" in d.reasons
    d = g.update(PowerReading(t=200, battery_pct=90), now=200 + STALE)   # exactly at the limit is not stale
    assert d.level == NORMAL and "stale" not in d.reasons


def test_stale_reading_can_still_tighten_but_its_signals_never_relax():
    g = Governor()
    d = g.update(PowerReading(t=0, skin_c=43.5), now=1000)
    assert d.level == CRITICAL and d.reasons == ("skin_critical", "stale")
    d = g.update(PowerReading(t=500, skin_c=30), now=2000)
    assert d.level == CRITICAL and "skin_critical" in d.reasons


def test_non_finite_t_is_stale():
    g = Governor()
    feed(g, 0, battery_pct=10)
    assert "stale" in g.update(PowerReading(t=float("nan"), battery_pct=90), now=500).reasons


def test_unknown_values_are_never_guessed():
    g = Governor()
    feed(g, 0, skin_c=42)
    d = feed(g, 500)                                   # every field None: the signals keep their state
    assert d.level == HOT and d.reasons == ("skin_hot",)
    for bad in (float("nan"), float("inf"), -1.0, 101.0, True):
        assert feed(Governor(), 0, battery_pct=bad).level == NORMAL
    assert feed(Governor(), 0, skin_c=float("nan")).level == NORMAL


def test_charging_cancels_the_battery_axis_not_heat():
    g = Governor()
    d = feed(g, 0, battery_pct=5, charging=True)
    assert d.level == NORMAL and d.reasons == ("charging",)
    d = feed(g, 1, battery_pct=5, charging=True, skin_c=41.5)
    assert d.level == HOT and d.reasons == ("charging", "skin_hot")
    g = Governor()
    feed(g, 0, battery_pct=5)
    d = feed(g, 1, battery_pct=5, charging=True)
    assert d.level == CRITICAL                          # the battery cause is gone, but relaxing waits for the dwell
    assert feed(g, 1 + DWELL, battery_pct=5, charging=True).level == HOT
    d = feed(g, 2 + 2 * DWELL, battery_pct=5, charging=False)   # unplugged: the battery axis is back at once
    assert d.level == CRITICAL and "battery_critical" in d.reasons


def test_charging_unknown_does_not_cancel():
    assert feed(Governor(), 0, battery_pct=5, charging=None).level == CRITICAL


@pytest.mark.parametrize("status,level", [(0, NORMAL), (1, NORMAL), (2, HOT), (3, CRITICAL), (4, CRITICAL),
                                          (5, CRITICAL), (6, CRITICAL), (7, None), (-1, None), (None, None),
                                          (True, None), (2.0, None), ("2", None)])
def test_thermal_status_mapping(status, level):
    assert thermal_status_level(status) == level


def test_thermal_status_feeds_the_heat_axis():
    g = Governor()
    d = feed(g, 0, thermal_status=2)
    assert d.level == HOT and d.reasons == ("thermal_status_hot",)
    d = feed(g, 1, thermal_status=3)
    assert d.level == CRITICAL and d.reasons == ("thermal_status_critical",)
    assert feed(g, 2, thermal_status=None).level == CRITICAL    # unknown: held


def test_more_restrictive_axis_wins_and_reasons_are_codes():
    d = feed(Governor(), 0, battery_pct=15, skin_c=41.5, device_c=86, thermal_status=2)
    assert d.level == CRITICAL
    assert d.reasons == ("battery_low", "skin_hot", "device_critical", "thermal_status_hot")
    assert set(d.reasons) <= set(REASONS)
    out = d.to_dict()
    assert out["level"] == CRITICAL and out["plan"]["camera_max_fps"] == 0.0


def test_decision_plan_is_the_levels_plan():
    s = PowerSettings()
    for lv, kw in ((NORMAL, {}), (SAVER, {"battery_pct": 19}), (HOT, {"skin_c": 41.2}), (CRITICAL, {"battery_pct": 1})):
        assert feed(Governor(s), 0, **kw).plan == s.plans[lv]


@pytest.mark.parametrize("bad", [
    {"battery_saver_exit_pct": 20.0},                       # exit must be above enter
    {"battery_critical_enter_pct": 21.0, "battery_critical_exit_pct": 30.0},   # critical below saver
    {"skin_hot_exit_c": 41.0},                              # exit must be below enter
    {"skin_critical_enter_c": 40.0, "skin_critical_exit_c": 39.0},
    {"device_hot_exit_c": 76.0},
    {"skin_hot_enter_c": 70.0}, {"battery_saver_enter_pct": -1}, {"min_dwell_s": -1}, {"stale_after_s": 0},
    {"min_dwell_s": True}, {"skin_hot_enter_c": float("nan")}, {"nonsense": 1}, {"plans": []},
])
def test_bad_power_config_is_a_config_error(bad):
    assert validate_power(bad)
    with pytest.raises(ConfigError):
        Config(power=bad).validate()
    with pytest.raises(ConfigError):
        PowerSettings.from_config(bad)


def test_config_without_power_or_budgets_still_validates():
    c = Config().validate()
    assert c.power == {} and c.budgets == {}
    assert PowerSettings.from_config(c.power).thresholds == POWER_DEFAULTS
    s = PowerSettings.from_config({"skin_hot_enter_c": 40.0, "skin_hot_exit_c": 38.0})
    assert feed(Governor(s), 0, skin_c=40.0).level == HOT
