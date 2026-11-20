"""Power simulation (closed loop, scenes) and the Mac pmset reader (fixtures only, no subprocess)."""
from __future__ import annotations

from oiris.adapters import power_sim
from oiris.adapters.power_macos import MacBattery, parse_pmset
from oiris.power.duty import CRITICAL, HOT, NORMAL, SAVER, build_plans
from oiris.power.governor import POWER_DEFAULTS, PowerReading

MAX_LEVEL_CHANGES_HOT_DAY = 2      # in 30 simulated minutes: normal -> hot, at most one more (no flapping)
THIRTY_MIN_S = 1800.0


def _first(rows, level):
    return next((r["t"] for r in rows if r["level"] == level), None)


def test_hot_day_reaches_hot_and_settles():
    rows = power_sim.run("hot_day", THIRTY_MIN_S)
    assert rows[-1]["t"] == THIRTY_MIN_S
    assert _first(rows, HOT) is not None
    assert power_sim.level_changes(rows) <= MAX_LEVEL_CHANGES_HOT_DAY
    assert rows[-1]["level"] == HOT and _first(rows, CRITICAL) is None
    # Throttling really cools: the skin stays under the critical entry and ends between hot's exit and entry.
    assert max(r["skin_c"] for r in rows) < POWER_DEFAULTS["skin_critical_enter_c"]
    assert POWER_DEFAULTS["skin_hot_exit_c"] < rows[-1]["skin_c"] < POWER_DEFAULTS["skin_hot_enter_c"]


def test_hot_day_without_throttling_would_go_critical():
    # The same device held at the normal plan heats past the critical entry: the governor is what prevents it.
    sim = power_sim.PowerSim("hot_day")
    normal = build_plans(None)[NORMAL]
    for _ in range(int(THIRTY_MIN_S)):
        sim.advance(1.0, normal)
    assert sim.skin_c >= POWER_DEFAULTS["skin_critical_enter_c"]


def test_cool_room_stays_normal():
    rows = power_sim.run("cool_room", THIRTY_MIN_S)
    assert {r["level"] for r in rows} == {NORMAL} and power_sim.level_changes(rows) == 0


def test_low_battery_passes_saver_then_critical():
    rows = power_sim.run("low_battery", THIRTY_MIN_S)
    t_saver, t_crit = _first(rows, SAVER), _first(rows, CRITICAL)
    assert t_saver is not None and t_crit is not None and t_saver < t_crit
    assert [b["level"] for a, b in zip(rows, rows[1:]) if a["level"] != b["level"]] == [SAVER, CRITICAL]
    assert rows[-1]["level"] == CRITICAL and "battery_critical" in rows[-1]["reasons"]
    # Each throttle lowers the draw.
    p = {r["level"]: r["power_w"] for r in rows if r["t"] > 0}
    assert p[NORMAL] > p[SAVER] > p[CRITICAL]


def test_charging_hot_cancels_the_battery_axis_only():
    rows = power_sim.run("charging_hot", THIRTY_MIN_S)
    assert rows[0]["level"] == NORMAL and rows[0]["battery_pct"] < POWER_DEFAULTS["battery_saver_enter_pct"]
    assert all("charging" in r["reasons"] for r in rows)
    assert not any(c in r["reasons"] for r in rows for c in ("battery_low", "battery_critical"))
    assert _first(rows, SAVER) is None and _first(rows, HOT) is not None and rows[-1]["level"] == HOT


def test_two_runs_are_identical():
    for scene in power_sim.SCENES:
        assert power_sim.run(scene, THIRTY_MIN_S) == power_sim.run(scene, THIRTY_MIN_S)


def test_sim_adapter_shape():
    a = power_sim.make("hot_day", period_s=0)
    s1, s2 = a.read(1.0), a.read(1.0)
    assert s1 is not None and s2 is not None
    assert s1.data["t_s"] == power_sim.READING_EVERY_S and s2.data["t_s"] == 2 * power_sim.READING_EVERY_S
    assert s2.data["device_c"] > s1.data["device_c"] and s1.data["charging"] is False
    assert s1.data["thermal_status"] is None and "power" in a.caps
    crit = build_plans(None)[CRITICAL]
    b = power_sim.make("hot_day", period_s=0, plan_source=lambda: crit)
    assert b.read(1.0).data["power_w"] < s1.data["power_w"]


CHARGING = """Now drawing from 'AC Power'
 -InternalBattery-0 (id=4653155)\t87%; charging; 0:42 remaining present: true
"""
DISCHARGING = """Now drawing from 'Battery Power'
 -InternalBattery-0 (id=4653155)\t64%; discharging; 4:12 remaining present: true
"""
CHARGED = """Now drawing from 'AC Power'
 -InternalBattery-0 (id=4653155)\t100%; charged; 0:00 remaining present: true
"""
NOT_CHARGING = """Now drawing from 'AC Power'
 -InternalBattery-0 (id=4653155)\t80%; AC attached; not charging present: true
"""
DESKTOP = "Now drawing from 'AC Power'\n"


def test_parse_pmset():
    assert parse_pmset(CHARGING) == (87.0, True)
    assert parse_pmset(DISCHARGING) == (64.0, False)
    assert parse_pmset(CHARGED) == (100.0, True)
    assert parse_pmset(NOT_CHARGING) == (80.0, True)
    assert parse_pmset(DESKTOP) == (None, None)


def test_parse_pmset_garbage():
    for junk in ("", "garbage\n\x00\xff", " -InternalBattery-0 (id=1)\t250%; charging;", None, 42, b"87%; charging;",
                 "x" * 100000):
        assert parse_pmset(junk) == (None, None)


def test_mac_battery_reads_through_the_injected_runner():
    r = MacBattery(runner=lambda: DISCHARGING, clock=lambda: 12.5).read()
    assert r == PowerReading(t=12.5, battery_pct=64.0, charging=False)
    assert r.skin_c is None and r.device_c is None and r.thermal_status is None   # the Mac gives no temperature
    assert MacBattery(runner=lambda: DESKTOP, clock=lambda: 1.0).read() == PowerReading(t=1.0)


def test_mac_battery_never_raises():
    def boom() -> str:
        raise FileNotFoundError("pmset")
    assert MacBattery(runner=boom, clock=lambda: 3.0).read() == PowerReading(t=3.0)

    def bad_clock() -> float:
        raise RuntimeError("clock")
    r = MacBattery(runner=lambda: CHARGING, clock=bad_clock).read()
    assert r.battery_pct is None and r.t != r.t   # NaN t: the governor treats it as stale
