"""Per-adapter max age (AdapterBase.max_age_s) in the engine's Forge F1 staleness and cache rules."""
from __future__ import annotations
import time

import pytest

from oiris.adapters.base import AdapterBase
from oiris.config import Config
from oiris.engine import Engine
from oiris.schema import Health, Sample, Score

S = 1_000_000_000
GLOBAL_S = 5.0


class _Adapter(AdapterBase):
    bus = "test"

    def __init__(self, aid: str, cap: str, max_age_s=None):
        super().__init__(aid)
        self.caps = {cap}
        if max_age_s is not None:
            self.max_age_s = max_age_s


class _Mod:
    module_flag = None

    def __init__(self, name: str, inputs: set):
        self.name, self.inputs, self.windows = name, inputs, []

    def update(self, window):
        self.windows.append(list(window))
        return Score(self.name, 1.0 if window else 0.0, 1.0 if window else 0.0)


@pytest.fixture
def clock(monkeypatch):
    now = [10 * S]
    monkeypatch.setattr("oiris.engine.time.monotonic_ns", lambda: now[0])
    return now


def _engine(*adapters_mods):
    c = Config(); c.sensor_max_age_s = GLOBAL_S
    e = Engine(c, score_modules=[m for _a, m in adapters_mods])
    for a, _m in adapters_mods:
        e.add_adapter(a)
    return e


def _push(e, aid, now):
    e.bus.queues[aid].append(Sample(aid, now[0], time.time(), 0, "u", 1.0, {"v": 1}))


def _health(res, aid):
    return res["health"][aid]["health"]


def test_slow_adapter_ok_and_scored_within_its_own_age(clock):
    a, m = _Adapter("slow", "slow_cap", max_age_s=35.0), _Mod("S_slow", {"slow_cap"})
    e = _engine((a, m))
    _push(e, "slow", clock)
    e.step()
    for dt in (6, 15, 34):                      # past the global 5 s, within the adapter's 35 s
        clock[0] = 10 * S + dt * S
        res = e.step()
        assert _health(res, "slow") == Health.OK.value, dt
        assert len(m.windows[-1]) == 1, f"cached reading not scored at {dt}s"


def test_slow_adapter_degraded_and_not_scored_past_its_own_age(clock):
    a, m = _Adapter("slow", "slow_cap", max_age_s=35.0), _Mod("S_slow", {"slow_cap"})
    e = _engine((a, m))
    _push(e, "slow", clock)
    e.step()
    clock[0] += 36 * S
    res = e.step()
    assert _health(res, "slow") == Health.DEGRADED.value
    assert "stale: no fresh reading for 36.0s" in res["health"]["slow"]["reason"]
    assert m.windows[-1] == [], "stale reading was scored"
    assert next(s for s in res["scores"] if s["module"] == "S_slow")["value"] == 0.0
    clock[0] += 1 * S                           # dropped from the cache: never comes back
    e.step()
    assert m.windows[-1] == [] and "S_slow" not in e._windows
    _push(e, "slow", clock)                     # a fresh reading restores it
    res = e.step()
    assert _health(res, "slow") == Health.OK.value and len(m.windows[-1]) == 1


def test_adapter_without_max_age_keeps_global_behaviour(clock):
    slow, ms = _Adapter("slow", "slow_cap", max_age_s=35.0), _Mod("S_slow", {"slow_cap"})
    fast, mf = _Adapter("fast", "fast_cap"), _Mod("S_fast", {"fast_cap"})
    assert fast.max_age_s is None
    e = _engine((slow, ms), (fast, mf))
    _push(e, "slow", clock); _push(e, "fast", clock)
    e.step()
    clock[0] += 5 * S                           # exactly the global age: still current
    res = e.step()
    assert _health(res, "fast") == Health.OK.value and len(mf.windows[-1]) == 1
    clock[0] += S // 10                         # 5.1 s: stale for the plain adapter only
    res = e.step()
    assert _health(res, "fast") == Health.DEGRADED.value and mf.windows[-1] == []
    assert _health(res, "slow") == Health.OK.value and len(ms.windows[-1]) == 1


@pytest.mark.parametrize("bad", [1.0, 0, -10.0, float("nan"), float("inf"), True, "60"])
def test_max_age_never_below_global_and_bad_values_ignored(clock, bad):
    a, m = _Adapter("odd", "odd_cap"), _Mod("S_odd", {"odd_cap"})
    a.max_age_s = bad
    e = _engine((a, m))
    _push(e, "odd", clock)
    e.step()
    clock[0] += 4 * S
    assert _health(e.step(), "odd") == Health.OK.value and len(m.windows[-1]) == 1
    clock[0] += 2 * S                           # 6 s: past the global age whatever the adapter says
    res = e.step()
    assert _health(res, "odd") == Health.DEGRADED.value and m.windows[-1] == []


def test_cached_reading_judged_by_its_own_adapter_age(clock):
    """Forge F1 per reading: a fast adapter's cached reading keeps the global age even when it
    shares a module with a slow adapter; a slow adapter's reading keeps its own longer age."""
    slow, fast = _Adapter("slow", "shared", max_age_s=35.0), _Adapter("fast", "shared")
    m = _Mod("S_shared", {"shared"})
    e = _engine((slow, m))
    e.add_adapter(fast)
    _push(e, "fast", clock)
    e.step()
    clock[0] += 20 * S
    res = e.step()
    assert m.windows[-1] == []                             # fast reading is 20 s old: never scored
    assert _health(res, "fast") == Health.DEGRADED.value
    _push(e, "slow", clock)
    e.step()
    clock[0] += 20 * S
    res = e.step()
    assert len(m.windows[-1]) == 1                         # slow reading at 20 s: within its 35 s
    clock[0] += 16 * S
    assert e.step() and m.windows[-1] == []                # 36 s: past its own age


def test_max_age_read_each_step(clock):
    a, m = _Adapter("slow", "slow_cap", max_age_s=35.0), _Mod("S_slow", {"slow_cap"})
    e = _engine((a, m))
    _push(e, "slow", clock)
    e.step()
    clock[0] += 20 * S
    assert _health(e.step(), "slow") == Health.OK.value
    a.max_age_s = 10.0                          # e.g. the adapter switched to a faster mode
    res = e.step()
    assert _health(res, "slow") == Health.DEGRADED.value and m.windows[-1] == []
