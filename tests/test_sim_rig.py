"""Simulated 360-degree rig (oiris/adapters/sim_rig.py) through the engine. Synthetic scenes only."""
from __future__ import annotations
import json

import pytest

from oiris.adapters import sim, sim_rig
from oiris.config import Config
from oiris.engine import Engine
from oiris.headless import run_scenario, run_sync
from oiris.registry import default_registry
from oiris.spatial import Rig
from oiris.spatial.detections import DETECTIONS_KEY, SPATIAL_CAP
from oiris.spatial.frames import wrap_az

STEPS = 3


def seen_by(scenario: str, t: float = 0.1) -> dict[str, list]:
    r = Rig.from_config(sim_rig.SIM_RIG)
    return {m.id: sim_rig.observe(m, sim_rig.SCENES[scenario], t) for m in r.mounts.values()}


def run(scenario: str, steps: int = STEPS, rig_cfg: dict | None = None, extra=()) -> list[dict]:
    e = Engine(Config(rig=sim_rig.SIM_RIG if rig_cfg is None else rig_cfg).validate())
    for a in [*sim_rig.make(scenario, rig=rig_cfg), *extra]:
        a.open({})
        e.add_adapter(a)
    return run_sync(e, steps)


def test_registered_and_shaped_like_a_sensor_adapter():
    r = default_registry()
    adapters = r.get("sensor", "sim_rig").factory("behind")
    assert {a.id for a in adapters} == {"rig_front", "rig_left", "rig_rear", "rig_right", "rig_front_thermal", "rig_mics"}
    s = adapters[0].read(0.0)
    assert all(a.caps == {SPATIAL_CAP} for a in adapters) and set(s.data) == {"mount_id", DETECTIONS_KEY}
    with pytest.raises(ValueError):
        sim_rig.make("nonexistent")


def test_an_object_behind_is_seen_only_by_the_rear_mount():
    seen = seen_by("behind")
    assert [k for k, v in seen.items() if v] == ["rear"]
    d = seen["rear"][0]
    assert d.pixel is not None and abs(d.pixel[0] - 320) < 1e-9 and abs(d.pixel[1] - 240) < 1e-9
    ev = run("behind")[-1]["spatial"]
    assert len(ev) == 1 and ev[0]["direction"] == "behind" and ev[0]["sensors"] == ["rear"] and abs(wrap_az(ev[0]["az"] - 180)) < 1e-6


def test_an_object_between_two_cameras_merges_into_one_event():
    seen = seen_by("between")
    assert sorted(k for k, v in seen.items() if v) == ["front", "left"]
    ev = run("between")[-1]["spatial"]
    assert len(ev) == 1
    e = ev[0]
    assert e["sensors"] == ["front", "left"] and e["count"] == 2 and abs(e["az"] - 45) < 1e-6 and e["direction"] == "ahead"
    assert e["evidence"] == 1 and abs(e["confidence"] - (1 - 0.3 ** 2)) < 1e-6


def test_sides_heat_and_sound():
    ev = run("sides")[-1]["spatial"]
    assert sorted((e["direction"], tuple(e["sensors"])) for e in ev) == [("at left", ("left",)), ("at right", ("right",))]
    ev = run("heat_and_sound_ahead")[-1]["spatial"]
    assert len(ev) == 1 and ev[0]["families"] == ["thermal", "voice"] and ev[0]["evidence"] == 2
    assert ev[0]["sensors"] == ["front_thermal", "mics"] and ev[0]["direction"] == "ahead" and 10 < ev[0]["az"] < 15
    ev = run("above")[-1]["spatial"]
    assert len(ev) == 1 and ev[0]["direction"] == "above" and abs(ev[0]["el"] - 70) < 1e-6


def test_orbit_hands_over_between_cameras_and_words():
    words, sensors = [], []
    for res in run("orbit", steps=130):   # 0.1 s a step at 30 deg/s: 13 s of scene time, a full turn and a bit
        (e,) = res["spatial"]
        words.append(e["direction"])
        sensors.append(tuple(e["sensors"]))
    order = [w for i, w in enumerate(words) if i == 0 or w != words[i - 1]]
    assert order[:5] == ["ahead", "at left", "behind", "at right", "ahead"]
    assert ("front", "left") in sensors and ("left", "rear") in sensors and ("rear",) in sensors


def test_a_rig_without_rear_has_no_coverage_behind():
    cfg = {"mounts": [m for m in sim_rig.SIM_RIG["mounts"] if m["id"] != "rear"]}
    res = run("behind", rig_cfg=cfg)
    assert all(r["spatial"] == [] for r in res)
    assert run("quiet")[-1]["spatial"] == []


def test_engine_reports_detections_it_cannot_place():
    # samples placed with the sim rig's mounts, resolved against the default rig (front only)
    e = Engine(Config())
    for a in sim_rig.make("sides"):
        a.open({}); e.add_adapter(a)
    res = run_sync(e, 1)[-1]
    assert res["spatial"] == [] and res["spatial_unplaced"] == 2


def test_spatial_does_not_change_fusion():
    plain = run_scenario("camera_lens", STEPS)
    e = Engine(Config(rig=sim_rig.SIM_RIG))
    for a in [*sim.make("camera_lens"), *sim_rig.make("heat_and_sound_ahead")]:
        a.open({}); e.add_adapter(a)
    mixed = run_sync(e, STEPS)
    for p, m in zip(plain, mixed):
        assert "spatial" not in p and m["spatial"]
        assert {k: v for k, v in m.items() if k not in ("spatial",)} == p


def test_without_spatial_samples_results_are_unchanged():
    for scen in sim.SCENARIOS:
        res = run_scenario(scen, STEPS)
        assert all("spatial" not in r and "spatial_unplaced" not in r for r in res)
        with open(f"golden/expected/{scen}.json", encoding="utf-8") as f:
            assert json.load(f) == json.loads(json.dumps(res))
