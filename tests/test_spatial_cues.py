"""Direction cues (oiris/spatial/cues.py) and what the wearer sees and hears: the glasses edge marker, the live
snapshot, speech only with output.speak_alerts, and the dashboard's Located list. Synthetic sim_rig scenes only."""
# ruff: noqa: F811
from __future__ import annotations
import time

import pytest

from oiris.adapters import sim_rig
from oiris.config import Config
from oiris.dashboard.server import render as page
from oiris.engine import Engine
from oiris.guard.phrases import SLOT_VALUES, render as phrase
from oiris.headless import run_sync
from oiris.spatial import Rig, locate, merge
from oiris.spatial.cues import Cue, cues, current, distance_word, nearest_event
from oiris.spatial.merge import NS_PER_S
from oiris.ui.glasses import CUE_EDGES, GlassesState, display_list, render as svg, state_from_result
from tests.test_live import SIM, _runner, _step_until, live_env  # noqa: F401  (fixture reuse)

ORBIT_LEFT_S = 3.0   # the "orbit" scene's emitter turns 30 deg/s from ahead: at 3 s it is at +90, "at left"


def scene(name: str, t: float = 0.1) -> list[dict]:
    """The merged spatial events a sim_rig scene gives at scene time t (as res["spatial"] carries them)."""
    rig = Rig.from_config(sim_rig.SIM_RIG)
    dets = [d for m in rig.mounts.values() for d in sim_rig.observe(m, sim_rig.SCENES[name], t)]
    located, bad = locate(rig, dets)
    assert not bad
    return [e.to_dict() for e in merge(located)]


def result(spatial: list[dict], signatures=("A1",), level="attention") -> dict:
    return {"level": level, "signatures": list(signatures), "spatial": spatial}


# ---- cues() -------------------------------------------------------------------------------------------
def test_a_glint_seen_behind_says_behind_through_render():
    [c] = cues(result(scene("behind")))
    assert c.text == phrase("glint_bearing", direction="behind") == "Candidate camera lens behind"
    assert (c.signature, c.phrase_key, c.direction, c.slots) == ("A1", "glint_bearing", "behind", {"direction": "behind"})
    assert abs(abs(c.az) - 180) < 1e-6 and c.evidence == 1 and c.level == "attention"


def test_an_ir_emitter_at_left_says_at_left():
    [c] = cues(result(scene("orbit", ORBIT_LEFT_S), ["A2"]))
    assert c.text == "Infrared emitter at left" and c.direction == "at left" and abs(c.az - 90) < 1e-6


def test_through_the_engine_with_the_simulated_rig():
    e = Engine(Config(rig=sim_rig.SIM_RIG).validate())
    for a in sim_rig.make("behind"):
        a.open({})
        e.add_adapter(a)
    headless = dict(run_sync(e, 3)[-1])                       # headless output drops times: same step, still matched
    timed = dict(e.results[-1])
    assert "t_mono_ns" not in headless["spatial"][0] and "t_mono_ns" in timed["spatial"][0]
    for res in (headless, timed):
        res.update(level="attention", signatures=["A1", "A4"])   # the sim rig carries no score modules: signatures injected
        assert [c.text for c in cues(res)] == ["Candidate camera lens behind"]


def test_no_spatial_event_no_cue_and_a_mismatched_kind_no_cue():
    assert cues(result([])) == [] and cues({"level": "attention", "signatures": ["A1"]}) == []
    assert cues(result(scene("quiet"))) == []
    assert cues(result(scene("behind"), ["A2", "A3"])) == []          # a glint does not place an emitter or heat
    assert cues(result(scene("heat_and_sound_ahead"), ["A1"])) == []   # neither does heat or sound place a glint
    assert cues(result(scene("behind"), ["A4", "A10"])) == []         # templates without a {direction} slot


def test_no_level_no_cue():
    assert cues(result(scene("behind"), level=None)) == []


def test_each_signature_gets_its_own_kind_in_signature_order():
    cs = cues(result(scene("sides"), ["A1", "A2"]))
    assert [(c.signature, c.direction) for c in cs] == [("A1", "at left"), ("A2", "at right")]
    assert [c.text for c in cs] == ["Candidate camera lens at left", "Infrared emitter at right"]


def test_heat_ahead_gives_no_marker_word_but_still_a_phrase():
    [c] = cues(result(scene("heat_and_sound_ahead"), ["A3"]))
    assert c.text == "Unusual heat pattern ahead" and c.direction == "ahead"


def glint(name: str, t: float = 0.1) -> dict:
    return next(e for e in scene(name, t) if "glint" in e["kinds"])


def test_nearest_in_time_within_the_merge_window():
    old, new = glint("sides", 0.1), glint("behind", 0.6)
    assert {old["direction"], new["direction"]} == {"at left", "behind"}
    assert cues(result([old, new]))[0].direction == "behind"
    # the reference is the newest event: one 0.5 s older is still in the window, 0.6 s older is not
    edge = {**old, "t_mono_ns": new["t_mono_ns"] - NS_PER_S // 2}
    assert nearest_event([edge], "glint", new["t_mono_ns"]) is edge
    stale = {**old, "t_mono_ns": new["t_mono_ns"] - 6 * NS_PER_S // 10}
    assert cues(result([stale, {**new, "kinds": ["sound"]}])) == []
    assert cues(result([stale]), t_ref_ns=new["t_mono_ns"]) == []


def test_ties_are_deterministic():
    a = {**glint("sides"), "confidence": 0.5}
    b = {**a, "az": -100.0, "direction": "at right", "confidence": 0.9}
    assert cues(result([a, b]))[0].direction == "at right" == cues(result([b, a]))[0].direction


def test_distance_only_from_a_measured_range():
    ev = scene("behind")[0]
    assert cues(result([{**ev, "range_m": 0.9}]))[0].text == "Candidate camera lens behind, one metre"
    for r, word in ((0.5, "half a metre"), (2.0, "two metres"), (3.0, "three metres"), (9.0, "far")):
        assert distance_word(r) == word and word in SLOT_VALUES["distance"]
    for r in (None, float("nan"), float("inf"), -1.0, True, "2"):
        assert distance_word(r) is None          # no usable range: no distance word at all, never a guessed "far"
    assert cues(result([{**ev, "range_m": 9.0}]))[0].text == "Candidate camera lens behind, far"   # measured, past 3.5 m


def test_a_distance_phrase_without_a_no_range_variant_makes_no_cue(monkeypatch):
    import oiris.spatial.cues as cues_module
    monkeypatch.setattr(cues_module, "NO_RANGE_PHRASE", {})
    assert cues(result(scene("behind"))) == []
    assert cues(result([{**scene("behind")[0], "range_m": 2.0}]))[0].text == "Candidate camera lens behind, two metres"


def test_the_guard_applies_an_unlisted_direction_gives_no_cue():
    ev = {**scene("behind")[0], "direction": "north"}
    assert cues(result([ev])) == []


def test_pure_and_deterministic():
    res = result(scene("sides"), ["A2", "A1"])
    before = repr(res)
    assert cues(res) == cues(res) and repr(res) == before
    assert all(isinstance(c, Cue) and c.to_dict()["text"] == c.text for c in cues(res))


def test_current_cue_is_the_most_corroborated_then_first():
    a = Cue("A1", "glint", {}, "behind", 180.0, 0.0, 1, "attention", "x")
    b = Cue("A2", "ir_emitter", {}, "at left", 90.0, 0.0, 2, "attention", "y")
    c = Cue("A3", "thermal", {}, "at right", -90.0, 0.0, 1, "attention", "z")
    assert current([a, b, c]) is b and current([a, c]) is a and current([]) is None


# ---- glasses edge marker ------------------------------------------------------------------------------
EXPECTED_EDGE = {"at left": ("left", 0, 184.0, 8, 32), "at right": ("right", 632, 184.0, 8, 32),
                 "behind": ("bottom", 0, 392, 640, 8), "above": ("top", 304.0, 0, 32, 8), "below": ("bottom", 304.0, 392, 32, 8)}


@pytest.mark.parametrize("word", sorted(SLOT_VALUES["direction"]))
@pytest.mark.parametrize("moving", [False, True])
def test_edge_cue_op_for_every_direction_and_the_moving_rule(word, moving):
    ops = display_list(GlassesState(level="medium", alert_text="Infrared emitter " + word, moving=moving, cue_direction=word))
    cue_ops = [o for o in ops if o["op"] == "cue"]
    if word == "ahead":
        assert cue_ops == []
    else:
        [o] = cue_ops
        assert (o["edge"], o["x"], o["y"], o["w"], o["h"]) == EXPECTED_EDGE[word] and o["direction"] == word
        assert o["colour"] == "#E6E6E6" and CUE_EDGES[word] == o["edge"]
    # the marker stays while moving; the alert text (a pop-up) does not
    assert any(o["op"] == "text" and o["id"] == "G2" for o in ops) is (not moving)


def test_svg_without_a_cue_is_unchanged_and_a_cue_draws_one_rect():
    base = GlassesState(level="low", alert_text="Heat source", health={"thermal": "ok"})
    assert svg(GlassesState(**{**base.__dict__, "cue_direction": None})) == svg(base)
    assert svg(GlassesState(**{**base.__dict__, "cue_direction": "ahead"})) == svg(base)
    s = svg(GlassesState(**{**base.__dict__, "cue_direction": "behind"}))
    rect = ('<rect id="cue" class="cue" data-direction="behind" data-edge="bottom" x="0" y="392" width="640" height="8" '
            'fill="#E6E6E6"/>')
    assert s.count('class="cue"') == 1 and s.replace(rect, "") == svg(base)


def test_state_from_result_carries_the_cue_direction():
    st = state_from_result({"level": "attention", "signatures": ["A1"]}, "walking", "at left")
    assert st.cue_direction == "at left" and st.moving and any(o["op"] == "cue" for o in display_list(st))


# ---- live ---------------------------------------------------------------------------------------------
RIG_BEHIND = {"name": "sim_rig", "args": {"scenario": "behind", "period_s": 0.01}}


class Said:
    def __init__(self):
        self.lines: list[str] = []

    def chime(self):
        pass

    def say(self, text):
        self.lines.append(text)


def _live(live_env, speak: bool, player: Said):
    cfg, reg, tmp = live_env
    # no M-06 attention spacing here, so the cue phrase plays right after the level words instead of 10 s later
    cfg = Config(data_dir=cfg.data_dir, rig=sim_rig.SIM_RIG, output={"speak_alerts": speak},
                 voice_attention_min_interval_s=0.0).validate()
    from oiris.live import LiveRunner, LiveSettings
    return LiveRunner(cfg, LiveSettings(adapters=[SIM, RIG_BEHIND], hz=20.0, site="lab"), registry=reg, announce_player=player)


def _until_cue(r, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        time.sleep(0.02)
        r.step()
        snap = r.snapshot()
        if snap["cues"]:
            return snap
    raise AssertionError("no cue within the timeout")


def _wait(cond, timeout=3.0):
    t0 = time.monotonic()
    while not cond() and time.monotonic() - t0 < timeout:
        time.sleep(0.02)
    return cond()


def test_live_snapshot_carries_cues_and_speech_only_with_speak_alerts(live_env):
    for speak in (False, True):
        player = Said()
        r = _live(live_env, speak, player)
        r.start(background=False)
        requests: list[tuple] = []
        real = r.output.request
        r.output.request = lambda kind, *a, **kw: (requests.append((kind, a, kw)), real(kind, *a, **kw))[1]
        try:
            snap = _until_cue(r)
            assert {"signature": "A1", "direction": "behind", "evidence": 1} in snap["cues"]
            assert snap["glasses"]["cue_direction"] == "behind"
            assert any(e["direction"] == "behind" and "glint" in e["kinds"] and e["sensors"] == 1 and e["seconds_ago"] >= 0
                       for e in snap["located"])
            cue_reqs = [q for q in requests if q[0] == "cue"]
            if speak:
                assert cue_reqs and all(q[2]["text"] == "Candidate camera lens behind" for q in cue_reqs)
                assert _wait(lambda: "Candidate camera lens behind" in player.lines)
            else:
                assert cue_reqs == [] and "Candidate camera lens behind" not in player.lines
        finally:
            r.stop()


def test_live_without_spatial_adapters_has_empty_cues(live_env):
    r = _runner(live_env, [SIM])
    r.start(background=False)
    try:
        _step_until(r, 2)
        snap = r.snapshot()
        assert snap["cues"] == [] and snap["located"] == [] and snap["glasses"]["cue_direction"] is None
    finally:
        r.stop()


# ---- dashboard ----------------------------------------------------------------------------------------
def _snap(**kw):
    return {"mode": "live", "running": True, "started": 0.0, "steps": 3, "hz": 1.0, "site": "live", "sweep_id": None,
            "adapters": {}, "health": {}, "result": None, **kw}


def test_dashboard_located_list_is_escaped():
    snap = _snap(located=[{"direction": "<b>behind</b>", "kinds": ["glint", "<script>"], "sensors": 2, "seconds_ago": 1.4},
                          {"direction": "at left", "kinds": ["ir_emitter"], "sensors": 1, "seconds_ago": 0.0}])
    html = page(None, snap)
    assert "<h2>Located</h2>" in html and "<script>" not in html and "<b>behind</b>" not in html
    assert "&lt;b&gt;behind&lt;/b&gt;" in html and "glint, &lt;script&gt;" in html
    assert "2 sensors agree · 1 s ago" in html and "<b>at left</b> · ir_emitter · 1 sensor · 0 s ago" in html


def test_dashboard_unchanged_without_located_events():
    assert page(None, _snap()) == page(None, _snap(located=[]))
    assert "Located" not in page(None, _snap(located=[]))
