"""The face camera in live mode (ADR-0057 decision 5): refused while M-16 is off, opened only under the capture
interlock's grant (ADR-0063), limited by the power plan (ADR-0061), frames zeroed after M-16 saw them and never
recorded, and "voice and face agree" on the heard card. Synthetic only: synth_face_frame, synthetic voices, a fake
VideoCapture and the deterministic test backend. No real camera, microphone, face, voice or network."""
# ruff: noqa: F811
from __future__ import annotations
import json, time

import numpy as np

from oiris import live as live_mod
from oiris.adapters import sim
from oiris.adapters.camera import FaceCameraAdapter, MAX_FPS
from oiris.capture.interlock import CAMERA, INDICATOR_FAILED
from oiris.config import Config
from oiris.dashboard.server import render
from oiris.engine import Engine
from oiris.live import (CAMERA_PLAN_FIELDS, CAMERA_RETRY_S, FACE_CAMERA, POWER_NO_CONSUMER, LiveRunner,
                        LiveSettings)
from oiris.modules.face_match.backends.test_deterministic import synth_face_frame
from oiris.modules.voice_link.announce import NullPlayer
from oiris.power.governor import PowerReading
from oiris.registry import default_registry
from oiris.schema import Sample
from tests.test_camera_adapter import FakeVideoCapture
from tests.test_dashboard_live import _snap
from tests.test_live import SIM
from tests.test_live_capture import CONFIRM_S, Clock, LoggedIndicator, interlock
from tests.test_voice_link import (InjectedBackend, _inj_registry, count, enroll_face, face_on, slots,
                                   synth_voice)
from tests.test_voice_link import env  # noqa: F401  (fixture reuse)
from tests.test_voice_link_live import FakeCapture, live_cfg, quiet, wait_until
from tests.test_voice_link_live import enrolled  # noqa: F401  (fixture reuse)

ALEX_SEED, STRANGER_SEED = 11, 99          # synthetic "faces" (enroll_face(face, 11, ...) in the enrolled fixture)
FRAME_NOISE = 50                           # a fresh synthetic frame of the same "face", not an enrollment shot
CAM = {"name": FACE_CAMERA, "options": {"fps": MAX_FPS}}
SETTLE_S = 0.3                             # real seconds to show that nothing more happens
SAVER_PCT, CRITICAL_PCT, FULL_PCT = 15.0, 5.0, 90.0
SKIN_HOT_C = 41.5                          # over skin_hot_enter_c (41), under skin_critical_enter_c (43)
STEP_S = 10.0                              # one power interval per step
MAX_STEPS = 20


class Cameras:
    """FaceCameraAdapter's video_capture: every device opened is kept; each frame is one synthetic face."""

    def __init__(self, log: list | None = None, seed: int = ALEX_SEED):
        self.log = log if log is not None else []
        self.seed = seed
        self.made: list[FakeVideoCapture] = []

    def __call__(self, device: int) -> FakeVideoCapture:
        vc = FakeVideoCapture(device, self.log, frame=synth_face_frame(self.seed, noise_seed=FRAME_NOISE))
        self.made.append(vc)
        return vc


class Power:
    """A power reading source the test steers (battery and skin), timed in the runner's clock."""

    def __init__(self, clock, battery_pct: float = FULL_PCT, skin_c: float | None = None):
        self.clock, self.battery_pct, self.skin_c = clock, battery_pct, skin_c

    def read(self, plan):
        return PowerReading(t=self.clock(), battery_pct=self.battery_pct, charging=False, skin_c=self.skin_c)


def camera_runner(env, cfg: Config, cams: Cameras, adapters=None, **kw) -> LiveRunner:
    reg = default_registry({"entry_points": False})
    reg.register("sensor", FACE_CAMERA, lambda **a: FaceCameraAdapter(video_capture=cams, **a), port_version="1.0")
    kw.setdefault("face_backend", InjectedBackend())
    kw.setdefault("speaker", slots("inj", reg=_inj_registry()))
    kw.setdefault("announce_player", NullPlayer())
    return LiveRunner(cfg, LiveSettings(adapters=[CAM] if adapters is None else adapters, hz=20.0, site="lab"),
                      registry=reg, store=(env["ks"], env["db"]), **kw)


def m16_cfg(env, **extra) -> Config:
    cfg = Config(data_dir=str(env["dir"]))
    cfg.flags["M-16"] = True
    for k, v in extra.items():
        setattr(cfg, k, v)
    return cfg.validate()


def camera_line(snap: dict) -> str:
    page = render(None, snap)
    return page.split("<p id=CAM", 1)[1].split("</p>", 1)[0] if "<p id=CAM" in page else ""


# ---- off unless configured, refused unless M-16 -------------------------------------------------------------
def test_without_the_camera_in_config_nothing_changes(env):
    face_on(env)
    cams = Cameras()
    r = camera_runner(env, m16_cfg(env), cams, adapters=[SIM])
    r.start(background=False)
    try:
        r.step()
        snap = r.snapshot()
        assert "camera" not in snap and "<p id=CAM" not in render(None, snap)
        assert r.engine.face_match is None and r._camera_gate is None and FACE_CAMERA not in r.engine.bus.adapters
        assert cams.made == []
    finally:
        r.stop()


def test_refused_with_a_plain_reason_while_m16_is_off(env):
    cams = Cameras()
    cfg = Config(data_dir=str(env["dir"])).validate()
    r = camera_runner(env, cfg, cams)
    r.start(background=False)
    try:
        r.step()
        snap = r.snapshot()
        a = snap["adapters"][FACE_CAMERA]
        assert a["available"] is False and "face matching (M-16) is off" in a["reason"]
        assert snap["camera"] == {"state": "off", "code": "m16_off", "reason": "", "fps": None, "frames": 0}
        assert "Camera: <b>off (face matching is off)</b>" in camera_line(snap)
        assert r.engine.face_match is None and FACE_CAMERA not in r.engine.bus.adapters
        time.sleep(SETTLE_S)
        assert cams.made == []                                             # never opened
    finally:
        r.stop()


def test_m16_on_without_the_opt_in_leaves_the_camera_closed(env):
    cams = Cameras()
    r = camera_runner(env, m16_cfg(env), cams)
    r.start(background=False)
    try:
        snap = r.snapshot()
        assert snap["camera"]["state"] == "unavailable" and "opt" in snap["camera"]["reason"]
        assert snap["adapters"][FACE_CAMERA]["available"] is False
        assert cams.made == [] and r.engine.face_match is None
    finally:
        r.stop()


# ---- the capture interlock (ADR-0063) ----------------------------------------------------------------------
def test_the_camera_opens_only_after_the_light_confirms_and_a_revoke_releases_it(env):
    face_on(env)
    clk, log = Clock(), []
    light = LoggedIndicator(clk, log, confirm_delay_s=CONFIRM_S)
    cams = Cameras(log)
    r = camera_runner(env, m16_cfg(env), cams, interlock=interlock(clk, light), clock=clk)
    r.start(background=False)
    try:
        assert wait_until(lambda: cams.made and r._camera_gate.passed >= 1)
        first_true = log.index(("read", True))
        assert log[0] == ("set", True) and ("read", False) in log[:first_true]   # it had to wait for the light
        assert log.index(("vc_open", 0)) > first_true                          # the device only after the grant
        assert step_until_seen(r)
        snap = r.snapshot()
        assert snap["capture"]["camera"] == "active" and snap["camera"]["state"] == "on"
        assert "Camera: <b>on, 10 frames a second</b>" in camera_line(snap)
        assert snap["camera"]["frames"] >= 1
        light.stuck_off = True                                                 # the light stops confirming
        r.step()                                                               # check() revokes; the camera follows
        assert wait_until(lambda: cams.made[0].released)
        assert not r.interlock.held(CAMERA)
        passed, frames = r._camera_gate.passed, r._camera_face.frames
        time.sleep(SETTLE_S)
        r.step()
        assert r._camera_gate.passed == passed and r._camera_face.frames == frames   # no frame after the revoke
        snap = r.snapshot()
        assert snap["camera"]["state"] == "waiting" and snap["camera"]["code"] == INDICATOR_FAILED
        assert "Camera: <b>waiting for the recording light</b>" in camera_line(snap)
        assert len(cams.made) == 1                                             # no reopen while the light is out
    finally:
        r.stop()
    assert all(vc.released for vc in cams.made)


def test_a_light_that_never_confirms_keeps_the_camera_closed_and_retries_slowly(env):
    face_on(env)
    clk, log = Clock(), []
    cams = Cameras(log)
    r = camera_runner(env, m16_cfg(env), cams, interlock=interlock(clk, LoggedIndicator(clk, log, stuck_off=True)),
                      clock=clk)
    r.start(background=False)
    try:
        assert wait_until(lambda: r._camera_gate.state == "waiting")
        time.sleep(SETTLE_S)
        tries = log.count(("set", True))
        assert tries == 1 and cams.made == []                                  # one try, then the retry interval
        clk.t += CAMERA_RETRY_S
        assert wait_until(lambda: log.count(("set", True)) == tries + 1)
        assert cams.made == [] and r.snapshot()["camera"]["state"] == "waiting"
    finally:
        r.stop()


# ---- the power plan (ADR-0061) -----------------------------------------------------------------------------
def test_the_plan_rate_and_stride_apply_and_critical_pauses_the_camera(env):
    face_on(env)
    clk = Clock(1000.0)
    power = Power(clk, battery_pct=SAVER_PCT)
    cams = Cameras()
    r = camera_runner(env, m16_cfg(env), cams, power_source=power, clock=clk)
    r.start(background=False)
    try:
        assert wait_until(lambda: r._camera_gate.passed >= 1)                 # a frame went through the plan's gate
        snap = r.snapshot()
        plan = snap["power"]["plan"]
        assert snap["power"]["level"] == "saver" and plan["frame_stride"] == 2
        assert snap["camera"]["fps"] == min(MAX_FPS, plan["camera_max_fps"]) / plan["frame_stride"]
        assert "Camera: <b>on, 5 frames a second</b>" in camera_line(snap)
        assert snap["power"]["no_consumer"] == [k for k in POWER_NO_CONSUMER if k not in CAMERA_PLAN_FIELDS]
        assert r._camera_gate._gate_rate == (plan["camera_max_fps"], plan["frame_stride"])
        power.battery_pct = CRITICAL_PCT
        for _ in range(MAX_STEPS):
            clk.t += STEP_S
            r.step()
            if r.snapshot()["camera"]["state"] == "paused":
                break
        snap = r.snapshot()
        assert snap["power"]["level"] == "critical"
        assert snap["camera"]["state"] == "paused" and snap["camera"]["code"] == "battery"
        assert "Camera: <b>paused to save the battery</b>" in camera_line(snap)
        assert wait_until(lambda: cams.made[0].released) and not r.interlock.held(CAMERA)
        passed = r._camera_gate.passed
        time.sleep(SETTLE_S)
        assert r._camera_gate.passed == passed and len(cams.made) == 1
    finally:
        r.stop()


def test_hot_turns_face_matching_off_and_the_camera_pauses_to_cool_down(env):
    face_on(env)
    clk = Clock(1000.0)
    power = Power(clk)
    cams = Cameras()
    r = camera_runner(env, m16_cfg(env), cams, power_source=power, clock=clk)
    r.start(background=False)
    try:
        assert wait_until(lambda: r._camera_gate.state == "on")
        power.skin_c = SKIN_HOT_C
        for _ in range(MAX_STEPS):
            clk.t += STEP_S
            r.step()
            if r.snapshot()["camera"]["state"] == "paused":
                break
        snap = r.snapshot()
        assert snap["power"]["level"] == "hot" and snap["power"]["plan"]["face_match"] is False
        assert snap["power"]["plan"]["camera_max_fps"] > 0                     # the rate alone would allow frames
        assert snap["camera"]["state"] == "paused" and snap["camera"]["code"] == "heat"
        assert "Camera: <b>paused to cool down</b>" in camera_line(snap)
        assert wait_until(lambda: cams.made[0].released)
    finally:
        r.stop()


# ---- frames: zeroed after use, never recorded ------------------------------------------------------------
class SeenFace:
    def __init__(self):
        self.seen: list[tuple[int, int]] = []

    def process_frame(self, frame, place=None):
        self.seen.append((id(frame), int(np.asarray(frame).sum())))
        return {"faces": 0, "matched": 0, "unrecognized": 0}


class FrameSource:
    id, caps = "cam", {"face_frame"}


def engine_with_frames(face, frames) -> Engine:
    e = Engine(Config(), face_match=face)
    e.add_adapter(FrameSource())
    for i, f in enumerate(frames):
        e.bus.queues["cam"].append(Sample("cam", i + 1, 0.0, i + 1, "dn", 1.0, f))
    return e


def test_frames_are_zeroed_after_process_frame_and_never_reach_the_results():
    frames = [synth_face_frame(ALEX_SEED), synth_face_frame(STRANGER_SEED)]
    face = SeenFace()
    e = engine_with_frames(face, frames)
    res = e.step()
    assert [i for i, _ in face.seen] == [id(f) for f in frames] and all(s > 0 for _, s in face.seen)   # seen whole
    assert all(not f.any() for f in frames)                                     # then zeroed
    json.dumps(res)                                                             # no array anywhere in the result
    assert len(e.bus.queues["cam"]) == 0


def test_frames_are_zeroed_with_m16_off_too_and_read_only_frames_do_not_break_the_step():
    f, ro = synth_face_frame(ALEX_SEED), synth_face_frame(ALEX_SEED)
    ro.flags.writeable = False
    e = engine_with_frames(None, [f, ro])
    e.step()
    assert not f.any() and ro.any()


def test_no_recorder_or_replay_source_carries_face_frames():
    """record_clip and the replay clips come from the sim scenarios: none has the face_frame capability."""
    for scenario in sorted(sim.SCENARIOS):
        assert all("face_frame" not in a.caps for a in sim.make(scenario)), scenario


# ---- end to end: "voice and face agree" ----------------------------------------------------------------------
def step_until_seen(r: LiveRunner, frames: int = 1) -> bool:
    """Step until M-16 has seen `frames` camera frames (the bus thread queues them on its own time)."""
    def seen() -> bool:
        r.step()
        return r._camera_face is not None and r._camera_face.frames >= frames
    return wait_until(seen)


def heard_card(r: LiveRunner, cap: FakeCapture, voice: str = "A") -> dict:
    cap.feed(np.concatenate([synth_voice(voice, seed=50), quiet(1.5)]))
    assert wait_until(lambda: r.snapshot()["voice"]["heard"] == 1)
    return r.snapshot()["voice"]["cards"][0]


def test_an_enrolled_face_and_voice_within_the_window_agree_on_the_card(enrolled):
    env, alex = enrolled.env, enrolled.a
    cap, cams = FakeCapture(), Cameras(seed=ALEX_SEED)
    r = camera_runner(env, live_cfg(env), cams, voice_capture=cap)
    r.start(background=False)
    try:
        assert r._camera_face._face is r._face                                  # one M-16 module for both
        assert step_until_seen(r), r.snapshot()["camera"]
        assert r._face.last_match_at(alex) is not None
        assert count(env["db"], "fm_encounter") >= 1                            # the encounter log, as spec 6 allows
        assert r.engine.events == [] and r.snapshot()["output"]["popup"] is None   # never an alert for a face
        card = heard_card(r, cap)
        assert card["name"] == "Alex" and card["face_agrees"] is True
        assert "Recent face match for this person" in render(None, r.snapshot())
    finally:
        r.stop()


def test_a_strangers_face_does_not_agree(enrolled):
    env = enrolled.env
    cap, cams = FakeCapture(), Cameras(seed=STRANGER_SEED)
    r = camera_runner(env, live_cfg(env), cams, voice_capture=cap)
    r.start(background=False)
    try:
        assert step_until_seen(r), r.snapshot()["camera"]
        assert heard_card(r, cap)["face_agrees"] is False
    finally:
        r.stop()


def test_a_strangers_face_leaves_nothing_beyond_counts(env):
    face = face_on(env, modes=("personal", "research"))
    enroll_face(face, ALEX_SEED, "Alex")
    db = env["db"]
    tables = [t for (t,) in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'fm_%'")]
    before = {t: count(db, t) for t in tables}
    cams = Cameras(seed=STRANGER_SEED)
    r = camera_runner(env, m16_cfg(env), cams)
    r.start(background=False)
    try:
        assert step_until_seen(r, frames=2), r.snapshot()["camera"]
        snap = r.snapshot()
        assert len(json.dumps(snap)) < len(synth_face_frame(STRANGER_SEED).tobytes())   # no frame in the snapshot
    finally:
        r.stop()
    after = {t: count(db, t) for t in tables}
    changed = {t for t in tables if after[t] != before[t]}
    assert changed == {"fm_unrecognized_count"}                                  # a count, nothing else
    total = db.execute("SELECT SUM(count) FROM fm_unrecognized_count").fetchone()[0]
    assert total >= 1 and after["fm_encounter"] == before["fm_encounter"] == 0


# ---- the dashboard line ----------------------------------------------------------------------------------
def test_the_camera_line_in_plain_words_and_escaped():
    def line(**cam):
        return camera_line({**_snap(None), "camera": {"state": "", "code": "", "reason": "", "fps": None, "frames": 0,
                                                      **cam}})
    assert line(state="on", fps=2.0) == ' style="grid-column:1/4">Camera: <b>on, 2 frames a second</b>'
    assert "on, 1 frame a second" in line(state="on", fps=1.0)
    assert "on, 0.5 frames a second" in line(state="on", fps=0.5)
    assert "off (face matching is off)" in line(state="off", code="m16_off")
    assert "waiting for the recording light" in line(state="waiting", code="indicator_unconfirmed")
    assert "paused to cool down" in line(state="paused", code="heat")
    assert "paused to save the battery" in line(state="paused", code="battery")
    evil = line(state="unavailable", reason="<script>alert(1)</script> & co")
    assert "<script>" not in evil and "&lt;script&gt;alert(1)&lt;/script&gt; &amp; co" in evil
    assert "unknown" in line(state="<b>")
    assert "<p id=CAM" not in render(None, _snap(None))                          # not configured: no line


def test_live_module_names_the_camera_fields():
    assert set(CAMERA_PLAN_FIELDS) <= set(POWER_NO_CONSUMER) and live_mod.FACE_CAMERA == "face_camera"
