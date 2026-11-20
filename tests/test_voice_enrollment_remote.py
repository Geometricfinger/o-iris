"""Phone-guided voice enrollment over the EnrollmentPort (ADR-0057; oiris/modules/face_match/remote.py voice ops).
Synthetic audio (tests/test_voice_link.py synth_voice: tones, not voices) and the fake deterministic speaker plugin
only; no microphone, no real model."""
import base64, json, logging, subprocess, sys

import numpy as np
import pytest

from oiris.config import VOICE_ENROLL_MAX_WINDOWS, VOICE_LINK_DEFAULTS, VOICE_LINK_RANGES
from oiris.modules.face_match import remote as R
from oiris.modules.face_match.remote import EnrollmentServer, dumps
from oiris.modules.voice_link import core as vl_core
from oiris.modules.voice_link.prompts import PROMPTS
from oiris.security.channel import MAX_MESSAGE, Reassembler, fragment
from tests.test_phone_link import _pair
from tests.test_voice_link import (ROOT, SR, enroll_face, env, face_on, synth_voice, voice_on)  # noqa: F401

PID = PROMPTS[0].id
HIGHER_TARGET_S = 8.0     # total-speech target for the "more speech" test: above three 1.8 s windows
SHORT_SPEECH_S = 1.4      # synth speech per short window: about 1.9 s after the VAD hangover, above the 1.8 s minimum


def pcm16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.asarray(x) * R.INT16_SCALE), -R.INT16_SCALE, R.INT16_SCALE - 1).astype(np.int16)


def b64(pcm: np.ndarray) -> str:
    return base64.b64encode(pcm.astype("<i2").tobytes()).decode()


@pytest.fixture
def pod(env):  # noqa: F811
    face = face_on(env)
    a, b = enroll_face(face, 11, "Alex"), enroll_face(face, 12, "Blake")
    vm = voice_on(env, face)                              # 3 windows, 5 s total speech
    srv = EnrollmentServer(env["db"], env["chain"], "phone1", lambda: face, module=face, voice=vm)
    return srv, vm, face, a, b


def call(srv, req: str | dict) -> dict:
    return json.loads(srv.handle(req if isinstance(req, str) else dumps(req)))


def sample(srv, person: str, x, prompt: str = PID, rate: int = SR) -> dict:
    return call(srv, R.voice_sample_request(person, prompt, rate, pcm16(x)))


def raw_sample(srv, person: str, text, rate=SR, prompt=PID) -> dict:
    return call(srv, {"op": "voice_sample", "person": person, "prompt_id": prompt, "rate": rate, "pcm16": text})


# ---- the order start, samples, finish ------------------------------------------------------------------------
def test_end_to_end_over_the_noise_channel(pod, env):  # noqa: F811
    srv, vm, face, a, _ = pod
    tx, rx = _pair()
    to_pod, to_phone = Reassembler(tx), Reassembler(rx)

    def over(req: str) -> dict:
        got = None
        for f in fragment(rx, req.encode()):
            got = to_pod.feed(f)
        out = None
        for f in fragment(tx, srv.handle(got).encode()):
            out = to_phone.feed(f)
        return R.parse_voice_sample(out) if '"voice_sample"' in req else json.loads(out)

    ps = over(R.voice_prompts_request())["prompts"]
    assert [p["id"] for p in ps] == [p.id for p in PROMPTS] and all(set(p) == {"id", "text", "tone"} for p in ps)
    st = over(R.voice_start_request(a, "c-voice"))
    assert st == {"ok": True, "windows": 3, "target_speech_s": 5.0, "window_s": vm.cfg.enroll_window_s,
                  "max_windows": VOICE_ENROLL_MAX_WINDOWS}
    k, r = 0, {"remaining": 1}
    while r["remaining"]:
        r = over(R.voice_sample_request(a, ps[k % len(ps)]["id"], SR, pcm16(synth_voice("A", seconds=7.0, seed=k + 1))))
        assert r["ok"] and r["accepted"] and r["code"] == "ok", r
        assert set(r) == {"ok", "error", "accepted", "code", "message", "speech_s", "total_speech_s", "level_dbfs", "remaining"}
        k += 1
    fin = over(R.voice_finish_request(a))
    assert fin["ok"] and fin["saved"] and fin["code"] == "done" and fin["remaining"] == 0 and "Alex" in fin["message"]
    assert env["db"].execute("SELECT windows FROM vl_voice WHERE person_id=?", (a,)).fetchone()[0] == k
    assert vm.capture_buffer_size() == 0


def test_a_seven_second_16k_sample_fits_the_caps():
    req = R.voice_sample_request("p", PID, SR, pcm16(synth_voice("A", seconds=7.0)))
    assert len(json.loads(req)["pcm16"]) < R.MAX_PCM_B64 and len(req) < MAX_MESSAGE // R.MAX_PCM_SHARE_OF_MESSAGE
    assert R.MAX_PCM_BYTES == R.MAX_SAMPLE_S * R.VOICE_RATE_MAX * 2 and R.MAX_PCM_B64 * 4 < MAX_MESSAGE


def test_need_more_total_speech_then_finish(pod):
    srv, vm, _, a, _ = pod
    vm.cfg.enroll_min_total_speech_s = HIGHER_TARGET_S
    assert call(srv, R.voice_start_request(a, "c-voice"))["target_speech_s"] == HIGHER_TARGET_S
    short = [sample(srv, a, synth_voice("A", seed=s, max_speech=SHORT_SPEECH_S), PROMPTS[s].id) for s in range(3)]
    assert all(r["accepted"] for r in short) and short[-1]["remaining"] == 1 and short[-1]["total_speech_s"] < HIGHER_TARGET_S
    fin = call(srv, R.voice_finish_request(a))
    assert fin == {**fin, "ok": True, "saved": False, "code": "speech_total", "remaining": 1} and "more speech" in fin["message"]
    r = sample(srv, a, synth_voice("A", seed=9), PROMPTS[3].id)
    assert r["accepted"] and r["remaining"] == 0
    assert call(srv, R.voice_finish_request(a))["saved"] is True


# ---- refusal codes and errors come through ---------------------------------------------------------------------
def test_each_window_refusal_code_comes_through(pod):
    srv, vm, _, a, _ = pod
    call(srv, R.voice_start_request(a, "c-voice"))
    assert sample(srv, a, synth_voice("A"), "no-such")["code"] == "prompt"
    assert sample(srv, a, synth_voice("A"), rate=4000)["code"] == "rate"
    assert sample(srv, a, synth_voice("A"), rate=vl_core.SAMPLE_RATE_MAX + 1)["code"] == "rate"
    assert raw_sample(srv, a, "")["code"] == "invalid"
    assert sample(srv, a, np.sign(synth_voice("A")) * 0.999)["code"] == "loud"
    r = sample(srv, a, synth_voice("A", max_speech=0.3))
    assert (r["code"], r["accepted"]) == ("speech", False) and r["message"] == vl_core.REASONS["speech"]
    vm.speaker._inst["speaker_embed"].embed = lambda *_: (_ for _ in ()).throw(RuntimeError("model crashed"))
    r = sample(srv, a, synth_voice("A"))
    assert r["code"] == "backend" and "crashed" not in json.dumps(r)
    del vm.speaker._inst["speaker_embed"].embed
    for s in (1, 2, 3):
        assert sample(srv, a, synth_voice("A", seed=s))["accepted"]
    assert sample(srv, a, synth_voice("A", seed=4))["code"] == "enough"
    # "mono" cannot arrive: pcm16 is mono by definition


def test_finish_refusal_codes_and_start_errors(pod, env):  # noqa: F811
    srv, vm, face, a, b = pod
    assert call(srv, R.voice_finish_request(a))["ok"] is False                 # no open capture
    assert "lacks scope voice_enrollment" in call(srv, R.voice_start_request(a, "c-face"))["error"]
    assert "not enrolled in face matching" in call(srv, R.voice_start_request("nobody", "c-voice"))["error"]
    minor = enroll_face(face, 13, "Kid", consent="c-minor", guardian=True)
    err = call(srv, R.voice_start_request(minor, "c-v-minor-noguard"))["error"]
    assert err.startswith("ConsentError") and "guardian" in err
    assert call(srv, R.voice_start_request(minor, "c-minor"))["ok"]
    call(srv, R.voice_start_request(a, "c-voice"))
    fin = call(srv, R.voice_finish_request(a))
    assert (fin["saved"], fin["code"], fin["remaining"]) == (False, "windows", 3)
    for e in (call(srv, {"op": "voice_start", "person": a}), call(srv, {"op": "voice_finish"})):
        assert e["ok"] is False and "Traceback" not in e["error"] and len(e["error"]) <= 200


def test_withdrawn_consent_and_disabled_module_refuse_plainly(pod, env):  # noqa: F811
    srv, vm, _, a, _ = pod
    call(srv, R.voice_start_request(a, "c-voice"))
    for s in (1, 2, 3):
        sample(srv, a, synth_voice("A", seed=s))
    env["consents"].withdraw("c-voice")
    r = call(srv, R.voice_finish_request(a))
    assert r["ok"] is False and r["error"].startswith("ConsentError")
    assert env["db"].execute("SELECT COUNT(*) FROM vl_voice").fetchone()[0] == 0
    vm.disable()
    r = call(srv, R.voice_start_request(a, "c-voice"))
    assert r["ok"] is False and "not opted in" in r["error"]


def test_model_unavailable_at_start(env):  # noqa: F811
    face = face_on(env)
    a = enroll_face(face, 11, "Alex")
    vm = voice_on(env, face, speaker="broken")
    vm.speaker._inst.pop("speaker_embed", None)
    vm.speaker.slots["speaker_embed"] = ""
    srv = EnrollmentServer(env["db"], env["chain"], "phone1", lambda: face, module=face, voice=vm)
    r = call(srv, R.voice_start_request(a, "c-voice"))
    assert r["ok"] is False and r["error"].startswith("VoiceLinkUnavailable")


# ---- input limits ----------------------------------------------------------------------------------------------
def test_oversize_long_odd_and_bad_base64_are_refused_without_decoding(pod, monkeypatch):
    srv, vm, _, a, _ = pod
    call(srv, R.voice_start_request(a, "c-voice"))
    decoded = []
    real = R._decode_pcm16
    monkeypatch.setattr(R, "_decode_pcm16", lambda *a_: decoded.append(1) or real(*a_))
    big = "A" * (R.MAX_PCM_B64 + 4)
    assert "too large" in raw_sample(srv, a, big, rate=R.VOICE_RATE_MAX)["error"]
    long_ = b64(np.zeros(R.MAX_SAMPLE_S * SR + 1, np.int16))
    assert "longer than 12 s" in raw_sample(srv, a, long_)["error"]
    for bad in ("AAA", "AA=A", "A===", "====", "AAAAA", 5, None, ["AAAA"]):
        assert raw_sample(srv, a, bad)["ok"] is False, bad
    for odd in ("AA==", "AAAA", "AAAAAAA="):
        assert "whole 16-bit samples" in raw_sample(srv, a, odd)["error"], odd
    assert decoded == []
    for bad in ("AA*AAA==", "AAB=", "AAAAAB==", "ÀÀÀÀÀÀ==", "AA-_AA=="):
        r = raw_sample(srv, a, bad)
        assert r["ok"] is False and "not valid base64" in r["error"], bad
    for rate in ("16000", 16000.0, True, None):
        assert "rate must be an integer" in raw_sample(srv, a, "", rate=rate)["error"]
    assert "prompt_id must be text" in raw_sample(srv, a, "", prompt=5)["error"]
    assert vm.capture_buffer_size() == 0


def test_decoder_is_strict_and_exact():
    rng = np.random.default_rng(3)
    for n in (0, 1, 2, 3, 5, 300):
        x = rng.integers(-32768, 32768, n).astype(np.int16)
        text = b64(x)
        got = R._decode_pcm16(text, R._pcm16_bytes(text), [])
        assert np.array_equal(got, x.astype(np.float32) / np.float32(R.INT16_SCALE))
    edge = np.array([0, 1, -1, 32767, -32768, 256, -256], np.int16)
    assert R._decode_pcm16(b64(edge), len(edge) * 2, []).tolist() == (edge / R.INT16_SCALE).tolist()


# ---- buffers zeroed; no audio in replies or logs -----------------------------------------------------------------
def test_every_buffer_is_zeroed_after_use(pod, monkeypatch):
    srv, vm, _, a, _ = pod
    call(srv, R.voice_start_request(a, "c-voice"))
    handed, wiped = [], []
    real_submit, real_wipe = vm.submit_window, R._wipe

    def spy_submit(person, audio, rate, prompt_id=None):
        handed.append(audio)
        return real_submit(person, audio, rate, prompt_id)

    def spy_wipe(bufs):
        real_wipe(bufs)
        wiped.extend(bufs)
    monkeypatch.setattr(vm, "submit_window", spy_submit)
    monkeypatch.setattr(R, "_wipe", spy_wipe)
    assert sample(srv, a, synth_voice("A", seed=1))["accepted"]
    assert sample(srv, a, synth_voice("A", max_speech=0.3))["code"] == "speech"
    assert len(handed) == 2 and all(h.size and not np.any(h) for h in handed)
    assert len(wiped) >= 2 * 6 and all(not any(bytes(w) if isinstance(w, bytearray) else w.tobytes()) for w in wiped)
    assert any(w is handed[0] for w in wiped)

    def boom(*_):
        raise RuntimeError("module failed")
    handed.clear(); wiped.clear()
    monkeypatch.setattr(vm, "submit_window", lambda p, audio, r, i=None: handed.append(audio) or boom())
    assert sample(srv, a, synth_voice("A", seed=2))["ok"] is False
    assert handed and wiped and all(not np.any(w) for w in wiped if isinstance(w, np.ndarray))


def test_no_audio_in_replies_or_logs(pod, caplog):
    srv, vm, _, a, _ = pod
    caplog.set_level(logging.DEBUG)
    x = pcm16(synth_voice("A", seed=5))
    text = b64(x)
    replies = [call(srv, R.voice_start_request(a, "c-voice"))]
    vm.speaker._inst["speaker_embed"].embed = lambda *_: (_ for _ in ()).throw(RuntimeError("model crashed"))
    replies.append(call(srv, R.voice_sample_request(a, PID, SR, x)))                           # logged: the slot failure
    del vm.speaker._inst["speaker_embed"].embed
    replies += [call(srv, R.voice_sample_request(a, PID, SR, x)), raw_sample(srv, a, text[:-4] + "AA*A"),
                raw_sample(srv, a, text + "A"), raw_sample(srv, a, text, rate=4000), call(srv, R.voice_finish_request(a))]
    out = json.dumps(replies) + "\n".join(r.getMessage() for r in caplog.records)
    assert caplog.records, "the slot failure should have been logged (without audio)"
    for chunk in (text[:24], text[len(text) // 2:len(text) // 2 + 24], text[-24:]):
        assert chunk not in out
    assert all("pcm16" not in r for r in replies)
    assert all(len(json.dumps(r)) < 400 for r in replies)


# ---- voice link off; face ops unchanged ---------------------------------------------------------------------------
def test_voice_ops_with_the_module_off_are_refused_plainly(env):  # noqa: F811
    face = face_on(env)
    for srv in (EnrollmentServer(env["db"], env["chain"], "phone1", lambda: face, module=face),
                EnrollmentServer(env["db"], env["chain"], "phone1", lambda: None)):
        for req in (R.voice_prompts_request(), R.voice_start_request("p", "c"), R.voice_finish_request("p"),
                    R.voice_cancel_request("p"), R.voice_sample_request("p", PID, SR, np.zeros(8, np.int16)),
                    dumps({"op": "voice_sample"})):
            assert call(srv, req) == {"ok": False, "error": f"PermissionError: {R.VOICE_OFF}"}


def _face_transcript(srv, seed: int) -> list:
    from oiris.modules.face_match.backends.test_deterministic import synth_face_frame
    from oiris.modules.face_match.port import REQUIRED_ANGLES
    out = [call(srv, {"op": "status"}), call(srv, {"op": "warning", "jurisdiction": "us-other"}),
           call(srv, {"op": "nope"}), call(srv, "not json")]
    sid = call(srv, {"op": "start_session"})["session"]
    pid = call(srv, {"op": "add_person", "session": sid, "label": "Casey", "consent_id": "c-voice"})["person"]
    for i, ang in enumerate(REQUIRED_ANGLES):
        g = np.clip(synth_face_frame(seed, ang, noise_seed=i), 0, 255).astype(np.uint8)
        out.append(call(srv, {"op": "shot", "person": pid, "angle": ang.value, "w": g.shape[1], "h": g.shape[0],
                              "fmt": "bgr" if g.ndim == 3 else "gray", "pixels_b64": base64.b64encode(g.tobytes()).decode()}))
    out += [call(srv, {"op": "finish", "person": pid}), call(srv, {"op": "end_session", "session": sid}),
            call(srv, {"op": "shot", "person": pid, "angle": "front", "w": 0, "h": 1, "pixels_b64": ""})]
    return out


def test_face_ops_unchanged_with_voice_wired(pod, env):  # noqa: F811
    srv, vm, face, _, _ = pod
    plain = EnrollmentServer(env["db"], env["chain"], "phone1", lambda: face, module=face)
    with_voice = _face_transcript(srv, 21)
    without = _face_transcript(plain, 22)
    assert with_voice == without
    assert with_voice[2] == {"ok": False, "error": "ValueError: unknown op 'nope'"}
    assert with_voice[-3]["complete"] is True


# ---- constants and import isolation ---------------------------------------------------------------------------
def test_constants_match_the_module():
    assert (R.VOICE_RATE_MIN, R.VOICE_RATE_MAX) == (vl_core.SAMPLE_RATE_MIN, vl_core.SAMPLE_RATE_MAX)
    assert R.INT16_SCALE == vl_core.INT16_SCALE
    lo, hi = VOICE_LINK_RANGES["enroll_window_s"]
    assert lo <= VOICE_LINK_DEFAULTS["enroll_window_s"] <= R.MAX_SAMPLE_S <= hi   # holds the default window


def test_importing_the_enrollment_port_does_not_import_voice_link():
    code = ("import sys; import oiris.modules.face_match.remote, oiris.link.pod_service;"
            "bad = [k for k in sys.modules if k.startswith('oiris.modules.voice_link')]; assert not bad, bad")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr


def test_phone_side_parsers_never_raise():
    for junk in ("", "{", "[]", "null", '{"ok":true,"prompts":[null]}', '{"ok":NaN}', '{"ok":false}', b"\xff"):
        for p in (R.parse_voice_prompts, R.parse_voice_start, R.parse_voice_sample, R.parse_voice_finish, R.parse_voice_ack):
            ack = p is R.parse_voice_ack and str(junk).startswith('{"ok":true')   # any {"ok": true} is an ack
            assert p(junk) == ({"ok": True, "error": None} if ack else {"ok": False, "error": "bad reply"})
    assert R.parse_voice_ack('{"ok":false,"error":"voice link is off"}') == {"ok": False, "error": "voice link is off"}
