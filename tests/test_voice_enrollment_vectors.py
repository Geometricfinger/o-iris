"""tests/vectors/voice_enrollment.json (scripts/make_android_vectors.py, from the real pod handlers) is current and
covers every voice op, and android/core Enrollment.kt / Checks.kt are wired to it. The Kotlin side itself runs in
`:core:check` (Checks.kt voiceEnrollmentChecks)."""
import importlib.util, json, re
from pathlib import Path

from oiris.modules.face_match import remote as R

ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "tests/vectors/voice_enrollment.json"
KOTLIN = ROOT / "android/core/src/main/kotlin/org/oiris/core/Enrollment.kt"
CHECKS = ROOT / "android/core/src/jvmCheck/kotlin/org/oiris/core/Checks.kt"


def _gen():
    spec = importlib.util.spec_from_file_location("make_android_vectors", ROOT / "scripts/make_android_vectors.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_voice_enrollment_vectors_are_current():
    mod = _gen()
    assert mod.differences(json.loads(VECTORS.read_text()), json.loads(json.dumps(mod.voice_enrollment()))) == []


def test_every_op_request_and_reply_parses():
    v = json.loads(VECTORS.read_text())
    mod = _gen()
    assert {c["op"] for c in v["ops"]} == set(R.VOICE_OPS) == {c["op"] for c in v["requests"]}
    for op in R.VOICE_OPS:   # each op: an accepted reply, and the plain "voice link is off" refusal
        assert any(c["op"] == op and c["parsed"]["ok"] for c in v["ops"]), op
        assert any(c["op"] == op and c["parsed"].get("error") == f"PermissionError: {R.VOICE_OFF}" for c in v["ops"]), op
    for c in v["requests"]:
        assert mod._ve_request(c["op"], c["args"]) == c["request"]
    for c in v["ops"]:
        assert mod._ve_parser(c["op"])(c["reply"]) == c["parsed"]
        assert ("request" in c) != ("request_sha256" in c)
    well_formed = {"integer target", "integer seconds and level"}   # JSON integers where seconds are expected: fine
    for r in v["replies"]:
        assert mod._ve_parser(r["parser"])(r["reply"]) == r["parsed"]
        assert r["parsed"]["ok"] is (r["name"] in well_formed), r["name"]
        if r["parsed"]["ok"]:
            assert all(isinstance(r["parsed"][k], float) for k in r["parsed"] if k.endswith("_s") or k == "level_dbfs")


def test_vectors_cover_accepted_refused_more_speech_and_finish():
    v = json.loads(VECTORS.read_text())
    codes = {c["parsed"].get("code") for c in v["ops"]}
    assert {"ok", "prompt", "rate", "invalid", "loud", "speech", "enough", "windows", "speech_total", "done"} <= codes
    errors = " ".join(c["parsed"].get("error") or "" for c in v["ops"])
    for word in ("longer than 12 s", "too large", "whole 16-bit samples", "not valid base64", "rate must be an integer",
                 "lacks scope voice_enrollment"):
        assert word in errors, word
    short = [c for c in v["requests"] if c["op"] == "voice_sample" and len(c["args"]["pcm"]) > 100]
    assert short and all(len(c["args"]["pcm"]) <= 1000 for c in short)    # a few hundred samples, listed in full
    flows = {f["name"]: f["final"] for f in v["flows"]}
    windows = _gen().VE_CFG["enroll_windows"]   # more accepted than the window count: the pod asked for more speech
    assert any(f["stage"] == "saved" and f["refused"] >= 1 and f["accepted"] > windows for f in flows.values())
    assert any(f["stage"] == "stopped" and f["refused"] == v["constants"]["max_refused"] for f in flows.values())
    assert sum(f["stage"] == "stopped" and f["refused"] == 0 for f in flows.values()) == 2
    # no PCM is ever echoed back
    replies = [c["reply"] for c in v["ops"]] + [s["reply"] for f in v["flows"] for s in f["steps"]]
    assert all('"pcm16"' not in r for r in replies)
    sent = [json.loads(c["request"])["pcm16"] for c in v["ops"] if c["op"] == "voice_sample" and "request" in c]
    assert all(p not in r for p in sent if len(p) >= 8 for r in replies)


def test_kotlin_constants_and_checks_are_wired():
    kt = KOTLIN.read_text()

    def const(name: str) -> str:
        m = re.search(rf"const val {name} = (.+)", kt)
        assert m, name
        return m.group(1).split("//")[0].strip()
    assert const("MAX_SAMPLE_S") == str(R.MAX_SAMPLE_S) and const("RATE_MIN") == str(R.VOICE_RATE_MIN)
    assert const("RATE_MAX") == str(R.VOICE_RATE_MAX) and const("MAX_PROMPTS") == str(R.MAX_PROMPTS)
    assert const("MAX_REPLY_TEXT") == str(R.MAX_REPLY_TEXT) and const("VOICE_OFF") == json.dumps(R.VOICE_OFF)
    for name in ("voicePrompts", "voiceStart", "voiceSample", "voiceFinish", "voiceCancel", "parsePrompts", "parseStart",
                 "parseSample", "parseFinish", "class VoiceEnrollmentFlow"):
        assert name in kt, name
    checks = CHECKS.read_text()
    assert "voiceEnrollmentChecks(root)" in checks and "tests/vectors/voice_enrollment.json" in checks
