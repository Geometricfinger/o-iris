"""M-17 Android parity (ADR-0059): tests/vectors/voice_link.json is current, and the pure core.decide()
that android/core VoiceLink.kt reproduces gives process_audio's outcomes. Synthetic audio only."""
# ruff: noqa: F811
import json, subprocess, sys
from pathlib import Path

import pytest

from oiris.modules.voice_link import core
from oiris.modules.voice_link.core import HEARD, NONE, SELF, SPOOFED, UNSURE, decide
from tests.test_voice_link import SR, count, enroll_face, enroll_voice, face_on, synth_voice, voice_on
from tests.test_voice_link import env, two  # noqa: F401  (fixture reuse)
from tests.test_voice_link_announce import spoof_vm

ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "tests/vectors/voice_link.json"
TH, MG, ST = 0.5, 0.05, 0.5


def test_voice_link_vectors_are_current():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/make_android_vectors.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_vectors_cover_every_outcome_and_the_name_rules():
    v = json.loads(VECTORS.read_text())
    assert {d["outcome"] for d in v["decide"]} == {HEARD, UNSURE, NONE, SELF, SPOOFED}
    assert any(d["spoof_score"] is None and d["outcome"] == HEARD for d in v["decide"])
    assert any(len(d["scores"]) == 1 for d in v["decide"])
    names = {n["name"]: n["speakable"] for n in v["names"]}
    assert names["Alex"] == "Alex" and names["Robot Smith"] is None and names["Alеx"] is None   # Cyrillic e
    assert names["山田さくら"] and names["O’Brien"] and names["Alex2"] is None
    assert {len(a["segments"]) for a in v["vad"]} >= {0, 1, 2}


def test_decide_order_and_lazy_spoof():
    asked = []

    def spoof():
        asked.append(1)
        return 0.9

    assert decide([], TH, MG, None, spoof, ST) == (NONE, None)
    assert decide([(0.4, "a")], TH, MG, None, spoof, ST) == (NONE, None)
    assert decide([(0.9, "a"), (0.87, "b")], TH, MG, None, spoof, ST) == (UNSURE, None)
    assert decide([(0.9, "a")], TH, MG, "a", spoof, ST) == (SELF, None)
    assert not asked                                                   # never asked before the spoof step
    assert decide([(0.9, "a")], TH, MG, None, spoof, ST) == (SPOOFED, None) and asked == [1]
    assert decide([(0.2, "b"), (0.9, "a")], TH, MG, None, None, ST) == (HEARD, "a")
    assert decide([(0.9, "a")], TH, MG, None, lambda: None, ST) == (HEARD, "a")


@pytest.fixture
def spy(monkeypatch):
    """Records each decide() call made by process_audio, with the spoof score it resolved to."""
    calls: list[dict] = []

    def wrapped(scores, th, mg, self_id, spoof, st):
        seen: list = []

        def resolved():
            seen.append(spoof() if callable(spoof) else spoof)
            return seen[-1]

        out = decide(scores, th, mg, self_id, resolved, st)
        calls.append({"args": (list(scores), th, mg, self_id, seen[0] if seen else None, st), "out": out})
        return out

    monkeypatch.setattr(core, "decide", wrapped)
    return calls


def outcome_of(res: dict, call: dict) -> str:
    """process_audio's result read back as a decide() outcome."""
    if res["heard"] is not None:
        assert call["out"] == (HEARD, res["heard"]["person_id"])
        return HEARD
    if res["unsure"]:
        return UNSURE
    return call["out"][0]


def check(res: dict, calls: list, expected: str) -> None:
    assert calls, "process_audio did not reach the decision"
    call = calls[-1]
    assert outcome_of(res, call) == call["out"][0] == expected
    assert decide(*call["args"]) == call["out"]                       # the pure function alone agrees
    assert (res["heard"] is not None) == (expected == HEARD) and res["unsure"] == (expected == UNSURE)


def test_decide_matches_process_audio_heard_none_and_self(two, spy):
    vm = two.vm
    check(vm.process_audio(synth_voice("A", seed=50), SR), spy, HEARD)
    check(vm.process_audio(synth_voice("B", seed=51), SR), spy, HEARD)
    check(vm.process_audio(synth_voice("C", seed=52), SR), spy, NONE)   # never enrolled
    vm.set_self(two.a)
    check(vm.process_audio(synth_voice("A", seed=53), SR), spy, SELF)
    assert count(two.env["db"], "vl_heard") == 2


def test_decide_matches_process_audio_unsure(env, spy):
    face = face_on(env)
    a, a2 = enroll_face(face, 11, "Alex"), enroll_face(face, 14, "Avery")
    vm = voice_on(env, face)
    enroll_voice(vm, a, "A", seeds=(1, 2, 3)); enroll_voice(vm, a2, "A", seeds=(4, 5, 6))   # two nearly identical voices
    check(vm.process_audio(synth_voice("A", seed=60), SR), spy, UNSURE)


def test_decide_matches_process_audio_spoof(env, spy):
    vm = spoof_vm(env, 0.9)
    check(vm.process_audio(synth_voice("A", seed=50), SR), spy, SPOOFED)
    vm.speaker.get("voice_antispoof").score = 0.2
    check(vm.process_audio(synth_voice("A", seed=51), SR), spy, HEARD)


def _vectors_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_android_vectors", ROOT / "scripts/make_android_vectors.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_vector_check_tolerates_cpu_float_noise_but_nothing_else():
    """The last bits of float results differ between CPUs (x86 vs ARM); --check must not call that stale, and must
    still catch every real change."""
    m = _vectors_script()
    base = {"cosine": [{"v": 0.123456789012345, "n": 3, "s": "Alex", "ok": True}], "speech_s": 1.25}
    same = json.loads(json.dumps(base))
    same["cosine"][0]["v"] += 1e-12      # last-bit noise
    assert m.differences(base, same) == []
    for change in (lambda d: d["cosine"][0].__setitem__("v", 0.124),         # a real numeric change
                   lambda d: d["cosine"][0].__setitem__("s", "Alexa"),       # a string
                   lambda d: d["cosine"][0].__setitem__("ok", False),        # a boolean
                   lambda d: d["cosine"][0].__setitem__("n", 3.0),           # int vs float
                   lambda d: d.pop("speech_s"),                              # a missing key
                   lambda d: d["cosine"].append({})):                        # list length
        other = json.loads(json.dumps(base))
        change(other)
        assert m.differences(base, other), change
    many = {f"k{i}": i for i in range(20)}
    assert len(m.differences(many, {k: v + 1 for k, v in many.items()})) == m.MAX_REPORTED
