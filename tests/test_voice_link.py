"""M-17 voice link core (ADR-0057, docs/VOICE_LINK_MODULE_SPEC.md). Synthetic audio only: harmonic tones
with different pitch and formant shapes, plus noise. No real voices, no recordings, no audio files. The
speaker model is a fake deterministic `speaker_embed` plugin whose embedding is a band spectrum of the
audio itself, so synthetic "speakers" are separable; it is test-only and refused unless allowed."""
import json, subprocess, sys, time
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from oiris.config import DEFAULT_OFF, LOCKED_ON, MODULES, VOICE_LINK_DEFAULTS, VOICE_LINK_KEYS, Config, validate_voice_link
from oiris.guard.consent import ConsentError, ConsentRecord, ConsentRegistry
from oiris.guard.privacy import PrivacyError, check_persist
from oiris.guard.retention import purge as nightly_purge
from oiris.guard.vocabulary import check_output, check_text
from oiris.modules.face_match import optin as face_optin
from oiris.modules.face_match.backends.test_deterministic import TestDeterministicBackend, synth_face_frame
from oiris.modules.face_match.core import VOICE_AAD, FaceMatchConfig, FaceMatchModule
from oiris.modules.face_match.port import REQUIRED_ANGLES
from oiris.modules.voice_link import optin, vad
from oiris.modules.voice_link.capture import MIC_CHECK_S
from oiris.modules.voice_link.core import (REASONS, VoiceLinkConfig, VoiceLinkLocked, VoiceLinkModule, VoiceLinkUnavailable,
                                           wipe_voice_data)
from oiris.registry import PluginMeta, Registry
from oiris.security.keys import KeyStore
from oiris.slots import SlotManager
from oiris.store.db import DB

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
META = PluginMeta(license="Apache-2.0", data_tier="synthetic", commercial_ok=True)
# Synthetic speakers: (fundamental Hz, [(formant centre Hz, width Hz), ...]). Tones, not voices.
SPEAKERS = {
    "A": (120, [(700, 250), (1200, 300)]),
    "B": (210, [(400, 200), (2300, 300)]),
    "C": (300, [(1600, 300), (3200, 400)]),   # never enrolled: the "unknown" voice
}
BAND_EDGES = np.geomspace(80, 4000, 49)
NFFT, HOP = 512, 256


def synth_voice(spk: str, seconds: float = 5.0, seed: int = 0, amp: float = 0.3, lead: float = 0.5,
                max_speech: float | None = None, sr: int = SR) -> np.ndarray:
    """Syllable-like bursts of a harmonic tone shaped by formant bumps, over a faint noise floor."""
    f0, formants = SPEAKERS[spk]
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    x = rng.normal(0, 1e-3, n)
    pos, made = int(lead * sr), 0
    while pos < n and (max_speech is None or made < max_speech * sr):
        syl = int(rng.uniform(0.35, 0.6) * sr)
        if max_speech is not None:
            syl = min(syl, int(max_speech * sr) - made)
        end = min(n, pos + syl)
        tt = np.arange(end - pos) / sr
        vib = 1 + 0.01 * np.sin(2 * np.pi * rng.uniform(3, 6) * tt)
        s = np.zeros_like(tt)
        for k in range(1, int(4000 / f0)):
            w = sum(np.exp(-((k * f0 - fc) / bw) ** 2) for fc, bw in formants) + 0.02
            s += w * np.sin(2 * np.pi * k * f0 * tt * vib + rng.uniform(0, 2 * np.pi))
        x[pos:end] += amp * s / np.max(np.abs(s)) * np.hanning(end - pos)
        made += end - pos
        pos = end + int(rng.uniform(0.08, 0.18) * sr)
    return x.astype(np.float32)


class FakeSpeaker:
    """TEST-ONLY speaker_embed plugin: mean-centred, unit band spectrum of the audio. Deterministic."""
    name, dim, model_id, test_only = "fake_spectral", 48, "fake-spectral-v1", True

    def __init__(self, model_dir: str = "") -> None:
        self.model_dir = model_dir
        self.seen: list[np.ndarray] = []   # the arrays the core handed over (to check they get zeroed)

    def embed(self, audio, sample_rate):
        self.seen.append(audio)
        w = np.hanning(NFFT)
        frames = np.array([audio[i:i + NFFT] * w for i in range(0, len(audio) - NFFT, HOP)])
        p = np.mean(np.abs(np.fft.rfft(frames, axis=1)) ** 2, axis=0)
        f = np.fft.rfftfreq(NFFT, 1 / sample_rate)
        bands = np.array([p[(f >= a) & (f < b)].sum() + 1e-12 for a, b in zip(BAND_EDGES[:-1], BAND_EDGES[1:])])
        v = np.sqrt(bands / bands.max())
        v -= v.mean()
        return (v / np.linalg.norm(v)).astype(np.float32)


class OtherModelSpeaker(FakeSpeaker):
    name, model_id = "fake_spectral_v2", "fake-spectral-v2"


class InjectedSpeaker(FakeSpeaker):
    """The same fake, injected through the CLI (a config file may not set allow_test_backend)."""
    name, test_only = "inj", False


class BrokenSpeaker(FakeSpeaker):
    name = "broken"

    def embed(self, audio, sample_rate):
        raise RuntimeError("model crashed")


class InjectedBackend(TestDeterministicBackend):
    test_only = False


def registry() -> Registry:
    r = Registry()
    for cls in (FakeSpeaker, OtherModelSpeaker, InjectedSpeaker, BrokenSpeaker):
        r.register("speaker_embed", cls.name, cls, port_version="1.0", meta=META)
    return r


def slots(name: str = "fake_spectral", reg: Registry | None = None) -> SlotManager:
    return SlotManager(reg or registry(), {"speaker_embed": name}, {name: {"model_dir": "unused"}})


# ---- fixtures ---------------------------------------------------------------------------------
@pytest.fixture
def env(tmp_path):
    ks = KeyStore(tmp_path / "keys.json"); ks.create("pw")
    db = DB(tmp_path / "asb.db", ks.subkey("db"))
    chain = face_optin.DbChain(db, ks.subkey("face-chain"))
    consents = ConsentRegistry(db)
    for cid, scopes in (("c-voice", {"face_enrollment", "voice_enrollment"}), ("c-face", {"face_enrollment"}),
                        ("c-minor", {"face_enrollment", "voice_enrollment", "guardian"}),
                        ("c-minor-face", {"face_enrollment", "guardian"}), ("c-v-only", {"voice_enrollment"}),
                        ("c-v-minor-noguard", {"voice_enrollment"})):
        consents.add(ConsentRecord(cid, "s-" + cid, scopes))
    yield {"ks": ks, "db": db, "chain": chain, "consents": consents, "dir": tmp_path}
    db.close()


def face_on(env, modes=("personal",)) -> FaceMatchModule:
    face_optin.record_opt_in(env["db"], env["chain"], "I UNDERSTAND", list(modes), "dev1", "us-other",
                             warning_shown=face_optin.warning_for("us-other"))
    return FaceMatchModule(env["db"], env["ks"], env["dir"], TestDeterministicBackend(), env["chain"],
                           FaceMatchConfig(modes=list(modes), allow_test_backend=True), env["consents"], timeout_s=1.0)


def enroll_face(face, seed: int, name: str, consent: str = "c-voice", guardian: bool = False) -> str:
    sid = face.start_session()
    pid = face.add_person(sid, name, consent, guardian=guardian)
    for i, a in enumerate(REQUIRED_ANGLES):
        assert face.submit_shot(pid, synth_face_frame(seed, a, noise_seed=i), a).ok
    assert face.finish_person(pid).complete
    face.end_session(sid)
    return pid


def voice_opt_in(env):
    return optin.record_opt_in(env["db"], env["chain"], "I UNDERSTAND", "ambient", "dev1", warning_shown=optin.WARNING_TEXT)


def voice_on(env, face, speaker: str = "fake_spectral", opt: bool = True, **cfg) -> VoiceLinkModule:
    cfg.setdefault("enroll_windows", 3)   # these tests are about storage, consent and matching, not the window count
    cfg.setdefault("enroll_min_total_speech_s", 5.0)   # nor the total-speech target (default 20 s; test_voice_link_tighten.py)
    if opt:
        voice_opt_in(env)
    return VoiceLinkModule(env["db"], face, slots(speaker), VoiceLinkConfig(allow_test_backend=True, **cfg), chain=env["chain"])


def enroll_voice(vm, pid: str, spk: str, consent: str = "c-voice", seeds=(1, 2, 3)) -> None:
    assert vm.start_voice(pid, consent) == len(seeds)
    for s in seeds:
        r = vm.submit_window(pid, synth_voice(spk, seed=s), SR)
        assert r.ok, r.reason
    st = vm.finish_voice(pid)
    extra = max(seeds) + 1
    while st.code == "speech_total":   # a config with the default 20 s total-speech target: more windows, as the CLI adds
        assert vm.submit_window(pid, synth_voice(spk, seed=extra), SR).ok
        st, extra = vm.finish_voice(pid), extra + 1
    assert st.ok and st.done, st.reason


@pytest.fixture
def two(env):
    """M-16 with Alex and Blake enrolled; M-17 on with A's and B's synthetic voices."""
    face = face_on(env)
    a, b = enroll_face(face, 11, "Alex"), enroll_face(face, 12, "Blake")
    vm = voice_on(env, face)
    enroll_voice(vm, a, "A"); enroll_voice(vm, b, "B")
    return SimpleNamespace(face=face, vm=vm, a=a, b=b, env=env)


def count(db, table: str) -> int:
    return db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def m17_records(chain) -> list[dict]:
    return [r for r in chain.records() if r["module"] == "M-17"]


# ---- 1. off by default, gate, opt-in ---------------------------------------------------------------
def test_m17_flag_default_off_and_not_locked():
    assert MODULES["M-17"] == "voice_link" and "M-17" in DEFAULT_OFF and "M-17" not in LOCKED_ON
    assert not Config().enabled("M-17")


def test_gate_returns_none_while_off_and_refuses_without_optin(env):
    from oiris.modules import voice_link_gate
    face = face_on(env)
    cfg = Config(data_dir=str(env["dir"]))
    assert voice_link_gate.load(cfg, env["db"], env["ks"], registry(), face) is None
    cfg.flags["M-16"] = cfg.flags["M-17"] = True
    cfg.voice_link = {"allow_test_backend": True}   # tests only: a config file may not set it
    with pytest.raises(VoiceLinkLocked):            # no M-17 opt-in yet
        voice_link_gate.load(cfg, env["db"], env["ks"], registry(), face, speaker=slots())
    voice_opt_in(env)
    with pytest.raises(VoiceLinkLocked):            # M-16 not loaded
        voice_link_gate.load(cfg, env["db"], env["ks"], registry(), None, speaker=slots())
    vm = voice_link_gate.load(cfg, env["db"], env["ks"], registry(), face, chain=env["chain"], speaker=slots())
    assert isinstance(vm, VoiceLinkModule) and vm.listening == "ambient" and vm.health() == ("listening", "")
    cfg.flags["M-16"] = False
    with pytest.raises(VoiceLinkLocked):
        voice_link_gate.load(cfg, env["db"], env["ks"], registry(), face, speaker=slots())


def test_gate_builds_speaker_slot_from_config(env):
    from oiris.modules import voice_link_gate
    face = face_on(env); voice_opt_in(env)
    cfg = Config(data_dir=str(env["dir"]), flags={**Config().flags, "M-16": True, "M-17": True})
    cfg.voice_link = {"allow_test_backend": True, "speaker_model": "fake_spectral"}
    vm = voice_link_gate.load(cfg, env["db"], env["ks"], registry(), face)
    assert vm.speaker.slots == {"speaker_embed": "fake_spectral"}
    assert vm.speaker.options["fake_spectral"]["model_dir"] == str(env["dir"] / "models" / "voice_link")
    cfg.plugins = {"slots": {"speaker_embed": "fake_spectral_v2"}, "options": {"fake_spectral_v2": {"model_dir": "/x"}}}
    vm = voice_link_gate.load(cfg, env["db"], env["ks"], registry(), face)
    assert vm.model_id == "fake-spectral-v2" and vm.speaker.options["fake_spectral_v2"] == {"model_dir": "/x"}


def test_optin_needs_exact_phrase_warning_and_m16_personal(env):
    db, chain = env["db"], env["chain"]
    with pytest.raises(optin.OptInError):           # M-16 not opted in
        optin.record_opt_in(db, chain, "I UNDERSTAND", "ambient", "d", warning_shown=optin.WARNING_TEXT)
    face_optin.record_opt_in(db, chain, "I UNDERSTAND", ["research"], "d", "us-other", warning_shown=face_optin.WARNING_TEXT)
    with pytest.raises(optin.OptInError):           # research mode only: no enrolled people
        optin.record_opt_in(db, chain, "I UNDERSTAND", "ambient", "d", warning_shown=optin.WARNING_TEXT)
    face_optin.set_modes(db, chain, ["personal"])
    for typed in ("", "yes", "i understand", "I UNDERSTAND please"):
        with pytest.raises(optin.OptInError):
            optin.record_opt_in(db, chain, typed, "ambient", "d", warning_shown=optin.WARNING_TEXT)
    with pytest.raises(optin.OptInError):
        optin.record_opt_in(db, chain, "I UNDERSTAND", "ambient", "d", warning_shown="short")
    assert optin.opt_in_record(db) is None
    rec = optin.record_opt_in(db, chain, " I UNDERSTAND ", "ambient", "d", warning_shown=optin.WARNING_TEXT)
    assert rec["listening"] == "ambient" and optin.opt_in_record(db)["warning_version"] == "v2"
    assert m17_records(chain)[-1]["kind"] == "opt_in" and chain.verify() == len(chain.records())


def test_warning_text_covers_the_adr_points_and_passes_vocabulary():
    t = optin.WARNING_TEXT
    check_text(t); check_output(REASONS)
    for must in ("biometric", "hears everyone in the room", "never stored or counted", "No audio is ever saved",
                 "voice_enrollment", "guardian", "microphone indicator", "Illinois", "Texas", "Washington", "EU/UK",
                 "your decision and your risk", "deletes every voice template"):
        assert must in t, must
    assert optin.CONFIRM_PHRASE == face_optin.CONFIRM_PHRASE == "I UNDERSTAND"


def test_module_locked_without_optin_and_test_plugin_refused(env):
    face = face_on(env)
    with pytest.raises(VoiceLinkLocked):
        VoiceLinkModule(env["db"], face, slots(), VoiceLinkConfig(allow_test_backend=True))
    voice_opt_in(env)
    with pytest.raises(VoiceLinkLocked):
        VoiceLinkModule(env["db"], face, slots(), VoiceLinkConfig())
    assert validate_voice_link({"allow_test_backend": True})


def test_speaker_slot_fault_isolation(env):
    face = face_on(env); pid = enroll_face(face, 11, "Alex")
    vm = voice_on(env, face, speaker="not_installed")
    state, why = vm.health()
    assert state == "unavailable" and "no plugin" in why
    with pytest.raises(VoiceLinkUnavailable):
        vm.start_voice(pid, "c-voice")
    assert vm.process_audio(synth_voice("A"), SR)["health"] == "unavailable"
    broken = VoiceLinkModule(env["db"], face, slots("broken"), VoiceLinkConfig(allow_test_backend=True))
    broken.start_voice(pid, "c-voice")
    r = broken.submit_window(pid, synth_voice("A", seed=1), SR)
    assert not r.ok and r.code == "backend" and broken.health()[0] == "degraded" and "model crashed" in broken.health()[1]
    assert face.process_frame(synth_face_frame(11), place="P")["matched"] == 1   # nothing else breaks


# ---- 2. consent ---------------------------------------------------------------------------------
def test_consent_required_with_voice_scope(env):
    face = face_on(env)
    pid = enroll_face(face, 11, "Alex", consent="c-face")
    vm = voice_on(env, face)
    for cid in (None, "", "nope", "c-face"):
        with pytest.raises(ConsentError):
            vm.start_voice(pid, cid)
    env["consents"].add(ConsentRecord("c-exp", "s", {"voice_enrollment"}, expires_utc=time.time() - 1))
    with pytest.raises(ConsentError, match="expired"):
        vm.start_voice(pid, "c-exp")
    env["consents"].add(ConsentRecord("c-wd", "s", {"voice_enrollment"}, withdrawn=True))
    with pytest.raises(ConsentError, match="withdrawn"):
        vm.start_voice(pid, "c-wd")
    with pytest.raises(KeyError):
        vm.start_voice("not-enrolled", "c-v-only")
    assert vm.start_voice(pid, "c-v-only") == 3


def test_guardian_rule_for_minors(env):
    face = face_on(env)
    kid = enroll_face(face, 13, "Kid", consent="c-minor-face", guardian=True)
    vm = voice_on(env, face)
    with pytest.raises(ConsentError, match="guardian"):
        vm.start_voice(kid, "c-v-minor-noguard")
    assert vm.start_voice(kid, "c-minor") == 3


def test_consent_withdrawn_during_capture_stores_nothing(env):
    face = face_on(env); pid = enroll_face(face, 11, "Alex", consent="c-face")
    vm = voice_on(env, face)
    vm.start_voice(pid, "c-v-only")
    for s in (1, 2, 3):
        assert vm.submit_window(pid, synth_voice("A", seed=s), SR).ok
    env["consents"]._r["c-v-only"].withdrawn = True
    with pytest.raises(ConsentError):
        vm.finish_voice(pid)
    assert count(env["db"], "vl_voice") == 0


# ---- 3. enroll and match ------------------------------------------------------------------------
def test_enroll_two_and_match_each_unknown_dropped(two):
    vm, db = two.vm, two.env["db"]
    assert count(db, "vl_voice") == 2
    r = vm.process_audio(synth_voice("A", seed=50), SR, place="Kitchen", now=1_700_000_029)
    assert r["heard"]["person_id"] == two.a and r["heard"]["name"] == "Alex" and not r["unsure"] and r["health"] == "listening"
    assert set(r["heard"]) == {"person_id", "name", "score", "face_agrees", "entry_id"}
    r = vm.process_audio(synth_voice("B", seed=51), SR, place="Kitchen")
    assert r["heard"]["person_id"] == two.b and r["heard"]["name"] == "Blake"
    rows = db.execute("SELECT person_id, heard_at, score, expires_at FROM vl_heard ORDER BY heard_at").fetchall()
    assert len(rows) == 2 and rows[0][1] % 60 == 0 and rows[0][2] == round(rows[0][2], 4)
    assert rows[0][3] == rows[0][1] + 30 * 86400
    stats = dict(vm.stats)
    r = vm.process_audio(synth_voice("C", seed=52), SR, place="Kitchen")   # never enrolled
    assert r["heard"] is None and not r["unsure"]
    assert count(db, "vl_heard") == 2 and vm.stats == stats               # not stored, not counted
    assert m17_records(two.env["chain"])[-1]["kind"] == "voice_enrolled"


def test_close_call_is_unsure_and_logs_nothing(env):
    face = face_on(env)
    a, a2 = enroll_face(face, 11, "Alex"), enroll_face(face, 14, "Avery")
    vm = voice_on(env, face)
    enroll_voice(vm, a, "A", seeds=(1, 2, 3)); enroll_voice(vm, a2, "A", seeds=(4, 5, 6))   # two nearly identical voices
    r = vm.process_audio(synth_voice("A", seed=60), SR, place="P")
    assert r["unsure"] and r["heard"] is None and count(env["db"], "vl_heard") == 0 and vm.stats["heard"] == 0


def test_too_little_speech_does_nothing(two):
    spy = two.vm.speaker.get("speaker_embed"); n = len(spy.seen)
    r = two.vm.process_audio(synth_voice("A", seed=70, max_speech=0.8), SR, place="P")
    assert r["heard"] is None and 0 < r["speech_s"] < 1.5 and count(two.env["db"], "vl_heard") == 0
    assert len(spy.seen) == n   # never even embedded
    r = two.vm.process_audio(np.zeros(5 * SR, np.float32), SR)
    assert r["heard"] is None and r["speech_s"] == 0


@pytest.mark.parametrize("make,rate,code", [
    (lambda: np.clip(synth_voice("A", seed=1, amp=1.6), -1, 1), SR, "loud"),
    (lambda: np.zeros(5 * SR, np.float32), SR, "speech"),
    (lambda: (np.random.default_rng(1).normal(0, 1e-3, 5 * SR)).astype(np.float32), SR, "speech"),
    (lambda: synth_voice("A", seed=1, max_speech=1.0), SR, "speech"),   # enroll_min_speech_s is 1.8
    (lambda: synth_voice("A", seed=1, sr=4000), 4000, "rate"),
    (lambda: synth_voice("A", seed=1), 96000, "rate"),
    (lambda: np.stack([synth_voice("A", seed=1)] * 2, axis=1), SR, "mono"),
    (lambda: np.full(SR, np.nan, np.float32), SR, "invalid"),
    (lambda: np.zeros(0, np.float32), SR, "invalid"),
])
def test_refused_windows(env, make, rate, code):
    face = face_on(env); pid = enroll_face(face, 11, "Alex")
    vm = voice_on(env, face)
    vm.start_voice(pid, "c-voice")
    buf = make()
    r = vm.submit_window(pid, buf, rate)
    assert not r.ok and r.code == code and r.reason == REASONS[code] and r.needed == 3 and not r.done
    assert vm.capture_buffer_size() == 0 and not np.nan_to_num(buf).any()   # refused buffers are zeroed too


def test_window_feedback_finish_needs_enough_windows_and_cancel(env):
    face = face_on(env); pid = enroll_face(face, 11, "Alex")
    vm = voice_on(env, face, enroll_windows=3)   # the minimum is 3
    vm.start_voice(pid, "c-voice")
    r = vm.submit_window(pid, synth_voice("A", seed=1), SR)
    assert r.ok and r.needed == 2 and r.speech_s >= 2.5 and -40 < r.level_dbfs < -10
    assert vm.submit_window(pid, synth_voice("A", seed=2), SR).needed == 1
    st = vm.finish_voice(pid)
    assert not st.ok and st.needed == 1 and "1 more good window" in st.reason
    assert vm.submit_window(pid, synth_voice("A", seed=3), SR).done
    assert vm.submit_window(pid, synth_voice("A", seed=4), SR).code == "enough"
    vm.cancel_voice(pid)
    assert vm.capture_buffer_size() == 0 and count(env["db"], "vl_voice") == 0
    with pytest.raises(KeyError):
        vm.finish_voice(pid)


def test_int16_and_float64_and_column_inputs_accepted_and_zeroed(env):
    face = face_on(env); pid = enroll_face(face, 11, "Alex")
    vm = voice_on(env, face)
    vm.start_voice(pid, "c-voice")
    bufs = [(synth_voice("A", seed=1) * 32767).astype(np.int16), synth_voice("A", seed=2).astype(np.float64),
            synth_voice("A", seed=3).reshape(-1, 1)]
    for b in bufs:
        assert vm.submit_window(pid, b, SR).ok
        assert not b.any()
    assert vm.finish_voice(pid).ok


# ---- 4. no audio on disk; buffers zeroed ----------------------------------------------------------
def test_no_audio_on_disk_and_buffers_zeroed(two):
    env, vm = two.env, two.vm
    spy = vm.speaker.get("speaker_embed")
    needles = []
    face = two.face
    c = enroll_face(face, 15, "Casey")
    vm.start_voice(c, "c-voice")
    for s in (21, 22, 23):
        buf = synth_voice("B", seed=s) * 0.5 + synth_voice("A", seed=s) * 0.5
        mid = buf.size // 2
        needles += [buf[mid:mid + 64].tobytes(), buf[SR:SR + 64].tobytes()]
        assert vm.submit_window(c, buf, SR).ok
        assert not buf.any()
    assert vm.finish_voice(c).ok
    for s in (31, 32, 33):
        buf = synth_voice("A", seed=s)
        needles.append(buf[buf.size // 2:buf.size // 2 + 64].tobytes())
        vm.process_audio(buf, SR, place="P")
        assert not buf.any()
    assert spy.seen and all(not a.any() for a in spy.seen)   # VAD speech copies handed to the model are zeroed
    assert vm.capture_buffer_size() == 0
    env["db"].execute("PRAGMA wal_checkpoint(TRUNCATE)")
    files = [p for p in env["dir"].rglob("*") if p.is_file()]
    assert any(p.name == "asb.db" for p in files)
    for p in files:
        blob = p.read_bytes()
        assert not any(n in blob for n in needles), p
    with pytest.raises(PrivacyError):
        check_persist("audio", True, source="M-17", m17_enabled=True)
    cols = {r[1] for t in ("vl_state", "vl_voice", "vl_heard") for r in env["db"].execute(f"PRAGMA table_info({t})")}
    assert not [c for c in cols if "audio" in c or "wav" in c or "sample" in c]


def test_templates_are_sealed_with_the_person_key(two):
    db, face = two.env["db"], two.face
    blob = db.execute("SELECT template FROM vl_voice WHERE person_id=?", (two.a,)).fetchone()[0]
    vec = np.frombuffer(face.person_sealer(two.a).open(blob, VOICE_AAD + two.a.encode()), np.float32)
    assert vec.size == FakeSpeaker.dim and abs(float(np.linalg.norm(vec)) - 1) < 1e-5
    with pytest.raises(Exception):
        face.person_sealer(two.b).open(blob, VOICE_AAD + two.a.encode())
    with pytest.raises(Exception):
        face.person_sealer(two.a).open(blob, VOICE_AAD + two.b.encode())
    with pytest.raises(KeyError):
        face.person_sealer("nobody")


# ---- 5. wipes, all verified -----------------------------------------------------------------------
def test_delete_voice_verified_face_untouched(two):
    db, face, vm = two.env["db"], two.face, two.vm
    vm.process_audio(synth_voice("A", seed=50), SR, place="P")
    blob = db.execute("SELECT template FROM vl_voice WHERE person_id=?", (two.a,)).fetchone()[0]
    old_dek = face.person_sealer(two.a)
    rep = vm.delete_voice(two.a)
    assert rep.verified and rep.message == "1 voice template destroyed, verified"
    assert rep.checks == {"row_absent": True, "undecryptable": True, "ciphertext_absent": True}
    assert old_dek.open(blob, VOICE_AAD + two.a.encode())   # only the replaced key could have opened it
    assert db.execute("SELECT COUNT(*) FROM vl_heard WHERE person_id=?", (two.a,)).fetchone()[0] == 0
    assert vm.process_audio(synth_voice("A", seed=51), SR)["heard"] is None
    assert vm.process_audio(synth_voice("B", seed=52), SR)["heard"]["person_id"] == two.b
    assert face.process_frame(synth_face_frame(11), place="P")["matched"] == 1 and face.person_name(two.a) == "Alex"
    assert m17_records(two.env["chain"])[-1]["kind"] == "voice_deleted"
    with pytest.raises(KeyError):
        vm.delete_voice(two.a)


def test_m16_delete_person_erases_the_voice(two):
    db, face = two.env["db"], two.face
    two.vm.process_audio(synth_voice("A", seed=50), SR, place="P")
    blob = db.execute("SELECT template FROM vl_voice WHERE person_id=?", (two.a,)).fetchone()[0]
    rep = face.delete_person(two.a)   # the voice module's RAM cache is not told; it must notice
    assert rep.verified and rep.message == "1 template and 1 voice template destroyed, verified"
    assert rep.checks["voice_row_absent"] and rep.checks["voice_undecryptable"] and rep.checks["voice_ciphertext_absent"]
    assert db.execute("SELECT COUNT(*) FROM vl_voice WHERE person_id=?", (two.a,)).fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM vl_heard WHERE person_id=?", (two.a,)).fetchone()[0] == 0
    assert blob not in Path(db.path).read_bytes()
    assert two.vm.process_audio(synth_voice("A", seed=51), SR)["heard"] is None
    assert two.vm.process_audio(synth_voice("B", seed=52), SR)["heard"]["person_id"] == two.b


def test_m16_disable_module_wipes_voices_and_m17_optin(two):
    db, chain = two.env["db"], two.env["chain"]
    blobs = [r[0] for r in db.execute("SELECT template FROM vl_voice")]
    rep = two.face.disable()
    assert rep.verified and rep.checks["voice_rows"] == 0 and rep.checks["voice_undecryptable"] and rep.checks["voice_ciphertext_absent"]
    assert count(db, "vl_voice") == count(db, "vl_heard") == 0 and optin.opt_in_record(db) is None
    assert not any(b in Path(db.path).read_bytes() for b in blobs)
    assert m17_records(chain)[-1]["kind"] == "disabled_wiped" and chain.verify() == len(chain.records())
    with pytest.raises(VoiceLinkLocked):
        two.vm.process_audio(synth_voice("A", seed=50), SR)


def test_m16_disable_without_module_wipes_voices(two):
    from oiris.modules.face_match.core import disabled_wipe_without_module, voice_rows
    db = two.env["db"]
    disabled_wipe_without_module(db, two.env["dir"])
    assert voice_rows(db) == 0 and count(db, "vl_heard") == 0 and optin.opt_in_record(db) is None


def test_m17_disable_wipes_voices_and_keeps_m16(two):
    db, face, vm = two.env["db"], two.face, two.vm
    vm.process_audio(synth_voice("A", seed=50), SR, place="P")
    old = {pid: (blob, face.person_sealer(pid)) for pid, blob in db.execute("SELECT person_id, template FROM vl_voice")}
    rep = vm.disable()
    assert rep.verified and rep.templates_destroyed == 2 and rep.message == "2 voice templates destroyed, verified"
    assert count(db, "vl_voice") == count(db, "vl_heard") == 0 and optin.opt_in_record(db) is None
    for pid, (blob, dek) in old.items():
        assert dek.open(blob, VOICE_AAD + pid.encode())   # the replaced keys are gone from the store
        assert blob not in Path(db.path).read_bytes()
    assert count(db, "fm_enrolled") == 2 and face_optin.opt_in_record(db) is not None
    assert face.process_frame(synth_face_frame(11), place="P")["matched"] == 1
    assert face.process_frame(synth_face_frame(12), place="P")["matched"] == 1
    with pytest.raises(VoiceLinkLocked):
        vm.process_audio(synth_voice("A", seed=50), SR)
    with pytest.raises(VoiceLinkLocked):
        VoiceLinkModule(db, face, slots(), VoiceLinkConfig(allow_test_backend=True))   # re-enabling needs a new opt-in


def test_wipe_without_face_module_reports_honestly(two):
    db = two.env["db"]
    rep = wipe_voice_data(db, None)
    assert count(db, "vl_voice") == 0 and rep.checks["ciphertext_absent"]
    assert not rep.verified and not rep.checks["undecryptable"]   # their face keys still exist; not claimed


def test_separate_voice_consent_withdrawal_erases_only_the_voice(env):
    face = face_on(env)
    a = enroll_face(face, 11, "Alex", consent="c-face")
    vm = voice_on(env, face)
    enroll_voice(vm, a, "A", consent="c-v-only")
    assert vm.process_audio(synth_voice("A", seed=50), SR)["heard"]["person_id"] == a
    env["consents"].withdraw("c-v-only")
    assert count(env["db"], "vl_voice") == 0 and count(env["db"], "fm_enrolled") == 1
    assert m17_records(env["chain"])[-1]["kind"] == "consent_withdrawn" and m17_records(env["chain"])[-1]["verified"]
    assert vm.process_audio(synth_voice("A", seed=51), SR)["heard"] is None


def test_shared_consent_withdrawal_erases_face_and_voice(env):
    face = face_on(env)
    a = enroll_face(face, 11, "Alex", consent="c-voice")
    vm = voice_on(env, face)
    enroll_voice(vm, a, "A", consent="c-voice")
    env["consents"].withdraw("c-voice")
    assert count(env["db"], "vl_voice") == 0 and count(env["db"], "fm_enrolled") == 0


def test_expired_voice_consent_purged_when_templates_load(env):
    face = face_on(env)
    a = enroll_face(face, 11, "Alex", consent="c-face")
    env["consents"].add(ConsentRecord("c-v-exp", "s", {"voice_enrollment"}, expires_utc=time.time() + 3600))
    vm = voice_on(env, face)
    enroll_voice(vm, a, "A", consent="c-v-exp")
    assert vm.process_audio(synth_voice("A", seed=50), SR)["heard"]   # templates cached in RAM
    env["consents"]._r["c-v-exp"].expires_utc = 1.0                     # expiry, no hook fires
    assert vm.process_audio(synth_voice("A", seed=51), SR)["heard"] is None
    assert count(env["db"], "vl_voice") == 0 and count(env["db"], "fm_enrolled") == 1


# ---- 6. face re-enrollment keeps the voice ---------------------------------------------------------
def test_face_reenrollment_reseals_and_keeps_voice(two):
    db, face = two.env["db"], two.face
    old_blob = db.execute("SELECT template FROM vl_voice WHERE person_id=?", (two.a,)).fetchone()[0]
    old_dek = face.person_sealer(two.a)
    sid = face.reenroll(two.a)
    for i, a in enumerate(REQUIRED_ANGLES):
        assert face.submit_shot(two.a, synth_face_frame(11, a, noise_seed=40 + i), a).ok
    assert face.finish_person(two.a).complete
    face.end_session(sid)
    new_blob = db.execute("SELECT template FROM vl_voice WHERE person_id=?", (two.a,)).fetchone()[0]
    assert new_blob != old_blob
    assert face.person_sealer(two.a).open(new_blob, VOICE_AAD + two.a.encode()) == old_dek.open(old_blob, VOICE_AAD + two.a.encode())
    with pytest.raises(Exception):
        face.person_sealer(two.a).open(old_blob, VOICE_AAD + two.a.encode())
    assert two.vm.process_audio(synth_voice("A", seed=50), SR)["heard"]["person_id"] == two.a


# ---- 7. retention ---------------------------------------------------------------------------------
def test_retention_purges_heard_entries(env):
    face = face_on(env); a = enroll_face(face, 11, "Alex")
    vm = voice_on(env, face, heard_retention_days=1)
    enroll_voice(vm, a, "A")
    t0 = 1_700_000_000
    assert vm.process_audio(synth_voice("A", seed=50), SR, place="P", now=t0)["heard"]
    assert nightly_purge(env["db"], days=30, now=t0 + 3600) == 0
    assert nightly_purge(env["db"], days=30, now=t0 + 2 * 86400) == 1
    assert count(env["db"], "vl_heard") == 0 and count(env["db"], "vl_voice") == 1   # templates kept until withdrawal


# ---- 8. voice and face agree -------------------------------------------------------------------
def test_face_agrees_only_within_window(two):
    t = 1_700_000_000.0
    assert two.face.last_match_at(two.a) is None
    assert two.face.process_frame(synth_face_frame(11), place="P", now=t)["matched"] == 1
    assert two.face.last_match_at(two.a) == t and two.face.last_match_at(two.b) is None
    assert two.vm.process_audio(synth_voice("A", seed=50), SR, place="P", now=t + 3)["heard"]["face_agrees"] is True
    assert two.vm.process_audio(synth_voice("A", seed=51), SR, place="P", now=t + 6)["heard"]["face_agrees"] is False
    assert two.vm.process_audio(synth_voice("B", seed=52), SR, place="P", now=t + 1)["heard"]["face_agrees"] is False
    agrees = [r[0] for r in two.env["db"].execute("SELECT face_agrees FROM vl_heard ORDER BY rowid")]
    assert agrees == [1, 0, 0]


def test_review_describes_a_recent_face_match_after_another_face_appears(two, monkeypatch):
    from oiris.voice_link_cli import run

    t = time.time() - 30.0
    assert two.face.process_frame(synth_face_frame(11), now=t)["matched"] == 1
    assert two.face.process_frame(synth_face_frame(12), now=t + 1)["matched"] == 1
    recent = two.vm.process_audio(synth_voice("A", seed=50), SR, now=t + 2)["heard"]
    older = two.vm.process_audio(synth_voice("A", seed=51), SR, now=t + 6)["heard"]
    assert recent["person_id"] == older["person_id"] == two.a
    assert recent["face_agrees"] and not older["face_agrees"]

    cfg = Config(data_dir=str(two.env["dir"]))
    cfg.flags.update({"M-16": True, "M-17": True})
    monkeypatch.setenv("OIRIS_PASSPHRASE", "pw")
    lines = []
    assert run(cli_args("review"), cfg, two.env["dir"] / "config.json", out=lines.append,
               face_backend=InjectedBackend(), speaker=slots("inj")) == 0
    assert next(line for line in lines if recent["entry_id"] in line).endswith("Recent face match for this person")
    assert "Recent face match for this person" not in next(line for line in lines if older["entry_id"] in line)


# ---- 9. templates of another model -----------------------------------------------------------------
def test_templates_from_another_model_not_used(two):
    vm2 = VoiceLinkModule(two.env["db"], two.face, slots("fake_spectral_v2"), VoiceLinkConfig(allow_test_backend=True))
    assert vm2.process_audio(synth_voice("A", seed=50), SR, place="P")["heard"] is None
    people = vm2.review()["people"]
    assert len(people) == 2 and all(not p["model_current"] and p["note"] == "re-enroll for the current model" for p in people)
    assert all(p["model_current"] and p["note"] == "" for p in two.vm.review()["people"])


def test_review_and_delete_heard(two):
    vm = two.vm
    vm.process_audio(synth_voice("A", seed=50), SR, place="Kitchen")
    vm.process_audio(synth_voice("A", seed=51), SR, place="Garden")
    rev = vm.review()
    alex = next(p for p in rev["people"] if p["person_id"] == two.a)
    assert alex["name"] == "Alex" and alex["windows"] == 3 and [e["place"] for e in alex["entries"]] == ["Kitchen", "Garden"]
    assert b"Kitchen" not in Path(two.env["db"].path).read_bytes()   # places sealed with M-16's log key
    assert vm.delete_heard(alex["entries"][0]["entry_id"]) and not vm.delete_heard("nope")
    assert len(next(p for p in vm.review()["people"] if p["person_id"] == two.a)["entries"]) == 1
    check_output(vm.review())


# ---- 10. config ---------------------------------------------------------------------------------
def test_config_keys_and_defaults_in_sync():
    assert VOICE_LINK_KEYS == set(VoiceLinkConfig.__dataclass_fields__)
    assert asdict(VoiceLinkConfig()) == VOICE_LINK_DEFAULTS
    assert VOICE_LINK_DEFAULTS["match_threshold"] == 0.5 and VOICE_LINK_DEFAULTS["listening"] == "ambient"
    assert Config().voice_link == {} and not validate_voice_link({})


@pytest.mark.parametrize("key,ok,bad", [
    ("listening", ["ambient", "push_to_talk"], ["always", 1]),
    ("match_threshold", [0.2, 0.5, 0.95], [0.19, 0.96, True, "0.5", float("nan")]),
    ("unsure_margin", [0.0, 0.3], [-0.01, 0.31]),
    ("min_speech_s", [0.5, 10], [0.4, 10.5]),
    ("enroll_windows", [3, 10], [2, 11, 2.5]),
    ("enroll_window_s", [3, 15], [2.9, 15.1]),
    ("enroll_min_speech_s", [0.5, 7], [0.4, 7.1]),
    ("enroll_range_min_rise", [0, 0.08, 0.5], [-0.01, 0.51, "0.1"]),
    ("heard_retention_days", [1, 365], [0, 366]),
    ("agree_window_s", [1, 30], [0.9, 31]),
    ("card_seconds", [3, 60], [2.9, 61]),
    ("vad_snr_db", [3, 30], [2.9, 31]),
    ("vad_floor_dbfs", [-80, -20], [-81, -19]),
    ("speaker_model", ["sherpa_campp", "other"], ["", 3]),
    ("input_device", ["", "iPhone", 2], [True, 1.5]),
])
def test_config_ranges(key, ok, bad):
    for v in ok:
        assert not validate_voice_link({key: v}), (key, v)
        VoiceLinkConfig.from_dict({key: v})
    for v in bad:
        assert validate_voice_link({key: v}), (key, v)
        with pytest.raises(ValueError):
            VoiceLinkConfig.from_dict({key: v})


def test_config_rejects_unknown_keys_and_test_backend():
    assert validate_voice_link({"record_audio": True}) and validate_voice_link([])
    assert validate_voice_link({"enroll_window_s": 3, "enroll_min_speech_s": 3.5})
    with pytest.raises(Exception):
        Config(voice_link={"allow_test_backend": True}).validate()
    with pytest.raises(Exception):
        Config(voice_link={"match_threshold": 2}).validate()
    c = Config(voice_link={"listening": "push_to_talk", "match_threshold": 0.6}); c.flags["M-17"] = True
    assert c.validate().enabled("M-17")


def test_privacy_guard_voice_template_and_face_photo():
    with pytest.raises(PrivacyError):
        check_persist("voice_template", True)
    with pytest.raises(PrivacyError):
        check_persist("voice_template", True, source="M-16", m16_enabled=True, m17_enabled=True)
    with pytest.raises(PrivacyError):
        check_persist("voice_template", True, source="M-17", m17_enabled=False)
    with pytest.raises(PrivacyError):
        check_persist("voice_template", True, payload=b"raw", source="M-17", m17_enabled=True)
    check_persist("voice_template", True, source="M-17", m17_enabled=True)
    check_persist("face_photo", True, source="M-16", m16_enabled=True, photo_consent=True)
    check_persist("face_photo", True, source="M-17", m17_enabled=True, photo_consent=True)
    for kw in ({"source": "M-16", "m16_enabled": True}, {"source": "M-17", "m16_enabled": True, "photo_consent": True},
               {"source": "M-03", "m16_enabled": True, "m17_enabled": True, "photo_consent": True},
               {"source": "M-16", "m17_enabled": True, "photo_consent": True}, {}):
        with pytest.raises(PrivacyError):
            check_persist("face_photo", True, **kw)
    for kind in ("audio", "raw_audio", "voice_wav", "face_crop", "face_embedding", "face_id", "face_hash"):
        with pytest.raises(PrivacyError):
            check_persist(kind, True, source="M-17", m16_enabled=True, m17_enabled=True, photo_consent=True)


def test_vad_frames_segments_and_floor():
    x = np.zeros(4 * SR, np.float32)
    rng = np.random.default_rng(0)
    x += rng.normal(0, 1e-3, x.size).astype(np.float32)
    t = np.arange(SR) / SR
    x[SR:2 * SR] += (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)        # 1 s tone
    x[3 * SR:3 * SR + int(0.15 * SR)] += 0.3                                        # 150 ms blip: dropped
    r = vad.detect(x, SR, 10, -55)
    assert len(r.segments) == 1
    a, b = r.segments[0]
    assert abs(a - SR) <= int(vad.FRAME_S * SR) and 2 * SR <= b <= 2 * SR + int((vad.HANGOVER_S + vad.FRAME_S) * SR)
    assert 1.0 <= r.speech_s <= 1.0 + vad.HANGOVER_S + 2 * vad.FRAME_S and r.speech.size == b - a
    assert abs(r.level_dbfs - 20 * np.log10(0.3 / np.sqrt(2))) < 1.5
    quiet = vad.detect((x * 0.001).astype(np.float32), SR, 10, -55)   # below the absolute floor
    assert quiet.segments == [] and quiet.speech_s == 0
    assert vad.detect(np.zeros(100, np.float32), SR, 10, -55).speech_s == 0
    r.zero(); assert not r.speech.any()


# ---- 11. CLI ------------------------------------------------------------------------------------
class FakeSd:
    """Stands in for sounddevice: synthetic windows only. The CLI's microphone check before window 1
    (capture.MIC_CHECK_S long) gets `check_audio` (faint noise by default) and is kept apart in `checks`."""
    def __init__(self, windows=(), check_audio=None):
        self.windows, self.recorded, self.devices_used = list(windows), [], []
        self.check_audio = np.random.default_rng(0).normal(0, 1e-2, SR).astype(np.float32) if check_audio is None else check_audio
        self.checks: list = []

    def query_devices(self):
        return [{"name": "Built-in Microphone", "max_input_channels": 1}, {"name": "Built-in Output", "max_input_channels": 0},
                {"name": "Test iPhone Microphone", "max_input_channels": 1}]

    def rec(self, frames, samplerate, channels, dtype, device):
        if frames == int(round(MIC_CHECK_S * samplerate)):
            chk = np.resize(self.check_audio, frames).reshape(-1, 1).astype(np.float32)
            self.checks.append(chk)
            return chk
        buf = self.windows.pop(0)[:frames].reshape(-1, 1).copy()
        assert channels == 1 and dtype == "float32" and samplerate == SR
        self.recorded.append(buf); self.devices_used.append(device)
        return buf

    def wait(self):
        pass


def enter_skip_redo(prompt: str) -> str:
    """Enter at every window; "s" when a redo is offered (one synthetic pitch for every sentence)."""
    return "s" if "redo" in prompt.lower() else ""


def cli_args(sub, **kw):
    base = {"mode": None, "jurisdiction": "us-other", "camera": None, "id": None, "subject": None, "days": None,
            "withdraw": False, "scope": None, "person": None, "consent": None, "input": None}
    return SimpleNamespace(sub=sub, **{**base, **kw})


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("OIRIS_PASSPHRASE", "pw")
    cfgp = tmp_path / "config.json"
    cfgp.write_text(json.dumps({"data_dir": str(tmp_path), "voice_link": {"enroll_windows": 3, "enroll_min_total_speech_s": 5.0}}))
    return tmp_path, cfgp


def open_store(d):
    ks = KeyStore(d / "keys.json"); ks.unlock("pw")
    db = DB(d / "asb.db", ks.subkey("db"))
    return ks, db, face_optin.DbChain(db, ks.subkey("face-chain"))


def cli_face_person(d, seed=11, name="Alex", consent="c-cli") -> str:
    """Enroll a face directly (the face CLI's camera flow is tested elsewhere)."""
    ks, db, chain = open_store(d)
    try:
        face = FaceMatchModule(db, ks, d, InjectedBackend(), chain, FaceMatchConfig(modes=["personal"]), ConsentRegistry(db))
        return enroll_face(face, seed, name, consent=consent)
    finally:
        db.close()


def test_cli_full_flow(store, monkeypatch):
    from oiris.face_match_cli import run as face_run
    from oiris.voice_link_cli import run as voice_run
    d, cfgp = store
    out: list[str] = []
    sm = lambda: slots("inj", reg=_inj_registry())   # noqa: E731
    # voice link needs face matching first
    assert face_run(cli_args("enable", mode=["personal"]), Config.load(cfgp), cfgp, inp=lambda p: "I UNDERSTAND", out=out.append) == 0
    out.clear()
    assert voice_run(cli_args("enable"), Config.load(cfgp), cfgp, inp=lambda p: "nope", out=out.append) == 1
    assert "not turned on" in out[-1] and json.loads(cfgp.read_text())["flags"].get("M-17") is not True
    assert voice_run(cli_args("enable"), Config.load(cfgp), cfgp, inp=lambda p: "I UNDERSTAND", out=out.append) == 0
    assert optin.WARNING_TEXT in out and json.loads(cfgp.read_text())["flags"]["M-17"] is True
    # consent with both scopes, then a face
    assert face_run(cli_args("consent", subject="S-01", id="c-cli", scope="face_enrollment,voice_enrollment"), Config.load(cfgp), cfgp,
                    inp=lambda p: "YES", out=out.append) == 0
    pid = cli_face_person(d)
    # status
    out.clear()
    assert voice_run(cli_args("status"), Config.load(cfgp), cfgp, out=out.append, speaker=sm()) == 0
    st = json.loads(out[0])
    assert st["flag_on"] and st["opted_in"] and st["listening"] == "ambient" and st["speaker_slot"] == "active"
    assert st["voice_templates"] == 0 and st["heard_entries"] == 0 and st["log_entries"] >= 2 and st["model_file_present"] is False
    # devices (names printed, never stored)
    out.clear()
    assert voice_run(cli_args("devices"), Config.load(cfgp), cfgp, out=out.append, sd=FakeSd()) == 0
    assert "  0: Built-in Microphone" in out and "  2: Test iPhone Microphone" in out and not any("Output" in o for o in out)
    # enroll: one refused (silent) window, then three good ones, from the iPhone microphone chosen by name
    sd = FakeSd([np.zeros(5 * SR, np.float32)] + [synth_voice("A", seed=s) for s in (1, 2, 3)])
    out.clear()
    assert voice_run(cli_args("enroll", person=pid[:6], consent="c-cli", input="iphone"), Config.load(cfgp), cfgp,
                     inp=enter_skip_redo, out=out.append, face_backend=InjectedBackend(), speaker=sm(), sd=sd) == 0
    assert sd.devices_used == [2] * 4 and all(not b.any() for b in sd.recorded)
    assert any("refused: not enough speech" in o for o in out) and sum(o.startswith("  ok:") for o in out) == 3
    assert out[-1] == "Voice template complete for Alex"
    assert any(o.startswith("Window 1/3 (") and o.endswith('"') for o in out)   # the sentence to read
    ks, db, _ = open_store(d)
    assert count(db, "vl_voice") == 1; db.close()
    # review
    out.clear()
    assert voice_run(cli_args("review"), Config.load(cfgp), cfgp, out=out.append, face_backend=InjectedBackend(), speaker=sm()) == 0
    assert out[0].startswith("Alex (id ") and "3 windows" in out[0]
    # delete-voice
    out.clear()
    assert voice_run(cli_args("delete-voice", person=pid[:6]), Config.load(cfgp), cfgp, out=out.append,
                     face_backend=InjectedBackend(), speaker=sm()) == 0
    assert out == ["1 voice template destroyed, verified"]
    assert voice_run(cli_args("delete-voice", person=pid[:6]), Config.load(cfgp), cfgp, out=out.append,
                     face_backend=InjectedBackend(), speaker=sm()) == 1
    assert voice_run(cli_args("delete-heard", id="nope"), Config.load(cfgp), cfgp, out=out.append,
                     face_backend=InjectedBackend(), speaker=sm()) == 0 and out[-1] == "no such entry"
    # enroll again, then disable: verified, face untouched
    sd = FakeSd([synth_voice("A", seed=s) for s in (4, 5, 6)])
    assert voice_run(cli_args("enroll", person=pid[:6], consent="c-cli"), Config.load(cfgp), cfgp, inp=enter_skip_redo, out=out.append,
                     face_backend=InjectedBackend(), speaker=sm(), sd=sd) == 0
    assert sd.devices_used == [None] * 3   # "" = system default
    out.clear()
    assert voice_run(cli_args("disable"), Config.load(cfgp), cfgp, out=out.append, face_backend=InjectedBackend()) == 0
    assert out[-1] == "Voice link is off. 1 voice templates destroyed, verified."
    assert json.loads(cfgp.read_text())["flags"]["M-17"] is False and json.loads(cfgp.read_text())["flags"]["M-16"] is True
    ks, db, chain = open_store(d)
    try:
        assert count(db, "vl_voice") == 0 and count(db, "fm_enrolled") == 1 and optin.opt_in_record(db) is None
        kinds = [r["kind"] for r in m17_records(chain)]
        assert kinds == ["opt_in", "voice_enrolled", "voice_deleted", "voice_enrolled", "disabled_wiped"] and chain.verify()
    finally:
        db.close()
    out.clear()
    assert voice_run(cli_args("review"), Config.load(cfgp), cfgp, out=out.append, face_backend=InjectedBackend(), speaker=sm()) == 2
    assert "Voice link is off" in out[-1]


def _inj_registry() -> Registry:
    r = Registry()
    r.register("speaker_embed", "inj", InjectedSpeaker, port_version="1.0", meta=META)
    return r


def test_cli_enroll_refuses_without_voice_consent_and_unknown_device(store):
    from oiris.face_match_cli import run as face_run
    from oiris.voice_link_cli import run as voice_run
    d, cfgp = store
    face_run(cli_args("enable", mode=["personal"]), Config.load(cfgp), cfgp, inp=lambda p: "I UNDERSTAND", out=[].append)
    voice_run(cli_args("enable"), Config.load(cfgp), cfgp, inp=lambda p: "I UNDERSTAND", out=[].append)
    face_run(cli_args("consent", subject="S-02", id="c-cli"), Config.load(cfgp), cfgp, inp=lambda p: "YES", out=[].append)
    pid = cli_face_person(d)
    out: list[str] = []
    kw = {"face_backend": InjectedBackend(), "speaker": slots("inj", reg=_inj_registry())}
    assert voice_run(cli_args("enroll", person=pid[:6], consent="c-cli"), Config.load(cfgp), cfgp, inp=lambda p: "", out=out.append,
                     sd=FakeSd(), **kw) == 1
    assert "lacks scope voice_enrollment" in out[-1] and "--scope voice_enrollment" in out[-1]
    out.clear()
    assert voice_run(cli_args("enroll", person=pid[:6], consent="c-cli", input="USB headset"), Config.load(cfgp), cfgp,
                     inp=lambda p: "", out=out.append, sd=FakeSd(), **kw) == 1
    assert "no input device matches" in out[-1]
    out.clear()
    assert voice_run(cli_args("enroll", person="zzz", consent="c-cli"), Config.load(cfgp), cfgp, inp=lambda p: "", out=out.append,
                     sd=FakeSd(), **kw) == 1 and out == ["no such person"]


def test_cli_enable_needs_face_matching(store):
    from oiris.voice_link_cli import run as voice_run
    d, cfgp = store
    out: list[str] = []
    assert voice_run(cli_args("enable"), Config.load(cfgp), cfgp, inp=lambda p: "I UNDERSTAND", out=out.append) == 1
    assert "face matching (M-16)" in out[-1]


def test_cli_face_match_disable_erases_voices(store):
    from oiris.face_match_cli import run as face_run
    from oiris.voice_link_cli import run as voice_run
    d, cfgp = store
    face_run(cli_args("enable", mode=["personal"]), Config.load(cfgp), cfgp, inp=lambda p: "I UNDERSTAND", out=[].append)
    voice_run(cli_args("enable"), Config.load(cfgp), cfgp, inp=lambda p: "I UNDERSTAND", out=[].append)
    face_run(cli_args("consent", subject="S-03", id="c-cli", scope="voice_enrollment,face_enrollment"), Config.load(cfgp), cfgp,
             inp=lambda p: "YES", out=[].append)
    pid = cli_face_person(d)
    sd = FakeSd([synth_voice("A", seed=s) for s in (1, 2, 3)])
    assert voice_run(cli_args("enroll", person=pid[:6], consent="c-cli"), Config.load(cfgp), cfgp, inp=enter_skip_redo, out=[].append,
                     face_backend=InjectedBackend(), speaker=slots("inj", reg=_inj_registry()), sd=sd) == 0
    out: list[str] = []
    assert face_run(cli_args("disable"), Config.load(cfgp), cfgp, out=out.append) == 0
    assert "1 templates destroyed, verified. 1 voice templates destroyed; voice link is off." in out[-1]
    flags = json.loads(cfgp.read_text())["flags"]
    assert flags["M-16"] is False and flags["M-17"] is False
    ks, db, chain = open_store(d)
    try:
        assert count(db, "vl_voice") == 0 and count(db, "fm_enrolled") == 0 and optin.opt_in_record(db) is None
        assert m17_records(chain)[-1]["kind"] == "disabled_wiped" and chain.verify()
    finally:
        db.close()


def test_cli_fetch_models_with_fake_backend_module(store, monkeypatch):
    from oiris.voice_link_cli import SHERPA_MODULE, run as voice_run
    d, cfgp = store
    fake = ModuleType(SHERPA_MODULE)

    class ModelMissing(RuntimeError):
        pass

    calls = []

    def fetch_models(model_dir, opener=None):
        calls.append(model_dir)
        if "bad" in str(model_dir):
            raise ModelMissing("checksum mismatch; nothing kept")
        return Path(model_dir) / "speaker.onnx"

    fake.ModelMissing, fake.fetch_models = ModelMissing, fetch_models
    fake.model_path, fake.model_ok = lambda model_dir: Path(model_dir) / "speaker.onnx", lambda p: False   # not present yet
    fake.MODEL = {"file": "speaker.onnx", "url": "https://example.invalid/m", "size": 1, "sha256": "0" * 64,
                  "license": "Apache-2.0", "weights_license": "CC-BY-4.0"}
    monkeypatch.setitem(sys.modules, SHERPA_MODULE, fake)
    out: list[str] = []
    assert voice_run(cli_args("fetch-models"), Config.load(cfgp), cfgp, out=out.append) == 0
    assert calls == [str(d / "models" / "voice_link")]
    assert "speaker.onnx" in out[-1] and "Apache-2.0" in out[-1] and "CC-BY-4.0" in out[-1] and out[-1].endswith("sha256 ok")
    cfg = Config.load(cfgp); cfg.voice_link = {"model_dir": str(d / "bad")}
    out.clear()
    assert voice_run(cli_args("fetch-models"), cfg, cfgp, out=out.append) == 1 and "checksum mismatch" in out[-1]


def test_cli_consent_scopes(store):
    from oiris.face_match_cli import run as face_run
    d, cfgp = store
    out: list[str] = []
    assert face_run(cli_args("consent", subject="S-04", id="c-all", scope="guardian, voice_enrollment,photo,face_enrollment"),
                    Config.load(cfgp), cfgp, inp=lambda p: "nope", out=out.append) == 1
    assert "  scope:        face_enrollment, voice_enrollment, photo, guardian" in out
    for s in ("face_enrollment", "voice_enrollment", "photo", "guardian"):
        assert any(o.startswith(f"  - {s}:") for o in out)
    assert face_run(cli_args("consent", subject="S-04", id="c-all", scope="guardian,voice_enrollment,photo,face_enrollment"),
                    Config.load(cfgp), cfgp, inp=lambda p: "YES", out=out.append) == 0
    for bad in ("", "voice", "face_enrollment,stalk", "everything"):
        out.clear()
        assert face_run(cli_args("consent", subject="S-05", scope=bad), Config.load(cfgp), cfgp,
                        inp=lambda p: pytest.fail("asked for YES with a bad scope"), out=out.append) == 2
        assert "--scope must be" in out[-1]
    face_run(cli_args("consent", subject="S-06", id="c-default"), Config.load(cfgp), cfgp, inp=lambda p: "YES", out=out.append)
    ks, db, _ = open_store(d)
    try:
        reg = ConsentRegistry(db)
        assert reg.get("c-all").scopes == {"face_enrollment", "voice_enrollment", "photo", "guardian"}
        assert reg.get("c-default").scopes == {"face_enrollment"}
    finally:
        db.close()


def test_cli_argument_parsing(monkeypatch, tmp_path):
    from oiris.cli import main
    got: dict = {}
    monkeypatch.setattr("oiris.voice_link_cli.run", lambda a, cfg, p: got.setdefault("a", a) and 0)
    main(["voice-link", "enroll", "--person", "ab12", "--consent", "c-1", "--input", "2", "--config", str(tmp_path / "none.json")])
    assert (got["a"].command, got["a"].sub, got["a"].person, got["a"].consent, got["a"].input) == ("voice-link", "enroll", "ab12", "c-1", "2")
    monkeypatch.setattr("oiris.face_match_cli.run", lambda a, cfg, p: got.__setitem__("f", a) or 0)
    main(["face-match", "consent", "--subject", "S-1", "--scope", "face_enrollment,voice_enrollment", "--config", str(tmp_path / "none.json")])
    assert got["f"].scope == "face_enrollment,voice_enrollment"


# ---- 12. not imported unless used -----------------------------------------------------------------
def test_cli_and_registry_do_not_import_voice_link():
    code = ("import sys; import oiris.cli; from oiris.registry import default_registry; default_registry();"
            "from oiris.config import Config; from oiris.modules import voice_link_gate;"
            "assert voice_link_gate.load(Config(), None, None, None, None) is None;"
            "bad = [k for k in sys.modules if k == 'oiris.modules.voice_link' or k.startswith('oiris.modules.voice_link.')];"
            "assert not bad, bad")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
