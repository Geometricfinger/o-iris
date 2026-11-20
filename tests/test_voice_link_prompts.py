"""M-17 enrollment sentences, pitch measure and advisory range check (docs/VOICE_LINK_MODULE_SPEC.md "Reading
sentences"). Synthetic audio only: harmonic tones, glides and noise. No real voices, no audio files, no model,
no microphone, no network. The fake speaker, fake sounddevice and CLI helpers come from test_voice_link."""
import json, logging, sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import test_voice_link as tvl
from oiris.config import Config
from oiris.guard.consent import ConsentRecord, ConsentRegistry
from oiris.guard.vocabulary import check_output, check_text
from oiris.modules.face_match import optin as face_optin
from oiris.modules.voice_link import pitch, prompts
from oiris.modules.voice_link.capture import MIC_CHECK_S
from oiris.modules.voice_link.core import REDO_CUE, VoiceLinkConfig, VoiceLinkModule
from oiris.security.keys import KeyStore
from oiris.store.db import DB

SR = tvl.SR
TABLE = [  # id, text, cue, tone: the owner's wording, in enrollment order
    ("calm-low", "Mom hums a long, calm tune in the warm room at noon.", "Calmly, a little lower than usual.", "low"),
    ("bright-high", "She sees three green leaves and six shiny seashells.", "Light and bright, a little higher.", "high"),
    ("question", "Do you know how many blue boats are out on the bay today?", "Ask it as a question; let your voice rise at the end.", "rising"),
    ("count", "One, two, three, four, five, six, seven, eight, nine, ten.", "Count at your normal pace.", "steady"),
    ("excited", "Wow, that huge yellow kite is flying high above the park!", "With excitement, a bit louder.", "high"),
    ("normal", "The orange jar of cherries in the beige garage is very heavy.", "Your normal speaking voice.", "normal"),
    ("story-low", "Joy's old wooden toy rolled with a soft thud across the floor.", "Slowly and softly, like telling a quiet story.", "low"),
    ("cheerful", "The black cat and her purple bird sat happily on the mat.", "Cheerfully, at a normal pace.", "normal"),
]
TONE_HZ = {"low": 110.0, "high": 170.0, "rising": 175.0, "steady": 140.0, "normal": 140.0}   # a wide synthetic range


def harmonic(f0, seconds=2.0, sr=SR, f1=None, amp=0.3) -> np.ndarray:
    """A steady (or, with f1, gliding) harmonic tone: 1/k harmonics up to 4 kHz."""
    t = np.arange(int(seconds * sr)) / sr
    f = np.full(t.size, float(f0)) if f1 is None else np.linspace(f0, f1, t.size)
    ph = 2 * np.pi * np.cumsum(f) / sr
    s = sum(np.sin(k * ph + k) / k for k in range(1, int(4000 / max(f0, f1 or f0))))
    return (amp * s / np.max(np.abs(s))).astype(np.float32)


def voice(f0, seconds=5.0, seed=0, amp=0.3) -> np.ndarray:
    """Syllable-like bursts of a harmonic tone at f0 (with slight vibrato) over a faint noise floor."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    x = rng.normal(0, 1e-3, n)
    pos = int(0.4 * SR)
    while pos < n:
        end = min(n, pos + int(rng.uniform(0.35, 0.6) * SR))
        tt = np.arange(end - pos) / SR
        vib = 1 + 0.01 * np.sin(2 * np.pi * rng.uniform(3, 6) * tt)
        s = sum(np.sin(2 * np.pi * k * f0 * tt * vib + rng.uniform(0, 2 * np.pi)) / k for k in range(1, int(4000 / f0)))
        x[pos:end] += amp * s / np.max(np.abs(s)) * np.hanning(end - pos)
        pos = end + int(rng.uniform(0.08, 0.18) * SR)
    return x.astype(np.float32)


def hiss(seconds=5.0, seed=0) -> np.ndarray:
    """Noise bursts: loud enough to count as speech for the VAD, but unvoiced."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1e-3, int(seconds * SR))
    for a in range(int(0.4 * SR), x.size - SR // 2, int(0.6 * SR)):
        x[a:a + int(0.45 * SR)] += rng.normal(0, 0.1, int(0.45 * SR)) * np.hanning(int(0.45 * SR))
    return x.astype(np.float32)


# ---- prompts ------------------------------------------------------------------------------------
def test_prompt_wording_order_and_ids_exact():
    assert [(p.id, p.text, p.cue, p.tone) for p in prompts.PROMPTS] == TABLE
    assert len({p.id for p in prompts.PROMPTS}) == len(prompts.PROMPTS) == 8
    for p in prompts.PROMPTS:
        check_text(p.text); check_text(p.cue)
        assert p.covers and prompts.by_id(p.id) is p
    with pytest.raises(KeyError):
        prompts.by_id("nope")


def test_enrollment_prompts_order_and_cycling():
    ids = [t[0] for t in TABLE]
    assert [p.id for p in prompts.enrollment_prompts(6)] == ids[:6]
    assert [p.id for p in prompts.enrollment_prompts(10)] == ids + ids[:2]
    assert prompts.enrollment_prompts(0) == []


def test_prompt_list_shape_for_other_devices():
    pl = prompts.prompt_list()
    assert pl == [{"id": i, "text": t, "cue": c, "tone": tone} for i, t, c, tone in TABLE]
    check_output(pl)
    json.dumps(pl)


def test_first_six_cover_the_sound_classes_and_tones():
    six = prompts.enrollment_prompts(6)
    covers = {c for p in six for c in p.covers}
    for need in ("nasals", "sibilants", "affricates", "front vowels", "back vowels", "open vowels", "diphthongs",
                 "liquids", "glides", "plosives"):
        assert need in covers, need
    assert {"low", "high", "rising", "steady"} <= {p.tone for p in six}


# ---- pitch --------------------------------------------------------------------------------------
@pytest.mark.parametrize("f0", [100, 220, 350])
def test_pitch_of_steady_tones(f0):
    ps = pitch.estimate(harmonic(f0), SR)
    assert ps is not None and abs(ps.median_hz - f0) <= 0.03 * f0
    assert ps.low_hz <= ps.median_hz <= ps.high_hz and ps.voiced_frames >= pitch.MIN_VOICED_FRAMES


def test_pitch_of_a_glide_spans_the_range():
    ps = pitch.estimate(harmonic(120, seconds=3.0, f1=240), SR)
    assert ps is not None and ps.low_hz < 140 and ps.high_hz > 210


def test_pitch_none_for_noise_silence_and_too_short():
    assert pitch.estimate(np.random.default_rng(0).normal(0, 0.1, 2 * SR).astype(np.float32), SR) is None
    assert pitch.estimate(np.zeros(2 * SR, np.float32), SR) is None
    assert pitch.estimate(harmonic(150, seconds=0.05), SR) is None
    assert pitch.estimate(harmonic(150), 0) is None


def test_pitch_leaves_the_input_unchanged():
    x = harmonic(150)
    before = x.copy()
    pitch.estimate(x, SR)
    assert np.array_equal(x, before)


# ---- core ---------------------------------------------------------------------------------------
@pytest.fixture
def env(tmp_path):
    ks = KeyStore(tmp_path / "keys.json"); ks.create("pw")
    db = DB(tmp_path / "asb.db", ks.subkey("db"))
    chain = face_optin.DbChain(db, ks.subkey("face-chain"))
    consents = ConsentRegistry(db)
    consents.add(ConsentRecord("c-voice", "s-c-voice", {"face_enrollment", "voice_enrollment"}))
    yield {"ks": ks, "db": db, "chain": chain, "consents": consents, "dir": tmp_path}
    db.close()


def vm_with_person(env, windows=3) -> tuple[VoiceLinkModule, str]:
    face = tvl.face_on(env)
    pid = tvl.enroll_face(face, 11, "Alex")
    return tvl.voice_on(env, face, enroll_windows=windows), pid


def submit(vm, pid, hz_by_tone, seed0=1):
    """One window per enrollment sentence, each at the pitch its tone gets in hz_by_tone."""
    out = []
    for i, p in enumerate(prompts.enrollment_prompts(vm.cfg.enroll_windows)):
        r = vm.submit_window(pid, voice(hz_by_tone[p.tone], seed=seed0 + i), SR, prompt_id=p.id)
        assert r.ok, r.reason
        out.append(r)
    return out


def test_new_enrollment_defaults():
    c = VoiceLinkConfig()
    assert (c.enroll_windows, c.enroll_window_s, c.enroll_min_speech_s, c.enroll_range_min_rise) == (6, 7.0, 1.8, 0.08)


def test_windows_report_pitch(env):
    vm, pid = vm_with_person(env)
    vm.start_voice(pid, "c-voice")
    r = vm.submit_window(pid, voice(150, seed=1), SR, prompt_id="calm-low")
    assert r.ok and r.pitch_hz is not None and abs(r.pitch_hz - 150) <= 0.05 * 150
    assert r.pitch_low_hz <= r.pitch_hz <= r.pitch_high_hz
    r = vm.submit_window(pid, hiss(seed=2), SR, prompt_id="bright-high")   # speech-loud but unvoiced
    assert r.ok and r.pitch_hz is None and r.pitch_low_hz is None and r.pitch_high_hz is None
    r = vm.submit_window(pid, voice(150, seed=3), SR)                       # no sentence named: still fine
    assert r.ok and r.pitch_hz is not None and r.done


def test_unknown_prompt_refused_and_zeroed(env):
    vm, pid = vm_with_person(env)
    vm.start_voice(pid, "c-voice")
    buf = voice(150, seed=1)
    r = vm.submit_window(pid, buf, SR, prompt_id="nope")
    assert not r.ok and r.code == "prompt" and r.needed == 3 and not buf.any() and vm.capture_buffer_size() == 0


def test_range_report_ok_when_high_windows_are_higher(env):
    vm, pid = vm_with_person(env, windows=6)
    vm.start_voice(pid, "c-voice")
    submit(vm, pid, TONE_HZ)
    rep = vm.range_report(pid)
    assert rep.ok and rep.measured and rep.redo_prompt is None and rep.high_hz > rep.low_hz
    check_text(rep.message)
    assert vm.finish_voice(pid).ok


def test_range_report_names_a_redo_and_redo_window_rerecords(env):
    vm, pid = vm_with_person(env, windows=3)   # calm-low, bright-high, question
    vm.start_voice(pid, "c-voice")
    submit(vm, pid, {"low": 140, "high": 138, "rising": 145})
    rep = vm.range_report(pid)
    assert not rep.ok and rep.measured and rep.redo_prompt == "bright-high" and rep.redo_cue == REDO_CUE == "a bit higher and livelier"
    check_text(rep.message)
    cap = vm._captures[pid]
    dropped = cap.embeddings[1]
    speech = cap.speech_s
    assert vm.redo_window(pid, "bright-high") == 1
    assert not dropped.any() and vm.capture_buffer_size() == 2 and cap.speech_s < speech
    assert [m.prompt_id for m in cap.marks] == ["calm-low", "question"]
    assert not vm.finish_voice(pid).ok                                   # one window short now
    r = vm.submit_window(pid, voice(185, seed=9), SR, prompt_id="bright-high")
    assert r.ok and r.done
    assert vm.range_report(pid).ok
    assert vm.finish_voice(pid).ok
    assert env["db"].execute("SELECT windows FROM vl_voice WHERE person_id=?", (pid,)).fetchone()[0] == 3


def test_finish_works_when_the_range_check_fails(env):
    vm, pid = vm_with_person(env, windows=3)
    vm.start_voice(pid, "c-voice")
    submit(vm, pid, {"low": 140, "high": 140, "rising": 140})
    assert not vm.range_report(pid).ok
    st = vm.finish_voice(pid)
    assert st.ok and st.done


def test_range_not_measured_is_ok(env):
    vm, pid = vm_with_person(env, windows=3)
    vm.start_voice(pid, "c-voice")
    for s in (1, 2, 3):
        assert vm.submit_window(pid, voice(150, seed=s), SR).ok      # no sentence ids: no tones known
    rep = vm.range_report(pid)
    assert rep.ok and not rep.measured and "not measured" in rep.message
    with pytest.raises(KeyError):
        vm.redo_window(pid, "calm-low")


def test_pitch_never_stored_or_logged(env, caplog):
    caplog.set_level(logging.DEBUG)
    vm, pid = vm_with_person(env, windows=6)
    vm.start_voice(pid, "c-voice")
    submit(vm, pid, TONE_HZ)
    vm.range_report(pid)
    cap = vm._captures[pid]
    assert vm.finish_voice(pid).ok
    assert not cap.marks and not cap.embeddings                         # the RAM capture is dropped
    cols = [r[1] for r in env["db"].execute("PRAGMA table_info(vl_voice)")]
    assert cols == ["person_id", "template", "model_id", "dim", "consent_id", "windows", "speech_s", "enrolled_at"]
    for rec in tvl.m17_records(env["chain"]):
        assert not [k for k in rec if "pitch" in k or "hz" in k.lower() or "prompt" in k or "tone" in k], rec
    assert not [r for r in caplog.records if "pitch" in r.getMessage().lower() or "hz" in r.getMessage().lower()]


def test_cancel_wipes_the_marks(env):
    vm, pid = vm_with_person(env)
    vm.start_voice(pid, "c-voice")
    assert vm.submit_window(pid, voice(150, seed=1), SR, prompt_id="calm-low").ok
    cap = vm._captures[pid]
    vm.cancel_voice(pid)
    assert not cap.marks and not cap.embeddings


# ---- CLI ----------------------------------------------------------------------------------------
@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("OIRIS_PASSPHRASE", "pw")
    cfgp = tmp_path / "config.json"
    cfgp.write_text(json.dumps({"data_dir": str(tmp_path)}))   # default enrollment: 6 sentences of 7 s
    return tmp_path, cfgp


def ready(store) -> str:
    from oiris.face_match_cli import run as face_run
    from oiris.voice_link_cli import run as voice_run
    d, cfgp = store
    face_run(tvl.cli_args("enable", mode=["personal"]), Config.load(cfgp), cfgp, inp=lambda p: "I UNDERSTAND", out=[].append)
    voice_run(tvl.cli_args("enable"), Config.load(cfgp), cfgp, inp=lambda p: "I UNDERSTAND", out=[].append)
    face_run(tvl.cli_args("consent", subject="S-01", id="c-cli", scope="face_enrollment,voice_enrollment"), Config.load(cfgp), cfgp,
             inp=lambda p: "YES", out=[].append)
    return tvl.cli_face_person(d)


class EventSd(tvl.FakeSd):
    def __init__(self, windows, events):
        super().__init__(windows)
        self.events = events

    def rec(self, frames, samplerate, channels, dtype, device):
        self.events.append(("rec", frames))
        return super().rec(frames, samplerate, channels, dtype, device)


def windows_for(hz_by_tone, n=6, seconds=7.0):
    return [voice(hz_by_tone[p.tone], seconds=seconds, seed=i + 1) for i, p in enumerate(prompts.enrollment_prompts(n))]


def run_enroll(store, pid, sd, inp, say=None, **kw):
    from oiris.voice_link_cli import run as voice_run
    d, cfgp = store
    out: list[str] = []
    rc = voice_run(tvl.cli_args("enroll", person=pid[:6], consent="c-cli", **kw), Config.load(cfgp), cfgp, inp=inp, out=out.append,
                   face_backend=tvl.InjectedBackend(), speaker=tvl.slots("inj", reg=tvl._inj_registry()), sd=sd, say=say)
    return rc, out


def test_cli_shows_sentence_and_cue_per_window_with_pitch(store):
    pid = ready(store)
    sd = tvl.FakeSd(windows_for(TONE_HZ))
    asked: list[str] = []
    rc, out = run_enroll(store, pid, sd, lambda p: asked.append(p) or "")
    assert rc == 0 and out[-1] == "Voice template complete for Alex"
    assert len(sd.recorded) == 6 and all(b.size == 7 * SR and not b.any() for b in sd.recorded)
    assert 'Window 2/6 (light and bright, a little higher): "She sees three green leaves and six shiny seashells."' in out
    assert 'Window 1/6 (calmly, a little lower than usual): "Mom hums a long, calm tune in the warm room at noon."' in out
    assert 'Window 6/6 (your normal speaking voice): "The orange jar of cherries in the beige garage is very heavy."' in out
    oks = [o for o in out if o.startswith("  ok:")]
    assert len(oks) == 6 and all("pitch about " in o and " Hz (" in o for o in oks)
    assert any(o.startswith("voice range ok") for o in out) and not any("redo" in a.lower() for a in asked)


def test_cli_pitch_not_measured_feedback(store):
    pid = ready(store)
    sd = tvl.FakeSd([hiss(seconds=7.0, seed=s) for s in range(6)])
    rc, out = run_enroll(store, pid, sd, lambda p: "")
    assert rc == 0 and sum("pitch not measured" in o for o in out) == 6
    assert any("not measured" in o and o.startswith("voice range") for o in out)


def test_cli_offers_one_redo_when_the_range_is_narrow(store):
    pid = ready(store)
    flat = {t: 140.0 for t in TONE_HZ}
    sd = tvl.FakeSd(windows_for(flat) + [voice(190, seconds=7.0, seed=40)])
    asked: list[str] = []
    rc, out = run_enroll(store, pid, sd, lambda p: asked.append(p) or "")   # Enter: redo
    assert rc == 0 and out[-1] == "Voice template complete for Alex"
    assert len(sd.recorded) == 7 and sum("redo" in a.lower() for a in asked) == 1
    redo = [o for o in out if o.startswith("Redo (a bit higher and livelier): ")]
    assert len(redo) == 1 and any(o.endswith(f'"{p.text}"') for p in prompts.PROMPTS if p.tone in ("high", "rising") for o in redo)
    assert any("not much higher" in o for o in out)


def test_cli_redo_can_be_skipped(store):
    pid = ready(store)
    sd = tvl.FakeSd(windows_for({t: 140.0 for t in TONE_HZ}))
    rc, out = run_enroll(store, pid, sd, lambda p: "s" if "redo" in p.lower() else "")
    assert rc == 0 and len(sd.recorded) == 6 and out[-1] == "Voice template complete for Alex"
    assert not any(o.startswith("Redo") for o in out)


def test_cli_redo_q_cancels(store):
    pid = ready(store)
    sd = tvl.FakeSd(windows_for({t: 140.0 for t in TONE_HZ}))
    rc, out = run_enroll(store, pid, sd, lambda p: "q" if "redo" in p.lower() else "")
    assert rc == 1 and out[-1] == "Voice enrollment cancelled; nothing saved."
    ks, db, _ = tvl.open_store(store[0])
    try:
        assert tvl.count(db, "vl_voice") == 0
    finally:
        db.close()


def test_cli_speak_reads_each_sentence_before_its_recording(store):
    pid = ready(store)
    events: list = []
    sd = EventSd(windows_for(TONE_HZ), events)
    rc, out = run_enroll(store, pid, sd, lambda p: "", say=lambda text: events.append(("say", text)), speak=True)
    assert rc == 0
    texts = [p.text for p in prompts.enrollment_prompts(6)]
    check = ("rec", int(round(MIC_CHECK_S * SR)))   # the microphone check before window 1 (nothing spoken)
    assert events == [check] + [e for t in texts for e in (("say", t), ("rec", 7 * SR))]


def test_cli_speak_without_say_prints_a_note_and_continues(store, monkeypatch):
    import oiris.voice_link_cli as vl
    monkeypatch.setattr(vl, "SAY", str(store[0] / "no-such-say"))
    pid = ready(store)
    rc, out = run_enroll(store, pid, tvl.FakeSd(windows_for(TONE_HZ)), lambda p: "", speak=True)
    assert rc == 0 and any(o.startswith("Note:") and "not be read aloud" in o for o in out)


def test_cli_speak_failure_prints_a_note_and_continues(store):
    pid = ready(store)

    def broken_say(text):
        raise OSError("audio busy")

    rc, out = run_enroll(store, pid, tvl.FakeSd(windows_for(TONE_HZ)), lambda p: "", say=broken_say, speak=True)
    assert rc == 0 and sum("could not be read aloud" in o for o in out) == 1


def test_cli_speak_flag_parsed(monkeypatch, tmp_path):
    from oiris.cli import main
    got: dict = {}
    monkeypatch.setattr("oiris.voice_link_cli.run", lambda a, cfg, p: got.setdefault("a", a) and 0)
    main(["voice-link", "enroll", "--person", "ab12", "--speak", "--config", str(tmp_path / "none.json")])
    assert got["a"].speak is True


def _fake_backend(monkeypatch, present: bool) -> list:
    from oiris.voice_link_cli import SHERPA_MODULE
    fake = ModuleType(SHERPA_MODULE)
    calls: list = []

    class ModelMissing(RuntimeError):
        pass

    def fetch_models(model_dir, opener=None):
        calls.append(model_dir)
        return Path(model_dir) / "speaker.onnx"

    fake.ModelMissing, fake.fetch_models = ModelMissing, fetch_models
    fake.model_path = lambda model_dir: Path(model_dir) / "speaker.onnx"
    fake.model_ok = lambda p: present
    fake.MODEL = {"file": "speaker.onnx", "license": "Apache-2.0", "weights_license": "CC-BY-4.0"}
    monkeypatch.setitem(sys.modules, SHERPA_MODULE, fake)
    return calls


@pytest.mark.parametrize("present", [True, False])
def test_cli_fetch_models_wording(store, monkeypatch, present):
    from oiris.voice_link_cli import run as voice_run
    d, cfgp = store
    calls = _fake_backend(monkeypatch, present)
    out: list[str] = []
    assert voice_run(tvl.cli_args("fetch-models"), Config.load(cfgp), cfgp, out=out.append) == 0
    if present:
        assert calls == [] and not any("Downloading" in o for o in out)
        assert out[-1].endswith("already present, sha256 ok, nothing downloaded")
    else:
        assert calls == [str(d / "models" / "voice_link")] and out[0].startswith("Downloading the speaker model")
        assert out[-1].endswith("sha256 ok") and "nothing downloaded" not in out[-1]
