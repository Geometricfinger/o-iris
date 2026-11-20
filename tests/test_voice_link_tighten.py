"""M-17 tightening after the first live check (docs/VOICE_LINK_MODULE_SPEC.md, section 7 step 6): the
microphone check before window 1, the total-speech target enforced in the core, and the 2.0 s matching
minimum. Synthetic audio, a fake capture and the fake speaker only; no microphone, model or network."""
import json
from dataclasses import asdict

import numpy as np
import pytest

from oiris.config import (VOICE_ENROLL_MAX_WINDOWS, VOICE_LINK_DEFAULTS, VOICE_LINK_KEYS, VOICE_LINK_RANGES, Config, ConfigError,
                          validate_voice_link)
from oiris.modules.voice_link import capture as vcap
from oiris.modules.voice_link.core import VoiceLinkConfig
from oiris.modules.voice_link.prompts import enrollment_prompts
from tests import test_voice_link as tvl
from tests.test_voice_link import SR, FakeSd, count, enroll_face, face_on, synth_voice, voice_on
from tests.test_voice_link import env as env  # noqa: F401  (fixture reuse)

TARGET = 20.0
LOUD_NOISE, QUIET_NOISE = 1e-2, 1e-4     # about -40 and -80 dBFS


def noise(sigma: float, seconds: float = vcap.MIC_CHECK_S, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, sigma, int(round(seconds * SR))).astype(np.float32)


# ---- microphone check (capture.mic_check) ---------------------------------------------------------
def grabber(audio: np.ndarray, log: list):
    def grab(seconds, sample_rate):
        log.append((seconds, sample_rate))
        return audio
    return grab


def test_mic_check_digital_silence_gives_the_permission_fix_and_zeroes():
    buf, log = np.zeros(int(vcap.MIC_CHECK_S * SR), np.float32), []
    r = vcap.mic_check(grabber(buf, log))
    assert not r.ok and r.status == vcap.MIC_CHECK_SILENT and r.level_dbfs is None
    assert vcap.MIC_SETTINGS in r.message and "Terminal" in r.message and "--input" in r.message
    assert log == [(vcap.MIC_CHECK_S, SR)]


def test_mic_check_quiet_warns_and_passes():
    buf = noise(QUIET_NOISE)
    r = vcap.mic_check(grabber(buf, []))
    assert r.ok and r.status == vcap.MIC_CHECK_QUIET and r.level_dbfs < vcap.MIC_QUIET_DBFS
    assert "move closer or speak up" in r.message and "dBFS" in r.message
    assert not buf.any()


def test_mic_check_normal_input_passes_with_level_and_zeroes():
    buf = noise(LOUD_NOISE)
    r = vcap.mic_check(grabber(buf, []))
    assert r.ok and r.status == vcap.MIC_CHECK_OK and abs(r.level_dbfs - 20 * np.log10(LOUD_NOISE)) < 1.0
    assert r.message == f"microphone ok (level {r.level_dbfs:.0f} dBFS)"
    assert not buf.any()


def test_mic_check_zeroes_a_converted_int16_buffer_too():
    buf = (noise(LOUD_NOISE) * 32767).astype(np.int16)
    assert vcap.mic_check(grabber(buf, [])).ok and not buf.any()


def test_mic_check_threshold_is_named_and_injectable():
    buf = noise(LOUD_NOISE)
    assert vcap.mic_check(grabber(buf, []), quiet_dbfs=-20.0).status == vcap.MIC_CHECK_QUIET
    assert vcap.MIC_QUIET_DBFS == -60.0 and vcap.MIC_CHECK_S == 1.5


# ---- CLI enroll -------------------------------------------------------------------------------------
@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("OIRIS_PASSPHRASE", "pw")
    return tmp_path


def ready(d, **voice) -> tuple:
    from oiris.face_match_cli import run as face_run
    from oiris.voice_link_cli import run as voice_run
    cfgp = d / "config.json"
    cfgp.write_text(json.dumps({"data_dir": str(d), "voice_link": {"enroll_windows": 3, **voice}}))
    face_run(tvl.cli_args("enable", mode=["personal"]), Config.load(cfgp), cfgp, inp=lambda p: "I UNDERSTAND", out=[].append)
    voice_run(tvl.cli_args("enable"), Config.load(cfgp), cfgp, inp=lambda p: "I UNDERSTAND", out=[].append)
    face_run(tvl.cli_args("consent", subject="S-09", id="c-cli", scope="face_enrollment,voice_enrollment"), Config.load(cfgp), cfgp,
             inp=lambda p: "YES", out=[].append)
    return cfgp, tvl.cli_face_person(d)


def enroll(d, cfgp, pid, sd) -> tuple[int, list[str]]:
    from oiris.voice_link_cli import run as voice_run
    out: list[str] = []
    rc = voice_run(tvl.cli_args("enroll", person=pid[:6], consent="c-cli"), Config.load(cfgp), cfgp, inp=tvl.enter_skip_redo,
                   out=out.append, face_backend=tvl.InjectedBackend(), speaker=tvl.slots("inj", reg=tvl._inj_registry()), sd=sd)
    return rc, out


def stored(d) -> int:
    ks, db, _ = tvl.open_store(d)
    try:
        return count(db, "vl_voice")
    finally:
        db.close()


def test_cli_digital_silence_stops_before_window_1(store):
    cfgp, pid = ready(store)
    sd = FakeSd([synth_voice("A", seed=s) for s in (1, 2, 3)], check_audio=np.zeros(SR, np.float32))
    rc, out = enroll(store, cfgp, pid, sd)
    assert rc == 1 and len(sd.checks) == 1 and sd.recorded == [] and len(sd.windows) == 3   # no window opened
    fix = next(o for o in out if vcap.MIC_SETTINGS in o)
    assert "allow the app running O Iris (for example Terminal)" in fix and "--input" in fix and "nothing saved" in fix
    assert not any(o.startswith("Window ") for o in out) and stored(store) == 0


def test_cli_quiet_input_warns_and_continues(store):
    cfgp, pid = ready(store)
    sd = FakeSd([synth_voice("A", seed=s) for s in range(1, 8)], check_audio=noise(QUIET_NOISE, seconds=1.0))
    rc, out = enroll(store, cfgp, pid, sd)
    assert rc == 0 and stored(store) == 1
    warn = next(i for i, o in enumerate(out) if "move closer or speak up" in o)
    assert "dBFS" in out[warn] and warn < next(i for i, o in enumerate(out) if o.startswith("Window 1/3"))
    assert len(sd.checks) == 1 and not sd.checks[0].any()   # the check's audio is zeroed, never kept


def test_cli_normal_input_shows_level_and_check_comes_first(store):
    cfgp, pid = ready(store)
    sd = FakeSd([synth_voice("A", seed=s) for s in range(1, 8)], check_audio=noise(LOUD_NOISE, seconds=1.0))
    rc, out = enroll(store, cfgp, pid, sd)
    assert rc == 0
    i = next(i for i, o in enumerate(out) if o.startswith("Microphone check (1.5 s"))
    assert out[i + 1].startswith("  microphone ok (level -") and out[i + 1].endswith(" dBFS)")
    assert i < next(i for i, o in enumerate(out) if o.startswith("Window 1/3"))
    assert not sd.checks[0].any() and all(not b.any() for b in sd.recorded)


# ---- total-speech target (core) -----------------------------------------------------------------------
@pytest.fixture
def vm_pid(env):
    face = face_on(env)
    pid = enroll_face(face, 11, "Alex")
    vm = voice_on(env, face, enroll_min_total_speech_s=TARGET)   # the default target, three windows
    return vm, pid, env


def short_window(seed: int) -> np.ndarray:
    return synth_voice("A", seed=seed, max_speech=2.0)   # about 2.6-2.8 s of speech


def test_short_windows_make_finish_ask_for_more_speech(vm_pid):
    vm, pid, store_env = vm_pid
    assert vm.start_voice(pid, "c-voice") == 3
    results = [vm.submit_window(pid, short_window(s), SR) for s in (1, 2, 3)]
    assert all(r.ok for r in results)
    total = results[-1].total_speech_s
    assert abs(total - sum(r.speech_s for r in results)) < 0.1 and total < TARGET
    assert not results[-1].done and results[-1].needed == 1      # three windows, but short of the target
    st = vm.finish_voice(pid)
    assert not st.ok and st.code == "speech_total" and st.needed == 1
    assert st.reason == f"still needed: {TARGET - total:.1f} s more speech (total {total:.1f} of 20 s); read another sentence"
    assert count(store_env["db"], "vl_voice") == 0


def test_target_reached_finishes_without_a_note(vm_pid):
    vm, pid, store_env = vm_pid
    vm.start_voice(pid, "c-voice")
    for s in range(1, 6):   # about 4.4 s each: 22 s after five
        r = vm.submit_window(pid, synth_voice("A", seed=s), SR)
        assert r.ok and r.done == (s == 5)
    assert vm.submit_window(pid, synth_voice("A", seed=9), SR).code == "enough"
    st = vm.finish_voice(pid)
    assert st.ok and st.reason == "Voice template complete for Alex" and count(store_env["db"], "vl_voice") == 1


def test_at_the_maximum_finish_accepts_below_the_target(env):
    face = face_on(env)
    pid = enroll_face(face, 11, "Alex")
    vm = voice_on(env, face, enroll_min_total_speech_s=60.0)
    vm.start_voice(pid, "c-voice")
    for s in range(1, VOICE_ENROLL_MAX_WINDOWS + 1):
        r = vm.submit_window(pid, synth_voice("A", seed=s), SR)
        assert r.ok and r.done == (s == VOICE_ENROLL_MAX_WINDOWS)
        if s < VOICE_ENROLL_MAX_WINDOWS:
            assert not vm.finish_voice(pid).ok
    assert vm.submit_window(pid, synth_voice("A", seed=50), SR).code == "enough"   # no 11th window
    st = vm.finish_voice(pid)
    assert st.ok and st.reason.startswith("Voice template complete for Alex (")
    assert f"from {VOICE_ENROLL_MAX_WINDOWS} windows, below the 60 s target)" in st.reason
    assert count(env["db"], "vl_voice") == 1


def test_redo_keeps_the_total_right(vm_pid):
    vm, pid, store_env = vm_pid
    vm.start_voice(pid, "c-voice")
    for s, p in zip(range(1, 6), enrollment_prompts(5)):
        r = vm.submit_window(pid, synth_voice("A", seed=s), SR, prompt_id=p.id)
    assert r.done
    assert vm.redo_window(pid, "excited") == 1                   # dropping a window falls below 20 s again
    r = vm.submit_window(pid, synth_voice("A", seed=1, max_speech=1.5), SR, prompt_id="excited")   # about 2.0 s
    assert r.ok and r.total_speech_s < TARGET and not r.done
    assert vm.finish_voice(pid).code == "speech_total"


# ---- total-speech target (CLI) ------------------------------------------------------------------------
def window_lines(out: list[str]) -> list[str]:
    return [o for o in out if o.startswith(("Window ", "Extra window "))]


def test_cli_adds_sentences_in_order_until_the_target(store):
    cfgp, pid = ready(store)   # three windows, the default 20 s target
    sd = FakeSd([synth_voice("A", seed=s) for s in range(1, 9)], check_audio=noise(LOUD_NOISE, seconds=1.0))
    rc, out = enroll(store, cfgp, pid, sd)
    assert rc == 0 and stored(store) == 1 and len(sd.recorded) == 5   # 4.4 s each: 22 s after five
    lines = window_lines(out)
    plan = enrollment_prompts(5)
    assert len(lines) == 5 and all(line.endswith(f'"{p.text}"') for line, p in zip(lines, plan))
    assert lines[0].startswith("Window 1/3 (") and lines[3].startswith("Extra window 4 (for 20 s of speech in total)")
    oks = [o for o in out if o.startswith("  ok:")]
    assert len(oks) == 5 and ", total 4.5 of 20 s," in oks[0] and ", total 22.2 of 20 s," in oks[-1]
    assert any(o.startswith("still needed: ") and "s more speech (total 13.3 of 20 s)" in o for o in out)
    assert out[-1] == "Voice template complete for Alex"


def test_cli_cycles_the_sentences_and_stops_at_the_maximum(store):
    cfgp, pid = ready(store, enroll_min_total_speech_s=60.0)
    sd = FakeSd([synth_voice("A", seed=s) for s in range(1, 13)], check_audio=noise(LOUD_NOISE, seconds=1.0))
    rc, out = enroll(store, cfgp, pid, sd)
    assert rc == 0 and stored(store) == 1 and len(sd.recorded) == VOICE_ENROLL_MAX_WINDOWS
    lines = window_lines(out)
    plan = enrollment_prompts(VOICE_ENROLL_MAX_WINDOWS)   # past the eighth sentence the list starts again
    assert len(lines) == VOICE_ENROLL_MAX_WINDOWS and all(line.endswith(f'"{p.text}"') for line, p in zip(lines, plan))
    assert plan[8] == plan[0]
    assert out[-1].startswith("Voice template complete for Alex (") and "from 10 windows, below the 60 s target)" in out[-1]


# ---- the 2.0 s matching minimum -----------------------------------------------------------------------
def test_matching_minimum_skips_1_8_s_and_processes_2_2_s(env):
    face = face_on(env)
    a, b = enroll_face(face, 11, "Alex"), enroll_face(face, 12, "Blake")
    vm = voice_on(env, face)
    tvl.enroll_voice(vm, a, "A"); tvl.enroll_voice(vm, b, "B")
    assert vm.cfg.min_speech_s == 2.0
    spy = vm.speaker.get("speaker_embed")
    n = len(spy.seen)
    r = vm.process_audio(synth_voice("A", seed=2, max_speech=1.35), SR, place="P")   # 1.8 s of speech
    assert r["speech_s"] == pytest.approx(1.8, abs=0.05) and r["heard"] is None and len(spy.seen) == n   # never embedded
    r = vm.process_audio(synth_voice("A", seed=1, max_speech=1.7), SR, place="P")    # 2.2 s of speech
    assert r["speech_s"] == pytest.approx(2.2, abs=0.05) and len(spy.seen) == n + 1
    assert r["heard"] is not None and r["heard"]["person_id"] == a


# ---- config ---------------------------------------------------------------------------------------------
def test_new_defaults():
    assert VOICE_LINK_DEFAULTS["min_speech_s"] == 2.0 and VOICE_LINK_DEFAULTS["enroll_min_total_speech_s"] == 20.0
    c = VoiceLinkConfig()
    assert (c.min_speech_s, c.enroll_min_total_speech_s) == (2.0, 20.0)
    assert VOICE_ENROLL_MAX_WINDOWS == 10 and VOICE_LINK_RANGES["enroll_windows"][1] == VOICE_ENROLL_MAX_WINDOWS


def test_voice_link_config_stays_in_sync():
    assert VOICE_LINK_KEYS == set(VoiceLinkConfig.__dataclass_fields__)
    assert asdict(VoiceLinkConfig()) == VOICE_LINK_DEFAULTS


@pytest.mark.parametrize("ok", [5, 5.0, 20.0, 64.5, 120])
def test_total_speech_range_accepts(ok):
    assert not validate_voice_link({"enroll_min_total_speech_s": ok})
    assert VoiceLinkConfig.from_dict({"enroll_min_total_speech_s": ok}).enroll_min_total_speech_s == ok


@pytest.mark.parametrize("bad", [4.9, 120.1, 0, -20, True, "20", float("nan"), float("inf"), None])
def test_total_speech_range_refuses(bad):
    assert validate_voice_link({"enroll_min_total_speech_s": bad}) == ["voice_link.enroll_min_total_speech_s must be in [5.0, 120.0]"]
    with pytest.raises(ValueError):
        VoiceLinkConfig.from_dict({"enroll_min_total_speech_s": bad})


def test_config_file_rejects_a_bad_target(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"data_dir": str(tmp_path), "voice_link": {"enroll_min_total_speech_s": 200}}))
    with pytest.raises(ConfigError, match="enroll_min_total_speech_s must be in"):
        Config.load(p)
