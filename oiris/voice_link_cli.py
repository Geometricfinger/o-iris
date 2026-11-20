"""`oiris voice-link ...` (M-17, ADR-0057): opt-in, voice enrollment, review, deletes, disable + wipe.
Only reached through an explicit `voice-link` command; normal runs never import M-17.

  oiris voice-link status
  oiris voice-link enable                     warning + typed phrase; needs face matching in personal mode
  oiris voice-link fetch-models               download the speaker model (SHA-256 pinned)
  oiris voice-link devices                    list microphones by index and name
  oiris voice-link enroll --person ID-PREFIX [--consent ID] [--input N|NAME] [--speak]
  oiris voice-link review [--days N]
  oiris voice-link delete-heard --id ENTRY | delete-voice --person ID-PREFIX
  oiris voice-link disable                    wipes every voice template, verified; face matching untouched
  oiris voice-link set-self --person ID-PREFIX   the wearer/owner: never announced, never logged (ADR-0058)
  oiris voice-link clear-self
  oiris voice-link test-sound [--name NAME]   plays the chime and "sounds like NAME" once; records nothing

Consent first, per person: oiris face-match consent --subject CODE --scope face_enrollment,voice_enrollment
(add guardian for a minor). The person must already be enrolled in face matching.

Microphones: with an iPhone nearby, macOS also lists the phone's microphone (Continuity), so a phone can be
the microphone: choose it with --input and its index or part of its name (see `devices`). Device names are
printed, never stored. Enrollment audio is recorded into memory only and zeroed after each window.

Before window 1 a short microphone check (about 1.5 s, nothing kept) opens the input, so macOS asks for
microphone permission then and not during a real window; it shows the level, and stops with the fix when the
input is digitally silent. Enrollment keeps adding sentences (in order, cycling) until the accepted windows
hold voice_link.enroll_min_total_speech_s of speech, or the maximum number of windows is reached.

Enrollment shows one designed sentence per window with a cue for how to say it (prompts.py); every capture
device shows the same sentences. --speak has the Mac read each sentence aloud first (/usr/bin/say), for
children who cannot read yet; recording starts only after it has finished speaking.
"""
from __future__ import annotations
import importlib, json, subprocess, sys, time, uuid
from pathlib import Path
from typing import Any, Callable

from .config import Config

SHERPA_MODULE = "oiris.modules.voice_link.backends.sherpa_campp"   # Agent L28's speaker backend (lazy)
ENROLL_SAMPLE_RATE = 16000
MAX_REFUSED_WINDOWS = 5
SAY = "/usr/bin/say"
SAY_TIMEOUT_S = 60
CONSENT_HINT = "record one with: oiris face-match consent --subject CODE --scope voice_enrollment"


def _save_flag(cfg_path: Path, on: bool) -> None:
    raw = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    raw.setdefault("flags", {})["M-17"] = on
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(raw, indent=1), encoding="utf-8")


def _model_dir(cfg: Config) -> str:
    return (cfg.voice_link or {}).get("model_dir") or str(Path(cfg.data_dir).expanduser() / "models" / "voice_link")


def _capture_mod() -> Any:
    """Microphone access lives in oiris/modules/voice_link/capture.py (imported only by voice-link commands)."""
    return importlib.import_module("oiris.modules.voice_link.capture")


def _sounddevice(sd: Any = None) -> Any:
    """sounddevice (or the injected stand-in); ImportError when it is not installed."""
    cap = _capture_mod()
    try:
        return cap.load_sounddevice(sd)
    except cap.CaptureUnavailable as e:
        raise ImportError(str(e)) from e


def _mic(sd: Any) -> Callable[[float, int, Any], Any]:
    """Records one window into RAM through capture.record. Returns a view of sounddevice's buffer, so
    zeroing it zeroes the buffer."""
    cap = _capture_mod()

    def capture(seconds: float, sample_rate: int, device: Any) -> Any:
        return cap.record(seconds, device, sample_rate, sd=sd)
    return capture


def _inputs(sd: Any) -> list[tuple[int, str]]:
    return _capture_mod().inputs(sd)


def _device(value: Any, sd: Any) -> int | None:
    """None = system default; an index; or the first input device whose name contains the text."""
    return _capture_mod().resolve_device(value, sd)


def _person(db, table: str, prefix: str | None) -> tuple[str | None, str]:
    if not prefix:
        return None, "give --person ID-PREFIX (see: oiris face-match review)"
    ids = [r[0] for r in db.execute(f"SELECT person_id FROM {table}") if r[0].startswith(prefix)]
    if not ids:
        return None, "no such person"
    if len(ids) > 1:
        return None, "more than one person matches; give more of the ID"
    return ids[0], ""


def _mac_say(text: str) -> None:
    """Reads `text` aloud and returns when it has finished speaking."""
    subprocess.run([SAY, text], check=True, timeout=SAY_TIMEOUT_S)


def run(a, cfg: Config, cfg_path: Path, inp: Callable[[str], str] = input, out=print, face_backend: Any = None,
        speaker: Any = None, capture: Callable[[float, int, Any], Any] | None = None, sd: Any = None,
        say: Callable[[str], Any] | None = None, player: Any = None) -> int:
    sub = a.sub or "status"
    if sub == "fetch-models":
        return _fetch_models(cfg, out)
    if sub == "test-sound":
        return _test_sound(getattr(a, "name", None), cfg, out, player)
    if sub == "devices":
        try:
            dev = _sounddevice(sd)
        except ImportError:
            out("The sounddevice package is not installed, so microphones cannot be listed."); return 1
        out("Audio input devices (index: name). With an iPhone nearby, macOS lists its microphone too (Continuity).")
        for i, name in _inputs(dev):
            out(f"  {i}: {name}")
        return 0
    from .face_match_cli import _ctx
    from .modules.voice_link import optin
    d, ks, db, chain = _ctx(cfg)
    try:
        cfg = Config.load(cfg_path) if cfg_path.exists() else cfg
        if sub == "status":
            return _status(cfg, db, chain, speaker, out)
        if sub == "enable":
            out(optin.WARNING_TEXT)
            typed = inp(f'\nType "{optin.CONFIRM_PHRASE}" to turn voice link on: ')
            listening = (cfg.voice_link or {}).get("listening", "ambient")
            try:
                optin.record_opt_in(db, chain, typed, listening, device_id=uuid.uuid5(uuid.NAMESPACE_URL, str(d)).hex[:12],
                                    warning_shown=optin.WARNING_TEXT)
            except optin.OptInError as e:
                out(f"Voice link not turned on: {e}"); return 1
            _save_flag(cfg_path, True)
            out(f"Voice link is on (listening: {listening}). Logged. Next: record consent ({CONSENT_HINT}), then "
                "oiris voice-link enroll --person ID. Turn it off any time: oiris voice-link disable")
            return 0
        if sub == "disable":
            return _disable(cfg, cfg_path, db, ks, chain, face_backend, out)
        from .guard.consent import ConsentError
        from .modules import face_match_gate, voice_link_gate
        from .registry import default_registry
        reg = default_registry(cfg.plugins)
        try:
            face = face_match_gate.load(cfg, db, ks, reg, chain=chain, backend=face_backend)
            if face is None:
                out("Face matching is off. Voice link needs it: oiris face-match enable"); return 2
            vm = voice_link_gate.load(cfg, db, ks, reg, face, chain=chain, speaker=speaker)
        except PermissionError as e:
            out(f"Voice link cannot start: {e}"); return 2
        if vm is None:
            out("Voice link is off. Turn it on first: oiris voice-link enable"); return 2
        if sub == "review":
            for p in vm.review(days=getattr(a, "days", None))["people"]:
                note = f"  [{p['note']}]" if p["note"] else ""
                out(f"{p['name']} (id {p['person_id'][:8]}) voice template from {p['windows']} windows{note}")
                for h in p["entries"]:
                    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(h["heard_at"]))
                    agree = "  Recent face match for this person" if h["face_agrees"] else ""
                    out(f"    entry {h['entry_id']}  heard at {h['place']}, {when}  score {h['score']:.2f}{agree}")
            return 0
        if sub == "set-self":
            pid, why = _person(db, "fm_enrolled", getattr(a, "person", None))
            if pid is None:
                out(why); return 1
            vm.set_self(pid)
            out("Self is set: this person's voice is never announced, logged or shown. Undo: oiris voice-link clear-self")
            return 0
        if sub == "clear-self":
            out("Self cleared." if vm.clear_self() else "No self person was set.")
            return 0
        if sub == "delete-heard":
            out("deleted" if vm.delete_heard(getattr(a, "id", None) or "") else "no such entry"); return 0
        if sub == "delete-voice":
            pid, why = _person(db, "vl_voice", getattr(a, "person", None))
            if pid is None:
                out(why); return 1
            rep = vm.delete_voice(pid)
            out(rep.message); return 0 if rep.verified else 1
        if sub == "enroll":
            try:
                return _enroll(vm, db, a, inp, out, capture, sd, say)
            except ConsentError as e:
                out(f"Not enrolled: {e}. To record consent, {CONSENT_HINT}"); return 1
        out(f"unknown voice-link command {sub}"); return 2
    finally:
        db.close()


def _fetch_models(cfg: Config, out) -> int:
    md = _model_dir(cfg)
    try:
        mod = importlib.import_module(SHERPA_MODULE)
    except ImportError as e:
        out(f"The speaker backend is not installed ({e})."); return 1
    m = mod.MODEL
    have = mod.model_path(md)
    if mod.model_ok(have):
        out(f"  {m['file']} ({have}) already present, sha256 ok, nothing downloaded")
        return 0
    out(f"Downloading the speaker model ({m['license']} code, {m['weights_license']} weights; see THIRD_PARTY.md) to {md}")
    try:
        p = Path(mod.fetch_models(md))
    except mod.ModelMissing as e:
        out(f"Not downloaded: {e}"); return 1
    out(f"  {m['file']} ({p}) licence {m['license']}, weights {m['weights_license']}: sha256 ok")
    return 0


def _status(cfg: Config, db, chain, speaker: Any, out) -> int:
    from .modules.voice_link import optin
    from .modules.voice_link.core import SLOT, VoiceLinkConfig, self_person, speaker_slots
    from .registry import default_registry
    vcfg = VoiceLinkConfig.from_dict(cfg.voice_link)
    sm = speaker if speaker is not None else speaker_slots(cfg, vcfg, default_registry(cfg.plugins))
    sm.get(SLOT)
    md = Path(_model_dir(cfg)).expanduser()
    out(json.dumps({"flag_on": cfg.enabled("M-17"), "opted_in": optin.opt_in_record(db) is not None,
                    "listening": vcfg.listening, "announce": vcfg.announce,
                    "self_set": "yes" if self_person(db) is not None else "no",
                    "speaker_model": sm.slots.get(SLOT), "speaker_slot": sm.status.get(SLOT, "empty"),
                    "model_file_present": md.is_dir() and any(md.glob("*.onnx")),
                    "voice_templates": db.execute("SELECT COUNT(*) FROM vl_voice").fetchone()[0],
                    "heard_entries": db.execute("SELECT COUNT(*) FROM vl_heard").fetchone()[0],
                    "log_entries": chain.verify()}, indent=1))
    return 0


def _disable(cfg: Config, cfg_path: Path, db, ks, chain, face_backend: Any, out) -> int:
    """Verified wipe of all voice data; M-16 templates stay. Works even when the speaker model is missing."""
    from .modules.voice_link.core import wipe_voice_data
    face = None
    if cfg.enabled("M-16"):
        from .modules import face_match_gate
        from .registry import default_registry
        try:
            face = face_match_gate.load(cfg, db, ks, default_registry(cfg.plugins), chain=chain, backend=face_backend)
        except Exception as e:  # noqa: BLE001 - the wipe goes ahead without M-16; the report says what it could verify
            out(f"(face matching could not load, so data keys cannot be replaced: {e})")
    rep = wipe_voice_data(db, face, chain)
    _save_flag(cfg_path, False)
    out(f"Voice link is off. {rep.message if rep.verified else f'{rep.templates_destroyed} voice templates removed, wipe NOT verified'}.")
    if not rep.verified:
        out("  checks: " + json.dumps(rep.checks))
    return 0 if rep.verified else 1


def _test_sound(name: str | None, cfg: Config, out, player: Any = None) -> int:
    """Preview: the chime, then "sounds like NAME" when the name passes the same rules. Opens no store,
    records nothing. Plays even when voice_link.announce is off (it says so)."""
    from .modules.voice_link import announce
    from .modules.voice_link.core import VoiceLinkConfig
    mode = VoiceLinkConfig.from_dict(cfg.voice_link).announce
    ok = announce.speakable_name(name) if name else None
    text = announce.phrase(ok) if ok else None
    if mode != announce.OFF:
        note = f"voice_link.announce is {mode}"
    else:
        note = "voice_link.announce is off, so live mode plays nothing"
    if name and ok is None:
        out("That name is not spoken (at most 4 words and 40 characters; letters of one script, spaces, hyphens and "
            "apostrophes; no guarded words). Live mode would play the chime only.")
    played = announce.play(player if player is not None else announce.platform_player(), text)
    what = f'the chime and "{text}"' if text else "the chime"
    out(f"Played {what} ({note})." if played else f"Could not play {what}; see the log ({note}).")
    return 0 if played else 1


def _cue(text: str) -> str:
    """A prompt's cue as shown inside the window line: lower-case first letter, no final full stop."""
    return (text[:1].lower() + text[1:]).rstrip(".")


def _pitch_note(r) -> str:
    if r.pitch_hz is None:
        return "pitch not measured"
    return f"pitch about {r.pitch_hz:.0f} Hz ({r.pitch_low_hz:.0f}-{r.pitch_high_hz:.0f})"


def _speaker_voice(a, say: Callable[[str], Any] | None, out) -> Callable[[str], Any] | None:
    """The function that reads a sentence aloud for --speak, or None (with a note) when it cannot."""
    if not getattr(a, "speak", False):
        return None
    if say is not None:
        return say
    if not Path(SAY).exists():
        out(f"Note: {SAY} is not available, so the sentences will not be read aloud; read them to the person instead.")
        return None
    return _mac_say


def _enroll(vm, db, a, inp, out, capture: Callable[[float, int, Any], Any] | None, sd: Any,
            say: Callable[[str], Any] | None = None) -> int:
    from .config import VOICE_ENROLL_MAX_WINDOWS
    from .modules.voice_link.core import VoiceLinkUnavailable
    from .modules.voice_link.prompts import by_id, enrollment_prompts
    pid, why = _person(db, "fm_enrolled", getattr(a, "person", None))
    if pid is None:
        out(why); return 1
    want = getattr(a, "input", None) or vm.cfg.input_device
    dev_mod = None
    if capture is None or want not in (None, ""):
        try:
            dev_mod = _sounddevice(sd)
        except ImportError:
            out("The sounddevice package is not installed, so the microphone cannot be used."); return 1
    try:
        device = _device(want, dev_mod)
    except ValueError as e:
        out(str(e)); return 1
    grab = capture or _mic(dev_mod)
    speak = _speaker_voice(a, say, out)
    cid = getattr(a, "consent", None) or inp("Consent record ID (with scope voice_enrollment): ").strip()
    try:
        needed = vm.start_voice(pid, cid)
    except (KeyError, VoiceLinkUnavailable) as e:
        out(f"Not enrolled: {e}"); return 1
    total, secs, target = needed, vm.cfg.enroll_window_s, vm.cfg.enroll_min_total_speech_s
    plan = enrollment_prompts(VOICE_ENROLL_MAX_WINDOWS)
    refused = 0
    shown = 0   # new sentences recorded (a redo repeats one)

    def record(p, label: str, cue: str) -> int | None:
        """Records windows for prompt `p` until one is accepted. Returns the windows still needed, or None
        when cancelled or refused too often (nothing saved)."""
        nonlocal refused, speak
        while True:
            if refused >= MAX_REFUSED_WINDOWS:
                out("Too many windows were refused; nothing saved. Try a quieter room or another microphone (--input)."); return None
            out(f'{label} ({cue}): "{p.text}"')
            if inp(f"  press Enter, then read it aloud; recording runs {secs:g} s (q to cancel) ").strip().lower() == "q":
                out("Voice enrollment cancelled; nothing saved."); return None
            if speak is not None:
                try:
                    speak(p.text)   # returns when the Mac has finished speaking; recording starts after
                except (OSError, subprocess.SubprocessError) as e:
                    out(f"  Note: the sentence could not be read aloud ({e}); read it to the person instead.")
                    speak = None
            r = vm.submit_window(pid, grab(secs, ENROLL_SAMPLE_RATE, device), ENROLL_SAMPLE_RATE, prompt_id=p.id)
            level = "n/a" if r.level_dbfs is None else f"{r.level_dbfs:.0f} dBFS"
            if r.ok:
                out(f"  ok: {r.speech_s:.1f} s of speech, total {r.total_speech_s:.1f} of {target:g} s, level {level}, {_pitch_note(r)}")
                return r.needed
            refused += 1
            out(f"  refused: {r.reason} ({r.speech_s:.1f} s of speech, level {level}); "
                f"{MAX_REFUSED_WINDOWS - refused} retries left")
            if r.code == "enough":
                return r.needed

    def next_sentence() -> int | None:
        nonlocal shown
        p = plan[shown]
        shown += 1
        label = f"Window {shown}/{total}" if shown <= total else f"Extra window {shown} (for {target:g} s of speech in total)"
        return record(p, label, _cue(p.cue))

    try:
        cap = _capture_mod()
        out(f"Microphone check ({cap.MIC_CHECK_S:g} s, nothing is kept): say a few words now.")
        chk = cap.mic_check(lambda s, rate: grab(s, rate, device), cap.MIC_CHECK_S, ENROLL_SAMPLE_RATE)
        if not chk.ok:
            out(f"  {chk.message}. Enrollment not started; nothing saved."); return 1
        out(f"  {chk.message}")
        out(f"Read each sentence aloud in the way shown, then keep talking naturally until the {secs:g} s window ends.")
        while needed > 0 and shown < total:
            left = next_sentence()
            if left is None:
                return 1
            needed = left
        rep = vm.range_report(pid)
        out(rep.message)
        if not rep.ok and rep.redo_prompt:
            p = by_id(rep.redo_prompt)
            ans = inp(f'Redo "{p.text}" {rep.redo_cue}? Enter to redo, s to skip (q to cancel) ').strip().lower()
            if ans == "q":
                out("Voice enrollment cancelled; nothing saved."); return 1
            if ans != "s":
                vm.redo_window(pid, p.id)
                if record(p, "Redo", rep.redo_cue) is None:
                    return 1
        while True:
            st = vm.finish_voice(pid)
            out(st.reason)
            if st.ok or st.needed <= 0 or shown >= len(plan):
                return 0 if st.ok else 1
            if next_sentence() is None:   # still short of windows or of total speech: the next sentence in order
                return 1
    finally:
        vm.cancel_voice(pid)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(1)
