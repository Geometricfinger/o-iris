#!/usr/bin/env python3
"""Generate shared vectors the Kotlin core (android/core, JVM-checked) must reproduce:
  tests/vectors/ble_framing.json   - BLE GATT segmenting of Noise frames (oiris/link/ble.py)
  tests/vectors/thermal_render.json - thermal grid -> grayscale / colour pixels (oiris/ui/glasses.thermal_image)
  tests/vectors/eye_android.json   - blinks + dark-pupil diameter (oiris/features/eye.py)
  tests/vectors/voice_link.json    - M-17 decision rules, names, repeat limit, energy VAD (oiris/modules/voice_link)
  tests/vectors/glasses_hud.json   - glasses display lists: HUD states and enrollment guide (oiris/ui/glasses)
  tests/vectors/time_sync.json     - clock offset/delay, sample selection, drift and conversion (oiris/link/timesync)
  tests/vectors/link_ops.json      - glasses link ops: request builders, pod replies, reply parsers, hostile HUD
                                     frames (oiris/link/ops.py, oiris/link/sideload.py)
  tests/vectors/voice_enrollment.json - phone-guided voice enrollment: request builders, real pod replies (temporary
                                     store, deterministic fake speaker, integer synthetic PCM), reply parsers, scripted
                                     flows (oiris/modules/face_match/remote.py voice ops)
  tests/vectors/fusion.json        - Fusion v2: Score() clamping and refusal, family values, logit, fuse output,
                                     seeded random score sets, weights JSON (oiris/fusion/stack.py, oiris/schema.py)
  tests/vectors/power_governor.json - power governor timelines (thresholds, hysteresis, dwell, stale, charging,
                                     thermal status), config validation, the power_sim scenes' readings, FrameGate
                                     and ListenDuty (oiris/power/governor.py, duty.py, oiris/adapters/power_sim.py)
  tests/vectors/hud_paint.json     - draw calls for HUD / guide display lists and parsed hostile HUD frames, float
                                     floor division, colour parsing (oiris/ui/paint.py)
usage: make_android_vectors.py [--check]"""
from __future__ import annotations
import copy, hashlib, json, math, random, sys, tempfile, uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# --check compares numbers within FLOAT_TOL, the Kotlin checks' own tolerance (Checks.kt VL_TOL): the last bits of
# floating-point results differ between CPUs (x86 vs ARM vector instructions), so a file generated on one machine
# must still count as current on another. Strings, booleans, integers and the JSON structure are compared exactly.
FLOAT_TOL = 1e-6
MAX_REPORTED = 5   # differing paths printed for a stale file
sys.path.insert(0, str(ROOT))
from oiris.features.eye import blinks, pupil_diameter_px  # noqa: E402
from oiris.link.ble import segment  # noqa: E402
from oiris.link import timesync as tsync  # noqa: E402
from oiris.link.pod_service import PodService  # noqa: E402
from oiris.link import ops as LO, sideload as SL  # noqa: E402
from oiris.security.keys import KeyStore  # noqa: E402
from oiris.security.panic import PanicController  # noqa: E402
from oiris.guard.phrases import SLOT_VALUES  # noqa: E402
from oiris.guard.vocabulary import VocabularyViolation  # noqa: E402
from oiris.ui import tokens as T  # noqa: E402
from oiris.ui.glasses import (COORD_DECIMALS, CUE_EDGES, CUE_NONE, CUE_THICK_STEP, GUIDE_ORDER, OP_KEYS,  # noqa: E402
                              TICK_POINTS, TICK_SIZE, EnrollGuide,
                              GlassesState, LayoutError, display_list, enroll_guide_display_list, moving_from_motion,
                              thermal_image)
from oiris.ui import paint as P  # noqa: E402
from oiris.config import VOICE_LINK_DEFAULTS  # noqa: E402
from oiris.modules.face_match.port import unit  # noqa: E402
from oiris.modules.voice_link import announce, vad  # noqa: E402
from oiris.modules.voice_link.core import decide  # noqa: E402
from oiris.fusion import stack as FS  # noqa: E402
from oiris.schema import Health, Level, Score  # noqa: E402
from oiris.power import duty as DUTY, governor as PG  # noqa: E402
from oiris.adapters import power_sim as PSIM  # noqa: E402


def ble() -> dict:
    cases = []
    for n, mtu, seq0 in [(1, 23, 0), (17, 23, 250), (300, 23, 7), (1000, 247, 0), (4000, 185, 255)]:
        frame = bytes((i * 31 + 7) % 256 for i in range(n))
        cases.append({"frame": frame.hex(), "mtu": mtu, "seq0": seq0, "segments": [s.hex() for s in segment(frame, mtu, seq0)]})
    return {"cases": cases}


def thermal() -> dict:
    rng = np.random.default_rng(42)
    cases = []
    for h, w in [(4, 4), (6, 8), (24, 32)]:
        f = np.round(rng.normal(25, 6, (h, w)), 2)
        g = thermal_image(f, False)
        c = thermal_image(f, True)
        cases.append({"h": h, "w": w, "celsius": f.ravel().tolist(), "gray": g.ravel().tolist(), "rgb": c.reshape(-1, 3).ravel().tolist()})
    flat = np.full((3, 3), 30.0)
    cases.append({"h": 3, "w": 3, "celsius": flat.ravel().tolist(), "gray": thermal_image(flat, False).ravel().tolist(),
                  "rgb": thermal_image(flat, True).reshape(-1, 3).ravel().tolist()})
    return {"cases": cases}


def eye() -> dict:
    rng = np.random.default_rng(7)
    bl = []
    for fps in (30.0, 60.0):
        o = np.ones(int(fps * 6))
        for start, dur in [(int(fps), 0.12), (int(2.5 * fps), 0.03), (int(4 * fps), 0.8), (int(5 * fps), 0.3)]:
            o[start:start + max(1, int(dur * fps))] = 0.05
        o = np.clip(o + rng.normal(0, 0.02, o.size), 0, 1).round(3)
        bl.append({"fps": fps, "openness": o.tolist(), "blinks": [list(map(int, b)) for b in blinks(o, fps)]})
    pu = []
    for r in (3, 6, 9):
        y, x = np.mgrid[:32, :40]
        roi = np.full((32, 40), 170.0) + rng.normal(0, 4, (32, 40))
        roi[(y - 16) ** 2 + (x - 20) ** 2 <= r * r] = 30
        roi = np.clip(roi, 0, 255).astype(int)
        d, q = pupil_diameter_px(roi.astype(float))
        pu.append({"h": 32, "w": 40, "roi": roi.ravel().tolist(), "diameter": d, "quality": q})
    return {"blinks": bl, "pupil": pu}


# ---- M-17 voice link (android/core VoiceLink.kt) ------------------------------------------------------
VL_RATE = 16000
VL_FRAME = 480                 # vad.FRAME_S at VL_RATE: envelope edges sit on frame edges
VL_MARGIN_DB = 1.0             # every frame level is at least this far from the speech threshold
VL_BED = [[97.0, 0.0006], [233.0, 0.0004], [3001.0, 0.0003]]   # faint, about -66 dBFS
VL_TONES = [[220.0, 0.2], [440.0, 0.1], [1330.0, 0.05]]        # loud, about -15 dBFS
VL_REPEAT_S = 60.0


def vl_signal(rate: int, samples: int, bed: list, tones: list, envelope: list) -> np.ndarray:
    """Sum of sines: the bed always, the tones times a piecewise-constant gain ([start, end) samples).
    VoiceLink checks (Checks.kt) generate the same audio from the same parameters."""
    t = np.arange(samples) / rate
    x = np.zeros(samples)
    for f, a in bed:
        x += a * np.sin(2 * np.pi * f * t)
    env = np.zeros(samples)
    for start, end, gain in envelope:
        env[start:end] = gain
    s = np.zeros(samples)
    for f, a in tones:
        s += a * np.sin(2 * np.pi * f * t)
    return (x + env * s).astype(np.float32)


def vl_vad_case(name: str, frames: int, spans: list, extra: int = 0, rate: int = VL_RATE, bed: list | None = None,
                tones: list | None = None, gain: float = 1.0, snr_db: float = VOICE_LINK_DEFAULTS["vad_snr_db"],
                floor_dbfs: float = VOICE_LINK_DEFAULTS["vad_floor_dbfs"]) -> dict:
    frame = int(round(vad.FRAME_S * rate))
    p = {"name": name, "rate": rate, "samples": frames * frame + extra, "bed": VL_BED if bed is None else bed,
         "tones": VL_TONES if tones is None else tones, "envelope": [[a * frame, b * frame, gain] for a, b in spans],
         "snr_db": snr_db, "floor_dbfs": floor_dbfs}
    x = vl_signal(p["rate"], p["samples"], p["bed"], p["tones"], p["envelope"])
    levels, _ = vad.frame_levels(x, rate)
    if levels.size:
        thr = max(float(np.percentile(levels, vad.FLOOR_PERCENTILE)) + snr_db, floor_dbfs)
        margin = float(np.min(np.abs(levels - thr)))
        assert margin >= VL_MARGIN_DB, f"{name}: frame level only {margin:.3f} dB from the threshold"
    r = vad.detect(x, rate, snr_db, floor_dbfs)
    p["segments"] = [[a, b] for a, b in r.segments]
    p["speech_s"] = r.speech_s
    return p


def voice_link() -> dict:
    D = VOICE_LINK_DEFAULTS
    constants = {"match_threshold": D["match_threshold"], "unsure_margin": D["unsure_margin"],
                 "spoof_threshold": D["spoof_threshold"], "vad_snr_db": D["vad_snr_db"], "vad_floor_dbfs": D["vad_floor_dbfs"],
                 "announce_repeat_s": D["announce_repeat_s"], "name_max_words": announce.NAME_MAX_WORDS,
                 "name_max_chars": announce.NAME_MAX_CHARS, "phrase_prefix": announce.PHRASE_PREFIX,
                 "frame_s": vad.FRAME_S, "hangover_s": vad.HANGOVER_S, "min_segment_s": vad.MIN_SEGMENT_S,
                 "floor_percentile": vad.FLOOR_PERCENTILE, "power_eps": vad.POWER_EPS, "silence_dbfs": vad.SILENCE_DBFS}

    # cosine against a stored template, and the unit-mean template, on small fixed embeddings (dim 8)
    e = {"a1": [1, 0.2, 0, 0.1, 0, 0, 0.3, 0], "a2": [0.9, 0.3, 0.1, 0, 0, 0.1, 0.2, 0], "a3": [1.2, 0.1, 0, 0.2, 0.1, 0, 0.4, 0],
         "b1": [0, 0.1, 1, 0, 0.5, 0, 0, 0.2], "b2": [0.1, 0, 0.8, 0.1, 0.6, 0, 0, 0.3], "neg": [-1, -0.2, 0, -0.1, 0, 0, -0.3, 0],
         "zero": [0] * 8, "big": [300, -120, 45, 0, 7, 0, 0, 1]}
    emb = {k: [float(x) for x in v] for k, v in e.items()}
    cos, tpl = [], []
    for group in (["a1", "a2", "a3"], ["b1", "b2"], ["a1"], ["big", "a1"]):
        t = unit(np.mean(np.stack([unit(np.array(emb[k])) for k in group]), axis=0)).astype(np.float32)
        tpl.append({"embeddings": [emb[k] for k in group], "template": [float(v) for v in t]})
        for q in ("a1", "a2", "b1", "neg", "zero", "big"):
            cos.append({"embedding": emb[q], "template": [float(v) for v in t], "cosine": float(np.dot(unit(np.array(emb[q])), t))})

    # decide: every outcome, in the core's order
    th, mg, st = D["match_threshold"], D["unsure_margin"], D["spoof_threshold"]
    dec = []
    for name, scores, self_id, spoof in [
        ("clear match", [(0.82, "p-a"), (0.31, "p-b")], None, None),
        ("below threshold", [(0.49, "p-a"), (0.2, "p-b")], None, None),
        ("at threshold", [(0.5, "p-a")], None, None),
        ("close call", [(0.80, "p-a"), (0.77, "p-b")], None, None),
        ("margin met", [(0.80, "p-a"), (0.70, "p-b")], None, None),
        ("unordered input", [(0.1, "p-c"), (0.3, "p-b"), (0.9, "p-a")], None, None),
        ("tie ranks the larger id first", [(0.9, "p-a"), (0.9, "p-b")], None, None),
        ("close call below threshold is none", [(0.45, "p-a"), (0.44, "p-b")], None, None),
        ("self", [(0.9, "p-a"), (0.2, "p-b")], "p-a", None),
        ("self is someone else", [(0.9, "p-a"), (0.2, "p-b")], "p-b", None),
        ("close call before self", [(0.9, "p-a"), (0.88, "p-b")], "p-a", None),
        ("spoofed", [(0.9, "p-a"), (0.2, "p-b")], None, 0.9),
        ("spoof at threshold", [(0.9, "p-a")], None, 0.5),
        ("spoof below threshold", [(0.9, "p-a")], None, 0.49),
        ("self before spoof", [(0.9, "p-a")], "p-a", 0.99),
        ("no spoof slot", [(0.9, "p-a")], None, None),
        ("single enrolled voice", [(0.61, "p-only")], None, None),
        ("single enrolled voice below", [(0.3, "p-only")], None, None),
        ("nobody enrolled", [], None, None),
    ]:
        outcome, pid = decide(list(scores), th, mg, self_id, spoof, st)
        dec.append({"name": name, "scores": [[s, p] for s, p in scores], "match_threshold": th, "unsure_margin": mg,
                    "self_id": self_id, "spoof_score": spoof, "spoof_threshold": st, "outcome": outcome, "person_id": pid})

    # names (announce.speakable_name / phrase)
    names = []
    for n in ["Alex", "Mary Jane van Dyke", "Mary Jane van der Berg", "A" * announce.NAME_MAX_CHARS,
              "A" * (announce.NAME_MAX_CHARS + 1), "Robot Smith", "ROBOTS", "Bot-Smith", "Abbott", "Dead Man", "Bot́",
              "Alex2", "R2D2", "Alеx", "Αlex", "山田太郎", "山田さくら",
              "ヤマダ", "김민준", "金민준", "Anne-Marie", "O'Brien", "O’Brien",
              "Anne--Marie", "-Anne", "'Alex", "José", "José", "Иван Петров",
              "Νίκος", "محمد", "प्रिया",
              "Ｓａｍ", "Ｓam", "ʻOlelo", "Alex  Smith", "Alex ", " Alex", "", "Alex!", "Dr. Smith",
              "Alex\tSmith", "Alex \U0001f600", None]:
        ok = announce.speakable_name(n)
        names.append({"name": n, "speakable": ok, "phrase": None if ok is None else announce.phrase(ok)})

    # repeat limit (announce.Announcer: the same person at most once per repeat_s; RAM only)
    rep = []
    for seq in [[("p-a", 0.0), ("p-a", 10.0), ("p-b", 10.0), ("p-a", 59.999), ("p-a", 60.0), ("p-b", 69.0), ("p-b", 70.0)],
                [("p-a", 5.0), ("p-b", 5.0), ("p-c", 5.0), ("p-a", 64.0), ("p-a", 65.0), ("p-a", 125.0), ("p-a", 1000.0)]]:
        clock = [0.0]
        a = announce.Announcer(announce.CHIME, VL_REPEAT_S, announce.NullPlayer(), clock=lambda: clock[0], queue_jobs=len(seq))
        steps = []
        for pid, t in seq:
            clock[0] = t
            steps.append({"person_id": pid, "t": t, "allowed": a.announce(pid, "x")})
        a.close()
        rep.append({"repeat_s": VL_REPEAT_S, "steps": steps})

    # energy VAD: generator parameters (sums of sines, piecewise gains), frame-aligned with clear margins
    vads = [
        vl_vad_case("one phrase", 100, [(20, 60)]),
        vl_vad_case("two phrases", 100, [(10, 30), (50, 70)]),
        vl_vad_case("gap bridged by the hangover", 100, [(10, 30), (34, 50)]),
        vl_vad_case("gap of exactly the hangover", 100, [(10, 30), (37, 50)]),
        vl_vad_case("gap one frame past the hangover", 100, [(10, 30), (38, 50)]),
        vl_vad_case("click dropped", 100, [(40, 45)]),
        vl_vad_case("one frame short of the minimum", 100, [(40, 48)]),
        vl_vad_case("exactly the minimum", 100, [(40, 49)]),
        vl_vad_case("speech from the start", 100, [(0, 20)]),
        vl_vad_case("speech to the end", 100, [(80, 100)]),
        vl_vad_case("trailing partial frame", 100, [(20, 60)], extra=123),
        vl_vad_case("quiet speech under the absolute floor", 100, [(20, 60)], bed=[], gain=0.002),
        vl_vad_case("quiet speech over a lower absolute floor", 100, [(20, 60)], bed=[], gain=0.002, floor_dbfs=-80.0),
        vl_vad_case("digital silence", 50, [], bed=[]),
        vl_vad_case("mostly speech: the floor rises", 100, [(5, 100)]),
        vl_vad_case("loud bed, SNR too small", 100, [(20, 60)], bed=[[150.0, 0.03], [2700.0, 0.02]], gain=0.4, snr_db=12.0),
        vl_vad_case("loud bed, SNR enough", 100, [(20, 60)], bed=[[150.0, 0.03], [2700.0, 0.02]], gain=0.4, snr_db=6.0),
        vl_vad_case("8 kHz", 80, [(10, 40), (55, 70)], rate=8000),
        vl_vad_case("shorter than a frame", 0, [], extra=300),
    ]
    return {"constants": constants, "embeddings": emb, "cosine": cos, "template": tpl, "decide": dec, "names": names,
            "repeat": rep, "vad": vads}


# ---- Glasses HUD display list (android/core GlassesHud.kt) ------------------------------------------------
GH_FRAME = np.zeros((2, 2))      # any frame: the display list carries only the G3 placeholder, no pixels
GH_HEALTH = {"thermal": "ok", "ble": "degraded", "radar": "down", "camera": "offline"}


def gh_state(**kw) -> dict:
    """A vector state: GlassesState fields, with thermal as a bool (a frame is present) and health as an object."""
    st = {"level": None, "alert_text": "", "new_sighting": False, "thermal": False, "thermal_colour": False, "health": {},
          "voice_tier": "low", "locked": False, "panic_armed": False, "moving": False, "cue_direction": None}
    st.update(kw)
    return st


def gh_tokens(patch: list) -> dict:
    tok = copy.deepcopy(T.load())
    for path, value in patch:
        d = tok
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = value
    return tok


def gh_result(build, tok: dict) -> dict:
    try:
        return {"ops": build(tok)}
    except (LayoutError, VocabularyViolation) as e:
        return {"error": type(e).__name__, "message": str(e)}


def glasses_hud() -> dict:
    six = "Cold object near the front door"
    hud = [
        ("nothing shown", gh_state(), []),
        ("voice tier only, empty", gh_state(voice_tier=""), []),
        ("empty level is no level", gh_state(level=""), []),
        ("unknown level", gh_state(level="severe"), []),
        ("all zones", gh_state(level="medium", alert_text="Cold object near door", new_sighting=True, thermal=True,
                               health=dict(GH_HEALTH), locked=True, panic_armed=True, voice_tier="high"), []),
        ("thermal colour overlay", gh_state(thermal=True, thermal_colour=True), []),
        ("health: every status, unsorted names", gh_state(health={"z": "ok", "a": "down", "m": "degraded", "b": "ok"}), []),
        ("health: nine sensors", gh_state(health={f"s{i}": ("ok" if i % 3 else "degraded") for i in range(9)}), []),
        ("health: code-point order and escapes", gh_state(health={"Ａ": "ok", "\U0001f600 cam": "down", "x<y": "a&b"}), []),
        ("locked only", gh_state(locked=True), []),
        ("panic armed only", gh_state(panic_armed=True), []),
        ("new sighting while moving", gh_state(new_sighting=True, moving=True), []),
        ("exactly the word limit", gh_state(level="low", alert_text=six), []),
        ("one word over the limit", gh_state(level="low", alert_text=six + " now"), []),
        ("over the limit while moving: hidden, not checked", gh_state(level="low", alert_text=six + " now", moving=True), []),
        ("banned word", gh_state(level="low", alert_text="robot detected"), []),
        ("banned word while moving: hidden, not checked", gh_state(level="low", alert_text="robot detected", moving=True), []),
        ("unicode whitespace splits words", gh_state(alert_text=" a b c d　e\x1ff\u0085"), []),
        ("unicode whitespace: one word over", gh_state(alert_text="a b c d　e\tf\u0085g"), []),
        ("zero-width space is not a word break", gh_state(alert_text="a​b c d e f g"), []),
        ("markup and unicode text kept raw", gh_state(alert_text="Objet <froid> & \"près\" → \U0001f525"), []),
        ("long single word is not truncated", gh_state(alert_text="Supercalifragilistic" * 3), []),
        ("larger canvas", gh_state(level="confirmed", alert_text="Heat source", new_sighting=True, thermal=True,
                                   health={"radio": "ok"}), [(["glasses", "canvas"], [800, 480])]),
        ("brightness and type from tokens", gh_state(level="low", alert_text="Heat source"),
         [(["glasses", "default_brightness"], 0.6), (["type", "glasses_body"], 24), (["type", "font_family"], "serif")]),
        ("brightness rounded to the named decimals", gh_state(), [(["glasses", "default_brightness"], 0.123456)]),
        ("tighter word limit", gh_state(alert_text="Cold object near door"), [(["glasses", "max_words_alert_line"], 3)]),
        ("centre not clear", gh_state(), [(["glasses", "centre_clear_fraction"], 0.99)]),
        ("thermal window outside the canvas", gh_state(), [(["glasses", "thermal_window"], [500, 230, 180, 135])]),
        ("thermal window over G5", gh_state(), [(["glasses", "thermal_window"], [440, 250, 180, 135])]),
        ("fractional canvas", gh_state(), [(["glasses", "canvas"], [640.5, 400])]),
    ]
    for lv in T.LEVELS:
        for moving in (False, True):
            hud.append((f"level {lv}, moving {moving}", gh_state(level=lv, alert_text="Signatures A1 A3", moving=moving), []))
    # Direction cues: every phrase direction word, still and moving (the marker stays, the G2 text hides while moving)
    for word in sorted(SLOT_VALUES["direction"]):
        for moving in (False, True):
            hud.append((f"cue {word}, moving {moving}", gh_state(level="medium", alert_text="Infrared emitter " + word,
                                                                 moving=moving, cue_direction=word), []))
    hud += [
        ("cue alone", gh_state(cue_direction="at left"), []),
        ("cue with every zone", gh_state(level="confirmed", alert_text="Heat source", new_sighting=True, thermal=True,
                                         health=dict(GH_HEALTH), locked=True, panic_armed=True, cue_direction="behind"), []),
        ("empty cue is no cue", gh_state(cue_direction=""), []),
        ("unknown cue direction", gh_state(cue_direction="north"), []),
        ("cue on a larger canvas", gh_state(cue_direction="at right"), [(["glasses", "canvas"], [800, 480])]),
        ("cue centred on an odd canvas", gh_state(cue_direction="above"), [(["glasses", "canvas"], [641, 401])]),
        ("cue sizes from the spacing scale", gh_state(cue_direction="below"), [(["space_px"], [2, 6, 10, 48])]),
        ("fractional cue size", gh_state(cue_direction="at left"), [(["space_px"], [4, 8.5, 32])]),
        ("cue reaching the clear centre", gh_state(cue_direction="at left"), [(["space_px"], [4, 200, 32])]),
    ]
    cases = []
    for name, s, patch in hud:
        st = GlassesState(**{**s, "thermal": GH_FRAME if s["thermal"] else None})
        cases.append({"name": name, "state": s, "tokens_patch": [[p, v] for p, v in patch],
                      **gh_result(lambda tok: display_list(st, tok), gh_tokens(patch))})

    every = list(GUIDE_ORDER)
    guide = []
    for i in range(len(every) + 1):          # each step: the next angle, with and without the tick
        for last_ok in ((None, True) if i == 0 else (True, False)):
            reason = "" if last_ok is not False else "move closer"
            guide.append((f"step {i}, last {last_ok}", {"needed": every[i:], "last_ok": last_ok, "last_reason": reason,
                                                        "complete": i == len(every)}, []))
    guide += [
        ("rejected, reason truncated", {"needed": every, "last_ok": False, "complete": False,
                                        "last_reason": "face too small, move closer, hold still, more light please"}, []),
        ("rejected, empty reason shows the angle", {"needed": ["up", "down"], "last_ok": False, "last_reason": "", "complete": False}, []),
        ("rejected, reason only commas", {"needed": ["down"], "last_ok": False, "last_reason": ", ,", "complete": False}, []),
        ("rejected, banned word", {"needed": ["left"], "last_ok": False, "last_reason": "fake face", "complete": False}, []),
        ("out-of-order needed", {"needed": ["down", "left"], "last_ok": None, "last_reason": "", "complete": False}, []),
        ("arrow counts as a word", {"needed": ["right"], "last_ok": None, "last_reason": "", "complete": False},
         [(["glasses", "max_words_alert_line"], 3)]),
        ("no arrow at front", {"needed": ["front"], "last_ok": None, "last_reason": "", "complete": False},
         [(["glasses", "max_words_alert_line"], 3)]),
        ("reason cut to a tighter limit", {"needed": ["front"], "last_ok": False, "last_reason": "move closer, face too small",
                                           "complete": False}, [(["glasses", "max_words_alert_line"], 2)]),
    ]
    gcases = []
    for name, g, patch in guide:
        eg = EnrollGuide(list(g["needed"]), g["last_ok"], g["last_reason"], g["complete"])
        gcases.append({"name": name, "guide": g, "tokens_patch": [[p, v] for p, v in patch],
                       **gh_result(lambda tok: enroll_guide_display_list(eg, tok), gh_tokens(patch))})

    motion = [{"motion": m, "moving": moving_from_motion(m)} for m in ("still", "walking", "fast", "unknown", "", None, "Walking")]
    rounding = [[x, round(x, COORD_DECIMALS)] for x in (0.125, 0.135, 0.375, 2.675, 1.005, -0.125, 0.35, 382.0, 1e-9, 123456.789)]
    constants = {"coord_decimals": COORD_DECIMALS, "op_keys": {k: list(v) for k, v in OP_KEYS.items()},
                 "tick_points": [list(p) for p in TICK_POINTS], "tick_size": TICK_SIZE, "guide_order": list(GUIDE_ORDER),
                 "cue_edges": dict(CUE_EDGES), "cue_none": CUE_NONE, "cue_thick_step": CUE_THICK_STEP,
                 "direction_words": sorted(SLOT_VALUES["direction"])}
    return {"constants": constants, "rounding": rounding, "hud": cases, "guide": gcases, "motion": motion}


# ---- Clock sync between devices (android/core TimeSync.kt) -------------------------------------------------
TS_SEED = 17
TS_BOOT = 3 * 86_400 * 10**9            # host uptime at the first exchange: realistic monotonic magnitudes
TS_TURNAROUND = (200_000, 2_000_000)    # device t2 - t1, ns
TS_LEG = (7_500_000, 15_000_000)        # one BLE leg, ns: one or two 7.5 ms connection events


def ts_simulate(n: int, offset_ns: int, drift_ppm: int, period_ns: int, seed: int = TS_SEED) -> list[list[int]]:
    """Seeded exchanges against a remote clock remote = host + offset + drift_ppm * (host - TS_BOOT) / 1e6 (integer
    ns), with random leg delays; tests/test_timesync.py simulates the same way."""
    rng = random.Random(seed)

    def remote(h: int) -> int:
        return h + offset_ns + drift_ppm * (h - TS_BOOT) // tsync.PPM

    out = []
    for k in range(n):
        t0 = TS_BOOT + k * period_ns
        h1 = t0 + rng.randint(*TS_LEG)
        h2 = h1 + rng.randint(*TS_TURNAROUND)
        t3 = h2 + rng.randint(*TS_LEG)
        out.append([t0, remote(h1), remote(h2), t3])
    return out


def ts_session(name: str, exchanges: list, probes: list, conversions: list, **params) -> dict:
    s = tsync.TimeSync(**params)
    accepted = [s.add(*e) is None for e in exchanges]
    b = s.best()
    conv = []
    for t_remote, at in conversions:
        try:
            conv.append({"t_remote": t_remote, "at_host": at, "t_host": s.remote_to_host(t_remote, at)})
        except tsync.TimeSyncError:  # python-brain: allow SILENT_EXCEPTION - a refused conversion is recorded as t_host None; Kotlin must refuse it too
            conv.append({"t_remote": t_remote, "at_host": at, "t_host": None})
    return {"name": name, "params": {"max_samples": s.samples.maxlen, "max_delay_ns": s.max_delay_ns, "stale_ns": s.stale_ns},
            "exchanges": exchanges, "accepted": accepted, "best": None if b is None else [b.t0, b.t1, b.t2, b.t3],
            "drift_ppm": s.drift_ppm(), "quality": [{"now": t, "quality": s.quality(t)} for t in probes], "conversions": conv}


def time_sync() -> dict:
    B, ms, s = TS_BOOT, 1_000_000, 1_000_000_000
    R = 40 * 86_400 * s                   # the device booted long before the host: a large positive offset
    ex = [  # name, t0, t1, t2, t3
        ("symmetric, positive offset", B, B + R + 10 * ms, B + R + 11 * ms, B + 21 * ms),
        ("symmetric, negative offset", B + R, B + 10 * ms, B + 11 * ms, B + R + 21 * ms),
        ("same clock", B, B + 5 * ms, B + 6 * ms, B + 11 * ms),
        ("odd sum, positive: floor", 0, 1000, 1000, 1001),
        ("odd sum, negative: floor away from zero", 1000, 0, 0, 1001),
        ("sum of -1: floor is -1, not 0", 0, 0, 1, 2),
        ("odd negative sum, small", 10, 5, 5, 11),
        ("asymmetric legs: slow uplink", B, B + R + 40 * ms, B + R + 41 * ms, B + 46 * ms),
        ("asymmetric legs: slow downlink", B, B + R + 5 * ms, B + R + 6 * ms, B + 46 * ms),
        ("zero delay", B, B + 7, B + 9, B + 2),
        ("exactly the maximum delay", B, B + R, B + R, B + tsync.MAX_DELAY_NS),
        ("rejected: one ns over the maximum", B, B + R, B + R, B + tsync.MAX_DELAY_NS + 1),
        ("rejected: negative delay", B, B + R, B + R + 30 * ms, B + 20 * ms),
        ("rejected: device replied before it received", B, B + R + 11 * ms, B + R + 10 * ms, B + 21 * ms),
        ("rejected: host received before it sent", B + 21 * ms, B + R + 10 * ms, B + R + 10 * ms, B),
        ("rejected: negative timestamp", -1, 5, 6, 10),
        ("rejected: timestamp at the bound", 0, tsync.MAX_TIMESTAMP_NS, tsync.MAX_TIMESTAMP_NS, 10),
        ("largest timestamps in range", tsync.MAX_TIMESTAMP_NS - 30, 0, 1, tsync.MAX_TIMESTAMP_NS - 1),
    ]
    exchanges = [{"name": n, "t0": a, "t1": b, "t2": c, "t3": d, "offset_ns": tsync.offset_ns(a, b, c, d),
                  "delay_ns": tsync.delay_ns(a, b, c, d), "reject": tsync.reject_reason(a, b, c, d)} for n, a, b, c, d in ex]

    def ex4(t0: int, up: int, turn: int, down: int, off: int) -> list[int]:
        return [t0, t0 + up + off, t0 + up + turn + off, t0 + up + turn + down]

    best_of = [ex4(B + k * s, up, ms, down, R) for k, (up, down) in
               enumerate([(30 * ms, 25 * ms), (12 * ms, 9 * ms), (8 * ms, 40 * ms), (5 * ms, 4 * ms), (20 * ms, 6 * ms)])]
    ties = [ex4(B, 5 * ms, ms, 5 * ms, R), ex4(B + s, 4 * ms, ms, 6 * ms, R + 3), ex4(B + 2 * s, 9 * ms, ms, 9 * ms, R)]
    mixed = best_of[:2] + [[B, B + R, B + R + 30 * ms, B + 20 * ms], ex4(B + 9 * s, ms, ms, ms, R)[:3] + [B + 9 * s + 400 * ms]]
    window = [ex4(B + k * s, up, ms, up, R + k) for k, up in enumerate([2 * ms, 9 * ms, 8 * ms, 7 * ms, 6 * ms, 10 * ms])]
    drift = ts_simulate(48, R, 40, 20 * s)
    Rn = -(B // 2)                        # the device booted after the host (its clock stays positive)
    drift_neg = ts_simulate(30, Rn, -25, 20 * s, seed=TS_SEED + 1)
    short = ts_simulate(20, R, 40, 100 * ms)
    few = ts_simulate(tsync.DRIFT_MIN_SAMPLES - 1, R, 40, 5 * s)
    last = drift[-1][3]
    sessions = [
        ts_session("unsynced", [], [B], [[B + R, B]]),
        ts_session("best of N", best_of, [best_of[-1][3], best_of[-1][3] + tsync.STALE_NS, best_of[-1][3] + tsync.STALE_NS + 1],
                   [[B + R + 5 * s, B + 5 * s], [R, 0]]),
        ts_session("delay tie goes to the most recent", ties, [], [[B + R, B]]),
        ts_session("rejected samples are not kept", mixed, [mixed[1][3]], [[B + R + 3 * s, B + 3 * s]]),
        ts_session("all rejected", [mixed[2], mixed[3]], [B + 9 * s], [[B + R, B]]),
        ts_session("window drops the best", window, [], [[B + R + 6 * s, B + 6 * s]], max_samples=4),
        ts_session("drift over many samples", drift, [last, last + tsync.STALE_NS + 1],
                   [[B + R + k * 7 * s, B + k * 7 * s] for k in range(0, 20, 3)] + [[B + R + 600 * s, B + 600 * s]]),
        ts_session("negative drift and offset", drift_neg, [drift_neg[-1][3]],
                   [[B + Rn + 10 * s, B + 10 * s], [B + Rn + 400 * s, B + 400 * s]]),
        ts_session("span too short for drift", short, [], [[B + R + s, B + s]]),
        ts_session("too few samples for drift", few, [], [[B + R + 60 * s, B + 60 * s]]),
        ts_session("tighter delay limit", best_of, [], [[B + R, B]], max_delay_ns=15 * ms),
        ts_session("shorter stale age", best_of, [best_of[-1][3] + s, best_of[-1][3] + s + 1], [], stale_ns=s),
    ]
    # pod replies: the real op (PodService.handle with a fixed pod clock), then malformed or refused ones
    t0 = B + 5 * s
    replies = []
    for t1, t2 in [(B + R + 5 * s + 9 * ms, B + R + 5 * s + 10 * ms), (B + R + 5 * s + 9 * ms, B + R + 5 * s + 9 * ms)]:
        ticks = iter([t1, t2])
        replies.append(("pod reply", PodService(clock=lambda: next(ticks)).handle(tsync.request(t0)), t0, t0 + 20 * ms))
    replies.append(("pod reply, too slow", replies[0][1], t0, t0 + tsync.MAX_DELAY_NS + 2 * ms))
    replies.append(("pod refusal", PodService().handle('{"op":"time_sync"}'), t0, t0 + 20 * ms))
    for name, text in [("another request's t0", f'{{"ok":true,"t0":{t0 + 1},"t1":5,"t2":6}}'),
                       ("t1 not an integer", f'{{"ok":true,"t0":{t0},"t1":5.0,"t2":6}}'),
                       ("t2 a boolean", f'{{"ok":true,"t0":{t0},"t1":5,"t2":true}}'),
                       ("t2 missing", f'{{"ok":true,"t0":{t0},"t1":5}}'),
                       ("ok missing", f'{{"t0":{t0},"t1":5,"t2":6}}'),
                       ("integer beyond a long", f'{{"ok":true,"t0":{t0},"t1":5,"t2":{1 << 63}}}'),
                       ("device replied before it received", f'{{"ok":true,"t0":{t0},"t1":{B + R + 2},"t2":{B + R + 1}}}'),
                       ("not an object", "[1,2,3]"), ("not JSON", '{"ok":tru'), ("empty", "")]:
        replies.append((name, text, t0, t0 + 20 * ms))
    reply_cases = []
    for name, text, sent, t3 in replies:
        ts = tsync.TimeSync()
        why = ts.add_reply(text, sent, t3)
        b = ts.best()
        reply_cases.append({"name": name, "reply": text, "t0": sent, "t3": t3, "reject": why,
                            "offset_ns": None if b is None else b.offset_ns, "delay_ns": None if b is None else b.delay_ns})
    requests = [{"t0": t, "request": tsync.request(t)} for t in (0, B, tsync.MAX_TIMESTAMP_NS - 1)]

    constants = {"max_samples": tsync.MAX_SAMPLES, "max_delay_ns": tsync.MAX_DELAY_NS, "stale_ns": tsync.STALE_NS,
                 "drift_min_samples": tsync.DRIFT_MIN_SAMPLES, "drift_min_span_ns": tsync.DRIFT_MIN_SPAN_NS,
                 "max_timestamp_ns": tsync.MAX_TIMESTAMP_NS, "ppm": tsync.PPM,
                 "reasons": [tsync.REJECT_RANGE, tsync.REJECT_ORDER, tsync.REJECT_NEGATIVE, tsync.REJECT_SLOW,
                             tsync.REJECT_REPLY],
                 "quality": [tsync.OK, tsync.UNSYNCED, tsync.STALE]}
    return {"constants": constants, "exchanges": exchanges, "sessions": sessions, "requests": requests,
            "replies": reply_cases}


# ---- Link ops for glasses (ADR-0062) -------------------------------------------------------------------------
LO_SECRET = bytes(range(0x40, 0x60))           # the phone's unlock secret (synthetic)
LO_WRONG = bytes(32)
LO_NONCE = bytes(range(0xA0, 0xC0))            # the pod's wipe challenge, fixed for the vectors
LO_SEED = bytes(range(32))                     # the phone's Ed25519 seed (synthetic): signatures are deterministic
LO_CHUNK = 512
LO_MODEL = bytes((i * 31 + 7) % 256 for i in range(2 * LO_CHUNK + 100))   # 3 chunks, the last one short
LO_PASS = "vector passphrase"
LO_OPS = ("unlock", "add_unlock_slot", "panic", "wipe_challenge", "wipe", "hud", "model_offer", "model_chunk",
          "model_commit", "model_status")


class _LoClock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def _lo_parser(op: str):
    return {"unlock": LO.parse_ack, "add_unlock_slot": LO.parse_ack, "panic": LO.parse_ack,
            "wipe_challenge": LO.parse_wipe_challenge, "wipe": LO.parse_wipe, "hud": LO.parse_hud,
            "model_offer": SL.parse_offer, "model_chunk": SL.parse_chunk, "model_commit": SL.parse_commit,
            "model_status": SL.parse_status}[op]


def _lo_build(op: str, a: dict) -> str:
    b = bytes.fromhex
    return {"unlock": lambda: LO.unlock_request(a["slot"], b(a["secret"])),
            "add_unlock_slot": lambda: LO.add_unlock_slot_request(a["slot"], b(a["secret"])),
            "panic": LO.panic_request, "wipe_challenge": LO.wipe_challenge_request,
            "wipe": lambda: LO.wipe_request(b(a["command"]), b(a["signature"])),
            "hud": lambda: LO.hud_request(a["since"]),
            "model_offer": lambda: SL.offer_request(a["name"], a["slot"], a["sha256"], a["size"], a["chunk_size"]),
            "model_chunk": lambda: SL.chunk_request(a["sha256"], a["index"], b(a["data"])),
            "model_commit": lambda: SL.commit_request(a["sha256"]), "model_status": SL.status_request}[op]()


def _lo_frame(ops, version=1) -> str:
    """A hud reply as the pod sends it (pod_service._dumps: non-ASCII text raw)."""
    return json.dumps({"ok": True, "version": version, "ops": ops}, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def link_ops() -> dict:
    from nacl.signing import SigningKey
    phone = SigningKey(LO_SEED)
    sha = hashlib.sha256(LO_MODEL).hexdigest()
    pin = {"name": "detector-v1.onnx", "slot": "detector", "sha256": sha, "size": len(LO_MODEL)}
    hud_state = GlassesState(level="low", alert_text="Signal nearby", health={"thermal": "ok"}, voice_tier="low")
    cmd = LO.wipe_command(LO_NONCE)
    cases: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / SL.PINS_FILE).write_text(json.dumps([pin]))
        ks = KeyStore(d / "key.json")
        ks.create(LO_PASS)
        clock = _LoClock()
        panic = PanicController(soft=ks.lock, erase=ks.crypto_erase, clock=clock)
        feed = LO.HudFeed()
        svc = PodService(keystore=ks, panic=panic, phone_verify_key=phone.verify_key, hud=feed, now=clock,
                         rand=lambda n: LO_NONCE[:n],
                         sideload=SL.Sideload(d / SL.PINS_FILE, d / "models", free_bytes=lambda p: 1 << 40))
        bare = PodService()

        def run(name: str, op: str, args: dict, pod: PodService = svc, advance: float = LO.UNLOCK_MIN_INTERVAL_S) -> None:
            clock.t += advance
            req = _lo_build(op, args)
            reply = pod.handle(req)
            cases.append({"name": name, "op": op, "args": args, "request": req, "reply": reply,
                          "parsed": _lo_parser(op)(reply)})

        secret = {"slot": "phone", "secret": LO_SECRET.hex()}
        run("add_unlock_slot outside the pairing window", "add_unlock_slot", secret)
        svc.open_pairing_window()
        run("add_unlock_slot inside the window, unlocked", "add_unlock_slot", secret)
        run("panic", "panic", {})
        run("unlock, wrong secret", "unlock", {"slot": "phone", "secret": LO_WRONG.hex()})
        run("unlock, too soon after the last attempt", "unlock", secret, advance=0.0)
        run("unlock", "unlock", secret)
        run("unlock, slot not whitelisted", "unlock", {"slot": "watch", "secret": LO_SECRET.hex()})
        run("wipe_challenge", "wipe_challenge", {})
        o = {k: pin[k] for k in ("name", "slot", "sha256", "size")}
        run("model_offer, not pinned", "model_offer", {**o, "sha256": "f" * 64, "chunk_size": LO_CHUNK})
        run("model_offer, chunk_size too small", "model_offer", {**o, "chunk_size": SL.CHUNK_MIN - 1})
        run("model_offer", "model_offer", {**o, "chunk_size": LO_CHUNK})
        for i in (0, 2):
            run(f"model_chunk {i}", "model_chunk",
                {"sha256": sha, "index": i, "data": LO_MODEL[i * LO_CHUNK:(i + 1) * LO_CHUNK].hex()})
        run("model_chunk, out of range", "model_chunk", {"sha256": sha, "index": 3, "data": "00" * LO_CHUNK})
        run("model_commit, incomplete", "model_commit", {"sha256": sha})
        run("model_offer, resumed", "model_offer", {**o, "chunk_size": LO_CHUNK})
        run("model_chunk 1", "model_chunk", {"sha256": sha, "index": 1, "data": LO_MODEL[LO_CHUNK:2 * LO_CHUNK].hex()})
        run("model_commit", "model_commit", {"sha256": sha})
        run("model_status", "model_status", {})
        feed.set(display_list(hud_state))
        run("hud, first poll", "hud", {"since": 0})
        run("hud, unchanged", "hud", {"since": feed.version})
        bad_sig = bytearray(phone.sign(cmd).signature)
        bad_sig[0] ^= 1
        run("wipe, bad signature (uses the nonce up)", "wipe", {"command": cmd.hex(), "signature": bytes(bad_sig).hex()})
        run("wipe_challenge again", "wipe_challenge", {})
        run("wipe", "wipe", {"command": cmd.hex(), "signature": phone.sign(cmd).signature.hex()})
        run("wipe, replayed", "wipe", {"command": cmd.hex(), "signature": phone.sign(cmd).signature.hex()})
        run("unlock, not available", "unlock", secret, pod=bare)
        run("model_status, not available", "model_status", {}, pod=bare)
        run("hud, not available", "hud", {"since": 0}, pod=bare)
    if {c["op"] for c in cases} != set(LO_OPS):
        raise AssertionError("every link op needs a vector")

    canvas = display_list(hud_state)[0]
    W, H = canvas["w"], canvas["h"]
    text = {"op": "text", "id": "G2", "x": 8, "y": 58, "text": "Signal nearby", "size": 14, "colour": "#e6e6e6",
            "anchor": "start"}
    shape = {"op": "shape", "id": "G1", "kind": "circle", "x": 16, "y": 14.5, "size": 16, "fill": "#fff",
             "stroke": "", "stroke_width": 0, "tag": ""}
    zone = {"op": "zone", "id": "G3", "x": 600, "y": 380, "w": 100, "h": 100}
    hostile = [
        ("clean frame", _lo_frame([canvas, text, shape])),
        ("unknown op kinds dropped", _lo_frame([canvas, {"op": "script", "src": "x"}, text, {"op": 7}, {"op": None}, "text", 5])),
        ("a second canvas dropped", _lo_frame([canvas, text, canvas])),
        ("first op not a canvas: everything dropped", _lo_frame([text, canvas, shape])),
        ("canvas too large", _lo_frame([{**canvas, "w": LO.MAX_CANVAS_PX + 1}, text])),
        ("canvas size not an integer", _lo_frame([{**canvas, "w": float(W)}, text])),
        ("canvas size zero", _lo_frame([{**canvas, "h": 0}, text])),
        ("op count capped", _lo_frame([canvas] + [dict(text, id=f"t{i}") for i in range(LO.MAX_HUD_OPS + 10)])),
        ("text capped", _lo_frame([canvas, {**text, "text": "x" * (LO.MAX_HUD_TEXT + 40)}])),
        ("text capped in code points", _lo_frame([canvas, {**text, "text": "\U0001F600" * (LO.MAX_HUD_TEXT + 1)}])),
        ("every string field capped", _lo_frame([canvas, {**text, "id": "i" * 300, "colour": "c" * 100}])),
        ("coordinates clamped", _lo_frame([canvas, {**text, "x": -40, "y": H + 500}, {**shape, "x": W + 0.75, "y": -3.25},
                                           {**shape, "size": 9000, "stroke_width": -2}])),
        ("width and height clamped to the canvas", _lo_frame([canvas, zone, {**zone, "x": 10.5, "w": 1e9}])),
        ("huge coordinates", _lo_frame([canvas, {**text, "x": 1 << 62}, {**text, "y": -(1 << 62)}])),
        ("opacity and font size clamped", _lo_frame([{**canvas, "opacity": 7, "font_size": 10000.5}, text])),
        ("missing key dropped", _lo_frame([canvas, {k: v for k, v in text.items() if k != "anchor"}])),
        ("wrong types dropped", _lo_frame([canvas, {**text, "x": True}, {**text, "x": "1"}, {**text, "text": 5},
                                           {**text, "colour": None}, {**shape, "size": [1]}])),
        ("extra keys removed", _lo_frame([canvas, {**text, "onclick": "steal()", "href": "x"}])),
        ("infinite number dropped", _lo_frame([canvas, text]).replace('"x":8,', '"x":1e400,')),
        ("NaN is not JSON", _lo_frame([canvas, text]).replace('"x":8,', '"x":NaN,')),
        ("integer beyond a long", _lo_frame([canvas, text]).replace('"x":8,', f'"x":{1 << 64},')),
        ("unchanged", LO.dumps({"ok": True, "version": 4, "unchanged": True})),
        ("ops not a list", LO.dumps({"ok": True, "version": 4, "ops": {"0": canvas}})),
        ("version negative", _lo_frame([canvas], version=-1)),
        ("version a float", _lo_frame([canvas], version=1.5)),
        ("empty ops", _lo_frame([])),
    ]
    hud = [{"name": n, "reply": r, "parsed": LO.parse_hud(r)} for n, r in hostile]

    good_nonce = LO.b64url(LO_NONCE)
    malformed = [
        ("ack", "not JSON", "{"), ("ack", "empty", ""), ("ack", "not an object", "[true]"), ("ack", "ok missing", "{}"),
        ("ack", "ok a string", LO.dumps({"ok": "true"})), ("ack", "refusal without an error", LO.dumps({"ok": False})),
        ("ack", "refusal error not a string", LO.dumps({"ok": False, "error": 5})),
        ("ack", "refusal passed through", LO.dumps({"ok": False, "error": LO.UNLOCK_REFUSED})),
        ("wipe_challenge", "nonce missing", LO.dumps({"ok": True})),
        ("wipe_challenge", "nonce padded", LO.dumps({"ok": True, "nonce": good_nonce + "="})),
        ("wipe_challenge", "nonce short", LO.dumps({"ok": True, "nonce": LO.b64url(LO_NONCE[:-1])})),
        ("wipe_challenge", "nonce standard base64", LO.dumps({"ok": True, "nonce": "+" + good_nonce[1:]})),
        ("wipe_challenge", "nonce with stray bits", LO.dumps({"ok": True, "nonce": good_nonce[:-1] + "B"})),
        ("wipe", "wiped missing", LO.dumps({"ok": True})), ("wipe", "wiped false", LO.dumps({"ok": True, "wiped": False})),
        ("offer", "range beyond chunks", LO.dumps({"ok": True, "chunks": 3, "missing": [[0, 4]], "missing_count": 4})),
        ("offer", "empty range", LO.dumps({"ok": True, "chunks": 3, "missing": [[1, 1]], "missing_count": 0})),
        ("offer", "negative count", LO.dumps({"ok": True, "chunks": 3, "missing": [], "missing_count": -1})),
        ("offer", "range not a pair", LO.dumps({"ok": True, "chunks": 3, "missing": [[0, 1, 2]], "missing_count": 1})),
        ("offer", "too many ranges", LO.dumps({"ok": True, "chunks": 1000, "missing_count": SL.MAX_MISSING_RANGES + 1,
                                                "missing": [[2 * i, 2 * i + 1] for i in range(SL.MAX_MISSING_RANGES + 1)]})),
        ("chunk", "missing_count a float", LO.dumps({"ok": True, "missing_count": 1.0})),
        ("commit", "installed missing", LO.dumps({"ok": True})),
        ("status", "models not a list", LO.dumps({"ok": True, "models": {}})),
        ("status", "model with a traversal name", LO.dumps({"ok": True, "models": [{**pin, "name": "../x"}]})),
        ("status", "model with an extra key", LO.dumps({"ok": True, "models": [{**pin, "path": "/etc"}]})),
        ("status", "model size a boolean", LO.dumps({"ok": True, "models": [{**pin, "size": True}]})),
    ]
    parsers = {"ack": LO.parse_ack, "wipe_challenge": LO.parse_wipe_challenge, "wipe": LO.parse_wipe,
               "offer": SL.parse_offer, "chunk": SL.parse_chunk, "commit": SL.parse_commit, "status": SL.parse_status}
    replies = [{"parser": p, "name": n, "reply": r, "parsed": parsers[p](r)} for p, n, r in malformed]

    constants = {
        "secret_bytes": LO.SECRET_BYTES, "nonce_bytes": LO.NONCE_BYTES, "signature_bytes": LO.SIGNATURE_BYTES,
        "wipe_prefix": LO.WIPE_PREFIX.decode(), "unlock_slots": list(LO.UNLOCK_SLOTS),
        "max_hud_ops": LO.MAX_HUD_OPS, "max_hud_text": LO.MAX_HUD_TEXT, "max_canvas_px": LO.MAX_CANVAS_PX,
        "number_keys": list(LO.NUMBER_KEYS), "max_model_bytes": SL.MAX_MODEL_BYTES, "chunk_min": SL.CHUNK_MIN,
        "chunk_max": SL.CHUNK_MAX, "max_missing_ranges": SL.MAX_MISSING_RANGES,
        "name_re": SL.NAME_RE.pattern, "slot_re": SL.SLOT_RE.pattern, "sha256_re": SL.SHA256_RE.pattern,
        "errors": {"not_available": LO.NOT_AVAILABLE, "bad_reply": LO.BAD_REPLY, "unlock_refused": LO.UNLOCK_REFUSED,
                   "add_slot_refused": LO.ADD_SLOT_REFUSED, "wipe_refused": LO.WIPE_REFUSED},
    }
    wipe = {"seed": LO_SEED.hex(), "verify_key": bytes(phone.verify_key).hex(), "nonce": LO_NONCE.hex(),
            "command": cmd.hex(), "signature": phone.sign(cmd).signature.hex()}
    chunking = [{"size": n, "chunk_size": c, "chunks": SL.chunk_count(n, c)}
                for n, c in [(1, SL.CHUNK_MIN), (SL.CHUNK_MIN, SL.CHUNK_MIN), (SL.CHUNK_MIN + 1, SL.CHUNK_MIN),
                             (len(LO_MODEL), LO_CHUNK), (SL.MAX_MODEL_BYTES, SL.CHUNK_MIN),
                             (SL.MAX_MODEL_BYTES, SL.CHUNK_MAX), (SL.MAX_MODEL_BYTES - 1, SL.CHUNK_MAX)]]
    b64 = [{"hex": x.hex(), "b64url": LO.b64url(x)} for x in (b"", b"\x00", b"\xff\xfe", b"\xfb\xff\xbf", LO_SECRET)]
    return {"constants": constants, "ops": cases, "replies": replies, "hud": hud, "wipe": wipe, "chunking": chunking,
            "base64url": b64}


# ---- Phone-guided voice enrollment (ADR-0057; android/core Enrollment.kt) --------------------------------------
VE_RATE = 8000
VE_SHORT = 300                  # samples in the builder vectors' PCM, listed in full
VE_FULL_REQUEST_MAX = 2000      # longer requests are recorded as their SHA-256 (Kotlin rebuilds them from "gen")
VE_PROMPT_ID = "calm-low"
VE_SPEECH = {"samples": 12000, "period": 64, "amp": 8000, "on": 2400, "off": 800, "lead": 800}   # about 1.2 s of speech
VE_SHORT_SPEECH = {**VE_SPEECH, "on": 1600, "off": 20000}   # one 0.2 s burst: too little speech
VE_LOUD = {**VE_SPEECH, "amp": 32767}                       # clipped
VE_CFG = {"enroll_windows": 3, "enroll_window_s": 3.0, "enroll_min_speech_s": 0.5, "enroll_min_total_speech_s": 5.0}
VE_MAX_REFUSED = 5              # = oiris/voice_link_cli.MAX_REFUSED_WINDOWS (the flow gives up after this many)
VE_STAGES = ("idle", "recording", "saved", "stopped")
VE_FACE_SEEDS = {"Alex": 11, "Blake": 12}


def ve_pcm(samples: int, period: int, amp: int, on: int, off: int, lead: int) -> np.ndarray:
    """Integer-only synthetic 'speech' (Kotlin VoiceVectors.pcm makes the same int16 samples): a triangle wave of
    `period` samples and peak `amp`, gated on for `on` samples and off for `off`, after `lead` silent samples."""
    i = np.arange(samples, dtype=np.int64)
    t = i % period
    half = period // 2
    tri = 2 * amp * np.minimum(t, period - t) // half - amp
    k = i - lead
    gate = (k >= 0) & (k % (on + off) < on)
    return np.where(gate, tri, 0).astype(np.int16)


def ve_gen(period: int | None = None, **kw) -> dict:
    g = dict(VE_SPEECH, **kw)
    if period is not None:
        g["period"] = period
    return g


class _VeSpeaker:
    """TEST-ONLY deterministic speaker_embed plugin for the vectors: unit, mean-centred band energies."""
    name, dim, model_id, test_only = "vector_bands", 16, "vector-bands-v1", True

    def __init__(self, model_dir: str = "") -> None:
        self.model_dir = model_dir

    def embed(self, audio, sample_rate):
        p = np.abs(np.fft.rfft(np.asarray(audio, np.float64))) ** 2
        v = np.sqrt(np.array([b.sum() for b in np.array_split(p, self.dim)]) + 1e-12)
        v -= v.mean()
        return (v / np.linalg.norm(v)).astype(np.float32)


def _ve_ids():
    n = [0]

    def uuid4() -> uuid.UUID:
        n[0] += 1
        return uuid.UUID(int=n[0])
    return SimpleNamespace(uuid4=uuid4)


def _ve_request(op: str, a: dict) -> str:
    from oiris.modules.face_match import remote as R
    if op == "voice_sample":
        pcm = np.array(a["pcm"], np.int16) if "pcm" in a else ve_pcm(**a["gen"])
        return R.voice_sample_request(a["person"], a["prompt_id"], a["rate"], pcm)
    return {"voice_prompts": lambda: R.voice_prompts_request(),
            "voice_start": lambda: R.voice_start_request(a["person"], a["consent"]),
            "voice_finish": lambda: R.voice_finish_request(a["person"]),
            "voice_cancel": lambda: R.voice_cancel_request(a["person"])}[op]()


def _ve_parser(op: str):
    from oiris.modules.face_match import remote as R
    return {"voice_prompts": R.parse_voice_prompts, "voice_start": R.parse_voice_start, "voice_sample": R.parse_voice_sample,
            "voice_finish": R.parse_voice_finish, "voice_cancel": R.parse_voice_ack, "ack": R.parse_voice_ack}[op]


def _ve_req_record(req: str) -> dict:
    if len(req) <= VE_FULL_REQUEST_MAX:
        return {"request": req}
    return {"request_sha256": hashlib.sha256(req.encode()).hexdigest(), "request_length": len(req)}


def _ve_pod(d: Path):
    """A temporary store with M-16 (two people enrolled) and M-17 (fake speaker); returns (server, ids, bare server)."""
    from oiris.guard.consent import ConsentRecord, ConsentRegistry
    from oiris.modules.face_match import core as fm_core, optin as face_optin
    from oiris.modules.face_match.backends.test_deterministic import TestDeterministicBackend, synth_face_frame
    from oiris.modules.face_match.port import REQUIRED_ANGLES
    from oiris.modules.face_match.remote import EnrollmentServer
    from oiris.modules.voice_link import optin as vl_optin
    from oiris.modules.voice_link.core import VoiceLinkConfig, VoiceLinkModule
    from oiris.registry import PluginMeta, Registry
    from oiris.slots import SlotManager
    from oiris.store.db import DB
    ks = KeyStore(d / "keys.json")
    ks.create(LO_PASS)
    db = DB(d / "asb.db", ks.subkey("db"))
    chain = face_optin.DbChain(db, ks.subkey("face-chain"))
    consents = ConsentRegistry(db)
    for cid, scopes in (("c-voice", {"face_enrollment", "voice_enrollment"}), ("c-face", {"face_enrollment"})):
        consents.add(ConsentRecord(cid, "s-" + cid, scopes))
    face_optin.record_opt_in(db, chain, "I UNDERSTAND", ["personal"], "dev1", "us-other",
                             warning_shown=face_optin.warning_for("us-other"))
    face = fm_core.FaceMatchModule(db, ks, d, TestDeterministicBackend(), chain,
                                   fm_core.FaceMatchConfig(modes=["personal"], allow_test_backend=True), consents, timeout_s=1.0)
    ids = {}
    sid = face.start_session()
    for name, seed in VE_FACE_SEEDS.items():
        pid = face.add_person(sid, name, "c-voice")
        for i, a in enumerate(REQUIRED_ANGLES):
            face.submit_shot(pid, synth_face_frame(seed, a, noise_seed=i), a)
        if not face.finish_person(pid).complete:
            raise AssertionError("vector face enrollment failed")
        ids[name] = pid
    face.end_session(sid)
    vl_optin.record_opt_in(db, chain, "I UNDERSTAND", "ambient", "dev1", warning_shown=vl_optin.WARNING_TEXT)
    reg = Registry()
    reg.register("speaker_embed", _VeSpeaker.name, _VeSpeaker, port_version="1.0",
                 meta=PluginMeta(license="Apache-2.0", data_tier="synthetic", commercial_ok=True))
    sm = SlotManager(reg, {"speaker_embed": _VeSpeaker.name}, {_VeSpeaker.name: {"model_dir": "unused"}})
    vm = VoiceLinkModule(db, face, sm, VoiceLinkConfig(allow_test_backend=True, **VE_CFG), chain=chain)
    srv = EnrollmentServer(db, chain, "phone1", lambda: face, module=face, voice=vm)
    bare = EnrollmentServer(db, chain, "phone1", lambda: face, module=face)
    return srv, ids, bare, db


def _ve_flow(srv, name: str, person: str, consent: str, gens: list[dict], prompt_ids: list[str]) -> dict:
    """The scripted flow (VoiceEnrollmentFlow in Enrollment.kt follows the same rules) against the real pod:
    prompts, start; each attempt reads the current sentence; accepted -> the next sentence (cycling); finish when the
    pod says nothing remains or the window maximum is reached; a finish that still needs speech -> the next sentence;
    VE_MAX_REFUSED refusals -> cancel. `gens` are the PCM attempts in order."""
    from oiris.modules.face_match import remote as R
    steps: list[dict] = []

    def call(req: str) -> str:
        reply = srv.handle(req)
        steps.append({**_ve_req_record(req), "reply": reply})
        return reply

    st = {"stage": "idle", "accepted": 0, "refused": 0, "shown": 0, "saved": False}
    if not R.parse_voice_prompts(call(R.voice_prompts_request()))["ok"]:
        st["stage"] = "stopped"
    start = None if st["stage"] == "stopped" else R.parse_voice_start(call(R.voice_start_request(person, consent)))
    if start is not None and not start["ok"]:
        st["stage"] = "stopped"
    used = 0
    if start is not None and start["ok"]:
        st["stage"], st["shown"] = "recording", 1
        max_w = start["max_windows"]

        def finish() -> None:
            f = R.parse_voice_finish(call(R.voice_finish_request(person)))
            if f["ok"] and f["saved"]:
                st["stage"], st["saved"] = "saved", True
            elif f["ok"] and f["remaining"] > 0 and st["accepted"] < max_w:
                st["shown"] += 1
            else:
                st["stage"] = "stopped"

        for g in gens:
            if st["stage"] != "recording":
                break
            used += 1
            pid = prompt_ids[(st["shown"] - 1) % len(prompt_ids)]
            r = R.parse_voice_sample(call(R.voice_sample_request(person, pid, VE_RATE, ve_pcm(**g))))
            if r["ok"] and r["accepted"]:
                st["accepted"] += 1
                if r["remaining"] == 0 or st["accepted"] >= max_w:
                    finish()
                else:
                    st["shown"] += 1
            elif r["ok"] and r["code"] == "enough":
                finish()
            else:
                st["refused"] += 1
                if st["refused"] >= VE_MAX_REFUSED:
                    call(R.voice_cancel_request(person))
                    st["stage"] = "stopped"
    return {"name": name, "person": person, "consent": consent, "rate": VE_RATE, "attempts": gens[:used] if used else [],
            "steps": steps, "final": st}


def voice_enrollment() -> dict:
    from oiris.config import VOICE_ENROLL_MAX_WINDOWS
    from oiris.modules.face_match import core as fm_core, remote as R
    from oiris.modules.voice_link import prompts as P
    from oiris.voice_link_cli import MAX_REFUSED_WINDOWS
    if MAX_REFUSED_WINDOWS != VE_MAX_REFUSED:
        raise AssertionError("VE_MAX_REFUSED must equal voice_link_cli.MAX_REFUSED_WINDOWS")
    short = ve_pcm(VE_SHORT, 40, 9000, 200, 50, 10).tolist()
    edge = [0, 1, -1, 255, 256, -256, 32767, -32768, 12345, -12345]
    requests = []
    for name, op, a in [
        ("voice_prompts", "voice_prompts", {}),
        ("voice_start", "voice_start", {"person": "P1", "consent": "c-voice"}),
        ("voice_start, non-ASCII", "voice_start", {"person": "P é", "consent": "c\n1"}),
        ("voice_sample, short synthetic PCM", "voice_sample", {"person": "P1", "prompt_id": VE_PROMPT_ID, "rate": VE_RATE, "pcm": short}),
        ("voice_sample, edge values (little-endian)", "voice_sample", {"person": "P1", "prompt_id": "count", "rate": 16000, "pcm": edge}),
        ("voice_sample, one sample (padding ==)", "voice_sample", {"person": "P1", "prompt_id": "count", "rate": 48000, "pcm": [-2]}),
        ("voice_sample, two samples (padding =)", "voice_sample", {"person": "P1", "prompt_id": "count", "rate": 48000, "pcm": [1, 2]}),
        ("voice_sample, empty", "voice_sample", {"person": "P1", "prompt_id": "count", "rate": 8000, "pcm": []}),
        ("voice_finish", "voice_finish", {"person": "P1"}),
        ("voice_cancel", "voice_cancel", {"person": "P1"}),
    ]:
        requests.append({"name": name, "op": op, "args": a, "request": _ve_request(op, a)})

    prompt_ids = [p.id for p in P.PROMPTS]
    cases: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp, mock.patch.object(fm_core, "uuid", _ve_ids()):
        srv, ids, bare, db = _ve_pod(Path(tmp))
        alex, blake = ids["Alex"], ids["Blake"]

        def run(name: str, op: str, args: dict, pod=srv) -> dict:
            req = _ve_request(op, args)
            reply = pod.handle(req)
            p = _ve_parser(op)(reply)
            cases.append({"name": name, "op": op, "args": args, **_ve_req_record(req), "reply": reply, "parsed": p})
            return p

        def raw(name: str, op: str, req: str) -> None:
            reply = srv.handle(req)
            cases.append({"name": name, "op": op, "args": None, "request": req, "reply": reply, "parsed": _ve_parser(op)(reply)})

        def sample(name: str, gen: dict, prompt: str, person: str = alex, rate: int = VE_RATE) -> dict:
            return run(name, "voice_sample", {"person": person, "prompt_id": prompt, "rate": rate, "gen": gen})

        run("voice_prompts", "voice_prompts", {})
        run("voice_start, consent without voice_enrollment", "voice_start", {"person": alex, "consent": "c-face"})
        run("voice_start, unknown consent", "voice_start", {"person": alex, "consent": "c-none"})
        run("voice_start, person not enrolled in face matching", "voice_start", {"person": "nobody", "consent": "c-voice"})
        run("voice_sample before voice_start", "voice_sample", {"person": alex, "prompt_id": VE_PROMPT_ID, "rate": VE_RATE, "gen": VE_SPEECH})
        run("voice_start", "voice_start", {"person": alex, "consent": "c-voice"})
        run("voice_finish, no windows yet", "voice_finish", {"person": alex})
        sample("voice_sample refused: unknown sentence", VE_SPEECH, "no-such-sentence")
        sample("voice_sample refused: sample rate", VE_SPEECH, VE_PROMPT_ID, rate=4000)
        run("voice_sample refused: empty audio", "voice_sample", {"person": alex, "prompt_id": VE_PROMPT_ID, "rate": VE_RATE, "pcm": []})
        sample("voice_sample refused: too loud", VE_LOUD, VE_PROMPT_ID)
        sample("voice_sample refused: too little speech", VE_SHORT_SPEECH, VE_PROMPT_ID)
        k = 0
        while True:
            r = sample(f"voice_sample accepted, sentence {k + 1}", ve_gen(48 + 8 * k), prompt_ids[k % len(prompt_ids)])
            k += 1
            if k == VE_CFG["enroll_windows"]:
                if r["remaining"] < 1:
                    raise AssertionError("the vectors need a 'more total speech' step: shorten VE_SPEECH")
                run("voice_finish refused: more total speech needed", "voice_finish", {"person": alex})
            if r["remaining"] == 0:
                break
        sample("voice_sample refused: enough windows", VE_SPEECH, prompt_ids[k % len(prompt_ids)])
        run("voice_finish, saved", "voice_finish", {"person": alex})
        run("voice_sample after finish", "voice_sample", {"person": alex, "prompt_id": VE_PROMPT_ID, "rate": VE_RATE, "pcm": short})
        run("voice_start, second person", "voice_start", {"person": blake, "consent": "c-voice"})
        sample("voice_sample refused: longer than the limit", ve_gen(samples=R.MAX_SAMPLE_S * VE_RATE + 1), VE_PROMPT_ID, person=blake)
        sample("voice_sample refused: larger than the cap", ve_gen(samples=R.MAX_PCM_BYTES // R.PCM_SAMPLE_BYTES + 1),
               VE_PROMPT_ID, person=blake, rate=R.VOICE_RATE_MAX)
        raw("voice_sample refused: odd-length PCM", "voice_sample", R.dumps({"op": "voice_sample", "person": blake,
            "prompt_id": VE_PROMPT_ID, "rate": VE_RATE, "pcm16": "AAAA"}))
        raw("voice_sample refused: one byte of PCM", "voice_sample", R.dumps({"op": "voice_sample", "person": blake,
            "prompt_id": VE_PROMPT_ID, "rate": VE_RATE, "pcm16": "AA=="}))
        raw("voice_sample refused: not base64", "voice_sample", R.dumps({"op": "voice_sample", "person": blake,
            "prompt_id": VE_PROMPT_ID, "rate": VE_RATE, "pcm16": "AA*AAA=="}))
        raw("voice_sample refused: base64 with stray bits", "voice_sample", R.dumps({"op": "voice_sample", "person": blake,
            "prompt_id": VE_PROMPT_ID, "rate": VE_RATE, "pcm16": "AAB="}))
        raw("voice_sample refused: rate not an integer", "voice_sample", R.dumps({"op": "voice_sample", "person": blake,
            "prompt_id": VE_PROMPT_ID, "rate": 8000.0, "pcm16": ""}))
        run("voice_cancel", "voice_cancel", {"person": blake})
        run("voice_sample after cancel", "voice_sample", {"person": blake, "prompt_id": VE_PROMPT_ID, "rate": VE_RATE, "pcm": short})
        run("voice_cancel, nothing open", "voice_cancel", {"person": blake})
        for op, a in [("voice_prompts", {}), ("voice_start", {"person": alex, "consent": "c-voice"}),
                      ("voice_sample", {"person": alex, "prompt_id": VE_PROMPT_ID, "rate": VE_RATE, "pcm": short}),
                      ("voice_finish", {"person": alex}), ("voice_cancel", {"person": alex})]:
            run(f"{op}, voice link off", op, a, pod=bare)

        speech = [ve_gen(48 + 8 * k) for k in range(len(prompt_ids))]
        flows = [
            _ve_flow(srv, "saved, with a refusal and more total speech", blake, "c-voice",
                     speech[:1] + [VE_SHORT_SPEECH] + speech[1:], prompt_ids),
            _ve_flow(srv, "too many refusals: cancelled", blake, "c-voice",
                     [VE_SHORT_SPEECH, VE_LOUD] + [VE_SHORT_SPEECH] * (VE_MAX_REFUSED - 2) + speech, prompt_ids),
            _ve_flow(srv, "start refused: no voice consent", blake, "c-face", speech, prompt_ids),
            _ve_flow(bare, "voice link off", blake, "c-voice", speech, prompt_ids),
        ]
        db.close()
    if {c["op"] for c in cases} != set(R.VOICE_OPS):
        raise AssertionError("every voice op needs a vector")

    malformed = [
        ("ack", "not JSON", "{"), ("ack", "empty", ""), ("ack", "ok missing", "{}"),
        ("ack", "refusal without an error", R.dumps({"ok": False})),
        ("voice_prompts", "prompts missing", R.dumps({"ok": True})),
        ("voice_prompts", "prompts empty", R.dumps({"ok": True, "prompts": []})),
        ("voice_prompts", "prompt with an extra key", R.dumps({"ok": True, "prompts": [{"id": "a", "text": "b", "tone": "low", "x": 1}]})),
        ("voice_prompts", "prompt text not a string", R.dumps({"ok": True, "prompts": [{"id": "a", "text": 5, "tone": "low"}]})),
        ("voice_prompts", "prompt text too long", R.dumps({"ok": True, "prompts": [{"id": "a", "text": "x" * (R.MAX_REPLY_TEXT + 1), "tone": "low"}]})),
        ("voice_prompts", "too many prompts", R.dumps({"ok": True, "prompts": [{"id": f"p{i}", "text": "t", "tone": "low"}
                                                                             for i in range(R.MAX_PROMPTS + 1)]})),
        ("voice_start", "windows zero", R.dumps({"ok": True, "windows": 0, "max_windows": 10, "target_speech_s": 20.0, "window_s": 7.0})),
        ("voice_start", "windows above the maximum", R.dumps({"ok": True, "windows": 11, "max_windows": 10, "target_speech_s": 20.0, "window_s": 7.0})),
        ("voice_start", "target a string", R.dumps({"ok": True, "windows": 6, "max_windows": 10, "target_speech_s": "20", "window_s": 7.0})),
        ("voice_start", "window length zero", R.dumps({"ok": True, "windows": 6, "max_windows": 10, "target_speech_s": 20.0, "window_s": 0})),
        ("voice_start", "integer target", R.dumps({"ok": True, "windows": 6, "max_windows": 10, "target_speech_s": 20, "window_s": 7})),
        ("voice_sample", "accepted missing", R.dumps({"ok": True, "code": "ok", "message": "m", "speech_s": 1.0, "total_speech_s": 1.0,
                                                     "level_dbfs": None, "remaining": 1})),
        ("voice_sample", "accepted a number", R.dumps({"ok": True, "accepted": 1, "code": "ok", "message": "m", "speech_s": 1.0,
                                                      "total_speech_s": 1.0, "level_dbfs": None, "remaining": 1})),
        ("voice_sample", "remaining negative", R.dumps({"ok": True, "accepted": True, "code": "ok", "message": "m", "speech_s": 1.0,
                                                       "total_speech_s": 1.0, "level_dbfs": -20.5, "remaining": -1})),
        ("voice_sample", "speech seconds negative", R.dumps({"ok": True, "accepted": True, "code": "ok", "message": "m", "speech_s": -1.0,
                                                            "total_speech_s": 1.0, "level_dbfs": -20.5, "remaining": 1})),
        ("voice_sample", "level a string", R.dumps({"ok": True, "accepted": True, "code": "ok", "message": "m", "speech_s": 1.0,
                                                   "total_speech_s": 1.0, "level_dbfs": "-20", "remaining": 1})),
        ("voice_sample", "infinite total", R.dumps({"ok": True, "accepted": True, "code": "ok", "message": "m", "speech_s": 1.0,
                                                   "total_speech_s": 1.0, "level_dbfs": -20.5, "remaining": 1}).replace(
                                                       '"total_speech_s":1.0', '"total_speech_s":1e400')),
        ("voice_sample", "NaN level", R.dumps({"ok": True, "accepted": True, "code": "ok", "message": "m", "speech_s": 1.0,
                                              "total_speech_s": 1.0, "level_dbfs": -20.5, "remaining": 1}).replace("-20.5", "NaN")),
        ("voice_sample", "integer seconds and level", R.dumps({"ok": True, "accepted": False, "code": "speech", "message": "m",
                                                              "speech_s": 0, "total_speech_s": 3, "level_dbfs": -40, "remaining": 2})),
        ("voice_sample", "remaining beyond a long", R.dumps({"ok": True, "accepted": True, "code": "ok", "message": "m", "speech_s": 1.0,
                                                            "total_speech_s": 1.0, "level_dbfs": None, "remaining": 1 << 64})),
        ("voice_finish", "saved missing", R.dumps({"ok": True, "code": "done", "message": "m", "total_speech_s": 5.0, "remaining": 0})),
        ("voice_finish", "message not a string", R.dumps({"ok": True, "saved": True, "code": "done", "message": None,
                                                         "total_speech_s": 5.0, "remaining": 0})),
        ("voice_finish", "remaining a float", R.dumps({"ok": True, "saved": False, "code": "windows", "message": "m",
                                                      "total_speech_s": 5.0, "remaining": 1.0})),
    ]
    replies = [{"parser": p, "name": n, "reply": r, "parsed": _ve_parser(p)(r)} for p, n, r in malformed]

    constants = {"ops": list(R.VOICE_OPS), "voice_off": R.VOICE_OFF, "bad_reply": LO.BAD_REPLY,
                 "max_sample_s": R.MAX_SAMPLE_S, "max_pcm_bytes": R.MAX_PCM_BYTES, "max_pcm_b64": R.MAX_PCM_B64,
                 "rate_min": R.VOICE_RATE_MIN, "rate_max": R.VOICE_RATE_MAX, "max_windows": VOICE_ENROLL_MAX_WINDOWS,
                 "max_prompts": R.MAX_PROMPTS, "max_reply_text": R.MAX_REPLY_TEXT, "max_refused": VE_MAX_REFUSED,
                 "stages": list(VE_STAGES), "full_request_max": VE_FULL_REQUEST_MAX}
    pcm = [{"gen": g, "sha256": hashlib.sha256(ve_pcm(**g).astype("<i2").tobytes()).hexdigest()}
           for g in (VE_SPEECH, VE_SHORT_SPEECH, VE_LOUD, ve_gen(50, samples=777, lead=3, on=101, off=7))]
    return {"constants": constants, "requests": requests, "ops": cases, "replies": replies, "flows": flows, "pcm": pcm}


# ---- Fusion v2 (android/core FusionCore.kt) ----------------------------------------------------------------
FU_SEED = 58
FU_RANDOM_CASES = 120
FU_MODULES = sorted(FS.SCORE_FAMILY) + ["S_other"]       # one unknown module: fuse ignores it
FU_HEALTH = ["ok"] * 6 + ["degraded"] * 2 + ["failed"] * 2
FU_EDGE_VALUES = [0.0, 0.5, 1.0, 0.4999999, 0.5000001]
FU_EDGE_SHARE, FU_EDGE_CONF_SHARE = 0.2, 0.15
FU_MAX_SCORES = 10
NON_FINITE = ("nan", "inf", "-inf")   # how a vector spells a non-finite float; both sides read it back as one


def nf(x):
    """A vector value as the Python call receives it: "nan" / "inf" / "-inf" become those floats."""
    return float(x) if isinstance(x, str) and x in NON_FINITE else x


def _fu_weights(text: str) -> FS.FusionModel | None:
    """FusionModel.from_json, None where it raises (load_weights then falls back)."""
    try:
        return FS.FusionModel.from_json(text)
    except Exception:  # noqa: BLE001  # python-brain: allow SILENT_EXCEPTION - a refused weights file is the recorded outcome (null); Kotlin must refuse it too
        return None


def _fu_score(raw: list):
    module, value, conf, health = raw
    try:
        s = Score(module, nf(value), nf(conf), health)
    except ValueError:  # python-brain: allow SILENT_EXCEPTION - Score() refusing a health name is the recorded outcome (null); Kotlin must refuse it too
        return None
    return [s.module, s.value, s.confidence, s.health.value]


def _fu_case(name: str, raw: list, weights: str | None = None, fallback_node: bool = False, **kw) -> dict:
    model = _fu_weights(weights) if weights is not None else None
    built = [_fu_score(r) for r in raw]
    scores = [Score(m, v, c, h) for m, v, c, h in (b for b in built if b is not None)]
    x, m, contrib, unhealthy = FS.family_values(scores)
    out = {"name": name, "scores": raw, "accepted": [b is not None for b in built],
           "family": {"x": x.tolist(), "mask": m.tolist(), "contrib": contrib, "unhealthy": unhealthy},
           "logit": (model or FS.FusionModel()).logit(x, m), "result": FS.fuse(scores, model, **kw)}
    if weights is not None:
        out["weights"] = weights
    if kw:
        out["params"] = kw
    if fallback_node:
        out["fallback_node"] = True   # Kotlin-only advisory flag; Python's output is unchanged
    return out


def _fu_random(rng: random.Random) -> list:
    out = []
    for _ in range(rng.randint(0, FU_MAX_SCORES)):
        value = rng.choice(FU_EDGE_VALUES) if rng.random() < FU_EDGE_SHARE else round(rng.random(), 6)
        conf = rng.choice([0.0, 1.0, 0.5]) if rng.random() < FU_EDGE_CONF_SHARE else round(rng.random(), 6)
        out.append([rng.choice(FU_MODULES), value, conf, rng.choice(FU_HEALTH)])
    return out


def fusion() -> dict:
    d = FS.FusionModel()
    constants = {"families": FS.FAMILIES, "score_family": FS.SCORE_FAMILY, "score_signature": FS.SCORE_SIGNATURE,
                 "family_threshold": FS.FAMILY_THRESHOLD, "min_families": 2, "max_unhealthy": 1,
                 "health": [h.value for h in Health], "levels": [lv.value for lv in Level],
                 "default_model": json.loads(d.to_json())}
    raws = [["S_rf", 0.7, 0.9, "ok"], ["S_rf", "0.7", " 1e-1 ", "degraded"], ["S_rf", "nan", "inf", "ok"],
            ["S_rf", "-inf", -0.5, "ok"], ["S_rf", 1.5, 2, "failed"], ["S_rf", None, True, "ok"], ["S_rf", "abc", "1_0", "ok"],
            ["S_rf", ".5", "1.", "ok"], ["S_rf", "Infinity", "-NaN", "ok"], ["S_rf", "0x10", "1e5", "ok"], ["S_rf", [1], {}, "ok"],
            ["S_rf", 0.5, 0.5, "bogus"], ["S_rf", 0.5, 0.5, "OK"], ["S_rf", 0.5, 0.5, None], [5, 0.9, 0.9, "ok"]]
    assert _fu_score(["S_rf", float("nan"), float("inf"), "ok"]) == _fu_score(["S_rf", "nan", "inf", "ok"])
    scores = [{"raw": r, "score": _fu_score(r)} for r in raws]
    T = FS.FAMILY_THRESHOLD
    custom = json.dumps({"w": [0.3, 2.0, 1.1, -0.4, 0.9, 1.7], "w_mask": [-0.2, 0.1, 0.0, 0.3, -0.5, 0.05], "b": -1.25,
                         "version": "custom-1"})
    cases = [
        _fu_case("empty", []),
        _fu_case("exactly at the threshold: two families", [["S_glint", T, 0.8, "ok"], ["S_thermal", T, 0.6, "ok"]]),
        _fu_case("just below the threshold", [["S_glint", 0.4999999, 0.8, "ok"], ["S_thermal", 0.4999999, 0.6, "ok"]]),
        _fu_case("at threshold with zero confidence", [["S_glint", T, 0.0, "ok"], ["S_thermal", 0.9, 0.0, "ok"]]),
        _fu_case("one family: fyi", [["S_glint", 0.9, 0.9, "ok"], ["S_motion", 0.8, 0.9, "ok"]]),
        _fu_case("S_wifi plus rf: still one family", [["S_wifi", 0.9, 0.9, "ok"], ["S_rf", 0.8, 0.7, "ok"],
                                                       ["S_rf_drift", 0.6, 0.9, "ok"]]),
        _fu_case("S_wifi plus optical: two families", [["S_wifi", 0.9, 0.9, "ok"], ["S_glint", 0.7, 0.7, "ok"]]),
        _fu_case("largest value*confidence wins in a family", [["S_rf", 0.9, 0.2, "ok"], ["S_wifi", 0.6, 0.9, "ok"]]),
        _fu_case("zero-confidence first score is replaced", [["S_rf", 0.9, 0.0, "ok"], ["S_wifi", 0.1, 0.1, "ok"]]),
        _fu_case("suppression: two unhealthy families", [["S_glint", 0.9, 0.9, "ok"], ["S_thermal", 0.9, 0.9, "ok"],
                                                          ["S_radar", 0.5, 0.5, "failed"], ["S_eye", 0.5, 0.5, "failed"]]),
        _fu_case("one unhealthy family: not suppressed", [["S_glint", 0.9, 0.9, "ok"], ["S_thermal", 0.9, 0.9, "ok"],
                                                           ["S_radar", 0.5, 0.5, "failed"]]),
        _fu_case("a failed score beside a healthy one: not unhealthy", [["S_radar", 0.9, 0.9, "failed"],
                                                                         ["S_live", 0.2, 0.9, "ok"], ["S_eye", 0.3, 0.3, "failed"]]),
        _fu_case("degraded still counts", [["S_voice", 0.9, 0.9, "degraded"], ["S_eye", 0.9, 0.9, "degraded"]]),
        _fu_case("A11: two of vital, eye, voice, thermal", [["S_radar", 0.9, 0.9, "ok"], ["S_voice", 0.9, 0.9, "ok"],
                                                             ["S_live", 0.7, 0.9, "ok"]]),
        _fu_case("signature without an active family (zero confidence)", [["S_newobj", 0.8, 0.0, "ok"]]),
        _fu_case("NaN and out-of-range are clamped", [["S_glint", "nan", 0.9, "ok"], ["S_thermal", 1.7, "inf", "ok"],
                                                       ["S_rf", -3, 0.5, "ok"], ["S_eye", "-inf", "nan", "ok"]]),
        _fu_case("a bad health name is rejected before fuse", [["S_glint", 0.9, 0.9, "ok"], ["S_thermal", 0.9, 0.9, "hot"],
                                                                ["S_rf", 0.9, 0.9, "ok"]]),
        _fu_case("unknown module ignored", [["S_other", 0.9, 0.9, "ok"], [7, 0.9, 0.9, "ok"], ["S_glint", 0.9, 0.9, "ok"]]),
        _fu_case("fallback node: optical + rf, advisory", [["S_glint", 0.8, 0.9, "ok"], ["S_wifi", 0.7, 0.8, "ok"]],
                 fallback_node=True),
        _fu_case("custom weights", [["S_glint", 0.8, 0.9, "ok"], ["S_radar", 0.3, 0.9, "ok"], ["S_voice", 0.6, 0.5, "ok"]],
                 weights=custom),
        _fu_case("custom weights, everything missing", [], weights=custom),
        _fu_case("min_families 3", [["S_glint", 0.9, 0.9, "ok"], ["S_thermal", 0.9, 0.9, "ok"]], min_families=3),
        _fu_case("max_unhealthy 0", [["S_glint", 0.9, 0.9, "ok"], ["S_radar", 0.5, 0.5, "failed"]], max_unhealthy=0),
        _fu_case("rejected weights fall back to the prior", [["S_glint", 0.9, 0.9, "ok"]], weights='{"w": [1, 2]}'),
    ]
    rng = random.Random(FU_SEED)
    for k in range(FU_RANDOM_CASES):
        cases.append(_fu_case(f"random {k}", _fu_random(rng)))
    weight_texts = [
        ("valid", custom), ("version missing", '{"w":[1,1,1,1,1,1],"w_mask":[0,0,0,0,0,0],"b":0}'),
        ("version int", '{"w":[1,1,1,1,1,1],"w_mask":[0,0,0,0,0,0],"b":0,"version":3}'),
        ("version null", '{"w":[1,1,1,1,1,1],"w_mask":[0,0,0,0,0,0],"b":0,"version":null}'),
        ("numbers as strings and bools", '{"w":["1.5",true,false,1,2,3],"w_mask":[0,0,0,0,0,0],"b":"-2","version":"s"}'),
        ("wrong shape", '{"w":[1,1,1],"w_mask":[0,0,0,0,0,0],"b":0}'),
        ("nested", '{"w":[[1],[1],[1],[1],[1],[1]],"w_mask":[0,0,0,0,0,0],"b":0}'),
        ("NaN weight", '{"w":[NaN,1,1,1,1,1],"w_mask":[0,0,0,0,0,0],"b":0}'),
        ("huge b overflows to inf", '{"w":[1,1,1,1,1,1],"w_mask":[0,0,0,0,0,0],"b":1e999}'),
        ("b missing", '{"w":[1,1,1,1,1,1],"w_mask":[0,0,0,0,0,0]}'),
        ("b null", '{"w":[1,1,1,1,1,1],"w_mask":[0,0,0,0,0,0],"b":null}'),
        ("bad string", '{"w":["x",1,1,1,1,1],"w_mask":[0,0,0,0,0,0],"b":0}'),
        ("not an object", "[1,2,3]"), ("not JSON", "{w:"), ("empty", ""),
    ]
    weights = []
    for name, text in weight_texts:
        m = _fu_weights(text)
        weights.append({"name": name, "json": text, "accepted": m is not None,
                        "model": json.loads(m.to_json()) if m is not None else None})
    return {"constants": constants, "scores": scores, "cases": cases, "weights": weights}


# ---- Power governor (android/core PowerGovernor.kt) -----------------------------------------------------------
PG_SIM_DURATION_S = 1800.0   # power_sim.DEFAULT_DURATION_S: every reading the closed loop gave the governor
PG_EPS = 1e-9                # just inside a dwell edge


def _pg_reading(r: dict) -> PG.PowerReading:
    return PG.PowerReading(**{k: nf(v) for k, v in r.items()})


def _pg_timeline(name: str, steps: list, power: dict | None = None) -> dict:
    """steps: reading dicts (optionally with "now"). Each step records the Python governor's level and reasons."""
    settings = PG.PowerSettings.from_config(power)
    gov = PG.Governor(settings)
    out = []
    for st in steps:
        r = {k: v for k, v in st.items() if k != "now"}
        d = gov.update(_pg_reading(r), st.get("now"))
        out.append({**st, "level": d.level, "reasons": list(d.reasons)})
    tl: dict = {"name": name, "steps": out}
    if power:
        tl["power"] = power
        tl["plans"] = {lv: p.to_dict() for lv, p in settings.plans.items()}
        tl["thresholds"] = settings.thresholds
    return tl


def _pg_steps(spec: list, **common) -> list:
    """(t, {fields}) pairs -> reading dicts."""
    return [{"t": t, **common, **f} for t, f in spec]


def _pg_sim(scene: str) -> dict:
    seen: list = []

    class Recording(PG.Governor):
        def update(self, reading, now=None):
            seen.append(reading)
            return super().update(reading, now)

    with mock.patch.object(PSIM, "Governor", Recording):
        rows = PSIM.run(scene, PG_SIM_DURATION_S)
    readings = [[r.t, r.battery_pct, r.charging, r.skin_c, r.device_c] for r in seen]
    return {"scene": scene, "readings": readings, "levels": [r["level"] for r in rows],
            "reasons": [r["reasons"] for r in rows]}


def _pg_gate(max_fps, stride, ts: list) -> dict:
    out = {"max_fps": max_fps, "stride": stride}
    try:
        g = DUTY.FrameGate(nf(max_fps), stride)
    except ValueError:  # python-brain: allow SILENT_EXCEPTION - FrameGate refusing its arguments is the recorded outcome; Kotlin must refuse too
        return {**out, "rejected": True}
    return {**out, "rejected": False, "t": ts, "admit": [g.admit(nf(t)) for t in ts]}


def _pg_listen(duty, period, ts: list) -> dict:
    out = {"duty": duty, "period_s": period}
    try:
        ld = DUTY.ListenDuty(nf(duty), nf(period))
    except ValueError:  # python-brain: allow SILENT_EXCEPTION - ListenDuty refusing its arguments is the recorded outcome; Kotlin must refuse too
        return {**out, "rejected": True}
    return {**out, "rejected": False, "t": ts, "active": [ld.active(nf(t)) for t in ts]}


def power_governor() -> dict:
    P = PG.POWER_DEFAULTS
    constants = {"levels": list(DUTY.LEVELS), "reasons": list(PG.REASONS), "defaults": P,
                 "ranges": {k: list(v) for k, v in PG.POWER_RANGES.items()},
                 "battery_pct_range": list(PG.BATTERY_PCT_RANGE), "temp_plausible_c": list(PG.TEMP_PLAUSIBLE_C),
                 "plan_keys": list(DUTY.PLAN_KEYS), "plan_ranges": {k: list(v) for k, v in DUTY.PLAN_RANGES.items()},
                 "default_plans": DUTY.DEFAULT_PLANS, "frame_gate_tolerance_s": DUTY.FrameGate.TOLERANCE_S,
                 "thermal_status": {"none": PG.THERMAL_STATUS_NONE, "light": PG.THERMAL_STATUS_LIGHT,
                                    "moderate": PG.THERMAL_STATUS_MODERATE, "severe": PG.THERMAL_STATUS_SEVERE,
                                    "critical": PG.THERMAL_STATUS_CRITICAL, "emergency": PG.THERMAL_STATUS_EMERGENCY,
                                    "shutdown": PG.THERMAL_STATUS_SHUTDOWN}}
    thermal = [{"status": s, "level": PG.thermal_status_level(s)}
               for s in [None, True, False, -1, 0, 1, 2, 3, 4, 5, 6, 7, 2.0, "2"]]
    dw, st = P["min_dwell_s"], P["stale_after_s"]
    timelines = [
        _pg_timeline("battery edges, hysteresis and dwell", _pg_steps([
            (0.0, {"battery_pct": P["battery_saver_enter_pct"] + 1e-4}), (1.0, {"battery_pct": P["battery_saver_enter_pct"]}),
            (5.0, {"battery_pct": P["battery_saver_exit_pct"] - 0.01}), (dw, {"battery_pct": P["battery_saver_exit_pct"]}),
            (1.0 + dw - PG_EPS, {"battery_pct": P["battery_saver_exit_pct"]}),
            (1.0 + dw, {"battery_pct": P["battery_saver_exit_pct"]}),
            (40.0, {"battery_pct": P["battery_critical_enter_pct"]}), (50.0, {"battery_pct": P["battery_critical_exit_pct"] - 0.01}),
            (80.0, {"battery_pct": P["battery_critical_exit_pct"] - 0.01}), (81.0, {"battery_pct": P["battery_critical_exit_pct"]}),
            (100.0, {"battery_pct": 90.0}), (111.0, {"battery_pct": 90.0}), (112.0, {"battery_pct": 90.0}),
            (120.0, {"battery_pct": 3.0}), (121.0, {"battery_pct": 0.0}), (160.0, {"battery_pct": 100.0}),
            (190.0, {"battery_pct": 100.0}), (220.0, {"battery_pct": 100.0}), (250.0, {"battery_pct": 100.0})],
            charging=False)),
        _pg_timeline("skin edges", _pg_steps([
            (0.0, {"skin_c": P["skin_hot_enter_c"] - 0.01}), (1.0, {"skin_c": P["skin_hot_enter_c"]}),
            (40.0, {"skin_c": P["skin_hot_exit_c"] + 0.1}), (41.0, {"skin_c": P["skin_hot_exit_c"]}),
            (50.0, {"skin_c": P["skin_critical_enter_c"]}), (90.0, {"skin_c": P["skin_critical_exit_c"] + 0.1}),
            (91.0, {"skin_c": P["skin_critical_exit_c"]}), (100.0, {"skin_c": P["skin_hot_exit_c"]}),
            (121.0, {"skin_c": P["skin_hot_exit_c"]}), (122.0, {"skin_c": 25.0})])),
        _pg_timeline("device edges", _pg_steps([
            (0.0, {"device_c": P["device_hot_enter_c"]}), (10.0, {"device_c": P["device_critical_enter_c"] - 0.01}),
            (20.0, {"device_c": P["device_critical_enter_c"]}), (60.0, {"device_c": P["device_critical_exit_c"] + 0.01}),
            (61.0, {"device_c": P["device_critical_exit_c"]}), (95.0, {"device_c": P["device_hot_exit_c"] + 0.01}),
            (96.0, {"device_c": P["device_hot_exit_c"]})])),
        _pg_timeline("the tighter axis wins; relaxing goes one step per dwell", _pg_steps([
            (0.0, {"battery_pct": 15.0, "skin_c": 44.0}), (10.0, {"battery_pct": 50.0, "skin_c": 30.0}),
            (30.0, {"battery_pct": 50.0, "skin_c": 30.0}), (59.0, {"battery_pct": 50.0, "skin_c": 30.0}),
            (60.0, {"battery_pct": 50.0, "skin_c": 30.0}), (90.0, {"battery_pct": 50.0, "skin_c": 30.0}),
            (100.0, {"battery_pct": 15.0, "skin_c": 42.0}), (130.0, {"battery_pct": 15.0, "skin_c": 30.0})])),
        _pg_timeline("stale: tightens, never relaxes", [
            {"t": 0.0, "skin_c": 42.0}, {"t": 10.0, "skin_c": 30.0, "now": 10.0 + st},
            {"t": 40.0, "skin_c": 30.0, "now": 40.0 + st + 0.001}, {"t": 50.0, "skin_c": 44.0, "now": 200.0},
            {"t": 210.0, "skin_c": 30.0, "now": 400.0}, {"t": None, "skin_c": 30.0, "now": 500.0},
            {"t": True, "skin_c": 30.0, "now": 501.0}, {"t": "nan", "skin_c": 30.0, "now": 502.0},
            {"t": 600.0, "skin_c": 30.0}, {"t": 700.0, "skin_c": 30.0}, {"t": 800.0, "skin_c": 30.0}]),
        _pg_timeline("no t and no now: time is the last level change", [
            {"t": None, "battery_pct": 10.0}, {"t": None, "battery_pct": 90.0}, {"t": 0.0, "battery_pct": 90.0},
            {"t": 31.0, "battery_pct": 90.0}]),
        _pg_timeline("charging cancels battery, not heat", [
            {"t": 0.0, "battery_pct": 5.0, "charging": True}, {"t": 1.0, "battery_pct": 5.0, "charging": None},
            {"t": 2.0, "battery_pct": 5.0, "charging": False}, {"t": 3.0, "battery_pct": 5.0, "charging": True, "skin_c": 42.0},
            {"t": 4.0, "battery_pct": 5.0, "charging": True, "skin_c": 42.0, "now": 100.0},
            {"t": 100.0, "battery_pct": 5.0, "charging": False, "skin_c": 30.0, "now": 200.0},
            {"t": 200.0, "battery_pct": 50.0, "charging": "yes"}, {"t": 240.0, "battery_pct": 50.0, "charging": 1}]),
        _pg_timeline("thermal status", [{"t": float(k), "thermal_status": s} for k, s in enumerate(
            [0, 1, 2, None, 3, True, 2.0, -1, 7, 4, 5, 6, 0, 1])] + [
            {"t": 100.0, "thermal_status": 0}, {"t": 140.0, "thermal_status": 0}, {"t": 180.0, "thermal_status": 0}]),
        _pg_timeline("unknown values keep the last state", [
            {"t": 0.0, "battery_pct": 10.0, "skin_c": 42.0}, {"t": 40.0, "battery_pct": 150.0, "skin_c": -50.0},
            {"t": 80.0, "battery_pct": -1.0, "skin_c": 200.0, "device_c": "inf"}, {"t": 120.0, "battery_pct": True, "skin_c": "nan"},
            {"t": 160.0, "battery_pct": None, "skin_c": None}, {"t": 200.0, "battery_pct": 100, "skin_c": 30}]),
        _pg_timeline("custom thresholds and plans", _pg_steps([
            (0.0, {"battery_pct": 39.0}), (5.0, {"battery_pct": 50.0}), (6.0, {"battery_pct": 51.0}),
            (7.0, {"skin_c": 38.0}), (8.0, {"skin_c": 36.0}), (9.0, {"skin_c": 35.9}), (20.0, {"skin_c": 35.0})]),
            power={"battery_saver_enter_pct": 40, "battery_saver_exit_pct": 50.5, "skin_hot_enter_c": 37.0,
                   "skin_hot_exit_c": 36.0, "min_dwell_s": 0.0, "stale_after_s": 1.0,
                   "plans": {"saver": {"camera_max_fps": 12, "frame_stride": 3.0}, "hot": {"listen_duty": 0.0005}}}),
    ]
    validation_cfgs = [
        ("defaults", {}), ("not an object", [1]), ("unknown keys", {"zz": 1, "aa": 2}),
        ("out of range and wrong types", {"battery_saver_enter_pct": 101, "skin_hot_enter_c": True, "min_dwell_s": "30",
                                          "stale_after_s": 0.5, "device_hot_enter_c": None}),
        ("battery hysteresis inverted", {"battery_saver_exit_pct": 20.0, "battery_critical_exit_pct": 5.0}),
        ("critical above saver", {"battery_critical_enter_pct": 21.0, "battery_critical_exit_pct": 30.0}),
        ("heat hysteresis inverted", {"skin_hot_exit_c": 41.0, "device_critical_exit_c": 90.0}),
        ("hot above critical", {"device_hot_enter_c": 86.0, "device_hot_exit_c": 84.0}),
        ("plans not an object", {"plans": 3}), ("plans null", {"plans": None}),
        ("plans: unknown level and bad level", {"plans": {"turbo": {}, "saver": 1}}),
        ("plan: unknown key, ranges, bools", {"plans": {"saver": {"fps": 1, "camera_max_fps": 61, "frame_stride": 2.5,
                                                                  "voice_match": 1, "radio_scan_interval_s": 0}}}),
        ("plan: whole-number float stride is fine", {"plans": {"critical": {"frame_stride": 5.0}}}),
        ("plan order", {"plans": {"saver": {"camera_max_fps": 31, "frame_stride": 1, "ir_illuminator": False},
                                  "hot": {"ir_illuminator": True, "listen_duty": 0.6, "radio_scan_interval_s": 20}}}),
        ("plan order: repr of a small float", {"plans": {"saver": {"listen_duty": 0.0005}, "hot": {"listen_duty": 0.25}}}),
        ("critical minimum", {"plans": {"critical": {"camera_max_fps": 0.5, "voice_match": True, "face_match": True},
                                        "hot": {"voice_match": True, "face_match": True}, "saver": {}, "normal": {}}}),
    ]
    validation = []
    for name, cfg in validation_cfgs:
        errs = PG.validate_power(cfg)
        rec = {"name": name, "power": cfg, "errors": errs}
        if not errs:
            s = PG.PowerSettings.from_config(cfg)
            rec["thresholds"] = s.thresholds
            rec["plans"] = {lv: p.to_dict() for lv, p in s.plans.items()}
        validation.append(rec)
    fps30 = [k / 30 for k in range(12)]
    gates = [
        _pg_gate(30.0, 1, fps30), _pg_gate(30, 2, fps30), _pg_gate(15.0, 1, fps30), _pg_gate(10.0, 3, [k / 60 for k in range(40)]),
        _pg_gate(0.0, 1, [0.0, 1.0, 2.0]), _pg_gate(20.0, 1, [0.0, 0.05 - 1e-10, 0.05, 0.1 - 2e-9, 0.12, 0.02, 0.07, 0.069]),
        _pg_gate(30.0, 1, [0.0, 0.5, "nan", 0.6, 0.6]), _pg_gate(1.0, 2, [-5.0, -4.0, -3.0, -2.5, -2.0, 100.0]),
        _pg_gate(-1.0, 1, []), _pg_gate("nan", 1, []), _pg_gate("inf", 1, []), _pg_gate(True, 1, []), _pg_gate(30.0, 0, []),
        _pg_gate(30.0, 2.0, []), _pg_gate(30.0, True, []), _pg_gate("30", 1, []),
    ]
    listen_t = [0.0, 1.0, 4.999, 5.0, 9.99, 10.0, 12.5, 15.0, -0.5, -5.0, -5.000001, -10.0, 1e9 + 3.0, "inf", "nan"]
    listens = [
        _pg_listen(0.5, 10.0, listen_t), _pg_listen(1.0, 10.0, listen_t), _pg_listen(0.0, 10.0, listen_t),
        _pg_listen(0.25, 3.3, [k * 0.37 for k in range(-12, 30)]), _pg_listen(0.999, 1, [0.9989, 0.999, 1.9989, 2.0]),
        _pg_listen(1.5, 10.0, []), _pg_listen(-0.1, 10.0, []), _pg_listen(0.5, 0.0, []), _pg_listen(0.5, -1.0, []),
        _pg_listen("nan", 10.0, []), _pg_listen(0.5, "inf", []), _pg_listen(True, 10.0, []), _pg_listen(0.5, False, []),
    ]
    sims = [_pg_sim(s) for s in PSIM.SCENES]
    return {"constants": constants, "thermal_status": thermal, "validation": validation, "timelines": timelines,
            "sim": sims, "frame_gate": gates, "listen_duty": listens}


# ---- Painting the glasses display list (android/core HudPainter.kt) ---------------------------------------------
# Draw calls are JSON lists (paint.py tuples); Kotlin rebuilds each HUD / guide display list with GlassesHud, or parses
# each frame with HudFrame.parse, paints it on a RecordingCanvas and must record exactly these calls.
HP_FLOORDIV = [(1.0, 0.1), (13.7, 2.0), (11.0, 2.0), (-7.0, 2.0), (7.0, -2.0), (-0.0, 24.0), (0.0, 24.0),
               (8 * 13.7, 24.0), (24 * 7.3, 24.0), (267.5, 24.0), (-5.5, 0.5), (1e-9, 24.0)]
HP_COLOURS = ["#E6E6E6", "#e6e6e6", "#fff", "#FFF", "#000000", "#9e9E9e", "", "none", "red", "#12345", "#1234567",
              "E6E6E6", "#ggg", "#fff ", " #fff", "#ＦＦＦ", "#ffff", "#"]


def hud_paint() -> dict:
    hud = [("idle", gh_state(), [])]
    for lv in T.LEVELS:
        for moving in (False, True):
            hud.append((f"level {lv}, moving {moving}", gh_state(level=lv, alert_text="Signatures A1 A3", new_sighting=True,
                                                                 moving=moving), []))
    hud.append(("moving, alert hidden, cue text kept", gh_state(alert_text="Heat source", new_sighting=True, moving=True), []))
    for word in sorted(SLOT_VALUES["direction"]):
        for moving in (False, True):
            hud.append((f"cue {word}, moving {moving}", gh_state(level="medium", alert_text="Infrared emitter " + word,
                                                                 moving=moving, cue_direction=word), []))
    hud += [
        ("all zones, grayscale thermal", gh_state(level="confirmed", alert_text="Cold object near door", new_sighting=True,
                                                  thermal=True, health=dict(GH_HEALTH), locked=True, panic_armed=True,
                                                  voice_tier="high", cue_direction="behind"), []),
        ("thermal colour overlay", gh_state(thermal=True, thermal_colour=True), []),
        ("health: every status", gh_state(health={"z": "ok", "a": "down", "m": "degraded", "b": "ok"}), []),
        ("markup and unicode text kept raw", gh_state(alert_text="Objet <froid> & \"près\" → \U0001f525"), []),
        ("larger canvas", gh_state(level="low", alert_text="Heat source", cue_direction="at right"),
         [(["glasses", "canvas"], [800, 480])]),
        ("odd canvas: fractional cue", gh_state(cue_direction="above"), [(["glasses", "canvas"], [641, 401])]),
        ("brightness and type from tokens", gh_state(level="low", alert_text="Heat source"),
         [(["glasses", "default_brightness"], 0.6), (["type", "glasses_body"], 24), (["type", "font_family"], "serif")]),
    ]
    cases = []
    for name, s, patch in hud:
        st = GlassesState(**{**s, "thermal": GH_FRAME if s["thermal"] else None})
        cases.append({"name": name, "state": s, "tokens_patch": [[p, v] for p, v in patch],
                      "calls": P.paint(display_list(st, gh_tokens(patch)))})

    every = list(GUIDE_ORDER)
    guide = [("first angle", {"needed": every, "last_ok": None, "last_reason": "", "complete": False}),
             ("accepted: tick", {"needed": every[2:], "last_ok": True, "last_reason": "", "complete": False}),
             ("rejected: reason", {"needed": every[3:], "last_ok": False, "last_reason": "move closer", "complete": False}),
             ("complete", {"needed": [], "last_ok": True, "last_reason": "", "complete": True})]
    gcases = []
    for name, g in guide:
        eg = EnrollGuide(list(g["needed"]), g["last_ok"], g["last_reason"], g["complete"])
        gcases.append({"name": name, "guide": g, "tokens_patch": [], "calls": P.paint(enroll_guide_display_list(eg))})

    canvas = display_list(GlassesState())[0]
    W, H = canvas["w"], canvas["h"]
    text = {"op": "text", "id": "G2", "x": 8, "y": 58, "text": "Signal nearby", "size": 20, "colour": "#E6E6E6",
            "anchor": "start"}
    shape = {"op": "shape", "id": "s", "kind": "circle", "x": 16.25, "y": 14.5, "size": 13.7, "fill": "#56B4E9",
             "stroke": "#fff", "stroke_width": 1.5, "tag": ""}
    full = GlassesState(level="medium", alert_text="Cold object near door", new_sighting=True, thermal=GH_FRAME,
                        health=dict(GH_HEALTH), locked=True, panic_armed=True, cue_direction="at left")
    frames = [
        ("the HUD through the link", _lo_frame(display_list(full))),
        ("the guide through the link", _lo_frame(enroll_guide_display_list(EnrollGuide(["up", "down"], True, "", False)))),
        ("every shape kind, fractional sizes", _lo_frame([canvas] + [{**shape, "kind": k} for k in
                                                                     ("circle", "triangle", "diamond", "dot", "tick")])),
        ("integer sizes: odd dot, small tick", _lo_frame([canvas, {**shape, "kind": "dot", "x": 30, "y": 40, "size": 11,
                                                                   "stroke_width": 2},
                                                          {**shape, "kind": "tick", "x": 100, "y": 20, "size": 7}])),
        ("unknown shape kinds skipped", _lo_frame([canvas, {**shape, "kind": "hexagon"}, {**shape, "kind": ""}, text])),
        ("anchors: unknown is start", _lo_frame([canvas] + [{**text, "anchor": a} for a in ("start", "middle", "end", "left", "")])),
        ("colours passed as given", _lo_frame([canvas, {**text, "colour": "#fff"}, {**text, "colour": "red"},
                                               {**shape, "kind": "dot", "fill": "", "stroke": "#12345"}])),
        ("hostile values clamped, then painted", _lo_frame([canvas, {**text, "x": -40, "y": H + 500},
                                                            {**shape, "kind": "dot", "x": W + 0.75, "y": -3.25},
                                                            {**shape, "kind": "triangle", "size": 9000, "stroke_width": -2}])),
        ("opacity and font size clamped", _lo_frame([{**canvas, "opacity": 7, "font_size": 10000.5}, text])),
        ("cue and image from the pod", _lo_frame([canvas, {"op": "cue", "id": "cue", "direction": "x", "edge": "nowhere",
                                                          "x": 300.5, "y": 0, "w": 39, "h": 8, "colour": "#E6E6E6"},
                                                  {"op": "image", "id": "G3", "x": 440, "y": 230, "w": 180, "h": 135,
                                                   "source": "thermal", "mode": "sepia"}])),
        ("zones only: the frame alone", _lo_frame([canvas] + [{"op": "zone", "id": "G1", "x": 0, "y": 0, "w": W, "h": 28}])),
        ("markup and unicode text kept raw", _lo_frame([canvas, {**text, "text": "<b>x</b> & \U0001f525 →"}])),
        ("second canvas skipped", _lo_frame([canvas, text, canvas, shape])),
        ("first op not a canvas: nothing", _lo_frame([text, canvas, shape])),
        ("unchanged: nothing", LO.dumps({"ok": True, "version": 4, "unchanged": True})),
    ]
    fcases = [{"name": n, "reply": r, "calls": P.paint(LO.parse_hud(r)["ops"])} for n, r in frames]

    constants = {"weight_normal": P.WEIGHT_NORMAL, "anchors": list(P.ANCHORS), "no_paint": P.NO_PAINT,
                 "no_stroke_width": P.NO_STROKE_WIDTH, "square": P.SQUARE, "mark_kinds": list(P.MARK_KINDS),
                 "call_kinds": list(P.CALL_KINDS)}
    floordiv = [[a, b, a // b] for a, b in HP_FLOORDIV]
    colours = [[s, P.parse_colour(s)] for s in HP_COLOURS]
    return {"constants": constants, "hud": cases, "guide": gcases, "frames": fcases, "floordiv": floordiv,
            "colours": colours}


def differences(a, b, path: str = "$", out: list | None = None) -> list[str]:
    """Where `b` differs from `a`, at most MAX_REPORTED entries. Floats agree within FLOAT_TOL; every other value,
    the int/float distinction and the structure must match exactly."""
    out = [] if out is None else out
    if len(out) >= MAX_REPORTED:
        return out
    if type(a) is not type(b):
        out.append(f"{path}: {a!r} != {b!r}")
    elif isinstance(a, float):
        if not (a == b or (math.isfinite(a) and math.isfinite(b) and abs(a - b) <= FLOAT_TOL)):
            out.append(f"{path}: {a!r} != {b!r}")
    elif isinstance(a, dict):
        for k in a.keys() | b.keys():
            if k not in a or k not in b:
                out.append(f"{path}.{k}: only in {'the file' if k in a else 'the regenerated vectors'}")
            else:
                differences(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: {len(a)} items != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            differences(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")
    return out[:MAX_REPORTED]


def main() -> int:
    out = {"ble_framing.json": ble(), "thermal_render.json": thermal(), "eye_android.json": eye(), "voice_link.json": voice_link(),
           "glasses_hud.json": glasses_hud(), "time_sync.json": time_sync(), "link_ops.json": link_ops(),
           "voice_enrollment.json": voice_enrollment(), "fusion.json": fusion(),
           "power_governor.json": power_governor(), "hud_paint.json": hud_paint()}
    check = "--check" in sys.argv
    bad = 0
    for name, d in out.items():
        p = ROOT / "tests/vectors" / name
        txt = json.dumps(d, separators=(",", ":")) + "\n"
        if check:
            diffs = ["file missing"] if not p.exists() else differences(json.loads(p.read_text()), json.loads(txt))
            if diffs:
                print(f"stale: {p}")
                for line in diffs:
                    print("  " + line)
                bad += 1
        else:
            p.write_text(txt)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
