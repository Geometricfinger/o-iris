"""Startup config: validated, safe defaults, feature flags per module (M-01..M-17)."""
from __future__ import annotations
import math
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .guard.vocabulary import violations
# Pure stdlib modules that never import this one (ADR-0061): the "power" and "budgets" sections are validated there.
from .power.budgets import validate_budgets
from .power.governor import validate_power

# Allowed range for face_match.pose_up_delta / pose_down_delta (M-16 enrollment, up/down judged against the
# person's own front pitch). 0.02 keeps a tilt distinguishable from landmark jitter; 0.30 is about half the
# backend's whole level-to-tilt pitch span, beyond which no real head tilt would pass. Provisional.
POSE_DELTA_MIN, POSE_DELTA_MAX = 0.02, 0.30

MODULES = {
    "M-01": "eye_behavior", "M-02": "liveness", "M-03": "signal_fingerprint", "M-04": "security",
    "M-05": "sighting_log", "M-06": "voice_output", "M-07": "location_baselines", "M-08": "thermal_view",
    "M-09": "headless", "M-10": "build", "M-11": "sensor_bus", "M-12": "feedback_loop",
    "M-13": "android_fallback", "M-14": "sweep_timeline", "M-15": "app_gate",
    "M-16": "face_match",   # optional add-on (ADR-0049/0050): off by default, not imported until enabled
    "M-17": "voice_link",   # add-on to M-16 personal mode (ADR-0057): off by default, not imported until enabled
}
# Modules that cannot be disabled: security and the bus are load-bearing; guards are not modules.
LOCKED_ON = {"M-04", "M-11"}
# Gate-dependent modules ship disabled until their experiment passes (spec section 11, M-gates).
DEFAULT_OFF = {"M-01", "M-02", "M-16", "M-17"}


class ConfigError(ValueError):
    pass


# M-17 voice link settings (ADR-0057, docs/VOICE_LINK_MODULE_SPEC.md): defaults and allowed ranges.
# oiris.modules.voice_link.core.VoiceLinkConfig mirrors these; a test keeps the key sets in sync.
VOICE_LISTENING_MODES = ("ambient", "push_to_talk")   # ADR-0057 decision 2: ambient chosen
# ADR-0058: a clear match plays a chime and says "sounds like NAME" (the owner's choice), or a chime only, or nothing.
VOICE_ANNOUNCE_MODES = ("chime_and_name", "chime", "off")
ANNOUNCE_REPEAT_DEFAULT_S = 60.0               # the same person is announced at most once per this many seconds
ANNOUNCE_REPEAT_RANGE_S = (5.0, 3600.0)
# ADR-0058 decision 9: a voice_antispoof score at or above this drops the match. Only used when that slot is set.
SPOOF_THRESHOLD_DEFAULT = 0.5
SPOOF_THRESHOLD_RANGE = (0.05, 0.95)
VOICE_LINK_DEFAULTS: dict = {
    "listening": "ambient",
    # Raw cosine similarity between a speech embedding and an enrolled template. PROVISIONAL: set from the
    # model's published operating point; to be measured in the consented live check (spec 7, step 6).
    "match_threshold": 0.5,
    "unsure_margin": 0.05,       # second-best enrolled voice this close to the best => "unsure", nothing logged
    # Shortest utterance matched in live listening. PROVISIONAL (2.0, was 1.5) until more live checks: in the first
    # live check (2026-11-16, one person, built-in microphone, near field) a 1.8 s piece scored 0.15 and the 2.2 s and
    # 2.3 s pieces scored 0.93 and 0.88 (a 3.5 s piece scored 0.44, so length is not the only factor).
    "min_speech_s": 2.0,
    # Enrollment: one designed sentence per window (oiris/modules/voice_link/prompts.py), six of the eight by default.
    "enroll_windows": 6,
    "enroll_window_s": 7.0,
    "enroll_min_speech_s": 1.8,
    # Total speech over all accepted windows. Below it, enrollment asks for more sentences (up to
    # VOICE_ENROLL_MAX_WINDOWS windows); the first live check's six windows held about 20 s.
    "enroll_min_total_speech_s": 20.0,
    # Advisory range check: the "high"/"rising" windows' median pitch should be at least this fraction above
    # the "low" windows'. Never blocks saving; pitch is measured in RAM only and never stored.
    "enroll_range_min_rise": 0.08,
    "heard_retention_days": 30,
    "agree_window_s": 5.0,       # a face match this recent => "voice and face agree"
    "card_seconds": 10.0,        # dashboard "heard" card (a later task)
    "vad_snr_db": 10.0,
    "vad_floor_dbfs": -55.0,
    "speaker_model": "sherpa_campp",
    "model_dir": "",
    "input_device": "",          # index or name substring; "" = system default
    "allow_test_backend": False,  # tests only; the test speaker plugin is refused otherwise
    "announce": VOICE_ANNOUNCE_MODES[0],
    "announce_repeat_s": ANNOUNCE_REPEAT_DEFAULT_S,
    "spoof_threshold": SPOOF_THRESHOLD_DEFAULT,
}
VOICE_ENROLL_MAX_WINDOWS = 10   # at most this many accepted windows, extra sentences for the total-speech target included
# key -> (low, high); enroll_min_speech_s's upper bound is the configured enroll_window_s (None here).
VOICE_LINK_RANGES: dict = {
    "match_threshold": (0.2, 0.95), "unsure_margin": (0.0, 0.3), "min_speech_s": (0.5, 10.0),
    "enroll_windows": (3, VOICE_ENROLL_MAX_WINDOWS), "enroll_window_s": (3.0, 15.0), "enroll_min_speech_s": (0.5, None),
    "enroll_min_total_speech_s": (5.0, 120.0),
    "enroll_range_min_rise": (0.0, 0.5),
    "heard_retention_days": (1, 365), "agree_window_s": (1.0, 30.0), "card_seconds": (3.0, 60.0),
    "vad_snr_db": (3.0, 30.0), "vad_floor_dbfs": (-80.0, -20.0),
    "announce_repeat_s": ANNOUNCE_REPEAT_RANGE_S, "spoof_threshold": SPOOF_THRESHOLD_RANGE,
}
VOICE_LINK_INTS = {"enroll_windows", "heard_retention_days"}
VOICE_LINK_KEYS = set(VOICE_LINK_DEFAULTS)


def _finite_number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate_voice_link(d: dict, allow_test_backend: bool = False) -> list[str]:
    """Validate M-17 settings without importing the module (it stays unloaded while off). The module's
    VoiceLinkConfig runs the same checks, passing allow_test_backend=True (tests construct it directly)."""
    if not isinstance(d, dict):
        return ["voice_link must be an object"]
    errs = []
    unknown = set(d) - VOICE_LINK_KEYS
    if unknown:
        errs.append(f"voice_link: unknown keys {sorted(unknown)}")
    v = {**VOICE_LINK_DEFAULTS, **{k: x for k, x in d.items() if k in VOICE_LINK_KEYS}}
    if v["listening"] not in VOICE_LISTENING_MODES:
        errs.append(f"voice_link.listening must be one of {list(VOICE_LISTENING_MODES)}")
    if v["announce"] not in VOICE_ANNOUNCE_MODES:
        errs.append(f"voice_link.announce must be one of {list(VOICE_ANNOUNCE_MODES)}")
    for k, (lo, hi) in VOICE_LINK_RANGES.items():
        x = v[k]
        if hi is None:   # enroll_min_speech_s: at most one enrollment window
            hi = v["enroll_window_s"] if _finite_number(v["enroll_window_s"]) else VOICE_LINK_RANGES["enroll_window_s"][1]
        if not _finite_number(x) or (k in VOICE_LINK_INTS and int(x) != x) or not lo <= x <= hi:
            errs.append(f"voice_link.{k} must be in [{lo}, {hi}]")
    for k in ("speaker_model", "model_dir"):
        if not isinstance(v[k], str):
            errs.append(f"voice_link.{k} must be a string")
    if not v["speaker_model"]:
        errs.append("voice_link.speaker_model must name a speaker_embed plugin")
    if isinstance(v["input_device"], bool) or not isinstance(v["input_device"], (str, int)):
        errs.append("voice_link.input_device must be a device index or a name substring")
    if not isinstance(v["allow_test_backend"], bool):
        errs.append("voice_link.allow_test_backend must be true or false")
    elif v["allow_test_backend"] and not allow_test_backend:
        errs.append("voice_link.allow_test_backend is for tests only")
    return errs


# Output arbiter and motion gate (oiris/modules/output.py, docs/ARCHITECTURE.md "Outputs and motion").
# oiris.modules.output.OutputSettings mirrors these; a test keeps the key sets in sync.
OUTPUT_FAST_MIN = ("confirmed", "panic")      # while moving fast only this priority and above play
OUTPUT_DEFAULTS: dict = {
    "speak_alerts": False,          # sensor alerts are shown, not spoken (the Mac stays silent as before)
    "popups_while_walking": False,  # Module Map 6.1 glasses constraints: no pop-ups during movement; the G1 strip still shows the level
    "fast_min_priority": "confirmed",
    "popup_s": 6.0,                 # a pop-up stays this long unless a higher one replaces it
    "dedupe_s": 30.0,               # the same key again inside this window is deduped (panic and voice link exempt)
}
OUTPUT_RANGES: dict = {"popup_s": (1.0, 30.0), "dedupe_s": (0.0, 600.0)}
OUTPUT_BOOLS = ("speak_alerts", "popups_while_walking")
OUTPUT_KEYS = set(OUTPUT_DEFAULTS)


def validate_output(d: dict) -> list[str]:
    if not isinstance(d, dict):
        return ["output must be an object"]
    errs = []
    unknown = set(d) - OUTPUT_KEYS
    if unknown:
        errs.append(f"output: unknown keys {sorted(unknown)}")
    v = {**OUTPUT_DEFAULTS, **{k: x for k, x in d.items() if k in OUTPUT_KEYS}}
    for k in OUTPUT_BOOLS:
        if not isinstance(v[k], bool):
            errs.append(f"output.{k} must be true or false")
    if v["fast_min_priority"] not in OUTPUT_FAST_MIN:
        errs.append(f"output.fast_min_priority must be one of {list(OUTPUT_FAST_MIN)}")
    for k, (lo, hi) in OUTPUT_RANGES.items():
        if not _finite_number(v[k]) or not lo <= v[k] <= hi:
            errs.append(f"output.{k} must be in [{lo}, {hi}]")
    return errs


def validate_face_match(d: dict) -> list[str]:
    """Validate M-16 settings without importing the module (it stays unloaded while off).
    Mirrors oiris.modules.face_match.core.FaceMatchConfig; a test keeps the two in sync."""
    if not isinstance(d, dict):
        return ["face_match must be an object"]
    errs = []
    unknown = set(d) - FACE_MATCH_KEYS
    if unknown:
        errs.append(f"face_match: unknown keys {sorted(unknown)}")
    if any(m not in ("personal", "research") for m in d.get("modes", [])): errs.append("face_match.modes: personal/research only")
    if not 0.5 <= float(d.get("match_threshold", 0.8)) < 1.0: errs.append("face_match.match_threshold must be in [0.5, 1)")
    if not 1 <= int(d.get("encounter_retention_days", 30)) <= 365: errs.append("face_match.encounter_retention_days must be 1..365")
    if d.get("place_source", "named") not in ("named", "gps_0.01"): errs.append("face_match.place_source must be named|gps_0.01")
    if not 5 <= int(d.get("enroll_min_shots", 5)) <= int(d.get("enroll_max_shots", 10)):
        errs.append("face_match.enroll_min_shots must be >= 5 and <= enroll_max_shots")
    for k, dflt in (("pose_up_delta", 0.06), ("pose_down_delta", 0.08)):
        if not POSE_DELTA_MIN <= float(d.get(k, dflt)) <= POSE_DELTA_MAX:
            errs.append(f"face_match.{k} must be in [{POSE_DELTA_MIN}, {POSE_DELTA_MAX}]")
    if d.get("allow_test_backend"): errs.append("face_match.allow_test_backend is for tests only")
    return errs


# Sensor rig, the spatial "ports" (ADR-0060, oiris/spatial/rig.py): where each camera, microphone array or IMU
# sits on the head and which way it points. Validated here without importing numpy; oiris.spatial.rig.Rig
# runs the same checks. An empty "rig" is the default rig (one forward camera "front"); "mounts": [] is no
# coverage at all, never an error.
RIG_KEYS = {"mounts"}
MOUNT_KINDS = ("camera", "thermal", "nir", "mic_array", "imu")
IMAGING_KINDS = ("camera", "thermal", "nir")
MOUNT_KEYS = {"id", "kind", "position_m", "yaw_deg", "pitch_deg", "roll_deg", "hfov_deg", "vfov_deg", "image_size",
              "intrinsics", "mic_positions_m", "ir_illuminator", "ir_eye_safety_approved"}
INTRINSICS_KEYS = {"fx", "fy", "cx", "cy"}
RIG_MAX_MOUNTS = 16
MOUNT_ID_MAX_LEN = 32
MOUNT_ID_RE = re.compile(rf"[A-Za-z0-9_-]{{1,{MOUNT_ID_MAX_LEN}}}")
MOUNT_YAW_RANGE_DEG = (-180.0, 180.0)
MOUNT_PITCH_RANGE_DEG = (-90.0, 90.0)
MOUNT_ROLL_RANGE_DEG = (-180.0, 180.0)
MOUNT_POSITION_MAX_M = 1.0          # a head-worn sensor sits within a metre of the eyes
MOUNT_FOV_RANGE_DEG = (1.0, 170.0)  # a pinhole model cannot reach 180 degrees
IMAGE_SIDE_RANGE_PX = (8, 8192)
MIC_ARRAY_MIN_MICS = 2
# Provisional defaults for an imaging mount that gives neither intrinsics nor fields of view.
DEFAULT_IMAGE_SIZE = (640, 480)
DEFAULT_HFOV_DEG, DEFAULT_VFOV_DEG = 70.0, 55.0
# Build spec v2 section 7.3 (rules 4 and 5): IR LEDs are aimed at objects in front of the wearer, and use on
# strangers needs an accredited IEC 62471 test first. An illuminator aimed more than this far off straight
# ahead (horizontally or vertically) lights people the wearer cannot see, so it is refused unless the mount
# says ir_eye_safety_approved: true (the owner's recorded decision after that review).
IR_ILLUMINATOR_MAX_OFF_AXIS_DEG = 30.0
DEFAULT_RIG: dict = {"mounts": [{"id": "front", "kind": "camera", "position_m": [0.0, 0.0, 0.0],
                                 "yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}]}


def _finite_vec(v: object, n: int) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == n and all(_finite_number(x) for x in v)


def mount_forward_az_el(yaw_deg: float, pitch_deg: float) -> tuple[float, float]:
    """Head-frame azimuth and elevation of a mount's forward axis. Roll turns the image about that axis,
    so it does not move it: R @ [1, 0, 0] = [cos p cos y, cos p sin y, sin p]."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    x, yy, z = math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)
    return math.degrees(math.atan2(yy, x)), math.degrees(math.atan2(z, math.hypot(x, yy)))


def _validate_mount(m: object, i: int) -> list[str]:
    where = f"rig.mounts[{i}]"
    if not isinstance(m, dict):
        return [f"{where} must be an object"]
    errs = []
    unknown = set(m) - MOUNT_KEYS
    if unknown:
        errs.append(f"{where}: unknown keys {sorted(unknown)}")
    mid, kind = m.get("id"), m.get("kind")
    # Mount ids appear in every spatial event the guard checks, so they must be plain and pass the vocabulary guard.
    if not isinstance(mid, str) or not MOUNT_ID_RE.fullmatch(mid) or violations(mid.replace("_", " ").replace("-", " ")):
        errs.append(f"{where}.id must be 1-{MOUNT_ID_MAX_LEN} letters, digits, '_' or '-' in plain wording")
    else:
        where = f"rig mount {mid!r}"
    if kind not in MOUNT_KINDS:
        errs.append(f"{where}.kind must be one of {list(MOUNT_KINDS)}")
    if "position_m" in m and not (_finite_vec(m["position_m"], 3) and all(abs(x) <= MOUNT_POSITION_MAX_M for x in m["position_m"])):
        errs.append(f"{where}.position_m must be [x, y, z] finite metres within {MOUNT_POSITION_MAX_M} m of the eyes")
    for k, (lo, hi) in (("yaw_deg", MOUNT_YAW_RANGE_DEG), ("pitch_deg", MOUNT_PITCH_RANGE_DEG), ("roll_deg", MOUNT_ROLL_RANGE_DEG)):
        x = m.get(k, 0.0)
        if not _finite_number(x) or not lo <= x <= hi:
            errs.append(f"{where}.{k} must be a number in [{lo}, {hi}]")
    imaging = kind in IMAGING_KINDS
    for k in ("hfov_deg", "vfov_deg", "image_size", "intrinsics", "ir_illuminator", "ir_eye_safety_approved"):
        if k in m and not imaging:
            errs.append(f"{where}.{k} applies only to imaging mounts {list(IMAGING_KINDS)}")
    if "mic_positions_m" in m and kind != "mic_array":
        errs.append(f"{where}.mic_positions_m applies only to mic_array mounts")
    for k in ("hfov_deg", "vfov_deg"):
        lo, hi = MOUNT_FOV_RANGE_DEG
        if k in m and (not _finite_number(m[k]) or not lo <= m[k] <= hi):
            errs.append(f"{where}.{k} must be in [{lo}, {hi}] degrees")
    if "image_size" in m:
        s, (lo, hi) = m["image_size"], IMAGE_SIDE_RANGE_PX
        if not (isinstance(s, (list, tuple)) and len(s) == 2 and all(isinstance(x, int) and not isinstance(x, bool) and lo <= x <= hi for x in s)):
            errs.append(f"{where}.image_size must be [width, height] whole pixels in [{lo}, {hi}]")
    if "intrinsics" in m:
        k_ = m["intrinsics"]
        if not (isinstance(k_, dict) and set(k_) == INTRINSICS_KEYS and all(_finite_number(k_[x]) for x in INTRINSICS_KEYS)
                and k_["fx"] > 0 and k_["fy"] > 0):
            errs.append(f"{where}.intrinsics must be {{fx, fy, cx, cy}} finite pixels with fx, fy > 0")
        elif "hfov_deg" in m or "vfov_deg" in m:
            errs.append(f"{where}: give intrinsics or hfov_deg/vfov_deg, not both")
    if kind == "mic_array":
        mics = m.get("mic_positions_m")
        if not (isinstance(mics, (list, tuple)) and len(mics) >= MIC_ARRAY_MIN_MICS and all(_finite_vec(p, 3) for p in mics)):
            errs.append(f"{where}.mic_positions_m needs at least {MIC_ARRAY_MIN_MICS} microphones as [x, y, z] metres")
        elif len({tuple(p) for p in mics}) != len(mics):
            errs.append(f"{where}.mic_positions_m has two microphones at the same position")
    for k in ("ir_illuminator", "ir_eye_safety_approved"):
        if k in m and not isinstance(m[k], bool):
            errs.append(f"{where}.{k} must be true or false")
    if m.get("ir_illuminator") is True and m.get("ir_eye_safety_approved") is not True and not errs:
        az, el = mount_forward_az_el(m.get("yaw_deg", 0.0), m.get("pitch_deg", 0.0))
        if abs(az) > IR_ILLUMINATOR_MAX_OFF_AXIS_DEG or abs(el) > IR_ILLUMINATOR_MAX_OFF_AXIS_DEG:
            errs.append(f"{where}: an IR illuminator aimed more than {IR_ILLUMINATOR_MAX_OFF_AXIS_DEG:g} deg from straight ahead "
                        f"(az {az:.1f}, el {el:.1f}) is refused (build spec v2 section 7.3, rules 4 and 5: aimed at objects, "
                        "an accredited IEC 62471 test before use near strangers); set ir_eye_safety_approved only after that review")
    return errs


def validate_rig(d: dict) -> list[str]:
    """Validate the "rig" section without importing numpy (oiris.spatial stays unloaded until used)."""
    if not isinstance(d, dict):
        return ["rig must be an object"]
    errs = []
    unknown = set(d) - RIG_KEYS
    if unknown:
        errs.append(f"rig: unknown keys {sorted(unknown)}")
    mounts = d.get("mounts", DEFAULT_RIG["mounts"])
    if not isinstance(mounts, list) or len(mounts) > RIG_MAX_MOUNTS:
        return errs + [f"rig.mounts must be a list of at most {RIG_MAX_MOUNTS} mounts"]
    ids: list = []
    for i, m in enumerate(mounts):
        errs += _validate_mount(m, i)
        if isinstance(m, dict):
            ids.append(m.get("id"))
    dup = sorted({x for x in ids if isinstance(x, str) and ids.count(x) > 1})
    if dup:
        errs.append(f"rig.mounts: duplicate ids {dup}")
    return errs


# Capture rules (ADR-0063, oiris/capture/): the bystander recording light, the hardware mic mute and the interlock.
# Validated here; capture_settings() resolves the profile defaults that oiris/live.py builds the interlock from.
CAPTURE_PROFILES = ("mac", "rig", "glasses", "helmet")
CAPTURE_INDICATOR_MODES = ("required", "off")
CAPTURE_MUTE_MODES = ("required", "if_present", "off")
CAPTURE_ADAPTERS = ("none", "sim", "gpio")
CAPTURE_FORCED_INDICATOR = ("glasses", "helmet")   # worn among people: the recording light cannot be turned off
CAPTURE_PROFILE_DEFAULTS: dict = {
    # The Mac's recording light is off by default: macOS shows its own camera light and microphone dot, which
    # O Iris cannot control (and cannot read back), so a second software light would add nothing.
    "mac": {"indicator": "off", "mute_sense": "if_present", "indicator_adapter": "none", "mute_adapter": "none"},
    # The Pi rig: spec B-18 ("with the switch off, 0 non-zero samples") makes the mute sense required.
    "rig": {"indicator": "required", "mute_sense": "required", "indicator_adapter": "gpio", "mute_adapter": "gpio"},
    # Glasses and helmet are not chosen yet (ADR-0059): their light must be configured (sim until it is wired).
    "glasses": {"indicator": "required", "mute_sense": "if_present", "indicator_adapter": "none", "mute_adapter": "none"},
    "helmet": {"indicator": "required", "mute_sense": "if_present", "indicator_adapter": "none", "mute_adapter": "none"},
}
CAPTURE_CONFIRM_TIMEOUT_MS = (50, 2000)
CAPTURE_CONFIRM_TIMEOUT_DEFAULT_MS = 200
CAPTURE_TAIL_S = (0.0, 10.0)
CAPTURE_TAIL_DEFAULT_S = 2.0
CAPTURE_MUTE_PIN_DEFAULT = 23            # spec GPIO table: GPIO23 mic-mute sense
GPIO_BCM_PINS = (0, 27)
# Pins the spec's GPIO map (ADR-0040) already assigns: I2C 2/3, UART 14/15, IR LED gates 17/27, I2S 18-21, button 22,
# mute sense 23. The recording light may use none of them (driving an IR LED gate as a "light" would defeat its duty cap).
GPIO_MAP_ASSIGNED = frozenset({2, 3, 14, 15, 17, 18, 19, 20, 21, 22, 23, 27})
CAPTURE_KEYS = {"profile", "indicator", "mute_sense", "confirm_timeout_ms", "tail_s", "indicator_adapter", "mute_adapter",
                "indicator_pin", "mute_pin"}


def capture_settings(d: dict) -> dict:
    """The capture section with the profile's defaults filled in (no validation; see validate_capture)."""
    d = d if isinstance(d, dict) else {}
    profile = d.get("profile", "mac")
    base = CAPTURE_PROFILE_DEFAULTS.get(profile, CAPTURE_PROFILE_DEFAULTS["mac"])
    out = {"profile": profile, **base, "confirm_timeout_ms": CAPTURE_CONFIRM_TIMEOUT_DEFAULT_MS, "tail_s": CAPTURE_TAIL_DEFAULT_S,
           "indicator_pin": None, "mute_pin": CAPTURE_MUTE_PIN_DEFAULT}
    out.update({k: v for k, v in d.items() if k in CAPTURE_KEYS})
    return out


def _gpio_pin(v: object) -> bool:
    lo, hi = GPIO_BCM_PINS
    return isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi


def validate_capture(d: dict) -> list[str]:
    if not isinstance(d, dict):
        return ["capture must be an object"]
    errs = []
    unknown = set(d) - CAPTURE_KEYS
    if unknown:
        errs.append(f"capture: unknown keys {sorted(unknown)}")
    if d.get("profile", "mac") not in CAPTURE_PROFILES:
        return errs + [f"capture.profile must be one of {list(CAPTURE_PROFILES)}"]
    v = capture_settings(d)
    if v["indicator"] not in CAPTURE_INDICATOR_MODES:
        errs.append(f"capture.indicator must be one of {list(CAPTURE_INDICATOR_MODES)}")
    elif v["indicator"] == "off" and v["profile"] in CAPTURE_FORCED_INDICATOR:
        errs.append(f"capture.indicator cannot be off for {v['profile']}: the recording light is required there (ADR-0063)")
    if v["mute_sense"] not in CAPTURE_MUTE_MODES:
        errs.append(f"capture.mute_sense must be one of {list(CAPTURE_MUTE_MODES)}")
    for k in ("indicator_adapter", "mute_adapter"):
        if v[k] not in CAPTURE_ADAPTERS:
            errs.append(f"capture.{k} must be one of {list(CAPTURE_ADAPTERS)}")
    if v["indicator"] == "required" and v["indicator_adapter"] == "none":
        errs.append(f"capture.indicator is required for {v['profile']}: set capture.indicator_adapter (sim until the real light is wired)")
    if v["mute_sense"] == "required" and v["mute_adapter"] == "none":
        errs.append("capture.mute_sense is required: set capture.mute_adapter (sim or gpio)")
    lo, hi = CAPTURE_CONFIRM_TIMEOUT_MS
    if not _finite_number(v["confirm_timeout_ms"]) or not lo <= v["confirm_timeout_ms"] <= hi:
        errs.append(f"capture.confirm_timeout_ms must be in [{lo}, {hi}]")
    lo_s, hi_s = CAPTURE_TAIL_S
    if not _finite_number(v["tail_s"]) or not lo_s <= v["tail_s"] <= hi_s:
        errs.append(f"capture.tail_s must be in [{lo_s}, {hi_s}]")
    pin_lo, pin_hi = GPIO_BCM_PINS
    ind_pin, mute_pin = v["indicator_pin"], v["mute_pin"]
    if ind_pin is not None and not _gpio_pin(ind_pin):
        errs.append(f"capture.indicator_pin must be a BCM pin number in [{pin_lo}, {pin_hi}]")
    elif ind_pin is not None and ind_pin in GPIO_MAP_ASSIGNED:
        errs.append(f"capture.indicator_pin {ind_pin} is already assigned in the GPIO map (ADR-0040)")
    if not _gpio_pin(mute_pin):
        errs.append(f"capture.mute_pin must be a BCM pin number in [{pin_lo}, {pin_hi}]")
    elif mute_pin != CAPTURE_MUTE_PIN_DEFAULT and mute_pin in GPIO_MAP_ASSIGNED:
        errs.append(f"capture.mute_pin {mute_pin} is already assigned in the GPIO map (ADR-0040)")
    if ind_pin is not None and ind_pin == mute_pin:
        errs.append("capture.indicator_pin and capture.mute_pin must differ")
    if v["indicator"] == "required" and v["indicator_adapter"] == "gpio" and ind_pin is None:
        errs.append("capture.indicator_pin is needed for the gpio recording light (the GPIO map assigns no pin to it yet)")
    return errs


# Live wiring of the power governor (ADR-0061, docs/LIVE.md "Power and heat"). These two keys sit in the "power"
# section beside the governor's thresholds and plans; validate_power() sees the section without them.
POWER_SOURCE_NONE, POWER_SOURCE_MACOS, POWER_SIM_PREFIX = "none", "macos", "sim:"
POWER_SOURCE_DEFAULT = POWER_SOURCE_NONE      # no readings, no governor: live mode exactly as before
# Mirrors oiris.adapters.power_sim.SCENES (a test keeps them equal); not imported, so config stays light.
POWER_SIM_SCENES = ("cool_room", "hot_day", "low_battery", "charging_hot")
# One reading per interval. 10 s is a sixth of the governor's 60 s staleness limit and a third of its 30 s dwell, so
# a level can change within one dwell; the simulated thermal time constant is 300 s, so nothing moves faster.
POWER_INTERVAL_DEFAULT_S = 10.0
POWER_INTERVAL_RANGE_S = (1.0, 600.0)
POWER_LIVE_KEYS = ("source", "interval_s")
# Scheduled retention while live mode runs (docs/LIVE.md "Retention while running"): glasses and Pi rigs run for days.
RETENTION_CHECK_DEFAULT_S = 3600.0
RETENTION_CHECK_RANGE_S = (60.0, 86400.0)


def power_sources() -> tuple[str, ...]:
    return (POWER_SOURCE_NONE, POWER_SOURCE_MACOS, *(POWER_SIM_PREFIX + s for s in POWER_SIM_SCENES))


def power_governor_section(d: object) -> dict:
    """The "power" section without the live keys: what validate_power and PowerSettings.from_config take."""
    return {k: v for k, v in d.items() if k not in POWER_LIVE_KEYS} if isinstance(d, dict) else {}


def power_live_settings(d: object) -> tuple[str, float]:
    """(source, interval_s) from the "power" section, defaults filled in (no validation; see validate_power_live)."""
    d = d if isinstance(d, dict) else {}
    return str(d.get("source", POWER_SOURCE_DEFAULT)), float(d.get("interval_s", POWER_INTERVAL_DEFAULT_S))


def validate_power_live(d: object) -> list[str]:
    if not isinstance(d, dict):
        return []            # validate_power reports the wrong type
    errs = []
    if d.get("source", POWER_SOURCE_DEFAULT) not in power_sources():
        errs.append(f"power.source must be one of {list(power_sources())}")
    lo, hi = POWER_INTERVAL_RANGE_S
    v = d.get("interval_s", POWER_INTERVAL_DEFAULT_S)
    if not _finite_number(v) or not lo <= v <= hi:
        errs.append(f"power.interval_s must be in [{lo}, {hi}] seconds")
    return errs


PLUGIN_KEYS = {"paths", "entry_points", "slots", "options"}
SLOT_NAMES = {"speech_recognizer", "voice_antispoof", "speaker_embed", "eye_model", "fusion_weights", "face_match"}


def validate_plugins(d: dict) -> list[str]:
    if not isinstance(d, dict):
        return ["plugins must be an object"]
    errs = []
    if set(d) - PLUGIN_KEYS: errs.append(f"plugins: unknown keys {sorted(set(d) - PLUGIN_KEYS)}")
    if not all(isinstance(p, str) and ":" in p for p in d.get("paths", [])): errs.append("plugins.paths: 'module:function' strings")
    slots = d.get("slots", {})
    if not isinstance(slots, dict) or set(slots) - SLOT_NAMES: errs.append(f"plugins.slots: keys must be in {sorted(SLOT_NAMES)}")
    return errs


FACE_MATCH_KEYS = {"modes", "jurisdiction", "require_enrollment_consent", "match_threshold", "encounter_retention_days", "place_source",
                   "time_bucket_minutes", "allow_log_export", "enroll_min_shots", "enroll_max_shots", "enroll_required_angles",
                   "pose_up_delta", "pose_down_delta",
                   "reenroll_confidence_drop", "reenroll_window_matches", "adapter", "model_dir", "allow_test_backend"}


@dataclass
class Config:
    flags: dict = field(default_factory=lambda: {m: m not in DEFAULT_OFF for m in MODULES})
    data_dir: str = "~/.asb"
    retention_days: int = 30
    retention_check_s: float = RETENTION_CHECK_DEFAULT_S   # live mode re-runs the retention purge this often
    gps_export_decimals: int = 2
    thermal_palette: str = "grayscale"
    thermal_range_c: tuple = (10.0, 45.0)
    queue_size: int = 64
    adapter_timeout_s: float = 2.0
    sensor_max_age_s: float = 5.0      # cached readings older than this are stale: not scored, health degraded
    adapter_hung_disable_s: float = 30.0  # a read hung this long disables the adapter until it returns
    restart_backoff_max_s: float = 10.0
    breaker_threshold: int = 5
    breaker_cooldown_s: float = 30.0
    rekey_messages: int = 1000
    rekey_seconds: int = 600
    failed_unlock_wipe: int = 10
    voice_attention_min_interval_s: float = 10.0
    min_families_for_alert: int = 2
    max_unhealthy_families: int = 1
    allow_network: bool = False
    pled_duty_cap: float = 0.25
    face_match: dict = field(default_factory=dict)   # M-16 settings, validated by FACE_MATCH_KEYS below
    voice_link: dict = field(default_factory=dict)   # M-17 settings, validated by validate_voice_link()
    output: dict = field(default_factory=dict)       # output arbiter and motion gate, validated by validate_output()
    rig: dict = field(default_factory=dict)          # sensor mounts (ADR-0060), validated by validate_rig(); {} = one front camera
    # Power governor thresholds and per-level duty plans (ADR-0061), validated by oiris.power.validate_power(); {} = defaults
    power: dict = field(default_factory=dict)
    # Latency budgets {"<part>.<stage>": ms} over the defaults (ADR-0061), validated by oiris.power.validate_budgets()
    budgets: dict = field(default_factory=dict)
    # Recording light, mic mute and capture interlock (ADR-0063), validated by validate_capture(); {} = the mac profile
    capture: dict = field(default_factory=dict)
    # Replaceable model slots (docs/PLUGINS.md): {"paths": ["pkg.mod:register"], "entry_points": true,
    #  "slots": {"speech_recognizer": "name", ...}, "options": {name: {...}}}
    plugins: dict = field(default_factory=dict)

    def enabled(self, module: str) -> bool:
        return module in LOCKED_ON or bool(self.flags.get(module, False))

    def validate(self) -> "Config":
        errs = []
        unknown = set(self.flags) - set(MODULES)
        if unknown:
            errs.append(f"unknown module flags {sorted(unknown)}")
        for m in LOCKED_ON:
            self.flags[m] = True
        if not 1 <= self.retention_days <= 365: errs.append("retention_days must be 1..365")
        lo_r, hi_r = RETENTION_CHECK_RANGE_S
        if not (_finite_number(self.retention_check_s) and lo_r <= self.retention_check_s <= hi_r):
            errs.append(f"retention_check_s must be in [{lo_r}, {hi_r}] seconds")
        if self.gps_export_decimals not in (0, 1, 2, 3): errs.append("gps_export_decimals must be 0..3")
        if self.thermal_palette not in ("grayscale", "inferno", "viridis"): errs.append("bad thermal_palette")
        lo, hi = self.thermal_range_c
        if not lo < hi: errs.append("thermal_range_c low must be < high")
        if self.min_families_for_alert < 2: errs.append("min_families_for_alert cannot go below 2 (evidence rule)")
        if not 1 <= self.rekey_messages <= 1000: errs.append("rekey_messages must be 1..1000")
        if not 1 <= self.rekey_seconds <= 600: errs.append("rekey_seconds must be 1..600")
        if not 0 < self.pled_duty_cap <= 0.25: errs.append("LED duty cap must be in (0, 0.25]")
        if self.queue_size < 1 or self.adapter_timeout_s <= 0: errs.append("queue/timeout must be positive")
        if not (isinstance(self.sensor_max_age_s, (int, float)) and math.isfinite(self.sensor_max_age_s) and self.sensor_max_age_s > 0):
            errs.append("sensor_max_age_s must be a positive number")
        if not (isinstance(self.adapter_hung_disable_s, (int, float)) and math.isfinite(self.adapter_hung_disable_s)
                and self.adapter_hung_disable_s > 0):
            errs.append("adapter_hung_disable_s must be a positive number")
        errs += validate_face_match(self.face_match)
        errs += validate_voice_link(self.voice_link)
        errs += validate_output(self.output)
        errs += validate_rig(self.rig)
        errs += validate_power(power_governor_section(self.power) if isinstance(self.power, dict) else self.power)
        errs += validate_power_live(self.power)
        errs += validate_budgets(self.budgets)
        errs += validate_capture(self.capture)
        if not (_finite_number(self.voice_attention_min_interval_s) and self.voice_attention_min_interval_s >= 0):
            errs.append("voice_attention_min_interval_s must be a number >= 0")
        errs += validate_plugins(self.plugins)
        if errs:
            raise ConfigError("; ".join(errs))
        return self

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        if path is None or not Path(path).exists():
            return cls().validate()
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        if "flags" in known:
            base = cls().flags
            base.update(known["flags"])
            known["flags"] = base
        if "thermal_range_c" in known:
            known["thermal_range_c"] = tuple(known["thermal_range_c"])
        return cls(**known).validate()

    def to_dict(self) -> dict:
        return asdict(self)
