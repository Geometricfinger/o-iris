"""Minimal local dashboard (stdlib only). Binds 127.0.0.1 only. Zones follow the design
bot's C6 map: D1 status strip, D2 sighting table, D3 map pins, D4 sweep timeline,
D5 sensor health, D6 review queue. Headless scenario data when no rig is attached; with a
live-state provider (oiris.live.LiveRunner.snapshot) it shows the live run and never simulates.
With a live provider and `on_quit`, the page also has a Quit button: POST /quit carrying a
random per-run token, from a same-origin page on 127.0.0.1/localhost (docs/LIVE.md).
Every GET needs Host 127.0.0.1:<port> or localhost:<port> (no DNS-rebinding names); else 403.
With voice link (M-17) on in live mode, a "Voice link" area shows the listening state and "Sounds like NAME"
cards (ADR-0058: a voice match is a hint, not proof; "Sounds like NAME, on your left" when the card has a
direction, ADR-0065); in push-to-talk mode its Listen button is POST /voice/listen under the same token and origin rules.
With the output arbiter's state in the live snapshot ("output"), an "Outputs" area shows the motion state in
plain words and the current pop-up until it expires. With spatial events in the live snapshot ("located"), a
"Located" area lists each one's direction word, kinds, how many sensors agree and seconds ago; none, no area.
With the capture interlock's state in the live snapshot ("capture", ADR-0063), a "Capture" line says in plain words
whether the recording light is on, the mute switch's position and whether the microphone listens or why it is refused.
With a face camera configured ("camera" in the live snapshot, ADR-0057 decision 5) a "Camera" line says whether it is on
and at what rate, off because face matching is off, waiting for the recording light, or paused by the power plan.
With "power" in the live snapshot (ADR-0061, power.source not "none") a "Power" line gives the level, battery, charging
and skin temperature, and says when voice matching is paused; with stages over their latency budget a "Latency" line
lists them; with "chain" (ADR-0066) a "Record chain" line says whether the chain continued or restarted and why. D1
shows when the retention purge last ran.
A heard card whose person has a stored photo (ADR-0057 decision 4) shows it from GET /voice/photo?k=KEY&token=TOKEN:
the per-run token, an Origin or Sec-Fetch-Site proving this same origin, and only for a person on a current card (else
404); served no-store, image/jpeg, nosniff. There is no way to list photos."""
from __future__ import annotations
import html, json, re, secrets, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from urllib.parse import parse_qs, quote, urlsplit

from ..config import Config
from ..headless import run_scenario
from ..store.sightings import OSM_ATTRIBUTION

PAGE = """<!doctype html><meta charset=utf-8><title>O Iris dashboard</title>{refresh}
<style>body{{font:14px system-ui;margin:0;display:grid;grid-template:auto 1fr 1fr/1fr 1fr 1fr;gap:8px;padding:8px}}
section{{border:1px solid #888;padding:8px}} #D1{{grid-column:1/4}} h2{{font-size:13px;margin:0 0 6px}}
li small{{display:block;color:#666}} #quit{{float:right;margin:0}}</style>
<section id=D1>{quit}<h2>D1 Status</h2>{status}</section>
<section id=D2><h2>D2 Signatures (latest)</h2>{sigs}</section>
<section id=D3><h2>D3 Map pins</h2>Offline map layer. <small>{attr}</small></section>
<section id=D4><h2>D4 Sweep timeline</h2>{sweep}</section>
<section id=D5><h2>D5 Sensor health</h2>{health}</section>
<section id=D6><h2>D6 Review queue</h2>Confirmed/benign sightings awaiting review appear here.</section>
{voice}<p style="grid-column:1/4"><a href="/face">Face match (M-16, optional, off by default)</a></p>
<p style="grid-column:1/4"><small>Research tool. Signatures are observations with benign causes; never an identity.</small></p>"""

STOPPED_PAGE = ('<!doctype html><meta charset=utf-8><title>O Iris stopped</title>'
                '<p style="font:16px system-ui;padding:16px"><b>O Iris stopped.</b> '
                'Live mode has ended and the sweep is closed. You can close this window.</p>')
# img-src 'self': the heard card's photo, from this dashboard only.
CSP = "default-src 'none'; style-src 'unsafe-inline'; img-src 'self'; form-action 'self'"
PHOTO_PATH = "/voice/photo"
PHOTO_PX = 64            # the card's photo, shown this many CSS pixels square
SEC_FETCH_SAME_ORIGIN = "same-origin"

_BT_SETTINGS = "System Settings > Privacy & Security > Bluetooth"
_LOC_SETTINGS = "System Settings > Privacy & Security > Location Services"
# (pattern on the raw reason, plain status, what the user should do). First match wins.
KNOWN_REASONS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"waiting for Bluetooth permission", re.I), "Waiting for Bluetooth permission",
     f"Answer the macOS prompt, or allow O Iris in {_BT_SETTINGS}."),
    (re.compile(r"Bluetooth permission restricted", re.I), "Bluetooth blocked by device policy",
     "Ask the administrator of this Mac to allow Bluetooth for O Iris."),
    (re.compile(r"Bluetooth permission denied|DENIED_BY_USER|unauthori[sz]ed", re.I), "Bluetooth access denied",
     f"Allow O Iris in {_BT_SETTINGS}, then restart O Iris."),
    (re.compile(r"POWERED_OFF|Bluetooth (device )?is turned off", re.I), "Bluetooth is off",
     "Turn Bluetooth on in Control Center or System Settings > Bluetooth."),
    (re.compile(r"macOS withheld BSSIDs", re.I), "macOS is hiding Wi-Fi network identifiers",
     f"Allow Location for O Iris in {_LOC_SETTINGS}."),
    (re.compile(r"Wi-Fi is turned off", re.I), "Wi-Fi is off",
     "Turn Wi-Fi on in Control Center or System Settings > Wi-Fi."),
    (re.compile(r"pip install bleak", re.I), "Bluetooth driver not installed",
     "Install it with: pip install bleak (or reinstall O Iris)."),
]
_MIC_SETTINGS = "System Settings > Privacy & Security > Microphone"
# Voice link (M-17): (pattern on the plain reason, what the user should do). First match wins.
VOICE_ACTIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"sounddevice package is not installed", re.I), 'Install it with: pip install -e ".[voice]"'),
    (re.compile(r"no input device", re.I), "Choose a microphone from: oiris voice-link devices (config voice_link.input_device)."),
    (re.compile(r"microphone could not be opened", re.I), f"Allow the microphone in {_MIC_SETTINGS}, or check the input device."),
    (re.compile(r"face matching|M-16", re.I), "Turn on face matching in personal mode: oiris face-match enable"),
    (re.compile(r"M-17 is off|not opted in|voice link is off", re.I), "Turn it on with: oiris voice-link enable"),
    (re.compile(r"speaker model|no plugin|not downloaded", re.I), "Download the model: oiris voice-link fetch-models"),
]
VOICE_STATES = {"listening": "Listening (microphone on)", "idle": "Push-to-talk: press Listen"}
# bus/runner bookkeeping prefixes that mean nothing to the user; stripped from the plain line only
_JARGON = re.compile(r"^(?:\s*(?:circuit open|read (?:error|timeout|exception)|DriverUnavailable|\w+Error)\s*:\s*)+")


def plain_reason(reason: str) -> tuple[str, str]:
    """-> (plain status, action). Known reasons get a plain status and what to do; unknown reasons
    fall back to the raw text with bus jargon ("circuit open: read error:") stripped, no action."""
    for rx, status, action in KNOWN_REASONS:
        if rx.search(reason):
            return status, action
    return _JARGON.sub("", reason).strip() or reason, ""


def _sensors(health: dict, adapters: dict | None = None) -> list[tuple[str, str, str]]:
    """[(name, health, raw reason)] for every bus adapter plus every adapter that failed to open."""
    out = []
    for k, v in health.items():
        if isinstance(v, dict):
            out.append((k, str(v.get("health", "?")), str(v.get("reason") or "")))
        else:
            out.append((k, str(v), ""))
    out += [(k, "unavailable", str(v.get("reason") or "")) for k, v in (adapters or {}).items() if not v["available"]]
    return out


def _sensor_line(name: str, health: str, reason: str) -> str:
    """`name: health · plain status. action` first; the raw reason kept below in smaller text."""
    if not reason:
        return f"<li>{html.escape(name)}: {html.escape(health)}</li>"
    status, action = plain_reason(reason)
    act = f" <b>{html.escape(action)}</b>" if action else ""
    return (f"<li>{html.escape(name)}: {html.escape(health)} · {html.escape(status.rstrip('.'))}.{act}"
            f"<small>{html.escape(reason)}</small></li>")


def _health_items(health: dict) -> str:
    out = []
    for k, v in health.items():
        if isinstance(v, dict):
            reason = f" ({v['reason']})" if v.get("reason") else ""
            out.append(f"<li>{html.escape(k)}: {html.escape(str(v.get('health', '?')) + reason)}</li>")
        else:
            out.append(f"<li>{html.escape(k)}: {html.escape(str(v))}</li>")
    return "".join(out)


def _level(res: dict) -> str:
    return f"Level: <b>{html.escape(str(res['level'] or 'none'))}</b> · index {res['anomaly_index']:.2f}"


def _live_status(res: dict | None, sensors: list[tuple[str, str, str]], health: dict | None = None) -> str:
    """D1 in live mode. With no signal family active the fusion index is only the model's output
    on empty input, so neither index nor level is shown; instead which sensors wait and why.
    "Arriving" means some bus adapter has delivered a sample (live health `samples` > 0); a
    family score present with "no samples" does not count."""
    if res and res.get("families_active"):
        return _level(res)
    flowing = any(isinstance(v, dict) and (v.get("samples") or 0) > 0 for v in (health or {}).values())
    head = "<b>Sensor data arriving; no signal family active yet</b>" if flowing else "<b>No live data yet</b>"
    waiting = [f"{html.escape(n)} ({html.escape(plain_reason(r)[0] if r else h)})"
               for n, h, r in sensors if h != "ok"]
    if waiting:
        head += " · waiting: " + "; ".join(waiting) + " (what to do: see D5)"
    elif not sensors:
        head += " · no sensors configured"
    return head


def voice_action(reason: str) -> str:
    for rx, action in VOICE_ACTIONS:
        if rx.search(reason):
            return action
    return ""


# ADR-0065: a heard card's direction word in the words the announcement speaks. A copy of
# oiris.modules.voice_link.announce.DIRECTION_PHRASES (this module stays stdlib only; a test keeps them equal).
# Any other value shows nothing.
VOICE_DIRECTIONS = {"ahead": "ahead of you", "at left": "on your left", "at right": "on your right",
                    "behind": "behind you", "above": "above you", "below": "below you"}


def _photo_img(card: dict, photo_token: str | None) -> str:
    """The card's photo element, or "" when the card has no photo key or the page has no token."""
    key = card.get("photo")
    if not photo_token or not isinstance(key, str) or not key:
        return ""
    src = f"{PHOTO_PATH}?k={quote(key, safe='')}&token={quote(photo_token, safe='')}"
    return (f'<img src="{html.escape(src)}" alt="" width={PHOTO_PX} height={PHOTO_PX} '
            'style="vertical-align:middle;margin-right:6px">')


def _voice(v: dict, listen_token: str | None = None, photo_token: str | None = None) -> str:
    """The "Voice link" area: the state in plain words, the digital-silence notice, the heard cards (with the person's
    photo when one is stored) and, in push-to-talk mode, the Listen button. Every text is escaped."""
    state = str(v.get("state", ""))
    if state in VOICE_STATES:
        line = VOICE_STATES[state]
        if state == "listening" and v.get("mode") == "push_to_talk" and v.get("listen_left_s"):
            line += f" · {float(v['listen_left_s']):.0f} s left"
        out = [f"<p><b>{html.escape(line)}</b></p>"]
    else:
        reason = str(v.get("reason") or "no reason given")
        action = voice_action(reason)
        out = [f"<p><b>Voice link unavailable: {html.escape(reason.rstrip('.'))}</b>"
               + (f" - {html.escape(action)}" if action else "") + "</p>"]
    if v.get("notice"):
        out.append(f"<p><b>{html.escape(str(v['notice']))}</b></p>")
    for c in v.get("cards") or []:
        agree = " · Recent face match for this person" if c.get("face_agrees") else ""
        where = VOICE_DIRECTIONS.get(str(c.get("direction"))) if c.get("direction") else None
        where = f", {html.escape(where)}" if where else ""
        out.append(f"<p class=heard>{_photo_img(c, photo_token)}<b>Sounds like {html.escape(str(c.get('name', '')))}{where}</b>{agree}"
                   f" · {float(c.get('seconds_ago') or 0):.0f} s ago</p>")
    out.append(f"<small>{int(v.get('heard') or 0)} heard this run</small>")
    if listen_token and v.get("mode") == "push_to_talk" and state in VOICE_STATES:
        out.append(f'<form method=post action=/voice/listen><input type=hidden name=token value="{html.escape(listen_token)}">'
                   "<button>Listen</button></form>")
    return '<section id=VL style="grid-column:1/4"><h2>Voice link</h2>' + "".join(out) + "</section>\n"


MOTION_WORDS = {"still": "still", "walking": "walking", "fast": "moving fast", "unknown": "motion unknown"}
OUTPUT_COUNTS = ("played", "queued", "deduped", "suppressed", "dropped")
# No script runs (CSP): a CSS animation hides the pop-up when it expires, before the 2 s refresh drops it.
_POPUP_GONE = "<style>@keyframes oiris-gone{to{visibility:hidden}}</style>"


def _output(o: dict) -> str:
    """The "Outputs" area: the motion state in plain words, the current pop-up (from the output arbiter, gone
    after its expiry) and the counts by outcome. Every text is escaped."""
    motion = MOTION_WORDS.get(str(o.get("motion")), MOTION_WORDS["unknown"])
    if not o.get("imu"):
        motion += " (no motion sensor)"
    out = [f"<p>Motion: <b>{html.escape(motion)}</b></p>"]
    p: dict = o["popup"] if isinstance(o.get("popup"), dict) else {}
    try:
        left = float(p.get("expires_in_s") or 0)
    except (TypeError, ValueError):  # python-brain: allow SILENT_EXCEPTION - a malformed expiry renders as "No pop-up"; the page keeps rendering
        left = 0.0
    if left > 0:
        out.append(f'{_POPUP_GONE}<p class=popup style="animation:oiris-gone 0s linear {left:.1f}s forwards">'
                   f"<b>{html.escape(str(p.get('text', '')))}</b> · {html.escape(str(p.get('priority', '')))}"
                   f" · {left:.0f} s left</p>")
    else:
        out.append("<p>No pop-up</p>")
    counts: dict = o["counts"] if isinstance(o.get("counts"), dict) else {}
    out.append("<small>" + " · ".join(f"{k} {int(counts.get(k) or 0)}" for k in OUTPUT_COUNTS) + "</small>")
    return '<section id=OUT style="grid-column:1/4"><h2>Outputs</h2>' + "".join(out) + "</section>\n"


def _located(items: list) -> str:
    """The "Located" area (ADR-0060): each current spatial event as its direction word, kinds, how many sensors agree
    and how long ago. Signatures and directions only, never who. Every text is escaped; "" when there are none."""
    rows = []
    for e in items:
        if not isinstance(e, dict):
            continue
        kinds = ", ".join(str(k) for k in e.get("kinds") or [])
        n = int(e.get("sensors") or 0)
        agree = "1 sensor" if n == 1 else f"{n} sensors agree"
        rows.append(f"<li><b>{html.escape(str(e.get('direction', '')))}</b> · {html.escape(kinds)}"
                    f" · {agree} · {float(e.get('seconds_ago') or 0):.0f} s ago</li>")
    if not rows:
        return ""
    return '<section id=LOC style="grid-column:1/4"><h2>Located</h2><ul>' + "".join(rows) + "</ul></section>\n"


# Capture (ADR-0063): each state code of snapshot()["capture"] in plain words. Unknown codes show "unknown".
CAPTURE_LIGHT_WORDS = {"on": "on", "off": "off", "tail": "on (turning off shortly)", "failed": "FAILED",
                       "not_required": "not used"}
CAPTURE_LIGHT_MAC = "not used on this Mac"   # macOS shows its own camera light and microphone dot
CAPTURE_MUTE_WORDS = {"muted": "muted (microphone cut)", "open": "open", "unknown": "cannot be read",
                      "not_present": "none fitted"}
CAPTURE_KIND_WORDS = {"active": "listening", "off": "off"}
CAPTURE_CAMERA_WORDS = {"active": "on", "off": "off"}
CAPTURE_REFUSALS = {"indicator_unconfirmed": "the recording light could not be confirmed on",
                    "indicator_failed": "the recording light failed",
                    "muted": "the mute switch is set to mute",
                    "mute_unknown": "the mute switch cannot be read",
                    "mute_breach": "sound came through while muted (hardware fault; restart after a check)",
                    "closed": "live mode is stopping"}
CAPTURE_REFUSED = "refused:"


def _capture_kind(code: str, words: dict) -> str:
    if code.startswith(CAPTURE_REFUSED):
        return "refused: " + CAPTURE_REFUSALS.get(code[len(CAPTURE_REFUSED):], "unknown reason")
    return words.get(code, "unknown")


def _capture(c: dict) -> str:
    """The "Capture" line: recording light, mute switch and microphone (and the camera once it is used), in plain
    words from the tables above. Every text is escaped."""
    light_code = str(c.get("indicator"))
    if light_code == "not_required" and c.get("profile") == "mac":
        light = CAPTURE_LIGHT_MAC
    else:
        light = CAPTURE_LIGHT_WORDS.get(light_code, "unknown")
    parts = [f"Recording light: <b>{html.escape(light)}</b>",
             f"Mute switch: <b>{html.escape(CAPTURE_MUTE_WORDS.get(str(c.get('mute')), 'unknown'))}</b>",
             f"Microphone: <b>{html.escape(_capture_kind(str(c.get('mic')), CAPTURE_KIND_WORDS))}</b>"]
    if str(c.get("camera", "off")) != "off":
        parts.append(f"Camera: <b>{html.escape(_capture_kind(str(c.get('camera')), CAPTURE_CAMERA_WORDS))}</b>")
    return '<p id=CAP style="grid-column:1/4">Capture · ' + " · ".join(parts) + "</p>\n"


# Face camera (ADR-0057 decision 5): snapshot()["camera"] in plain words. Unknown states show "unknown".
CAMERA_OFF_WORDS = {"m16_off": "off (face matching is off)", "closed": "off (live mode is stopping)"}
CAMERA_PAUSE_WORDS = {"heat": "paused to cool down", "battery": "paused to save the battery"}
CAMERA_STATE_WORDS = {"starting": "starting", "waiting": "waiting for the recording light", "off": "off",
                      "paused": "paused to save power"}
ONE_FRAME = 1.0


def _frames_a_second(fps: object) -> str:
    if not isinstance(fps, (int, float)) or isinstance(fps, bool):
        return ""
    return f"{float(fps):g} frame{'' if float(fps) == ONE_FRAME else 's'} a second"


def _camera(c: dict) -> str:
    """The "Camera" line, only when a face camera is configured: on (with its rate), off and why, waiting for the
    recording light, paused by the power plan, or unavailable with the reason. Every text is escaped."""
    state, code = str(c.get("state", "")), str(c.get("code") or "")
    if state == "on":
        rate = _frames_a_second(c.get("fps"))
        text = "on" + (f", {rate}" if rate else "")
    elif state == "off" and code in CAMERA_OFF_WORDS:
        text = CAMERA_OFF_WORDS[code]
    elif state == "paused":
        text = CAMERA_PAUSE_WORDS.get(code, CAMERA_STATE_WORDS["paused"])
    elif state == "unavailable":
        reason = str(c.get("reason") or "").rstrip(".")
        text = "unavailable" + (f" ({reason})" if reason else "")
    else:
        text = CAMERA_STATE_WORDS.get(state, "unknown")
    return f'<p id=CAM style="grid-column:1/4">Camera: <b>{html.escape(text)}</b></p>\n'


# Power (ADR-0061): the governor's level and reason codes in plain words. Unknown codes show nothing extra.
POWER_LEVEL_WORDS = {"normal": "normal", "saver": "battery saver", "hot": "hot, throttled", "critical": "critical"}
POWER_REASON_WORDS = {"battery_low": "battery low", "battery_critical": "battery critically low",
                      "skin_hot": "warm against the skin", "skin_critical": "too hot against the skin",
                      "device_hot": "device hot", "device_critical": "device too hot",
                      "thermal_status_hot": "the system reports heat",
                      "thermal_status_critical": "the system reports critical heat",
                      "stale": "the last reading is old"}
FULL_DUTY = 1.0
PERCENT = 100.0


def _power(p: dict, voice_on: bool) -> str:
    """The "Power" line: level (with its reasons), battery, charging, skin temperature; with the voice link on, whether
    matching is paused or listening is cut back. Numbers and fixed words only; every text is escaped."""
    level = str(p.get("level", ""))
    words = POWER_LEVEL_WORDS.get(level, "unknown")
    why = [POWER_REASON_WORDS[r] for r in p.get("reasons") or [] if r in POWER_REASON_WORDS]
    parts = [words + (f" ({', '.join(why)})" if why else "")]
    battery = p.get("battery_pct")
    if isinstance(battery, (int, float)) and not isinstance(battery, bool):
        parts.append(f"battery {float(battery):.0f} %")
    if p.get("charging") is True:
        parts.append("charging")
    skin = p.get("skin_c")
    if isinstance(skin, (int, float)) and not isinstance(skin, bool):
        parts.append(f"skin {float(skin):.1f} C")
    line = "Power: <b>" + html.escape(", ".join(parts)) + "</b>"
    plan: dict = p["plan"] if isinstance(p.get("plan"), dict) else {}
    if voice_on and plan.get("voice_match") is False:
        line += " · <b>voice matching paused</b> to save power and heat"
    elif voice_on:
        duty = plan.get("listen_duty")
        if isinstance(duty, (int, float)) and not isinstance(duty, bool) and duty < FULL_DUTY:
            line += f" · voice listening {float(duty) * PERCENT:.0f} % of the time"
    return f'<p id=PWR style="grid-column:1/4">{line}</p>\n'


def _budgets(over: list) -> str:
    """The "Latency" line, only when a stage is over its budget (a measurement for the developer)."""
    items = [f"{html.escape(str(o.get('part', '')))} {html.escape(str(o.get('stage', '')).replace('_', ' '))} "
             f"p95 {float(o.get('p95_ms') or 0):.0f} ms (budget {float(o.get('budget_ms') or 0):.0f} ms)"
             for o in over if isinstance(o, dict)]
    if not items:
        return ""
    return '<p id=LAT style="grid-column:1/4">Latency over budget: ' + "; ".join(items) + "</p>\n"


# Record chain (ADR-0066): restart reason codes in plain words.
CHAIN_REASON_WORDS = {"legacy": "records from before the resumable chain", "edited": "a stored record was changed",
                      "deleted": "stored records are missing", "reordered": "stored records are out of order",
                      "truncated": "records after a signed anchor are missing",
                      "bad_anchor": "a signed anchor does not verify"}


def _chain(c: dict) -> str:
    state = str(c.get("state", ""))
    records = int(c.get("records") or 0)
    if state == "resumed":
        text = f"continued ({records} records)"
    elif state == "restarted":
        text = f"restarted ({CHAIN_REASON_WORDS.get(str(c.get('reason')), 'unknown reason')})"
    elif state == "new":
        text = "new"
    else:
        text = "unknown"
    return f'<p id=CHAIN style="grid-column:1/4">Record chain: <b>{html.escape(text)}</b></p>\n'


SECONDS_PER_MINUTE = 60.0


def _last_purge(records: dict) -> str:
    """D1's retention note: when the purge last ran (local time) and how often it runs."""
    at, every = records.get("last_purge_at"), records.get("retention_check_s")
    if not isinstance(at, (int, float)) or isinstance(at, bool):
        return ""
    out = f" · retention purge last run {time.strftime('%H:%M', time.localtime(float(at)))}"
    if isinstance(every, (int, float)) and not isinstance(every, bool):
        out += f", every {float(every) / SECONDS_PER_MINUTE:.0f} min"
    return out


def render(res: dict | None, live: dict | None = None, quit_token: str | None = None,
           listen_token: str | None = None, photo_token: str | None = None) -> str:
    """quit_token (live only): adds the Quit button, a POST form carrying the per-run token.
    listen_token (live only): the same token, for the voice link's Listen button (push-to-talk).
    photo_token (live only): the same token, in each heard card's photo URL."""
    sigs = ("".join(f"<li>{html.escape(s)}</li>" for s in res["signatures"]) or "none") if res else "none"
    if live is None:
        status = _level(res) if res else "Live: <b>waiting for the first step</b>"
        health = _health_items(res.get("health", {}) if res else {}) or "headless (no live adapters)"
        return PAGE.format(refresh="", quit="", voice="", status=status, sigs=f"<ul>{sigs}</ul>",
                           attr=html.escape(OSM_ATTRIBUTION), sweep="No sweeps in demo mode.", health=f"<ul>{health}</ul>")
    # "latency" is a measurement, not a sensor: its own line, never in D1's waiting list or D5.
    sensor_health = {k: v for k, v in (live.get("health") or {}).items() if k != "latency"}
    sensors = _sensors(sensor_health, live.get("adapters", {}))
    status = _live_status(res, sensors, sensor_health)
    status += f" · live · {live['steps']} steps at {live['hz']:g} Hz · {'running' if live['running'] else 'stopped'}"
    if isinstance(live.get("records"), dict):
        status += html.escape(_last_purge(live["records"]))
    health = "".join(_sensor_line(*s) for s in sensors)
    sweep = (f"Recording sweep <code>{html.escape(live['sweep_id'])}</code> at site {html.escape(live['site'])}."
             if live.get("sweep_id") else "Sweep timeline (M-14) off: not recording.")
    quit_form = (f'<form id=quit method=post action=/quit><input type=hidden name=token value="{html.escape(quit_token)}">'
                 "<button>Quit O Iris</button></form>") if quit_token else ""
    voice_on = isinstance(live.get("voice"), dict)
    voice = _voice(live["voice"], listen_token, photo_token) if voice_on else ""
    if isinstance(live.get("output"), dict):
        voice = _output(live["output"]) + voice
    if isinstance(live.get("located"), list):
        voice = _located(live["located"]) + voice
    if isinstance(live.get("chain"), dict):
        voice = _chain(live["chain"]) + voice
    if isinstance(live.get("budgets"), list):
        voice = _budgets(live["budgets"]) + voice
    if isinstance(live.get("power"), dict):
        voice = _power(live["power"], voice_on) + voice
    if isinstance(live.get("camera"), dict):
        voice = _camera(live["camera"]) + voice
    if isinstance(live.get("capture"), dict):
        voice = _capture(live["capture"]) + voice
    return PAGE.format(refresh='<meta http-equiv="refresh" content="2">', quit=quit_form, voice=voice, status=status,
                       sigs=f"<ul>{sigs}</ul>", attr=html.escape(OSM_ATTRIBUTION), sweep=sweep,
                       health=f"<ul>{health or 'no adapters'}</ul>")


def local_host(headers, port: int) -> bool:
    """Host is 127.0.0.1 or localhost on the dashboard port, case-insensitive (no DNS-rebinding names)."""
    return (headers.get("Host") or "").lower() in {f"127.0.0.1:{port}", f"localhost:{port}"}


def same_origin(headers, port: int) -> bool:
    """local_host, and an Origin header must be present and name the same scheme, host and port."""
    ok = {f"127.0.0.1:{port}", f"localhost:{port}"}
    origin = (headers.get("Origin") or "").lower()
    if not local_host(headers, port) or not origin:
        return False
    o = urlsplit(origin)
    return o.scheme == "http" and o.netloc in ok and o.path in ("", "/")


def same_origin_image(headers, port: int) -> bool:
    """The origin rule for an image GET. A browser sends no Origin header on a same-origin <img> request, so the
    proof is an Origin naming this dashboard (same_origin) or, without Origin, Sec-Fetch-Site: same-origin. A header
    that is present must agree, and a request carrying neither is refused (no proof of where it came from)."""
    origin, site = headers.get("Origin"), headers.get("Sec-Fetch-Site")
    if not local_host(headers, port) or (origin is None and site is None):
        return False
    if origin is not None and not same_origin(headers, port):
        return False
    return site is None or site.lower() == SEC_FETCH_SAME_ORIGIN


def make_server(cfg: Config, port: int = 8765, scenario: str = "quiet", face_module=None,
                live: Callable[[], dict] | None = None, bind_and_activate: bool = True,
                on_quit: Callable[[], None] | None = None,
                on_listen: Callable[[], object] | None = None,
                on_photo: Callable[[str], bytes | None] | None = None) -> ThreadingHTTPServer:
    """`live`: optional provider returning a live snapshot ({"result", "health", "adapters", ...});
    when given, "/" and "/api/latest" show it and no simulated scenario ever runs.
    `on_quit` (live only): called once by an accepted POST /quit, after the reply is sent.
    `on_listen` (live only): called by an accepted POST /voice/listen (push-to-talk). Passed explicitly
    (oiris/cli.py gives LiveRunner.voice_listen); without it there is no Listen button.
    `on_photo(key)` (live only): the JPEG for a heard card's photo key, or None. Passed explicitly (oiris/cli.py gives
    LiveRunner.voice_photo); None means no photos and no photo endpoint."""
    from .face import FaceReview
    face = FaceReview(face_module)
    photo_source = on_photo
    wants_token = on_quit is not None or on_listen is not None or photo_source is not None
    token = secrets.token_urlsafe(32) if live is not None and wants_token else None
    quit_token = token if on_quit is not None else None
    listen_token = token if on_listen is not None else None
    photo_token = token if photo_source is not None else None
    quit_lock = threading.Lock()
    quit_state = {"done": False}

    class H(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str = "text/html; charset=utf-8", allow: str = "") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Security-Policy", CSP)
            if allow:
                self.send_header("Allow", allow)
            self.end_headers(); self.wfile.write(body)

        def _port(self) -> int | None:
            addr = self.server.server_address   # AF_INET: (host, port)
            return addr[1] if isinstance(addr, tuple) and isinstance(addr[1], int) else None

        def _allowed(self, body: str, expected: str) -> bool:
            """Same-origin request on 127.0.0.1/localhost carrying the per-run token (constant-time compare)."""
            token = parse_qs(body).get("token", [""])[0]
            port = self._port()
            return (port is not None and same_origin(self.headers, port)
                    and secrets.compare_digest(token.encode(), expected.encode()))

        def _quit(self, body: str) -> None:
            if quit_token is None:
                return self._send(404, b"no quit in this mode", "text/plain; charset=utf-8")
            if not self._allowed(body, quit_token):
                return self._send(403, b"quit refused: bad token or origin", "text/plain; charset=utf-8")
            with quit_lock:
                first, quit_state["done"] = not quit_state["done"], True
            self._send(200, STOPPED_PAGE.encode())
            self.wfile.flush()
            if first and on_quit is not None:   # always set when quit_token is
                on_quit()

        def _listen(self, body: str) -> None:
            if listen_token is None or on_listen is None:
                return self._send(404, b"no voice link in this mode", "text/plain; charset=utf-8")
            if not self._allowed(body, listen_token):
                return self._send(403, b"listen refused: bad token or origin", "text/plain; charset=utf-8")
            if quit_state["done"]:
                return self._send(409, b"O Iris has stopped", "text/plain; charset=utf-8")
            on_listen()
            self.send_response(303)
            self.send_header("Location", "/")
            self.send_header("Content-Security-Policy", CSP)
            self.end_headers()

        def _photo(self, query: str) -> None:
            """GET /voice/photo?k=KEY&token=TOKEN: the token, the image origin rule, then only a current card's photo."""
            if photo_token is None or photo_source is None:
                return self._send(404, b"no photos in this mode", "text/plain; charset=utf-8")
            q = parse_qs(query)
            tok, key = q.get("token", [""])[0], q.get("k", [""])[0]
            port = self._port()
            if not (port is not None and same_origin_image(self.headers, port)
                    and secrets.compare_digest(tok.encode(), photo_token.encode())):
                return self._send(403, b"photo refused: bad token or origin", "text/plain; charset=utf-8")
            img = photo_source(key) if key and not quit_state["done"] else None
            self.send_response(200 if img else 404)
            self.send_header("Content-Type", "image/jpeg" if img else "text/plain; charset=utf-8")
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            body = img or b"no photo"
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = min(int(self.headers.get("Content-Length") or 0), 65536)
            body = self.rfile.read(n).decode("utf-8", "replace")
            if urlsplit(self.path).path == "/quit":
                return self._quit(body)
            if urlsplit(self.path).path == "/voice/listen":
                return self._listen(body)
            code, msg = face.post(self.path, body)
            self._send(code, face.render(msg).encode())

        def do_GET(self):
            port = self._port()
            if port is None or not local_host(self.headers, port):   # Origin not required: navigations omit it
                return self._send(403, b"forbidden: bad Host", "text/plain; charset=utf-8")
            if urlsplit(self.path).path == "/quit":
                return self._send(405, b"use the Quit button", "text/plain; charset=utf-8", allow="POST")
            if urlsplit(self.path).path == "/voice/listen":
                return self._send(405, b"use the Listen button", "text/plain; charset=utf-8", allow="POST")
            if urlsplit(self.path).path == PHOTO_PATH:
                return self._photo(urlsplit(self.path).query)
            if self.path == "/face":
                return self._send(200, face.render().encode())
            api = self.path == "/api/latest"
            if live is not None:
                snap = live()
                if quit_state["done"] and not api:
                    return self._send(200, STOPPED_PAGE.encode())
                body = json.dumps(snap).encode() if api else render(snap["result"], snap, quit_token, listen_token,
                                                                       photo_token).encode()
            else:
                res = run_scenario(scenario, 1, config=cfg)[-1]
                body = json.dumps(res).encode() if api else render(res).encode()
            self._send(200, body, "application/json" if api else "text/html; charset=utf-8")

        def log_message(self, fmt, *args):  # keep the live console readable
            if live is None:
                super().log_message(fmt, *args)
    return ThreadingHTTPServer(("127.0.0.1", port), H, bind_and_activate=bind_and_activate)


def serve(cfg: Config, port: int = 8765, scenario: str = "quiet", face_module=None,
          live: Callable[[], dict] | None = None) -> None:
    make_server(cfg, port, scenario, face_module, live).serve_forever()
