"""Live mode (`oiris run`): real adapters resolved by name from the registry, the Engine
stepping continuously in a background thread, each live session recorded as one sweep in the
encrypted store (M-14) with alert events in the sighting log (M-05): a changed (signature, level)
pair at once, an unchanged one at most once per SIGHTING_REPEAT_S. The retention purge
(guard/retention.py, config retention_days) runs when the store opens and every config retention_check_s (default
3600 s) while running; with the sighting log on it goes through SightingLog.purge, which chains a purge boundary first.

Record chain (ADR-0066): the sighting log resumes one hash chain across sessions. Every step calls its tick() (the
60 s idle anchor), stop() calls its close() under the store lock before the DB closes (the session's tail is anchored),
and snapshot()["chain"] is its chain_status from the open: new, resumed, or restarted with a reason code.

Power (ADR-0061): with config power.source "macos" (pmset) or "sim:<scene>" (oiris/adapters/power_sim.py), a reading
goes to the Governor every power.interval_s. Its plan's listen_duty / listen_period_s and voice_match go to the voice
listener (no new utterance while off; one being heard is finished). camera_max_fps, frame_stride and face_match go to the
face camera when one is configured (below). The plan's other fields have no live consumer yet and are shown, not
dropped: snapshot()["power"]["no_consumer"]. The default source "none" changes nothing.

Latency budgets (ADR-0061): every step, BudgetMonitor checks the StageTimer summaries against config budgets; the
stages over budget are snapshot()["budgets"], and the snapshot's health map gains a degraded "latency" entry. A breach
never changes behaviour.

Config (raw JSON key "live" in the config file; Config.load ignores unknown keys):
  {"live": {"adapters": [{"name": "ble_scan", "options": {}}, {"name": "wifi_macos"}],
            "hz": 1.0, "site": "live"}}
  "options" go to the adapter's open(cfg); optional "args" go to its registry factory
  (e.g. {"name": "sim", "args": {"scenario": "quiet", "period_s": 0.05}}).
An adapter that is unknown, missing a driver or fails to open is reported unavailable with
its reason; it never stops the run. Research signals only: no identity output from the sensors.

Voice link (M-17, ADR-0057): only when the M-16 and M-17 flags are both on, the runner builds the face
and voice modules through their gates and listens to the microphone in RAM (voice_link/listener.py).
Recognised enrolled people appear in snapshot()["voice"] and are announced on this device's speaker
(ADR-0058: a chime and "sounds like NAME", per voice_link.announce); any voice failure is a plain status
there and never stops live mode. With a mic_array mount in the rig config (ADR-0065), the microphone opens with
that many channels and a clear match also says where the voice came from ("sounds like NAME, on your left") when
the microphones can tell; a device that cannot open them falls back to mono without direction. Without such a
mount (the Mac today) everything is mono, exactly as before.

Outputs and motion (docs/ARCHITECTURE.md): one OutputArbiter carries every output. Engine alerts become
pop-ups (snapshot()["output"]["popup"]; spoken only with output.speak_alerts) and the voice-link
announcement goes through it at voice_link priority. An adapter with the "imu" capability (the simulated
"imu" adapter today) feeds a MotionEstimator whose state gates the outputs; with none, motion is unknown
and counts as still.

Directions (ADR-0060, oiris/spatial/cues.py): after each step, the signatures that have a {direction} phrase and a
matching spatial event become cues. snapshot()["cues"] lists them (signature, direction, evidence), the current
cue's direction is snapshot()["glasses"]["cue_direction"] (the edge marker), snapshot()["located"] lists the step's
spatial events, and each cue's phrase is spoken through the arbiter only with output.speak_alerts.

Timing (oiris/timing.py, docs/LIVE.md "Timing"): the runner owns one StageTimer each for the engine, the voice
listener and the output arbiter, passes them in, and shows their summaries in snapshot()["timing"]. Numbers only.

Capture (ADR-0063, oiris/capture/): the runner builds a CaptureInterlock from the config "capture" section and wraps the
voice capture in it: the microphone opens only after a mic grant (the recording light confirmed, the mute switch open),
every block passes guard_samples (a non-zero sample while muted is a latched mute_breach), and a revoked grant stops
listening within one block (the gate zeroes and drops every block after the revocation; the step closes the stream).
check() and tick() run at every step; snapshot()["capture"] carries the interlock's state codes and the profile. The
camera's acquire/release is camera_acquire()/camera_release(). The default mac profile (no light, no mute sense) always
grants, so live mode on the Mac behaves exactly as before.

Photo on the heard card (ADR-0057 decision 4): a card whose person has a stored photo carries "photo", a random per-run
key. voice_photo(key) returns the JPEG (FaceMatchModule.photo) only while that person is on a current card; the
dashboard serves it under its token and origin rules. Photos are never listed.

Face camera (ADR-0057 decision 5, docs/LIVE.md "Camera"): a "face_camera" entry in live.adapters is refused with a plain
reason while M-16 is off. Otherwise the runner loads M-16 (sharing the voice link's module when it runs), gives it to the
engine (process_frame under the store lock: counts and the encounter log only, never an alert) and wraps the adapter in
_GatedCamera: the device opens only after camera_acquire() grants (a refusal waits CAMERA_RETRY_S), a revoked grant or a
plan with face_match off (hot, critical) closes the device and drops the frames, and with the governor running every
frame passes FrameGate(camera_max_fps, frame_stride). Frames stay in RAM; the engine zeroes each one after M-16 saw it.
snapshot()["camera"] (only when configured) holds state, code, reason, fps and frames. Without the entry nothing changes."""
from __future__ import annotations
import dataclasses, getpass, ipaddress, json, logging, os, secrets, signal, socket, threading, time
from collections import Counter
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .capture.indicator import NoIndicator
from .capture.interlock import (CAMERA, CLOSED, INDICATOR_FAILED, INDICATOR_UNCONFIRMED, MIC, MUTE_BREACH, MUTE_UNKNOWN,
                                REFUSE_MUTED, CaptureInterlock, Grant, Refusal)
from .capture.mute import UnknownMute
from .config import (POWER_SIM_PREFIX, POWER_SOURCE_MACOS, POWER_SOURCE_NONE, Config, capture_settings,
                     power_governor_section, power_live_settings)
from .engine import Engine
from .guard.vocabulary import check_output, violations
from .modules.motion import IMU_CAP, MotionEstimator
from .modules.output import ALERT_LEVELS, FYI, OutputArbiter, OutputSettings, alert as output_alert
from .power.budgets import BudgetMonitor
from .power.duty import CAMERA_OFF_FPS, NORMAL, DutyPlan, FrameGate
from .power.governor import Decision, Governor, PowerReading, PowerSettings
from .registry import Registry, default_registry
from .schema import Health
from .spatial.cues import Cue, cues as spatial_cues, current as current_cue
from .spatial.merge import NS_PER_S
from .timing import StageTimer
from .ui.glasses import state_from_result

if TYPE_CHECKING:
    from .store.db import DB
    from .store.sightings import SightingLog, SweepTimeline

log = logging.getLogger(__name__)

DEFAULT_ADAPTERS = ["ble_scan", "wifi_macos"]
MAX_HZ = 10.0
CHECKPOINT_EVERY = 10   # steps between sweep-record checkpoints (outcome stays "open")
STOP_JOIN_S = 5.0       # stop(): bounded wait for the loop thread, once before and once after the bus stops
SIGHTING_REPEAT_S = 60.0   # an unchanged (signature, level) pair is written at most once per this many seconds
VOICE_FACE_CHAIN = "face-chain"   # key-store subkey of the M-16/M-17 change log (as face_match_cli uses)
TIMING_PARTS = ("engine", "voice", "output")   # snapshot()["timing"] keys, one StageTimer each
MS_PER_S = 1000.0
CAPTURE_RETRY_S = 5.0      # an ambient mic refused for the light tries again at most this often (no blinking light)
AMBIENT = "ambient"
LIGHT_CODES = (INDICATOR_UNCONFIRMED, INDICATOR_FAILED)
# The voice status reason when the interlock refuses or revokes the microphone (plain words, vocabulary-checked).
MIC_REFUSAL_REASONS = {
    INDICATOR_UNCONFIRMED: "microphone refused: the recording light could not be confirmed on",
    INDICATOR_FAILED: "microphone refused: the recording light failed",
    REFUSE_MUTED: "microphone refused: the mute switch is set to mute",
    MUTE_UNKNOWN: "microphone refused: the mute switch cannot be read",
    MUTE_BREACH: "microphone refused: sound came through while muted, a hardware fault (restart O Iris after a check)",
    CLOSED: "microphone refused: live mode is stopping",
}
# Plan fields with no live consumer yet (the camera, IR, glasses display and radio paths): shown in snapshot()["power"].
POWER_NO_CONSUMER = ("camera_max_fps", "frame_stride", "face_match", "ir_illuminator", "display_brightness_cap",
                     "radio_scan_interval_s")
POWER_DECIMALS = 1         # battery % and skin C in the snapshot
# The face camera (ADR-0057 decision 5): its live.adapters name, the plan fields it consumes, and its states and codes.
FACE_CAMERA = "face_camera"
CAMERA_PLAN_FIELDS = ("camera_max_fps", "frame_stride", "face_match")
CAMERA_RETRY_S = 5.0       # a camera refused by the interlock asks again at most this often (no blinking light)
CAMERA_IDLE_S = 0.1        # a camera read without an open device waits this long before the bus asks again
CAMERA_ENFORCE_WAIT_S = 0.25   # the live loop waits at most this long for a frame grab in progress before closing
CAMERA_FPS_DECIMALS = 1
CAM_STARTING, CAM_ON, CAM_OFF, CAM_WAITING, CAM_PAUSED, CAM_UNAVAILABLE = (
    "starting", "on", "off", "waiting", "paused", "unavailable")
CAM_M16_OFF, CAM_PAUSE_HEAT, CAM_PAUSE_BATTERY = "m16_off", "heat", "battery"   # codes beside CLOSED and the refusals
CAMERA_M16_OFF_REASON = "face matching (M-16) is off; set flags M-16 and opt in with `oiris face-match enable`"
BATTERY_REASONS = frozenset({"battery_low", "battery_critical"})
CAMERA_HEALTH_WORDS = {CAM_STARTING: "starting", CAM_OFF: "off", CAM_WAITING: "waiting for the recording light",
                       CAM_PAUSED: "paused by the power plan"}
PHOTO_KEY_BYTES = 16       # the per-run random key a heard card's photo is fetched by
LIVE_SIGHTING_CHAIN, LIVE_SIGHTING_ANCHOR = "live-sighting-chain", "live-sighting-anchor"   # key-store subkeys


class LiveConfigError(ValueError):
    pass


@dataclass
class LiveSettings:
    adapters: list = field(default_factory=lambda: [{"name": n, "options": {}} for n in DEFAULT_ADAPTERS])
    hz: float = 1.0
    site: str = "live"


def load_live_settings(path: str | Path | None) -> LiveSettings:
    """Read the raw "live" key from the config file. No file or no key -> defaults."""
    if path is None or not Path(path).exists():
        return LiveSettings()
    raw = json.loads(Path(path).read_text(encoding="utf-8")).get("live")
    if raw is None:
        return LiveSettings()
    if not isinstance(raw, dict) or set(raw) - {"adapters", "hz", "site"}:
        raise LiveConfigError("live must be an object with keys adapters, hz, site")
    s = LiveSettings()
    if "adapters" in raw:
        ads = raw["adapters"]
        if not isinstance(ads, list):
            raise LiveConfigError("live.adapters must be a list")
        s.adapters = []
        for a in ads:
            a = {"name": a} if isinstance(a, str) else a
            if (not isinstance(a, dict) or not isinstance(a.get("name"), str) or set(a) - {"name", "options", "args"}
                    or not isinstance(a.get("options", {}), dict) or not isinstance(a.get("args", {}), dict)):
                raise LiveConfigError(f"live.adapters entry {a!r}: needs name, optional options/args objects")
            s.adapters.append({"name": a["name"], "options": dict(a.get("options", {})), "args": dict(a.get("args", {}))})
    if "hz" in raw:
        hz = raw["hz"]
        if isinstance(hz, bool) or not isinstance(hz, (int, float)) or not 0 < hz <= MAX_HZ:
            raise LiveConfigError(f"live.hz must be a number in (0, {MAX_HZ}]")
        s.hz = float(hz)
    if "site" in raw:
        if not isinstance(raw["site"], str) or not raw["site"]:
            raise LiveConfigError("live.site must be a non-empty string")
        s.site = raw["site"]
    return s


# ---- network guard: local radios yes, outbound connections no ----------------------------

class OutboundBlocked(RuntimeError):
    pass


def _local_target(sock: socket.socket, address: Any) -> bool:
    if getattr(socket, "AF_UNIX", None) is not None and sock.family == socket.AF_UNIX:
        return True   # local IPC (e.g. BlueZ over D-Bus on Linux)
    host = address[0] if isinstance(address, tuple) and address else address
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(str(host).split("%", 1)[0]).is_loopback
    except ValueError:
        return False   # hostnames other than localhost would need DNS: refused


_GUARDED = ("connect", "connect_ex", "sendto", "sendmsg")


@contextmanager
def local_only_network():
    """Refuse every Python-level socket connect, and every sendto/sendmsg with an explicit
    address (UDP without connect), that is not loopback or AF_UNIX while live mode runs.
    Process-wide; restored on exit. It does not police subprocesses or native code."""
    real = {n: getattr(socket.socket, n) for n in _GUARDED if hasattr(socket.socket, n)}
    own = {n: socket.socket.__dict__[n] for n in real if n in socket.socket.__dict__}

    def check(sock, address, what):
        if not _local_target(sock, address):
            raise OutboundBlocked(f"outbound {what} to {address!r} refused in live mode")

    def connect(self, address):
        check(self, address, "connection")
        return real["connect"](self, address)

    def connect_ex(self, address):
        check(self, address, "connection")
        return real["connect_ex"](self, address)

    def sendto(self, data, *args):
        if args:                                  # sendto(data, address) or sendto(data, flags, address)
            check(self, args[-1], "datagram")
        return real["sendto"](self, data, *args)

    def sendmsg(self, buffers, *args, **kw):
        address = kw.get("address", args[2] if len(args) > 2 else None)
        if address is not None:                   # no address: a connected socket, already guarded
            check(self, address, "datagram")
        return real["sendmsg"](self, buffers, *args, **kw)

    guards = {"connect": connect, "connect_ex": connect_ex, "sendto": sendto, "sendmsg": sendmsg}
    for n in real:
        setattr(socket.socket, n, guards[n])
    try:
        yield
    finally:
        for n in real:
            if n in own:
                setattr(socket.socket, n, own[n])
            else:
                delattr(socket.socket, n)         # inherited from _socket.socket again


# ---- store ------------------------------------------------------------------------------

def open_store(cfg: Config):
    """Open the encrypted store the same way face_match_cli._ctx does (KeyStore in data_dir,
    passphrase from OIRIS_PASSPHRASE or a prompt). Returns (KeyStore, DB)."""
    from .security.keys import KeyStore
    from .store.db import DB
    d = Path(cfg.data_dir).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    ks = KeyStore(d / "keys.json")
    pw = os.environ.get("OIRIS_PASSPHRASE") or getpass.getpass("O Iris passphrase: ")
    ks.unlock(pw) if ks.exists() else ks.create(pw)
    return ks, DB(d / "asb.db", ks.subkey("db"))


def _safe_text(s: str) -> str:
    return s if not violations(s) else "reason withheld by vocabulary guard"


def _voice_unavailable(mode: str, reason: str) -> dict:
    """snapshot()["voice"] when the listener could not start."""
    return {"state": "unavailable", "reason": _safe_text(reason), "mode": mode, "notice": "", "listen_left_s": 0.0,
            "cards": [], "heard": 0}


def _sounddevice_capture(device: Any, channels: int) -> Any:
    """The default microphone capture: PortAudio, `channels` channels (1 = mono, exactly as before ADR-0065)."""
    from .modules.voice_link.capture import SoundDeviceCapture
    return SoundDeviceCapture(device, channels=channels)


class _GatedCapture:
    """Wraps the voice capture in the capture interlock (ADR-0063). start() opens the microphone only after a mic grant,
    else raises CaptureUnavailable with the plain refusal (the listener shows it). Every block passes guard_samples:
    a block arriving without a grant, or while muted, is zeroed and dropped. stop() closes it and releases the grant."""

    def __init__(self, inner: Any, interlock: CaptureInterlock):
        self._inner, self._il = inner, interlock
        self.is_open = False
        self.refused: str | None = None   # the last refusal or revocation code, until the microphone opens again

    def start(self, callback: Callable[[Any], None]) -> None:
        from .modules.voice_link.capture import CaptureUnavailable
        g = self._il.acquire(MIC)
        if isinstance(g, Refusal):
            self.refused = g.code
            raise CaptureUnavailable(MIC_REFUSAL_REASONS.get(g.code, "microphone refused"))

        def on_block(block: Any) -> None:
            if self._il.guard_samples(MIC, block):
                callback(block)

        try:
            self._inner.start(on_block)
        except BaseException:
            self._il.release(MIC)
            raise
        self.is_open, self.refused = True, None

    def stop(self) -> None:
        try:
            self._inner.stop()
        finally:
            self.is_open = False
            self._il.release(MIC)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _capture_ports(cfg: Config, clock: Callable[[], float]) -> tuple[Any, Any, dict]:
    """(indicator, mute sense, resolved settings) from the "capture" section. A configured port whose driver cannot
    open becomes one that never confirms (the light) or always reads unknown (the mute): capture fails closed."""
    s = capture_settings(cfg.capture)
    indicator: Any = None
    mute: Any = None
    if s["indicator"] == "required":
        indicator = NoIndicator()
        try:
            if s["indicator_adapter"] == "sim":
                from .adapters.capture_sim import SimIndicator
                indicator = SimIndicator(clock)
            elif s["indicator_adapter"] == "gpio":
                from .adapters.capture_gpio import GpioIndicator
                indicator = GpioIndicator(s["indicator_pin"])
        except Exception as e:  # noqa: BLE001 - DriverUnavailable or a GPIO error: no light, so nothing is granted
            log.warning("recording light unavailable: %s", e)
            indicator = NoIndicator(str(e))
    if s["mute_sense"] != "off" and s["mute_adapter"] != "none":
        try:
            if s["mute_adapter"] == "sim":
                from .adapters.capture_sim import SimMuteSwitch
                mute = SimMuteSwitch(clock)
            else:
                from .adapters.capture_gpio import GpioMuteSense
                mute = GpioMuteSense(s["mute_pin"])
        except Exception as e:  # noqa: BLE001 - DriverUnavailable or a GPIO error: the mute state is unknown
            log.warning("mute sense unavailable: %s", e)
            mute = UnknownMute(str(e))
    return indicator, mute, s


def build_interlock(cfg: Config, clock: Callable[[], float] = time.monotonic) -> CaptureInterlock:
    """The capture interlock the config "capture" section describes (ADR-0063)."""
    indicator, mute, s = _capture_ports(cfg, clock)
    return CaptureInterlock(indicator, mute, indicator_required=s["indicator"] == "required",
                            mute_required=s["mute_sense"] == "required",
                            confirm_timeout_s=float(s["confirm_timeout_ms"]) / MS_PER_S, tail_s=float(s["tail_s"]), clock=clock)


class _LockedFaceMatch:
    """M-16 as the engine sees it in live mode: process_frame runs under the store lock (the voice link's lock, so the
    two share one module safely) and never after the store is sealed. Counts only, as process_frame returns them."""

    def __init__(self, face: Any, lock: threading.Lock, closed: Callable[[], bool]):
        self._face, self._lock, self._closed = face, lock, closed
        self.frames = 0

    def process_frame(self, frame: Any, place: str | None = None) -> dict:
        with self._lock:
            if self._closed():
                return {"faces": 0, "matched": 0, "unrecognized": 0}
            self.frames += 1
            return self._face.process_frame(frame, place=place)


class _GatedCamera:
    """The face camera under the capture interlock and the power plan (ADR-0063, ADR-0061). The bus thread calls
    open/read/stop/close; the live loop calls enforce() every step.

    - The device opens only after acquire() grants. A refusal leaves it closed, degraded with the reason, and the next
      try comes retry_s later. A missing driver or bad options make it unavailable until restart.
    - A grant that is no longer held (revoked() returns its code) closes the device, then releases the grant; a frame
      read across the revocation is zeroed and dropped.
    - limits() -> (pause code or None, camera_max_fps or None, frame_stride or None). A pause code (the plan's
      face_match off, or camera_max_fps 0) closes the device; with a rate, every frame passes FrameGate(rate, stride).
    - stop()/close() close the device and release the grant."""

    def __init__(self, inner: Any, acquire: Callable[[], Any], release: Callable[[], None],
                 revoked: Callable[[], str | None], limits: Callable[[], tuple], clock: Callable[[], float],
                 retry_s: float = CAMERA_RETRY_S, idle_s: float = CAMERA_IDLE_S):
        self._inner = inner
        self.id = getattr(inner, "id", FACE_CAMERA)
        self.caps = set(getattr(inner, "caps", ()))
        self.bus = getattr(inner, "bus", "usb")
        self._acquire, self._release, self._revoked, self._limits = acquire, release, revoked, limits
        self._clock, self.retry_s, self.idle_s = clock, retry_s, idle_s
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._cfg: dict = {}
        self._device = False
        self._stopped = False
        self._dead = False
        self._retry_at: float | None = None
        self._gate: FrameGate | None = None
        self._gate_rate: tuple | None = None   # (camera_max_fps, frame_stride) the FrameGate was built with
        self.state, self.code, self.reason = CAM_STARTING, "", ""
        self.passed = 0                       # frames handed to the bus

    # -- lifecycle (bus thread) --
    def open(self, cfg: dict) -> None:
        with self._lock:
            self._cfg, self._stopped = dict(cfg or {}), False
            self._wake.clear()
            self._sync(self._clock(), may_open=True)

    def start(self) -> None:
        """Nothing to start: the device opens in open() or a later read(), only under a grant."""

    def stop(self) -> None:
        self._stopped = True
        self._wake.set()
        with self._lock:
            self._close_device()

    def close(self) -> None:
        self.stop()

    def _set(self, state: str, code: str = "", reason: str = "") -> None:
        self.state, self.code, self.reason = state, code, reason

    def _close_device(self) -> None:
        """Caller holds _lock. The device is closed (released) before the grant is given back."""
        if not self._device:
            return
        self._device, self._gate, self._gate_rate = False, None, None
        try:
            self._inner.close()
        except Exception as e:  # noqa: BLE001 - the grant is released whatever the driver does
            log.warning("face camera close failed: %s", e)
        finally:
            self._release()

    def _sync(self, now: float, may_open: bool) -> bool:
        """Caller holds _lock. Applies the pause and the grant, and opens the device when allowed. True: frames flow."""
        pause = self._limits()[0]
        if pause is not None:
            self._close_device()
            self._retry_at = None             # the plan allowing it again reopens at once
            self._set(CAM_PAUSED, pause)
            return False
        if self._device:
            code = self._revoked()
            if code is None:
                return True
            self._close_device()
            self._retry_at = now + self.retry_s
            self._set(CAM_OFF if code == CLOSED else CAM_WAITING, code)
            return False
        if self.state == CAM_PAUSED:
            self._set(CAM_STARTING)
        if self._stopped or self._dead or not may_open or (self._retry_at is not None and now < self._retry_at):
            return False
        return self._open_device(now)

    def _open_device(self, now: float) -> bool:
        """Caller holds _lock. The interlock first; the device only under a grant."""
        from .adapters.drivers import DriverUnavailable
        g = self._acquire()
        if not g:
            code = str(getattr(g, "code", CLOSED))
            self._retry_at = now + self.retry_s
            self._set(CAM_OFF if code == CLOSED else CAM_WAITING, code)
            return False
        try:
            self._inner.open(self._cfg)
        except (DriverUnavailable, ValueError) as e:   # no driver, no such camera or bad options: until restart
            self._release()
            self._dead = True
            self._set(CAM_UNAVAILABLE, reason=_safe_text(f"{type(e).__name__}: {e}"))
            return False
        except Exception as e:  # noqa: BLE001 - any other open error: released, shown, tried again later
            self._release()
            self._retry_at = now + self.retry_s
            self._set(CAM_UNAVAILABLE, reason=_safe_text(f"{type(e).__name__}: {e}"))
            return False
        self._device, self._retry_at = True, None
        self._set(CAM_ON)
        return True

    # -- frames (bus thread) --
    def read(self, timeout_s: float) -> Any:
        """One frame or None. The device is only touched under _lock; the wait for the next frame time (the inner
        adapter's wait_s(), else idle_s) happens outside it, so enforce() is never held up by the frame pacing."""
        from .adapters.camera import READ_WAIT_SHARE, zero_frame
        wait = self.idle_s
        with self._lock:
            if self._sync(self._clock(), may_open=True):
                s = self._inner.read(0.0)
                if s is None:
                    pace = getattr(self._inner, "wait_s", None)
                    wait = float(pace()) if callable(pace) else self.idle_s
                elif not self._sync(self._clock(), may_open=False):   # revoked or paused during the read
                    zero_frame(s.data)
                    return None
                else:
                    _, max_fps, stride = self._limits()
                    if max_fps is not None:
                        rate = (float(max_fps), int(stride or 1))
                        if rate != self._gate_rate:
                            self._gate, self._gate_rate = FrameGate(*rate), rate
                        if self._gate is not None and not self._gate.admit(self._clock()):
                            zero_frame(s.data)
                            return None
                    self.passed += 1
                    return s
        self._wake.wait(min(max(wait, 0.0), max(timeout_s, 0.0) * READ_WAIT_SHARE))
        return None

    # -- the live loop --
    def enforce(self) -> bool:
        """Every step: True while frames may flow. Otherwise the device is closed now; if the bus thread is still inside
        one grab after CAMERA_ENFORCE_WAIT_S, that read closes it as it returns and drops its frame."""
        if self._limits()[0] is None and self._device and self._revoked() is None:
            return True
        if self._lock.acquire(timeout=CAMERA_ENFORCE_WAIT_S):
            try:
                self._sync(self._clock(), may_open=False)
            finally:
                self._lock.release()
        return False

    def health(self) -> tuple[Health, str]:
        if self.state == CAM_ON:
            return self._inner.health()
        if self.state == CAM_UNAVAILABLE:
            return Health.FAILED, self.reason or "unavailable"
        return Health.DEGRADED, CAMERA_HEALTH_WORDS.get(self.state, self.state)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _Opened:
    """The runner opens each adapter itself (to report unavailability up front). The bus opens
    adapters again in its thread; the first of those calls is skipped, restarts reopen."""

    def __init__(self, inner: Any):
        self._inner, self._fresh = inner, True

    def open(self, cfg: dict) -> None:
        if self._fresh:
            self._fresh = False
            return
        self._inner.open(cfg)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _ImuTap:
    """Feeds every sample an "imu" adapter reads to the motion estimator, in the bus thread, then passes it on."""

    def __init__(self, inner: Any, motion: MotionEstimator):
        self._inner, self._motion = inner, motion

    def read(self, timeout_s: float) -> Any:
        s = self._inner.read(timeout_s)
        if s is not None:
            try:
                self._motion.add_sample(s.data)
            except Exception as e:  # noqa: BLE001 - a bad IMU sample must not stop the bus; motion goes stale
                log.warning("imu sample not used: %s", e)
        return s

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _SimPower:
    """config power.source "sim:<scene>": the deterministic power_sim model, advanced by the runner's clock under the
    plan in force (a closed loop: throttling cools it). Readings are timed in the runner's clock."""

    def __init__(self, scene: str, clock: Callable[[], float]):
        from .adapters.power_sim import PowerSim
        self.sim, self.clock = PowerSim(scene), clock
        self._t: float | None = None

    def read(self, plan: DutyPlan) -> PowerReading:
        now = self.clock()
        if self._t is not None and now > self._t:
            self.sim.advance(now - self._t, plan)
        self._t = now
        return dataclasses.replace(self.sim.reading(), t=now)


class _MacPower:
    """config power.source "macos": `pmset -g batt` (battery and charging; no temperature on a Mac). Never raises."""

    def __init__(self, clock: Callable[[], float]):
        from .adapters.power_macos import MacBattery
        self.battery = MacBattery(clock=clock)

    def read(self, plan: DutyPlan) -> PowerReading:
        return self.battery.read()


def build_power_source(cfg: Config, clock: Callable[[], float]) -> Any:
    """The reading source config power.source names (an object with read(plan) -> PowerReading), or None for "none"."""
    source, _ = power_live_settings(cfg.power)
    if source == POWER_SOURCE_NONE:
        return None
    hosts: dict[str, Callable[[Callable[[], float]], Any]] = {POWER_SOURCE_MACOS: _MacPower}
    if source in hosts:
        return hosts[source](clock)
    if source.startswith(POWER_SIM_PREFIX):
        return _SimPower(source[len(POWER_SIM_PREFIX):], clock)
    raise LiveConfigError(f"power.source {source!r} is not known")


def _budget_reason(over: list[dict]) -> str:
    """The health map's plain "latency" reason: each stage over budget with its p95 and budget, in milliseconds."""
    return "over budget: " + "; ".join(f"{o['part']} {o['stage'].replace('_', ' ')} p95 {o['p95_ms']:.0f} ms "
                                       f"(budget {o['budget_ms']:.0f} ms)" for o in over)


class LiveRunner:
    def __init__(self, cfg: Config, settings: LiveSettings | None = None, registry: Registry | None = None,
                 store: tuple | None = None, join_s: float = STOP_JOIN_S, clock: Callable[[], float] | None = None,
                 face_backend: Any = None, speaker: Any = None, voice_capture: Any = None,
                 announce_player: Any = None, haptic: Callable[[], None] | None = None,
                 voice_capture_factory: Callable[[Any, int], Any] | None = None,
                 interlock: CaptureInterlock | None = None, power_source: Any = None):
        """store: (KeyStore, DB) already open, or None to open it with open_store(cfg).
        join_s: how long stop() waits for the loop thread in each of its two joins.
        clock: monotonic seconds for the sighting repeat, purge, power-reading and chain-anchor intervals and the
        listening duty (default time.monotonic).
        face_backend, speaker, voice_capture: injected M-16 backend, M-17 speaker SlotManager and
        microphone capture (tests); None builds them from config (voice only with M-16 and M-17 on).
        voice_capture_factory(device, channels): builds the microphone capture when none is injected (default
        SoundDeviceCapture). channels is the rig's mic_array microphone count (ADR-0065), else 1.
        announce_player: the output arbiter's player for the chime and speech (tests inject a fake); None =
        MacPlayer on macOS, else the generic player (announce.platform_player).
        haptic: the arbiter's haptic pulse; None = no haptic device (the Mac).
        interlock: the capture interlock (tests inject one with simulated ports); None builds it from the config
        "capture" section at start (ADR-0063).
        power_source: an object with read(plan) -> PowerReading (tests inject one); None builds it from config
        power.source at start ("none": no governor)."""
        self.cfg = cfg
        self.settings = settings or LiveSettings()
        self.registry = registry or default_registry(cfg.plugins)
        self._store = store
        self.adapters: dict[str, dict] = {}    # configured name/id -> {"available", "reason", "name"}
        self.steps = 0
        self.sweep_id: str | None = None
        self.started: float | None = None
        self.outcome: str | None = None
        self.join_s = join_s
        self._lock = threading.Lock()
        # Every store write (sighting, sweep checkpoint/finish, step counts) and the close happen
        # under _store_lock; once _closed is set no write from the loop can reach the store.
        self._store_lock = threading.Lock()
        self._closed = False
        self._close_deferred = False      # loop outlived stop(): the loop thread closes the DB on exit
        self._loop_done = False           # set by the loop thread itself, under _store_lock
        self.loop_alive_at_stop = False
        self._latest: dict | None = None
        self._latest_at: float | None = None   # _clock() when _latest arrived (the Located list's "seconds ago")
        self._cues: list[Cue] = []            # direction cues of _latest (oiris/spatial/cues.py)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._counts: Counter = Counter()
        self._sightings = 0               # sightings written
        self._events = 0                  # every engine event, written or not
        self._suppressed = 0              # unchanged (signature, level) repeats inside SIGHTING_REPEAT_S
        self._last_pair: tuple | None = None
        self._last_written = 0.0
        self._purged = 0                  # rows deleted by retention purges this run
        self._last_purge: float | None = None       # _clock() of the last purge attempt (schedules the next)
        self._last_purge_at: float | None = None    # wall time of the last successful purge (the dashboard)
        self._clock = clock or time.monotonic
        self._net: AbstractContextManager[None] | None = None
        self.engine: Engine | None = None
        self.db: DB | None = None
        self._timeline: SweepTimeline | None = None
        self._log: SightingLog | None = None
        self._face_backend, self._speaker, self._voice_capture = face_backend, speaker, voice_capture
        self._announce_player = announce_player
        self._voice_capture_factory = voice_capture_factory or _sounddevice_capture
        self.voice: Any = None                 # VoiceListener while voice link listens
        self.announcer: Any = None             # Announcer while voice link listens (ADR-0058)
        self._voice_status: dict | None = None  # plain status when it could not start
        self._haptic = haptic
        self.motion = MotionEstimator()        # unknown (counts as still) until an "imu" adapter delivers
        self.output: OutputArbiter | None = None   # every audible, visual and haptic output while running
        self._output_final: dict | None = None     # its last state after stop
        self.timers: dict[str, StageTimer] = {k: StageTimer() for k in TIMING_PARTS}   # kept after stop
        self.interlock: CaptureInterlock | None = interlock   # camera and microphone rules (ADR-0063); kept after stop
        self._capture_profile = str(capture_settings(cfg.capture)["profile"])
        self._mic_gate: _GatedCapture | None = None   # the interlock's wrapper around the voice capture
        self._mic_retry_at = 0.0
        # Power governor (ADR-0061): source, governor and the latest decision; all None with power.source "none".
        self._power_source: Any = power_source
        self._power_source_name = "injected" if power_source is not None else power_live_settings(cfg.power)[0]
        self._power_interval_s = power_live_settings(cfg.power)[1]
        self._governor: Governor | None = None
        self._power_decision: Decision | None = None
        self._power_reading: PowerReading | None = None
        self._power_next: float | None = None
        # Latency budgets (ADR-0061): the stages over budget after the last step.
        self._budgets = BudgetMonitor(cfg.budgets)
        self._over: list[dict] = []
        # Heard-card photos: per-run random key <-> person, for the people on the current cards only.
        self._face: Any = None                 # the M-16 module while voice link listens (photo lookup)
        self._photo_keys: dict[str, str] = {}  # person_id -> key, kept for the run so a card's key is stable
        # Face camera (ADR-0057 decision 5): its live.adapters entries, the gate around it and M-16 as the engine sees it.
        self._camera_specs: list[tuple[dict, dict]] = []     # (spec, open options) of each "face_camera" entry
        self._camera_gate: _GatedCamera | None = None
        self._camera_face: _LockedFaceMatch | None = None
        self._camera_status: dict | None = None               # snapshot()["camera"] when the camera was refused

    # -- setup ------------------------------------------------------------------------------
    def _resolve(self, ks) -> None:
        engine = self.engine
        if engine is None:
            raise RuntimeError("adapters resolved before the engine exists")
        for spec in self.settings.adapters:
            name, opts = spec["name"], dict(spec.get("options", {}))
            if ks is not None:
                # per-site HMAC key for radio identifiers; never taken from the config file
                opts["site_key"] = ks.subkey("site:" + self.settings.site)
            if name == FACE_CAMERA:
                self._camera_specs.append((spec, opts))   # never opened here: _start_camera, under its gates
                continue
            try:
                made = self.registry.create("sensor", name, **spec.get("args", {}))
            except Exception as e:  # noqa: BLE001 - unknown name, missing plugin, bad args
                self._unavailable(name, name, f"{type(e).__name__}: {e}")
                continue
            for a in (made if isinstance(made, list) else [made]):
                aid = getattr(a, "id", name)
                if aid in engine.bus.adapters:
                    self._unavailable(aid, name, "duplicate adapter id")
                    continue
                try:
                    a.open(opts)
                except Exception as e:  # noqa: BLE001 - DriverUnavailable or any open error
                    self._unavailable(aid, name, f"{type(e).__name__}: {e}")
                    try:
                        a.close()
                    except Exception as ce:  # noqa: BLE001
                        log.warning("live adapter %s close after failed open: %s", aid, ce)
                    continue
                tapped = _ImuTap(a, self.motion) if IMU_CAP in getattr(a, "caps", ()) else a
                engine.add_adapter(_Opened(tapped), opts)
                self.adapters[aid] = {"name": name, "available": True, "reason": ""}

    def _unavailable(self, key: str, name: str, reason: str) -> None:
        log.warning("live adapter %s unavailable: %s", key, reason)
        self.adapters[key] = {"name": name, "available": False, "reason": _safe_text(reason)}

    def start(self, background: bool = True) -> "LiveRunner":
        if self._thread is not None:
            raise RuntimeError("already started")
        net = local_only_network()
        net.__enter__()
        self._net = net
        try:
            ks, db = self._store or open_store(self.cfg)
            self.db = db
            self.engine = Engine(self.cfg, registry=self.registry, timer=self.timers["engine"])
            self.engine.on_event = self._on_event
            self._resolve(ks)
            from .store.sightings import SightingLog, SweepTimeline
            self._timeline = SweepTimeline(db) if self.cfg.enabled("M-14") else None
            self._log = None
            if self.cfg.enabled("M-05"):
                from nacl.signing import SigningKey
                from .security.hashchain import HashChain
                self._log = SightingLog(db, HashChain(ks.subkey(LIVE_SIGHTING_CHAIN),
                                                           SigningKey(ks.subkey(LIVE_SIGHTING_ANCHOR)),
                                                           clock=self._clock))
                log.info("record chain %s", self._log.chain_status.get("state"))
            with self._store_lock:
                self._purge()                    # retention (ETHICS.md): expired rows go at every open
            if self.interlock is None:
                self.interlock = build_interlock(self.cfg, self._clock)
            self._start_power()                  # the first reading, so the voice listener starts under its plan
            self._start_output()
            self._start_voice(ks, db)
            self._start_camera(ks, db)           # after the voice link: one M-16 module for both
            self.sweep_id = self._timeline.start(self.settings.site) if self._timeline else None
            self.started = time.time()
            self.engine.bus.start()
        except BaseException:
            self._stop_voice()
            self._close_capture()
            self._close_output()
            if self.engine is not None:
                self.engine.bus.stop()
            if self._store is None and self.db is not None:
                self.db.close()
            net.__exit__(None, None, None)
            self._net = None
            raise
        # background=False: the caller drives step() itself (tests); stop() still cleans up.
        self._thread = threading.Thread(target=self._loop, args=(background,), name="oiris-live", daemon=True)
        self._thread.start()
        return self

    # -- outputs and motion ----------------------------------------------------------------------
    def _start_output(self) -> None:
        """The host's player only when something can sound (spoken alerts, or the voice link may start);
        otherwise nothing probes the audio devices and the arbiter carries pop-ups and haptics only."""
        settings = OutputSettings.from_config(self.cfg)
        player = self._announce_player
        audible = settings.speak_alerts or (self.cfg.enabled("M-16") and self.cfg.enabled("M-17"))
        if player is None and audible:
            from .modules.voice_link.announce import platform_player
            player = platform_player()
        self.output = OutputArbiter(player, haptic=self._haptic, settings=settings, motion=self.motion.state,
                                    timer=self.timers["output"])

    def _close_output(self) -> None:
        arb, self.output = self.output, None
        if arb is None:
            return
        try:
            arb.close()
            self._output_final = arb.state()
        except Exception as e:  # noqa: BLE001 - stopping live mode must go on
            log.error("output arbiter close failed: %s", e)

    def _output_snapshot(self) -> dict:
        """snapshot()["output"]: motion state, whether an IMU feeds it, the current pop-up, counts by outcome."""
        arb = self.output
        if arb is not None:
            st = arb.state()
        else:
            st = dict(self._output_final or {"popup": None, "speaking": None, "queued": 0, "counts": {}})
        adapters = self.engine.bus.adapters.values() if self.engine is not None else []
        return {"motion": self.motion.state(), "imu": any(IMU_CAP in getattr(a, "caps", ()) for a in adapters), **st}

    # -- voice link (M-17) -------------------------------------------------------------------
    def _start_voice(self, ks, db) -> None:
        """Start listening when M-16 and M-17 are both on. Every failure (not opted in, models missing,
        sounddevice missing, no device, model failure) becomes a plain status; live mode carries on."""
        if not (self.cfg.enabled("M-16") and self.cfg.enabled("M-17")):
            return
        mode = str((self.cfg.voice_link or {}).get("listening", "ambient"))
        try:
            if ks is None:
                raise RuntimeError("no key store, so the encrypted voice templates cannot be opened")
            from .modules import face_match_gate, voice_link_gate
            from .modules.face_match.optin import DbChain
            chain = DbChain(db, ks.subkey(VOICE_FACE_CHAIN))
            try:
                face = face_match_gate.load(self.cfg, db, ks, self.registry, chain=chain, backend=self._face_backend)
            except Exception as e:
                raise RuntimeError(f"face matching could not load: {e}") from e
            vm = voice_link_gate.load(self.cfg, db, ks, self.registry, face, chain=chain, speaker=self._speaker)
            mode = vm.cfg.listening
            state, why = vm.health()
            if state == "unavailable":
                raise RuntimeError(f"the speaker model is unavailable ({why})")
            from .modules.voice_link.announce import Announcer
            from .modules.voice_link.listener import VoiceListener
            if self.output is None:
                raise RuntimeError("the output arbiter is not running")
            cap, direction = self._open_voice_capture(vm.cfg.input_device)
            if self.interlock is None:
                raise RuntimeError("the capture interlock is not running")
            cap = self._mic_gate = _GatedCapture(cap, self.interlock)   # the microphone opens only on a mic grant
            # The chime and phrase play through the arbiter (and its player); "off" requests nothing.
            ann = self.announcer = Announcer(vm.cfg.announce, vm.cfg.announce_repeat_s, arbiter=self.output)
            self.voice = VoiceListener(vm, cap, process_lock=self._store_lock, place=self.settings.site,
                                       active=lambda: not self._closed,
                                       on_heard=lambda card: ann.announce(card["person_id"], card["name"],
                                                                          card.get("direction")),
                                       paused=ann.playing, timer=self.timers["voice"], direction=direction,
                                       duty_clock=self._clock)
            self._face = face
            self._apply_plan()                   # the governor's plan in force before the first utterance
            self.voice.start()
            st = self.voice.status()
            if st["state"] == "unavailable":
                log.warning("voice link unavailable: %s", st["reason"])
        except Exception as e:  # noqa: BLE001 - live mode never dies because of voice; the reason is shown
            log.warning("voice link unavailable: %s", e)
            if self.voice is not None:           # built but failed later: its worker and microphone stop
                self._stop_voice()
            self.voice, self._face = None, None
            self._close_announcer()
            self._voice_status = _voice_unavailable(mode, str(e))

    def _voice_mics_head_m(self) -> list | None:
        """Head-frame positions of the first mic_array mount's microphones (ADR-0065), or None (no such mount: mono).
        The capture's channel order is the mount's mic_positions_m order."""
        from .spatial.rig import Rig
        for m in Rig.from_config(self.cfg.rig).mounts.values():
            if m.kind == "mic_array":
                return [[float(c) for c in m.mount_vec_to_head(p) + m.position_m] for p in m.mic_positions_m]
        return None

    def _open_voice_capture(self, device: Any) -> tuple[Any, Any]:
        """(capture, VoiceDirection or None). With a mic_array mount, an N-channel capture and a direction estimate;
        a device that cannot open N channels falls back to mono, with a plain log line and no direction. Without one,
        mono exactly as before. An injected voice_capture is used as given; it gets a direction only when its
        `channels` matches the mount."""
        from .modules.voice_link.capture import CHANNELS, SAMPLE_RATE
        from .modules.voice_link.direction import VoiceDirection
        mics = self._voice_mics_head_m()
        n = len(mics) if mics else CHANNELS
        cap = self._voice_capture
        if cap is None and n > CHANNELS:
            try:
                cap = self._voice_capture_factory(device, n)
            except Exception as e:  # noqa: BLE001 - no direction is fine; no voice is not
                log.warning("voice direction off: the microphone could not open %d channels (%s); listening in mono", n, e)
        if cap is None:
            cap = self._voice_capture_factory(device, CHANNELS)
        if mics is None or int(getattr(cap, "channels", CHANNELS)) != n:
            return cap, None
        return cap, VoiceDirection(mics, int(getattr(cap, "sample_rate", SAMPLE_RATE)))

    def _close_announcer(self) -> None:
        ann, self.announcer = self.announcer, None
        if ann is not None:
            try:
                ann.close()
            except Exception as e:  # noqa: BLE001 - stopping live mode must go on
                log.error("announcer close failed: %s", e)

    # -- capture interlock (ADR-0063) -------------------------------------------------------------
    def camera_acquire(self) -> Grant | Refusal:
        """For the camera path: a Grant before any frame is taken, or a Refusal with its code."""
        il = self.interlock
        if il is None or self._closed:
            return Refusal(CAMERA, CLOSED)
        return il.acquire(CAMERA)

    def camera_release(self) -> None:
        if self.interlock is not None:
            self.interlock.release(CAMERA)

    # -- face camera (ADR-0057 decision 5) ----------------------------------------------------------
    def _start_camera(self, ks, db) -> None:
        """A configured face_camera: refused while M-16 is off (or M-16 cannot load), else wrapped in _GatedCamera and
        added to the bus, with M-16 handed to the engine. Nothing opens here; the bus opens it under its gates."""
        if not self._camera_specs:
            return
        engine = self.engine
        if engine is None:
            raise RuntimeError("the camera started before the engine exists")
        (spec, opts), extra = self._camera_specs[0], self._camera_specs[1:]
        for _ in extra:
            self._unavailable(FACE_CAMERA + "#2", FACE_CAMERA, "only one face camera is supported")
        if not self.cfg.enabled("M-16"):
            self._unavailable(FACE_CAMERA, FACE_CAMERA, CAMERA_M16_OFF_REASON)
            self._camera_status = {"state": CAM_OFF, "code": CAM_M16_OFF, "reason": "", "fps": None, "frames": 0}
            return
        try:
            face = self._face or self._load_face(ks, db)
            made = self.registry.create("sensor", FACE_CAMERA, **spec.get("args", {}))
            if getattr(made, "id", FACE_CAMERA) in engine.bus.adapters:
                raise RuntimeError("duplicate adapter id")
        except Exception as e:  # noqa: BLE001 - not opted in, no backend, bad args: shown, live mode goes on
            reason = _safe_text(f"{type(e).__name__}: {e}")
            self._unavailable(FACE_CAMERA, FACE_CAMERA, reason)
            self._camera_status = {"state": CAM_UNAVAILABLE, "code": "", "reason": reason, "fps": None, "frames": 0}
            return
        gate = _GatedCamera(made, self.camera_acquire, self.camera_release, self._camera_revoked, self._camera_limits,
                            self._clock)
        self._camera_face = _LockedFaceMatch(face, self._store_lock, lambda: self._closed)
        engine.face_match, engine.face_place = self._camera_face, self.settings.site
        engine.add_adapter(gate, opts)
        self._camera_gate = gate
        self.adapters[gate.id] = {"name": FACE_CAMERA, "available": True, "reason": ""}

    def _load_face(self, ks, db) -> Any:
        """M-16 through its gate when the voice link did not load it (M-17 off or not running)."""
        if ks is None:
            raise RuntimeError("no key store, so face matching cannot open its templates")
        from .modules import face_match_gate
        from .modules.face_match.optin import DbChain
        face = face_match_gate.load(self.cfg, db, ks, self.registry, chain=DbChain(db, ks.subkey(VOICE_FACE_CHAIN)),
                                    backend=self._face_backend)
        if face is None:
            raise RuntimeError("face matching (M-16) is off")
        return face

    def _camera_revoked(self) -> str | None:
        """None while the camera holds its grant, else the refusal or revocation code."""
        il = self.interlock
        if il is None or self._closed:
            return CLOSED
        if il.held(CAMERA):
            return None
        return il.refusal(CAMERA) or CLOSED

    def _camera_limits(self) -> tuple[str | None, float | None, int | None]:
        """(pause code, camera_max_fps, frame_stride) from the plan in force; (None, None, None) without a governor."""
        if self._governor is None:
            return None, None, None
        plan = self._plan()
        if not plan.face_match or plan.camera_max_fps <= CAMERA_OFF_FPS:
            d = self._power_decision
            reasons = set(d.reasons) if d is not None else set()
            return (CAM_PAUSE_BATTERY if reasons and reasons <= BATTERY_REASONS else CAM_PAUSE_HEAT), None, None
        return None, plan.camera_max_fps, plan.frame_stride

    def _camera_step(self) -> None:
        """Every step, after the interlock's check: a camera that may not deliver is closed and its queued frames are
        zeroed and dropped, so no frame taken before a revocation or a pause reaches M-16."""
        gate, engine = self._camera_gate, self.engine
        if gate is None or engine is None or gate.enforce():
            return
        from .adapters.camera import zero_frame
        for s in engine.bus.drain(gate.id):
            zero_frame(s.data)

    def _camera_snapshot(self) -> dict | None:
        """snapshot()["camera"], only when a face camera is configured: state, code, reason, fps (on only), frames."""
        gate = self._camera_gate
        if gate is None:
            return dict(self._camera_status) if self._camera_status is not None else None
        fps: float | None = None
        if gate.state == CAM_ON:
            _, max_fps, stride = self._camera_limits()
            fps = float(getattr(gate, "fps", 0.0) or 0.0)
            if max_fps is not None:
                fps = min(fps, float(max_fps)) / int(stride or 1)
            fps = round(fps, CAMERA_FPS_DECIMALS)
        frames = self._camera_face.frames if self._camera_face is not None else 0
        return {"state": gate.state, "code": gate.code, "reason": _safe_text(gate.reason), "fps": fps, "frames": frames}

    def _capture_step(self) -> None:
        """Every step: check() and tick(), then the microphone follows the grant. A revoked grant closes the stream
        (its blocks were already dropped by the gate); an ambient microphone refused by the interlock tries again once
        the mute allows it (and, after a light failure, at most every CAPTURE_RETRY_S, so the light does not blink)."""
        il = self.interlock
        if il is None or self._closed:
            return
        now = self._clock()
        try:
            il.check(now)
            il.tick(now)
        except Exception as e:  # noqa: BLE001 - the loop never dies; capture fails closed instead
            log.error("capture check failed, capture closed: %s", e)
            il.close()
        self._camera_step()
        listener, gate = self.voice, self._mic_gate
        if listener is None or gate is None:
            return
        if gate.is_open and not il.held(MIC):
            gate.refused = il.refusal(MIC) or CLOSED
            if gate.refused in LIGHT_CODES:
                self._mic_retry_at = now + CAPTURE_RETRY_S
            listener.pause_mic()           # the listener's own stop path: the stream stops, the grant is released
        elif (not gate.is_open and gate.refused is not None and listener.mode == AMBIENT and now >= self._mic_retry_at
              and il.mic_may_grant()):
            listener.resume_mic()          # opens through the gate; a refusal keeps its plain reason
            if gate.refused in LIGHT_CODES:
                self._mic_retry_at = now + CAPTURE_RETRY_S

    def _close_capture(self) -> None:
        """Every grant revoked and the light off; the GPIO ports released."""
        il = self.interlock
        if il is None:
            return
        try:
            il.close()
        except Exception as e:  # noqa: BLE001 - stopping live mode must go on
            log.error("capture interlock close failed: %s", e)
        for port in (il.indicator, il.mute):
            close = getattr(port, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as e:  # noqa: BLE001 - stopping live mode must go on
                    log.error("capture port close failed: %s", e)

    # -- power governor and latency budgets (ADR-0061) -----------------------------------------------
    def _start_power(self) -> None:
        """Build the reading source and governor from config power (none: nothing at all), then take the first reading."""
        if self._power_source is None:
            self._power_source = build_power_source(self.cfg, self._clock)
        if self._power_source is None:
            return
        self._governor = Governor(PowerSettings.from_config(power_governor_section(self.cfg.power)))
        self._power_step()

    def _plan(self) -> DutyPlan:
        d = self._power_decision
        if d is not None:
            return d.plan
        return self._governor.settings.plans[NORMAL] if self._governor is not None else PowerSettings().plans[NORMAL]

    def _power_step(self) -> None:
        """Every power.interval_s: a reading, the governor's decision, the plan applied. A failure keeps the last plan."""
        gov, src = self._governor, self._power_source
        if gov is None or src is None or self._closed:
            return
        now = self._clock()
        if self._power_next is not None and now < self._power_next:
            return
        self._power_next = now + self._power_interval_s
        try:
            reading = src.read(self._plan())
            d = gov.update(reading, now)
        except Exception as e:  # noqa: BLE001 - a broken reading must not stop live mode; the last plan stays
            log.error("power reading failed, plan unchanged: %s", e)
            return
        prev = self._power_decision
        with self._lock:
            self._power_decision, self._power_reading = d, reading
        if prev is None or prev.level != d.level:
            log.info("power level %s (%s)", d.level, ", ".join(d.reasons) or "no reason")
        self._apply_plan()

    def _apply_plan(self) -> None:
        """The plan's listening duty and voice matching go to the voice listener. Its other fields wait for their paths."""
        listener = self.voice
        if listener is None or self._governor is None:
            return
        plan = self._plan()
        try:
            listener.set_duty(plan.listen_duty, plan.listen_period_s)
            listener.set_matching(plan.voice_match)
        except Exception as e:  # noqa: BLE001 - the listener keeps its last plan; live mode goes on
            log.error("power plan not applied to the voice listener: %s", e)

    def _power_snapshot(self) -> dict | None:
        """snapshot()["power"]: codes and numbers only. None with power.source "none"."""
        if self._governor is None:
            return None
        with self._lock:
            d, r = self._power_decision, self._power_reading
        plan = self._plan()

        def num(v: Any) -> float | None:
            return round(float(v), POWER_DECIMALS) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
        return {"source": self._power_source_name, "level": d.level if d else NORMAL,
                "reasons": list(d.reasons) if d else [], "plan": plan.to_dict(),
                "battery_pct": num(r.battery_pct) if r else None,
                "charging": r.charging if r is not None and isinstance(r.charging, bool) else None,
                "skin_c": num(r.skin_c) if r else None,
                "no_consumer": [k for k in POWER_NO_CONSUMER
                                if self._camera_gate is None or k not in CAMERA_PLAN_FIELDS]}

    def _budget_step(self) -> None:
        """Every step: the stages over budget. Logged once per change; a breach never changes behaviour."""
        try:
            over = self._budgets.check(self.timing())
        except Exception as e:  # noqa: BLE001 - a measurement must never stop live mode
            log.error("latency budget check failed: %s", e)
            return
        with self._lock:
            prev, self._over = self._over, over
        keys = [(o["part"], o["stage"]) for o in over]
        if keys != [(o["part"], o["stage"]) for o in prev]:
            if over:
                log.warning("latency %s", _budget_reason(over))
            else:
                log.info("latency back within budget")

    # -- heard-card photos -------------------------------------------------------------------------------
    def _photo_key(self, person_id: str) -> str:
        """Caller holds _lock. A random key per person for this run: the person's ID never reaches the page."""
        key = self._photo_keys.get(person_id)
        if key is None:
            key = self._photo_keys[person_id] = secrets.token_urlsafe(PHOTO_KEY_BYTES)
        return key

    def _enrolled_voice_cards(self, cards: list[dict]) -> list[dict]:
        """A withdrawal from the CLI can erase an enrollment before the listener's cached card expires."""
        if not cards:
            return []
        with self._store_lock:
            db = self.db
            if self._closed or db is None:
                return []
            people = {card["person_id"] for card in cards}
            enrolled = {pid for pid in people if db.execute(
                "SELECT 1 FROM vl_voice WHERE person_id=?", (pid,)).fetchone() is not None}
        return [card for card in cards if card["person_id"] in enrolled]

    def _has_photo(self, person_id: str) -> bool:
        face = self._face
        if face is None:
            return False
        with self._store_lock:
            if self._closed:
                return False
            try:
                return bool(face.has_photo(person_id))
            except Exception as e:  # noqa: BLE001 - no photo shown is the safe fallback
                log.warning("photo lookup failed: %s", e)
                return False

    def voice_photo(self, key: str) -> bytes | None:
        """The JPEG of the person behind `key`, only while that person is on a current heard card, else None.
        FaceMatchModule.photo checks enrollment, the module and the photo consent itself. No list exists."""
        listener, face = self.voice, self._face
        if listener is None or face is None or self._closed or not isinstance(key, str):
            return None
        try:
            current = {c["person_id"] for c in self._enrolled_voice_cards(listener.cards(with_person=True))}
        except Exception as e:  # noqa: BLE001 - no current card known: nothing is served
            log.warning("heard cards unavailable: %s", e)
            return None
        with self._lock:
            pid = next((p for p, k in self._photo_keys.items() if secrets.compare_digest(k, key)), None)
        if pid is None or pid not in current:
            return None
        with self._store_lock:
            if self._closed:
                return None
            try:
                img = face.photo(pid)
            except Exception as e:  # noqa: BLE001 - a failed read shows no photo
                log.warning("photo read failed: %s", e)
                return None
        return bytes(img) if img else None

    def _stop_voice(self) -> None:
        listener, self.voice, self._face = self.voice, None, None
        if listener is not None:
            try:
                listener.stop()
            except Exception as e:  # noqa: BLE001 - stopping live mode must go on
                log.error("voice listener stop failed: %s", e)
            self._voice_status = _voice_unavailable(listener.mode, "live mode stopped")
        self._close_announcer()

    def voice_listen(self, seconds: float | None = None) -> bool:
        """Push-to-talk (the dashboard's Listen button). True when the microphone is now listening."""
        listener = self.voice
        if listener is None or self._closed:
            return False
        return listener.listen_for() if seconds is None else listener.listen_for(seconds)

    def _voice_snapshot(self) -> dict | None:
        listener = self.voice
        if listener is None:
            return dict(self._voice_status) if self._voice_status else None
        try:
            st = listener.status(with_person=True)
            cards = []
            for c in self._enrolled_voice_cards(st["cards"]):
                pid = c.pop("person_id")
                if self._has_photo(pid):
                    with self._lock:
                        c["photo"] = self._photo_key(pid)
                cards.append(c)
            st["cards"] = cards
        except Exception as e:  # noqa: BLE001 - the dashboard must keep rendering
            log.error("voice status failed: %s", e)
            return _voice_unavailable(listener.mode, f"voice status failed: {e}")
        gate = self._mic_gate
        if st["state"] == "unavailable" and not st["reason"] and gate is not None and gate.refused:
            st = {**st, "reason": MIC_REFUSAL_REASONS.get(gate.refused, "microphone refused")}
        return {**st, "reason": _safe_text(st["reason"])}

    # -- loop -------------------------------------------------------------------------------
    def _on_event(self, ev) -> None:
        """Every event is counted; a sighting is written when the (signature, level) pair changes,
        and an unchanged pair at most once per SIGHTING_REPEAT_S (the rest counted as suppressed)."""
        from .store.sightings import GpsFix
        d = ev.to_dict()
        arb = self.output
        if arb is not None and not self._closed:
            output_alert(arb, str(d.get("level")), str(d.get("signature")))
        pair = (d.get("signature"), d.get("level"))
        with self._store_lock:
            if self._closed:
                log.warning("live event after stop dropped (store closed)")
                return
            self._events += 1
            if self._log is None:
                return
            now = self._clock()
            if pair == self._last_pair and now - self._last_written < SIGHTING_REPEAT_S:
                self._suppressed += 1
                return
            self._log.add(d, GpsFix(None, None))   # no GPS fix in live Mac mode: never guessed
            self._sightings += 1
            self._last_pair, self._last_written = pair, now

    def _purge(self) -> None:
        """Caller holds _store_lock. With the sighting log, SightingLog.purge chains the purge boundary first
        (ADR-0066), so verification tells this purge from a deletion. A failure is logged and never stops live mode."""
        from .guard.retention import purge
        if self._closed or self.db is None:
            return
        self._last_purge = self._clock()
        try:
            if self._log is not None:
                n = self._log.purge(days=self.cfg.retention_days)
            else:
                n = purge(self.db, days=self.cfg.retention_days)
        except Exception as e:  # noqa: BLE001 - retention must not kill live mode
            log.error("retention purge failed: %s", e)
            return
        self._purged += n
        self._last_purge_at = time.time()
        log.info("retention purge: %d expired rows deleted (older than %d days)", n, self.cfg.retention_days)

    def _loop(self, background: bool = True) -> None:
        period = 1.0 / self.settings.hz
        try:
            while background and not self._stop.wait(period):
                self.step()
        finally:
            with self._store_lock:
                self._loop_done = True
                if self._close_deferred:       # stop() gave up waiting and left the DB to us
                    self._close_deferred = False
                    self._close_db()

    def step(self) -> dict | None:
        assert self.engine is not None
        self._power_step()
        self._capture_step()
        try:
            res = self.engine.step()
        except Exception as e:  # noqa: BLE001 - the loop never dies
            log.error("live step failed: %s", e)
            return None
        # Engine keeps every result/event; a long live run must not grow without bound.
        del self.engine.results[:-1]
        del self.engine.events[:-1]
        self._budget_step()
        with self._store_lock:
            if self._closed:                   # a step that finished after stop(): not counted
                return None
            if self._last_purge is not None and self._clock() - self._last_purge >= self.cfg.retention_check_s:
                self._purge()                  # scheduled retention: rigs run for days
            self._chain_tick()
            self._counts["steps"] += 1
            if res.get("level"):
                self._counts["level:" + str(res["level"])] += 1
            for sig in res.get("signatures", []):
                self._counts[str(sig)] += 1
            cs = self._step_cues(res)
            with self._lock:
                self._latest, self._latest_at, self._cues = res, self._clock(), cs
                self.steps += 1
                n = self.steps
            if self.sweep_id and n % CHECKPOINT_EVERY == 0:
                self._write_sweep("open")
        self._speak_cues(cs)
        return res

    def _chain_tick(self) -> None:
        """Caller holds _store_lock and has checked _closed. The record chain's 60 s idle anchor (ADR-0066)."""
        if self._log is None:
            return
        try:
            self._log.tick()
        except Exception as e:  # noqa: BLE001 - a late anchor is reported by verification; the loop goes on
            log.error("record chain anchor failed: %s", e)

    def _step_cues(self, res: dict) -> list[Cue]:
        """Direction cues for this step's signatures (oiris/spatial/cues.py); a failure means no cues, never a dead loop."""
        try:
            return spatial_cues(res)
        except Exception as e:  # noqa: BLE001 - cues are an added output; the loop never dies
            log.error("direction cues failed: %s", e)
            return []

    def _speak_cues(self, cs: list[Cue]) -> None:
        """Each cue's guarded phrase goes to the output arbiter at its event's level, only when output.speak_alerts is
        on (off by default on the Mac; the level pop-up from _on_event is unchanged either way). fyi is haptic only
        (M-06), so it gets no phrase. One key per (signature, direction, level): the dedupe window paces repeats."""
        arb = self.output
        if arb is None or self._closed or not arb.settings.speak_alerts:
            return
        for c in cs:
            pr = ALERT_LEVELS.get(c.level)
            if pr is not None and pr != FYI:
                arb.request("cue", pr, f"cue:{c.signature}:{c.direction}:{c.level}", text=c.text)

    def _located(self, latest: dict | None, at: float | None) -> list[dict]:
        """snapshot()["located"]: the latest step's spatial events as direction word, kinds, how many sensors agree
        and seconds ago (time since that step plus each event's age before the step's newest detection)."""
        events = [e for e in ((latest or {}).get("spatial") or []) if isinstance(e, dict)]
        if not events or at is None:
            return []
        ref = max(int(e.get("t_mono_ns") or 0) for e in events)
        since = max(0.0, self._clock() - at)
        return [{"direction": str(e.get("direction", "")), "kinds": [str(k) for k in e.get("kinds") or []],
                 "sensors": len(e.get("sensors") or []),
                 "seconds_ago": since + (ref - int(e.get("t_mono_ns") or 0)) / NS_PER_S} for e in events]

    def _write_sweep(self, outcome: str) -> None:
        """Caller holds _store_lock and has checked _closed."""
        timeline, sweep_id = self._timeline, self.sweep_id
        if timeline is None or sweep_id is None:   # no M-14 sweep open: nothing to record
            return
        try:
            counts = {**self._counts, "sightings": self._sightings, "events": self._events,
                      "sightings_suppressed": self._suppressed, "purged_rows": self._purged,
                      "adapters_unavailable": sorted(k for k, v in self.adapters.items() if not v["available"])}
            prev = [s for s in timeline.history(self.settings.site)
                    if s["sweep_id"] != sweep_id and s["outcome"] != "open"]
            seen_before = set(prev[-1]["counts"]) if prev else set()
            new = sorted(k for k in self._counts if k != "steps" and not k.startswith("level:") and k not in seen_before)
            timeline.finish(sweep_id, counts, new, outcome=outcome)
        except Exception as e:  # noqa: BLE001 - recording must not kill the loop
            log.error("sweep record failed: %s", e)

    # -- state for the dashboard ------------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            latest, steps, at, cs = self._latest, self.steps, self._latest_at, list(self._cues)
            over = [dict(o) for o in self._over]
        health = dict(latest.get("health", {})) if latest else {}
        if self.engine is not None and not latest:
            health = self.engine.bus.health()
        if over:   # a measurement for the developer, beside the sensors' health (never an alert)
            health["latency"] = {"health": Health.DEGRADED.value, "reason": _budget_reason(over)}
        for k, v in health.items():
            if isinstance(v, dict) and v.get("reason"):
                health[k] = {**v, "reason": _safe_text(str(v["reason"]))}
        snap = {"mode": "live", "running": self._thread is not None and self.outcome is None,
                "started": self.started, "steps": steps, "hz": self.settings.hz, "site": self.settings.site,
                "sweep_id": self.sweep_id, "adapters": {k: dict(v) for k, v in self.adapters.items()},
                "health": health, "result": latest,
                "records": {"events": self._events, "sightings": self._sightings,
                            "sightings_suppressed": self._suppressed, "purged_rows": self._purged,
                            "last_purge_at": self._last_purge_at,
                            "retention_check_s": float(self.cfg.retention_check_s)},
                "budgets": over}
        if self._log is not None:
            snap["chain"] = dict(self._log.chain_status)
        power = self._power_snapshot()
        if power is not None:
            snap["power"] = power
        voice = self._voice_snapshot()
        if voice is not None:
            snap["voice"] = voice
        output = snap["output"] = self._output_snapshot()
        cue = current_cue(cs)
        g = state_from_result(latest or {}, output.get("motion"), cue.direction if cue else None)
        snap["glasses"] = {"level": g.level, "alert_text": g.alert_text, "moving": g.moving, "cue_direction": g.cue_direction}
        snap["cues"] = [{"signature": c.signature, "direction": c.direction, "evidence": c.evidence} for c in cs]
        snap["located"] = self._located(latest, at)
        snap["timing"] = self.timing()
        if self.interlock is not None:
            snap["capture"] = {**self.interlock.state(), "profile": self._capture_profile}
        camera = self._camera_snapshot()
        if camera is not None:
            snap["camera"] = camera
        check_output({k: v for k, v in snap.items() if k != "result"})
        return snap

    def timing(self) -> dict:
        """{"engine": {...}, "voice": {...}, "output": {...}}: each stage's count, p50_ms, p95_ms, max_ms."""
        return {k: self.timers[k].summary() for k in TIMING_PARTS}

    # -- shutdown ---------------------------------------------------------------------------
    def _close_db(self) -> None:
        if self._store is None and self.db is not None:
            try:
                self.db.close()
            except Exception as e:  # noqa: BLE001
                log.warning("live store close failed: %s", e)

    def _close_chain(self) -> None:
        """Caller holds _store_lock with _closed set: no loop write can follow, so the anchor covers the whole tail."""
        if self._log is None or self.db is None:
            return
        try:
            self._log.close()
        except Exception as e:  # noqa: BLE001 - stopping must go on; the next open reports the unanchored tail
            log.error("record chain close failed: %s", e)

    def stop(self, outcome: str = "done") -> None:
        """Idempotent and bounded: at most two joins of join_s plus the bus stop. The store is
        sealed (no further loop writes) before the final sweep record; if the loop thread is
        still inside a step, the DB is left open for it and closed when the loop exits."""
        if self._thread is None or self.outcome is not None:
            return
        self.outcome = outcome                  # claim stop first: a second call returns at once
        self._stop.set()
        self._stop_voice()                      # before the store closes: no utterance reaches it after this
        self._close_capture()                   # after the microphone stopped: every grant revoked, the light off
        self._camera_step()                     # the face camera's device closes now (or as its read returns)
        self._close_output()
        self._thread.join(self.join_s)
        try:
            if self.engine is not None:         # set whenever start() succeeded and made the thread
                self.engine.bus.stop()          # stops and closes every adapter; blocked reads return
        finally:
            if self._thread.is_alive():
                self._thread.join(self.join_s)
            with self._store_lock:
                alive = not self._loop_done     # decided under the lock the loop's exit path takes
                self._closed = True
                if alive:
                    self.loop_alive_at_stop = True
                    self._counts["loop_alive_at_stop"] = 1
                    log.warning("live loop still running %.1fs after stop; store sealed, DB close deferred to it",
                                2 * self.join_s)
                if self.sweep_id:
                    self._write_sweep(outcome)
                self._close_chain()             # the session's tail anchored before the DB closes (ADR-0066)
                if alive:
                    self._close_deferred = True
                else:
                    self._close_db()
            if self._net is not None:
                self._net.__exit__(None, None, None)
                self._net = None

    def wait(self, duration: float = 0.0, install_signals: bool = False) -> str:
        """Block until SIGINT/SIGTERM (when install_signals) or `duration` seconds (0 = until
        stopped), then stop cleanly. Returns the sweep outcome."""
        why = {"outcome": "done"}
        old = {}
        if install_signals:
            def handler(signum, _frame):
                why["outcome"] = "interrupted"
                self._stop.set()
            for s in (signal.SIGINT, signal.SIGTERM):
                old[s] = signal.signal(s, handler)
        try:
            deadline = time.monotonic() + duration if duration > 0 else None
            while not self._stop.is_set():
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    break
                self._stop.wait(0.2 if left is None else min(0.2, left))
        except KeyboardInterrupt:
            log.info("live mode interrupted (KeyboardInterrupt); stopping")
            why["outcome"] = "interrupted"     # recorded as the sweep outcome and returned
        finally:
            for s, h in old.items():
                signal.signal(s, h)
            self.stop(why["outcome"])
        return why["outcome"]
