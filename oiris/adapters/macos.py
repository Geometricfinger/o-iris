"""macOS sensor drivers. Wi-Fi comes only from the O Iris.app launcher's feed: a pipe inherited as
OIRIS_WIFI_FEED_FD carrying one JSON line every 15 s. The launcher (macos/launcher/main.swift) holds
the app's Location grant and reads CWInterface.cachedScanResults(), the networks macOS already saw in
its own scans. Neither the launcher nor this module starts a scan, associates or transmits; macOS's
Wi-Fi daemon log (macOS 26.5) confirms the launcher made no scan requests, only power and
supported-channel queries. There is no other source: `system_profiler SPAirPortDataType` makes macOS
start a live scan on every run (same log) and still had every BSSID withheld inside the app, so it is
not used, and `oiris run` started from a terminal has no Wi-Fi source on macOS.
Samples have the WifiBeaconAdapter payload shape plus a class: {"aps": [{"key", "tsf_us", "rssi", "local_s",
"class"}]}. "class" (camera_like, randomized or access_point, features/rf.ap_class) is worked out from the raw
BSSID in memory before it is hashed; only the class string leaves the adapter, never the vendor prefix.

What macOS withholds, reported rather than invented:
- beacon TSF: never exposed, so `tsf_us` is always None (AP clock skew is unavailable);
- BSSIDs unless the app has Location Services permission. An AP without a BSSID has no stable key,
  so it is dropped and counted in health; if every visible AP is withheld, read() raises
  DriverUnavailable with the reason."""
from __future__ import annotations
import json, logging, math, os, re, select, stat, sys, threading, time
from typing import TypeGuard

from ..guard import radio
from ..schema import Health
from .base import AdapterBase
from .drivers import DriverUnavailable

log = logging.getLogger(__name__)

_MAC = re.compile(r"^[0-9a-f]{2}(:[0-9a-f]{2}){5}$")
TSF_NOTE = "beacon TSF not exposed by macOS (tsf_us=None; AP clock skew unavailable)"
LOCATION_HINT = ("grant Location Services to the app running O Iris "
                 "(System Settings > Privacy & Security > Location Services)")
LOCATION_WAITING = ("waiting for Location permission (macOS hides Wi-Fi network IDs until then): answer the "
                    "macOS prompt, or allow O Iris in System Settings > Privacy & Security > Location Services")
NO_FEED = ("Wi-Fi on macOS comes from the O Iris app (its launcher reads macOS's cached scan under the app's "
           "Location permission); oiris run started from a terminal has no Wi-Fi source")


# ---- the O Iris.app launcher's Wi-Fi feed (macos/launcher/main.swift) ----

FEED_ENV = "OIRIS_WIFI_FEED_FD"
FEED_STATUSES = frozenset({"ok", "location_not_authorized", "no_interface", "wifi_off", "no_cached_results"})
FEED_INTERVAL_S = 15.0         # the launcher writes one line this often
FEED_MAX_AGE_S = 60.0          # an older line counts as no data
FEED_READING_AGE_S = 2 * FEED_INTERVAL_S + 5.0   # engine max age in feed mode: one missed line + slack
FEED_MAX_LINE = 1 << 20        # a longer line without a newline is dropped (and counted)


def _number(v) -> TypeGuard[int | float]:   # a finite JSON number (bool is not one)
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def parse_feed_line(line: bytes) -> dict:
    """One feed line -> {"t", "status", "aps": [{"bssid", "rssi"}], "withheld", "bad"}. Raises ValueError
    if the line as a whole is malformed; a malformed AP entry (bad MAC, wrong types) is dropped and
    counted in "bad". Raw BSSIDs stay inside the adapter."""
    try:
        doc = json.loads(line)
    except RecursionError as e:
        raise ValueError("nested too deeply") from e
    if not isinstance(doc, dict):
        raise ValueError("not a JSON object")
    t, status, aps, withheld = doc.get("t"), doc.get("status"), doc.get("aps"), doc.get("withheld", 0)
    if not _number(t) or status not in FEED_STATUSES or not isinstance(aps, list):
        raise ValueError("bad t, status or aps")
    if isinstance(withheld, bool) or not isinstance(withheld, int) or withheld < 0:
        raise ValueError("bad withheld")
    out, seen, bad = [], set(), 0
    for ap in aps:
        b = ap.get("bssid") if isinstance(ap, dict) else None
        r = ap.get("rssi") if isinstance(ap, dict) else None
        b = b.strip().lower() if isinstance(b, str) else ""
        if not _MAC.match(b) or b == "00:00:00:00:00:00" or not _number(r):
            bad += 1
        elif b not in seen:
            seen.add(b)
            out.append({"bssid": b, "rssi": float(r)})
    return {"t": float(t), "status": status, "aps": out, "withheld": withheld, "bad": bad}


class WifiFeed:
    """Read end of the launcher's pipe: non-blocking, with a partial-line buffer. poll() returns the
    latest valid complete line (parsed) or None. Malformed lines and entries are counted in
    `rejected`, never raised. On EOF (the launcher exited or stopped the feed) the fd is closed and
    `fd` becomes None. Raw BSSIDs only ever sit in this object's memory."""

    def __init__(self, fd: int):
        os.set_blocking(fd, False)
        os.set_inheritable(fd, False)   # not passed on to anything we start
        self.fd: int | None = fd
        self.rejected = 0
        self.error: str | None = None   # the last OS error on the pipe, shown in the adapter's health
        self._buf = b""
        self._skip = False              # inside an over-long line: drop up to its newline
        self._lock = threading.Lock()   # a hung read may overlap a restarted adapter's read

    def _note_error(self, what: str, e: Exception) -> None:
        self.error = f"{what}: {e}"
        log.warning("wifi_macos feed %s", self.error)

    def _drain(self) -> bytes:
        chunks = []
        while self.fd is not None:
            try:
                b = os.read(self.fd, 65536)
            except (BlockingIOError, InterruptedError):
                break
            except OSError as e:        # a broken pipe end: recorded, then handled as EOF
                self._note_error("read failed", e)
                b = b""
            if not b:
                self.close()
                break
            chunks.append(b)
        return b"".join(chunks)

    def poll(self, wait_s: float) -> dict | None:
        with self._lock:
            if self.fd is None:
                return None
            try:
                select.select([self.fd], [], [], max(0.0, wait_s))
            except (OSError, ValueError) as e:   # recorded; the read below then reports a broken fd as EOF
                self._note_error("select failed", e)
            *lines, self._buf = (self._buf + self._drain()).split(b"\n")
            if self._skip and lines:
                lines, self._skip = lines[1:], False
            if len(self._buf) > FEED_MAX_LINE or (self.fd is None and self._buf.strip()):   # too long, or cut off by EOF
                self.rejected += 1
                self._buf, self._skip = b"", self.fd is not None
            latest = None
            for ln in lines:
                if not ln.strip():
                    continue
                try:
                    latest = parse_feed_line(ln)
                except ValueError:      # includes JSONDecodeError and UnicodeDecodeError
                    self.rejected += 1
                    continue
                self.rejected += latest["bad"]
            return latest

    def close(self) -> None:
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError as e:        # the fd is unusable either way; recorded, not raised
                self._note_error("close failed", e)
            self.fd = None
        self._buf = b""


def _open_feed(raw: str | None) -> WifiFeed | None:
    try:
        fd = int(raw or "")
        if fd < 3 or not stat.S_ISFIFO(os.fstat(fd).st_mode):   # never stdio, only a pipe
            return None
        return WifiFeed(fd)
    except (ValueError, OSError):
        return None


_feed_lock = threading.Lock()
_feed: WifiFeed | None = None
_feed_claimed = False
_feed_ended: str | None = None   # why the feed ended, once it has (shown again when the bus reopens)


def feed_ended_reason(error: str | None) -> str:
    why = f" ({error})" if error else ""
    return f"the app launcher's Wi-Fi feed ended{why}; Wi-Fi stays unavailable until O Iris is restarted"


def wifi_feed() -> WifiFeed | None:
    """The launcher's feed, claimed from OIRIS_WIFI_FEED_FD once per process. The variable is then
    removed, so nothing we start sees it and a later reuse of the fd number is never taken for the
    feed; the feed outlives adapter close/reopen (the bus reopens after a failed read). None without
    a valid feed, or after its EOF (then `_feed_ended` holds the reason)."""
    global _feed, _feed_claimed, _feed_ended
    with _feed_lock:
        if not _feed_claimed:
            _feed_claimed = True
            _feed = _open_feed(os.environ.pop(FEED_ENV, None))
        if _feed is not None and _feed.fd is None:
            _feed_ended = feed_ended_reason(_feed.error)
            _feed = None
        return _feed


class WifiMacosAdapter(AdapterBase):
    """Wi-Fi APs on macOS 14+ from the O Iris.app launcher's feed (one line every 15 s with macOS's
    cached scan, read under the app's Location grant). A read waits for the next line for part of the
    bus timeout and returns None without one. Without a feed, open() fails with the reason; when the
    feed ends, the adapter fails and stays failed (no other source, never a subprocess). The feed
    stays open across close/reopen: it belongs to the process (see wifi_feed).
    cfg: site_key. max_age_s (how long the engine treats a reading as current) is FEED_READING_AGE_S."""
    id, bus, caps, units = "wifi_macos", "wifi", {"ap_scan"}, "dBm"
    WAIT_SHARE = 0.8   # of the bus read timeout

    def open(self, cfg: dict) -> None:
        super().open(cfg)
        # macOS has no `iw`, so the monitor-mode query finds nothing here; O Iris never enables
        # monitor mode itself. Kept so the shared radio guard runs on every Wi-Fi adapter.
        radio.assert_passive()
        self._key = cfg["site_key"]
        self._feed = wifi_feed()
        if self._feed is None:
            raise self._fail(_feed_ended or NO_FEED)
        self._feed_last = time.monotonic()
        self.max_age_s = FEED_READING_AGE_S
        self.set_health(Health.DEGRADED, "waiting for the app launcher's Wi-Fi feed (one reading every 15 s)")

    def _fail(self, reason: str) -> DriverUnavailable:
        self.set_health(Health.FAILED, reason)
        return DriverUnavailable(reason)

    def _emit(self, aps: list[dict], withheld: int, empty: str, note: str = ""):
        """Health and sample for one scan: [{"bssid", "rssi"}] plus the count without a BSSID. Each AP
        also gets its coarse "class" (rf.ap_class), computed here from the raw BSSID; only the class
        string leaves the adapter, never the BSSID or its OUI."""
        from ..features.rf import ap_class
        from ..guard.privacy import hash_identifier
        n_vis = len(aps) + withheld
        if withheld and not aps:
            raise self._fail(f"macOS withheld BSSIDs for all {n_vis} visible networks: {LOCATION_HINT}{note}")
        if withheld:
            self.set_health(Health.DEGRADED, f"macOS withheld BSSIDs for {withheld} of {n_vis} networks "
                                             f"(dropped; {LOCATION_HINT}); {TSF_NOTE}{note}")
        elif not n_vis:
            self.set_health(Health.DEGRADED, f"{empty}; {TSF_NOTE}{note}")
        else:
            self.set_health(Health.OK, TSF_NOTE + note)
        t = time.time()
        return self._sample({"aps": [{"key": hash_identifier(a["bssid"], self._key), "tsf_us": None,
                                      "rssi": a["rssi"], "local_s": t, "class": ap_class(a["bssid"])}
                                     for a in aps]})

    def read(self, timeout_s: float):
        feed = self._feed
        if feed is None:
            raise self._fail(_feed_ended or NO_FEED)
        rep = feed.poll(max(0.05, self.WAIT_SHARE * timeout_s))
        if feed.fd is None:   # EOF or a broken pipe: the launcher is gone or stopped the feed
            self._feed = None
            wifi_feed()       # records the reason for later opens
            raise self._fail(feed_ended_reason(feed.error))
        if rep is None:
            idle = time.monotonic() - self._feed_last
            if idle > FEED_MAX_AGE_S:
                self.set_health(Health.DEGRADED, f"no reading from the app launcher's Wi-Fi feed for {idle:.0f}s")
            return None
        if abs(time.time() - rep["t"]) > FEED_MAX_AGE_S:
            self.set_health(Health.DEGRADED, f"Wi-Fi feed reading older than {FEED_MAX_AGE_S:.0f}s ignored")
            return None
        self._feed_last = time.monotonic()
        note = f"; {feed.rejected} malformed Wi-Fi feed lines/entries dropped" if feed.rejected else ""
        status = rep["status"]
        if status == "location_not_authorized":
            # Like Bluetooth's undecided permission: wait without a sample and without failing. A
            # failure here would count toward the bus's circuit breaker, which then held Wi-Fi off for
            # its 30 s cooldown after the user allowed Location (seen live, 2026-11-15). Waiting lets
            # the first OK line after the grant count at once; a denial reads the same, and the reason
            # says where to change it.
            self.set_health(Health.DEGRADED, LOCATION_WAITING + note)
            return None
        if status == "wifi_off":
            raise self._fail("Wi-Fi is turned off" + note)
        if status == "no_interface":
            raise self._fail("no Wi-Fi interface reported" + note)
        if status == "no_cached_results":
            return self._emit([], 0, "macOS has no cached Wi-Fi scan yet", note)
        return self._emit(rep["aps"], rep["withheld"], "no networks in the OS scan", note)


# ---- Bluetooth permission (CoreBluetooth via PyObjC, installed with bleak on macOS) ----

BT_NOT_DETERMINED, BT_RESTRICTED, BT_DENIED, BT_ALLOWED = 0, 1, 2, 3   # CBManagerAuthorization*
BT_SETTINGS = "System Settings > Privacy & Security > Bluetooth"
BT_WAITING = f"waiting for Bluetooth permission: answer the macOS prompt or allow O Iris in {BT_SETTINGS}"
BT_DENIED_REASON = f"Bluetooth permission denied for O Iris: allow it in {BT_SETTINGS}"
BT_RESTRICTED_REASON = "Bluetooth permission restricted by device policy"


def ble_authorization() -> int | None:
    """CBManager.authorization() on macOS: instant, no scan, no prompt. None when not on macOS or
    PyObjC/CoreBluetooth is unavailable (callers then keep their usual behaviour)."""
    if sys.platform != "darwin":
        return None
    try:
        from CoreBluetooth import CBManager  # type: ignore
        return int(CBManager.authorization())
    except Exception:  # noqa: BLE001 - ImportError, missing selector on old macOS
        return None


def ble_request_permission():
    """macOS shows its Bluetooth prompt only when the app uses Bluetooth: creating a central
    manager is enough and returns at once (no scan). Keep the result alive until decided."""
    try:
        from CoreBluetooth import CBCentralManager  # type: ignore
        return CBCentralManager.alloc().initWithDelegate_queue_options_(None, None, None)
    except Exception:  # noqa: BLE001
        return None


class BlePermission:
    """Gate before each BLE scan. check() -> None: scan (allowed, or state unknown);
    str: do not scan, still waiting for the user. Raises DriverUnavailable if denied/restricted.
    The prompt is requested once per gate (i.e. once per adapter open); after that it only polls."""

    def __init__(self):
        self._requested, self._manager = False, None

    def check(self) -> str | None:
        st = ble_authorization()
        if st == BT_NOT_DETERMINED:
            if not self._requested:
                self._requested, self._manager = True, ble_request_permission()
            return BT_WAITING
        self._manager = None
        if st == BT_DENIED:
            raise DriverUnavailable(BT_DENIED_REASON)
        if st == BT_RESTRICTED:
            raise DriverUnavailable(BT_RESTRICTED_REASON)
        return None


MACOS = [WifiMacosAdapter]
