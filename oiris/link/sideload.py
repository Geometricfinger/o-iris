"""Model side-load from the phone (ADR-0062): the glasses or pod may be offline by design, so a model the owner has
pinned arrives in chunks over the paired link and is installed only when its SHA-256 matches the pin.

Pins: `<data_dir>/models.pins.json`, edited by the owner on the device itself: [{"name","slot","sha256","size"}].
load_pins refuses the whole file on any bad entry. No op adds or changes a pin: the phone can only offer what the
owner already pinned, byte for byte.

  model_offer  {"op","name","slot","sha256","size","chunk_size"}
               -> {"ok":true,"chunks":n,"missing":[[start,stop],...],"missing_count":k}
  model_chunk  {"op","sha256","index","data"}  -> {"ok":true,"missing_count":k}
  model_commit {"op","sha256"}                 -> {"ok":true,"installed":"<slot>/<name>"}
  model_status {"op"}                          -> {"ok":true,"models":[{"name","slot","sha256","size"}]}

`missing` lists half-open index ranges, at most MAX_MISSING_RANGES of them (missing_count is the total), so a reply
stays small however many chunks there are; the phone sends those and offers again for the rest. An offer of a
transfer already under way resumes it: the partial file `<models_dir>/.partial/<sha256>` and its record
`<sha256>.json` (received ranges, written every PERSIST_EVERY chunks after an fsync of the data) survive a dropped
link or a pod restart, so at most PERSIST_EVERY - 1 chunks are sent twice. Chunks are idempotent.

Paths: names and slots match strict patterns (no separators, no leading dot), every file here is opened with
O_NOFOLLOW and every directory created here must be a real directory, so a symlink planted in the models tree is
refused rather than followed. Free space is checked before an offer is accepted. The ops are transport-independent:
BLE is slow (ADR-0062), a faster link can carry the same requests later."""
from __future__ import annotations
import errno, hashlib, hmac, json, logging, os, re, shutil, stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .ops import b64url, bad_reply, dumps, from_b64url, load_reply, refusal

log = logging.getLogger(__name__)

PINS_FILE = "models.pins.json"
MAX_MODEL_BYTES = 256 * 1024 * 1024
CHUNK_MIN, CHUNK_MAX = 512, 65536
MAX_MISSING_RANGES = 64
PERSIST_EVERY = 16                  # chunks between fsync + progress record
FREE_MARGIN_BYTES = 32 * 1024 * 1024   # left free on the data volume after a model is accepted
HASH_BLOCK = 1024 * 1024
PARTIAL_DIR = ".partial"
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
SLOT_RE = re.compile(r"[a-z][a-z0-9_]{0,31}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
PIN_KEYS = ("name", "sha256", "size", "slot")

NOT_PINNED = "not pinned"
PINS_INVALID = "pins invalid"
BAD_OFFER = "bad offer"
TOO_LARGE = "model too large"
BAD_CHUNK_SIZE = "bad chunk_size"
NO_SPACE = "not enough free space"
UNSAFE_PATH = "unsafe path"
NO_OFFER = "no offer"
CHUNK_RANGE = "chunk out of range"
CHUNK_LENGTH = "bad chunk length"
INCOMPLETE = "incomplete"
HASH_MISMATCH = "hash mismatch"
INSTALL_FAILED = "install failed"


class PinError(ValueError):
    pass


@dataclass(frozen=True)
class Pin:
    name: str
    slot: str
    sha256: str
    size: int

    def as_dict(self) -> dict:
        return {"name": self.name, "slot": self.slot, "sha256": self.sha256, "size": self.size}


def valid_pin_fields(name: Any, slot: Any, sha256: Any, size: Any) -> str | None:
    """Why these fields cannot be a pin, or None."""
    if not (isinstance(name, str) and NAME_RE.fullmatch(name) and isinstance(slot, str) and SLOT_RE.fullmatch(slot)):
        return "bad name or slot"
    if not (isinstance(sha256, str) and SHA256_RE.fullmatch(sha256)):
        return "sha256 must be 64 lowercase hex characters"
    if type(size) is not int or size < 1:
        return "size must be a positive integer"
    if size > MAX_MODEL_BYTES:
        return TOO_LARGE
    return None


def load_pins(path: str | Path) -> list[Pin]:
    """The owner's pins. A missing file is no pins; any bad entry refuses the whole file (PinError)."""
    p = Path(path)
    try:
        fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return []
    except OSError as e:
        raise PinError(f"pins file unreadable: {e.strerror}") from None
    with os.fdopen(fd, "rb") as f:
        try:
            raw = json.loads(f.read())
        except ValueError:
            raise PinError("pins file is not JSON") from None
    if not isinstance(raw, list):
        raise PinError("pins file must be a list")
    pins: list[Pin] = []
    seen: set[tuple[str, str]] = set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict) or tuple(sorted(entry)) != PIN_KEYS:
            raise PinError(f"pin {i}: keys must be exactly {', '.join(PIN_KEYS)}")
        why = valid_pin_fields(entry["name"], entry["slot"], entry["sha256"], entry["size"])
        if why:
            raise PinError(f"pin {i}: {why}")
        if (entry["slot"], entry["name"]) in seen:
            raise PinError(f"pin {i}: duplicate {entry['slot']}/{entry['name']}")
        seen.add((entry["slot"], entry["name"]))
        pins.append(Pin(entry["name"], entry["slot"], entry["sha256"], entry["size"]))
    return pins


def chunk_count(size: int, chunk_size: int) -> int:
    return -(-size // chunk_size)


def _merge(ranges: list[list[int]]) -> list[list[int]]:
    """Half-open ranges as sorted, disjoint, non-adjacent ones."""
    out: list[list[int]] = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _add(ranges: list[list[int]], i: int) -> list[list[int]]:
    return _merge([*ranges, [i, i + 1]])


def _has(ranges: list[list[int]], i: int) -> bool:
    return any(a <= i < b for a, b in ranges)


def _missing(ranges: list[list[int]], n: int) -> list[list[int]]:
    out, at = [], 0
    for a, b in ranges:
        if a > at:
            out.append([at, a])
        at = b
    if at < n:
        out.append([at, n])
    return out


@dataclass
class _Transfer:
    pin: Pin
    chunk_size: int
    chunks: int
    have: list[list[int]]
    unsaved: int = 0

    def missing_count(self) -> int:
        return self.chunks - sum(b - a for a, b in self.have)


class Sideload:
    """Pod side of the side-load ops. `free_bytes(dir)`: free space on the volume (injectable for tests)."""

    def __init__(self, pins_path: str | Path, models_dir: str | Path,
                 free_bytes: Callable[[Path], int] = lambda d: shutil.disk_usage(d).free):
        self.pins_path, self.models_dir, self.free_bytes = Path(pins_path), Path(models_dir), free_bytes
        self.transfers: dict[str, _Transfer] = {}

    # ---- paths ----
    @staticmethod
    def _real_dir(d: Path) -> Path:
        """d as a real directory (created 0700 if absent); OSError(ELOOP) when it is a symlink or not a directory."""
        try:
            os.mkdir(d, 0o700)
        except FileExistsError:   # python-brain: allow SILENT_EXCEPTION - an existing directory is checked just below
            pass
        st = os.lstat(d)
        if not stat.S_ISDIR(st.st_mode):
            raise OSError(errno.ELOOP, "not a real directory", str(d))
        return d

    def _partial(self, sha256: str) -> tuple[Path, Path]:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        d = self._real_dir(self.models_dir / PARTIAL_DIR)
        return d / sha256, d / f"{sha256}.json"

    @staticmethod
    def _open(p: Path, flags: int) -> int:
        return os.open(p, flags | os.O_NOFOLLOW, 0o600)

    def _save(self, t: _Transfer) -> None:
        part, rec = self._partial(t.pin.sha256)
        fd = self._open(part, os.O_RDWR)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        tmp = rec.with_name(rec.name + ".tmp")
        fd = self._open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps({"v": 1, "size": t.pin.size, "chunk_size": t.chunk_size, "have": t.have}))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, rec)
        t.unsaved = 0

    def _resume(self, pin: Pin, chunk_size: int, n: int) -> list[list[int]]:
        """Received ranges from the progress record, or [] (and a truncated partial file) when it does not fit."""
        part, rec = self._partial(pin.sha256)
        try:
            fd = self._open(rec, os.O_RDONLY)
            with os.fdopen(fd, "rb") as f:
                d = json.loads(f.read())
            have = d["have"]
            ok = (d.get("size") == pin.size and d.get("chunk_size") == chunk_size and isinstance(have, list)
                  and all(isinstance(r, list) and len(r) == 2 and all(type(x) is int for x in r) and 0 <= r[0] < r[1] <= n
                          for r in have))
            if ok and os.lstat(part).st_size <= pin.size:
                return _merge(have)
        except FileNotFoundError:   # python-brain: allow SILENT_EXCEPTION - no record means a fresh transfer
            pass
        except (ValueError, KeyError, TypeError):   # python-brain: allow SILENT_EXCEPTION - a damaged record restarts the transfer
            pass
        fd = self._open(part, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        os.close(fd)
        return []

    def _drop(self, sha256: str) -> None:
        self.transfers.pop(sha256, None)
        part, rec = self._partial(sha256)
        for p in (part, rec):
            try:
                os.unlink(p)
            except FileNotFoundError:   # python-brain: allow SILENT_EXCEPTION - already gone is the goal
                pass

    # ---- ops ----
    def handle(self, op: str, req: dict) -> dict:
        try:
            return getattr(self, "_" + op)(req)
        except PinError as e:
            log.warning("side-load: %s", e)
            return {"ok": False, "error": PINS_INVALID}
        except OSError as e:
            log.warning("side-load %s refused: %s", op, e)
            return {"ok": False, "error": UNSAFE_PATH if e.errno == errno.ELOOP else INSTALL_FAILED}

    def _model_offer(self, r: dict[str, Any]) -> dict:
        name: Any = r.get("name")
        slot: Any = r.get("slot")
        sha256: Any = r.get("sha256")
        size: Any = r.get("size")
        chunk_size: Any = r.get("chunk_size")
        why = valid_pin_fields(name, slot, sha256, size)
        if why == TOO_LARGE:
            return {"ok": False, "error": TOO_LARGE}
        if why:
            return {"ok": False, "error": BAD_OFFER}
        pin = Pin(name, slot, sha256, size)
        if pin not in load_pins(self.pins_path):
            return {"ok": False, "error": NOT_PINNED}
        if type(chunk_size) is not int or not CHUNK_MIN <= chunk_size <= CHUNK_MAX:
            return {"ok": False, "error": BAD_CHUNK_SIZE}
        n = chunk_count(size, chunk_size)
        t = self.transfers.get(sha256)
        if t is None or t.pin != pin or t.chunk_size != chunk_size:
            part, _ = self._partial(sha256)
            have_bytes = os.lstat(part).st_size if os.path.lexists(part) else 0
            if self.free_bytes(part.parent) < size - min(have_bytes, size) + FREE_MARGIN_BYTES:
                return {"ok": False, "error": NO_SPACE}
            t = _Transfer(pin, chunk_size, n, self._resume(pin, chunk_size, n))
            self._save(t)
            self.transfers[sha256] = t
        missing = _missing(t.have, n)
        return {"ok": True, "chunks": n, "missing": missing[:MAX_MISSING_RANGES], "missing_count": t.missing_count()}

    def _model_chunk(self, r: dict) -> dict:
        sha256, index = r.get("sha256"), r.get("index")
        t = self.transfers.get(sha256) if isinstance(sha256, str) else None
        if t is None:
            return {"ok": False, "error": NO_OFFER}
        if type(index) is not int or not 0 <= index < t.chunks:
            return {"ok": False, "error": CHUNK_RANGE}
        expect = t.chunk_size if index < t.chunks - 1 else t.pin.size - index * t.chunk_size
        data = from_b64url(r.get("data"), max_bytes=t.chunk_size)
        if data is None or len(data) != expect:
            return {"ok": False, "error": CHUNK_LENGTH}
        if not _has(t.have, index):
            part, _ = self._partial(t.pin.sha256)
            fd = self._open(part, os.O_WRONLY)
            try:
                os.pwrite(fd, bytes(data), index * t.chunk_size)
            finally:
                os.close(fd)
            t.have = _add(t.have, index)
            t.unsaved += 1
            if t.unsaved >= PERSIST_EVERY:
                self._save(t)
        return {"ok": True, "missing_count": t.missing_count()}

    def _model_commit(self, r: dict) -> dict:
        sha256 = r.get("sha256")
        t = self.transfers.get(sha256) if isinstance(sha256, str) else None
        if t is None:
            return {"ok": False, "error": NO_OFFER}
        if t.missing_count():
            return {"ok": False, "error": INCOMPLETE}
        if t.pin not in load_pins(self.pins_path):     # the owner may have removed the pin meanwhile
            self._drop(t.pin.sha256)
            return {"ok": False, "error": NOT_PINNED}
        part, rec = self._partial(t.pin.sha256)
        fd = self._open(part, os.O_RDONLY)
        try:
            h = hashlib.sha256()
            size = 0
            while block := os.read(fd, HASH_BLOCK):
                h.update(block)
                size += len(block)
            good = size == t.pin.size and hmac.compare_digest(h.hexdigest(), t.pin.sha256)
            if good:
                os.fsync(fd)
        finally:
            os.close(fd)
        if not good:
            self._drop(t.pin.sha256)
            log.warning("side-load: %s/%s failed its hash check; partial file removed", t.pin.slot, t.pin.name)
            return {"ok": False, "error": HASH_MISMATCH}
        slot_dir = self._real_dir(self.models_dir / t.pin.slot)
        os.replace(part, slot_dir / t.pin.name)
        dfd = os.open(slot_dir, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
        self.transfers.pop(t.pin.sha256, None)
        try:
            os.unlink(rec)
        except FileNotFoundError:   # python-brain: allow SILENT_EXCEPTION - the record may not have been written yet
            pass
        log.info("side-load: installed %s/%s", t.pin.slot, t.pin.name)
        return {"ok": True, "installed": f"{t.pin.slot}/{t.pin.name}"}

    def installed(self) -> list[Pin]:
        """Pinned models present as regular files (never through a symlink) of the pinned size. The hash was
        checked at commit; this does not re-hash (a 256 MiB read per status request)."""
        out = []
        for pin in load_pins(self.pins_path):
            try:
                d = os.lstat(self.models_dir / pin.slot)
                st = os.lstat(self.models_dir / pin.slot / pin.name)
            except OSError:   # python-brain: allow SILENT_EXCEPTION - absent means not installed
                continue
            if stat.S_ISDIR(d.st_mode) and stat.S_ISREG(st.st_mode) and st.st_size == pin.size:
                out.append(pin)
        return out

    def _model_status(self, r: dict) -> dict:
        return {"ok": True, "models": [p.as_dict() for p in self.installed()]}


# ---- request builders and reply parsers (LinkOps.kt mirrors them) --------------------------------------------
def offer_request(name: str, slot: str, sha256: str, size: int, chunk_size: int) -> str:
    return dumps({"op": "model_offer", "name": name, "slot": slot, "sha256": sha256, "size": size,
                  "chunk_size": chunk_size})


def chunk_request(sha256: str, index: int, data: bytes) -> str:
    return dumps({"op": "model_chunk", "sha256": sha256, "index": index, "data": b64url(data)})


def commit_request(sha256: str) -> str:
    return dumps({"op": "model_commit", "sha256": sha256})


def status_request() -> str:
    return dumps({"op": "model_status"})


def _count(v: Any) -> bool:
    return type(v) is int and v >= 0


def parse_offer(reply: str | bytes) -> dict:
    r = load_reply(reply)
    if r is None:
        return bad_reply()
    if not r["ok"]:
        return refusal(r)
    n, missing, k = r.get("chunks"), r.get("missing"), r.get("missing_count")
    if not (_count(n) and _count(k) and isinstance(missing, list) and len(missing) <= MAX_MISSING_RANGES):
        return bad_reply()
    for m in missing:
        if not (isinstance(m, list) and len(m) == 2 and all(_count(x) for x in m) and m[0] < m[1] <= n):
            return bad_reply()
    return {"ok": True, "error": None, "chunks": n, "missing": missing, "missing_count": k}


def parse_chunk(reply: str | bytes) -> dict:
    r = load_reply(reply)
    if r is None:
        return bad_reply()
    if not r["ok"]:
        return refusal(r)
    return {"ok": True, "error": None, "missing_count": r["missing_count"]} if _count(r.get("missing_count")) else bad_reply()


def parse_commit(reply: str | bytes) -> dict:
    r = load_reply(reply)
    if r is None:
        return bad_reply()
    if not r["ok"]:
        return refusal(r)
    return {"ok": True, "error": None, "installed": r["installed"]} if isinstance(r.get("installed"), str) else bad_reply()


def parse_status(reply: str | bytes) -> dict:
    r = load_reply(reply)
    if r is None:
        return bad_reply()
    if not r["ok"]:
        return refusal(r)
    models = r.get("models")
    if not isinstance(models, list):
        return bad_reply()
    for m in models:
        if not (isinstance(m, dict) and tuple(sorted(m)) == PIN_KEYS and valid_pin_fields(m["name"], m["slot"], m["sha256"], m["size"]) is None):
            return bad_reply()
    return {"ok": True, "error": None, "models": [{k: m[k] for k in PIN_KEYS} for m in models]}
