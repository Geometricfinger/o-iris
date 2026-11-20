"""Tamper-evident log: h_i = HMAC-SHA256(k_log, h_{i-1} || canonical(record_i)).
Head (index, hash) signed with Ed25519 and anchored every 100 records AND at least every
60 seconds while records are arriving (`tick()` is the timer hook for idle periods). Verification
detects edits, deletions, reordering and truncation past the last anchor.

A chain can resume over persisted history (ADR-0066): `base_idx`/`base_head` are the index and
head of the last stored record, so `append` continues at base_idx + 1 with prev = base_head while
`entries` holds only this session's records. `link`, `anchor_message` and `anchor_valid` are the
primitives the stored-log verifier (oiris.store.sightings.verify_log) uses."""
from __future__ import annotations
import hashlib, hmac, json, time
from typing import Callable
from dataclasses import dataclass, field
from nacl.signing import SigningKey, VerifyKey
from nacl.exceptions import BadSignatureError

from .ct import equal

GENESIS = b"\x00" * 32
ANCHOR_EVERY_RECORDS = 100     # ADR-0024
ANCHOR_EVERY_SECONDS = 60.0    # ADR-0024


def canonical(rec: dict) -> bytes:
    return json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def link(key: bytes, prev: bytes, rec: dict) -> bytes:
    """h_i = HMAC-SHA256(k_log, h_{i-1} || canonical(record_i))."""
    return hmac.new(key, prev + canonical(rec), hashlib.sha256).digest()


def anchor_message(idx: int, head: bytes) -> bytes:
    return idx.to_bytes(8, "big") + head


@dataclass
class Anchor:
    idx: int
    head: bytes
    sig: bytes


@dataclass
class HashChain:
    key: bytes
    signer: SigningKey
    entries: list = field(default_factory=list)   # (record, h_this)
    anchors: list = field(default_factory=list)
    anchor_every: int = ANCHOR_EVERY_RECORDS
    anchor_seconds: float = ANCHOR_EVERY_SECONDS
    clock: Callable[[], float] = time.monotonic
    base_idx: int = 0             # records stored before entries[0] (a resumed chain)
    base_head: bytes = GENESIS    # h of record base_idx (GENESIS for a new chain or segment)

    def __post_init__(self) -> None:
        self._last_anchor_t = self.clock()
        self._resumed_at = self.base_idx

    @property
    def length(self) -> int:
        """Index of the last record, stored history included."""
        return self.base_idx + len(self.entries)

    @property
    def head(self) -> bytes:
        return self.entries[-1][1] if self.entries else self.base_head

    @property
    def last_anchor_idx(self) -> int:
        return self.anchors[-1].idx if self.anchors else 0

    def resume(self, idx: int, head: bytes, last_anchor: Anchor | None = None) -> None:
        """Continue after stored record `idx` whose h is `head` (GENESIS starts a new segment).
        Resumed records are not this session's: `tick` and `has_unanchored` ignore them until
        this session appends, so nothing anchors over a tail it did not write."""
        self.base_idx, self.base_head = idx, head
        self.entries = []
        self.anchors = [last_anchor] if last_anchor is not None else []
        self._resumed_at = idx
        self._last_anchor_t = self.clock()

    def has_unanchored(self) -> bool:
        """True when this session appended records that no anchor covers yet."""
        return self.length > max(self.last_anchor_idx, self._resumed_at)

    def append(self, rec: dict) -> bytes:
        h = link(self.key, self.head, rec)
        self.entries.append((rec, h))
        # every 100th index (a resumed chain keeps the global index), so no more than 100
        # records ever follow the last anchor
        if self.length % self.anchor_every == 0:
            self.anchor()
        else:
            self.tick()
        return h

    def tick(self) -> Anchor | None:
        """60-second anchor timer. Call from the engine loop; anchors only if records were
        added since the last anchor, so an idle log does not grow."""
        if self.clock() - self._last_anchor_t < self.anchor_seconds:
            return None
        if not self.has_unanchored():
            self._last_anchor_t = self.clock()
            return None
        return self.anchor()

    def anchor(self) -> Anchor:
        idx, head = self.length, self.head
        a = Anchor(idx, head, self.signer.sign(anchor_message(idx, head)).signature)
        self.anchors.append(a)
        self._last_anchor_t = self.clock()
        return a


class ChainInvalid(ValueError):
    pass


def anchor_valid(a: Anchor, verify_key: VerifyKey) -> bool:
    try:
        verify_key.verify(anchor_message(a.idx, a.head), a.sig)
    except BadSignatureError:
        return False
    return True


def verify(entries: list, key: bytes, anchors: list, verify_key: VerifyKey) -> int:
    prev = GENESIS
    heads = {}
    for i, (rec, h) in enumerate(entries, 1):
        exp = link(key, prev, rec)
        if not equal(exp, h):
            raise ChainInvalid(f"entry {i} does not chain")
        heads[i] = h
        prev = h
    for a in anchors:
        if not anchor_valid(a, verify_key):
            raise ChainInvalid("anchor signature bad")
        if a.idx > len(entries):
            raise ChainInvalid(f"truncated past anchor {a.idx}")
        if a.idx and not equal(heads[a.idx], a.head):
            raise ChainInvalid(f"anchor {a.idx} head mismatch")
    return len(entries)
