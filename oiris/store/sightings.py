"""M-05 sighting log (encrypted, hash-chained, auto-GPS, export rounding) and
M-14 sweep history timeline.

Resumable chain (ADR-0066). One HMAC chain runs across sessions over the stored history:
- records are the `sighting` rows (seq = chain_idx, h_prev, h_this) plus side records kept in
  `audit_log` under CHAIN_KINDS (a verdict change, a purge boundary, a segment restart);
- opening a log verifies the stored chain (`verify_log`). If its current segment verifies, the
  next record continues it (seq + 1, prev = last h). If not, a new segment starts at GENESIS with
  a `chain_restart` record (reason code and counts only) and `chain_status` says "restarted";
  past rows are never rewritten or re-signed;
- an Ed25519 anchor (idx, head, sig) goes into `chain_anchor` every ANCHOR_EVERY_RECORDS records
  or ANCHOR_EVERY_SECONDS with new records, and at `close()`. Records stored after the last anchor
  are the unanchored tail, reported at open and never anchored over at open;
- retention: `purge()` (or `record_purge_boundary` before any other deletion) chains a purge
  boundary first, so a purged range verifies while any other missing row is "deleted".
Wiring still owed by the live runner (oiris/live.py): call `close()` at stop, and purge through
`SightingLog.purge` (or call `record_purge_boundary(cutoff)` with the same cutoff) instead of
calling guard.retention.purge directly."""
from __future__ import annotations
import bisect, json, time, uuid
from dataclasses import dataclass

from nacl.signing import VerifyKey

from ..guard.privacy import round_gps
from ..security.ct import equal
from ..security.hashchain import GENESIS, Anchor, HashChain, anchor_valid, link
from .db import DB

VERDICTS = {"confirmed", "benign", "unresolved"}
OSM_ATTRIBUTION = "© OpenStreetMap contributors"

KIND_RESTART, KIND_VERDICT, KIND_PURGE = "chain_restart", "chain_verdict", "chain_purge"
CHAIN_KINDS = (KIND_RESTART, KIND_VERDICT, KIND_PURGE)
_SIDE_MARK = {KIND_RESTART: "chain_restart", KIND_VERDICT: "verdict_change", KIND_PURGE: "chain_purge"}
_LINK_FIELDS = ("seq", "prev", "h")      # stored beside a side record's chained fields
SECONDS_PER_DAY = 86400
MAX_ISSUES_REPORTED = 20
# Reason codes, in the order a reader should worry about them.
OK, EDITED, DELETED, REORDERED, TRUNCATED, BAD_ANCHOR = "ok", "edited", "deleted", "reordered", "truncated", "bad_anchor"
LEGACY = "legacy"   # restart reason only: rows from before ADR-0066 (no anchors, no side records)


@dataclass
class _Rec:
    seq: int
    prev: bytes
    h: bytes
    rec: dict | None          # the chained fields; None when the stored record cannot be rebuilt
    kind: str                 # "sighting" or one of CHAIN_KINDS
    sound: bool = True        # rebuilt and HMAC-consistent


def _side_records(db: DB) -> tuple[list[_Rec], int]:
    out, unreadable = [], 0
    marks = ",".join("?" * len(CHAIN_KINDS))
    for kind, detail in db.execute(f"SELECT kind, detail FROM audit_log WHERE kind IN ({marks}) ORDER BY id",
                                   CHAIN_KINDS):
        try:
            d = json.loads(detail)
            rec = {k: v for k, v in d.items() if k not in _LINK_FIELDS}
            if _SIDE_MARK[kind] not in rec:
                raise ValueError("side record kind does not match its fields")
            out.append(_Rec(int(d["seq"]), bytes.fromhex(d["prev"]), bytes.fromhex(d["h"]), rec, kind))
        except (ValueError, KeyError, TypeError):   # python-brain: allow SILENT_EXCEPTION - counted as an edit
            unreadable += 1
    return out, unreadable


def _sighting_records(db: DB, side: list[_Rec]) -> tuple[list[_Rec], list[int]]:
    """Rebuild each row's chained record. The chained verdict is the one at creation: the first
    later verdict change's `verdict_from`, else the column. Returns rows and seqs whose verdict
    column disagrees with the last chained verdict change."""
    changes: dict[str, list[dict]] = {}
    for s in sorted(side, key=lambda r: r.seq):
        if s.kind == KIND_VERDICT and s.rec is not None:
            changes.setdefault(str(s.rec.get("sighting_id")), []).append({"seq": s.seq, **s.rec})
    out, verdict_bad = [], []
    for sid, t_wall, codes, body, verdict, seq, prev, h in db.execute(
            "SELECT sighting_id,t_wall,codes,body,verdict,chain_idx,h_prev,h_this FROM sighting"):
        if seq is None:
            continue
        rec: dict | None
        try:
            later = [c for c in changes.get(sid, []) if c["seq"] > seq]
            v0 = later[0]["verdict_from"] if later else verdict
            if later and later[-1]["verdict_change"] != verdict:
                verdict_bad.append(seq)
            digest = json.dumps(db.open_json(body, sid.encode()), sort_keys=True)
            rec = {"sighting_id": sid, "t_wall": t_wall, "codes": json.loads(codes), "verdict": v0,
                   "body_digest": digest}
        except Exception:  # noqa: BLE001  # python-brain: allow SILENT_EXCEPTION - unreadable row is reported as edited
            rec = None
        out.append(_Rec(int(seq), bytes(prev or b""), bytes(h or b""), rec, "sighting"))
    return out, verdict_bad


def _anchors(db: DB) -> list[Anchor]:
    return [Anchor(int(i), bytes(hd), bytes(sg)) for i, hd, sg in
            db.execute("SELECT idx, head, sig FROM chain_anchor ORDER BY idx")]


def _purge_ranges(recs: list[_Rec]) -> tuple[list[tuple[int, int]], dict[int, bytes]]:
    """Merged [first, last] seq ranges from sound purge records, and last seq -> head."""
    spans, heads = [], {}
    for r in recs:
        if r.kind == KIND_PURGE and r.sound and r.rec is not None:
            for first, last, head in r.rec["chain_purge"]:
                spans.append((int(first), int(last)))
                heads[int(last)] = bytes.fromhex(head)
    merged: list[tuple[int, int]] = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged, heads


def _covered(merged: list[tuple[int, int]], lo: int, hi: int) -> bool:
    return any(a <= lo and hi <= b for a, b in merged)


def _scan(db: DB, key: bytes, verify_key: VerifyKey) -> dict:
    side, unreadable = _side_records(db)
    rows, verdict_bad = _sighting_records(db, side)
    recs = sorted(rows + side, key=lambda r: r.seq)
    anchors = _anchors(db)
    issues: list[tuple[int, str, str]] = [(0, EDITED, "unreadable chain side record")] * unreadable
    for r in recs:
        r.sound = r.rec is not None and equal(link(key, r.prev, r.rec), r.h)
        if not r.sound:
            issues.append((r.seq, EDITED, f"{r.kind} record does not match its hash"))
    issues += [(s, EDITED, "verdict differs from the chained verdict change") for s in verdict_bad]
    merged, purge_heads = _purge_ranges(recs)
    signed = [anchor_valid(a, verify_key) for a in anchors]
    good_anchors = [a for a, ok in zip(anchors, signed) if ok]
    anchor_at = {a.idx: a for a in good_anchors}
    known = {r.h: r.seq for r in recs if r.sound}
    purged_gaps, last = 0, None
    for r in recs:
        if last is not None and r.seq == last.seq:
            issues.append((r.seq, REORDERED, "two records share one seq"))
            last = r
            continue
        lo, hi = (last.seq + 1 if last is not None else 1), r.seq - 1
        expected = last.h if last is not None else GENESIS
        if r.kind == KIND_RESTART:          # a segment starts at GENESIS; only its gap is checked
            if lo <= hi:
                if _covered(merged, lo, hi) or (last is None and hi in anchor_at):
                    purged_gaps += 1
                else:
                    issues.append((lo, DELETED, f"seqs {lo}-{hi} missing without a purge boundary"))
        elif lo <= hi:
            head = purge_heads.get(hi)
            if _covered(merged, lo, hi) and head is not None and equal(head, r.prev):
                purged_gaps += 1
            elif last is None and hi in anchor_at and equal(anchor_at[hi].head, r.prev):
                purged_gaps += 1            # purged prefix ending exactly at an anchor
            else:
                where = "prefix" if last is None else "middle"
                issues.append((lo, DELETED, f"{where} seqs {lo}-{hi} missing without a purge boundary"))
        elif not equal(r.prev, expected):
            if r.prev in known and known[r.prev] != r.seq - 1:
                issues.append((r.seq, REORDERED, "record links to a record stored elsewhere"))
            else:
                issues.append((r.seq, DELETED, "record links to a record that is not stored"))
        last = r
    last_seq = recs[-1].seq if recs else 0
    by_seq = {r.seq: r for r in recs}
    for a, ok in zip(anchors, signed):
        if not ok:
            issues.append((a.idx, BAD_ANCHOR, "anchor signature does not verify"))
        elif a.idx > last_seq:
            issues.append((a.idx, TRUNCATED, f"anchor {a.idx} is past the last stored seq {last_seq}"))
        elif a.idx in by_seq and not equal(by_seq[a.idx].h, a.head):
            issues.append((a.idx, EDITED, f"record {a.idx} differs from its signed anchor"))
        # an anchor with no retained record at its idx (purged, or below the horizon) checks its signature only
    matched = [a for a in good_anchors if a.idx in by_seq and equal(by_seq[a.idx].h, a.head)]
    last_anchor = max(matched, key=lambda a: a.idx) if matched else None
    starts = sorted({recs[0].seq, *(r.seq for r in recs if r.kind == KIND_RESTART)}) if recs else []
    issues.sort(key=lambda i: i[0])
    seg_status = []
    for n, s in enumerate(starts):
        mine = [i for i in issues if bisect.bisect_right(starts, i[0]) - 1 == n]
        seg_status.append({"segment": n + 1, "first_seq": s, "status": mine[0][1] if mine else OK,
                           "records": sum(1 for r in recs if bisect.bisect_right(starts, r.seq) - 1 == n),
                           "issues": len(mine)})
    tail_from = last_anchor.idx if last_anchor is not None else 0
    return {
        "status": issues[0][1] if issues else OK,
        "records": len(recs), "anchors": len(anchors), "segments": len(starts),
        "first_seq": recs[0].seq if recs else None, "last_seq": last_seq if recs else None,
        "unanchored_tail": sum(1 for r in recs if r.seq > tail_from),
        "purged_gaps": purged_gaps,
        "issues": [{"seq": s, "reason": c, "note": t} for s, c, t in issues[:MAX_ISSUES_REPORTED]],
        "issue_count": len(issues),
        "segment_status": seg_status,
        # internal: used by SightingLog to resume; dropped from the public report
        "_last": recs[-1] if recs else None, "_last_anchor": last_anchor,
        "_top": max([last_seq, *(a.idx for a in anchors), *(b for _, b in merged)]),
        # pre-ADR-0066 shape: nothing persisted but rows, and a later session restarted at GENESIS
        "_legacy": not anchors and not side and unreadable == 0 and any(equal(r.prev, GENESIS) for r in recs[1:]),
    }


def verify_log(db: DB, chain_key: bytes, verify_key: VerifyKey) -> dict:
    """Verify the stored sighting chain across every session and segment. Report: status (ok,
    edited, deleted, reordered, truncated, bad_anchor: the first problem by seq), records and
    anchors checked, segments, first/last seq, unanchored tail, legal purged gaps, issues
    (seq, reason, note; first MAX_ISSUES_REPORTED) and per-segment status. Never content."""
    return {k: v for k, v in _scan(db, chain_key, verify_key).items() if not k.startswith("_")}


@dataclass
class GpsFix:
    lat: float | None
    lon: float | None
    quality: str = "no_fix"     # "no_fix" | "2d" | "3d"


class SightingLog:
    def __init__(self, db: DB, chain: HashChain):
        self.db, self.chain = db, chain
        self._persisted = 0          # highest anchor idx in chain_anchor
        self.chain_status = self._open()

    # -- chain continuity (ADR-0066) ------------------------------------------------------------
    def _open(self) -> dict:
        rep = _scan(self.db, self.chain.key, self.chain.signer.verify_key)
        self._persisted = max([0, *(a.idx for a in _anchors(self.db))])
        counts = {"records": rep["records"], "anchors": rep["anchors"], "first_seq": rep["first_seq"],
                  "last_seq": rep["last_seq"]}
        if rep["_last"] is None and not rep["anchors"]:
            return {"state": "new", "segment": 1, "unanchored_tail": 0, **counts}
        segs = rep["segment_status"]
        current = segs[-1]["status"] if segs else rep["status"]
        if current == OK:
            last, la = rep["_last"], rep["_last_anchor"]
            self.chain.resume(last.seq, last.h, la)
            return {"state": "resumed", "segment": len(segs), "unanchored_tail": rep["unanchored_tail"], **counts}
        reason = LEGACY if rep["_legacy"] else current
        segment = len(segs) + 1
        self.chain.resume(rep["_top"], GENESIS, None)
        self._append_side(KIND_RESTART, {"chain_restart": reason, "segment": segment, "records": rep["records"],
                                         "issues": rep["issue_count"], "t_wall": time.time()})
        if self.chain.has_unanchored():      # sign the new segment's start; it links to nothing older
            self.chain.anchor()
            self._persist_anchors()
        return {"state": "restarted", "reason": reason, "segment": segment, "unanchored_tail": 0, **counts}

    def _persist_anchors(self) -> None:
        for a in self.chain.anchors:
            if a.idx > self._persisted:
                self.db.execute("INSERT INTO chain_anchor(idx, head, sig) VALUES(?,?,?)", (a.idx, a.head, a.sig))
                self._persisted = a.idx

    def _append_side(self, kind: str, rec: dict) -> bytes:
        """Chain a record kept in audit_log (no sighting content): stored first, then appended."""
        seq, prev = self.chain.length + 1, self.chain.head
        h = link(self.chain.key, prev, rec)
        self.db.audit(kind, {**rec, "seq": seq, "prev": prev.hex(), "h": h.hex()})
        self.chain.append(rec)
        self._persist_anchors()
        return h

    def tick(self) -> None:
        """60-second anchor timer for idle periods (anchors only if this session added records)."""
        self.chain.tick()
        self._persist_anchors()

    def close(self) -> None:
        """Anchor this session's unanchored records. The live runner calls it at stop (wiring task)."""
        if self.chain.has_unanchored():
            self.chain.anchor()
        self._persist_anchors()

    def record_purge_boundary(self, cutoff: float) -> dict | None:
        """Before retention deletes rows older than `cutoff` (keep=0), chain the seq ranges about
        to go and the head of each range's last record, so verification can tell this purge from
        a deletion. Returns the boundary record, or None when nothing would be purged."""
        doomed = {int(s): bytes(h) for s, h in self.db.execute(
            "SELECT chain_idx, h_this FROM sighting WHERE t_wall < ? AND COALESCE(keep,0)=0 AND chain_idx IS NOT NULL",
            (cutoff,))}
        if not doomed:
            return None
        side, _ = _side_records(self.db)
        present = sorted({int(s) for (s,) in self.db.execute("SELECT chain_idx FROM sighting WHERE chain_idx IS NOT NULL")}
                         | {r.seq for r in side})
        ranges: list[list] = []
        run: list[int] = []
        for s in [*present, None]:
            if s is not None and s in doomed:
                run.append(s)
                continue
            if run:
                ranges.append([run[0], run[-1], doomed[run[-1]].hex()])
                run = []
        rec = {"chain_purge": ranges, "cutoff": cutoff, "rows": len(doomed), "t_wall": time.time()}
        self._append_side(KIND_PURGE, rec)
        return rec

    def purge(self, days: int = 30, now: float | None = None) -> int:
        """Retention purge (ETHICS.md) with a recorded boundary: the drop-in for
        guard.retention.purge(db, days) wherever a SightingLog exists."""
        from ..guard.retention import purge
        now = time.time() if now is None else now
        self.record_purge_boundary(now - days * SECONDS_PER_DAY)
        return purge(self.db, days=days, now=now)

    def verify(self) -> dict:
        return verify_log(self.db, self.chain.key, self.chain.signer.verify_key)

    def add(self, event: dict, gps: GpsFix, crop_path: str | None = None, verdict: str = "unresolved") -> str:
        if verdict not in VERDICTS:
            raise ValueError("bad verdict")
        sid = uuid.uuid4().hex
        if gps.quality == "no_fix":
            lat = lon = None   # never guessed
        else:
            lat, lon = gps.lat, gps.lon
        body = {"event": event, "lat": lat, "lon": lon, "fix_quality": gps.quality, "crop_path": crop_path}
        rec = {"sighting_id": sid, "t_wall": time.time(), "codes": [event.get("signature")], "verdict": verdict}
        full = {**rec, "body_digest": json.dumps(body, sort_keys=True)}
        seq, prev = self.chain.length + 1, self.chain.head
        h = link(self.chain.key, prev, full)     # stored first, then appended: no in-memory-only link
        self.db.execute("INSERT INTO sighting VALUES(?,?,?,?,?,?,?,?,0)",
                        (sid, rec["t_wall"], json.dumps(rec["codes"]), self.db.seal_json(body, sid.encode()),
                         verdict, seq, prev, h))
        self.chain.append(full)
        self._persist_anchors()
        return sid

    def set_verdict(self, sid: str, verdict: str) -> None:
        if verdict not in VERDICTS:
            raise ValueError("bad verdict")
        r = self.db.execute("SELECT verdict FROM sighting WHERE sighting_id=?", (sid,)).fetchone()
        self.db.execute("UPDATE sighting SET verdict=? WHERE sighting_id=?", (verdict, sid))
        self._append_side(KIND_VERDICT, {"sighting_id": sid, "verdict_change": verdict,
                                         "verdict_from": r[0] if r else None, "t_wall": time.time()})

    def get(self, sid: str) -> dict:
        r = self.db.execute("SELECT sighting_id,t_wall,codes,body,verdict FROM sighting WHERE sighting_id=?", (sid,)).fetchone()
        if not r:
            raise KeyError(sid)
        return {"sighting_id": r[0], "t_wall": r[1], "codes": json.loads(r[2]), "verdict": r[4],
                **self.db.open_json(r[3], r[0].encode())}

    def all(self) -> list[dict]:
        return [self.get(r[0]) for r in self.db.execute("SELECT sighting_id FROM sighting ORDER BY t_wall")]

    def export(self, decimals: int = 2) -> list[dict]:
        out = []
        for s in self.all():
            s = dict(s)
            s["lat"], s["lon"] = round_gps(s["lat"], s["lon"], decimals)
            s.pop("crop_path", None)
            out.append(s)
        return out

    def map_pins(self) -> dict:
        """Pins for the offline map layer (our own layer; never merged into OSM data)."""
        pins = [{"id": s["sighting_id"], "lat": s["lat"], "lon": s["lon"], "codes": s["codes"], "verdict": s["verdict"]}
                for s in self.all() if s["lat"] is not None]
        return {"attribution": OSM_ATTRIBUTION, "pins": pins}


class SweepTimeline:
    def __init__(self, db: DB):
        self.db = db

    def start(self, site_code: str) -> str:
        sid = uuid.uuid4().hex
        self.db.execute("INSERT INTO sweep(sweep_id,site_code,t_wall,body,outcome) VALUES(?,?,?,?,?)",
                        (sid, site_code, time.time(), self.db.seal_json({"counts": {}}, sid.encode()), "open"))
        return sid

    def finish(self, sweep_id: str, counts: dict, new_since_last: list, outcome: str = "done") -> None:
        body = {"counts": counts, "new_since_last": new_since_last}
        self.db.execute("UPDATE sweep SET t_end=?, body=?, outcome=? WHERE sweep_id=?",
                        (time.time(), self.db.seal_json(body, sweep_id.encode()), outcome, sweep_id))

    def history(self, site_code: str | None = None) -> list[dict]:
        q = "SELECT sweep_id,site_code,t_wall,t_end,body,outcome FROM sweep"
        args: tuple = ()
        if site_code:
            q += " WHERE site_code=?"; args = (site_code,)
        rows = self.db.execute(q + " ORDER BY t_wall", args).fetchall()
        return [{"sweep_id": r[0], "site_code": r[1], "start": r[2], "end": r[3], "outcome": r[5],
                 **self.db.open_json(r[4], r[0].encode())} for r in rows]
