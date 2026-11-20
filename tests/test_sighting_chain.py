"""ADR-0066 resumable sighting chain: resume across sessions, persisted anchors, restart on a
broken chain, verification across retention. Synthetic only: temp DB, generated keys."""
import json, os, time
import pytest
from nacl.signing import SigningKey

from oiris.security.hashchain import ANCHOR_EVERY_RECORDS, ANCHOR_EVERY_SECONDS, GENESIS, HashChain
from oiris.store.db import DB
from oiris.store.sightings import SECONDS_PER_DAY, GpsFix, SightingLog, verify_log

NO_FIX = GpsFix(None, None)


@pytest.fixture
def env(tmp_path, key, clock):
    e = {"path": tmp_path / "s.db", "key": key, "ck": os.urandom(32), "sk": SigningKey.generate(),
         "clock": clock, "dbs": []}
    yield e
    for d in e["dbs"]:
        d.close()


def _open(env) -> SightingLog:
    db = DB(env["path"], env["key"])
    env["dbs"].append(db)
    return SightingLog(db, HashChain(env["ck"], env["sk"], clock=env["clock"]))


def _add(log, n, start=0):
    return [log.add({"signature": f"A{(start + i) % 7}", "level": "attention"}, NO_FIX) for i in range(n)]


def _report(log):
    return verify_log(log.db, log.chain.key, log.chain.signer.verify_key)


def _rows(db):
    return db.execute("SELECT chain_idx, h_prev, h_this FROM sighting ORDER BY chain_idx").fetchall()


def _anchor_idx(db):
    return [i for (i,) in db.execute("SELECT idx FROM chain_anchor ORDER BY idx")]


# ---------------- resume ----------------

def test_two_sessions_continue_one_chain(env):
    log = _open(env)
    assert log.chain_status["state"] == "new"
    _add(log, 5)
    log.close()
    a1 = _anchor_idx(log.db)
    assert a1 == [5]
    log2 = _open(env)
    st = log2.chain_status
    assert st["state"] == "resumed" and st["last_seq"] == 5 and st["unanchored_tail"] == 0 and st["segment"] == 1
    _add(log2, 4, 5)
    log2.close()
    rows = _rows(log2.db)
    assert [r[0] for r in rows] == list(range(1, 10))                 # no gap
    assert bytes(rows[5][1]) == bytes(rows[4][2])                     # session 2 links to session 1's last h
    assert bytes(rows[0][1]) == GENESIS
    assert _anchor_idx(log2.db) == [5, 9]                             # anchors persist
    rep = _report(log2)
    assert rep["status"] == "ok" and rep["records"] == 9 and rep["segments"] == 1
    assert rep["first_seq"] == 1 and rep["last_seq"] == 9 and rep["anchors"] == 2 and rep["unanchored_tail"] == 0


def test_resume_reports_unanchored_tail_and_does_not_anchor_it(env):
    log = _open(env)
    _add(log, 3)                       # no close(): the app "crashed"
    assert _anchor_idx(log.db) == []
    log2 = _open(env)
    assert log2.chain_status["state"] == "resumed" and log2.chain_status["unanchored_tail"] == 3
    env["clock"].advance(ANCHOR_EVERY_SECONDS + 1)
    log2.tick()
    log2.close()
    assert _anchor_idx(log2.db) == []  # nothing new this session: the old tail is not blessed
    assert _report(log2)["unanchored_tail"] == 3


def test_verdict_change_is_chained_and_persisted(env):
    log = _open(env)
    a, _ = _add(log, 2)
    log.set_verdict(a, "confirmed")
    log.close()
    log2 = _open(env)
    assert log2.chain_status["state"] == "resumed" and log2.chain_status["last_seq"] == 3
    assert log2.get(a)["verdict"] == "confirmed"
    assert _report(log2)["status"] == "ok"
    log2.db.execute("UPDATE sighting SET verdict='benign' WHERE sighting_id=?", (a,))     # edited verdict
    assert _report(log2)["status"] == "edited"


# ---------------- anchors ----------------

def test_anchor_every_100_records(env):
    log = _open(env)
    _add(log, 2 * ANCHOR_EVERY_RECORDS + 5)
    assert _anchor_idx(log.db) == [ANCHOR_EVERY_RECORDS, 2 * ANCHOR_EVERY_RECORDS]
    log.close()
    assert _anchor_idx(log.db)[-1] == 2 * ANCHOR_EVERY_RECORDS + 5


def test_anchor_every_60_seconds_with_new_records(env):
    log = _open(env)
    _add(log, 3)
    assert _anchor_idx(log.db) == []
    env["clock"].advance(ANCHOR_EVERY_SECONDS)
    _add(log, 1)
    assert _anchor_idx(log.db) == [4]
    env["clock"].advance(ANCHOR_EVERY_SECONDS)
    log.tick()
    assert _anchor_idx(log.db) == [4]          # idle: no new records, no anchor
    _add(log, 1)
    env["clock"].advance(ANCHOR_EVERY_SECONDS)
    log.tick()
    assert _anchor_idx(log.db) == [4, 5]


def test_no_more_than_100_unanchored_across_a_resume(env):
    log = _open(env)
    _add(log, 30)
    log.close()
    log2 = _open(env)
    _add(log2, 2 * ANCHOR_EVERY_RECORDS)
    idx = _anchor_idx(log2.db)
    assert idx == [30, ANCHOR_EVERY_RECORDS, 2 * ANCHOR_EVERY_RECORDS]   # the 100-rule runs on the global seq
    assert all(b - a <= ANCHOR_EVERY_RECORDS for a, b in zip([0, *idx], idx))


# ---------------- tampering after a restart of the app ----------------

def _two_sessions(env, n1=6, n2=6, close=True):
    log = _open(env)
    _add(log, n1)
    log.close()
    log2 = _open(env)
    _add(log2, n2, n1)
    if close:
        log2.close()
    return log2


def test_edited_body_detected(env):
    log = _two_sessions(env)
    sid, = log.db.execute("SELECT sighting_id FROM sighting WHERE chain_idx=8").fetchone()
    body = log.db.open_json(log.db.execute("SELECT body FROM sighting WHERE sighting_id=?", (sid,)).fetchone()[0],
                            sid.encode())
    body["event"]["signature"] = "Z9"
    log.db.execute("UPDATE sighting SET body=? WHERE sighting_id=?", (log.db.seal_json(body, sid.encode()), sid))
    rep = _report(log)
    assert rep["status"] == "edited" and rep["issues"][0]["seq"] == 8


def test_deleted_middle_row_detected(env):
    log = _two_sessions(env)
    log.db.execute("DELETE FROM sighting WHERE chain_idx=4")
    rep = _report(log)
    assert rep["status"] == "deleted" and rep["issues"][0]["seq"] == 4


def test_swapped_rows_detected(env):
    log = _two_sessions(env)
    db = log.db
    db.execute("UPDATE sighting SET chain_idx=-1 WHERE chain_idx=7")
    db.execute("UPDATE sighting SET chain_idx=7 WHERE chain_idx=8")
    db.execute("UPDATE sighting SET chain_idx=8 WHERE chain_idx=-1")
    assert _report(log)["status"] == "reordered"


def test_deleted_tail_past_last_anchor_is_truncated(env):
    log = _two_sessions(env)                         # closed: anchor at 12
    log.db.execute("DELETE FROM sighting WHERE chain_idx >= 11")
    rep = _report(log)
    assert rep["status"] == "truncated" and rep["last_seq"] == 10


def test_deleted_unanchored_tail_shows_as_a_shorter_tail(env):
    log = _two_sessions(env, close=False)            # anchor at 6 only; 6 records unanchored
    assert _report(log)["unanchored_tail"] == 6
    log.db.execute("DELETE FROM sighting WHERE chain_idx >= 11")
    rep = _report(log)
    # cryptographically invisible past the last anchor: stated, and why close() anchors the tail
    assert rep["status"] == "ok" and rep["unanchored_tail"] == 4


def test_forged_anchor_signature_detected(env):
    log = _two_sessions(env)
    log.db.execute("UPDATE chain_anchor SET sig=? WHERE idx=6", (os.urandom(64),))
    assert _report(log)["status"] == "bad_anchor"
    log.db.execute("INSERT INTO chain_anchor(idx, head, sig) VALUES(?,?,?)", (99, os.urandom(32), os.urandom(64)))
    assert {i["reason"] for i in _report(log)["issues"]} == {"bad_anchor"}


def test_wrong_keys_do_not_verify(env):
    log = _two_sessions(env)
    rep = verify_log(log.db, os.urandom(32), log.chain.signer.verify_key)
    assert rep["status"] == "edited"
    rep = verify_log(log.db, log.chain.key, SigningKey.generate().verify_key)
    assert rep["status"] == "bad_anchor"


# ---------------- broken chain at open ----------------

def test_broken_chain_at_open_restarts_a_segment(env):
    log = _two_sessions(env)
    log.db.execute("DELETE FROM sighting WHERE chain_idx=3")
    before = log.db.execute("SELECT * FROM sighting ORDER BY chain_idx").fetchall()
    anchors_before = log.db.execute("SELECT * FROM chain_anchor ORDER BY idx").fetchall()
    log3 = _open(env)
    st = log3.chain_status
    assert st["state"] == "restarted" and st["reason"] == "deleted" and st["segment"] == 2
    audit = log3.db.execute("SELECT detail FROM audit_log WHERE kind='chain_restart'").fetchall()
    assert len(audit) == 1
    d = json.loads(audit[0][0])
    assert d["chain_restart"] == "deleted" and d["segment"] == 2 and d["records"] == 11 and d["seq"] == 13
    assert bytes.fromhex(d["prev"]) == GENESIS
    assert set(d) == {"chain_restart", "segment", "records", "issues", "t_wall", "seq", "prev", "h"}   # no content
    assert log3.db.execute("SELECT * FROM sighting ORDER BY chain_idx").fetchall() == before    # untouched
    assert log3.db.execute("SELECT * FROM chain_anchor ORDER BY idx").fetchall()[:2] == anchors_before
    _add(log3, 2, 20)
    log3.close()
    rep = _report(log3)
    assert rep["segments"] == 2 and rep["status"] == "deleted"          # the history keeps its finding
    assert [s["status"] for s in rep["segment_status"]] == ["deleted", "ok"]
    log4 = _open(env)                                                    # the new segment resumes
    assert log4.chain_status["state"] == "resumed" and log4.chain_status["segment"] == 2
    assert log4.chain_status["last_seq"] == 15
    assert log4.db.execute("SELECT count(*) FROM audit_log WHERE kind='chain_restart'").fetchone()[0] == 1


def test_legacy_rows_restart_with_legacy_reason(env):
    log = _open(env)
    _add(log, 3)
    db = log.db                      # pre-ADR-0066 shape: each session restarted at seq 1 from GENESIS
    db.execute("INSERT INTO sighting SELECT 'x'||sighting_id, t_wall, codes, body, verdict, chain_idx, h_prev, h_this, 0 "
               "FROM sighting")
    log2 = _open(env)
    assert log2.chain_status["state"] == "restarted" and log2.chain_status["reason"] == "legacy"


# ---------------- retention ----------------

def _aged(env, n_old=5, n_new=4, keep_seq=None):
    log = _open(env)
    old = time.time() - 40 * SECONDS_PER_DAY
    with pytest.MonkeyPatch.context() as mp:      # written 40 days ago (t_wall is chained: never edit it)
        mp.setattr(time, "time", lambda: old)
        _add(log, n_old)
    if keep_seq is not None:
        log.db.execute("UPDATE sighting SET keep=1 WHERE chain_idx=?", (keep_seq,))
    _add(log, n_new, n_old)
    log.close()
    return log


def test_recorded_prefix_purge_verifies_ok(env):
    _aged(env)
    log2 = _open(env)
    assert log2.chain_status["state"] == "resumed"
    assert log2.purge(days=30) >= 5
    rep = _report(log2)
    assert rep["status"] == "ok" and rep["first_seq"] == 6 and rep["purged_gaps"] == 1
    log2.close()
    assert _open(env).chain_status["state"] == "resumed"


def test_recorded_purge_around_a_kept_row_verifies_ok(env):
    log = _aged(env, keep_seq=3)
    log.purge(days=30)
    assert [r[0] for r in _rows(log.db)] == [3, 6, 7, 8, 9]
    rep = _report(log)
    assert rep["status"] == "ok" and rep["purged_gaps"] == 2
    log.db.execute("UPDATE sighting SET keep=0 WHERE chain_idx=3")   # a later purge takes the kept row too
    log.purge(days=30)
    assert _report(log)["status"] == "ok"


def test_unrecorded_purge_is_reported(env):
    from oiris.guard.retention import purge
    log = _aged(env)
    purge(log.db, days=30)                       # the old path: no boundary
    rep = _report(log)
    assert rep["status"] == "deleted" and "prefix" in rep["issues"][0]["note"]


def test_prefix_purge_ending_at_an_anchor_is_legal(env):
    log = _aged(env)                             # close() anchored at 9; anchor at 5 is made below
    log.db.execute("DELETE FROM chain_anchor")
    log2 = _open(env)
    from oiris.security.hashchain import Anchor, anchor_message
    h5 = bytes(log2.db.execute("SELECT h_this FROM sighting WHERE chain_idx=5").fetchone()[0])
    a = Anchor(5, h5, env["sk"].sign(anchor_message(5, h5)).signature)
    log2.db.execute("INSERT INTO chain_anchor VALUES(?,?,?)", (a.idx, a.head, a.sig))
    log2.db.execute("DELETE FROM sighting WHERE chain_idx <= 5")
    rep = _report(log2)
    assert rep["status"] == "ok" and rep["purged_gaps"] == 1


def test_middle_deletion_after_a_recorded_purge_still_detected(env):
    log = _aged(env)
    log.purge(days=30)
    log.db.execute("DELETE FROM sighting WHERE chain_idx=7")
    assert _report(log)["status"] == "deleted"


def test_purge_with_nothing_expired_adds_no_record(env):
    log = _open(env)
    _add(log, 2)
    assert log.record_purge_boundary(0.0) is None
    assert log.db.execute("SELECT count(*) FROM audit_log WHERE kind='chain_purge'").fetchone()[0] == 0


# ---------------- unchanged paths ----------------

def test_export_and_verdict_paths_unchanged(env):
    log = _open(env)
    a = log.add({"signature": "A1"}, GpsFix(12.345678, -45.678912, "3d"))
    log.set_verdict(a, "benign")
    with pytest.raises(ValueError):
        log.set_verdict(a, "is_robot")
    ex = log.export(2)
    assert ex[0]["lat"] == 12.35 and ex[0]["verdict"] == "benign" and "crop_path" not in ex[0]
    assert set(ex[0]) == {"sighting_id", "t_wall", "codes", "verdict", "event", "lat", "lon", "fix_quality"}
