# ADR-0066: A resumable sighting chain: one tamper-evident record across sessions and retention

- Status: Accepted (owner request, 2026-11-17)
- Date: 2026-11-17
- Source:
  - The owner, 2026-11-17, asked for every open item to be built as code plus reasoning.
  - docs/STATUS.md, open item "Live sighting chain": the live sighting hash chain restarts each session and its anchors are not persisted.
  - ADR-0024 (D-24): HMAC hash chain plus Ed25519 anchors every 60 s or 100 records.

## Context
- ADR-0024 chose an HMAC-SHA256 chain over the M-05 sighting log, with Ed25519-signed anchors (index, head) every 100 records or 60 s. `oiris/security/hashchain.py` implements it in memory.
- The sighting rows already store `chain_idx`, `h_prev` and `h_this`, and the `chain_anchor(idx, head, sig)` table exists. But every `SightingLog` started a fresh in-memory chain at GENESIS with index 1, so:
  - each session began a new, unconnected chain, and its `chain_idx` values repeated earlier sessions';
  - anchors were never written to `chain_anchor`, so nothing signed survived a restart;
  - a verdict change was chained in memory only, so the stored chain had a hole after every change;
  - retention (`guard/retention.purge`, run by live mode at every open and every 24 h) deletes old rows, and nothing could tell that from a deletion.
- A glasses or Pi rig may run for weeks and restart often. It needs one record whose edits, deletions, reordering and truncation are detectable over the whole retained history.

## Decision
1. **One chain across sessions.**
   - The chain's records are the `sighting` rows (seq = `chain_idx`) plus side records kept in `audit_log`:
     - `chain_verdict`: a verdict change (sighting id, new verdict, previous verdict, time);
     - `chain_purge`: a purge boundary (see 5);
     - `chain_restart`: a segment start (see 2).
   - A side record's detail is its chained fields plus `seq`, `prev` and `h`. It never holds sighting content: no event, place, crop or body.
   - A sighting row's chained record is rebuilt from the row. The body is opened with the store key and re-serialised for `body_digest`. The verdict at creation is the first later verdict change's `verdict_from`, else the column. The current column must equal the last chained verdict change.
   - Each record is stored before it is appended in memory, so no link exists only in memory.
2. **Resume, or restart with a reason.** When a `SightingLog` opens, it verifies the stored chain (`verify_log`, below).
   - Empty store: `chain_status = {"state": "new", ...}`.
   - The current segment verifies: the chain resumes. The next record has seq = last seq + 1 and prev = the last h. `chain_status` is `"resumed"`, with counts and `unanchored_tail`.
   - It does not verify: the log does not continue silently.
     - It starts segment n + 1. Its first record is a `chain_restart` side record with prev = GENESIS and seq above every stored seq, anchor and purge range.
     - That record is also the audit entry (kind `chain_restart`). Its detail is the reason code and counts only.
     - The new segment's start is anchored at once. It links to nothing older, so the anchor blesses nothing that was there before.
     - `chain_status = {"state": "restarted", "reason": code, "segment": n + 1, ...}`.
     - Past rows and anchors are never rewritten or re-signed.
   - Only the current segment decides this, so an old finding does not restart every session. `verify_log` keeps reporting the old finding for the whole history.
   - Reason `legacy` means rows from before this ADR: no anchors, no side records, and a later session restarted at GENESIS. Those rows age out by retention.
3. **Anchors.** `ANCHOR_EVERY_RECORDS = 100` and `ANCHOR_EVERY_SECONDS = 60` are ADR-0024's numbers, as named constants in `hashchain.py`.
   - After each record, the log persists any new anchor (idx, head, Ed25519 signature) into `chain_anchor`. Anchors come:
     - at every 100th global seq, so no more than 100 records follow the last anchor;
     - when 60 s have passed since the last anchor and this session has added records (`tick()` for idle periods).
   - `close()` anchors this session's unanchored records.
4. **The unanchored tail.** Records stored after the last anchor are the unanchored tail. At open it is reported as `chain_status["unanchored_tail"]`, and the log does not anchor over it: that would bless whatever happened while the app was off. The first anchor this session makes does cover it, because a chain head commits to everything before it. The open-time count is the record that the tail was not signed when the session found it.
5. **Verification across retention.** `verify_log(db, chain_key, verify_key)` returns:
   - status: the first problem by seq, as one reason code (`ok`, `edited`, `deleted`, `reordered`, `truncated` or `bad_anchor`);
   - records and anchors checked, segments, first and last seq;
   - the unanchored tail, legal purged gaps, and issues (seq, reason code, note; the first 20);
   - a status for each segment.

   Each status means:
   - `edited`: a record does not match its HMAC, its body cannot be opened, a verdict column disagrees with its chained change, or a record differs from its signed anchor;
   - `deleted`: a seq gap with no purge boundary, or a link to a record that is not stored;
   - `reordered`: two records share a seq, or a record links to one stored at another seq;
   - `truncated`: a valid anchor lies past the last stored seq;
   - `bad_anchor`: an anchor signature fails.

   Rules:
   - **Retention.**
     - `SightingLog.purge(days, now)` first chains a `chain_purge` boundary (`record_purge_boundary(cutoff)`), then runs `guard.retention.purge` with the same `now`.
     - The boundary lists every run of seqs about to go as [first, last, head of last].
     - Rows kept with `keep = 1` and side records split runs.
   - **Gap rule.** A seq gap is legal only when:
     - authenticated purge boundaries cover it, and the boundary ending at the gap carries exactly the next record's prev; or
     - it is a prefix gap ending exactly at a valid anchor whose head is the earliest retained row's prev.

     Any other gap, in the prefix, the middle or the tail, is `deleted`.
   - **Anchor horizon rule.** An anchor at a seq with no retained record (purged, or below the earliest retained row) is checked for its signature only. Such anchors may be deleted with the rows and their absence is not an error. An anchor at a retained seq must match that record. An anchor past the last seq means truncation.
   - **What it cannot see.** Deleting records past the last anchor, together with nothing else, leaves a shorter unanchored tail and status `ok`. That is why `close()` anchors at stop and the open-time tail count is reported. An attacker holding both keys, or one who deletes the tail rows and their anchors, is out of scope until the remote witness below.

## Built now and later
- **Built now:**
  - `hashchain.py`: `link`, `anchor_message`, `anchor_valid`, a resumable `HashChain` (`base_idx`, `base_head`, `resume`, `has_unanchored`, `length`, `head`);
  - `store/sightings.py`: resume or restart at open, `chain_status`, anchor persistence, `tick`, `close`, chained and persisted verdict changes, `record_purge_boundary`, `purge`, `verify` and `verify_log`;
  - tests: `tests/test_sighting_chain.py`.
- **Later: the wiring in `oiris/live.py`** (held by another task):
  - call `self._log.close()` in `stop()` under `_store_lock`, before the DB closes;
  - in `_purge()`, when `self._log` exists, call `self._log.purge(days=self.cfg.retention_days)` instead of `guard.retention.purge(self.db, ...)`, or call `self._log.record_purge_boundary(now - days * 86400)` and pass the same `now` to `purge`;
  - optionally call `self._log.tick()` from the loop for idle anchoring, and show `chain_status` in the snapshot.

  Until this is wired:
  - each live session leaves its records after the last 100 or 60 s anchor unanchored;
  - each retention purge of sighting rows is reported as `deleted`, and the next open starts a new segment. That is honest, and it is roughly once per retention period.
- **Later: export of the verification report** with the sighting export.
- **Later, an option: a remote anchor witness.** The paired phone would countersign anchors and keep its own copies, so that deleting the tail and its anchors on the Mac or Pi becomes detectable.

## Where it is enforced
- `oiris/security/hashchain.py`: the chain, the anchor rule and its constants, and the signature check.
- `oiris/store/sightings.py`: `SightingLog` (resume, restart, anchor persistence, verdict and purge side records) and `verify_log`.
- `tests/test_sighting_chain.py`: two-session continuity, the 100-record and 60 s anchors, `close()`, each tamper case with its reason code, restart at open, and recorded and unrecorded purges.
- `tests/test_security.py`, `tests/test_audit_gaps.py`, `tests/test_store_golden.py`: the in-memory chain behaviour is unchanged.

## Consequences
- Opening a log reads and opens every retained sighting body once to verify it. That is linear in the retained rows, which retention bounds (30 days by default).
- `audit_log` gains one row per verdict change, per purge that deletes sighting rows, and per restart. Retention does not purge `audit_log`. A verdict side record keeps the random sighting id and verdict words after its sighting expires, never content.
- The in-memory `HashChain` API is a superset of the old one. A fresh chain behaves exactly as before, and `verify()` on in-memory entries is unchanged.
- No schema change and no migration: the existing `sighting`, `chain_anchor` and `audit_log` tables carry everything.

## Changing it
Write a new ADR that supersedes this one and update CHANGELOG.md. Changing the anchor numbers also supersedes ADR-0024. A change that weakens detection (for example accepting a gap without a boundary) needs owner sign-off and negative tests. A change to a frozen test threshold cannot be used to declare a pass retroactively.
