# ADR-0062: Link operations for glasses: unlock, panic and wipe, model side-load and the HUD feed

- Status: Accepted (owner request, 2026-11-17)
- Date: 2026-11-17
- Source:
  - The owner, 2026-11-17, asked for every hardware-dependent part to be built now, as code plus reasoning, so the real glasses and helmet later need only fine-tuning.
  - This ADR covers the link operations a pair of glasses needs from the owner's phone: the pod-side ops in Python, the matching request builders and reply parsers in Kotlin, and shared test vectors.

## Context
- Glasses have no keyboard. They cannot take a passphrase, so they cannot unlock the key store on their own.
- Glasses may be offline by design: no Wi-Fi, no account and no download path. A new or updated model has to come from somewhere the owner controls.
- The owner's paired phone is their controller. It talks to the pod or glasses only over the paired Noise channel (`oiris/security/channel.py`), which BLE carries (`oiris/link/ble.py`). Everything on the air is already Noise ciphertext from the one paired peer.
- The existing ops are `time_sync`, `thermal` and the face-match enrollment ops. Panic, the 10-failure erase and the signed remote wipe already exist in `PanicController` (ADR-0023), and extra key slots exist in `KeyStore` (`add_slot` / `unlock_slot`). The phone just had no way to reach them.
- Some glasses will be only a display. The display is already defined once as a display list (`oiris/ui/glasses.py`, ADR-0052), so the glasses need only the list itself.

## Decision
Every request and reply is canonical JSON (sorted keys, compact). Binary values are base64url without padding, decoded strictly: only the canonical encoding of the expected number of bytes is accepted. An op whose collaborators are not configured on the pod replies `{"ok":false,"error":"not available"}`. These op names never reach the enrollment server.

1. **Unlock from the phone.**
   - The phone keeps a 32-byte random unlock secret in its hardware-backed vault. The app side comes later.
   - `{"op":"unlock","slot":"phone","secret":<32 bytes>}` calls `KeyStore.unlock_slot`. It replies `{"ok":true}` or `{"ok":false,"error":"unlock refused"}` and never says why: a wrong secret, an unknown slot, a malformed secret and the rate limit look the same.
   - The KDF input is the canonical base64url text of the 32 bytes, so it depends only on the bytes.
   - A refused phone-slot attempt does **not** count toward `PanicController`'s 10-failure erase. The secret is 32 random bytes, so it cannot be guessed and counting adds no protection; it would only let a buggy phone app erase the keys in about 20 s. A success still calls `unlock_result(True)`, which resets every failure count.
   - Instead the wait doubles. An attempt within `unlock_wait_s()` of the previous tried attempt is refused without trying and without counting. The wait is `UNLOCK_MIN_INTERVAL_S` (2 s) after one failure, then 4, 8, 16 s and so on, capped at `UNLOCK_BACKOFF_MAX_S` (300 s). A success resets it. Refused spam does not push back the owner's next real try.
   - The passphrase path keeps the 10-failure erase (D-23, ADR-0023) unchanged.
   - Slot names are whitelisted (`UNLOCK_SLOTS`, only `phone` for now).
   - The secret is never logged and never echoed. Its decoded bytes are a `bytearray` zeroed after use. Python cannot erase its immutable strings (the request text, the decoded JSON value, the KDF input), so this is best effort, and the docstring says so.
   - `{"op":"add_unlock_slot","slot":"phone","secret":...}` works only when the store is unlocked and a pairing window is open. The window is opened on the pod itself (`PodService.open_pairing_window(seconds)`, default 60 s, at most 300 s). It is a local action and never an op. One successful add closes the window.
2. **Panic and wipe.**
   - `{"op":"panic"}` runs the PanicController soft path (`soft()`, as the phone's volume-down gesture does) and then `KeyStore.lock()`, so the keys are dropped even if `soft` was wired to something else. It is always allowed on the paired channel. It does not arm the pod button's hard-panic step.
   - `{"op":"wipe_challenge"}` returns a fresh 32-byte nonce, valid for `WIPE_NONCE_TTL_S` (60 s). A new challenge replaces the outstanding one.
   - `{"op":"wipe","command":...,"signature":...}` consumes the outstanding nonce whatever the outcome, so it is single use. It then calls `PanicController.remote_wipe` with the paired phone's Ed25519 verify key and that nonce. The command is `ASB-WIPE:` followed by the nonce.
   - A missing, expired, used or wrong nonce refuses. So does a bad signature or a malformed command. The reply is `{"ok":false,"error":"wipe refused"}`. A good one erases (crypto-erase, ADR-0023) and replies `{"ok":true,"wiped":true}`.
3. **Model side-load.**
   - **Pins.** An owner-edited local file, `<data_dir>/models.pins.json`, holds `[{"name","slot","sha256","size"}]`.
     - `load_pins` refuses the whole file if any entry is bad.
     - Names match `[A-Za-z0-9][A-Za-z0-9._-]{0,127}` and slots match `[a-z][a-z0-9_]{0,31}`, so there are no separators and no leading dot.
     - `sha256` is 64 lowercase hex characters.
     - `size` is 1..`MAX_MODEL_BYTES` (256 MiB).
     - Keys must be exactly those four, with no duplicates.
     - No op adds or changes a pin.
   - `model_offer {name, slot, sha256, size, chunk_size}`:
     - refuses anything not pinned exactly (all four fields);
     - requires `chunk_size` to be 512..65536;
     - checks free space (what remains to be written plus a 32 MiB margin);
     - then opens or resumes `<models_dir>/.partial/<sha256>`.
     - It replies with `chunks`, the missing chunks as half-open index ranges (at most `MAX_MISSING_RANGES` = 64, so the reply stays small) and `missing_count`.
   - `model_chunk {sha256, index, data}`:
     - needs an open offer;
     - bounds-checks the index;
     - requires the data to be exactly the chunk size, except for a short last chunk;
     - is idempotent: a chunk already held is acknowledged and not rewritten.
   - **Resume.** Received ranges are written to `<sha256>.json` beside the partial file every 16 chunks, after an fsync of the data. A dropped link or a pod restart therefore resumes, and at most 15 chunks are sent twice. A different chunk size, or a damaged record, restarts the transfer.
   - `model_commit {sha256}`:
     - requires every chunk;
     - re-reads the pins (the owner may have removed one meanwhile);
     - streams a SHA-256 over the file and compares it, in constant time, with the pin.
     - On a match it fsyncs, `os.replace()`s the file into `<models_dir>/<slot>/<name>` and fsyncs the directory.
     - On a mismatch it deletes the partial file and its record and refuses.
   - `model_status` lists the pinned models present as regular files of the pinned size. The hash was checked at commit; status does not re-read 256 MiB.
   - **Symlinks are never followed.** Every file is opened with `O_NOFOLLOW`. The `.partial` and slot directories must be real directories (checked with `lstat`). A planted symlink is refused (`unsafe path`), not followed and not removed.
   - **Throughput.** BLE carries about 6 KB/s in practice, so a 28 MB model takes over an hour (28 MB / 6 KB/s is about 78 minutes). The ops are transport-independent, so a faster link (Wi-Fi Direct or USB) can carry the same requests later without changing the pod.
4. **HUD display list.**
   - The pod keeps a `HudFeed`. `set(display_list)` bumps the version only when the canonical content changes.
   - `{"op":"hud","since":v}` returns `{"ok":true,"version":n,"ops":[...]}`, or `{"ok":true,"version":n,"unchanged":true}` when `v` is current. Without `since` it always returns the list.
   - The caps are named constants:
     - `MAX_HUD_OPS` (64);
     - `MAX_HUD_TEXT` (80 code points, for every string field);
     - `MAX_CANVAS_PX` (4096);
     - coordinates within the canvas.
   - The pod refuses a list that breaks a cap: that is a bug on the pod, so it fails loudly.
   - The glasses side (`HudFrame.parse`) treats the list as untrusted:
     - the first op must be a valid canvas, otherwise everything is dropped;
     - only the first 64 ops are looked at;
     - unknown kinds, missing or mistyped keys, a second canvas and non-finite numbers are dropped and counted;
     - extra keys are removed;
     - strings are truncated by code points;
     - x and y are clamped into the canvas, w and h so the op ends inside it, sizes into [0, min(W, H)] and opacity into [0, 1];
     - numbers keep their type;
     - NaN and integers beyond a 64-bit long make the whole reply malformed.
5. **Kotlin parity.**
   - `android/core` `LinkOps.kt` has a builder for every request, byte-identical to Python, and a parser for every reply (plus `HudFrame.parse`).
   - Parsers never throw: a malformed reply parses as `ok=false`, `bad reply`.
   - `tests/vectors/link_ops.json` holds the shared cases, generated by `scripts/make_android_vectors.py` from the real pod handlers on a temp key store:
     - one or more request/reply pairs per op, with the expected parsed values;
     - malformed replies;
     - hostile HUD frames;
     - the wipe command with its deterministic Ed25519 signature;
     - chunk counts and base64url cases.
   - `Checks.kt` checks every vector.

## Built now vs later
- **Built now, tested on synthetic data:**
  - the pod ops: unlock, add_unlock_slot, panic, wipe_challenge, wipe, the four model ops and hud;
  - pins validation, resume and the symlink refusals;
  - the HUD feed and its caps;
  - the Kotlin builders and parsers with shared vectors;
  - Python tests with a temp KeyStore, a fake erase, fake clocks and generated keys.
- **Later:**
  - the Android app screens for these ops: unlock, panic, wipe confirmation, model transfer progress and a HUD mirror;
  - the vault holding the unlock secret and the Ed25519 wipe key (Android Keystore);
  - the glasses' own drawing backend for the HUD ops;
  - real BLE throughput measured on the hardware;
  - a faster transport for models (Wi-Fi Direct or USB) carrying the same ops;
  - a pod-side control (a button or console) that calls `open_pairing_window`.

## Where it is enforced
- `oiris/link/ops.py`: unlock, add_unlock_slot, panic, wipe_challenge, wipe, `HudFeed` and the reference builders and parsers.
- `oiris/link/sideload.py`: pins, offer, chunk, commit and status, and their builders and parsers.
- `oiris/link/pod_service.py`: routing, `not available`, and `open_pairing_window`.
- `oiris/security/keys.py` and `oiris/security/panic.py`: unchanged. These ops only call them.
- `android/core` `LinkOps.kt` and `Checks.kt` `linkOpsChecks`, against `tests/vectors/link_ops.json`.
- Tests: `tests/test_link_ops.py`, `tests/test_link_sideload.py` and `tests/test_link_ops_vectors.py`.

## Consequences
- Glasses with no keyboard can be unlocked, panicked and wiped by the paired phone alone. Nothing new is reachable by anyone else, because every op rides the paired Noise channel.
- A phone app with a bug that sends a wrong secret cannot erase the keys: phone failures only slow the next try, up to 5 minutes apart. Guessing the secret stays infeasible at any rate. The 10-failure erase still guards passphrase guessing.
- A model can only ever be one the owner pinned on the device, byte for byte. A stolen or compromised phone can push no other model.
- Side-load over BLE is slow, but it resumes, so long transfers survive a dropped link.
- The HUD can be drawn by glasses that do no computation. A buggy or hostile pod cannot make them draw outside the canvas, draw huge text or allocate without bound.

## Changing it
Write a new ADR that supersedes this one, update CHANGELOG.md, and get the owner's sign-off. Any change to an op's wire format must regenerate `tests/vectors/link_ops.json` and keep `Checks.kt` passing. Do not change the existing vector files.
