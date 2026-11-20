# oiris.link — phone ↔ pod / glasses transport
`ble.py`: BLE GATT service (`6f697269-7300-4c69-6e6b-…01`), RX write-without-response + TX notify, segmenting of Noise frames to the ATT MTU (seq | flags | [length] | payload), reassembly that drops only a damaged frame, `loopback_pair` for tests, `BlessGattServer` (pod, needs a radio, `bless` MIT). `pod_service.py`: pod request handler (thermal frames for the phone's live view; `time_sync`; the glasses link ops below; delegates face-match enrollment ops). Byte format shared with `android/core` `BleFraming.kt` via `tests/vectors/ble_framing.json`. Everything on the air is already Noise ciphertext. Tests: `tests/test_link_ble.py`. Radio path untested (needs hardware).

`timesync.py`: NTP-style clock offset and drift over the `time_sync` op (ADR-0060), matched by `android/core` `TimeSync.kt` (`tests/vectors/time_sync.json`).

## Link operations for glasses (ADR-0062)
Glasses have no keyboard and may be offline; the paired phone is the owner's controller. Each op needs its collaborators on `PodService` (`keystore`, `panic`, `phone_verify_key`, `hud`, `pins_path` + `models_dir`); without them it replies `{"ok":false,"error":"not available"}`.

| Op | Limits |
|---|---|
| `unlock` (`ops.py`) | slot whitelist (`phone`); 32-byte secret, strict base64url; one tried attempt per 2 s (refused without counting); every failure counts toward the 10-failure erase; reply never says why; secret never logged or echoed (zeroed bytearray, best effort in Python) |
| `add_unlock_slot` | store unlocked **and** a pairing window opened on the pod (`PodService.open_pairing_window`, default 60 s, max 300 s, one add per window; never an op) |
| `panic` | always allowed on the paired channel: PanicController soft path, then `KeyStore.lock()` |
| `wipe_challenge` / `wipe` | 32-byte nonce, single use (any wipe attempt consumes it), 60 s; command `ASB-WIPE:`+nonce signed by the paired phone's Ed25519 key → `PanicController.remote_wipe` |
| `model_offer` / `model_chunk` / `model_commit` / `model_status` (`sideload.py`) | only models pinned exactly in `<data_dir>/models.pins.json` (owner-edited; no op adds pins); ≤ 256 MiB; chunk 512..65536; free space checked; resumable (`.partial/<sha256>` + progress record); SHA-256 checked before `os.replace` into `<models_dir>/<slot>/<name>`; mismatch deletes the partial; `O_NOFOLLOW` everywhere, symlinked dirs refused |
| `hud` | `HudFeed` versions only on change, `since` → `unchanged`; ≤ 64 ops, ≤ 80 code points per string, inside a canvas ≤ 4096 px |

`android/core` `LinkOps.kt` builds every request byte-identically and parses every reply (`HudFrame.parse` drops, caps and clamps untrusted display lists); shared vectors `tests/vectors/link_ops.json` (`scripts/make_android_vectors.py`), checked by `Checks.kt`. BLE moves about 6 KB/s, so a 28 MB model takes over an hour; the ops are transport-independent for a faster link later. Tests: `tests/test_link_ops.py`, `tests/test_link_sideload.py`, `tests/test_link_ops_vectors.py`.

## Phone-guided voice enrollment (ADR-0057)
Glasses have no keyboard and may have a weak microphone, so the owner's phone guides M-17 voice enrollment. It shows each designed sentence, records it and sends the audio. The pod (or the glasses) runs every check and builds the sealed voice template exactly as `oiris voice-link enroll` does. The ops are EnrollmentPort ops (`oiris/modules/face_match/remote.py`, delegated by `PodService`) and need the loaded M-17 module (`EnrollmentServer(..., voice=VoiceLinkModule)`). Without it every voice op replies `{"ok":false,"error":"PermissionError: voice link is off"}`. Consent, guardian, speaker-model and opt-in checks are the module's own, unchanged. Errors are one plain line, never a stack trace.

| Op | Request → reply |
|---|---|
| `voice_prompts` | → `prompts`: `[{id, text, tone}]`, the designed sentences in enrollment order (text only) |
| `voice_start` | `person`, `consent` → `windows`, `target_speech_s` (total speech), `window_s`, `max_windows` |
| `voice_sample` | `person`, `prompt_id`, `rate`, `pcm16` → `accepted`, `code`, `message`, `speech_s`, `total_speech_s`, `level_dbfs`, `remaining` |
| `voice_finish` | `person` → `saved`, `code` (`done`, `windows`, `speech_total`), `message`, `total_speech_s`, `remaining` |
| `voice_cancel` | `person` → `{}`; the open capture's embeddings are zeroed |

The window result's `ok` is sent as `accepted` (and the finish result's as `saved`), because the reply's own `ok` means the op ran. Refusal codes come through unchanged: `prompt`, `rate`, `invalid`, `loud`, `speech`, `backend` and `enough` (`mono` cannot occur, because pcm16 is mono).

Limits (named constants in `remote.py`):
- `pcm16` is strict standard base64 (padded, canonical) of little-endian signed 16-bit mono samples.
- `rate` must be an integer. A rate outside 8000–48000 Hz is not decoded, and the module refuses it with `rate`.
- A sample may be at most 12 s (`MAX_SAMPLE_S`). The size cap is 1,152,000 bytes (12 s at 48 kHz), or 1,536,000 base64 characters (`MAX_PCM_B64`). That is under a fifth of `channel.MAX_MESSAGE` (8 MiB).
- An odd byte count, non-ASCII text, a stray-bit or padding error, or an oversize or too-long sample is refused before anything is decoded where possible.

Audio crosses only the paired Noise channel (fragmented over 60 kB frames). It is decoded into buffers that are zeroed after use: the base64 bytes, every intermediate array and the float buffer handed to `submit_window`, which the module zeroes too. The request string itself is an immutable Python str, so its zeroing is best effort. Audio is never stored, logged or echoed in a reply or an error.

BLE moves about 6 KB/s. A 7 s sample at 16 kHz is about 224 KB (about 300 KB as base64), so it takes roughly 40 s to send (about 50 s with the base64 overhead). Six sentences take about 4–5 minutes, so a faster link would help. 8 kHz halves this.

`android/core` `Enrollment.kt` has the matching pieces:
- `EnrollmentProtocol.voice*` builds each request byte-identically (`voiceSample` takes a `ShortArray`).
- `VoiceEnrollment.parse*` parses each reply and never throws: a malformed reply parses as `bad reply`.
- `VoiceEnrollmentFlow` runs the flow: prompts, then start. Each sentence is submitted and gets feedback; after a refusal the same sentence is read again. The flow cycles to the next sentence while the pod asks for more total speech, and finishes when nothing remains or at the window maximum. Five refusals cancel. Each `ShortArray` is zeroed before `submit` returns.

Shared vectors: `tests/vectors/voice_enrollment.json` (`scripts/make_android_vectors.py`). They come from the real pod handlers, with a temporary store, a deterministic fake speaker and integer-only synthetic PCM. `Checks.kt` checks them, including scripted flows against the recorded replies. The Android microphone screen needs the Android SDK and is later work, and so are the advisory range check and the redo of one sentence that the CLI offers. Tests: `tests/test_voice_enrollment_remote.py`, `tests/test_voice_enrollment_vectors.py`.
