# oiris.modules.face_match — M-16 face matching (optional, off by default)

Spec: `docs/FACE_MATCHING_MODULE_SPEC.md` · ADR-0049, ADR-0050 · ETHICS.md.

- `port.py` — `FaceMatchPort` (backends only compute), `Embedding` (RAM only, `zero()`), `Angle`.
- `core.py` — `FaceMatchModule`: storage, keys, modes, enrollment (`start_session`, `add_person`, `needed_angles`, `submit_shot`, `finish_person`, `end_session`, `reenroll`, `delete_person`), `process_frame` (counts only), `review_log`, `delete_entry`, `export_log` (blocked by default, never templates), `disable` (verified wipe).
- `quality.py` — per-shot checks: one face, size, light, blur, angle.
- `optin.py` — warning text v1, typed `I UNDERSTAND`, jurisdiction notes, tamper-evident `DbChain`.
- `supervisor.py` — every backend call in a thread with timeout and a circuit breaker.
- `backends/opencv_zoo.py` — YuNet (MIT) + SFace (Apache-2.0); `oiris face-match fetch-models`.
- `backends/test_deterministic.py` — **TEST ONLY**, synthetic frames, refused outside tests.

Loaded only by `oiris/modules/face_match_gate.load()` when `"M-16": true` and the opt-in is recorded.
No alerts, no events, no live lookup, no limit on enrolled people. Never labels anyone as anything but a user-chosen name or "unrecognized".

CLI: `oiris face-match status | enable --mode personal [--mode research] | fetch-models | enroll [--camera 0] | list | review | delete-entry --id E | delete-person --id P | photo-delete --person P | disable`.

## Photo (ADR-0057 decision 4)

- **What is kept.** At most one photo per enrolled person: a face crop from that person's own best FRONT enrollment shot (`photo.py`). "Best" is the highest sharpness, the blur measure `check_shot` already uses; a tie keeps the earlier shot. The crop is the face box plus a 25 % margin on each side (`PHOTO_MARGIN`), clipped to the frame and resized to 160 x 160 px (`PHOTO_SIZE_PX`), stored as JPEG quality 85 (`PHOTO_JPEG_QUALITY`, `cv2.imencode`, no metadata). Live frames are never used, and strangers' faces are never kept.
- **Why.** Only to show who was recognised (the dashboard card comes later and calls `FaceMatchModule.photo(person_id)`). There is no list, export, file path or log of the image; `face-match list` shows only "photo: yes/no".
- **Consent.** Only with a current consent record that has the `photo` scope (`oiris face-match consent --subject CODE --scope face_enrollment,photo`) and only if `check_persist("face_photo", ...)` allows it. Without it, no crop is even kept in RAM, and the enrollment message says "No photo kept".
- **How it is kept.** In `fm_photo`, sealed with the person's own data key (AAD `b"photo:" + person_id`), written in the same transaction as the template. Capture buffers are RAM only: a replaced candidate is zeroed, and everything is zeroed when the capture ends, finished or not. A re-enrollment replaces the photo; the old key is crypto-erased.
- **How it is erased** (verified wipe, reported in `WipeReport.photos_destroyed`): `delete-person`; `photo-delete --person P` (keeps the enrollment, replaces the person's key); withdrawing the consent or just its photo scope (`consent --withdraw --id ID [--scope photo]`); consent expiry; `disable`. A photo whose consent is no longer current is erased the next time the module loads or the photo is asked for.
