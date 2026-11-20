# ADR-0057: Voice link (M-17): recognise enrolled people by voice and show who it is

- Status: Accepted (owner sign-off, 2026-11-16: the recommended option for each owner decision); decision 5 amended by [ADR-0058](0058-voice-link-chime-and-name.md) (a clear match also plays a chime and speaks the name)
- Date: 2026-11-15
- Source: owner request of 2026-11-15 (family use: link voice to face enrollment); docs/VOICE_LINK_MODULE_SPEC.md; ADR-0017, ADR-0018, ADR-0039, ADR-0049, ADR-0050, ADR-0055, ADR-0056; docs/FACE_MATCHING_MODULE_SPEC.md

## Context
The owner wants O Iris to recognise the people the owner has enrolled, for example family members, by their voice, and to show who it is. The voice should be linked to the person's M-16 face enrollment.

Pieces O Iris already has:
- M-16 personal mode: consent records, guardian consent for minors, encrypted per-person templates, a review log, and verified wipe. A live check on a Mac webcam passed on 2026-11-15.
- The `speaker_embed` model slot (ADR-0055). It produces voice embeddings in RAM only, and no model is bundled.
- An encrypted `enrollment` table.
- Live mode and O Iris.app. The app asks macOS for its own permissions (ADR-0056).

Decisions this would change:
- **ADR-0018 (D-18 voice):** "push-to-talk, hardware mute, no ambient audio, no audio files; embeddings ephemeral unless written enrollment; liveness-consistency purpose only."
- **ADR-0017 and ADR-0049:** "no stored face crops". Face templates are allowed only for people enrolled in M-16.
- **M-16 spec, the no-action rule:** a match produces no event, sound or notification. It is only logged for later review.

## Decision
1. **M-17 voice link, off by default.**
   - It is an add-on to M-16 personal mode. A person already enrolled in M-16 can also get a voice template.
   - Voices are matched only against enrolled people. Unknown voices are dropped at once: not stored, not counted.
   - It is a separate switch with its own typed opt-in, following ADR-0050: the owner decides and carries the risk.
2. **Listening mode.**
   - Chosen: **ambient listening** while M-17 is on and live mode runs. The microphone is open, gated by voice-activity detection. Audio is processed in short windows (about 1–3 s) in RAM and discarded.
   - The dashboard shows a "listening" indicator, and macOS shows its own microphone indicator.
   - This supersedes ADR-0018 for M-17 only. Everything else keeps push-to-talk and the liveness-only purpose.
   - Not chosen: **push-to-talk only**, where you press to ask "who is speaking?". This keeps ADR-0018's "no ambient audio".
3. **Storage.**
   - A voice template is the average of several speaker embeddings. It is encrypted with the person's own key, like face templates.
   - Raw audio is never written anywhere (ADR-0039 unchanged).
   - Deleting the person, withdrawing their consent, or turning M-16 or M-17 off erases the voice template, with verified wipe.
4. **Photo.**
   - Chosen: **one photo per enrolled person**, cropped to the face from that person's own best front enrollment frame. It is used only to show who was recognised.
   - Not chosen (could be added later): a photo file the user picks.
   - It is stored encrypted with the person's key, never exported, and deleted with the person.
   - This needs a new privacy-guard kind, `face_photo`, accepted for M-16/M-17 enrolled, consenting people only.
   - It supersedes "no stored face crops" (ADR-0017/0049) for those people only. Strangers' faces are never stored.
   - Not chosen: **no photo**. The card shows the name only.
5. **What a match does.**
   - Chosen: the local dashboard (127.0.0.1) shows a **"heard: NAME" card** for a few seconds, with the photo if decision 4 allows it.
   - The review log records "heard at PLACE / TIME", with 30-day retention.
   - There is no sound, no voice output, no push notification and nothing on the glasses or phone.
   - This amends M-16's no-action rule for the local dashboard only.
   - When the camera also matches the same person within a few seconds, the card says "voice and face agree".
   - Not chosen: **log only**, reviewed later, which keeps the no-action rule.
6. **Consent.**
   - Each person needs a consent record with scope `voice_enrollment`, plus `photo` if decision 4 allows it. Records are made with `oiris face-match consent` (typed YES).
   - Minors need the guardian scope, as M-16 already requires.
   - The opt-in notice adds voiceprints to its biometric-law warning.
7. **Model and licences.**
   - Runtime: **sherpa-onnx** (Apache-2.0). It is offline and needs no PyTorch.
   - Speaker model: **3D-Speaker CAM++ English**. Its code is Apache-2.0. The weights were trained on VoxCeleb, so **CC BY 4.0 attribution** applies; it is recorded in THIRD_PARTY.md.
   - The speaker model is downloaded on request with SHA-256 pins (like M-16 `fetch-models`), and is never bundled or committed.
   - Python runtime: sherpa-onnx 2.0.1, which has a Python 3.14 Apple Silicon wheel.
   - Microphone capture: sounddevice (MIT, bundles PortAudio).
   - Voice-activity detection: a simple energy-based detector first, so no model is needed. Silero VAD (MIT per the official project) is optional later, and its upstream LICENSE file is to be confirmed before any download.
   - Thresholds are provisional until measured.
8. **Mac permissions.** O Iris.app's launcher asks for Microphone access itself, with a usage string, as it does for Bluetooth and Location.
9. **Capture is device-independent, and phones are first-class.**
   - Face and voice enrollment and listening work from any source:
     - the Mac's camera and microphone;
     - an iPhone through Continuity Camera, whose camera and microphone appear as Mac devices;
     - the Android app over the existing encrypted EnrollmentPort;
     - a future iPhone app using the same platform-neutral protocol.
   - Voice samples from a phone use a new `voice_sample` op that mirrors the `shot` op: processed in RAM on the device running O Iris, buffer zeroed, never written.
   - The Mac webcam used in the 2026-11-15 check was only the device at hand.
10. **Tests and data.**
   - Tests use synthetic audio only.
   - No family voices, photos or faces ever go into the repository or CI; that is the owner's rule.
   - A live check, like the webcam check, reports counts and scores only.

## Where it will be enforced (when built)
- `oiris/modules/voice_link/` (new): voice enrollment, matching, and review entries.
- `oiris/guard/privacy.py`: new kinds `voice_template`, and `face_photo` if decision 4 allows it. Both are refused for anything but enrolled, consenting people.
- A macOS microphone adapter: voice-activity gated, RAM only, nothing written.
- `macos/launcher/main.swift`: the Microphone permission request.
- `oiris/modules/face_match/remote.py` (EnrollmentPort): the `voice_sample` op, and the Android enrollment screen's voice capture. The input device (Mac microphone or the iPhone's through Continuity) is selectable.
- `oiris/dashboard/server.py`: the "heard" card, if decision 5 allows it.
- Tests: synthetic audio, plus no-audio-on-disk and wipe-verification tests.

## Consequences
- Voiceprints are biometric identifiers under the same laws as face templates: Illinois BIPA, Texas, Washington, the EU/UK GDPR and others. The user decides and carries the risk (ADR-0050).
- With ambient listening (decision 2), the microphone hears everyone in the room while M-17 runs. Unknown voices are dropped at once, and both indicators show that the microphone is open.
- New optional dependency: sherpa-onnx (Apache-2.0), in an extra. Models are downloaded on request.
- Rebuilding the ad-hoc-signed app resets the Microphone permission too (ADR-0056).

## Changing it
Write a new ADR that supersedes this one, update CHANGELOG.md, and (if it touches a guardrail) get owner sign-off. A change to a frozen test threshold cannot be used to declare a pass retroactively.
