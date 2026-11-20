# M-17 Voice link: recognise enrolled people by voice (design, proposed)

Status: **design accepted** (ADR-0057, owner sign-off 2026-11-16); being built in the steps of section 7. The owner's choices: ambient listening, a photo from the person's best front enrollment frame, a dashboard "heard" card plus a log entry, and 3D-Speaker CAM++ run by sherpa-onnx.

## 1. Purpose
O Iris recognises the people the user enrolled, for example family members, by their voice and shows who it is, linked to their M-16 face enrollment.

It does **not**:
- identify strangers;
- store audio;
- send anything off the device;
- make a judgement about anyone. The vocabulary guard still applies.

## 2. What the user does
Every capture step below works from any source in section 3: the Mac's camera or microphone, an iPhone through Continuity Camera, or the phone app.

1. **Turn it on.** Type the opt-in phrase: `oiris voice-link enable`. The notice extends M-16's warning to voiceprints. M-16 personal mode must be on.
2. **Record consent** for each person: `oiris face-match consent --subject CODE --scope voice_enrollment` (plus `photo`). Guardian scope is required for a minor.
3. **Enroll a voice** for someone already enrolled in M-16: `oiris voice-link enroll --person ID`. The person reads 6 designed sentences, one per 7-second window, each with a cue for how to say it (see "Reading sentences" below).
   - Each window is checked for speech (VAD), length and level, then turned into an embedding in RAM.
   - A short microphone check comes first (section 7, step 6): it shows the level and stops with the permission fix when the input is digitally silent.
   - Feedback per window: seconds of speech, the total so far against the 20 s target, level, and the pitch ("pitch about 180 Hz (150-230)", or "pitch not measured").
   - When the windows hold less than `voice_link.enroll_min_total_speech_s` (20 s) of speech, more sentences follow in order, up to 10 windows.
   - After the last window an advisory pitch check runs, and one sentence may be offered for a redo (below).
   - `--speak`: the Mac reads each sentence aloud before its window (`/usr/bin/say`), for children who cannot read yet. Recording starts only after it has finished speaking.
   - Only the averaged template is stored, encrypted with that person's key.
   - The recordings themselves are never written.
4. **Photo:** the person's best front enrollment frame, cropped to the face, is kept encrypted with the person's key. It is shown only when they are recognised and deleted with them.
5. **Live.** While live mode runs with M-17 on:
   - listening is ambient: the microphone stays open while voice link runs, gated by voice-activity detection, and push-to-talk is also available;
   - each speech segment is embedded and compared with the enrolled voice templates;
   - a match shows a "Sounds like NAME" card on the local dashboard (with the photo if allowed), plays a chime and says "sounds like NAME" on this device (ADR-0058, "Announcing a match" below), and the review log records "heard at PLACE/TIME";
   - if the camera matched that enrolled person within `voice_link.agree_window_s` of the voice decision (default 5 s), the card says "Recent face match for this person". Another face can have appeared more recently. The review entry describes that same moment.
6. **Review and delete** with the existing commands: `oiris face-match review`, `delete-person`, and `disable` (verified wipe of face, voice and photo).

### Reading sentences
The sentences are pre-made, with specific words embedded to bring out a certain range of tone. They are read in this order (`oiris/modules/voice_link/prompts.py`). The first 6 are used by default (`voice_link.enroll_windows`, 3 to 10; past 8 the list repeats).

| id | Sentence | Cue (how to say it) | Tone | Brings out |
|---|---|---|---|---|
| calm-low | Mom hums a long, calm tune in the warm room at noon. | Calmly, a little lower than usual. | low | nasals (m n ng), back and open vowels, liquids, glides |
| bright-high | She sees three green leaves and six shiny seashells. | Light and bright, a little higher. | high | front vowels (ee), sibilants (s sh z), dental fricatives (th), liquids, plosives |
| question | Do you know how many blue boats are out on the bay today? | Ask it as a question; let your voice rise at the end. | rising | back vowels (oo), diphthongs (ow oh ay), nasals, plosives |
| count | One, two, three, four, five, six, seven, eight, nine, ten. | Count at your normal pace. | steady | fricatives (f v s th), diphthongs, nasals, glides |
| excited | Wow, that huge yellow kite is flying high above the park! | With excitement, a bit louder. | high | open vowels, diphthongs (ow eye), affricates (j), glides (y w), plosives |
| normal | The orange jar of cherries in the beige garage is very heavy. | Your normal speaking voice. | normal | affricates (j ch), voiced fricatives (zh v th), liquids |
| story-low | Joy's old wooden toy rolled with a soft thud across the floor. | Slowly and softly, like telling a quiet story. | low | diphthongs (oy oh), back vowels, dental fricatives, fricatives, liquids |
| cheerful | The black cat and her purple bird sat happily on the mat. | Cheerfully, at a normal pace. | normal | open front vowel (a in cat), r-coloured vowels (er, ur), plosives |

Why these sentences:
- **Phonetic coverage.** Together they cover the main sound classes, so the template is not built from a few repeated sounds.
- **Nasal resonance and hiss sounds.** Family voices differ most in the nose and mouth shape behind m/n/ng and in s/sh/th; the "calm-low" and "bright-high" sentences are dense with them.
- **Tone range.** Low, high, rising and steady readings put a calm voice and an excited voice into the same template, so either is still recognised later.

**Advisory pitch check (RAM only).** Each accepted window's pitch is measured (normalised autocorrelation, 60 to 500 Hz; `oiris/modules/voice_link/pitch.py`). After the last window, the median pitch of the "high" and "rising" sentences is compared with that of the "low" ones. If it is not at least `voice_link.enroll_range_min_rise` (default 8%) higher, one high or rising sentence is offered for a redo, "a bit higher and livelier" (Enter to redo, s to skip). With too little pitch data the check says "not measured". Either way the template is saved: the check never blocks enrollment, and matching is unchanged. Pitch values stay in RAM: never stored, never written to the chain log, never logged.

**Every capture device shows the same sentences.** The Mac CLI uses them now; the phone app will show the same list (`prompts.prompt_list()`: id, text, cue, tone) and send the sentence id with each `voice_sample`.

## 3. Capture sources: device-independent, phones first-class
The Mac's built-in webcam was used for the first live check only because it was the device at hand. Enrollment and listening do **not** depend on it. O Iris treats every camera and microphone as a capture source, and phones are first-class.

| Source | Face | Voice | Status |
|---|---|---|---|
| **Mac camera and microphone** | `--camera auto` or an index | default input device | Face works (live check 2026-11-15); voice in build step 3 |
| **iPhone through Continuity Camera on the Mac** | The iPhone's camera appears as a Mac camera (`--camera auto` or an index) | macOS lists the iPhone's microphone as an input device ("<name>'s iPhone Microphone") | Works today with no phone app; choose the device |
| **Android app** (EnrollmentActivity) | Over the existing encrypted EnrollmentPort (Noise over BLE): the phone captures and guides, the pod checks and builds the template | New `voice_sample` op, same pattern (below) | Face built, never run on a phone; voice to add |
| **iPhone app** | Same EnrollmentPort protocol (canonical JSON over Noise; platform-neutral) | Same `voice_sample` op | Not built (an iOS client is future work) |

**The `voice_sample` op** on EnrollmentPort, mirroring the existing `shot` op:
- The phone sends a short clip: 16 kHz mono 16-bit PCM, at most 8 s, base64, fragmented over Noise frames like frames are.
- The pod or Mac checks it (speech present, length, level), embeds it in RAM, adds it to the person's open capture, and **zeroes the buffer**.
- The clip is never written, logged or returned. Raw audio never exists on disk on either side (ADR-0039).
- The phone shows the same guidance and ticks as for face shots, and the same sentences and cues as the Mac (section 2, "Reading sentences").
- As built, the ops, limits and Kotlin flow are in "Phone-guided voice enrollment" in [oiris/link/README.md](../oiris/link/README.md#phone-guided-voice-enrollment-adr-0057).

**Listening from a phone** (later): the phone can be the live microphone, as with Continuity on a Mac, or the Android fallback node over the link. The same rule holds: audio in RAM only, embeddings compared on the device running O Iris. A later option is on-device embeddings on the phone; sherpa-onnx has Android and iOS builds, and only the embedding would cross the link.

## 4. Components
| Piece | What | Notes |
|---|---|---|
| Microphone adapter (macOS) | 16 kHz mono capture through CoreAudio (sounddevice, MIT) | RAM ring buffer of a few seconds; nothing written; health shows "listening" |
| Voice activity | Energy-based detector first (numpy, no model); Silero VAD optional later | Drops non-speech before any embedding. Silero is MIT per the official project; its upstream LICENSE is to be confirmed before any download |
| `speaker_embed` slot plugin | sherpa-onnx (Apache-2.0) running 3D-Speaker CAM++ English | Weights CC BY 4.0 via VoxCeleb (attribution); downloaded on request; SHA-256 pinned |
| `oiris/modules/voice_link/` | Enroll, match, review entries, wipe | Mirrors M-16: per-person key, verified wipe, consent re-check at load |
| Privacy guard | New kinds `voice_template` and `face_photo`  | Refused for anyone not enrolled and consenting |
| Launcher | Microphone permission request | Usage string; macOS shows the prompt naming O Iris |
| Dashboard |  "Sounds like NAME" card; listening indicator | 127.0.0.1 only; no push |
| Announcer | chime and "sounds like NAME" on this device (ADR-0058), plus where the voice came from when a microphone array can tell (ADR-0065) | `announce.py`; afplay/say on macOS, chime + Flite elsewhere |
| Voice direction | one direction word for a clear match, or none (ADR-0065) | `direction.py`; `oiris/features/doa.py` per 100 ms speech window; only with a rig `mic_array` mount |

## 5. Matching
- Similarity between a segment's embedding and each enrolled template is the cosine score.
- A match needs a score at or above `voice_link.match_threshold` and at least 1.5 s of speech.
- The threshold is **provisional**: set from the model's published operating point, then measured in a consented live check.
- Unknown voices are dropped and not counted.
- Several people talking at once lowers confidence. Overlapping speech is reported as "unsure", never guessed.

### Live listening
Built in step 3 (`oiris/modules/voice_link/capture.py`, `listener.py`; wiring in `oiris/live.py`). It runs **only** in live mode (`oiris run`) and only when the M-16 and M-17 flags are both on; otherwise nothing of M-17 is loaded.
- **Capture.** `capture.py` is the one place that opens a microphone: a continuous 16 kHz mono float32 PortAudio stream (sounddevice, imported lazily) on `voice_link.input_device` (index, name substring, or the system default). Each block is copied for the listener and the stream's own block zeroed. Another backend (for example Apple's voice-processing input) can be added there without touching the listener. Enrollment windows use the same module (`capture.record`).
- **Utterance cutting.** Blocks go into a bounded RAM queue (about 10 s; a full queue drops blocks). A worker thread gates 30 ms frames on energy against a slowly adapting noise floor (the gate is `max(floor + vad_snr_db, vad_floor_dbfs)`). An utterance starts at the first loud frame, with about 0.2 s of pre-roll, and ends after about 0.7 s of quiet or at 8 s (a longer stretch continues as the next utterance). Utterances with under 0.3 s of gated speech (clicks) are dropped unprocessed. All numbers are named constants in `listener.py`.
- **Matching.** Each utterance goes to `process_audio` under the live store lock, and not at all once the store is closed or the listener stopped. The core then applies its own VAD, `min_speech_s`, threshold and "unsure" margin (section 5).
- **RAM only.** Every block, frame, pre-roll and utterance buffer is zeroed after use, and the queue is zeroed at stop. Nothing is written except the core's "heard" entry for an enrolled person.
- **The card.** A heard person becomes a card (name, score, face agreement, time) kept in RAM for `voice_link.card_seconds`; the dashboard shows "Sounds like NAME", "Recent face match for this person" when it applies, and how many seconds ago. Only heard people are counted; unknown voices and close calls leave no card, no count and no log.
- **Push-to-talk.** With `listening: "push_to_talk"` the microphone stays closed until the dashboard's Listen button (`POST /voice/listen`, the same token, Host and same-Origin rules as Quit). It then opens for about 8 s of audio and closes again.
- **Digital-silence notice.** macOS delivers exact zeros when the microphone permission is denied. After about 3 s of exact zeros the dashboard says "no sound from the microphone: check System Settings > Privacy & Security > Microphone, or the input device". A quiet room is never exact zeros, so it does not trigger the notice.
- **Never fatal.** If the voice link is not opted in, a model or `sounddevice` is missing, the device is unknown or the microphone cannot be opened, the dashboard shows "Voice link unavailable: reason - what to do" and live mode keeps running.

### Announcing a match
Built per ADR-0058 (`oiris/modules/voice_link/announce.py`; wiring in `oiris/live.py`). A voice match is a hint, not proof: a phone call, a video or a recording of an enrolled person can match. So O Iris says "sounds like NAME", never the bare name.
- **What plays.** A clear match (section 5) plays a short chime, then says "sounds like NAME" through this device's speaker. Unknown voices, close calls, the self person and anti-spoof drops play nothing.
  - macOS: `/usr/bin/afplay /System/Library/Sounds/Glass.aiff`, then `/usr/bin/say PHRASE`. Argument lists, absolute paths, each with a timeout.
  - Other hosts and the glasses: a generated two-note chime through `sounddevice` when present, and the phrase through the M-06 `tts` port (Flite) when installed; otherwise the chime only.
- **Setting.** `voice_link.announce`: `chime_and_name` (default), `chime` or `off`. `voice_link.announce_repeat_s` (default 60, range 5 to 3600): the same person is announced at most once per that time; different people can follow each other. The job queue is bounded (a burst beyond it is dropped).
- **The name rules.** The name is spoken only when it passes the vocabulary guard, has at most 4 words and 40 characters, and uses only letters of one script, spaces, hyphens and apostrophes. Mixed scripts are refused so a look-alike letter cannot slip a word past the guard. Otherwise the chime plays alone. The phrase is at most 6 words, within M-06's 8-word budget.
- **No feedback loop.** While a job waits or plays, and for 0.4 s after (`TAIL_S`), the listener zeroes and drops every microphone block.
- **Never fatal.** A missing sound device or failed command is logged; listening goes on.
- **Direction (ADR-0065).** When the microphones can tell, the phrase adds where the voice came from: "sounds like NAME, on your left". The rules are below, under "Saying the direction".
- **Preview.** `oiris voice-link test-sound [--name NAME]` plays the chime and the phrase once with the same name rules. It opens no store and records nothing.
- **Self.** `oiris voice-link set-self --person ID` marks one enrolled person as self: the wearer on glasses, the owner at the Mac. It is stored as `vl_state` key `self`. A clear match of self is dropped like an unknown voice: no heard entry, no card, no count, no sound. `clear-self` removes the mark; so do `disable`, a new opt-in, and deleting that person's voice. `status` says only whether a self person is set (yes/no), never who.
- **Anti-spoof hook.** When `plugins.slots.voice_antispoof` is set (ADR-0055), a clear match's speech is scored by that slot before anything is logged. A score at or above `voice_link.spoof_threshold` (default 0.5, range 0.05 to 0.95) drops the match without a trace. No slot, or a failed slot, lets the match go ahead. No anti-spoof model ships yet (licence terms, STATUS.md), and the opt-in notice says a recording or phone call can trigger an announcement.

### Saying the direction
Built per ADR-0065 (owner decision 2026-11-17; `oiris/modules/voice_link/direction.py`, wiring in `capture.py`, `listener.py`, `announce.py`, `oiris/live.py`). Synthetic signals only so far.
- **Multi-channel capture in live mode.** When the `rig` config has a `mic_array` mount (ADR-0060), live mode opens the voice microphone with that many channels (`SoundDeviceCapture(channels=N)`, blocks of shape (n, N), channel order = the mount's `mic_positions_m`). The microphone positions go to the head frame through the mount's position and yaw/pitch/roll. The first `mic_array` mount is used. Without one (the Mac today) the capture is mono and nothing changes.
- **Fallback.** A device with fewer input channels, or a stream that cannot open N channels, falls back to mono with a plain log line ("could not open N channels ... listening in mono") and no direction. The listener never fails because of it.
- **Matching is unchanged.** VAD, embedding and match use channel 0 (`REF_CHANNEL`) only; averaging delayed channels would smear the voice. A parallel RAM-only multi-channel buffer and the per-frame speech flags follow the mono buffer's lifecycle and are zeroed after every utterance, matched or not.
- **Only after a clear match.** The direction is computed only once `process_audio` returns a heard person. Never for unknown voices, close calls, self, a spoof-vetoed match, or while input is paused.
- **The estimate.** 100 ms windows (`WINDOW_S`) that the voice gate marked as speech throughout; `doa()` on each; a window counts only at or above `doa.MIN_CONFIDENCE`. A front/back-ambiguous result (a temple pair) counts only when every candidate gives the same word (60 degrees: "at left"; 30 degrees: no word). The word needs at least `MIN_WINDOWS` (3) confident windows and a share of at least `MAJORITY_SHARE` (0.6). Otherwise none. "above"/"below" only from a non-coplanar array.
- **What is spoken.** "sounds like NAME, PHRASE": ahead of you, on your left, on your right, behind you, above you, below you (`announce.DIRECTION_PHRASES`). When name plus direction would pass M-06's 8 words, the name alone. `check_text` checks the full text first; a refusal falls back to the name-only phrase, which is checked too. A direction is never spoken without the name. `chime` and `off` are unchanged. The repeat limit is per person: a new direction does not re-announce within `announce_repeat_s`.
- **Shown, never kept.** The card and the `on_heard` payload carry `direction` (a word or None); the dashboard shows "Sounds like NAME, on your left", escaped. `vl_heard` and every other record never contain a direction.

### Android parity
Wherever the core ends up running on the glasses (ADR-0059), the Android side must give the same answers as this Python reference. `android/core` `VoiceLink.kt` (pure Kotlin, no Android APIs) ports:
- the decision (`core.decide`, which `process_audio` uses): threshold, then the "unsure" margin, then self, then anti-spoof; the spoof score is asked for only at that last step;
- the cosine score and the unit-mean voice template;
- the name rules (`speakable_name`, `phrase`), with the banned words from the shared `schema/vocabulary.json`;
- the per-person repeat limit;
- the energy VAD (`vad.detect`): segments in samples and speech seconds.

`scripts/make_android_vectors.py` writes `tests/vectors/voice_link.json` by calling the Python functions. The VAD cases store generator parameters (sums of sines with piecewise gains, frame-aligned, at least 1 dB from the threshold), not audio. `:core:check` expects every case. `tests/test_voice_link_vectors.py` checks that the file is current and that `decide` gives `process_audio`'s outcomes.

Not ported yet: the speaker model on Android (sherpa-onnx has an Android build; the model is side-loaded later, ADR-0059), the listener (capture, utterance cutting) and the announcer's player and queue. Script names come from the Unicode tables of the JVM (Unicode 13 on JDK 17) and of Python (`unicodedata`). Characters assigned after Unicode 13 may be judged differently.

## 6. Privacy and consent (unchanged rules first)
- **Unchanged:**
  - raw voice is never stored (ADR-0039);
  - templates exist only for enrolled, consenting people (ADR-0049, extended to voice);
  - consent records are durable and withdrawal erases them;
  - guardian consent for minors;
  - nothing leaves the device;
  - everything stays off by default.
- **Changed if signed off:**
  - ambient listening for M-17 only (supersedes ADR-0018 for M-17) ;
  - one photo per enrolled person (supersedes "no stored face crops" for them only) ;
  - a local "heard" card (amends M-16's no-action rule for the local dashboard) .
- **Owner's rule:** no family voice, photo or face ever goes into the repository, tests or CI. Tests use synthetic audio, such as generated tones, noise and synthetic speech features.

## 7. Build steps (each through the workbench: checks, audit, review)
1. **Model slot plugin.**
   - sherpa-onnx in an optional extra.
   - `voice-link fetch-models`: pinned checksums, licence lines in THIRD_PARTY.md, and the same certificate handling as M-16.
   - Synthetic tests.
   - *Accept:* embeddings of the right size for synthetic input; checksum and licence enforcement.
2. **Enrollment and matching core** (`oiris/modules/voice_link/`), the privacy-guard kinds, wipe, and consent scopes.
   - *Accept:* enroll and match synthetic voices; no-audio-on-disk test; wipe verified; consent required.
3. **Microphone adapter, launcher permission and phone capture.**
   - The input device is selectable: the Mac microphone, or the iPhone's through Continuity.
   - The `voice_sample` op is added to EnrollmentPort, and voice capture to the Android enrollment screen.
   - Mic capture in live mode with VAD, and the launcher's Microphone request.
   - *Accept:* the macOS prompt names O Iris; nothing is written; a "listening" health state.
   - *Status:* mic capture in live mode with utterance cutting, the selectable input device, the dashboard "heard" card, push-to-talk and the digital-silence notice are built and tested with synthetic audio and a fake capture (section 5, "Live listening"). Since then the CLI enrollment and a near-field listening test have run once on a real built-in microphone (step 6). The launcher's Microphone request is built (it asks only when the app config turns M-17 on) but not yet observed in a built app. The `voice_sample` op and Android capture are not built yet.
4. **Dashboard card and review entries** , plus the voice-and-face agreement.
   - *Accept:* the card shows and expires; log entries carry 30-day retention.
5. **Photo**  (if signed off).
   - *Accept:* encrypted, deleted with the person, never exported.
6. **Live check** with the owner, and with family members who consent, as in the webcam check.
   - Counts and scores only, recorded in STATUS.
   - **Loudspeaker-replay test:** with the enrolled person's consent, play a recording of their voice through a loudspeaker near the microphone, and record whether it is announced (and, once an anti-spoof slot exists, whether it drops it). Delete the recording afterwards; it never enters the repository.
   - *Run once* (2026-11-16): one consenting enrolled user, a Mac laptop's built-in microphone, the CLI enrollment, then a near-field listening test. Counts and scores only; no audio was kept.
     - Enrollment: six windows of 7 s. Window 1's first try was refused with 0.0 s of speech at -54 dBFS: the macOS microphone permission prompt opened as recording started and the window ran out. The accepted windows had 1.8, 4.3, 2.3, 3.1, 2.9 and 4.0 s of speech (about 20 s in total) at -34 to -38 dBFS. The range check flagged a narrow range; one redo of the excited sentence widened it.
     - Near-field listening, four utterances: 3.5 s scored 0.44 and 1.8 s scored 0.15 (both dropped, below the provisional 0.5 threshold); 2.3 s scored 0.88 and 2.2 s scored 0.93 (both heard).
     - One manual check on one person: not a study, and no accuracy claim. The distance and other-voice phases (and the loudspeaker-replay test) are still to run.
   - *Changes after it* (tested with synthetic audio, a fake capture and a fake speaker; `tests/test_voice_link_tighten.py`):
     1. **Microphone check before window 1** (`capture.mic_check`, about 1.5 s, `MIC_CHECK_S`): opens the input so macOS asks for permission before any real window, and shows the level in dBFS. Digital silence (exact zeros) stops enrollment with the fix: System Settings > Privacy & Security > Microphone, allow the app running O Iris (for example Terminal), or choose another input with `--input`. Below -60 dBFS (`MIC_QUIET_DBFS`) it says "move closer or speak up" and goes on. The check's audio is zeroed and never kept.
     2. **Total-speech target**, enforced in the core so every capture device follows it: `voice_link.enroll_min_total_speech_s` (default 20 s, 5 to 120). After `enroll_windows` good windows, if the total is short, `finish_voice` says how many seconds are still needed and refuses for now; the CLI adds the next sentences in order (the list cycles) until the target or 10 windows (`VOICE_ENROLL_MAX_WINDOWS`). At 10 windows the template is saved with a note that it is below the target. Each window's feedback shows the total, for example "total 12.4 of 20 s".
     3. **Matching minimum** `voice_link.min_speech_s` 2.0 s (was 1.5): the 1.8 s piece scored 0.15, the 2.2 s and 2.3 s pieces scored 0.93 and 0.88. Length is not the only factor (the 3.5 s piece scored 0.44). Provisional until more live checks.

## 8. Open questions
- The match threshold and minimum speech length on this Mac's microphone: measure in step 6 (one near-field run so far; the minimum is now a provisional 2.0 s, the threshold still the provisional 0.5).
- Whether push-to-talk should stay available even with ambient listening on: proposed yes.
- How long the "heard" card stays on screen: proposed 10 s.
