# Evidence: what has worked, and what is not yet proven

This page sorts the project's evidence into three kinds and never mixes them:

1. **Seen working on real hardware or real data.** Someone ran the code on a real device, or on real recordings, and recorded what happened. Each entry is a single check, not a study, so no accuracy is claimed.
2. **Proven by automated checks.** Tests that run on every change. They show that the code does what its specification says on synthetic, simulated or replayed inputs. They cannot show that a real sensor behaves the way a simulation assumes.
3. **Built and tested, not yet proven.** Code that passes its tests against simulations and placeholder numbers but has never run on its target hardware (glasses, helmet, Pi rig, phone), plus the research hypotheses the project exists to test.

The numbers are from the release that added this page (2026-11-17); `pytest` and `gradle :core:check` print the current ones. Row-by-row detail is in [STATUS.md](STATUS.md). The glasses and helmet parts are mapped in [HARDWARE_READY_PROGRAM.md](HARDWARE_READY_PROGRAM.md).

## 1. Seen working on real hardware or real data

| What | Setup | What was seen | What it does not show |
|---|---|---|---|
| Live mode in O Iris.app | An Apple Silicon Mac (macOS 26.5), 2026-11-15, after the user allowed Bluetooth and Location for O Iris | The app starts natively (arm64), installs its bundled wheels offline, unlocks the encrypted store from the passphrase dialog, and shows live Bluetooth LE and Wi-Fi readings on the local dashboard. Wi-Fi comes from macOS's cached scan results; the system's Wi-Fi log showed no scan requests from O Iris. Bluetooth addresses and Wi-Fi network IDs are hashed before anything is stored. | One Mac in one place. A Mac feeds one signal family, so the alert level stops at FYI. No detection claim. |
| M-16 face matching (opt-in) | A Mac's built-in camera, one consenting enrolled user, 2026-11-15 | The first run found four bugs: a certificate error when fetching the model, up/down head poses a laptop camera could not reach, the wrong camera picked when a phone's Continuity Camera is present, and no command to record consent. All four were fixed and the check was run again. The camera was picked automatically and five-angle enrollment completed. Facing the camera, 20 of 20 frames matched (confidence 0.929 to 0.943, threshold 0.8). With the camera covered, 0 of 20 frames showed a face. | One person, one session, one room. No other faces were tried, so there is no false-match rate. The automated tests use synthetic frames only. |
| M-17 voice link (opt-in) | A Mac laptop's built-in microphone, one consenting enrolled user, enrollment from the command line, 2026-11-16 | Enrollment completed: six 7 s sentences, about 20 s of speech in all. In a near-field listening test, 2 of 4 utterances were recognised (scores 0.88 and 0.93). The other 2 scored below the provisional 0.5 threshold (0.44 and 0.15) and were dropped. The check led to three changes: a microphone check before the first sentence, a 20 s total-speech target, and a 2.0 s minimum per utterance. No audio was kept. | Not yet run at a distance, with other people's voices, or inside the app. The 0.5 threshold is provisional. |
| M-01 blink detection | Two public Tobii Pro Glasses 2 recordings (LaborTrack, CC BY 4.0 / MIT, recorded with consent). CI fetches and sha256-checks them on every change; they are never stored in the repository | Scored against the eye tracker's own blink events: precision 0.63 and 0.62, recall 0.86 and 0.83 (blinks of 50 to 500 ms). | Not the O Iris eye camera: no real video has been through the eye-camera pipeline yet. |
| Spoken output | CI on Linux, with real Flite speech synthesis | Synthesised speech has the expected length and level; barge-in stops a real playing process. | Not yet heard on the Pi's speaker or on glasses. |
| Builds | GitHub Actions on every push and pull request | Linux tests on Python 3.11, 3.12 and 3.13. A macOS job runs the Android core checks, builds the debug and unsigned release APKs, and builds the ad-hoc signed Mac app. | The APK has never been installed on a phone or an emulator. The Mac app is not Developer ID signed. |

## 2. Proven by automated checks

Every change passes these before it merges: first in the Architecture Workbench gate on the development Mac, then in CI on GitHub.

- **Lint, types and licences:** `ruff`, `mypy` and `scripts/license_check.py`.
- **Python suite:** 1808 tests in 70 files. On the development Mac, 1802 pass and 6 are skipped because they need extras that are not installed there: Flite, the eye recordings, the compiled Mac gate helper and the face-model weights. CI installs Flite and OpenCV and fetches the eye recordings, so those tests run there. The gate-helper test runs in the macOS build job. Only the face-model test, which needs downloaded weights, is skipped everywhere.
- **Headless golden runs:** five simulated scenarios (`quiet`, `camera_lens`, `warm_prop`, `fixed_emitter_drift`, `replay_screen`) must reproduce their expected JSON exactly, and the synthetic seed set must match its generator.
- **Android core:** 2509 JVM checks (`gradle :core:check`). The Kotlin code replays vectors that the Python code generates in `tests/vectors/`. They cover the Noise handshake, BLE framing, the enrollment protocol, eye features, thermal rendering, the voice link, voice enrollment, the glasses display list, the HUD painter, clock sync, link operations, fusion and the power governor. Numbers must agree within 1e-6 and everything else exactly; the Noise handshake is byte-identical. `scripts/make_android_vectors.py --check` fails when a vector file is stale.

| Area | Tests | What the tests establish |
|---|---|---|
| Runtime, live mode and dashboard | 173 | The sensor bus with fault isolation and staleness limits, live sessions and sightings, the network guard, the dashboard's Host and Quit rules, and per-stage timing |
| Guards, privacy, security and the encrypted store | 137 | The vocabulary, consent, privacy, retention and radio guards; encryption at rest; the hash chain and its anchors across sessions; the clip recorder's refusal of person imagery; the shared JSON contracts |
| Signals and fusion | 67 | Feature extractors on synthetic signals, blink detection on the two real recordings above, the fusion stack and its evidence rule, Wi-Fi AP scoring, the golden seed data |
| macOS app and sensors | 73 | Bluetooth and Wi-Fi adapters with fakes, permission states, the app bundle's structure |
| M-16 face matching and the live camera | 140 | Opt-in, enrollment, matching, review and verified wipe; the enrolled-person photo; the camera's gating order |
| M-17 voice link and phone voice enrollment | 311 | Enrollment and matching with a deterministic fake speaker model, live listening and push-to-talk, announcements, direction phrases, the phone enrollment operations |
| Spatial: 360-degree rig, calibration, sound direction, clock sync | 459 | Head-frame geometry and merge rules, calibration on synthetic renders, direction of arrival on synthetic plane waves, clock offset and drift |
| Glasses output: display, drawing, output arbiter, motion | 183 | The display list and painter, output priorities and barge-in, the motion gate |
| Power, heat and latency budgets | 129 | Governor levels with hysteresis, duty plans, the simulated battery and heat, budget warnings |
| Phone and glasses link | 60 | BLE framing, unlock backoff, panic, the signed wipe, pinned side-load |
| Recording light, mute and capture interlock | 76 | Capture only with the light confirmed on and the mute open; the mute always wins |
| **Total** | **1808** | |

These checks do not show that a real sensor, room, voice, face or battery behaves like the simulations and fakes the tests use.

## 3. Built and tested, not yet proven

### Glasses and helmet parts

Open-platform Android glasses come first, then possibly a helmet ([ADR-0059](adr/0059-code-first-build-order.md)); none has been chosen yet. These parts are built and pass the checks above, but none has run on glasses, a helmet, a phone or a Pi.

| Part | What the code is checked against | Still to prove on the device |
|---|---|---|
| 360-degree rig ports and merge | A simulated rig | Real mounts and real timing |
| Direction words and cues | Simulated detections | That the cue is clear on the glasses |
| Mount calibration | Synthetic renders: intrinsics within 0.03 %, angles within 0.05 degrees, positions within 0.6 mm | A real camera and a printed board |
| Sound direction | Synthetic plane waves | A real microphone array in real rooms |
| Voice-link direction | Synthetic multi-channel audio | A real array (a Mac has one microphone) |
| Clock sync | Loopback and its Kotlin twin | A real radio link; live mode does not use it yet |
| Power, heat and latency budgets | A closed-loop battery and heat simulation | The glasses' real power and heat coefficients, which are placeholders now |
| Phone link operations | Loopback and their Kotlin twins | A real phone over Bluetooth |
| Recording light, mute and interlock | Simulated ports | Real wiring, including that a broken mute line reads as muted |
| Resumable sighting chain | Tests | A run in the app |
| Wi-Fi AP scoring | Synthetic BSSIDs | A run in the app |
| Enrolled-person photo | Synthetic frames | A real enrollment |
| Phone-guided voice enrollment | Loopback and its Kotlin twin | A phone; the Android microphone screen is not built |
| Kotlin parity set | JVM checks | A phone |
| Glasses drawing layer | The Python reference painter and the Kotlin painter | A real display; the Android canvas classes are compiled only by CI |
| Live camera for M-16 | Fakes | A run in the app |

Each part's placeholders and device check are in [HARDWARE_READY_PROGRAM.md](HARDWARE_READY_PROGRAM.md).

### Research hypotheses

- No study shows that this stack detects a disguised humanoid machine, and nothing here has been tested against one.
- The eye-behaviour (M-01) and liveness (M-02) modules are hypotheses. They ship disabled until their experiments pass on real data.
- The fusion weights are a hand-set prior. There are no trained weights yet, because there is no real golden data.
- Every threshold in the test plan is a proposal, not a measured result.

## Test data and privacy

- All test inputs are synthetic, simulated or generated by code in this repository. The only real recordings used are the two public eye-tracker recordings above, which CI fetches and which are never committed.
- The repository holds no real faces, voices, photos, or Wi-Fi or Bluetooth captures. The live checks in section 1 ran on the owner's own machine; only their counts and scores are recorded here.
- Before each push, every new commit (files and messages), the whole tree and the pull-request text are scanned for the owner's network, Bluetooth, device and personal identifiers. The scan masks the values in its output, and nothing is pushed until it finds none. Commits use the project's no-reply identity.
