# O Iris — Anomaly Signature Bench

An open-source **research** rig and Android operator app that combines passive sensors (thermal, near-IR camera, eye camera, BLE/Wi-Fi, mmWave radar, microphone, GPS) and reports **anomaly signatures** such as "lens-glint candidate" or "known emitter changed vs its baseline", each with its common benign causes.

It is built for one question: *can a cheap, passive, consent-based sensor stack flag unusual devices, emitters and liveness inconsistencies (for example a hidden camera, a photo or screen replay, or a mannequin) in a room or at close range?*

> **Honest limits — read first**
> - **No study shows this detects a disguised humanoid machine.** Nothing here has been validated against one. Every threshold in the test plan is a proposal, not a benchmark result.
> - The **eye-behaviour (M-01)** and **liveness (M-02)** modules are **hypotheses**. They ship **disabled by default** until their experiments (E-11/E-14, E-15/E-16) pass on real data. Micro-saccades cannot be resolved with this hardware at a distance and are report-only.
> - A **silicone mask on a live person** keeps real eyes, pulse and voice; this stack does not catch it.
> - A heated mannequin is expected to defeat thermal checks.
> - The output **never labels a person**. There is no "robot" or "human" verdict anywhere in the code, by design (see `oiris/guard/vocabulary.py`).
> - No accuracy claims are made. Results are research signals.

## What has worked, and what is not yet proven

- **Seen working on real hardware or real data:**
  - Live mode in the Mac app reads real Bluetooth LE and Wi-Fi.
  - M-16 face matching ran on a Mac camera with one consenting user: 20 of 20 frames matched, and 0 of 20 with the camera covered.
  - M-17 voice link enrolled one consenting user's voice on a laptop microphone and recognised 2 of 4 near-field utterances. That check led to stricter defaults.
  - Blink detection is scored on two public eye-tracker recordings in CI.

  Each of these is a single check, not a study.
- **Proven by automated checks on every change:**
  - 1808 Python tests (6 need optional extras and skip without them).
  - Five headless golden runs.
  - 2509 Android JVM checks against vectors generated from the Python code.
  - CI also builds the Android APK and the Mac app.
- **Built and tested, not yet proven:** every part made for the glasses and helmet, including the 360-degree rig, calibration, sound direction, power and heat, the phone link, the recording light and mute, and the drawing layer. None has run on glasses, a helmet or a phone. The detection ideas themselves (eye behaviour, liveness, humanoid machines) are untested hypotheses.

Details and the limits of each check: [docs/EVIDENCE.md](docs/EVIDENCE.md). Row-by-row status: [docs/STATUS.md](docs/STATUS.md).

## What's in the repo

| Path | What it is |
|---|---|
| `oiris/` | Python 3.11+ runtime: one process, one thread per sensor, scores in [0,1], one fusion layer |
| `oiris/adapters/` | Sensor port (C2): real drivers (MLX90640, Lepton/PureThermal, Pi NoIR, BLE via bleak, Wi-Fi via `iw` on Linux and via the O Iris.app launcher's feed on macOS, GPS/NMEA, MR60BHA2 radar, RTL-SDR probe, Android node) + replay + simulated backends |
| `oiris/scores/`, `oiris/features/` | Score port (C1) and signal processing for the 11 scores |
| `oiris/fusion/` | Six signal families, logistic-regression stack, missing-family mask, ≥2-family evidence rule |
| `oiris/guard/` | **Required** guards: vocabulary, consent, privacy, retention, phrase whitelist, radio (no monitor mode, no transmit tools) |
| `oiris/security/` | AES-256-GCM at rest, Noise_XX rotating-key link, HMAC hash chain + Ed25519 anchors, constant-time compare, crypto-erase, panic gesture, 10-fail wipe |
| `oiris/store/` | Encrypted SQLite with versioned migrations; sighting log (GPS, map pins) and sweep timeline |
| `oiris/slots.py`, `examples/plugins/` | Replaceable model slots: swap or remove voice recognition, face match, eye and fusion models with one config line ([docs/PLUGINS.md](docs/PLUGINS.md)); no weights bundled |
| `oiris/modules/` | Voice tiers, golden dataset + feedback loop, fingerprints, liveness stack, app gate |
| `oiris/headless/` | Headless mode: replay, virtual clock, no network, LED stub |
| `oiris/live.py` | Live mode (`oiris run`, the default command): real adapters, encrypted session record, network guard |
| `oiris/dashboard/` | Minimal local dashboard (127.0.0.1 only); in live mode it shows live data and a Quit button |
| `android/` | Kotlin companion app: BiometricPrompt Class 3 gate bound to a Keystore key, SQLCipher, offline TTS tiers, fallback camera/BLE node, no INTERNET permission |
| `schema/` | Shared JSON contracts (Python and Kotlin read the same files) |
| `golden/` | Golden expectations for the headless suite (data stays out of git) |
| `macos/` | Swift sources for O Iris.app: the launcher (permission requests, Wi-Fi feed) and the M-15 Keychain gate helper |
| `scripts/` | Build (`build_all.sh`, `make_macos_app.py`), licence audit, table and test-vector generators, Pi LUKS2 setup |
| `docs/` | Architecture, live mode, macOS app, status, evidence, the hardware-ready program, extending, upgrading, ADRs, and the v2 build spec and module map PDFs. Ethics is in the root [ETHICS.md](ETHICS.md) |

## Quick start (no hardware needed)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                                   # full headless suite
oiris --headless --scenario camera_lens    # simulated rig run, JSON out
oiris --headless --scenario replay_screen --steps 3 --expect golden/expected/replay_screen.json
oiris check-phrases                        # every voice/display phrase passes the guard
oiris dashboard                            # http://127.0.0.1:8765
```

Scenarios: `quiet`, `camera_lens`, `warm_prop`, `fixed_emitter_drift`, `replay_screen` (lab proxies, never people).

Hardware drivers: `pip install -e ".[hw]"` on a Raspberry Pi 5. Parts list, wiring and IR eye-safety rules are in `docs/spec/ASB-Build-Spec-v2.pdf`.

## Live mode and the macOS app

`oiris run` (or plain `oiris`) runs against real local sensors. It asks for the store passphrase (or reads `OIRIS_PASSPHRASE`), creates or unlocks the encrypted store, and serves the dashboard on http://127.0.0.1:8765. On a Mac it reads two things:

- **Bluetooth LE** advertisements, through bleak and CoreBluetooth.
- **Wi-Fi** networks that macOS already found in its own scans (`CWInterface.cachedScanResults()`), read by the O Iris.app launcher every 15 s. `oiris run` started from a terminal has no Wi-Fi source on macOS.

```bash
pip install -e ".[mac]"     # bleak, and the pyobjc packages it needs on macOS
oiris run
```

CI builds `o-iris-<commit>-macos-app-adhoc-signed.zip`. The app runs only on **Apple Silicon with macOS 14 or later** and needs Python 3.11–3.14 installed (the bundled wheels cover those versions). It asks for Bluetooth and for Location (When In Use). Location is needed only because macOS hides Wi-Fi network IDs without it; O Iris never reads a position. The app is ad-hoc signed, not Developer ID signed, so Gatekeeper blocks the first open on other Macs, and macOS asks for both permissions again after every rebuild.

Radio use is normal client-radio activity, as the spec allows: scans can send scan requests (bleak scans actively, and macOS has no passive BLE mode). O Iris never connects to or pairs with others' devices and uses no monitor mode, injection, deauthentication or jamming. A Mac alone feeds one signal family (rf, from BLE), so the alert level can reach FYI but not higher.

Details: [docs/LIVE.md](docs/LIVE.md) and [docs/MACOS.md](docs/MACOS.md).

## Optional modules (off by default)

- **M-16 face matching:** recognises people the user enrolled, with their consent, by face, using OpenCV Zoo models downloaded on request. It needs its flag plus a typed opt-in. Run once on a Mac camera with one consenting user; the automated tests use synthetic frames only. Spec: [docs/FACE_MATCHING_MODULE_SPEC.md](docs/FACE_MATCHING_MODULE_SPEC.md).
- **M-17 voice link:** recognises people the user enrolled, by voice, and announces a clear match with a chime and "sounds like NAME" (a hint, not proof). With a microphone array it adds the direction ("on your left"). It is an add-on to M-16 face matching, needs its flag plus a typed opt-in, and writes no raw audio. Run once on a real voice (one consenting user, near field, from the command line); not yet run at a distance, with other people's voices, or in the app. Spec: [docs/VOICE_LINK_MODULE_SPEC.md](docs/VOICE_LINK_MODULE_SPEC.md) (ADR-0057, ADR-0058, ADR-0065).

## Hardware-ready

O Iris targets open-platform Android smart glasses, then possibly a helmet (ADR-0059). None have been chosen yet. The glasses parts are built now as code, tests, simulation and Kotlin twins checked on the JVM. These include the 360-degree rig ports, mount calibration, sound direction, power and heat budgets, phone link operations, the recording light and capture interlock, and the drawing layer. None of it has run on glasses. [docs/HARDWARE_READY_PROGRAM.md](docs/HARDWARE_READY_PROGRAM.md) lists, for each part, what is built, which placeholders the real parts will tune, and how to check it on the device.

## Use it responsibly

Read [ETHICS.md](ETHICS.md). Short version: consent first, private premises, normal client radio only (no monitor mode, injection or jamming), biometric laws apply (Illinois BIPA, Texas, Washington, California, GDPR), and never use it to accuse anyone of anything.

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) · [docs/EXTENDING.md](docs/EXTENDING.md) · [SECURITY.md](SECURITY.md) · [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

Licensed under Apache-2.0.

## Say hi 👋

Thanks for stopping by! It's totally okay to just say hi. If you've tried this project or just read through it, we'd love to hear what you think, so feel free to open an issue and share. Got an idea to make it better? Suggestions and pull requests are very welcome, big or small.
