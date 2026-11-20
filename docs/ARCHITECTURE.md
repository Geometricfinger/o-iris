# Architecture

## Shape
One process. One thread per sensor adapter (the **sensor bus**). Each `Engine.step()` drains the bus, runs every enabled **score module** (each returns a score in [0,1], a confidence and a health flag), and hands the scores to **one fusion layer**. A **guard** checks every output. The phone app and dashboard talk to the runtime only through the contracts below.

What drives `step()` depends on the mode:
- **Live mode** (`oiris run`, `oiris/live.py`): a background thread steps the Engine at `live.hz` (default 1 Hz, at most `MAX_HZ` = 10), records the session as one sweep, writes sightings and serves the dashboard (docs/LIVE.md).
- **Headless** (`oiris --headless`, `oiris/headless/`): a virtual clock, one step per pulled sample, so runs are repeatable.
- **Library use**: `Engine.run()` is an asyncio loop at a given rate (default 1 Hz).

```
 sensors ─► adapters (C2) ─► SensorBus ──► score modules (C1) ──► fusion ──► guard ──► events (C3)
   real │ replay │ sim      thread/adapter   safe_update()         families       vocabulary      │
                            watchdog, breaker                       ≥2-family rule                  ├─► voice tiers (M-06)
                                                                                                   ├─► sighting log (C4, M-05)
                                                                                                   └─► Noise link ─► Android app
```

## Ports (ports & adapters)
| Port | Interface | Version | Registry key |
|---|---|---|---|
| sensor | `oiris.adapters.base.SensorAdapter` (open/start/read/stop/close/health, `caps`, `license`) | 1.0 | `("sensor", id)` |
| score | `oiris.scores.base.ScoreModule` (`name`, `inputs`, `module_flag`, `update(window)->Score`) | 1.0 | `("score", name)` |
| model | `oiris.models.ModelPort` + C9 manifest; loaded into a `ModelSlot` | 1.0 | `("model", name)` |
| tts | `oiris.modules.voice.Tts` (`speak`, `stop`, `offline`) | 1.0 | `("tts", name)` |
| store | `oiris.store.db.DB` (migrations, sealed JSON) | 1.0 | `("store", name)` |

`oiris/registry.py` holds the ports; built-ins register in `oiris/adapters/builtin.py` and `oiris/scores/builtin.py`; third parties register through the `oiris.plugins` entry-point group. A plugin with a different **major** port version or a disallowed license is rejected at registration. A plugin that throws while loading is logged and skipped.

## Fault isolation
| Failure | Containment | Test |
|---|---|---|
| Adapter `read()` raises (even `SystemExit`) | caught in the adapter thread → DEGRADED, restart with exponential back-off | `test_fault_injection.py` |
| Adapter hangs | read runs in a helper thread with a timeout; timeout counts as a failure | same |
| N consecutive failures | circuit breaker opens → FAILED, no restarts until cooldown | same |
| Garbage samples | validator rejects; counter incremented | same |
| Score module raises / NaN / wrong type | `safe_update` → FAILED score; family masked | same |
| ≥2 families unhealthy | fusion suppresses the index | same |
| Model or fusion-weights load fails | last-good or disabled; never raises | `test_model_slot`, `test_model_load_failure_falls_back` |
| Event handler raises | logged; loop continues | `test_event_handler_crash_does_not_stop_loop` |
| Bad config | rejected at startup with a clear message; unknown keys ignored; safe defaults | `test_config_defaults_and_validation` |
| DB newer than code | refuses to open (no silent downgrade) | `test_refuses_newer_schema` |

## Contracts
| ID | Contract | Where | Owner |
|---|---|---|---|
| C1 | Score: module, value [0,1], confidence [0,1], health, timestamps, version; no class field | `oiris/schema.py`, `schema/score.json` | Core |
| C2 | Sensor adapter + Sample | `oiris/adapters/base.py`, `schema/sample.json` | Core |
| C3 | Event: signature A1–A11, level fyi/attention/confirmed, benign causes; no class field | `oiris/schema.py`, `schema/event.json` | Core |
| C4 | Sighting log: encrypted body, GPS policy, hash-chain fields, verdict, retention | `oiris/store/sightings.py`, migrations | Core |
| C5 | Golden dataset manifest (format v2) | `oiris/modules/golden.py` | Training |
| C6 | Layout-zone map (glasses G*, phone P1–P5, dashboard D1–D6) | `docs/spec/ASB-Module-Map-v1.1.pdf`, used by `android/` and `oiris/dashboard/` | Design |
| C7 | Design tokens | `oiris/ui/design_tokens.json` v0.1 (DEFAULT, validated by `oiris/ui/tokens.py`; ADR-0052) | Design |
| C8 | Versioning and change policy | `docs/UPGRADING.md` | Docs |
| C9 | Model package manifest | `oiris/models.py` (`REQUIRED` fields) | Training |

## Signal families and signatures
Families: optical, thermal, rf, vital, eye_liveness, voice. Signatures A1–A11 are defined in `oiris/schema.py` with their benign causes. An alert (ATTENTION) needs ≥2 families; one family is FYI; CONFIRMED is set only by an operator.

## Module map (spec M-01…M-15 → code)
| Module | Code |
|---|---|
| M-01 Eye behaviour | `features/eye.py`, `scores/builtin.EyeScore` (default off) |
| M-02 Liveness stacking | `modules/liveness.py`, `features/vital.py`, `LiveScore`, `VoiceScore` (default off) |
| M-03 Signal fingerprint | `features/rf.py`, `modules/fingerprint.py`, `RfDriftScore` |
| M-04 Security | `security/*`, `android/.../KeyVault.kt` |
| M-05 Sighting log | `store/sightings.SightingLog`, `android/.../SightingStore.kt` |
| M-06 Voice output | `modules/voice.py`, `guard/phrases.py`, `android/.../VoiceOut.kt` |
| M-07 Baselines / new object | `features/baseline.py`, `NewObjScore` |
| M-08 Thermal view | `features/thermal.render/overlay` |
| M-09 Headless + golden | `headless/`, `modules/golden.py`, `golden/expected/` |
| M-10 One-commit build | `scripts/build_all.sh`, `.github/workflows/ci.yml` |
| M-11 Sensor bus / adapters | `bus.py`, `adapters/*` (macOS: `adapters/macos.py`), `registry.py` |
| Live mode | `live.py` (runner, network guard), `dashboard/server.py`; macOS app: `scripts/make_macos_app.py`, `macos/launcher/` |
| M-12 Feedback loop | `modules/golden.FeedbackLoop` |
| M-13 Android fallback node | `android/.../FallbackNode.kt`, `GlintCamera.kt`, `adapters/drivers.AndroidBridgeAdapter` |
| Phone link (M-04) | `oiris/security/channel.py` ↔ `android/core/.../Noise.kt` (shared vectors `tests/vectors/`); M-16 EnrollmentPort over it: `face_match/remote.py` ↔ `core/.../Enrollment.kt` |
| UI (module 2) | `oiris/ui/` (C7 tokens, simulated glasses), `oiris/dashboard/` (D1–D6, `/face`), Android activities |
| M-14 Sweep timeline | `store/sightings.SweepTimeline` |
| M-15 App gate | `android/.../Gate.kt`, `modules/appgate.py` |
| M-17 Voice link (optional, off by default; ADR-0057, ADR-0058, [spec](VOICE_LINK_MODULE_SPEC.md)) | `modules/voice_link/` (`core.py` enrollment, matching, `self` marker and `voice_antispoof` hook; `capture.py`, `vad.py`, `listener.py` live listening; `announce.py` chime and "sounds like NAME"; `backends/sherpa_campp.py` CAM++), `voice_link_cli.py`, dashboard "Voice link" area; Android rules port `android/core/.../VoiceLink.kt` (shared vectors `tests/vectors/voice_link.json`; no speaker model on Android) |
| Outputs and motion | `modules/output.py`, `modules/motion.py`, `adapters/imu_sim.py` (below); dashboard "Outputs" area |
| Spatial sensing (ADR-0060) | `spatial/` (`frames.py`, `rig.py`, `detections.py`, `merge.py`), `adapters/sim_rig.py` (below) |

## Outputs and motion
On a wearable, alerts, the voice-link announcement, pop-ups and haptics must never stack, overlap or distract while moving. One `OutputArbiter` (`oiris/modules/output.py`) carries every audible, visual and haptic output; nothing else plays sound in live mode.

- **API.** `request(kind, priority, key, text=None, sound=None, haptic=False, person_id=None)` returns a `Decision` (`outcome`: `played`, `queued`, `deduped`, `suppressed` or `dropped`, plus a `reason`). It never raises; player and haptic failures are logged. `playing()`, `popup()`, `state()`, `cancel(key)` and `close()` complete it.
- **Priorities** (named constants): `PANIC` > `CONFIRMED` > `ATTENTION` > `VOICE_LINK` > `FYI`.
- **Channels.** Speech (one utterance at a time) and chime are played in order by one daemon worker from a bounded queue (`QUEUE_JOBS`; a full queue evicts its lowest job or drops the new one; a job not started within `QUEUE_MAX_AGE_S` is dropped). The pop-up is state the glasses/dashboard read (text, priority, expiry). The haptic is a short pulse. Alerts pop up and buzz, and speak only with `output.speak_alerts`; `VOICE_LINK` chimes and speaks but never pops up; `FYI` is haptic only (M-06). When a request uses several channels, the audible one decides the outcome if it played or queued.
- **Rules.** A higher priority barges in on lower speech (the player's `stop()`; the cut-off job is not replayed). On macOS `announce.MacPlayer` runs `afplay` and `say` with `Popen` and waits for them; its `stop()` terminates the command playing now and kills it after `STOP_GRACE_S`. The same key inside `output.dedupe_s`, or already queued or playing, is deduped (panic and voice link are not windowed: the Announcer keeps its per-person repeat limit). The M-06 budget, one attention utterance per `voice_attention_min_interval_s`, holds later ones in the queue for their slot. `playing()` covers queued-and-ready and playing jobs plus a short tail (`TAIL_S`).
- **Motion gate.** `MotionEstimator` (`oiris/modules/motion.py`) classifies accelerometer samples from any sensor adapter with the `imu` capability as `still`, `walking`, `fast` or `unknown`: the standard deviation of the acceleration magnitude over a 2 s window separates still from walking, a sustained running-level deviation or a speed input (GPS, later) gives fast, and enter/exit thresholds plus hold times stop it flickering. Unknown (no IMU, or stale) counts as still. While walking there are no pop-ups (the G1 strip still shows the level; Module Map §6.1) and speech is allowed; while fast only `output.fast_min_priority` (confirmed) and above play, without pop-ups. Suppressed outputs are dropped, never replayed; a queued job is checked again when its turn comes. All thresholds are provisional until measured on the glasses.
- **IMU.** `oiris/adapters/imu_sim.py` registers the sensor adapter `imu` (deterministic `still`, `walking` and `driving` profiles; driving carries a simulated speed, since an IMU alone cannot tell a smooth car from sitting still). A real glasses IMU adapter plugs into the same C2 interface and sample shape (`t_s`, `accel`, optional `speed_mps`).
- **Wiring.** `VoiceOutput` uses the arbiter when given one and is unchanged without it. The voice-link `Announcer` sends its chime and phrase at `VOICE_LINK` priority; its `playing()` (the listener's input pause) becomes the arbiter's. `LiveRunner` builds one arbiter, feeds `imu` samples to the estimator, reports `snapshot()["output"]` (`motion`, `imu`, `popup`, `speaking`, `queued`, `counts` by outcome) and closes it on stop. The live dashboard's "Outputs" area shows the motion state in plain words and the current pop-up (escaped; hidden at its expiry by CSS, since the page runs no script). `GlassesState.moving` comes from the motion state (`oiris/ui/glasses.py`).
- **Config** (`"output"`, validated in `oiris/config.py`): `speak_alerts` (false: the Mac stays silent for sensor signatures; the voice link still announces), `popups_while_walking` (false), `fast_min_priority` (`confirmed` or `panic`), `popup_s` (6, in [1, 30]), `dedupe_s` (30, in [0, 600]).

## Spatial sensing
Cameras (one forward now; rear and two side mounts planned), thermal and NIR cameras and microphone arrays plug into one rig, and what they locate merges into events that say where something is: ahead, behind, at left, at right, above or below. Conventions are fixed by [ADR-0060](adr/0060-spatial-sensing.md); this is built and tested in simulation only, and no mount has run on real hardware.

- **Head frame** (`oiris/spatial/frames.py`). Right-handed, origin midway between the eyes, x forward, y left, z up. Azimuth `atan2(y, x)` in (-180, 180] (0 ahead, +90 left, 180 behind), elevation `atan2(z, hypot(x, y))`. A mount's rotation is `Rz(yaw) @ Ry(-pitch) @ Rx(roll)` (yaw + turns left, pitch + tilts up, roll + is clockwise seen from behind). Pixels become rays through pinhole intrinsics, or intrinsics derived from `hfov_deg`/`vfov_deg` and the image size. Bearings ignore mount positions (parallax is negligible beyond about 1 m); positions are kept for later range work.
- **Direction words.** The `{direction}` phrase-slot values (`oiris/guard/phrases.py`): ahead |az| <= 45, at left 45 < az <= 135, at right -135 <= az < -45, behind |az| > 135; above el > 45 and below el < -45 override. A tracked bearing keeps its word until it is 10 degrees past a boundary (`DirectionTracker`; the engine's `EventTracker` carries tracks between steps).
- **The ports** (`oiris/spatial/rig.py`, config section `"rig"`, validated in `oiris/config.py` without importing numpy). `{"mounts": [{id, kind: camera|thermal|nir|mic_array|imu, position_m, yaw_deg, pitch_deg, roll_deg, hfov_deg?, vfov_deg?, image_size?, intrinsics? {fx, fy, cx, cy}, mic_positions_m? (mic_array, at least 2), ir_illuminator, ir_eye_safety_approved}]}`. An empty section is the default rig, one forward camera `front`, so existing setups need no config; `"mounts": []` and absent mounts mean no coverage, never an error.
- **Eye safety.** A mount with `ir_illuminator: true` whose forward axis is more than 30 degrees from straight ahead (horizontally or vertically) is refused at startup unless `ir_eye_safety_approved: true`: build spec v2 section 7.3 aims IR LEDs at objects (rule 4) and requires an accredited IEC 62471 test before use on strangers (rule 5), and a rear or side illuminator lights people the wearer cannot see.
- **Detections** (`oiris/spatial/detections.py`). A frozen `SpatialDetection` (kind glint|ir_emitter|thermal|object|sound, family, mount_id, t_mono_ns, confidence, and exactly one of pixel, mount_bearing or head_bearing; optional range_m and size), from any source including sound direction finding. Helpers place A1 glint candidates, A2 saturated NIR spots and A3 hot regions. Adapters with the capability `spatial` put them in `Sample.data["spatial_detections"]`.
- **Merging** (`oiris/spatial/merge.py`). Detections within 15 degrees and 0.5 s of each other are linked and each connected group is one `SpatialEvent`: confidence-weighted mean bearing, united families, sensors and kinds, noisy-OR confidence, evidence = distinct families. Order-independent and deterministic. When spatial samples arrive, `Engine.step()` adds `res["spatial"]` (and `res["spatial_unplaced"]`, a count, when a detection names an absent mount or lands off its image). Fusion, levels and the two-family alert rule are unchanged; without spatial samples results and golden outputs are identical to before.
- **Simulation** (`oiris/adapters/sim_rig.py`, sensor adapter `sim_rig`). Scripted scenes place objects at head-frame bearings by formula; one adapter per mount reports only what lies in that mount's field of view (pixels for imaging mounts, bearings for a microphone array) on the default simulated rig: front, left, rear and right cameras (hfov 100, overlapping by 10 degrees), a forward thermal camera and a four-microphone array.
