# ADR-0061: Power, heat and latency budgets for a wearable: a governor, duty plans and stage budgets

- Status: Accepted (owner request, 2026-11-17)
- Date: 2026-11-17
- Source:
  - The owner, 2026-11-17, asked for every hardware-dependent part to be built now, as code plus reasoning, so that the real glasses and helmet parts later only need fine-tuning. This ADR covers power, heat and latency budgets.
  - ADR-0059 deferred "power and heat budgets, and the duty cycling that meets them" until the glasses are chosen. This ADR builds the policy, its ports and a simulation that proves it; the numbers stay provisional until measured on the device.

## Context
- Smart glasses carry a small battery (around 1 to 2 Wh) next to the face. A camera, an IR illuminator, a speaker model and a bright display can drain it within an hour and warm the frame against the skin.
- Worn devices have a touch-temperature ceiling. About 43 C is the usual limit for long skin contact (IEC 62368-1 touch-temperature limits); it has to be confirmed against the standard and the chosen glasses' own thermal specification before shipping.
- Android reports battery through `BatteryManager` and heat through `PowerManager.getCurrentThermalStatus()` (API 29: NONE 0, LIGHT 1, MODERATE 2, SEVERE 3, CRITICAL 4, EMERGENCY 5, SHUTDOWN 6) and `getThermalHeadroom()`. Skin and device temperatures depend on the hardware.
- No wearable exists yet. The latency counters (`oiris/timing.py`, docs/LIVE.md "Timing") measure every stage but nothing compares them with a target.

## Decision
1. **Levels and axes.**
   - Four levels, least to most restrictive: `normal`, `saver`, `hot`, `critical`.
   - Two axes; the more restrictive one wins:
     - battery: `saver` or `critical`;
     - heat: `hot` or `critical`, from skin temperature, device temperature or the Android thermal status (MODERATE is hot; SEVERE and above are critical; `thermal_status_level()` is a pure mapping, so the Android side only passes the integer).
   - Charging cancels the battery axis but not heat: charging adds heat.
   - A reading's fields are all optional except its time. None means unknown and is never guessed: an unknown value keeps that signal's last state, so a sensor that stops reporting cannot relax the level.
2. **Hysteresis.** Every threshold has an enter and an exit value. Defaults (config `power`, named and provisional):

   | Signal | Enter | Exit |
   |---|---|---|
   | Battery saver | <= 20 % | >= 25 % |
   | Battery critical | <= 7 % | >= 12 % |
   | Skin hot | >= 41.0 C | <= 39.5 C |
   | Skin critical | >= 43.0 C | <= 41.5 C |
   | Device hot (placeholder) | >= 75 C | <= 70 C |
   | Device critical (placeholder) | >= 85 C | <= 80 C |

   Validation refuses an exit on the wrong side of its enter, or a critical entry not beyond the matching saver or hot entry.
3. **Dwell.** Tightening is immediate. Relaxing goes one level at a time (critical, hot, saver, normal) and only when the exit thresholds are met and the governor has been at its level for `min_dwell_s` (30 s). One level at a time can pass through a stricter plan than needed (critical to hot when only the battery was low); that errs on the cool side.
4. **Stale readings.** A reading older than `stale_after_s` (60 s), or with no finite time, relaxes nothing: the level holds and the reasons include `stale`. A stale reading may still tighten, the safe direction.
5. **Decisions.** `Decision(level, plan, reasons)`. Reasons are fixed codes: `battery_low`, `battery_critical`, `charging`, `skin_hot`, `skin_critical`, `device_hot`, `device_critical`, `thermal_status_hot`, `thermal_status_critical`, `stale`. Numbers and codes only.
6. **Duty plans,** one per level (config `power.plans`, laid over the defaults per field). A plan is a set of caps: it never turns on something the config or the owner has off, and the rig's IR eye-safety rule (ADR-0060) still applies.

   | Field | normal | saver | hot | critical |
   |---|---|---|---|---|
   | camera_max_fps (0 = off) | 30 | 15 | 10 | 0 |
   | frame_stride (every Nth admitted frame) | 1 | 2 | 3 | 3 |
   | listen_duty | 1.0 | 0.5 | 0.25 | 0.0 |
   | listen_period_s | 10 | 10 | 10 | 10 |
   | radio_scan_interval_s | 15 | 30 | 30 | 60 |
   | ir_illuminator | on | on | off | off |
   | voice_match | on | on | on | off |
   | face_match | on | on | off | off |
   | display_brightness_cap | 1.0 | 0.6 | 0.5 | 0.3 |

   - Each step cuts the largest users first: camera rate and processing, then the IR illuminator and the face model (heat comes mostly from the camera pipeline), then the voice model. Radio scanning, the cheapest family and the guard's minimum, never stops; in critical it only slows.
   - **Order rule:** each level is at least as restrictive as the one before it on every restricting field: lower or equal fps, higher or equal stride, lower or equal listening duty, longer or equal scan interval, lower or equal brightness cap, and a feature off stays off. `listen_period_s` only spreads the duty over time, so it is not ordered. A wrong order, an out-of-range field, or a critical plan with the camera, face model, voice model or IR illuminator on is a `ConfigError`.
   - `FrameGate(max_fps, stride).admit(t)` and `ListenDuty(duty, period_s).active(t)` are pure helpers. When listening is switched off, an utterance already being heard is finished, not cut.
7. **Latency budgets** (config `budgets`, `{"<part>.<stage>": ms}` laid over the defaults). Part is one of live mode's `TIMING_PARTS` (`engine`, `voice`, `output`); stage follows the timing stage-name pattern; values are positive milliseconds up to 60000. Defaults exist only for stages the code records today, each provisional until measured on the device:

   | Budget | ms | Reasoning from the code |
   |---|---|---|
   | `engine.step` | 50 | Live mode steps at `live.hz` (default 1, at most `MAX_HZ` 10), so the shortest interval is 100 ms. The step must finish well inside it, leaving the rest for the bus threads, the voice worker and the dashboard on the same CPU: half the interval. |
   | `voice.queue_wait` | 1000 | The microphone delivers 0.1 s blocks into a bounded queue of `QUEUE_BLOCKS` 100 (about 10 s), and a full queue drops audio. A block waiting a tenth of the queue's capacity shows the worker falling behind long before anything is dropped. |
   | `voice.decision` | 1000 | Utterance end to match result. The end is only declared after `END_QUIET_S` (0.7 s) of quiet; with this and the output's 1.5 s, "sounds like NAME" starts about 3 s after the speaker stops, the most a match can lag and still be about the person present. |
   | `output.chime_wait` | 1500 | Spec R-17 / E-20: attention speech starts within 1.5 s p95. The chime precedes the speech, so it has the same ceiling. |
   | `output.speech_wait` | 1500 | The same R-17 target. It is measured from the request, so speech after a chime includes the chime. |

   `BudgetMonitor(budgets, min_samples=20).check(timing_snapshot)` lists `{"part", "stage", "p95_ms", "budget_ms", "count"}` for each stage whose p95 is above its budget with at least 20 samples. Below 20 samples the nearest-rank p95 is just the maximum, so one slow start would read as a breach.
8. **How warnings will surface** (the wiring task): live mode will put the monitor's list in `snapshot()["budgets"]` and the dashboard will show it as a plain line in the sensor-status area, logged once per change. A breach never changes behaviour, raises an alert or reaches the glasses: it is a measurement for the developer. The power decision will go in `snapshot()["power"]` as `Decision.to_dict()`.

## Built now vs later
- **Built now, tested in simulation:**
  - the governor (`oiris/power/governor.py`), the duty plans and their validation, `FrameGate` and `ListenDuty` (`oiris/power/duty.py`), and the budgets and `BudgetMonitor` (`oiris/power/budgets.py`);
  - the config sections `power` and `budgets`;
  - a deterministic closed-loop power and heat simulation (`oiris/adapters/power_sim.py`): `P = base + camera * fps/max + compute (voice or face model) + IR + mic * duty + display * brightness`, a battery in Wh, and a first-order thermal model `dT/dt = (P * R - (T - ambient)) / tau` with the skin rise a fixed fraction of the device's. Every coefficient is a named placeholder. Scenes: `cool_room` stays normal; `hot_day` (35 C ambient) reaches hot and settles there, while the same device held at the normal plan passes the critical skin entry, so the throttling is what keeps it cool; `low_battery` passes saver, then critical; `charging_hot` heats from charging with the battery axis cancelled;
  - a Mac battery reader (`oiris/adapters/power_macos.py`, `pmset -g batt`; no temperature on a Mac), never raising into the caller.
- **Later:**
  - an Android adapter: `BatteryManager` (level, plugged state), `PowerManager.getCurrentThermalStatus()` into `thermal_status_level()`, and `getThermalHeadroom()` as an earlier warning;
  - skin and device temperatures from the chosen hardware, and the device thresholds from its datasheet;
  - measured coefficients for the simulation and measured budgets, replacing every provisional number here;
  - the live wiring (`oiris/live.py` is not changed by this ADR): a governor fed on the loop, `FrameGate` in the camera adapters, `ListenDuty` and `voice_match` through the voice listener's existing `active` check, between utterances, the radio scan interval in the Bluetooth and Wi-Fi adapters, the brightness cap in the glasses renderer, `face_match` as a skip in the engine's M-16 step, the budget list in the snapshot, and registering the simulated `power` adapter by name.

## Where it is enforced
- `oiris/power/` (governor, duty, budgets).
- `oiris/config.py`: the `power` and `budgets` sections are validated on every `Config.validate()`; an empty section is all defaults.
- `oiris/adapters/power_sim.py` and `oiris/adapters/power_macos.py`.
- Tests: `tests/test_power_governor.py`, `tests/test_power_duty.py`, `tests/test_latency_budgets.py` (including that its `TIMING_PARTS` equals live mode's), `tests/test_power_adapters.py`.

## Consequences
- The policy is fixed and tested before any hardware: a real device only needs its adapter, its measured thresholds and coefficients, and the wiring.
- Nothing changes in live mode, headless runs or the golden suite until the wiring task.
- The simulation proves the policy (no flapping, throttling cools, the battery axis order, charging), not any device's numbers. Every threshold, plan and budget is provisional.
- Throttling lowers what the device can sense (fewer frames, less listening) exactly when it is hot or low. That is the accepted trade: a wearable that burns the wearer or dies mid-walk senses nothing.

## Changing it
Write a new ADR that supersedes this one, update CHANGELOG.md, and get the owner's sign-off. Changing a default threshold, plan or budget after measurement on the device needs only the measurement recorded in docs/STATUS.md and a CHANGELOG entry.
