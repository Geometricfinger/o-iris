# ADR-0063: Bystander recording light, hardware microphone mute and the capture interlock

- Status: Accepted (owner request, 2026-11-17)
- Date: 2026-11-17
- Source:
  - The owner, 2026-11-17: build every hardware-dependent part now as code plus reasoning, so the real glasses and helmet later need only fine-tuning.
  - [ADR-0059](0059-code-first-build-order.md) decision 3 lists "a bystander notice: an outward-facing indicator while the camera or microphone is active" among the things decided with the hardware. This ADR builds the code side now; the choice of light stays with the glasses.
  - Spec D-18 ([ADR-0018](0018-d-18-voice.md)): push-to-talk and a hardware mute.
  - Spec B-18 (docs/spec/ASB-Build-Spec-v2.md, section 10): "With the switch off, 0 non-zero samples".
  - The spec's GPIO table: GPIO23 is the mic-mute sense, with pull-ups and a 30 ms debounce; the mute switch cuts the microphone's 3V3 in hardware.

## Context
- Live mode opens the microphone for the voice link (ADR-0057, ADR-0065), and a camera path comes next. Nothing so far tells the people nearby that the device is recording, and nothing in software checks the hardware mute.
- A Mac already shows its own camera light and microphone dot. O Iris cannot control them or read them back.
- Glasses and a helmet are worn among people. A light there is the only notice a bystander gets.
- The mute switch is hardware: software cannot make it stronger, but software can refuse to listen while it is on, and can notice when the hardware cut has failed.

## Decision
1. **Two ports** (`oiris/capture/`).
   - `IndicatorPort`: `set(on)` and `read_back()`, returning True (confirmed lit), False (confirmed dark) or None (cannot be confirmed). None counts as unconfirmed.
   - `MuteSense`: `muted()`, returning True, False or None (unknown). An absent sense is no port.
2. **One interlock** (`oiris/capture/interlock.py`), pure logic with an injected clock. The camera and the microphone each `acquire` a grant before they run and `release` it after.
   - Before the first grant the light is turned on and read back until it is confirmed or `confirm_timeout_ms` passes. Unconfirmed means refused (`indicator_unconfirmed`), and the light goes off again.
   - The microphone is refused while muted (`muted`), and while the mute state is unknown or absent when the mute sense is required (`mute_unknown`). The camera ignores the mute.
   - After the last release the light stays on for `tail_s`, then goes off, so a bystander sees that recording just ended.
   - Every live step runs `check()`: a light whose read-back turns False or None while anything holds a grant revokes every grant (`indicator_failed`, fail closed); a switch turned to mute revokes the microphone.
   - Every microphone block passes `guard_samples()`. While muted, any non-zero sample on any channel means the hardware cut failed: a `mute_breach`. It revokes the microphone, zeroes the block and latches until restart. This is the software side of B-18.
   - Settle time: a block can hold up to 0.1 s of audio from before the switch was sensed, plus the 30 ms debounce. Blocks inside `MUTE_SETTLE_S` (0.2 s) after the mute is first sensed are zeroed and dropped but are not a breach.
   - `state()` gives codes only: `indicator` on|off|tail|failed|not_required, `mute` muted|open|unknown|not_present, `camera` and `mic` active|off|refused:<code>, `fault` null|mute_breach. The refusal codes are indicator_unconfirmed, indicator_failed, muted, mute_unknown, mute_breach and closed.
3. **Profiles** (config section `capture`).

   | profile | light | mute sense | adapters by default |
   |---|---|---|---|
   | mac (default) | off | if_present | none |
   | rig | required | required (B-18) | gpio; the light needs `indicator_pin` |
   | glasses | required, cannot be turned off | if_present | none: an adapter must be set |
   | helmet | required, cannot be turned off | if_present | none: an adapter must be set |

   - On the Mac the light is off because macOS shows its own camera light and microphone dot, which O Iris cannot control. Grants then never touch a light and the state says `not_required`. A mute sense that is present still applies.
   - A light or mute sense marked required with no adapter is a config error, not a silent pass.
   - `confirm_timeout_ms` is 50 to 2000 (default 200). `tail_s` is 0 to 10 (default 2).
   - The recording light may not use a pin that the GPIO map (ADR-0040) already assigns. This includes the IR LED gates on GPIO17 and GPIO27, which keep their duty cap.
4. **Live wiring** (`oiris/live.py`).
   - Whatever capture `voice_capture_factory` returns, mono or an (n, N) microphone array, is wrapped. The microphone opens only after a mic grant, and every block passes `guard_samples`.
   - Blocks that arrive after a revocation are zeroed and dropped. The next step closes the stream, so a revoked grant stops listening within one block.
   - Ambient listening opens again once the switch allows it. After a light failure it tries at most every 5 s, so the light does not blink. Push-to-talk follows the same rule: a press while refused returns False with the reason.
   - `snapshot()["capture"]` carries the state plus the profile. The runner exposes `camera_acquire()` / `camera_release()` for the camera path that comes next.
   - The dashboard shows one "Capture" line in plain words.
5. **A failed driver fails closed.** If a GPIO port cannot open (no library, not a Pi), the light becomes one that never confirms and the mute sense becomes one that always reads unknown.

## Built now vs later
- **Built now, tested with simulated ports and a fake GPIO module:**
  - the ports and the interlock;
  - the deterministic sims (a light that is stuck off, slow to confirm or failing after a time, and a mute switch on a scripted timeline);
  - the config section and its validation;
  - the live wiring and the dashboard line;
  - Pi GPIO adapters (`oiris/adapters/capture_gpio.py`): the LED on a configured pin, read back as the pin level, and the mute sense on GPIO23 with a pull-up and a 30 ms debounce. These are **untested on hardware**.
    - The GPIO library (the RPi.GPIO API; on a Pi 5 the rpi-lgpio package) is imported only when a port opens, and is not a dependency.
    - High on the sense line means muted, so a broken sense wire reads muted. Confirm this against the wiring.
- **Later, with the hardware:**
  - **The chosen glasses' own light.** Many glasses wire the camera light to camera power in hardware, which is stronger than any software light: it cannot be skipped by a bug. Prefer it when the glasses offer it, and expose it through `IndicatorPort` (read back from the glasses' own status if they report it).
  - **A photodiode read-back** that proves the light is lit, not just that its pin is driven.
  - **The B-18 bench check** on the rig: the switch off gives 0 non-zero samples.
  - **A legal review of notice duties** per jurisdiction ([ADR-0014](0014-d-14-legal-review.md)). A light is a courtesy and a design rule here, not a claim that it meets any law's notice requirement.

## Where it is enforced
- `oiris/capture/` (the ports, the interlock).
- `oiris/adapters/capture_sim.py` and `oiris/adapters/capture_gpio.py`.
- `oiris/config.py`: the `capture` section (`validate_capture`, `capture_settings`).
- `oiris/live.py`: `build_interlock`, the gated voice capture, `_capture_step` at every step, `camera_acquire` / `camera_release` and `snapshot()["capture"]`.
- `oiris/dashboard/server.py`: the Capture line.
- Tests: `tests/test_capture_interlock.py`, `tests/test_capture_adapters.py` and `tests/test_live_capture.py`.

## Consequences
- On the Mac nothing changes: the default profile always grants, and the existing live tests and golden runs are unchanged.
- On the rig, glasses and helmet, no camera frame or microphone block is used unless the light is confirmed. A missing, stuck or failing light stops capture rather than letting it run unannounced.
- A software light proves only what its read-back proves. Until a photodiode or the glasses' own hardware light exists, "on" means "the pin is driven".
- A mute breach stops the microphone until restart. Fix the hardware first.

## Changing it
Write a new ADR that supersedes this one, update CHANGELOG.md, and get the owner's sign-off. A change that weakens a rule needs negative tests showing the old refusal still happens where it must: letting a profile turn the light off, granting on an unconfirmed read-back, or letting the microphone run while muted.
