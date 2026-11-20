# ADR-0060: Spatial sensing: one head frame, rig mounts, bearings and directional merging (the 360-degree plan)

- Status: Accepted (owner request, 2026-11-16)
- Date: 2026-11-16
- Source:
  - The owner, 2026-11-16, asked for:
    - the code infrastructure, the ports, so future cameras plug in for 360-degree coverage (one at the back, two on the sides) and their data merges;
    - vision detection, from IR and other readings, that says what it sees ahead, behind, left or right;
    - sound detection that merges the same way.

    The hardware does not exist yet, so the code plans for it and proves it in simulation (ADR-0059).
  - The gap review of 2026-11-16, item 15: phrases can say "at left" or "behind", but no code computes a direction.

## Context
- The planned platform is Android smart glasses, which have a front camera and may add thermal or NIR. A helmet would add rear and side cameras and a microphone array.
- Today no detection carries a direction. The phrase whitelist has a `{direction}` slot (`ahead`, `at left`, `at right`, `behind`, `above`, `below`) that nothing fills.
- Fusion counts signal families without knowing where each one was observed.
- Each device has its own monotonic clock, so detections from different devices cannot be called simultaneous.

## Decision
1. **One head frame.**
   - Right-handed, with its origin midway between the eyes: x forward, y left, z up.
   - Azimuth is `atan2(y, x)` in degrees, in (-180, 180]: 0 is straight ahead, +90 left, -90 right, 180 behind.
   - Elevation is `atan2(z, hypot(x, y))`, with +90 straight up.
   - Every bearing is expressed in this frame once it has been through calibration.
2. **The rig and its mounts, the "ports".**
   - The config section `rig` lists mounts. Each has:
     - an id and a kind: camera, thermal, NIR, microphone array or IMU;
     - a position in metres;
     - an orientation: yaw (+ to the left), pitch (+ up), roll (+ clockwise seen from behind), giving `R = Rz(yaw) Ry(-pitch) Rx(roll)`;
     - a field of view and image size, or pinhole intrinsics;
     - for a microphone array, its microphone positions.
   - An adapter attaches to a mount by id.
   - A missing mount means no coverage there, never an error. The default rig is a single forward camera, so existing setups need no config.
3. **Spatial detections.**
   - Any adapter or feature can emit a detection with:
     - a kind: glint, IR emitter, thermal, object or sound;
     - a family, a mount, a time and a confidence;
     - a pixel, a mount-frame bearing or a head-frame bearing;
     - optionally a range and a size.
   - The rig converts it to a head-frame bearing. A camera pixel becomes an optical-frame ray (OpenCV pinhole), then a mount-frame ray, then a head-frame ray.
   - Sensors without a direction (radio, vital signs) are unchanged.
4. **Direction words.**
   - Sectors:
     - `ahead` when |az| <= 45;
     - `at left` when 45 < az <= 135;
     - `at right` when -135 <= az < -45;
     - `behind` when |az| > 135.
   - `above` (el > 45) and `below` (el < -45) override the horizontal word.
   - 10 degrees of hysteresis keep a tracked word steady at the boundaries.
   - These fill the existing phrase slot. Distance words are used only when a range estimate exists.
5. **Merging.**
   - Detections from different sensors within 15 degrees and 0.5 s of each other merge into one spatial event.
   - Its bearing is the confidence-weighted mean of the unit vectors. Its families and sensors are united. Its confidence is the noisy-OR of the parts, and its evidence is the number of distinct families.
   - Two families agreeing on where something is is stronger corroboration than two families anywhere. The existing rule that an alert needs at least two families is unchanged: spatial events are an added output until fusion weights are trained on real data.
   - Without spatial inputs, every result is exactly as before.
6. **Sound direction.**
   - Microphone arrays give a direction of arrival from GCC-PHAT time differences between microphone pairs, in the head frame, with the speed of sound fixed at 343 m/s.
   - Two microphones, left and right, separate left from right but not front from back; the result is reported as ambiguous, with both candidates.
   - Three or more non-collinear microphones give a full azimuth; four or more non-coplanar ones add elevation.
   - Silence or an unclear correlation gives no answer, never a guess.
   - Sound bearings merge with vision bearings the same way. Later a voice-link match can also say where it was heard.
7. **One clock.**
   - Devices on the link align clocks with an NTP-style four-timestamp exchange over the Noise channel, in integer nanoseconds with floor division, matched exactly in Kotlin.
   - Each pairing keeps the lowest-delay sample and estimates drift with a robust fit.
   - Samples from another device are converted to the host clock before merging.
8. **Eye safety.**
   - An IR illuminator may be enabled only on a mount whose axis is within 30 degrees of straight ahead, unless the rig config records an accredited eye-safety approval for that mount (build spec section 7.3, rule 5).
   - Config validation refuses rear and side illuminators without that approval.
9. **No tracking of people.** Spatial outputs describe signatures and their direction, never a person's identity or path. The vocabulary, privacy and retention guards apply to every spatial output, and nothing about strangers is stored beyond what each module already allows.

## Built now vs later
- **Built now, tested in simulation:**
  - rig config and validation;
  - frame maths, pixel-to-bearing and direction words with hysteresis;
  - cross-sensor merging;
  - a simulated 360-degree rig adapter (front, rear, left, right);
  - sound direction on synthetic multi-microphone signals;
  - the clock-sync exchange in Python and Kotlin, with shared vectors.
- **Later, needing hardware:**
  - real calibration of each camera's intrinsics and mount pose (an ArUco-based procedure, using `oiris/features/aruco.py`);
  - mounting;
  - real microphone arrays;
  - latency and power budgets on the device;
  - direction cues on the glasses display and in speech: edge arrows and the phrase slot filled from spatial events, a follow-up on top of this.

## Where it is enforced
- `oiris/spatial/` (frames, rig, detections, merge) and `oiris/adapters/sim_rig.py`.
- `oiris/features/doa.py`.
- `oiris/link/timesync.py` and the pod service's `time_sync` op, with `android/core` `TimeSync.kt` matching it.
- `oiris/config.py`: the `rig` section and the eye-safety validation.
- `oiris/engine.py`: spatial events in the step result.

## Consequences
- Cameras and microphones can be added later by declaring a mount and attaching an adapter; nothing else needs to change.
- Merged directions are only as good as calibration. Until real mounts are calibrated, simulated rigs prove the maths, not the accuracy.

## Changing it
Write a new ADR that supersedes this one, update CHANGELOG.md, and get the owner's sign-off.
