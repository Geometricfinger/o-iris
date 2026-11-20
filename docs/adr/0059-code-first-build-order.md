# ADR-0059: Build order: code infrastructure first; Android glasses, then a helmet, as the target

- Status: Accepted (owner, 2026-11-16)
- Date: 2026-11-16
- Amends (build order only): [ADR-0007](0007-d-07-glasses-spike.md) ("skip until G1") and [ADR-0009](0009-d-09-platform.md) ("rig + Android operator app; glasses Phase 2 only after G1")
- Source: the owner, 2026-11-16:
  - O Iris is meant first for open-platform Android smart glasses, not yet bought, then possibly a helmet with smart glasses, sensors and cameras.
  - What can be built and tested now is what is built now.
  - No glasses are owned yet, so the code infrastructure comes first.
  - The gap review of the build spec against that goal (2026-11-16) found where the docs had to change.

## Context
- The build spec (v2, section 15) puts glasses after gate G1, which needs rig tests (E-03, E-07), and makes the Pi rig plus an Android operator app the platform.
- The owner's target is now glasses first, then possibly a helmet. Nothing can be measured on glasses until a model is chosen and bought.
- Much of the glasses path does not depend on the hardware:
  - the voice link;
  - output arbitration and the motion policy;
  - shared Python/Kotlin test vectors;
  - the glasses display renderer;
  - timing counters.
- Some decisions do depend on the hardware, and making them now would be guesswork.

## Decision
1. **Target platforms, in order:** open-platform Android smart glasses, then possibly a helmet with smart glasses, more sensors and cameras. The Mac and the Pi rig stay as test benches.
2. **Build now:** code infrastructure that can be built and tested without the glasses.
   - On the Mac.
   - In simulation: `oiris glasses-sim`, headless and golden runs, simulated adapters including an IMU.
   - In JVM checks for the Android code, with test vectors shared with Python.

   The spec's gates still govern every claim about real-device performance. Code built for glasses is "built", not "working on glasses", until it has run on them.
3. **Decided later, when the glasses are chosen, with evidence from them:**
   - **Where the core runs:** on the glasses, the phone, a pod, or split between them; and whether the glasses are a full node or a display client. Bluetooth Low Energy (BLE) bandwidth matters here: the spec plans about 6 KB/s, while 16 kHz microphone audio is 32 KB/s.
   - **The glasses gate:** what "working on glasses" means for latency, legibility, battery and heat.
   - **Moving enrolled templates between the owner's own devices:** re-enroll on each device, or transfer them between owner-paired devices over the Noise link with a verified wipe of the source. Today templates are never exported (ETHICS.md), so either needs an ADR.
   - **The wearer's own voice:** handling beyond the M-17 `self` marker (ADR-0058), and microphones that favour the wearer.
   - **A bystander notice:** an outward-facing indicator while the camera or microphone is active.
   - **Power and heat budgets,** and the duty cycling that meets them.
4. **Public use stays as ETHICS.md states it:**
   - The owner's own premises, with consenting adults.
   - A lawyer's review before any public use, any subject beyond household adults, or any sale or sharing (ADR-0014).
   - The one exception remains M-16 research mode on the user's own opt-in (ADR-0050).

   ADR-0050's principle (an on/off switch, the owner's call and risk) does not yet extend to M-17's ambient listening and spoken names in public. Extending it is the owner's decision and needs its own ADR. Either way, the M-17 warning names the recording and eavesdropping laws that can apply to listening even when nothing is stored, including places that need every party's consent (spec N-04, R-4).

## Where it is enforced
Process and documentation:
- docs/STATUS.md keeps "built" apart from "observed on hardware".
- Each deferred item above becomes its own ADR when the hardware exists.

## Consequences
- Glasses-path code can be built and tested now without waiting for G1. Nothing about it is claimed to work on glasses until it has.
- ADR-0007 and ADR-0009 keep their hardware gates; only their "nothing for glasses before G1" ordering changes.

## Changing it
Write a new ADR that supersedes this one, update CHANGELOG.md, and get the owner's sign-off.
