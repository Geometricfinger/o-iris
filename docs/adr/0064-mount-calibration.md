# ADR-0064: Mount calibration: camera intrinsics and head-frame mount pose from printed ArUco boards

- Status: Accepted (owner request, 2026-11-17)
- Date: 2026-11-17
- Source:
  - The owner, 2026-11-17, asked for every hardware-dependent part to be built now, as code plus reasoning, so that the real glasses and helmet cameras later only need fine-tuning.
  - ADR-0060 lists "real calibration of each camera's intrinsics and mount pose (an ArUco-based procedure, using `oiris/features/aruco.py`)" as later work. This ADR builds that procedure and proves it in simulation (ADR-0059).

## Context
- ADR-0060 turns every camera pixel into a head-frame bearing through the mount's intrinsics and its yaw, pitch and roll. Merging (15 degrees, 0.5 s) and the direction words are only as good as those numbers.
- Field-of-view guesses are wrong by several percent on real lenses, and a camera glued a few degrees off moves every bearing it reports by the same few degrees. Two cameras that disagree by more than the merge window never merge.
- The rig config already takes `intrinsics {fx, fy, cx, cy}`, `image_size`, `position_m` and the three angles. Nothing produced them.
- The project already detects ArUco DICT_4X4_50 markers in numpy only (`oiris/features/aruco.py`). Its corners are whole-pixel, about 1 px off, which is not enough for calibration.

## Decision
1. **The board.**
   - A planar grid of DICT_4X4_50 markers, ids row by row from the top left, each marker 6 x 6 cells (black border round 4 x 4 bits), with a white gap of at least one cell around every marker (the quiet zone).
   - Default: 3 rows x 4 columns, 70 mm markers, 15 mm gaps, 355 x 270 mm, printed on A3 landscape by `scripts/make_calibration_board.py`.
   - The script writes a plain-text SVG, byte-for-byte repeatable, with a title line telling the owner to print at 100 % and to measure the true printed marker side with a ruler. The measured side is what calibration uses (`Board.scaled_to`); the gap scales with it.
2. **The calibration ring.**
   - Four boards around a marked spot on the floor, ahead (azimuth 0), left (90), right (-90) and behind (180), each 0.6 m from the spot to the board centre, centred at eye height, facing the spot squarely.
   - Each board has its own 12 ids (0-11, 12-23, 24-35, 36-47), so a camera knows which board it sees.
   - The wearer stands on the spot facing the ahead board. The head frame of ADR-0060 is then the ring's frame.
3. **Corners.** Detection stays `aruco.detect`, unchanged. Each detected marker's four outer edges are then located to sub-pixel accuracy: profiles across each edge give an area estimate of where ink turns to paper (exact for an area-averaged edge whatever the blur or pixel phase), a line is fitted to each edge, and neighbouring lines meet at the corners. A marker whose edges cannot be found, or whose corners would move more than 3 px, is dropped.
4. **Intrinsics: Zhang's method.**
   - Hold one board at different tilts in front of the camera, 3 or more views (5 or more recommended).
   - Per view, a normalised DLT homography from the refined corners.
   - Closed form for fx, fy, cx, cy with zero skew, then Levenberg-Marquardt refinement of fx, fy, cx, cy and every view's pose on the reprojection error. Numpy only, deterministic.
   - Reported: the intrinsics, the RMS reprojection error in pixels, the number of views and the board tilt spread.
5. **Mount pose.**
   - The wearer stands on the mark; each camera captures several frames.
   - Per frame: the ring board with the most refined markers; its planar pose from the homography and the intrinsics, refined by Levenberg-Marquardt; composed with that board's known head-frame pose, giving the camera's optical frame in the head frame.
   - The optical frame (x right, y down, z forward) becomes the mount frame (x forward, y left, z up), and the rotation splits into yaw, pitch and roll by ADR-0060's `R = Rz(yaw) Ry(-pitch) Rx(roll)`. Position is the camera's optical centre.
   - Frames are combined: start from the medoid frame, drop frames beyond the outlier limits, then take the chordal mean of the rotations (the nearest rotation to their sum) and the mean of the positions, twice.
   - Output: a mount dict `{id, kind, position_m, yaw_deg, pitch_deg, roll_deg, image_size, intrinsics}` that `oiris.config.validate_rig` accepts, plus quality numbers (frames given, dropped and why, used, outliers, board, mean markers, RMS, angle and position spread). It is checked with `validate_rig` before it is returned.
6. **Quality limits** (named constants in `oiris/spatial/calibrate.py`). Below any of them, no numbers are output, only a reason code.

   | Limit | Value | Reason code |
   |---|---|---|
   | Intrinsics views | at least 3 | `too_few_views` |
   | Board tilt spread between views | at least 10 degrees | `views_near_parallel` |
   | Intrinsics RMS reprojection | at most 1.0 px | `rms_too_high` |
   | Refined markers per view or frame | at least 4 | (view or frame dropped) |
   | Mount frames with a ring board | at least 3 | `too_few_frames`, `no_ring_board` |
   | Frame pose RMS | at most 1.0 px | (frame dropped as `rms_too_high`) |
   | Outlier frame | more than 0.5 degrees or 1 cm from the consensus | (frame dropped) |
   | Surviving frames | at least 3 and at least 60 % | `frames_disagree` |
   | Spread of surviving frames (RMS) | at most 0.25 degrees and 5 mm | `frames_disagree` |
   | The resulting mount | must pass `validate_rig` | `invalid_mount` |

7. **What the output feeds.** The mount block is pasted into the `rig.mounts` list of the config (ADR-0060). Nothing reads calibration files at run time; the config is the single record, reviewed like any other config change.
8. **Lens distortion is not modelled yet.** Estimating it needs rig config keys that do not exist. It is the first fine-tuning item (below). Until then, a wide-angle camera shows it as a high RMS and is refused rather than silently mis-calibrated.

## Built now vs later
- **Built now, tested in simulation:**
  - the board model, the ring layout and a pinhole renderer with optional seeded noise and blur (`oiris/spatial/calib_sim.py`);
  - sub-pixel corners, Zhang's intrinsics, planar pose, mount-angle decomposition, outlier rejection and frame averaging (`oiris/spatial/calibrate.py`);
  - the printable board (`scripts/make_calibration_board.py`);
  - the owner's procedure (`docs/CALIBRATION.md`).
- **Measured in simulation** (`tests/test_spatial_calibrate.py`, 640 x 480 cameras): intrinsics within 0.03 % (0.12 % with noise of 8 grey levels and 1 px blur), RMS about 0.05 px; mount angles within 0.05 degrees and positions within 0.6 mm for the front, left, right, rear, pitched and rolled mounts; a head-frame point lands within 1 px of the same pixel through `oiris/spatial/rig.py`. This proves the maths and the code, not real accuracy: real corners, lenses and board placement are worse.
- **Later, needing hardware (the fine-tuning list):**
  1. Lens distortion (radial k1, k2 and tangential p1, p2) for wide-angle glasses and helmet cameras: new rig config keys, a new ADR, and undistortion in `pixel_to_mount_ray`.
  2. Real photos from each camera, and the limits re-checked against them.
  3. Thermal and NIR mounts: a printed board is invisible to a thermal camera; a heated or cut-out board is needed.
  4. Temperature drift checks: the same procedure warm and cold.
  5. Re-calibration after a knock: a quick check against the ahead board at power-on, and a full re-run when it fails.
  6. The wearer's stance: standing off the mark or turned shows up as the same error on every mount. A chin rest or a head-mounted reference would remove it.

## Where it is enforced
- `oiris/spatial/calibrate.py`: the limits, the reason codes and the `validate_rig` check before output.
- `oiris/spatial/calib_sim.py`: the board and ring geometry shared by the script, the procedure and the tests.
- `scripts/make_calibration_board.py`: the printed board matches the board model.
- `tests/test_spatial_calibrate.py` and `tests/test_calibration_board.py`.

## Consequences
- The rig config can carry measured intrinsics and mount angles instead of guesses, and a bad calibration is refused rather than written.
- Bearings from different cameras agree to the calibration's accuracy, so ADR-0060's merging works across cameras.
- The ring has to be set up carefully: a board 1 cm off sideways at 0.6 m, or turned 1 degree, moves that camera's yaw by about 1 degree. `docs/CALIBRATION.md` says how to place them.
- Until distortion is modelled, wide-angle cameras may be refused for a high RMS.

## Changing it
Write a new ADR that supersedes this one, update CHANGELOG.md, and get the owner's sign-off. Loosening a quality limit needs measured evidence from real captures.
