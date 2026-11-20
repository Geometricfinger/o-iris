# Calibrating the cameras on the glasses or helmet

This tells each camera on the glasses or helmet two things:
- its lens numbers (the intrinsics);
- exactly where it sits on the head and which way it points (the mount pose).

The result is a block of numbers you paste into the `rig` section of the config.

The decision behind this is [ADR-0064](adr/0064-mount-calibration.md). It has only been tested in simulation so far; the first real run is part of fine-tuning.

## What you need
- A printer that can print A3 (or A4 with smaller markers), and fairly stiff card or foam board to stick the prints on.
- A steel ruler and a tape measure.
- Masking tape to mark a spot on the floor.
- Somewhere with about 1.5 m of clear space all round, evenly lit, without glare on the boards.
- The glasses or helmet, running, with a way to save camera frames.

## 1. Print the boards
Print five boards: four for the ring, and one more for the lens step (it can be one of the ring boards).

```
python scripts/make_calibration_board.py --ring ahead  --out board_ahead.svg
python scripts/make_calibration_board.py --ring left   --out board_left.svg
python scripts/make_calibration_board.py --ring right  --out board_right.svg
python scripts/make_calibration_board.py --ring behind --out board_behind.svg
```

- Print each one at **100 %**. Turn off "fit to page".
- Glue it flat onto stiff board. A bent board gives wrong numbers.
- If you only have A4, add `--paper a4 --marker-mm 40` to every command.

## 2. Measure
Printers never print exactly to size, so:
- Measure the side of one black marker square with the steel ruler, outer edge to outer edge, in millimetres. Measure three markers and take the average.
- Write it down. You will enter it as the marker size, not the size the script printed in the title.

## 3. Find each camera's lens numbers
Do this for each camera, one at a time. The glasses do not need to be worn.
1. Hold one board 30-60 cm from the camera.
2. Take about eight frames. Between frames, tilt the board a different way, about 20-30 degrees: top towards the camera, bottom towards it, left, right, and a couple of diagonals. Keep the whole board in the picture and sharp.
3. Run the intrinsics step (`oiris.spatial.calibrate.calibrate_intrinsics`) with your frames and the board at your measured size.
4. Check the result:
   - it says `ok`;
   - the RMS is below 1 pixel. Below 0.3 is good.

If it refuses:
- `views_near_parallel`: tilt the board more between frames.
- `too_few_views`: the board was not found in enough frames. Get closer, add light, or hold it still.
- `rms_too_high`: blur, a bent board, or a very wide lens. See "Known limits".

## 4. Set up the ring
1. Tape a cross on the floor: the mark.
2. Place the four boards facing the mark:
   - **ahead** straight in front;
   - **left** to your left;
   - **right** to your right;
   - **behind** behind you.
3. Make each board's centre exactly **60 cm** from the point above the mark between your eyes, measured level, and at your eye height.
4. Square each board to the mark. Measure from the mark to the board's left and right edges; they must be equal to within 5 mm.
5. The four boards must be at right angles to each other. A board 1 cm off sideways, or turned by 1 degree, makes that camera's angle about 1 degree wrong.

## 5. Capture
1. Put on the glasses or helmet as you normally wear them.
2. Stand on the mark, facing the ahead board. Stand straight and look level at the centre of the ahead board.
3. Keep still and take at least five frames from every camera at once.
4. Each camera should see its own board: the front camera the ahead board, the left camera the left board, and so on. A camera that sees no board cannot be calibrated from here.

## 6. Check the numbers
Run the mount step (`oiris.spatial.calibrate.calibrate_mount`) for each camera, with:
- its frames;
- its lens numbers from step 3;
- the ring (`calib_sim.ring_layout`) built with your measured marker size and distance.

Each camera gives either a mount block or a refusal. Accept the mount block only when:
- it says `ok`;
- `frames_used` is at least 3 and `frames_rejected_as_outliers` is small;
- `rms_px` is below 1. Below 0.3 is good;
- `angle_spread_deg` is below 0.25 and `position_spread_m` is below 0.005;
- the angles make sense: the left camera's yaw is near +90, the right camera's near -90, the rear camera's near 180.

If it refuses:
- `frames_disagree`: you moved, or the camera is loose. Capture again.
- `no_ring_board`: that camera cannot see its board. Move the board, or check the camera.
- `invalid_mount`: the numbers are impossible for a head-worn camera. Check the ring distance you entered.

## 7. Paste the mount block
Each accepted camera gives a block like this:

```json
{"id": "left", "kind": "camera", "position_m": [0.0, 0.08, 0.02], "yaw_deg": 90.0, "pitch_deg": 0.0, "roll_deg": 0.0,
 "image_size": [640, 480], "intrinsics": {"fx": 268.5, "fy": 312.8, "cx": 320.0, "cy": 240.0}}
```

Paste it into the `rig.mounts` list of your config, replacing that camera's old entry. Then:
- add back any `ir_illuminator` and `ir_eye_safety_approved` settings the old entry had. The eye-safety rules of ADR-0060 still apply;
- start O Iris. It checks the config and refuses a bad entry.

## When to do it again
- After a camera is knocked, re-glued or replaced.
- If directions start to look wrong (something ahead is reported "at left").
- After a large temperature change, until drift has been measured.

## Known limits
- **Lens distortion is not corrected yet.** Wide-angle cameras bend straight lines, especially near the picture edges. Until the config has distortion settings, such a camera may be refused with `rms_too_high`. Keep the board near the middle of the picture as a workaround. Adding distortion is the first fine-tuning item.
- **Thermal cameras** cannot see a printed board. They need a heated board, which is later work.
- **How you stand matters.** If you stand off the mark or turned, every camera is off by the same amount.
