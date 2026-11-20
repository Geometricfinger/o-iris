# Live mode (`oiris run`)

`oiris run` (also the default when you type just `oiris`) runs O Iris against real local
sensors: it resolves adapters by name from the registry, steps the normal Engine
continuously in a background thread (`hz` below, default 1, at most 10), records the session
in the encrypted store and serves the local dashboard on `http://127.0.0.1:8765`. It asks for
the store passphrase (or reads `OIRIS_PASSPHRASE`) and creates the store on the first run or
unlocks it after that. It is a **research** tool: it reports anomaly signatures and sensor
health, never a label about who or what a person is (see `oiris/guard/vocabulary.py`). Live
mode loads the opt-in M-16 face matching (people the user enrolled;
docs/FACE_MATCHING_MODULE_SPEC.md) only for the voice link, when both M-16 and M-17 are on (see
"Voice link in live mode" below), and for the face camera, when M-16 is on and `face_camera` is
configured (see "Camera" below).

## Running it on a Mac

```bash
pip install -e ".[mac]"         # bleak and the pyobjc packages it needs on macOS
export OIRIS_PASSPHRASE='…'     # otherwise you are prompted; the first run creates the key store
oiris run                       # until Ctrl-C / SIGTERM
oiris run --duration 300        # stop after 5 minutes
oiris run --port 8800           # dashboard on another local port
oiris run --no-dashboard        # live mode only
oiris run --config my.json      # adapters, rate, site, data_dir (below)
```

On startup every configured adapter is listed as `available` or `unavailable (reason)`. An
adapter that is unknown, lacks its driver (for example `bleak` not installed, or the
`wifi_macos` adapter not present in this build) or fails to open is **reported, not fatal**:
the run continues with whatever is available, and the dashboard's D5 panel shows the reason.
Started from a terminal, live mode on a Mac has BLE only: `wifi_macos` reports that Wi-Fi
comes only from the O Iris app (docs/MACOS.md). macOS ties the Bluetooth permission to the
app that started the process (here the terminal); without it bleak's scan can hang. The O Iris
app asks for Bluetooth and Location under its own name.

All other commands (`--headless`, `selftest`, `dashboard`, `record`, …) behave as before;
`oiris dashboard` alone still shows the simulated `quiet` scenario (and has no Quit button).

## The live dashboard

The page ("O Iris dashboard") refreshes every 2 s.

- **D1 status** shows the alert level and fusion index only once at least one signal family is
  active. Before that the index is just the fusion model's output on empty input, so instead
  D1 says **No live data yet** (or "Sensor data arriving; no signal family active yet" when
  sensors deliver but nothing crosses a family threshold) and lists the sensors that are not
  `ok`, each with a plain status.
- **D5 sensor health** leads each line with a plain status and what to do, and keeps the raw
  reason below in smaller text. Recognised reasons: waiting for Bluetooth permission, Bluetooth
  denied / restricted by policy / off, Bluetooth driver (`bleak`) missing, Wi-Fi off, and macOS
  withholding Wi-Fi identifiers (→ allow Location for O Iris). Anything else shows the raw
  reason with bus prefixes such as `circuit open: read error:` stripped from the plain line.
- **Quit O Iris** (top right) wakes `runner.wait()` the same way the signal handler does:
  the adapters stop, the sweep is finished with outcome `done` (a deliberate stop; SIGTERM and
  Ctrl-C record `interrupted`), the store is closed and `oiris run` exits 0. The page then shows "O Iris stopped". It is a `POST /quit`
  only, accepted when the form carries the random per-run token embedded in the page (compared
  in constant time) **and** the `Host` is `127.0.0.1`/`localhost` on the dashboard port **and**
  an `Origin` header names that same origin. A wrong or missing token, a missing or foreign
  `Origin`, or another host name gets 403; `GET /quit` gets 405.
- **Every GET** (the page, `/api/latest`, `/face`, `/voice/photo`) needs `Host` `127.0.0.1:<port>` or
  `localhost:<port>`; any other host name gets 403, so a DNS-rebinding page cannot read it.
- **D1** also says when the retention purge last ran and how often it runs ("retention purge last run 14:02, every
  60 min"; see "Retention while running").

## Voice link in live mode

Only when the `M-16` and `M-17` flags are both on (and both opted in, with voices enrolled:
`oiris voice-link enroll`), `oiris run` also listens to the microphone (ADR-0057;
docs/VOICE_LINK_MODULE_SPEC.md, "Live listening"):

- The runner builds the face and voice modules through their gates, with the M-16/M-17 change log
  (`face-chain` subkey), and starts a listener on `voice_link.input_device` (an index, part of a
  name, or `""` for the system default; list them with `oiris voice-link devices`). Needs
  `pip install -e ".[voice]"` and the speaker model (`oiris voice-link fetch-models`).
- Audio stays in RAM: it is cut into utterances (about 0.7 s of quiet ends one; at most 8 s each),
  each is matched against enrolled voices only, and every buffer is zeroed after use.
- A recognised person shows a **Sounds like NAME** card in the dashboard's "Voice link" area for
  `voice_link.card_seconds` (default 10 s), with "Recent face match for this person" when the camera matched
  the same person within `agree_window_s`. The review log gets the usual "heard" entry (30-day
  retention). Unknown voices and close calls leave no trace and are not counted.
- **Announcing (ADR-0058).** A recognised person also plays a chime and "sounds like NAME" on
  this device's speaker (macOS: `afplay` and `say`; elsewhere a generated chime and Flite if
  installed), at most once per person per `voice_link.announce_repeat_s` (60 s). Set
  `voice_link.announce` to `chime` or `off` to change it, and preview it with `oiris voice-link
  test-sound --name NAME`. The microphone input is dropped while it plays. The person marked
  with `oiris voice-link set-self` (you) is never announced or logged. A recording or phone call
  of an enrolled person can trigger it: a match is a hint, not proof.
- `voice_link.listening: "push_to_talk"` keeps the microphone closed until the dashboard's
  **Listen** button (about 8 s per press). It is a `POST /voice/listen` with the same per-run
  token, `Host` and same-`Origin` rules as Quit.
- Several seconds of exact digital silence shows "no sound from the microphone: check System
  Settings > Privacy & Security > Microphone, or the input device" (what macOS delivers when the
  microphone permission is denied). A quiet room does not trigger it.
- Anything that stops the voice link (not opted in, model or `sounddevice` missing, no such
  device, the microphone failing to open) is shown as "Voice link unavailable: reason - what to
  do"; live mode keeps running. `/api/latest` carries the same state under `voice`:
  `state` (`listening`/`idle`/`unavailable`), `reason`, `mode`, `notice`, `listen_left_s`,
  `cards` (`name`, `score`, `face_agrees`, `direction`, `seconds_ago`, and `photo` only when one is stored) and
  `heard` (count this run).
- Stopping live mode stops the microphone before the store is closed. The network guard is
  unchanged: nothing in the voice path opens a socket.
- A push-to-talk window counts microphone time. Blocks dropped while an announcement plays still use it up, so the
  window always ends after the seconds asked for.

### Saying the direction

With a `mic_array` mount in the `rig` config (ADR-0065), the microphone opens with that many channels and a clear
match also says where the voice came from: "sounds like NAME, on your left" (also `ahead of you`, `on your right`,
`behind you`, `above you`, `below you`). The card shows the same words. A direction is added only after a clear match
of an enrolled person, and only when enough of the utterance's speech windows agree (three confident windows, 60 %
of them); otherwise the announcement is exactly "sounds like NAME". It is never stored or logged. Matching uses
channel 0 only. A device that cannot open the channels falls back to mono, with a log line and no direction; without a
`mic_array` mount (the Mac) everything is mono, exactly as before.

### Photo on the card

A recognised person who has a stored photo (consent scope `photo`, ADR-0057 decision 4) shows it on their heard
card. The card carries a random key for this run, never the person's ID. The image comes from
`GET /voice/photo?k=KEY&token=TOKEN`, which is served only:

- with the per-run token of the page (compared in constant time);
- from this same origin: an `Origin` naming this dashboard, or, since browsers send no `Origin` on a same-origin
  image request, `Sec-Fetch-Site: same-origin`. A header that is present must agree; a request with neither is 403;
- for a person on a heard card shown right now. Any other key, or a card that has expired, gets 404;
- with `Cache-Control: no-store`, `Content-Type: image/jpeg` and `X-Content-Type-Options: nosniff`.

There is no list of photos. No photo means no image element. `oiris face-match list` shows "photo: yes/no" (never the
image), and `oiris face-match photo-delete --person ID` erases one person's photo, verified, keeping the enrollment.

## Outputs and motion

Every output in live mode goes through one output arbiter (docs/ARCHITECTURE.md, "Outputs and
motion"): the voice-link announcement plays through it, so it never overlaps another sound, and
sensor alerts become a pop-up (`/api/latest` → `output.popup`) rather than speech. The Mac stays
silent for sensor signatures unless the config sets `"output": {"speak_alerts": true}`. An
adapter named `imu` (simulated today: `{"name": "imu", "args": {"profile": "walking"}}`, also
`still` and `driving`) feeds the motion gate: while walking no pop-ups, while moving fast only
confirmed and panic. With no IMU, motion is `unknown` and counts as still. `output` in
`/api/latest` also carries `motion`, `imu`, `speaking`, `queued` and `counts` by outcome.

## Directions

When spatial adapters run (ADR-0060; simulated today with the `sim_rig` adapter and a `rig`
config section), each step's lens-glint (A1), infrared-emitter (A2) and heat-pattern (A3)
signatures are matched with the spatial event of the same kind nearest in time (within 0.5 s).
A match is a cue: `/api/latest` → `cues` lists its signature, direction word and evidence, and
`glasses.cue_direction` is the current one. The glasses show it as a small edge marker: left or
right edge for `at left` / `at right`, the whole bottom edge for `behind`, the top centre for
`above`, the bottom centre for `below`, nothing for `ahead`. The marker stays while walking (it
is not a pop-up); the alert text still hides. The phrase, for example "Candidate camera lens
behind", is spoken only with `"output": {"speak_alerts": true}`. A distance is said only when a
sensor measured a range ("Candidate camera lens behind, two metres"); without one the phrase names
the direction only, never a guessed "far". No match means no cue. The dashboard's "Located" list shows the current spatial events:
direction, kinds, how many sensors agree and seconds ago. Cues describe signatures, never people.

## Capture

The camera and the microphone run only while the capture interlock allows it (ADR-0063, `oiris/capture/`).

- The microphone opens only after a mic grant. On the rig, glasses or helmet that needs the outward-facing
  recording light to be confirmed on. It also needs the mute switch to be open: the microphone is refused while
  muted, and while the switch cannot be read if `capture.mute_sense` is `required`.
- Every microphone block is checked. A non-zero sample while the switch is on is a mute breach: the hardware cut
  failed. The microphone then stays off until restart.
- Every step re-checks. A light that stops confirming revokes every grant. Switching to mute closes the microphone
  within one block. Ambient listening comes back when the switch opens again; push-to-talk is refused while muted.
- After the last grant ends, the light stays on for `tail_s` (default 2 s).
- `/api/latest` → `capture` holds codes and the profile: `indicator` (`on`/`off`/`tail`/`failed`/`not_required`),
  `mute` (`muted`/`open`/`unknown`/`not_present`), `camera` and `mic` (`active`/`off`/`refused:<code>`) and
  `fault` (`null`/`mute_breach`). The dashboard shows them as one "Capture" line in plain words.
- On the Mac (the default `"capture": {"profile": "mac"}`) there is no O Iris light, because macOS shows its own
  camera light and microphone dot, and no mute sense. Live mode behaves exactly as before.
- For the rig, set `{"capture": {"profile": "rig", "indicator_pin": <BCM pin>}}`. The mute sense is GPIO23. The
  GPIO adapters have not yet been tested on hardware. Other keys are `indicator`, `mute_sense`,
  `confirm_timeout_ms` (50-2000, default 200), `tail_s` (0-10), `indicator_adapter` / `mute_adapter`
  (`none`/`sim`/`gpio`) and `mute_pin`.

## Camera

The face camera (ADR-0057 decision 5, `oiris/adapters/camera.py`) lets face matching (M-16) see who is in front of
you, so a voice card can say **Recent face match for this person**. It is off unless you turn it on.

How to turn it on:

1. Opt in to face matching and enroll people as usual: set `"flags": {"M-16": true}` in the config, then run
   `oiris face-match enable` (the typed opt-in) and `oiris face-match enroll`.
2. Add the adapter to live mode:

   ```json
   {"flags": {"M-16": true},
    "live": {"adapters": [{"name": "ble_scan"}, {"name": "face_camera", "options": {"device": 0, "fps": 2}}]}}
   ```

   `device` is the camera index (default 0). `fps` is frames a second, 0.2 to 10, default 2. Needs OpenCV
   (`pip install opencv-python-headless`); without it the camera is shown as unavailable and live mode goes on.
3. In the Mac app, the launcher asks macOS for camera access only when the app config has both of these. It never
   asks otherwise.

What happens:

- With M-16 off, the camera is refused: "Camera: off (face matching is off)". It is never opened.
- The camera opens only after the capture interlock grants it (see "Capture"): on the rig or glasses, after the
  recording light is confirmed on. A refusal is tried again every 5 s; a light that fails closes the camera at once
  and drops its frames.
- With the power governor running (see "Power and heat"), frames are limited to the plan's `camera_max_fps` and
  `frame_stride`. At hot and critical the plan turns face matching off, and the camera closes until it is allowed
  again.
- Each frame goes to M-16, which compares faces only with the people you enrolled. A match is written to the
  face-match review log (the encounter entry M-16 always makes) and remembered in memory for a few seconds
  (`voice_link.agree_window_s`, default 5 s). A voice card for the same person within that window says
  "Recent face match for this person".
- The dashboard shows one "Camera" line: "Camera: on, 2 frames a second", "Camera: off (face matching is off)",
  "Camera: waiting for the recording light", "Camera: paused to cool down" (or "to save the battery"), or
  "unavailable" with the reason. `/api/latest` → `camera` holds `state`, `code`, `reason`, `fps` and `frames` (how
  many frames M-16 has seen this run). Without `face_camera` in the config there is no line and nothing changes.

What it never does:

- **No alerts.** A face match makes no sound, pop-up, announcement, haptic or event. It only marks a voice card
  "Recent face match for this person" (docs/FACE_MATCHING_MODULE_SPEC.md section 6).
- **No lookup of strangers.** People you did not enroll are never named or identified. In research mode they are only
  counted (place, date, time bucket); otherwise they leave no trace.
- **No frames saved.** Frames are held in memory only. They are never written, logged or recorded, the engine zeroes
  each one after M-16 has seen it, and frames that are not used are zeroed and dropped.

## Power and heat

The power governor (ADR-0061, `oiris/power/`) runs in live mode when the config names a reading source:

```json
{"power": {"source": "sim:hot_day", "interval_s": 10}}
```

- `power.source`: `"none"` (the default: no readings, no governor, live mode exactly as before), `"macos"` (battery
  and charging from `pmset -g batt`; a Mac reports no temperature) or `"sim:<scene>"` with a scene of
  `oiris/adapters/power_sim.py` (`cool_room`, `hot_day`, `low_battery`, `charging_hot`; the simulation runs on live
  mode's clock under the plan in force, so throttling really cools it).
- `power.interval_s`: one reading every this many seconds, 1 to 600, default 10 (a sixth of the governor's 60 s
  staleness limit). The governor's thresholds and `plans` sit in the same section, as in ADR-0061.
- Each reading goes to the governor, and its plan is applied at once:
  - `listen_duty` / `listen_period_s` go to the voice listener: it listens for the first duty share of every period.
    While it is off, no new utterance starts. An utterance already being heard is finished and matched, never cut.
    The microphone itself stays as the capture interlock set it.
  - `voice_match` off (the critical level) pauses voice matching the same way. The dashboard says "voice matching
    paused".
  - `camera_max_fps`, `frame_stride` and `face_match` go to the face camera when one is configured (see "Camera").
  - The other fields (`ir_illuminator`, `display_brightness_cap`, `radio_scan_interval_s`, and the three camera
    fields without a face camera) have no live consumer yet. They are not dropped: they are in the snapshot's plan and
    listed under `no_consumer` for the glasses paths.
- `/api/latest` → `power`: `source`, `level` (`normal`/`saver`/`hot`/`critical`), `reasons` (codes such as `skin_hot`,
  `battery_low`, `charging`, `stale`), `plan`, `battery_pct`, `charging`, `skin_c` and `no_consumer`. Numbers and
  codes only. A level change is logged once.
- The dashboard shows one line, for example "Power: normal, battery 64 %, charging" or "Power: hot, throttled (warm
  against the skin), battery 97 %, skin 41.3 C · voice listening 25 % of the time". With the source `none` the line is
  hidden.
- A failing reading is logged and keeps the last plan; it never stops live mode.

## Latency budgets

Every step, the budget monitor (ADR-0061, `oiris/power/budgets.py`) compares each stage's p95 (see "Timing") with
its budget: config `budgets` (`{"<part>.<stage>": ms}`) laid over the defaults `engine.step` 50, `voice.queue_wait`
1000, `voice.decision` 1000, `output.chime_wait` 1500 and `output.speech_wait` 1500 ms. A stage counts only after 20
samples.

- `/api/latest` → `budgets` lists the stages over budget (`part`, `stage`, `p95_ms`, `budget_ms`, `count`); `[]` when
  none is.
- The snapshot's health map then gains a `latency` entry, `degraded`, with a plain reason ("over budget: voice
  decision p95 2000 ms (budget 1000 ms)"). It is a measurement, not a sensor: the dashboard shows it as its own
  "Latency over budget" line, only while something is over, and never in D1's waiting list or D5.
- A breach is logged once per change. It never changes behaviour, raises an alert or reaches the glasses.

## Timing

The plan has almost no latency budgets yet. Budgets have to come from measurements, so every stage
that will need one reports its own timing first (`oiris/timing.py`). Each part has its own
`StageTimer`: a ring buffer of the last 512 durations per stage, summarised as `count` (every
record this run), `p50_ms`, `p95_ms` and `max_ms` (over the kept window). `snapshot()["timing"]`
and `/api/latest` carry `{"engine": {...}, "voice": {...}, "output": {...}}`, keyed by stage.
Numbers only: no names, audio or identifiers. A part that has not run yet is `{}`.

- **engine**: `step` (the whole step), and inside it `collect` (bus drain), `face_match` (M-16
  on only), `health` (bus health and staleness), `scores`, `fusion`, `guard` (the vocabulary
  check of the result) and `events` (only when a level is raised: the event, its check and the
  live handler, which includes the sighting write and the alert request).
- **voice** (voice link listening): `queue_wait` (a microphone block queued until the worker
  takes it), `decision` (utterance end to the match result: copy, lock wait and the core),
  `lock_wait` (waiting for the store lock), `process` (the whole `process_audio`), and `embed` /
  `spoof` (the speaker model calls inside it). The core's VAD is not timed on its own; it is
  inside `process`.
- **output** (the arbiter), per sound channel: `chime_wait` / `speech_wait` (request accepted
  to the start of that sound; speech after a chime includes the chime) and `chime_play` /
  `speech_play` (how long the player took). Pop-ups and haptics are immediate and not timed.

The timers cost about a microsecond per stage and change no output (`tests/test_timing.py`
checks both, and the golden suite is unchanged). The only targets in the spec today are R-17 /
E-20 (docs/spec/ASB-Build-Spec-v2.md): Attention speech starts within 1.5 s p95, and a
Confirmed barge-in cuts speech within 300 ms p95. They have not been measured on the target
hardware (no glasses yet); these counters are how they will be.

## Configuration

Live settings live under a `"live"` key in the config file given with `--config` (the normal
`Config` loader ignores this key; `oiris/live.py` reads it). No file, or no key, means the
defaults shown here:

```json
{
  "data_dir": "~/.asb",
  "live": {
    "adapters": [{"name": "ble_scan", "options": {}}, {"name": "wifi_macos", "options": {}}],
    "hz": 1.0,
    "site": "live"
  }
}
```

- `options` are passed to the adapter's `open(cfg)` (e.g. `self_allowlist` for `ble_scan`).
  `site_key` is always derived from your key store (`site:<site>` subkey) and cannot be set
  from the file, so raw MAC/BSSIDs are hashed with a key that never leaves the store.
- `args` (optional) are passed to the registry factory, e.g. a sim or replay rig:
  `{"name": "sim", "args": {"scenario": "quiet", "period_s": 0.2}}` or
  `{"name": "replay", "args": {"adapter_id": "rep", "path": "clip/sim_ble.jsonl", "caps": ["rf_adv"]}}`.
- `hz`: engine steps per second, `(0, 10]` (`MAX_HZ` in `oiris/live.py`).
- `site`: the site code the sweep is recorded under.

## What it records

Everything goes to the existing encrypted SQLite store in `data_dir` (`asb.db`, key from
`keys.json` unlocked with your passphrase); no new storage format.

- **One sweep per live session** (M-14 `SweepTimeline`, table `sweep`, sealed body). It is
  opened at start with outcome `open`, checkpointed every 10 steps (still `open`), and
  finished on stop as `done` (duration reached, or the dashboard's Quit) or `interrupted`
  (Ctrl-C / SIGTERM / SIGINT). The body holds counts only: steps, signature codes seen, alert
  levels, `events`, `sightings`, `sightings_suppressed`, `purged_rows`, the unavailable adapter
  names, and the signature codes new since the previous finished sweep at the same site. A crash
  leaves the sweep `open` with its last checkpoint. The dashboard's `/api/latest` shows the same
  four record counts under `records`.
- **Stopping** is bounded: the loop thread gets up to 5 s, then the adapters are stopped and
  closed (so blocked reads can return) and the loop gets up to 5 s more. The store is then
  sealed: every write (sighting, checkpoint, step counts) checks a closed flag under one lock,
  so nothing from the loop reaches the store after the final sweep record. If the loop is still
  inside a step, the sweep body records `loop_alive_at_stop: 1`, the late step and any alert it
  raises are dropped, and the DB is closed by the loop thread when it exits, never underneath it.
- **Alert events** (when fusion raises a level) go to the M-05 `SightingLog` with
  `fix_quality: no_fix` (live Mac mode has no GPS; positions are never guessed). Every event is
  counted. A sighting is written when the (signature, level) pair changes; an unchanged pair is
  written at most once per 60 s, and the rest are counted as suppressed.
- **Retention.** The purge (`oiris/guard/retention.py`, config `retention_days`, default 30)
  runs when the store opens and then while live mode runs (see "Retention while running").
- Turning off M-14 or M-05 in `flags` turns off the matching record.

### Record chain

The sighting log is one tamper-evident chain across sessions (ADR-0066). When live mode opens the store it verifies
the stored chain and either continues it or, if the current segment does not verify, starts a new segment with a
recorded reason. Past rows are never rewritten.

- Every step gives the chain its 60 s idle anchor (`tick()`); stopping anchors the session's last records (`close()`,
  under the store lock, before the DB closes), so the next session finds no unanchored tail.
- `/api/latest` → `chain` is the open-time status: `state` (`new`, `resumed` or `restarted`), `reason` (when
  restarted: `legacy`, `edited`, `deleted`, `reordered`, `truncated`, `bad_anchor`), `segment`, `records`, `anchors`,
  `first_seq`, `last_seq` and `unanchored_tail`.
- The dashboard shows "Record chain: continued (N records)" or "Record chain: restarted (reason in plain words)".

### Retention while running

Glasses and Pi rigs run for days, so the purge does not wait for a restart. It runs at store open and then every
`retention_check_s` seconds (top-level config key, 60 to 86400, default 3600), under the store lock. With the sighting
log on it goes through `SightingLog.purge`, which chains a purge boundary first, so the purged range still verifies;
without it, the plain retention purge runs as before. A failure is logged and never stops live mode. `/api/latest` →
`records` carries `last_purge_at` (wall time of the last successful purge) and `retention_check_s`, and D1 shows the
time.

Raw samples, frames and per-step results are not stored. The dashboard keeps only the latest
result in memory.

## Network

Live mode uses local radios but opens no outbound connections. While it runs, Python-level
socket `connect`, `connect_ex`, `sendto` and `sendmsg` are refused for anything other than
loopback or local (AF_UNIX) targets, and the dashboard binds `127.0.0.1` only. This guard covers
Python sockets only: it does not police subprocesses or native code inside drivers.

Radio use is normal client-radio activity, which the spec allows. `ble_scan` calls bleak's
`BleakScanner.discover()` with bleak's default (active) scanning on every platform, so it may
send BLE scan requests; it does not go through `guard/radio.py`. On Linux, `wifi_beacons` runs
`iw dev <if> scan` through `guard/radio.py` (allow-listed arguments, monitor-mode refusal); `iw`
has no passive keyword. On macOS, `wifi_macos` starts no scan: the O Iris.app launcher (Swift)
reads macOS's cached scan and pipes it in. O Iris never connects to or pairs with others'
devices and uses no monitor mode, injection, deauthentication or jamming.

## Limits (read this)

- Research signals only. No study shows this stack detects anything specific in the wild;
  thresholds are proposals and default fusion weights are a hand-set prior.
- A Mac alone feeds one signal family: rf, from BLE (no score module uses the Wi-Fi AP scans
  yet). One family can raise FYI; ATTENTION needs ≥2 families. Observed on a real Mac
  (2026-11-15): with one strong Bluetooth device nearby the level stayed at FYI with signature
  A4 for hours. That is a research signal about a nearby emitter, nothing more.
- Observed live on an Apple Silicon Mac (macOS 26.5, 2026-11-15, O Iris.app after the user
  allowed Bluetooth and Location): `ble_scan` delivered real samples and `wifi_macos` stayed OK
  with a new sample about every 15 s (feed: 67–72 networks, 0 withheld). The runner's own tests
  use sim and replay adapters (`tests/test_live.py`); the Mac adapters are tested with fakes
  (`tests/test_macos_adapters.py`). Nothing here is validated against real subjects.
- The power governor, the duty plans and the latency budgets are tested in simulation only (`power_sim` scenes, fake
  clocks); every threshold and budget is provisional until measured on the glasses or helmet (ADR-0061). The record
  chain detects edits, deletions and reordering over the retained history; deleting the unanchored tail together with
  nothing else is out of its reach until a remote witness exists (ADR-0066).
- Never point it at people who have not agreed to it; read ETHICS.md.
