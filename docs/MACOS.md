# macOS app

CI (`build` job, macOS runner) produces `o-iris-<commit>-macos-app-adhoc-signed.zip` containing
`O Iris.app` (built by `scripts/make_macos_app.py`). It has no **Developer ID** signature (there is
no paid Apple Developer account), but the build gives it an **ad-hoc** signature; see "Code identity"
below. Double-clicking the app runs **live mode** (`oiris run`, see docs/LIVE.md): the real Mac
sensors (`ble_scan` on CoreBluetooth, `wifi_macos`) plus the local dashboard on http://127.0.0.1:8765/.

**It runs only on Apple Silicon with macOS 14 or later.** The launcher is built for
`arm64-apple-macos14.0` and the bundled wheels are arm64. Intel Macs are not supported.

## Building it on this Mac
```bash
python3 -m pip wheel --no-deps -w dist/local .            # the project wheel
python3 scripts/make_macos_app.py --wheel dist/local/o_iris-0.1.0-py3-none-any.whl \
        --out dist/local --commit "$(git rev-parse HEAD)" --bundle-deps
```
`--bundle-deps` is the only step that uses the network: it downloads binary wheels for
`o-iris[mac,voice]` (numpy, cryptography, PyNaCl, noiseprotocol, bleak and the pyobjc packages bleak
needs on macOS, plus the M-17 voice link's sherpa-onnx, sherpa-onnx-core and sounddevice with its
cffi) for Python 3.11–3.14 into `Contents/Resources/wheels`. No speaker model and no OpenCV are
bundled (see "Voice link in the app"). Without it the bundle
holds only the O Iris wheel and the first launch cannot install. After building, the app never
goes online.

If Xcode or the Command Line Tools are installed (`xcode-select -p` succeeds), the script also
compiles two Swift programs with the same flags (`-O -target arm64-apple-macos14.0`):
`macos/launcher/main.swift` into `Contents/MacOS/o-iris`, the app's executable, which asks for
Bluetooth and Location itself and feeds Wi-Fi to oiris (see "Code identity", "Permissions" and
"Wi-Fi" below), and the M-15 Keychain gate helper (`macos/keychain_gate/`) into
`Contents/MacOS/keychain-gate`. The bash launcher then lives in
`Contents/Resources/o-iris-launcher.sh`. Without Swift it prints a note and builds the earlier
layout: the bash script itself is `Contents/MacOS/o-iris`. Such an app starts, but nothing asks for
the permissions, macOS has no proper code identity for them, and there is no Wi-Fi source. A real
compile error fails the build.

Last, the script ad-hoc signs the finished bundle (`codesign --force --sign -`; the identifier is the
Info.plist `CFBundleIdentifier`, `io.github.geometricfinger.oiris`). It then runs
`codesign --verify --deep --strict` against that identifier, and a failed verification fails the
build. There is no hardened runtime (`--options runtime`): it would stop the launcher's unsigned child
Python from running. Without `codesign` (not macOS) it prints a note and leaves the app unsigned.
`--no-swift` (or `OIRIS_SKIP_SWIFT=1`) skips both the compiles and the signing on purpose; the tests
use this. **Anything changed inside the bundle after signing breaks the signature.**
`scripts/build_all.sh` therefore changes nothing in the bundle after `make_macos_app.py`: it runs the
`keychain-gate` checks, verifies the signature again (`codesign --verify --deep --strict`), and zips
the app with `ditto`. The Android APK from the same script stays unsigned.

## Running the app
1. Install Python 3.11–3.14 (python.org or Homebrew). Finder starts apps with a minimal `PATH`, so
   the launcher looks for `python3.14` … `python3.11` in the python.org framework, `/opt/homebrew/bin`,
   `/usr/local/bin` and `PATH`; set `OIRIS_PYTHON` to force one.
2. Move `O Iris.app` to Applications. An app you built on this Mac opens normally. A copy
   downloaded onto another Mac has only an ad-hoc signature (not Developer ID, not notarized), so
   Gatekeeper blocks the first open. On macOS 14, Control-click the app → Open → Open. On macOS 15
   and later, try to open it once, then use System Settings → Privacy & Security → Open Anyway.
   Alternatively: `xattr -dr com.apple.quarantine "/Applications/O Iris.app"`.
3. **Permissions.** At once, before anything else, macOS may ask whether *O Iris* may use Bluetooth
   and whether it may use your location (see "Permissions" below). These prompts come only while
   the answer is undecided; the app keeps starting while they are open.
4. First launch creates `~/Library/Application Support/O Iris/venv` and installs `o-iris[mac,voice]`
   from the bundled wheels offline (`pip --no-index`). It reinstalls only when the app version changes.
5. **Passphrase.** The encrypted store (`~/.asb`) needs one, and a Finder launch has no terminal, so a
   hidden-input macOS dialog asks for it. The first run asks twice to create the store (both entries
   must match; it cannot be recovered if lost). Later runs ask once and check it before starting
   (three tries). Cancel quits without starting anything. If `OIRIS_PASSPHRASE` is already set in
   the environment, no dialog appears. The passphrase goes to oiris only through the environment of
   the exec'd process. It is never written to disk, to the log or onto a command line.
6. The launcher then `exec`s `oiris run --port 8765` (`OIRIS_PORT` changes the port), adding
   `--config "~/Library/Application Support/O Iris/config.json"` if that app config exists (see
   "Voice link in the app"), and opens the dashboard in your browser once the port answers. If an instance is already running, a second
   double-click just opens its dashboard.
7. To stop it, use the dashboard's **Quit O Iris**: the sweep is recorded as `done`. Quitting the
   process instead (Activity Monitor: the `o-iris` launcher, which passes SIGTERM/SIGINT on, or
   `pkill -f "venv/bin/oiris run"`) records the sweep as `interrupted` (docs/LIVE.md). The `o-iris`
   launcher exits when oiris does, with its exit status.

**Apple Silicon: runs native (arm64).** The compiled launcher is arm64, so bash, Python and the
bundled arm64 wheels run native under it. macOS may start an app whose executable is a shell script
(the no-Swift layout) under Rosetta; then `uname -m` and the universal2 python.org Python both
report x86_64 and the bundled arm64 wheels (numpy, cffi, cryptography) fail to install ("Could not
find a version that satisfies the requirement numpy>=1.26", seen on the first Finder launch on this
Mac, macOS 26.5). So `Info.plist` sets `LSArchitecturePriority` to `arm64, x86_64`, and as a guard the
bash launcher's first step re-`exec`s itself with `/usr/bin/arch -arm64` if `sysctl.proc_translated`
is 1 (at most once). Under the compiled launcher the guard does not fire. The log's `start` line
records the architecture.

**Log:** `~/Library/Logs/O Iris/o-iris.log` (the launcher's steps plus everything oiris prints,
including each adapter's `available` / `unavailable (reason)`). It rotates to `o-iris.log.1` past 5 MB.
The compiled launcher appends its own `o-iris:` lines there: its start with the Bluetooth
authorization, then every change of the Bluetooth state/authorization and of the Location
authorization (for example `o-iris: bluetooth state poweredOn, authorization allowedAlways` and
`o-iris: location authorization authorizedWhenInUse`), and each change of its Wi-Fi feed state with
counts only (`o-iris: wifi feed: ok, 67 networks (0 withheld)`; see "Wi-Fi" below). At each start
it also logs the microphone authorization and whether the app config turns the voice link on (for
example `o-iris: microphone authorization notDetermined; voice link: no app config`).

## Code identity: compiled launcher + ad-hoc signature
macOS ties privacy permissions (Bluetooth, Location) to the code identity of the *responsible*
app, normally the signed executable that started the process tree. Earlier builds had no such
identity: the bundle was unsigned (`codesign -dv` said "code object is not signed at all") and its
`CFBundleExecutable` was a bash script. On this Mac (macOS 26.5) O Iris therefore never appeared in
System Settings → Privacy & Security → Bluetooth and macOS never asked on its behalf. Bluetooth
ended up working under some other identity, Location stayed "not determined", and Wi-Fi identifiers
stayed withheld. So now:
- `Contents/MacOS/o-iris` is a small compiled Swift launcher (`macos/launcher/`). It runs the bash
  launcher as a **child** process and stays alive until it ends. It never `exec`s, so the Mach-O
  remains the responsible process for bash and `oiris run`.
- The whole bundle, including `keychain-gate`, is ad-hoc signed with the identifier
  `io.github.geometricfinger.oiris`.
- The launcher itself asks for Bluetooth and Location (next section). A signed identity alone was
  not enough: with the first signed build (macOS 26.5), O Iris still appeared under neither
  Bluetooth nor Location and no prompt named it. The requests came from processes that are not the
  app's main executable: the Python child (`CBManager.authorization()` stayed `notDetermined`, so
  the dashboard kept "waiting for Bluetooth permission") and a separate CLI Location helper without
  its own Info.plist (it returned `notDetermined` after its 20 s wait). macOS dropped both silently.
  With the earlier unsigned build, Bluetooth had been granted to "Python" instead of O Iris, because
  the Python process was then the one macOS treated as asking. The helper has been removed.

**Ad-hoc is not Developer ID.** An ad-hoc signature gives macOS a code identity on this Mac, but
it proves nothing about who built the app, so Gatekeeper still blocks the first open on other Macs
(see "Running the app"). An ad-hoc identity is the exact code hash, so **every rebuild makes macOS
ask for Bluetooth and Location again** (seen three times on 2026-11-15), and old entries may need
removing in System Settings. A Developer ID signature (below) would keep the grants across rebuilds.

## Permissions macOS asks for, and what it withholds
The compiled launcher, the app's main executable, asks for both permissions at launch, before it
starts the bash launcher, so on a first run the prompts appear right away (while the rest starts
behind them). Its main thread then runs the main run loop so macOS can deliver the answers, and it
logs each change. The Python side also asks for Bluetooth: while the Bluetooth authorization is
undecided, `ble_scan` creates a CoreBluetooth central manager once per adapter open (`BlePermission`
in `oiris/adapters/macos.py`, no scan) and then only polls the authorization, with no scan until it
is decided. Earlier builds showed macOS can drop such a request from the child silently; the
launcher's request is the one that reliably names O Iris. Python never asks for Location.
- **Bluetooth (required for `ble_scan`).** The launcher creates a CoreBluetooth central manager
  (with no "Bluetooth is off" alert). While the authorization is undecided this makes macOS show
  "“O Iris” would like to use Bluetooth" with the purpose string (`NSBluetoothAlwaysUsageDescription`:
  O Iris reads nearby Bluetooth advertisements using macOS's normal Bluetooth scanning, never
  connects to or pairs with devices, hashes identifiers with a local key, and nothing leaves the
  Mac). macOS's Bluetooth scanning is active (bleak: "macOS does not support passive scanning"), so
  `ble_scan` may send BLE scan requests: normal client-radio activity, which the spec allows. The
  launcher never scans or connects; only `ble_scan` in the Python child scans, once allowed. Without
  that plist key macOS kills a process when CoreBluetooth starts. macOS attributes the permission
  to the responsible app, so oiris stays a descendant of the signed `o-iris` launcher (bash `exec`s
  oiris; nothing goes through `open`/`launchctl`) and the dashboard's "waiting for Bluetooth
  permission" clears once you allow it. That is also why running `oiris run` from a terminal hung in
  bleak's discover on this Mac (2026-11-15): the terminal had no Bluetooth authorization for it.
  Find or change the grant in System Settings → Privacy & Security → Bluetooth, where the entry is
  named **O Iris**.
- **Location, When In Use (optional; it only unlocks Wi-Fi network IDs).** macOS hides Wi-Fi
  BSSIDs and SSIDs from any process without Location Services permission. While the authorization
  is undecided the launcher calls `requestWhenInUseAuthorization()`: macOS shows the Location prompt
  for O Iris with the plist's purpose string (`NSLocationUsageDescription` /
  `NSLocationWhenInUseUsageDescription`). Startup never waits for the answer. Find or change it in
  System Settings → Privacy & Security → Location Services (entry **O Iris**).
  **On the Mac, O Iris never reads your position.** The launcher only reads and requests the
  authorization *status*. It never starts location updates or asks for a location fix
  (`tests/test_macos_app.py` checks the launcher source for this), and live mode records every
  sighting with no GPS fix (`no_fix`), so nothing on the Mac reads, stores or sends coordinates. (The
  Pi rig's NMEA GPS adapter, which stamps sightings when it has a fix, is not used on the Mac.)
  Until Location is allowed (undecided or denied), `wifi_macos` shows "waiting for Location
  permission (macOS hides Wi-Fi network IDs until then): answer the macOS prompt, or allow O Iris in
  System Settings > Privacy & Security > Location Services" as degraded health in the dashboard's
  D5 panel, with no samples and no failure, and the first reading after the grant counts at once.
  Either way live mode carries on with Bluetooth.
- **Never available:** Wi-Fi beacon TSF timestamps. macOS does not expose them, so the AP
  clock-skew family stays masked on a Mac.

## Wi-Fi: the launcher's feed only
**Why the launcher reads it (signed O Iris.app, macOS 26.5, 2026-11-15):** after the user allowed
Bluetooth and Location, BLE samples arrived but a `system_profiler SPAirPortDataType` run by the
Python grandchild (launcher → bash → python → system_profiler) still had every BSSID withheld. Only
the launcher, the app's main executable, holds the app's Location grant.

So the launcher reads Wi-Fi itself and hands it down:
- **Its own Location grant, the OS's cached scan only.** Every 15 s, while Location is authorized,
  it reads `CWInterface.cachedScanResults()` (CoreWLAN): the networks macOS already saw in its own
  scans. It never starts a scan (`scanForNetworks`) and never associates; SSIDs are left out.
  macOS's Wi-Fi daemon log (macOS 26.5) showed the launcher made no scan requests, only power and
  supported-channel queries.
- **A pipe, nothing on disk.** Before it starts bash, the launcher creates a pipe. The child inherits
  the read end, named in its environment as `OIRIS_WIFI_FEED_FD`; the write end stays in the
  launcher (close-on-exec, non-blocking). Each reading is one JSON line:
  `{"t": unix_s, "status": "ok", "aps": [{"bssid": "…", "rssi": -60}], "withheld": n}`, or a status
  `location_not_authorized`, `no_interface`, `wifi_off`, `no_cached_results` with no APs. If oiris
  lags, the line is dropped; if oiris is gone, the feed stops. Raw BSSIDs exist only in that pipe and
  in oiris's memory: the log records only feed state changes with counts (for example
  `o-iris: wifi feed: ok, 67 networks (0 withheld)`), never an identifier.
- **`wifi_macos` reads the feed.** At open it takes `OIRIS_WIFI_FEED_FD` (once per process; the
  variable is then removed so nothing oiris starts sees it) if it names a pipe. Each read drains what
  is available (non-blocking, partial lines kept for the next read) and uses the newest complete
  line; lines older than 60 s count as no data, and a read without a new line returns nothing. Each
  BSSID is hashed with the local site key (`hash_identifier`); the sample has `tsf_us` None, plus
  `rssi` and `local_s`. Health: some networks without a BSSID → degraded, all of them → failed with
  the Location hint; `location_not_authorized` → "waiting for Location permission" (degraded, no
  sample, never a failure: a failure would count toward the bus's circuit breaker, which then kept
  Wi-Fi off for its 30 s cooldown after the user allowed Location, as seen live); `wifi_off` → failed
  "Wi-Fi is turned off"; `no_interface` → failed; `no_cached_results` or no networks → an empty
  sample with degraded health. Malformed lines (bad JSON, wrong types, a bad MAC) are dropped and
  counted in the health reason, never a crash.
- **No other source.** The earlier `system_profiler` fallback was removed: the same daemon log
  showed that each `system_profiler SPAirPortDataType` run makes macOS start a live scan, and inside
  the app every BSSID stayed withheld anyway. Without a feed (`oiris run` from a terminal, a build
  without the compiled launcher) `wifi_macos` is unavailable with a plain reason in its health, and
  once the feed ends it stays failed until O Iris is restarted. It never runs a subprocess.
- The monitor-mode check in `oiris/guard/radio.py` still runs when `wifi_macos` opens, but macOS has
  no `iw`, so it has nothing to query there. O Iris never enables monitor mode.

## Voice link in the app
The M-17 voice link (ADR-0057, ADR-0058, docs/VOICE_LINK_MODULE_SPEC.md) is off unless you turn it on
in an **app config** and opt in. Without that file the app runs exactly as described above.

**The app config.** If `~/Library/Application Support/O Iris/config.json` exists, the bash launcher
passes `--config` with that path to `oiris run` and to its passphrase check (so both open the same
store); otherwise it passes nothing and `oiris run` uses the defaults (data in `~/.asb`). It is a
normal O Iris config file (`oiris/config.py`); for the app the useful keys are:
- `flags`: `"M-16": true` and `"M-17": true` turn face matching (needed for each person's keys) and
  the voice link on. `M-17` is also what the launcher reads before it asks for the microphone.
- `data_dir`: where the encrypted store lives (default `~/.asb`).
- `voice_link`: the M-17 settings (`listening`, `announce`, `input_device`, `match_threshold`, ...).
- `face_match.adapter`: set it to `"store_only"` (below).

Example for the voice link:
```json
{
  "flags": {"M-16": true, "M-17": true},
  "data_dir": "~/.asb",
  "face_match": {"modes": ["personal"], "adapter": "store_only"},
  "voice_link": {"listening": "ambient", "announce": "chime_and_name"}
}
```
A config that does not validate stops `oiris run` with `config invalid: ...` in the log.

**The store-only face adapter.** The voice link needs M-16 loaded, because each person's voice
template is sealed with that person's own key. The app bundles no OpenCV and no face models, and live
mode does no face matching yet. `face_match.adapter = "store_only"`
(`oiris/modules/face_match/backends/store_only.py`) opens M-16 without models or OpenCV. It never
finds a face, so nothing is embedded or compared, and the dashboard never reports a face match. It
has its own model ID, so templates made by a real face model are kept but never matched with it. It
cannot enroll anyone: `oiris face-match enroll` with it prints "Not enrolled: face enrollment needs a
real face backend; ...". (With `--camera auto` the camera probe runs first; it needs OpenCV and finds
no face.)

**Where enrollments live: `data_dir`.** People are enrolled with the CLI, not in the app, into the
same `data_dir` with the same passphrase:
1. Face enrollment needs a real face backend, so use a Python environment with OpenCV
   (`pip install 'o-iris[hw]'`, for example a checkout of this repository), not the app's venv. With
   the default config (`data_dir` `~/.asb`, adapter `opencv_zoo`):
   `oiris face-match enable --mode personal`, `oiris face-match fetch-models`,
   `oiris face-match consent --subject CODE --scope face_enrollment,voice_enrollment`, then
   `oiris face-match enroll --camera auto`. If the app config sets another `data_dir`, pass a config
   with that `data_dir` (and the default adapter) to every command.
2. Write the app config above.
3. Quit O Iris, then use the app's own venv, which has the voice packages, with the app config:
   ```bash
   APP="$HOME/Library/Application Support/O Iris"
   "$APP/venv/bin/oiris" voice-link enable --config "$APP/config.json"
   "$APP/venv/bin/oiris" voice-link fetch-models --config "$APP/config.json"
   "$APP/venv/bin/oiris" voice-link enroll --person ID-PREFIX --consent ID --config "$APP/config.json"
   ```
   `fetch-models` is the only download: the speaker model (sherpa-onnx CAM++, size and SHA-256
   pinned) goes to `<data_dir>/models/voice_link`, only when you run it. The app never bundles or
   downloads it. Voice enrollment from Terminal uses Terminal's microphone permission, not O Iris's.
4. Open O Iris.

**The microphone prompt.** With `flags."M-17"` true in the app config, the compiled launcher (the
app's main executable, as for Bluetooth and Location) asks for the microphone while the answer is
undecided. macOS shows "“O Iris” would like to access the microphone" with the purpose string
(`NSMicrophoneUsageDescription`): "O Iris listens only when you turn on the voice link, to recognise
the voices of people you enrolled. Sound is processed in memory and never saved." Without the flag it
never asks; it only logs the authorization (reading it never prompts). The Python child opens the
microphone under the app's grant. Find or change it in System Settings → Privacy & Security →
Microphone (entry **O Iris**). **Every rebuild asks again**: the ad-hoc identity is the exact code
hash (ADR-0056, "Code identity" above). A build without the compiled launcher never asks.

Not yet observed on hardware: the microphone prompt in a built app, and live listening inside the
app. Tested without hardware: the purpose string, the launcher source (the request only behind the
M-17 check), the bundled wheel list (faked `pip download`), the launcher script's `--config` handling
under `/bin/bash` 3.2 with fake venv programs, and the store-only adapter with M-16 and M-17
(`tests/test_macos_app.py`, `tests/test_face_match_store_only.py`).

## Honest limits
- Research signals only: anomaly signatures and sensor health, never a person label
  (`oiris/guard/vocabulary.py`). A Mac alone feeds one signal family (rf, from BLE; no score module
  uses the Wi-Fi AP scans yet). One family can raise FYI; ATTENTION needs ≥2 families. Observed
  (2026-11-15): with one strong Bluetooth device nearby, the level stayed at FYI with signature A4
  for hours. That is an observation about a nearby emitter, not a finding about anyone.
- Apple Silicon and macOS 14 or later only.
- Ad-hoc signed only: Gatekeeper blocks the first open on other Macs (Control-click → Open on
  macOS 14; System Settings → Privacy & Security → Open Anyway on macOS 15 and later), and every
  rebuild makes macOS ask for Bluetooth and Location again (seen three times on 2026-11-15). A
  Developer ID signature (stable team and identifier) would keep the grants across rebuilds.
- Observed on hardware (Apple Silicon, macOS 26.5, 2026-11-15): macOS showed prompts naming O Iris
  and the user allowed Bluetooth and Location. Before the grant, `wifi_macos` showed "waiting for
  Location permission" with 0 restarts and turned OK 1 s after the first OK feed line. Real BLE
  samples arrived. The feed reported `ok, 67–72 networks (0 withheld)`, and `wifi_macos` stayed OK
  with a new sample about every 15 s. No score module uses Wi-Fi AP scans yet: without beacon TSF the
  AP clock-skew family has nothing to measure.
- Tested without hardware: the feed reader in-process with `os.pipe()` (`tests/test_macos_adapters.py`);
  the bundle structure, the launcher source, the purpose strings and the recorded compile and
  codesign argv (`tests/test_macos_app.py`; no downloads, no compile, no signing, no launch). The
  launcher's spawn, signal and exit-status code was also run earlier in a fake bundle with a
  stand-in script, and the bash launcher's dialog, unlock and exec paths with fakes under macOS
  `/bin/bash` 3.2.
- Never point it at people who have not agreed to it; read ETHICS.md.

## Signing and notarizing (needs an Apple Developer ID, paid; not done here)
```
codesign --force --deep --options runtime --timestamp --sign "Developer ID Application: NAME (TEAMID)" "O Iris.app"
ditto -c -k --keepParent "O Iris.app" o-iris.zip
xcrun notarytool submit o-iris.zip --apple-id YOU --team-id TEAMID --password APP_SPECIFIC_PW --wait
xcrun stapler staple "O Iris.app"
spctl --assess --type execute -vv "O Iris.app"
```
This replaces the build's ad-hoc signature. Note that the hardened runtime (`--options runtime`) needs to be tested with the
launcher's unsigned child Python first, because it may stop that child from running. A Developer ID
build would give the Bluetooth and Location grants a stable identity across rebuilds.
Never put signing identities or passwords in this public repository's CI.
