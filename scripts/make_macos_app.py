#!/usr/bin/env python3
"""Build an AD-HOC signed (not Developer ID) macOS `O Iris.app` bundle around the project wheel (M-10 artifact).

Layout: Contents/Info.plist, Contents/MacOS/o-iris (compiled Swift launcher, macos/launcher/), which
asks macOS for Bluetooth and Location itself and runs Contents/Resources/o-iris-launcher.sh (bash) as
its child, Contents/MacOS/keychain-gate (M-15 Keychain gate helper, macos/keychain_gate/),
Contents/Resources/wheels/*.whl. Both Swift programs are compiled when swiftc is available (skip
with --no-swift or OIRIS_SKIP_SWIFT=1); without it the bash script itself is Contents/MacOS/o-iris,
as in earlier builds, and nothing asks for the permissions. The finished bundle is ad-hoc signed
(`codesign --sign -`) so macOS has a code identity to tie Bluetooth/Location permissions to.
First launch creates a private venv in ~/Library/Application Support/O Iris using python3
(>= 3.11), installs `o-iris[mac,voice]` from the bundled wheels with --no-index (offline), asks for
the store passphrase in a macOS dialog, then runs live mode (`oiris run`: real Mac sensors and
the local dashboard on 127.0.0.1; `--config` the app config if it exists) and opens the dashboard.
The launcher asks for the microphone only when that app config turns the M-17 voice link on, and for the camera only
when it turns M-16 face matching on and lists the face_camera adapter in live.adapters. No network, no paid service.
Developer ID signing/notarization steps for someone with an Apple Developer account: docs/MACOS.md.

  python scripts/make_macos_app.py --wheel dist/X/o_iris-*.whl --out dist/X [--bundle-deps] [--commit SHA] [--no-swift]
"""
from __future__ import annotations
import argparse, os, plistlib, shutil, stat, subprocess, sys
from pathlib import Path

LAUNCHER = r"""#!/bin/bash
# O Iris launcher script. Normally Contents/Resources/o-iris-launcher.sh, run as a child of the
# signed Mach-O launcher Contents/MacOS/o-iris, which asks macOS for Bluetooth and Location itself and
# is the responsible process for those permissions; builds without swiftc use this script as
# Contents/MacOS/o-iris (and then nothing asks for them).
# Offline: installs only the wheels inside this bundle, then runs live mode (`oiris run`: real Mac
# sensors + dashboard on 127.0.0.1) in THIS process (exec), so oiris stays in O Iris.app's process
# tree. Never detach via open/launchctl.
# The compiled launcher's Wi-Fi feed (an inherited pipe named by OIRIS_WIFI_FEED_FD) passes through
# this script untouched to the exec'd oiris.
# App config: if "~/Library/Application Support/O Iris/config.json" exists, oiris (and the key store
# check) get `--config <that file>`; otherwise nothing changes (the defaults, data in ~/.asb).
# docs/MACOS.md, "Voice link in the app".
# Runs under macOS /bin/bash 3.2 (no bash 4 features). The passphrase lives only in a shell
# variable and, at the very end, in the environment of the exec'd oiris process.
set -euo pipefail
# Apple Silicon: macOS may start a script-executable bundle (the no-swiftc layout) under Rosetta
# (x86_64), and then the universal2 python.org Python picks x86_64 too and the bundled arm64 wheels
# do not install. The compiled launcher is arm64, so under it this guard does not fire.
# Info.plist LSArchitecturePriority asks for arm64 first; this guard re-execs natively if we are
# still translated. `arch -arm64` is never translated, and OIRIS_ARCH_REEXEC stops any second try.
if [[ -z "${OIRIS_ARCH_REEXEC:-}" && "$(/usr/sbin/sysctl -n sysctl.proc_translated 2>/dev/null || true)" == 1 ]]; then
  export OIRIS_ARCH_REEXEC=1
  exec /usr/bin/arch -arm64 /bin/bash "$0" ${1+"$@"}
fi
unset OIRIS_ARCH_REEXEC
HERE="$(cd "$(dirname "$0")/.." && pwd)"   # Contents (from Contents/Resources, or Contents/MacOS without swiftc)
SUPPORT="$HOME/Library/Application Support/O Iris"
VENV="$SUPPORT/venv"
APP_CONFIG="$SUPPORT/config.json"
CONFIG_ARGS=()   # expanded as ${CONFIG_ARGS[@]+"${CONFIG_ARGS[@]}"}: bash 3.2 with set -u rejects an empty array
if [[ -f "$APP_CONFIG" ]]; then CONFIG_ARGS=(--config "$APP_CONFIG"); fi
PORT="${OIRIS_PORT:-8765}"
LOGDIR="$HOME/Library/Logs/O Iris"
LOG="$LOGDIR/o-iris.log"
mkdir -p "$LOGDIR"
if [[ -f "$LOG" && "$(wc -c < "$LOG")" -gt 5000000 ]]; then mv -f "$LOG" "$LOG.1"; fi
exec >>"$LOG" 2>&1
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') launcher: $*"; }
alert() {
  /usr/bin/osascript -e 'on run argv' -e 'display alert (item 1 of argv) message (item 2 of argv) as critical' \
    -e 'end run' "$1" "$2" >/dev/null || true
}
ask() {  # $1 = prompt; prints the typed text; non-zero (error -128) when the user presses Cancel
  /usr/bin/osascript -e 'on run argv' \
    -e 'text returned of (display dialog (item 1 of argv) with title "O Iris" default answer "" with hidden answer)' \
    -e 'end run' "$1"
}
port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; }
log "start @VERSION@ (port $PORT, arch $(uname -m))"
if [[ -f "$APP_CONFIG" ]]; then log "app config: $APP_CONFIG"; else log "no app config; using the defaults"; fi

# Finder gives apps PATH=/usr/bin:/bin:/usr/sbin:/sbin, whose python3 is often too old.
find_python() {
  if [[ -n "${OIRIS_PYTHON:-}" ]]; then echo "$OIRIS_PYTHON"; return 0; fi
  local c p
  for c in python3.14 python3.13 python3.12 python3.11 python3; do
    for p in "/Library/Frameworks/Python.framework/Versions/Current/bin/$c" "/opt/homebrew/bin/$c" "/usr/local/bin/$c" \
             "$(command -v "$c" || true)"; do
      if [[ -x "$p" ]] && "$p" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then echo "$p"; return 0; fi
    done
  done
  return 1
}
PY="$(find_python || true)"
if [[ -z "$PY" ]] || ! "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
  log "no python3 >= 3.11 found"
  alert "O Iris needs Python 3.11 or newer" "Install it from python.org, then open O Iris again."
  exit 1
fi
STAMP="$VENV/.oiris-version"
if [[ ! -f "$STAMP" || "$(cat "$STAMP")" != "@VERSION@" ]]; then
  log "installing bundled wheels into $VENV with $PY"
  mkdir -p "$SUPPORT"
  if ! { "$PY" -m venv --clear "$VENV" && "$VENV/bin/pip" install --no-index --find-links "$HERE/Resources/wheels" "o-iris[mac,voice]"; }; then
    log "offline install failed"
    alert "O Iris could not install its bundled packages" "Details: ~/Library/Logs/O Iris/o-iris.log"
    exit 1
  fi
  echo "@VERSION@" > "$STAMP"
fi

if port_open; then   # already running (e.g. a second double-click): just show the dashboard
  log "port $PORT already answers; opening it instead of starting a second instance"
  /usr/bin/open "http://127.0.0.1:$PORT/"
  exit 0
fi

# Passphrase for the encrypted store: OIRIS_PASSPHRASE if set, else a hidden macOS dialog.
KEYCHECK='
import os, sys
from pathlib import Path
from oiris.config import Config
from oiris.security.keys import KeyStore, KeyStoreError
cfg = Config.load(sys.argv[3] if sys.argv[2:3] == ["--config"] else None)   # the store oiris run opens
ks = KeyStore(Path(cfg.data_dir).expanduser() / "keys.json")
if sys.argv[1] == "exists":
    sys.exit(10 if ks.exists() else 11)
try:
    ks.unlock(os.environ["OIRIS_PASSPHRASE"])
except KeyStoreError:
    sys.exit(12)
'
PW="${OIRIS_PASSPHRASE:-}"
unset OIRIS_PASSPHRASE
if [[ -z "$PW" ]]; then
  set +e; "$VENV/bin/python" -c "$KEYCHECK" exists ${CONFIG_ARGS[@]+"${CONFIG_ARGS[@]}"}; rc=$?; set -e
  if [[ $rc -eq 10 ]]; then
    tries=0
    while :; do
      PW="$(ask "Enter your O Iris passphrase to unlock the encrypted store.")" || { log "passphrase dialog cancelled"; exit 0; }
      if [[ -n "$PW" ]] && OIRIS_PASSPHRASE="$PW" "$VENV/bin/python" -c "$KEYCHECK" unlock ${CONFIG_ARGS[@]+"${CONFIG_ARGS[@]}"}; then break; fi
      tries=$((tries + 1))
      log "passphrase rejected ($tries of 3)"
      if [[ $tries -ge 3 ]]; then alert "O Iris: wrong passphrase" "The encrypted store was not unlocked."; exit 1; fi
    done
  elif [[ $rc -eq 11 ]]; then
    while :; do
      PW="$(ask "First run: choose a passphrase for the new encrypted O Iris store. It cannot be recovered if lost.")" \
        || { log "passphrase dialog cancelled"; exit 0; }
      PW2="$(ask "Type the same passphrase again.")" || { log "passphrase dialog cancelled"; exit 0; }
      if [[ -n "$PW" && "$PW" == "$PW2" ]]; then break; fi
      alert "Passphrases did not match (or were empty)" "Please try again."
    done
    PW2=""
  else
    log "key store check failed (exit $rc)"
    alert "O Iris could not check its key store" "Details: ~/Library/Logs/O Iris/o-iris.log"
    exit 1
  fi
fi

# Open the dashboard once it answers; give up if oiris (this PID after exec) exits or after 30 s.
LAUNCHER_PID=$$
( for _ in $(seq 1 120); do
    kill -0 "$LAUNCHER_PID" 2>/dev/null || exit 0
    if port_open; then /usr/bin/open "http://127.0.0.1:$PORT/"; exit 0; fi
    sleep 0.25
  done
  log "dashboard did not answer on port $PORT within 30 s" ) &
log "exec oiris run --port $PORT${CONFIG_ARGS[@]+ ${CONFIG_ARGS[*]}}"
export OIRIS_PASSPHRASE="$PW"
exec "$VENV/bin/oiris" run --port "$PORT" ${CONFIG_ARGS[@]+"${CONFIG_ARGS[@]}"}
"""

# Accurate on purpose: macOS's Bluetooth scanning is active (CoreBluetooth has no passive scan), so
# neither string claims "passive" or "never transmits" (spec non-goal: normal client radios allowed).
BLUETOOTH_PURPOSE = ("O Iris reads nearby Bluetooth advertisements for research anomaly signals, using macOS's normal "
                     "Bluetooth scanning. It never connects to or pairs with devices; identifiers are hashed with a "
                     "local key and nothing leaves this Mac.")
LOCATION_PURPOSE = ("Needed only because macOS hides Wi-Fi network identifiers without it. O Iris never reads, "
                    "records or sends your location.")
# ADR-0057: the voice link is off by default; audio is processed in memory and only templates of enrolled
# voices (and "heard" entries) are stored, never sound. The launcher asks only when flags."M-17" is on.
MICROPHONE_PURPOSE = ("O Iris listens only when you turn on the voice link, to recognise the voices of people you "
                      "enrolled. Sound is processed in memory and never saved.")
# ADR-0057 decision 5: the face camera is off by default; frames stay in memory, only counts and encounter entries are
# stored. The launcher asks only when flags."M-16" is on and live.adapters lists face_camera.
CAMERA_PURPOSE = ("O Iris uses the camera only when you turn on face matching (M-16) and add the face camera to live "
                  "mode, to recognise people you enrolled. Frames are processed in memory and never saved.")
# Bundled for every Python the app supports: live mode on macOS plus the M-17 voice link (sherpa-onnx,
# sounddevice and its cffi). The speaker model is not bundled: `oiris voice-link fetch-models`, on request.
APP_EXTRAS = "mac,voice"
PYTHON_VERSIONS = ("3.11", "3.12", "3.13", "3.14")


BUNDLE_ID = "io.github.geometricfinger.oiris"
LAUNCHER_SRC = Path(__file__).resolve().parents[1] / "macos" / "launcher" / "main.swift"
GATE_SRC = Path(__file__).resolve().parents[1] / "macos" / "keychain_gate" / "main.swift"   # M-15 helper


def find_swiftc() -> list[str] | None:
    """-> command prefix for swiftc, or None. Only on macOS (the launcher needs CoreBluetooth and
    CoreLocation), and only
    if Xcode or the Command Line Tools are installed: on a Mac without them, /usr/bin/swiftc and
    xcrun are shims that pop up an install dialog, so `xcode-select -p` is checked first."""
    if sys.platform != "darwin" or not shutil.which("xcode-select") or not shutil.which("xcrun"):
        return None
    for probe in (["xcode-select", "-p"], ["xcrun", "--find", "swiftc"]):   # fixed argv, no shell
        try:
            if subprocess.run(probe, capture_output=True, timeout=60).returncode != 0:
                return None
        except (OSError, subprocess.TimeoutExpired):   # a broken or hung toolchain means "no swiftc", not a crash
            return None
    return ["xcrun", "swiftc"]


def _compile(swiftc: list[str], src: Path, dest: Path) -> Path:
    # fixed argv, no shell; a compile error or a hang (>10 min) fails the build loudly
    subprocess.run([*swiftc, "-O", "-target", "arm64-apple-macos14.0", str(src), "-o", str(dest)],
                   check=True, timeout=600)
    return dest


def build_launcher(macos: Path, swift: bool) -> Path | None:
    """Compile macos/launcher into Contents/MacOS/o-iris, the bundle's executable, which asks for Bluetooth
    and Location (the microphone when the app config turns M-17 on, the camera when it turns M-16 on with the
    face_camera adapter), runs the bash launcher from
    Contents/Resources as its child and feeds it the OS's cached Wi-Fi scan (its AVFoundation, CoreBluetooth,
    CoreLocation and CoreWLAN imports autolink those frameworks, so no -framework flags are needed). With it,
    the same swiftc and flags build the M-15 Keychain gate helper (macos/keychain_gate) into
    Contents/MacOS/keychain-gate, before signing, so the signature covers it.
    None (with a note) when disabled or without swiftc:
    the caller then falls back to the bash script as the executable. A compile error fails."""
    if not swift:
        print("note: Swift launcher not compiled (--no-swift / OIRIS_SKIP_SWIFT=1); "
              "this build cannot ask for Bluetooth or Location")
        return None
    swiftc = find_swiftc()
    if swiftc is None:
        print("note: swiftc not available; O Iris.app's executable is the bash launcher itself, which "
              "gives macOS no code identity to tie Bluetooth/Location permissions to")
        return None
    launcher = _compile(swiftc, LAUNCHER_SRC, macos / "o-iris")
    _compile(swiftc, GATE_SRC, macos / "keychain-gate")
    return launcher


def sign_app(app: Path, sign: bool) -> bool:
    """Ad-hoc sign the finished bundle (`codesign --sign -`, identifier = CFBundleIdentifier from
    Info.plist) and verify it. Not Developer ID and no hardened runtime (that would stop the launcher's
    unsigned child python). Skipped, with a note, when disabled or off macOS / without codesign.
    A signing or verification failure fails the build."""
    if not sign:
        print("note: not signed (--no-swift / OIRIS_SKIP_SWIFT=1); macOS has no code identity for this app")
        return False
    if sys.platform != "darwin" or not shutil.which("codesign"):
        print("note: codesign not available; the app is unsigned")
        return False
    # fixed argv, no shell, bounded. Contents/MacOS holds the main executable (the launcher) and, when
    # compiled, keychain-gate; both are sealed by this signature and checked by --deep --strict.
    subprocess.run(["codesign", "--force", "--sign", "-", str(app)], check=True, timeout=300)
    subprocess.run(["codesign", "--verify", "--deep", "--strict", f'-R=identifier "{BUNDLE_ID}"', str(app)],
                   check=True, timeout=300)
    return True


def build(wheel: Path, out: Path, version: str, commit: str, bundle_deps: bool, swift: bool = True) -> Path:
    app = out / "O Iris.app"
    if app.exists():
        shutil.rmtree(app)
    macos, res = app / "Contents" / "MacOS", app / "Contents" / "Resources" / "wheels"
    macos.mkdir(parents=True); res.mkdir(parents=True)
    shutil.copy2(wheel, res / wheel.name)
    if bundle_deps:
        for pv in PYTHON_VERSIONS:   # the user's python3 may be any supported version
            subprocess.run([sys.executable, "-m", "pip", "download", "--only-binary=:all:", "--python-version", pv,
                            "-d", str(res), f"{wheel}[{APP_EXTRAS}]"], check=True)
    info = {"CFBundleName": "O Iris", "CFBundleDisplayName": "O Iris", "CFBundleIdentifier": BUNDLE_ID,
            "CFBundleVersion": version, "CFBundleShortVersionString": version, "CFBundlePackageType": "APPL",
            "CFBundleExecutable": "o-iris", "LSMinimumSystemVersion": "14.0", "NSHighResolutionCapable": True,
            "LSArchitecturePriority": ["arm64", "x86_64"],   # native on Apple Silicon (bundled arm64 wheels)
            "OIrisCommit": commit, "LSApplicationCategoryType": "public.app-category.developer-tools",
            "NSBluetoothAlwaysUsageDescription": BLUETOOTH_PURPOSE,
            "NSLocationUsageDescription": LOCATION_PURPOSE, "NSLocationWhenInUseUsageDescription": LOCATION_PURPOSE,
            "NSMicrophoneUsageDescription": MICROPHONE_PURPOSE, "NSCameraUsageDescription": CAMERA_PURPOSE}
    with open(app / "Contents" / "Info.plist", "wb") as f:
        plistlib.dump(info, f)
    # compiled launcher -> script in Resources; otherwise the script is the executable (earlier layout)
    lp = macos / "o-iris" if build_launcher(macos, swift) is None else res.parent / "o-iris-launcher.sh"
    lp.write_text(LAUNCHER.replace("@VERSION@", f"{version}+{commit[:12]}"), encoding="utf-8")
    lp.chmod(lp.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    for name in ("LICENSE", "NOTICE", "THIRD_PARTY.md"):
        src = Path(__file__).resolve().parents[1] / name
        if src.exists():
            shutil.copy2(src, app / "Contents" / "Resources" / name)
    sign_app(app, swift)   # last: any later change to the bundle breaks the signature
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wheel", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--commit", default=os.environ.get("ASB_COMMIT", "unknown"))
    ap.add_argument("--version", default="0.1.0")
    ap.add_argument("--bundle-deps", action="store_true")
    ap.add_argument("--no-swift", action="store_true", help="do not compile the Swift launcher or keychain-gate "
                    "and do not sign (also OIRIS_SKIP_SWIFT=1); the app then has no code identity for permissions")
    a = ap.parse_args()
    swift = not (a.no_swift or os.environ.get("OIRIS_SKIP_SWIFT", "") not in ("", "0"))
    print(build(a.wheel, a.out, a.version, a.commit, a.bundle_deps, swift))
