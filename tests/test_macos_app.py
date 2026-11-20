"""macOS .app bundle structure (built on any OS; no deps download, no Swift compile, no launch in tests)."""
import importlib.util, os, plistlib, re, subprocess, sys, zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NO_SWIFT = {**os.environ, "OIRIS_SKIP_SWIFT": "1"}   # tests never compile (or run) the Swift launcher


def test_app_bundle_layout(tmp_path):
    whl = tmp_path / "o_iris-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(whl, "w") as z:
        z.writestr("oiris/__init__.py", "")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_macos_app.py"), "--wheel", str(whl), "--out", str(tmp_path),
                        "--commit", "abc123def4567890"], capture_output=True, text=True, env=NO_SWIFT)
    assert r.returncode == 0, r.stderr
    app = tmp_path / "O Iris.app"
    assert not (app / "Contents" / "MacOS" / "o-iris-location").exists() and "not compiled" in r.stdout
    # without Swift: the earlier layout (the script is the executable), not signed, and it says so
    assert not (app / "Contents" / "Resources" / "o-iris-launcher.sh").exists() and "not signed" in r.stdout
    info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    assert info["CFBundleExecutable"] == "o-iris" and info["CFBundlePackageType"] == "APPL" and info["OIrisCommit"].startswith("abc123")
    launcher = app / "Contents" / "MacOS" / "o-iris"
    assert launcher.stat().st_mode & 0o111
    txt = launcher.read_text()
    assert "--no-index" in txt and "127.0.0.1" in txt and "0.1.0+abc123def456" in txt
    assert (app / "Contents" / "Resources" / "wheels" / whl.name).exists()
    assert (app / "Contents" / "Resources" / "LICENSE").exists()
    assert subprocess.run(["bash", "-n", str(launcher)]).returncode == 0

    # live app: permissions macOS requires before CoreBluetooth / Wi-Fi identifiers, and macOS 14 for wifi_macos
    bt = info["NSBluetoothAlwaysUsageDescription"]
    assert bt == _maker().BLUETOOTH_PURPOSE and "hashed" in bt and "never connects" in bt
    for k in ("NSLocationUsageDescription", "NSLocationWhenInUseUsageDescription"):
        assert info[k] == _maker().LOCATION_PURPOSE and "Wi-Fi" in info[k]
    assert info["NSMicrophoneUsageDescription"] == _maker().MICROPHONE_PURPOSE   # M-17 voice link (ADR-0057)
    assert info["NSCameraUsageDescription"] == _maker().CAMERA_PURPOSE           # M-16 face camera (ADR-0057 d. 5)
    assert info["LSMinimumSystemVersion"] == "14.0"

    # launcher: offline install of the mac and voice extras, live mode exec'd in-process, Finder evidence in a log
    assert '"o-iris[mac,voice]"' in txt and "--find-links" in txt and '"o-iris[mac]"' not in txt
    assert 'exec "$VENV/bin/oiris" run --port "$PORT"' in txt and " dashboard --port" not in txt
    code = "\n".join(ln for ln in txt.splitlines() if not ln.lstrip().startswith("#"))
    assert "launchctl" not in code and "nohup" not in code and "disown" not in code and "open -a" not in code
    assert code.count("/usr/bin/open ") == 2 and code.count('/usr/bin/open "http://127.0.0.1:$PORT/"') == 2  # only the URL
    assert "/dev/tcp/127.0.0.1/$PORT" in txt and "sleep 1;" not in txt   # polls the port before opening
    assert '"$HOME/Library/Logs/O Iris"' in txt and 'exec >>"$LOG" 2>&1' in txt
    for host in ("http://", "https://"):
        assert all(u.startswith(host + "127.0.0.1") for u in _urls(txt, host))

    # passphrase: hidden dialog only without OIRIS_PASSPHRASE, asked twice on first run, Cancel quits cleanly
    assert "with hidden answer" in txt and 'default answer ""' in txt
    assert txt.count('PW="$(ask ') == 2 and 'PW2="$(ask ' in txt and '"$PW" == "$PW2"' in txt
    assert txt.count('|| { log "passphrase dialog cancelled"; exit 0; }') == 3
    # ...and it only ever reaches oiris through the environment, never argv, a file or the log
    uses = [ln.strip() for ln in txt.splitlines() if "$PW" in ln]   # every read of the secret, listed exactly
    assert uses == ['if [[ -z "$PW" ]]; then',
                    'if [[ -n "$PW" ]] && OIRIS_PASSPHRASE="$PW" "$VENV/bin/python" -c "$KEYCHECK" unlock '
                    '${CONFIG_ARGS[@]+"${CONFIG_ARGS[@]}"}; then break; fi',
                    'if [[ -n "$PW" && "$PW" == "$PW2" ]]; then break; fi',
                    'export OIRIS_PASSPHRASE="$PW"'], uses
    assert 'export OIRIS_PASSPHRASE="$PW"\nexec "$VENV/bin/oiris" run' in txt   # exported only for the exec'd process
    assert "unset OIRIS_PASSPHRASE" in txt          # an inherited value is not passed on to the helpers
    assert txt.index('export OIRIS_PASSPHRASE="$PW"') > txt.index(") &")   # opener subshell forked before the export
    assert "set -x" not in txt

    # Apple Silicon: a script executable may be started under Rosetta, which makes the universal2 Python
    # x86_64 and the bundled arm64 wheels unusable. Plist asks for arm64 first; the launcher re-execs if translated.
    assert info["LSArchitecturePriority"] == ["arm64", "x86_64"]
    guard = ('if [[ -z "${OIRIS_ARCH_REEXEC:-}" && "$(/usr/sbin/sysctl -n sysctl.proc_translated 2>/dev/null || true)" == 1 ]]; then\n'
             '  export OIRIS_ARCH_REEXEC=1\n'
             '  exec /usr/bin/arch -arm64 /bin/bash "$0" ${1+"$@"}\n'
             'fi\n'
             'unset OIRIS_ARCH_REEXEC\n')
    assert guard in txt and txt.count("/usr/bin/arch") == 1   # exec keeps the PID (Bluetooth attribution); one try only
    g = txt.index(guard)
    assert txt.index("set -euo pipefail") < g
    for later in ("mkdir -p", 'exec >>"$LOG"', "log()", '"$VENV/bin/pip"', "find_python()"):   # before anything logs or installs
        assert g < txt.index(later), later


def _urls(txt, scheme):
    out, i = [], txt.find(scheme)
    while i != -1:
        out.append(txt[i:i + 40]); i = txt.find(scheme, i + 1)
    return out


def _maker():
    spec = importlib.util.spec_from_file_location("make_macos_app", ROOT / "scripts" / "make_macos_app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _launcher_code():
    src = (ROOT / "macos" / "launcher" / "main.swift").read_text()
    return src, "\n".join(ln.split("//")[0] for ln in src.splitlines())   # comments may mention exec, scans, ...


def test_launcher_source_asks_for_bluetooth_and_location_itself():
    """The app's main executable must ask: requests from the Python child or a bare CLI helper were dropped."""
    src, code = _launcher_code()
    assert re.findall(r"^import (\w+)", src, re.M) == ["AVFoundation", "CoreBluetooth", "CoreLocation", "CoreWLAN", "Foundation"]
    # Bluetooth: creating the manager raises the prompt while undecided; no power alert; state changes logged
    assert "CBCentralManager(delegate: watcher, queue: DispatchQueue.main" in code
    assert "CBCentralManagerOptionShowPowerAlertKey: false" in code
    assert "func centralManagerDidUpdateState(_ central: CBCentralManager)" in code and "CBManager.authorization" in code
    # Location: asks When In Use only while undecided; authorization changes logged
    assert "location.delegate = watcher" in code
    assert "if location.authorizationStatus == .notDetermined {" in code and "location.requestWhenInUseAuthorization()" in code
    assert "func locationManagerDidChangeAuthorization(_ manager: CLLocationManager)" in code
    assert code.count('log("bluetooth ') == 1 and code.count('log("location authorization ') == 2
    # never scans or connects, never a position: no updates, no one-shot request, no CLLocation object or coordinate
    for banned in ("scanForPeripherals", "connect(", "retrievePeripherals", "retrieveConnected", "CBPeripheral",
                   "startUpdatingLocation", "requestLocation", "requestAlwaysAuthorization", "didUpdateLocations",
                   "startMonitoring", "coordinate", "latitude", "longitude", ".location", "CLLocation("):
        assert banned not in code, banned
    assert re.search(r"\bCLLocation\b", code) is None
    # the requests start before the child is spawned, so a first-run prompt appears at once
    spawn = code.index("posix_spawn(&pid")
    assert code.index("CBCentralManager(") < spawn and code.index("requestWhenInUseAuthorization()") < spawn
    # the main thread then runs the main run loop (CoreLocation delivers there), kept alive by a timer
    assert "RunLoop.main.run()" in code and "RunLoop.main.add(Timer(" in code and "dispatchMain" not in code
    assert spawn < code.index("RunLoop.main.run()")
    # managers and watcher are top-level globals, alive for the whole process
    for g in ("\nlet watcher = PermissionWatcher()", "\nlet central = CBCentralManager(", "\nlet location = CLLocationManager()"):
        assert g in code, g
    assert '"/Library/Logs/O Iris/o-iris.log"' in code and "O_APPEND" in code   # the bash launcher's log


def test_launcher_build_is_skippable(tmp_path, monkeypatch, capsys):
    m = _maker()
    calls = []

    def no_run(*a, **k):
        calls.append(a)
        raise AssertionError(a)
    monkeypatch.setattr(m.subprocess, "run", no_run)
    assert m.build_launcher(tmp_path, swift=False) is None
    assert "not compiled" in capsys.readouterr().out
    monkeypatch.setattr(m, "find_swiftc", lambda: None)            # e.g. Linux CI, or a Mac without the CLT
    assert m.build_launcher(tmp_path, swift=True) is None
    assert calls == [] and list(tmp_path.iterdir()) == []
    monkeypatch.undo()
    monkeypatch.setattr(m.subprocess, "run", no_run)
    monkeypatch.setattr(m.sys, "platform", "linux")
    assert m.find_swiftc() is None and calls == []                  # never probes xcrun off macOS


def test_location_helper_is_gone():
    """The bare CLI helper's request was dropped by macOS (no Info.plist, not the main executable): removed."""
    assert not (ROOT / "macos" / "location_helper").exists()
    m = _maker()
    assert not hasattr(m, "LOCATION_HELPER_SRC") and not hasattr(m, "build_location_helper")
    src = (ROOT / "scripts" / "make_macos_app.py").read_text()
    assert "o-iris-location" not in src and "location-helper" not in src and "LOCHELPER" not in src
    assert "location helper" not in m.LAUNCHER.lower()


def test_find_swiftc_probes_are_bounded_and_never_crash(monkeypatch):
    m = _maker()
    monkeypatch.setattr(m.sys, "platform", "darwin")
    monkeypatch.setattr(m.shutil, "which", lambda name: "/usr/bin/" + name)
    seen = []

    def hang(cmd, **kw):
        seen.append((cmd, kw))
        raise m.subprocess.TimeoutExpired(cmd, kw["timeout"])
    monkeypatch.setattr(m.subprocess, "run", hang)
    assert m.find_swiftc() is None and seen[0][0] == ["xcode-select", "-p"] and seen[0][1]["timeout"] > 0
    monkeypatch.setattr(m.subprocess, "run", lambda cmd, **kw: (_ for _ in ()).throw(OSError("broken shim")))
    assert m.find_swiftc() is None
    monkeypatch.setattr(m.subprocess, "run", lambda cmd, **kw: m.subprocess.CompletedProcess(cmd, 0))
    assert m.find_swiftc() == ["xcrun", "swiftc"]


def test_launcher_source_spawns_child_and_forwards_signals():
    """The compiled launcher must stay the responsible process: child via posix_spawn, never exec."""
    _, code = _launcher_code()
    assert "posix_spawn(&pid, \"/bin/bash\", nil, &attr, &argv, &envp)" in code   # child, environment passed through
    assert "var ep = environ" in code and "childEnv.append(s)" in code
    assert re.search(r"\bexec\w*\s*\(", code) is None and "Process()" not in code
    assert '"o-iris-launcher.sh"' in code and '"Resources"' in code
    assert "CommandLine.arguments.dropFirst()" in code                     # arguments passed through
    assert "let forwarded: [Int32] = [SIGTERM, SIGINT]" in code and "kill(child, sig)" in code
    assert "makeSignalSource(signal: sig" in code and "signal(sig, SIG_IGN)" in code
    assert "posix_spawnattr_setsigdefault" in code and "POSIX_SPAWN_SETSIGDEF" in code   # child keeps default handlers
    # reaped off the main thread (which runs the run loop); the exit with its status happens on the main queue
    assert code.index("DispatchQueue.global().async {") < code.index("waitpid(pid, &status, 0)") < code.index("RunLoop.main.run()")
    assert "EINTR" in code
    assert "let code = termsig == 0 ? (status >> 8) & 0xff : 128 + termsig" in code   # child's status, 128+n if signalled
    assert "DispatchQueue.main.async { exit(code) }" in code
    for banned in ("URLSession", "NSWorkspace", "AppKit", "SwiftUI", "socket(", "NWConnection", "NWPathMonitor", "open "):
        assert banned not in code, banned
    assert (ROOT / "macos" / "launcher" / "README.md").exists()


def test_launcher_source_wifi_feed_is_passive_and_private():
    """Wi-Fi BSSIDs come from the launcher (it holds the Location grant) through a pipe, never via disk or log."""
    src, code = _launcher_code()
    # only the OS's cached scan: never a scan of our own, never an association or any other CWInterface action
    assert code.count("cachedScanResults()") == 1 and "CWWiFiClient.shared().interface()" in code
    for banned in ("scanForNetworks", "associate", "disassociate", "startMonitoringEvent", "setPower",
                   "CWNetworkProfile", "CWConfiguration", "commitConfiguration"):
        assert banned not in code, banned
    assert re.search(r"\bssid", code, re.I) is None                   # SSIDs are left out (only .bssid is read)
    # read only while Location is authorized; otherwise the line says so
    assert '["authorizedAlways", "authorizedWhenInUse"].contains(name(status))' in code
    assert "DispatchQueue.main.sync { location.authorizationStatus }" in code
    for st in ("ok", "location_not_authorized", "no_interface", "wifi_off", "no_cached_results"):
        assert f'"{st}"' in code, st
    assert "JSONSerialization.data(withJSONObject: snap)" in code and '"bssid": mac, "rssi": n.rssiValue' in code
    assert "let feedPeriod = 15.0" in code and "makeTimerSource(queue: feedQueue)" in code
    # the pipe: read end inherited (named in the child's environment), write end CLOEXEC + non-blocking here
    assert "pipe(&feedPipe)" in code and 'let feedEnv = "OIRIS_WIFI_FEED_FD"' in code
    assert 'childEnv.append("\\(feedEnv)=\\(feedPipe[0])")' in code and '!s.hasPrefix(feedEnv + "=")' in code
    assert "fcntl(feedPipe[1], F_SETFD, FD_CLOEXEC)" in code
    assert "fcntl(feedPipe[1], F_SETFL, fcntl(feedPipe[1], F_GETFL) | O_NONBLOCK)" in code
    assert "fcntl(feedPipe[0], F_SETFD, 0)" in code
    spawn = code.index("posix_spawn(&pid")
    assert code.index("pipe(&feedPipe)") < spawn < code.index("close(feedPipe[0])") < code.index("makeTimerSource(")
    # a lagging reader drops lines (EAGAIN); a gone reader stops the feed (EPIPE, SIGPIPE ignored)
    assert "signal(SIGPIPE, SIG_IGN)" in code and "EAGAIN" in code and 'errno == EPIPE ? "reader closed"' in code
    assert "for sig in forwarded + [SIGPIPE] { sigaddset(&defaults, sig) }" in code   # the child gets SIGPIPE back
    # BSSIDs reach only the pipe: no log call mentions them or the snapshot, nothing else is written
    logs = re.findall(r"\b(?:feedLog|log)\((.*)\)", code)
    leak = re.compile(r"\b(bssid|mac|b|snap|line|nets|n|feedPending|aps\[)\b", re.I)
    assert len(logs) > 10 and [ln for ln in logs if leak.search(ln)] == []
    assert "write(feedPipe[1]" in code and code.count("write(") == code.count("write(feedPipe[1]") + 3   # + the log's own writes
    assert "FileManager.default.createFile" not in code and ".write(to:" not in code and "UserDefaults" not in code


def _fake_toolchain(m, monkeypatch, swiftc=True, codesign=True, fail=None):
    """Record every subprocess.run; swiftc 'compiles' by writing the -o file. Nothing is compiled or signed."""
    seen = []

    def run(cmd, **kw):
        assert isinstance(cmd, list) and not kw.get("shell")              # fixed argv, no shell
        seen.append((cmd, kw))
        if fail and fail in cmd:
            raise m.subprocess.CalledProcessError(1, cmd)
        if cmd[:2] == ["xcrun", "swiftc"]:
            Path(cmd[cmd.index("-o") + 1]).write_bytes(b"\xcf\xfa\xed\xfe fake mach-o")
        return m.subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(m.subprocess, "run", run)
    monkeypatch.setattr(m, "find_swiftc", lambda: ["xcrun", "swiftc"] if swiftc else None)
    monkeypatch.setattr(m.sys, "platform", "darwin")
    monkeypatch.setattr(m.shutil, "which", lambda name: "/usr/bin/codesign" if codesign and name == "codesign" else None)
    return seen


def _wheel(tmp_path):
    whl = tmp_path / "o_iris-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(whl, "w") as z:
        z.writestr("oiris/__init__.py", "")
    return whl


def test_build_compiles_launcher_moves_script_and_signs(tmp_path, monkeypatch, capsys):
    m = _maker()
    seen = _fake_toolchain(m, monkeypatch)
    app = m.build(_wheel(tmp_path), tmp_path, "0.1.0", "abc123def4567890", bundle_deps=False, swift=True)
    c = app / "Contents"
    assert plistlib.loads((c / "Info.plist").read_bytes())["CFBundleExecutable"] == "o-iris"
    assert (c / "MacOS" / "o-iris").read_bytes().startswith(b"\xcf\xfa\xed\xfe")   # the compiled launcher
    script = c / "Resources" / "o-iris-launcher.sh"
    assert script.stat().st_mode & 0o111 and "0.1.0+abc123def456" in script.read_text()
    assert 'exec "$VENV/bin/oiris" run --port "$PORT"' in script.read_text()   # every behaviour kept
    cmds = [cmd for cmd, _ in seen]
    assert cmds[0][-3:] == [str(m.LAUNCHER_SRC), "-o", str(c / "MacOS" / "o-iris")]
    assert m.LAUNCHER_SRC == ROOT / "macos" / "launcher" / "main.swift" and "-O" in cmds[0]
    assert cmds[0][cmds[0].index("-target") + 1] == "arm64-apple-macos14.0"
    assert sorted(p.name for p in (c / "MacOS").iterdir()) == ["keychain-gate", "o-iris"]   # no Location helper
    assert cmds[1][-3:] == [str(m.GATE_SRC), "-o", str(c / "MacOS" / "keychain-gate")]
    # ad-hoc signing last, then a strict verification against the plist identifier
    assert cmds[2] == ["codesign", "--force", "--sign", "-", str(app)]
    assert cmds[3] == ["codesign", "--verify", "--deep", "--strict", '-R=identifier "io.github.geometricfinger.oiris"', str(app)]
    assert len(cmds) == 4 and all(kw.get("check") is True and kw.get("timeout") for _, kw in seen)
    assert not any("runtime" in a or "--options" in a for cmd in cmds for a in cmd)   # no hardened runtime
    assert "note:" not in capsys.readouterr().out


def test_build_falls_back_without_swiftc_or_codesign(tmp_path, monkeypatch, capsys):
    m = _maker()
    seen = _fake_toolchain(m, monkeypatch, swiftc=False, codesign=False)
    app = m.build(_wheel(tmp_path), tmp_path, "0.1.0", "abc", bundle_deps=False, swift=True)
    c = app / "Contents"
    assert seen == [] and not (c / "Resources" / "o-iris-launcher.sh").exists()
    assert (c / "MacOS" / "o-iris").read_text().startswith("#!/bin/bash") and (c / "MacOS" / "o-iris").stat().st_mode & 0o111
    out = capsys.readouterr().out
    assert "bash launcher itself" in out and "codesign not available" in out
    # swiftc missing but codesign present: the script-executable bundle is still signed and verified
    seen = _fake_toolchain(m, monkeypatch, swiftc=False)
    m.build(_wheel(tmp_path), tmp_path, "0.1.0", "abc", bundle_deps=False, swift=True)
    assert [cmd[:2] for cmd, _ in seen] == [["codesign", "--force"], ["codesign", "--verify"]]
    # disabled (tests, --no-swift): neither compiled nor signed, and off macOS codesign is never probed
    seen = _fake_toolchain(m, monkeypatch)
    m.build(_wheel(tmp_path), tmp_path, "0.1.0", "abc", bundle_deps=False, swift=False)
    assert seen == [] and "not signed" in capsys.readouterr().out
    monkeypatch.setattr(m.sys, "platform", "linux")
    assert m.sign_app(app, True) is False and seen == []


def test_build_fails_when_signature_does_not_verify(tmp_path, monkeypatch):
    m = _maker()
    seen = _fake_toolchain(m, monkeypatch, fail="--verify")
    with pytest.raises(m.subprocess.CalledProcessError) as err:   # a failed codesign --verify fails the build
        m.build(_wheel(tmp_path), tmp_path, "0.1.0", "abc", bundle_deps=False, swift=True)
    assert "--verify" in err.value.cmd and seen[-1][0] == err.value.cmd


def test_purpose_strings_are_accurate():
    """macOS's Bluetooth scanning is active (bleak: "macOS does not support passive scanning"), so no
    passive or no-transmit claim; the spec allows the Mac's normal client radio activity."""
    m = _maker()
    bt, loc, mic, cam = m.BLUETOOTH_PURPOSE, m.LOCATION_PURPOSE, m.MICROPHONE_PURPOSE, m.CAMERA_PURPOSE
    for txt in (bt, loc, mic, cam):
        assert "passiv" not in txt.lower() and "transmit" not in txt.lower(), txt
        assert len(txt) <= 250, len(txt)
    assert "Bluetooth advertisements" in bt and "normal Bluetooth scanning" in bt
    assert "never connects to or pairs with devices" in bt and "hashed with a local key" in bt
    assert "nothing leaves this Mac" in bt
    assert "only because macOS hides Wi-Fi network identifiers" in loc
    assert "never reads, records or sends your location" in loc
    # ADR-0057: off by default, only enrolled people, audio processed in RAM and discarded
    assert mic == ("O Iris listens only when you turn on the voice link, to recognise the voices of people you "
                   "enrolled. Sound is processed in memory and never saved.")
    # ADR-0057 decision 5: only with M-16 and the face camera configured; frames never saved
    assert cam == ("O Iris uses the camera only when you turn on face matching (M-16) and add the face camera to live "
                   "mode, to recognise people you enrolled. Frames are processed in memory and never saved.")


def test_build_compiles_keychain_gate_before_signing(tmp_path, monkeypatch):
    m = _maker()
    seen = _fake_toolchain(m, monkeypatch)
    app = m.build(_wheel(tmp_path), tmp_path, "0.1.0", "abc", bundle_deps=False, swift=True)
    cmds = [cmd for cmd, _ in seen]
    gate = app / "Contents" / "MacOS" / "keychain-gate"
    assert m.GATE_SRC == ROOT / "macos" / "keychain_gate" / "main.swift" and m.GATE_SRC.exists()
    i = next(n for n, c in enumerate(cmds) if str(m.GATE_SRC) in c)
    sign = cmds.index(["codesign", "--force", "--sign", "-", str(app)])
    assert i < sign and cmds[i][-3:] == [str(m.GATE_SRC), "-o", str(gate)] and gate.exists()
    launcher = cmds[0]                                                   # same swiftc, target and flags
    assert cmds[i][:cmds[i].index(str(m.GATE_SRC))] == launcher[:launcher.index(str(m.LAUNCHER_SRC))]
    # disabled: neither Swift program compiled, nothing signed
    seen = _fake_toolchain(m, monkeypatch)
    app = m.build(_wheel(tmp_path), tmp_path, "0.1.0", "abc", bundle_deps=False, swift=False)
    assert seen == [] and not (app / "Contents" / "MacOS" / "keychain-gate").exists()


def test_build_all_keeps_the_signed_bundle_intact():
    """build_all.sh must not change the bundle after make_macos_app.py signed it, and verifies that."""
    sh = (ROOT / "scripts" / "build_all.sh").read_text()
    code = "\n".join(ln.split("#")[0] for ln in sh.splitlines())
    after = code[code.index("scripts/make_macos_app.py"):]
    assert "swiftc" not in after and "swiftc" not in code
    gate = '"$OUT/O Iris.app/Contents/MacOS/keychain-gate"'
    assert gate + " status" in after and "OIRIS_MAC_GATE_HELPER=" + gate in after
    verify = 'codesign --verify --deep --strict "$OUT/O Iris.app"'
    assert verify in after and after.index("OIRIS_MAC_GATE_HELPER") < after.index(verify) < after.index("ditto ")
    assert '"$OUT/o-iris-$COMMIT-macos-app-adhoc-signed.zip"' in after and "macos-app-unsigned" not in sh
    assert "unsigned .app" not in sh
    assert '"$OUT/o-iris-$COMMIT-release-unsigned.apk"' in code              # the APK stays unsigned
    assert 'assert all(commit in n for n in arts if n.endswith((".apk", ".gz", ".zip")))' in sh
    assert subprocess.run(["/bin/bash", "-n", str(ROOT / "scripts" / "build_all.sh")]).returncode == 0


def test_launcher_script_here_is_contents_in_both_layouts(tmp_path):
    line = next(ln for ln in _maker().LAUNCHER.splitlines() if ln.startswith("HERE="))
    contents = tmp_path / "O Iris.app" / "Contents"
    for sub in ("Resources", "MacOS"):
        (contents / sub).mkdir(parents=True, exist_ok=True)
        script = contents / sub / "o-iris-launcher.sh"
        r = subprocess.run(["/bin/bash", "-c", line + '\necho "$HERE"', str(script)], capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout.strip() == str(contents.resolve()), (sub, r)


# ---- M-17 voice link in the app ----------------------------------------------------------------
def test_bundle_deps_downloads_mac_and_voice_wheels_for_every_python(tmp_path, monkeypatch):
    """--bundle-deps: one offline-able `pip download` of o-iris[mac,voice] per supported Python (faked here:
    nothing is downloaded), all before the compiles and the signature."""
    m = _maker()
    seen = _fake_toolchain(m, monkeypatch)
    whl = _wheel(tmp_path)
    app = m.build(whl, tmp_path, "0.1.0", "abc", bundle_deps=True, swift=True)
    cmds = [cmd for cmd, _ in seen]
    pips = [c for c in cmds if c[1:4] == ["-m", "pip", "download"]]
    assert m.PYTHON_VERSIONS == ("3.11", "3.12", "3.13", "3.14") and m.APP_EXTRAS == "mac,voice"
    assert [c[c.index("--python-version") + 1] for c in pips] == list(m.PYTHON_VERSIONS)
    res = app / "Contents" / "Resources" / "wheels"
    for c in pips:
        assert c[-1] == f"{whl}[mac,voice]" and "--only-binary=:all:" in c and c[c.index("-d") + 1] == str(res)
    assert cmds[:len(pips)] == pips                                     # downloads first
    rest = cmds[len(pips):]                                              # then the unchanged compile + sign order
    assert [c[:2] for c in rest] == [["xcrun", "swiftc"], ["xcrun", "swiftc"], ["codesign", "--force"], ["codesign", "--verify"]]
    assert rest[0][-1] == str(app / "Contents" / "MacOS" / "o-iris") and rest[1][-1] == str(app / "Contents" / "MacOS" / "keychain-gate")


def _fake_home(tmp_path, with_config: bool):
    """A HOME with the app's venv already installed (stamp matches) and fake python/oiris that record their argv."""
    home = tmp_path / ("home-cfg" if with_config else "home")
    support = home / "Library" / "Application Support" / "O Iris"
    vbin = support / "venv" / "bin"
    vbin.mkdir(parents=True)
    (support / "venv" / ".oiris-version").write_text("t\n")
    for prog in ("oiris", "python"):
        p = vbin / prog
        p.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$HOME/' + prog + '.argv"\n')
        p.chmod(0o755)
    if with_config:
        (support / "config.json").write_text('{"flags": {"M-16": true, "M-17": true}}')
    return home, support


def _run_launcher(tmp_path, home):
    script = tmp_path / "Contents" / "Resources" / "o-iris-launcher.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(_maker().LAUNCHER.replace("@VERSION@", "t"))
    env = {"HOME": str(home), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "OIRIS_PYTHON": sys.executable,
           "OIRIS_PASSPHRASE": "s3cret-Passphrase", "OIRIS_PORT": "1"}   # port 1: never answers, so a normal start
    r = subprocess.run(["/bin/bash", str(script)], env=env, capture_output=True, text=True, timeout=60)
    log = (home / "Library" / "Logs" / "O Iris" / "o-iris.log").read_text()
    return r, log


def test_launcher_script_passes_app_config_only_when_it_exists(tmp_path):
    """Runs the real launcher text under macOS /bin/bash 3.2 (set -u) with fake venv programs."""
    home, _ = _fake_home(tmp_path, with_config=False)
    r, log = _run_launcher(tmp_path, home)
    assert r.returncode == 0, (r, log)
    assert (home / "oiris.argv").read_text().splitlines() == ["run", "--port", "1"]   # exactly as before
    assert "no app config; using the defaults" in log and "--config" not in log
    home, support = _fake_home(tmp_path, with_config=True)
    r, log = _run_launcher(tmp_path, home)
    assert r.returncode == 0, (r, log)
    cfg = str(support / "config.json")
    assert (home / "oiris.argv").read_text().splitlines() == ["run", "--port", "1", "--config", cfg]
    assert f"app config: {cfg}" in log and "s3cret-Passphrase" not in log
    # the key store check opens the same store as oiris run: it gets the same --config
    txt = _maker().LAUNCHER
    code = "\n".join(ln.split("#")[0] if ln.lstrip().startswith("#") or "  # " in ln else ln for ln in txt.splitlines())
    assert code.count('${CONFIG_ARGS[@]+"${CONFIG_ARGS[@]}"}') == 3   # key store exists/unlock checks + exec
    assert 'APP_CONFIG="$SUPPORT/config.json"' in txt and 'if [[ -f "$APP_CONFIG" ]]; then CONFIG_ARGS=(--config "$APP_CONFIG"); fi' in txt


def test_keycheck_uses_the_app_config_data_dir(tmp_path):
    from oiris.security.keys import KeyStore
    m = _maker()
    keycheck = m.LAUNCHER.split("KEYCHECK='", 1)[1].split("\n'\n", 1)[0]
    data = tmp_path / "data"
    cfg = tmp_path / "config.json"
    cfg.write_text('{"data_dir": "%s"}' % data)
    run = lambda *a: subprocess.run([sys.executable, "-c", keycheck, *a], env={**os.environ, "OIRIS_PASSPHRASE": "pw"},  # noqa: E731
                                    cwd=ROOT, capture_output=True, text=True).returncode
    assert run("exists", "--config", str(cfg)) == 11                    # no key store in the config's data_dir yet
    data.mkdir()
    KeyStore(data / "keys.json").create("pw")
    assert run("exists", "--config", str(cfg)) == 10
    assert run("unlock", "--config", str(cfg)) == 0


def test_launcher_source_asks_for_the_microphone_only_when_m17_is_on():
    src, code = _launcher_code()
    # the state is read (never prompts) and logged at every launch
    assert "AVCaptureDevice.authorizationStatus(for: .audio)" in code
    assert 'log("microphone authorization \\(name(microphone)); voice link: \\(voice.why)")' in code
    # the only microphone request sits behind the app config's flags."M-17" and an undecided state
    assert code.count("requestAccess(for: .audio)") == 1
    gate = "if voice.on && microphone == .notDetermined {"
    assert gate in code
    block = code[code.index(gate):]
    block = block[:block.index("\n}\n")]
    assert "AVCaptureDevice.requestAccess(for: .audio)" in block and 'log("microphone access ' in block
    assert 'json["flags"] as? [String: Any]' in code and 'flags?["M-17"] as? Bool == true' in code
    assert 'NSHomeDirectory() + "/Library/Application Support/O Iris/config.json"' in code   # the bash launcher's app config
    assert "AVCaptureSession" not in code and "AVAudioEngine" not in code and "AVAudioRecorder" not in code   # never records
    assert code.index("requestAccess(") < code.index("posix_spawn(&pid")
    # the plist carries the purpose string the request needs (macOS kills a process that asks without it)
    assert len(_maker().MICROPHONE_PURPOSE) <= 250


def test_launcher_source_asks_for_the_camera_only_when_m16_and_the_face_camera_are_on():
    """ADR-0057 decision 5: mirrors the microphone path. The request sits behind flags."M-16", face_camera in
    live.adapters and an undecided state; the launcher itself never opens a camera."""
    _, code = _launcher_code()
    assert "AVCaptureDevice.authorizationStatus(for: .video)" in code                 # read at every launch, never prompts
    assert 'log("camera authorization \\(name(camera)); face camera: \\(faceCamera.why)")' in code
    assert code.count("requestAccess(for: .video)") == 1 and code.count("requestAccess(") == 2   # one mic, one camera
    gate = "if faceCamera.on && camera == .notDetermined {"
    assert gate in code
    block = code[code.index(gate):]
    block = block[:block.index("\n}\n")]
    assert "AVCaptureDevice.requestAccess(for: .video)" in block and 'log("camera access ' in block
    fn = code[code.index("func faceCameraOn()"):]
    fn = fn[:fn.index("\n}\n")]
    assert 'flags?["M-16"] as? Bool == true' in fn and '"face_camera"' in fn and '["adapters"]' in fn
    assert "appConfigPath" in fn                                                    # the same app config as the mic
    assert code.index("requestAccess(for: .video)") < code.index("posix_spawn(&pid")
    for banned in ("AVCaptureSession", "AVCaptureVideoDataOutput", "AVCapturePhotoOutput", "AVCaptureDeviceInput"):
        assert banned not in code, banned                                           # never opens or records the camera
    assert len(_maker().CAMERA_PURPOSE) <= 250
