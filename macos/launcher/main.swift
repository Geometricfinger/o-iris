// O Iris app launcher (macOS). Apache-2.0.
// The bundle's CFBundleExecutable (Contents/MacOS/o-iris). macOS ties privacy permissions
// (Bluetooth, Location) to the responsible app's code identity and only shows a prompt when that
// app's own main executable asks, so this signed Mach-O does two things:
// 1. It asks for both permissions ITSELF at launch: a CBCentralManager (macOS raises the Bluetooth
//    prompt while the authorization is not determined) and a CLLocationManager that calls
//    requestWhenInUseAuthorization() while undecided. It never scans, connects, starts location
//    updates or reads a position; it only logs the authorization state and its changes.
// 2. It stays the responsible process: it runs Contents/Resources/o-iris-launcher.sh with /bin/bash
//    as a CHILD (posix_spawn, never exec), passes the environment and arguments through, forwards
//    SIGTERM and SIGINT to the child, and exits with the child's status (128 + signal number if the
//    child was killed by a signal).
// 3. It feeds the child the OS's cached Wi-Fi scan. macOS reveals BSSIDs only to a process holding
//    the Location grant, and that is this process, not its grandchildren (system_profiler run by
//    Python got every BSSID withheld after the grant). So every 15 s, while Location is authorized,
//    it reads CWInterface.cachedScanResults() (never a scan of its own, never an association, never a
//    transmission) and writes one JSON line {"t", "status", "aps": [{"bssid", "rssi"}], "withheld"}
//    to a pipe whose read end the child inherits as OIRIS_WIFI_FEED_FD. No SSIDs. Raw BSSIDs exist
//    only in that pipe; they never reach disk or the log, which records only feed state changes.
// 4. The microphone (M-17 voice link, ADR-0057): it logs the authorization state at launch (reading it
//    never prompts) and asks with AVCaptureDevice.requestAccess(for: .audio) only while undecided AND only
//    when the app config (~/Library/Application Support/O Iris/config.json) has flags."M-17" true.
//    Otherwise it never asks. It never opens an audio device; the Python child does, under this grant.
// 5. The camera (M-16 face camera, ADR-0057 decision 5): the same, with AVCaptureDevice.requestAccess(for: .video),
//    only while undecided AND only when the app config has flags."M-16" true and lists "face_camera" in
//    live.adapters. Otherwise it never asks. It never opens a camera; the Python child does, under this grant.
// The main thread runs the main run loop so the permission callbacks arrive (CoreLocation calls its
// delegate on the run loop of the thread that created the manager; dispatchMain() would not run it).
// Log lines are appended to ~/Library/Logs/O Iris/o-iris.log (a Finder launch has no stdout).
// No network, no UI of its own.
import AVFoundation
import CoreBluetooth
import CoreLocation
import CoreWLAN
import Foundation

let logPath = NSHomeDirectory() + "/Library/Logs/O Iris/o-iris.log"
let stamp: DateFormatter = {
    let f = DateFormatter()
    f.locale = Locale(identifier: "en_US_POSIX")
    f.dateFormat = "yyyy-MM-dd HH:mm:ss"   // same format as the bash launcher's log()
    return f
}()

// One line per call, opened with O_APPEND each time so it interleaves cleanly with the bash launcher's
// writes and follows its log rotation. Called on the main thread only.
func log(_ message: String) {
    let line = "\(stamp.string(from: Date())) o-iris: \(message)\n"
    try? FileManager.default.createDirectory(atPath: (logPath as NSString).deletingLastPathComponent,
                                             withIntermediateDirectories: true)
    let fd = Darwin.open(logPath, O_WRONLY | O_APPEND | O_CREAT, 0o644)
    if fd < 0 { return }
    _ = line.withCString { write(fd, $0, strlen($0)) }
    close(fd)
}

// ---- permissions: requested by this process, the app's main executable ----
func name(_ a: CBManagerAuthorization) -> String {
    switch a {
    case .notDetermined: return "notDetermined"
    case .restricted: return "restricted"
    case .denied: return "denied"
    case .allowedAlways: return "allowedAlways"
    @unknown default: return "unknown(\(a.rawValue))"
    }
}

func name(_ s: CBManagerState) -> String {
    switch s {
    case .unknown: return "unknown"
    case .resetting: return "resetting"
    case .unsupported: return "unsupported"
    case .unauthorized: return "unauthorized"
    case .poweredOff: return "poweredOff"
    case .poweredOn: return "poweredOn"
    @unknown default: return "unknown(\(s.rawValue))"
    }
}

func name(_ s: CLAuthorizationStatus) -> String {
    switch s {
    case .notDetermined: return "notDetermined"
    case .restricted: return "restricted"
    case .denied: return "denied"
    case .authorizedAlways: return "authorizedAlways"
    case .authorizedWhenInUse: return "authorizedWhenInUse"
    @unknown default: return "unknown(\(s.rawValue))"
    }
}

func name(_ s: AVAuthorizationStatus) -> String {
    switch s {
    case .notDetermined: return "notDetermined"
    case .restricted: return "restricted"
    case .denied: return "denied"
    case .authorized: return "authorized"
    @unknown default: return "unknown(\(s.rawValue))"
    }
}

// The app config the bash launcher passes to oiris (--config) when it exists. Only flags."M-17" is read.
let appConfigPath = NSHomeDirectory() + "/Library/Application Support/O Iris/config.json"

func voiceLinkOn() -> (on: Bool, why: String) {
    guard FileManager.default.fileExists(atPath: appConfigPath) else { return (false, "no app config") }
    guard let data = FileManager.default.contents(atPath: appConfigPath),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        return (false, "app config unreadable")
    }
    let flags = json["flags"] as? [String: Any]
    return flags?["M-17"] as? Bool == true ? (true, "M-17 on in the app config") : (false, "M-17 off in the app config")
}

// Only flags."M-16" and live.adapters (a name string or {"name": ...}) are read.
func faceCameraOn() -> (on: Bool, why: String) {
    guard FileManager.default.fileExists(atPath: appConfigPath) else { return (false, "no app config") }
    guard let data = FileManager.default.contents(atPath: appConfigPath),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        return (false, "app config unreadable")
    }
    let flags = json["flags"] as? [String: Any]
    guard flags?["M-16"] as? Bool == true else { return (false, "M-16 off in the app config") }
    let adapters = (json["live"] as? [String: Any])?["adapters"] as? [Any] ?? []
    let listed = adapters.contains { ($0 as? String ?? ($0 as? [String: Any])?["name"] as? String) == "face_camera" }
    return listed ? (true, "M-16 on and face_camera in the app config") : (false, "face_camera not in the app config")
}

final class PermissionWatcher: NSObject, CBCentralManagerDelegate, CLLocationManagerDelegate {
    var lastBluetooth = ""
    var lastLocation = ""

    // Bluetooth: called with every state change (the first right after the manager is created).
    // Only the state and the authorization are logged; nothing is scanned or connected.
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        let now = "state \(name(central.state)), authorization \(name(CBManager.authorization))"
        if now != lastBluetooth { lastBluetooth = now; log("bluetooth \(now)") }
    }

    // Location: called right after the delegate is set and whenever the user answers or changes it.
    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        let now = name(manager.authorizationStatus)
        if now != lastLocation { lastLocation = now; log("location authorization \(now)") }
    }
}

let watcher = PermissionWatcher()
log("start (pid \(getpid())); bluetooth authorization \(name(CBManager.authorization)), asking macOS while not determined")
// Creating the manager is what makes macOS raise "O Iris would like to use Bluetooth" when undecided.
let central = CBCentralManager(delegate: watcher, queue: DispatchQueue.main,
                               options: [CBCentralManagerOptionShowPowerAlertKey: false])
let location = CLLocationManager()   // on the main thread: its delegate runs on the main run loop
location.delegate = watcher
if location.authorizationStatus == .notDetermined {
    log("location authorization notDetermined; requesting When In Use")
    location.requestWhenInUseAuthorization()
}
// Microphone: only for the voice link. Without flags."M-17" in the app config it is never requested.
let microphone = AVCaptureDevice.authorizationStatus(for: .audio)   // reading it never prompts
let voice = voiceLinkOn()
log("microphone authorization \(name(microphone)); voice link: \(voice.why)")
if voice.on && microphone == .notDetermined {
    log("microphone authorization notDetermined; requesting access for the voice link")
    AVCaptureDevice.requestAccess(for: .audio) { granted in
        DispatchQueue.main.async { log("microphone access \(granted ? "granted" : "not granted")") }
    }
}
// Camera: only for the face camera. Without flags."M-16" and face_camera in the app config it is never requested.
let camera = AVCaptureDevice.authorizationStatus(for: .video)   // reading it never prompts
let faceCamera = faceCameraOn()
log("camera authorization \(name(camera)); face camera: \(faceCamera.why)")
if faceCamera.on && camera == .notDetermined {
    log("camera authorization notDetermined; requesting access for the face camera")
    AVCaptureDevice.requestAccess(for: .video) { granted in
        DispatchQueue.main.async { log("camera access \(granted ? "granted" : "not granted")") }
    }
}

// ---- passive Wi-Fi feed: the OS's cached scan, read under this process's Location grant ----
let feedEnv = "OIRIS_WIFI_FEED_FD"
let feedPeriod = 15.0   // seconds between lines
var feedPipe: [Int32] = [-1, -1]   // [read end: inherited by the child, write end: stays here]
if pipe(&feedPipe) == 0 {
    _ = fcntl(feedPipe[1], F_SETFD, FD_CLOEXEC)   // the child never gets the write end (EOF when we exit)
    _ = fcntl(feedPipe[1], F_SETFL, fcntl(feedPipe[1], F_GETFL) | O_NONBLOCK)   // a lagging reader never blocks us
    _ = fcntl(feedPipe[0], F_SETFD, 0)            // the read end crosses posix_spawn and bash's exec
} else {
    log("wifi feed: no pipe (\(String(cString: strerror(errno)))); the child falls back to system_profiler")
    feedPipe = [-1, -1]
}
signal(SIGPIPE, SIG_IGN)   // a closed reader shows up as EPIPE from write(), which stops the feed

// "aa:b:..." -> "aa:0b:...": six hex octets, lower case; anything else is not a usable identifier
func normalizedMAC(_ s: String) -> String? {
    let parts = s.lowercased().split(separator: ":", omittingEmptySubsequences: false)
    guard parts.count == 6 else { return nil }
    var octets: [String] = []
    for p in parts {
        guard (1...2).contains(p.count), p.allSatisfy({ $0.isHexDigit }) else { return nil }
        octets.append(p.count == 1 ? "0" + p : String(p))
    }
    let mac = octets.joined(separator: ":")
    return mac == "00:00:00:00:00:00" ? nil : mac
}

// One reading of the OS's cached scan results. cachedScanResults() only returns what macOS already
// has; it starts no scan. SSIDs are left out.
func wifiSnapshot(authorized: Bool) -> [String: Any] {
    var out: [String: Any] = ["t": Date().timeIntervalSince1970, "aps": [[String: Any]](), "withheld": 0]
    guard authorized else { out["status"] = "location_not_authorized"; return out }
    guard let itf = CWWiFiClient.shared().interface() else { out["status"] = "no_interface"; return out }
    guard itf.powerOn() else { out["status"] = "wifi_off"; return out }
    guard let nets = itf.cachedScanResults() else { out["status"] = "no_cached_results"; return out }
    var aps: [[String: Any]] = []
    var withheld = 0
    for n in nets {
        if let b = n.bssid, let mac = normalizedMAC(b) {
            aps.append(["bssid": mac, "rssi": n.rssiValue])
        } else {
            withheld += 1
        }
    }
    out["status"] = "ok"
    out["aps"] = aps
    out["withheld"] = withheld
    return out
}

let feedQueue = DispatchQueue(label: "o-iris.launcher.wififeed")   // serial: owns the feed state below
var feedTimer: DispatchSourceTimer?
var feedPending = Data()   // the unwritten tail of a line (pipe writes over PIPE_BUF may be partial)
var feedLastState = ""

func feedLog(_ message: String) {   // log() runs on the main thread only
    DispatchQueue.main.async { log(message) }
}

func feedStop(_ why: String) {
    feedTimer?.cancel()
    feedTimer = nil
    if feedPipe[1] >= 0 { close(feedPipe[1]); feedPipe[1] = -1 }
    feedPending = Data()
    feedLog("wifi feed: stopped (\(why))")
}

// Non-blocking write of the pending bytes. false: the reader is gone (EPIPE) or the pipe broke.
func feedFlush() -> Bool {
    while !feedPending.isEmpty {
        let n = feedPending.withUnsafeBytes { write(feedPipe[1], $0.baseAddress, $0.count) }
        if n > 0 { feedPending.removeFirst(n); continue }
        if n < 0 && errno == EINTR { continue }
        if n < 0 && errno == EAGAIN { return true }   // full pipe: keep the tail for the next tick
        feedStop(n < 0 && errno == EPIPE ? "reader closed" : "write failed: \(String(cString: strerror(errno)))")
        return false
    }
    return true
}

func feedTick() {
    guard feedPipe[1] >= 0 else { return }
    let status = DispatchQueue.main.sync { location.authorizationStatus }   // the manager lives on main
    // by name: the Swift SDK marks .authorizedWhenInUse unavailable on macOS, yet macOS 26 reports it
    let snap = wifiSnapshot(authorized: ["authorizedAlways", "authorizedWhenInUse"].contains(name(status)))
    let state = snap["status"] as? String ?? "?"
    let aps = (snap["aps"] as? [[String: Any]])?.count ?? 0
    let withheld = snap["withheld"] as? Int ?? 0
    let key = state == "ok" ? "ok\(withheld > 0 ? "+withheld" : "")\(aps > 0 ? "" : "+empty")" : state
    if key != feedLastState {   // counts only, never an identifier
        feedLastState = key
        feedLog(state == "ok" ? "wifi feed: ok, \(aps + withheld) networks (\(withheld) withheld)"
                              : "wifi feed: \(state)")
    }
    guard feedFlush() else { return }
    guard feedPending.isEmpty else { return }   // the reader lags: drop this line
    guard var line = try? JSONSerialization.data(withJSONObject: snap) else { return }
    line.append(0x0a)
    let n = line.withUnsafeBytes { write(feedPipe[1], $0.baseAddress, $0.count) }
    if n < 0 {
        if errno == EAGAIN || errno == EINTR { return }   // drop the line; the next tick tries again
        feedStop(errno == EPIPE ? "reader closed" : "write failed: \(String(cString: strerror(errno)))")
        return
    }
    if n < line.count { feedPending = line.subdata(in: n..<line.count) }   // never tear a line
}

// ---- the bash launcher as a child process ----
let executable = (Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])).resolvingSymlinksInPath()
let script = executable.deletingLastPathComponent().deletingLastPathComponent()   // Contents/MacOS -> Contents
    .appendingPathComponent("Resources").appendingPathComponent("o-iris-launcher.sh").path

let forwarded: [Int32] = [SIGTERM, SIGINT]
let queue = DispatchQueue(label: "o-iris.launcher.signals")   // serial: owns `child`
var child: pid_t = 0
var sources: [DispatchSourceSignal] = []
for sig in forwarded {
    signal(sig, SIG_IGN)   // delivered through the dispatch source instead of killing the launcher
    let source = DispatchSource.makeSignalSource(signal: sig, queue: queue)
    source.setEventHandler {
        if child > 0 { kill(child, sig) } else { exit(128 + sig) }   // before the spawn: nothing to forward
    }
    source.resume()
    sources.append(source)
}

// The child starts with default handlers for the forwarded signals and SIGPIPE (ignored here only for
// the feed) and an empty signal mask.
var attr: posix_spawnattr_t?
posix_spawnattr_init(&attr)
var defaults = sigset_t()
sigemptyset(&defaults)
for sig in forwarded + [SIGPIPE] { sigaddset(&defaults, sig) }
var mask = sigset_t()
sigemptyset(&mask)
posix_spawnattr_setsigdefault(&attr, &defaults)
posix_spawnattr_setsigmask(&attr, &mask)
posix_spawnattr_setflags(&attr, Int16(POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETSIGMASK))

let args = ["/bin/bash", script] + CommandLine.arguments.dropFirst()
var argv: [UnsafeMutablePointer<CChar>?] = args.map { strdup($0) } + [nil]
// The child's environment: ours, passed through, with OIRIS_WIFI_FEED_FD set to the feed's read end
// (an inherited value is never passed on: only this launcher's pipe is a feed).
var childEnv: [String] = []
var ep = environ
while let entry = ep.pointee {
    let s = String(cString: entry)
    if !s.hasPrefix(feedEnv + "=") { childEnv.append(s) }
    ep += 1
}
if feedPipe[0] >= 0 { childEnv.append("\(feedEnv)=\(feedPipe[0])") }
var envp: [UnsafeMutablePointer<CChar>?] = childEnv.map { strdup($0) } + [nil]
let rc: Int32 = queue.sync {   // a signal is handled either before the spawn (exit) or after it (forward)
    var pid: pid_t = 0
    let r = posix_spawn(&pid, "/bin/bash", nil, &attr, &argv, &envp)
    if r == 0 { child = pid }
    return r
}
posix_spawnattr_destroy(&attr)
for p in argv { free(p) }
for p in envp { free(p) }
if feedPipe[0] >= 0 { close(feedPipe[0]); feedPipe[0] = -1 }   // only the child reads; EOF reaches it if we exit
if rc != 0 {
    log("cannot start /bin/bash \(script): \(String(cString: strerror(rc)))")
    FileHandle.standardError.write("o-iris: cannot start /bin/bash \(script): \(String(cString: strerror(rc)))\n".data(using: .utf8)!)
    exit(126)
}

// The feed starts once the child exists; its first line goes out at once, then every 15 s.
if feedPipe[1] >= 0 {
    feedQueue.sync {
        let t = DispatchSource.makeTimerSource(queue: feedQueue)
        t.schedule(deadline: .now(), repeating: feedPeriod, leeway: .seconds(1))
        t.setEventHandler { feedTick() }
        feedTimer = t
        t.resume()
    }
}

// Wait for the child off the main thread; exit from the main queue with its status when it ends.
let pid = queue.sync { child }
DispatchQueue.global().async {
    var status: Int32 = 0
    while waitpid(pid, &status, 0) == -1 {
        if errno != EINTR {
            let reason = String(cString: strerror(errno))
            DispatchQueue.main.async {
                FileHandle.standardError.write("o-iris: waitpid failed: \(reason)\n".data(using: .utf8)!)
                exit(1)
            }
            return
        }
    }
    let termsig = status & 0x7f   // WIFEXITED / WEXITSTATUS / WTERMSIG are C macros, not imported into Swift
    let code = termsig == 0 ? (status >> 8) & 0xff : 128 + termsig
    DispatchQueue.main.async { exit(code) }
}

// The main run loop delivers the Bluetooth and Location callbacks and the main-queue exit above.
// A timer keeps it alive: RunLoop.run() returns at once if no source or timer is attached.
RunLoop.main.add(Timer(timeInterval: 86_400, repeats: true) { _ in }, forMode: .default)
RunLoop.main.run()
withExtendedLifetime((central, location, watcher, sources)) {}
exit(0)   // not reached: the main queue exits with the child's status
