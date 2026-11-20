# ADR-0056: A Mac as an advisory live research node

- Status: Accepted (owner sign-off, 2026-11-15)
- Date: 2026-11-15
- Source: docs/spec/ASB-Build-Spec-v2.md section 2 (purpose; non-goals: "no transmit except IR LEDs and the Pi/phone's normal client radios"); ADR-0004, ADR-0009, ADR-0024, ADR-0047; observations on an Apple Silicon Mac, macOS 26.5, 2026-11-15

## Context
D-09 (ADR-0009) sets the platform as the rig plus the Android operator app. Live mode (`oiris run`) and `O Iris.app` now run O Iris on a Mac with real sensors: BLE through bleak and CoreBluetooth, and Wi-Fi from macOS's cached scan. No ADR covers that yet.

The docs had also said things the code does not do:
- "Never transmits" and "passive scanning". bleak's CoreBluetooth backend raises "macOS does not support passive scanning", and `BleScanAdapter` calls `BleakScanner.discover()` with bleak's default (active) scanning on every platform, so it may send BLE scan requests. On Linux, `iw dev <if> scan` has no `passive` keyword. This is normal client-radio activity, which the spec allows.
- "Artifacts are unsigned" (ADR-0047). The macOS app is now ad-hoc signed, because macOS ties Bluetooth and Location permissions to a code identity.

The radio guard's wording is guardrail text (`oiris/guard/README.md`: changes need an ADR and owner sign-off), so the corrected wording needs this record.

## Decision
Accepted with the owner's sign-off on 2026-11-15:

1. **Advisory Mac node.** A Mac may run live mode as an advisory research node with BLE and Wi-Fi only. This is in addition to D-09's rig and Android app; it does not supersede ADR-0009. The node reports anomaly signatures and sensor health, never a person label. A Mac alone feeds one signal family (rf, from BLE; Wi-Fi AP scans are not scored yet), so the level can reach FYI but not ATTENTION.
2. **Radio.** Radio use on the Mac stays within the spec's normal-client-radio rule. Scans may send scan requests. O Iris never connects to or pairs with others' devices and uses no monitor mode, injection, deauthentication or jamming (ADR-0004 still applies in full). On macOS there is no `iw`, so the monitor-mode refusal has nothing to query there. Wi-Fi on macOS starts no scan: the launcher reads `CWInterface.cachedScanResults()`, and macOS's Wi-Fi daemon log showed no scan requests from it. Documents must not say O Iris "never transmits" or "passively scans". ETHICS.md ("Normal client radio only"), README.md, THIRD_PARTY.md (bleak and `iw` rows) and the `oiris/guard/radio.py` docstring were reworded to match the code. Because the radio guard's text is guardrail text, that rewording took effect with the owner's sign-off on this ADR.
3. **Permissions.** The O Iris.app launcher requests Bluetooth, and Location (When In Use) only because macOS hides Wi-Fi network IDs (BSSIDs) without it. It never starts location updates or reads a position, and live mode records sightings with no GPS fix (`no_fix`), so on the Mac nothing reads, stores or sends coordinates. (The Pi rig's NMEA GPS adapter, which stamps sightings when it has a fix, is not part of the Mac node.)
4. **Signing.** The macOS artifact is ad-hoc signed (`codesign --sign -`, verified with `codesign --verify --deep --strict`) and named `o-iris-<commit>-macos-app-adhoc-signed.zip`. This supersedes ADR-0047's "Artifacts are unsigned" for the macOS app only; the Android APK stays unsigned. Developer ID signing and notarization stay an owner step (docs/MACOS.md). It runs only on Apple Silicon with macOS 14 or later.
5. **Storage in live mode.** The retention purge (config `retention_days`, default 30) runs when the store opens and every 24 h. A sighting is written when the (signature, level) pair changes, otherwise at most once per 60 s; suppressed repeats are counted. Open gap: the live sighting hash chain restarts each session and its anchors are not persisted, so live mode does not yet meet ADR-0024 (anchors every 60 s or 100 records). A resumable chain is a separate task.

## Where it is enforced
- `oiris/live.py`: live runner, retention purge, sighting rule, network guard (Python sockets only).
- `oiris/adapters/drivers.py` (`BleScanAdapter`), `oiris/adapters/macos.py` (`WifiMacosAdapter`, `BlePermission`).
- `oiris/guard/radio.py`: tool allow-list and monitor-mode refusal where `iw` exists.
- `macos/launcher/main.swift`: permission requests and the cached-scan Wi-Fi feed; `tests/test_macos_app.py` checks its source (no scan, no association, no position) and the purpose strings (no passive or no-transmit claim).
- `scripts/make_macos_app.py`, `scripts/build_all.sh`: ad-hoc signing, verification, artifact name.

## Consequences
- Every rebuild makes macOS ask for both permissions again: an ad-hoc signature's identity is its code hash (seen three times on 2026-11-15). A Developer ID would fix this.
- Gatekeeper blocks the first open on other Macs (macOS 14: Control-click → Open; macOS 15 and later: System Settings → Privacy & Security → Open Anyway).
- Intel Macs are not supported.
- The live sighting log is not tamper-evident across sessions until the chain is resumable.

## Changing it
Write a new ADR that supersedes this one, update CHANGELOG.md, and (if it touches a guardrail) get owner sign-off. A change to a frozen test threshold cannot be used to declare a pass retroactively.
