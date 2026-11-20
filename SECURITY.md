# Security policy

## Reporting
Please report vulnerabilities privately via GitHub's **"Report a vulnerability"** (Security → Advisories) on this repository. Do not open a public issue. We aim to acknowledge within 7 days.

## Scope
The security module (M-04) protects **the owner's own data on the owner's own devices**: AES-256-GCM at rest (LUKS2 on the Pi, SQLCipher on the phone), a Noise_XX_25519_ChaChaPoly_BLAKE2s link with rekey every 1,000 messages or 10 minutes, an HMAC-SHA256 hash-chained log with Ed25519 anchors, crypto-erase on panic / 10 failed unlocks / signed wipe from the paired phone.

Pi disk encryption: `sudo scripts/pi_luks2_setup.sh --device /dev/sdX1 --dry-run` prints the plan; drop `--dry-run` to apply (erases the device after a typed confirmation). It is idempotent and never stores the passphrase.

## Known limits
- Python cannot guarantee constant-time behaviour of surrounding code; secret comparisons use `hmac.compare_digest` and a lint test bans `==` on secret-named variables.
- Overwriting flash is unreliable; erasure relies on destroying keys (crypto-erase).
- The Android Noise channel is a Kotlin port (`android/core`, ADR-0053), byte-identical to the Python side on shared test vectors. The BLE transport that carries it has not run on a real radio (see docs/STATUS.md).
- The Android APK is unsigned. The macOS app is only ad-hoc signed: that seals the bundle against later changes but proves nothing about who built it (no Developer ID, no notarization). Verify artifacts against the SHA-256 list in `BUILD_INFO.json` from CI.
- Live mode's network guard refuses non-loopback `connect`, `connect_ex`, `sendto` and `sendmsg` on Python sockets only. It does not police subprocesses or native code.
- No secrets are stored in this repository or in CI.
