# oiris.dashboard
Stdlib HTTP server on 127.0.0.1 with zones D1–D6 from the C6 map and `/api/latest` JSON. Strict CSP, no external assets. Every GET needs `Host` `127.0.0.1:<port>` or `localhost:<port>` (else 403). Replace freely; it only reads C3/C4 data.

Two modes. `oiris dashboard` alone shows simulated headless data. In live mode (`oiris run`, docs/LIVE.md) it shows `LiveRunner.snapshot()` and never simulates: D1 says "No live data yet" until a signal family is active, D5 gives each sensor a plain status and what to do, and a **Quit O Iris** button stops live mode (POST `/quit` with a per-run token and a same-origin `Origin`; the sweep is recorded as `done`).

`/face` (face.py): M-16 review page. Present only when M-16 is on and opted in; otherwise shows the off notice and the enrollment guide. All changes are token-protected POSTs; disable requires typing WIPE and reports the verified wipe.
