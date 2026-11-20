"""`oiris` command line. Examples:
  oiris --headless --scenario camera_lens --steps 3
  oiris --headless --replay golden/clips/camera_lens
  oiris check-phrases | selftest | dashboard | record | plugins | version
  oiris glasses-sim --scenario camera_lens --out g.svg   (simulated glasses display)
  oiris face-match status|enable|fetch-models|consent|enroll|list|review|delete-entry|delete-person|photo-delete|disable
      (M-16, opt-in)
  oiris face-match consent --subject CODE [--days 30] [--scope face_enrollment,voice_enrollment,photo,guardian]
      then: oiris face-match enroll --camera auto
  oiris face-match list                       enrolled people, with "photo: yes/no" (never the image)
  oiris face-match photo-delete --person ID   erase that person's photo, verified; the enrollment stays
  oiris voice-link status|enable|fetch-models|devices|enroll|review|delete-heard|delete-voice|disable  (M-17, opt-in)
  oiris voice-link set-self --person ID-PREFIX | clear-self | test-sound [--name NAME]   (ADR-0058 announcements)
  oiris voice-link enroll --person ID-PREFIX [--consent ID] [--input N|NAME] [--speak]   (needs face matching, personal mode)
  oiris [run] [--duration S] [--port P] [--no-dashboard]   live mode, the default (docs/LIVE.md)
The licence audit is a separate script: python scripts/license_check.py
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

from . import __version__
from .config import Config, ConfigError


def _camera_arg(v: str) -> int | str:
    if v.strip().lower() == "auto":
        return "auto"
    try:
        return int(v)
    except ValueError:
        raise argparse.ArgumentTypeError("use a camera index (0, 1, ...) or auto") from None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="oiris", description="O Iris, Anomaly Signature Bench (research tool). Reports anomaly signatures and never "
                                "labels a person; only the opt-in M-16 face matching and M-17 voice link (off by "
                                "default) can recognise people you enrolled.")
    p.add_argument("command", nargs="?", default="run",
                   choices=["run", "selftest", "check-phrases", "dashboard", "record", "version", "face-match", "voice-link", "glasses-sim",
                            "plugins"])
    p.add_argument("sub", nargs="?", help="face-match / voice-link subcommand (face-match: status, enable, fetch-models, "
                                          "consent, enroll, list, review, delete-entry, delete-person, photo-delete, "
                                          "disable)")
    p.add_argument("--mode", action="append", choices=["personal", "research"], help="face-match enable: mode(s)")
    p.add_argument("--jurisdiction", help="face-match enable: jurisdiction profile (unknown -> strictest)")
    p.add_argument("--camera", type=_camera_arg,
                   help="face-match enroll: camera index, or auto (the first camera that sees one face). OpenCV's "
                        "numbering can differ from macOS's camera list; with an iPhone nearby (Continuity Camera) "
                        "camera 0 may be the phone, which may chime while it is probed")
    p.add_argument("--id", help="face-match delete-entry/delete-person: entry or person ID; consent: consent ID; "
                                "voice-link delete-heard: entry ID")
    p.add_argument("--subject", help="face-match consent: subject code (a code, not a name)")
    p.add_argument("--days", type=int, help="face-match consent: days until the record expires (1-365, default 30); "
                                            "voice-link review: days of entries to show")
    p.add_argument("--scope", help="face-match consent: comma-separated scopes from face_enrollment (default), "
                                   "voice_enrollment, photo, guardian")
    p.add_argument("--person", help="voice-link enroll/delete-voice/set-self, face-match photo-delete: person ID or its "
                                    "first characters")
    p.add_argument("--name", help="voice-link test-sound: the name to preview as 'sounds like NAME'; nothing is recorded")
    p.add_argument("--consent", help="voice-link enroll: consent record ID with scope voice_enrollment")
    p.add_argument("--input", help="voice-link enroll: microphone index or part of its name (see voice-link devices); "
                                   "with an iPhone nearby, macOS lists the phone's microphone too (Continuity)")
    p.add_argument("--speak", action="store_true",
                   help="voice-link enroll: the Mac reads each sentence aloud before its window (for children who cannot read yet)")
    p.add_argument("--withdraw", action="store_true", help="face-match consent: withdraw the record given by --id")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--scenario", default="quiet")
    p.add_argument("--replay", type=Path)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--config", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--expect", type=Path, help="compare output to an expected JSON file; exit 1 on mismatch")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--duration", type=float, default=0.0, help="run: stop after SECONDS (0 = until stopped)")
    p.add_argument("--no-dashboard", action="store_true", help="run: live mode without the local dashboard")
    a = p.parse_args(argv)
    try:
        cfg = Config.load(a.config)
    except ConfigError as e:
        print(f"config invalid: {e}", file=sys.stderr)
        return 2
    if a.command == "face-match":
        from .face_match_cli import run
        return run(a, cfg, a.config or Path(cfg.data_dir).expanduser() / "config.json")
    if a.command == "voice-link":
        from .voice_link_cli import run as run_voice
        return run_voice(a, cfg, a.config or Path(cfg.data_dir).expanduser() / "config.json")
    if a.command == "plugins":
        from .registry import default_registry
        from .slots import SLOTS, SlotManager
        reg = default_registry(cfg.plugins)
        for w in reg.warnings:
            print(w, file=sys.stderr)
        sm = SlotManager.from_config(cfg, reg)
        for slot in SLOTS:
            name = sm.slots.get(slot) or "-"
            st = "configured" if slot == "face_match" else (sm.get(slot) and "active") or sm.status.get(slot, "empty")
            avail = ", ".join(reg.names(slot)) or "none installed"
            print(f"{slot:18} -> {name:20} [{st}]  available: {avail}")
        return 0
    if a.command == "version":
        print(__version__); return 0
    if a.command == "check-phrases":
        from .guard.phrases import audit_all
        audit_all(); print("all phrase templates pass the vocabulary guard"); return 0
    if a.command == "record":
        from .headless import record_clip
        record_clip(a.scenario, a.out or Path(f"clip_{a.scenario}"), a.steps); return 0
    if a.command == "glasses-sim":
        from .headless import run_scenario
        from .ui.glasses import render, state_from_result
        svg = render(state_from_result(run_scenario(a.scenario, a.steps, config=cfg)[-1]))
        (a.out.write_text(svg, encoding="utf-8") if a.out else print(svg)); return 0
    if a.command == "dashboard":
        from .dashboard.server import serve
        face = None
        if cfg.enabled("M-16"):  # the face-match page is live only when M-16 is on AND opted in
            from .face_match_cli import _ctx
            from .modules import face_match_gate
            from .registry import default_registry
            _, ks, db, chain = _ctx(cfg)
            try:
                face = face_match_gate.load(cfg, db, ks, default_registry(cfg.plugins), chain=chain)
            except PermissionError as e:
                print(f"face match page disabled: {e}")
        serve(cfg, port=a.port, face_module=face); return 0
    if a.command == "selftest" or a.headless:
        from .headless import run_replay, run_scenario
        res = run_replay(a.replay, None, cfg) if a.replay else run_scenario(a.scenario, a.steps, config=cfg)
        txt = json.dumps(res, indent=1, sort_keys=True)
        if a.out:
            a.out.write_text(txt, encoding="utf-8")
        else:
            print(txt)
        if a.expect:
            ok = json.loads(a.expect.read_text(encoding="utf-8")) == json.loads(txt)
            print("golden: PASS" if ok else "golden: FAIL", file=sys.stderr)
            return 0 if ok else 1
        return 0
    return _run_live(a, cfg)


def _run_live(a, cfg: Config) -> int:
    """`oiris run`: live adapters + engine + encrypted sweep record, dashboard on 127.0.0.1."""
    import threading
    from .live import LiveConfigError, LiveRunner, load_live_settings
    from .security.keys import KeyStoreError
    from .store.db import MigrationError
    if a.duration < 0:
        print("--duration must be >= 0", file=sys.stderr); return 2
    try:
        settings = load_live_settings(a.config)
    except (LiveConfigError, ValueError) as e:
        print(f"config invalid: {e}", file=sys.stderr); return 2
    runner = LiveRunner(cfg, settings)
    try:
        runner.start()
    except (KeyStoreError, MigrationError, OSError) as e:
        print(f"cannot open the encrypted store: {e}", file=sys.stderr); return 2
    for k, v in runner.adapters.items():
        print(f"adapter {k}: " + ("available" if v["available"] else f"unavailable ({v['reason']})"), file=sys.stderr)
    server = None
    if not a.no_dashboard:
        from .dashboard.server import make_server
        try:
            # Quit button: wake runner.wait() exactly as the SIGTERM handler does, so the stop
            # (adapters, sweep record, store close) runs on this thread and main() returns 0.
            server = make_server(cfg, port=a.port, live=runner.snapshot, on_quit=runner._stop.set,
                                 on_listen=runner.voice_listen, on_photo=runner.voice_photo)
        except OSError as e:
            runner.stop("failed")
            print(f"dashboard could not bind 127.0.0.1:{a.port}: {e}", file=sys.stderr); return 2
        threading.Thread(target=server.serve_forever, name="oiris-dashboard", daemon=True).start()
        print(f"dashboard: http://127.0.0.1:{server.server_address[1]}", file=sys.stderr)
    print("live mode running; Ctrl-C or the dashboard's Quit button to stop" + (f" (or after {a.duration:g}s)" if a.duration else ""), file=sys.stderr)
    try:
        outcome = runner.wait(a.duration, install_signals=threading.current_thread() is threading.main_thread())
    finally:
        if server is not None:
            server.shutdown(); server.server_close()
    print(f"live mode stopped ({outcome}) after {runner.steps} steps; sweep {runner.sweep_id or 'not recorded'}"
          + ("; engine step still running at stop, its result was dropped" if runner.loop_alive_at_stop else ""),
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
