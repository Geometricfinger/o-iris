"""Model side-load over the link (ADR-0062): pins, offer / chunk / commit / status, resume, refusals, symlinks.
Synthetic only: generated model bytes in a temp models tree."""
import hashlib, json, os

import pytest

from oiris.link import sideload as S
from oiris.link.ops import b64url
from oiris.link.pod_service import PodService

CHUNK = 1024
MODEL = bytes((i * 7 + 3) % 251 for i in range(5 * CHUNK + 300))   # 6 chunks, the last one short
SHA = hashlib.sha256(MODEL).hexdigest()
PIN = {"name": "detector-v2.onnx", "slot": "detector", "sha256": SHA, "size": len(MODEL)}
PLENTY = 1 << 40


def pins_file(tmp_path, pins) -> str:
    p = tmp_path / S.PINS_FILE
    p.write_text(json.dumps(pins))
    return str(p)


@pytest.fixture
def sl(tmp_path):
    return S.Sideload(pins_file(tmp_path, [PIN]), tmp_path / "models", free_bytes=lambda d: PLENTY)


def svc_of(sl) -> PodService:
    return PodService(sideload=sl)


def call(svc, req: str) -> dict:
    return json.loads(svc.handle(req))


def offer(svc, pin=PIN, chunk_size=CHUNK) -> dict:
    return call(svc, S.offer_request(pin["name"], pin["slot"], pin["sha256"], pin["size"], chunk_size))


def send(svc, data: bytes, indices, sha=SHA, chunk_size=CHUNK) -> list[dict]:
    return [call(svc, S.chunk_request(sha, i, data[i * chunk_size:(i + 1) * chunk_size])) for i in indices]


def test_a_pinned_model_installs(sl, tmp_path):
    svc = svc_of(sl)
    o = offer(svc)
    assert o == {"ok": True, "chunks": 6, "missing": [[0, 6]], "missing_count": 6}
    assert [r["missing_count"] for r in send(svc, MODEL, range(6))] == [5, 4, 3, 2, 1, 0]
    assert call(svc, S.commit_request(SHA)) == {"ok": True, "installed": "detector/detector-v2.onnx"}
    target = tmp_path / "models" / "detector" / "detector-v2.onnx"
    assert target.read_bytes() == MODEL
    assert not any((tmp_path / "models" / S.PARTIAL_DIR).iterdir())
    assert call(svc, S.status_request()) == {"ok": True, "models": [PIN]}


def test_status_lists_only_installed_pinned_models(tmp_path):
    other = {**PIN, "name": "other.onnx", "sha256": "0" * 64}
    sl = S.Sideload(pins_file(tmp_path, [PIN, other]), tmp_path / "models", free_bytes=lambda d: PLENTY)
    svc = svc_of(sl)
    assert call(svc, S.status_request()) == {"ok": True, "models": []}
    (tmp_path / "models" / "detector").mkdir(parents=True)
    (tmp_path / "models" / "detector" / "other.onnx").write_bytes(b"short")      # wrong size: not installed
    os.symlink(tmp_path / S.PINS_FILE, tmp_path / "models" / "detector" / "detector-v2.onnx")
    assert call(svc, S.status_request()) == {"ok": True, "models": []}


def test_unpinned_and_inexact_offers_are_refused(sl):
    svc = svc_of(sl)
    refused = {"ok": False, "error": S.NOT_PINNED}
    assert offer(svc, {**PIN, "sha256": "a" * 64}) == refused
    assert offer(svc, {**PIN, "size": PIN["size"] + 1}) == refused
    assert offer(svc, {**PIN, "name": "detector-v3.onnx"}) == refused
    assert offer(svc, {**PIN, "slot": "face_match"}) == refused


def test_wrong_hash_at_commit_refuses_and_removes_the_partial(tmp_path):
    bad = bytes(len(MODEL))
    sl = S.Sideload(pins_file(tmp_path, [PIN]), tmp_path / "models", free_bytes=lambda d: PLENTY)
    svc = svc_of(sl)
    offer(svc)
    send(svc, bad, range(6))
    part = tmp_path / "models" / S.PARTIAL_DIR / SHA
    assert part.exists()
    assert call(svc, S.commit_request(SHA)) == {"ok": False, "error": S.HASH_MISMATCH}
    assert not part.exists() and not (tmp_path / "models" / "detector" / "detector-v2.onnx").exists()
    assert call(svc, S.commit_request(SHA)) == {"ok": False, "error": S.NO_OFFER}


def test_commit_needs_every_chunk(sl):
    svc = svc_of(sl)
    offer(svc)
    send(svc, MODEL, [0, 1, 2, 3, 5])
    assert call(svc, S.commit_request(SHA)) == {"ok": False, "error": S.INCOMPLETE}
    assert offer(svc)["missing"] == [[4, 5]]


def test_transfer_resumes_after_an_interruption(tmp_path):
    big = bytes((i * 13) % 256 for i in range(40 * 512))
    pin = {"name": "m.bin", "slot": "detector", "sha256": hashlib.sha256(big).hexdigest(), "size": len(big)}
    pins = pins_file(tmp_path, [pin])
    sl = S.Sideload(pins, tmp_path / "models", free_bytes=lambda d: PLENTY)
    svc = svc_of(sl)
    assert offer(svc, pin, 512)["missing_count"] == 40
    send(svc, big, range(S.PERSIST_EVERY + 3), pin["sha256"], 512)
    # the same pod, link dropped: the phone offers again and gets exactly what is missing
    assert offer(svc, pin, 512) == {"ok": True, "chunks": 40, "missing": [[19, 40]], "missing_count": 21}
    # pod restart: the progress record has the chunks up to the last persist; the rest are asked for again
    svc2 = svc_of(S.Sideload(pins, tmp_path / "models", free_bytes=lambda d: PLENTY))
    o = offer(svc2, pin, 512)
    assert o["missing"] == [[S.PERSIST_EVERY, 40]]
    send(svc2, big, range(S.PERSIST_EVERY, 40), pin["sha256"], 512)
    assert send(svc2, big, [0], pin["sha256"], 512) == [{"ok": True, "missing_count": 0}]   # idempotent
    assert call(svc2, S.commit_request(pin["sha256"]))["ok"] is True
    assert (tmp_path / "models" / "detector" / "m.bin").read_bytes() == big


def test_a_different_chunk_size_restarts(sl, tmp_path):
    svc = svc_of(sl)
    offer(svc)
    send(svc, MODEL, range(3))
    assert offer(svc, chunk_size=2048) == {"ok": True, "chunks": 3, "missing": [[0, 3]], "missing_count": 3}


def test_missing_reply_is_capped(tmp_path):
    size = 512 * (2 * S.MAX_MISSING_RANGES + 4)
    pin = {"name": "m.bin", "slot": "detector", "sha256": "ab" * 32, "size": size}
    svc = svc_of(S.Sideload(pins_file(tmp_path, [pin]), tmp_path / "models", free_bytes=lambda d: PLENTY))
    offer(svc, pin, 512)
    send(svc, bytes(size), range(0, size // 512, 2), pin["sha256"], 512)   # every other chunk
    o = offer(svc, pin, 512)
    assert len(o["missing"]) == S.MAX_MISSING_RANGES and o["missing_count"] == size // 512 // 2
    assert S.parse_offer(json.dumps(o))["ok"] is True


def test_chunk_bounds_and_lengths(sl):
    svc = svc_of(sl)
    assert send(svc, MODEL, [0]) == [{"ok": False, "error": S.NO_OFFER}]
    offer(svc)
    for i in (-1, 6, 1 << 40, True, "1", 1.0):
        assert call(svc, json.dumps({"op": "model_chunk", "sha256": SHA, "index": i, "data": b64url(MODEL[:CHUNK])})) \
            == {"ok": False, "error": S.CHUNK_RANGE}
    assert call(svc, S.chunk_request(SHA, 0, MODEL[:CHUNK - 1])) == {"ok": False, "error": S.CHUNK_LENGTH}
    assert call(svc, S.chunk_request(SHA, 0, MODEL[:CHUNK] + b"x")) == {"ok": False, "error": S.CHUNK_LENGTH}
    assert call(svc, S.chunk_request(SHA, 5, MODEL[5 * CHUNK:] + b"x")) == {"ok": False, "error": S.CHUNK_LENGTH}
    assert call(svc, S.chunk_request(SHA, 5, MODEL[5 * CHUNK:]))["ok"] is True                  # the short last chunk
    assert call(svc, json.dumps({"op": "model_chunk", "sha256": SHA, "index": 0, "data": "!!"}))["error"] == S.CHUNK_LENGTH


def test_offer_limits(sl, tmp_path):
    svc = svc_of(sl)
    for cs in (S.CHUNK_MIN - 1, S.CHUNK_MAX + 1, True, 1024.0):
        assert offer(svc, chunk_size=cs) == {"ok": False, "error": S.BAD_CHUNK_SIZE}
    assert offer(svc, chunk_size=S.CHUNK_MIN)["ok"] and offer(svc, chunk_size=S.CHUNK_MAX)["ok"]
    assert offer(svc, {**PIN, "size": S.MAX_MODEL_BYTES + 1}) == {"ok": False, "error": S.TOO_LARGE}
    full = svc_of(S.Sideload(pins_file(tmp_path, [PIN]), tmp_path / "m2", free_bytes=lambda d: len(MODEL)))
    assert offer(full) == {"ok": False, "error": S.NO_SPACE}


def test_traversal_names_are_refused(sl, tmp_path):
    svc = svc_of(sl)
    for name, slot in [("../evil", "detector"), ("a/b", "detector"), (".hidden", "detector"), ("x", "../up"),
                       ("x", "Detector"), ("", "detector"), ("x", "")]:
        assert offer(svc, {**PIN, "name": name, "slot": slot}) == {"ok": False, "error": S.BAD_OFFER}
    for bad in (["../evil"], ["ok", "a/b"]):
        pins = [{**PIN, "name": bad[-1]}]
        with pytest.raises(S.PinError):
            S.load_pins(pins_file(tmp_path, pins))


def test_load_pins_validates_strictly(tmp_path):
    assert S.load_pins(tmp_path / "absent.json") == []
    for bad in ([{**PIN, "sha256": SHA.upper()}], [{**PIN, "sha256": SHA[:-1]}], [{**PIN, "size": 0}],
                [{**PIN, "size": True}], [{**PIN, "size": S.MAX_MODEL_BYTES + 1}], [{**PIN, "extra": 1}],
                [{k: v for k, v in PIN.items() if k != "size"}], [PIN, PIN], {"pins": []}):
        with pytest.raises(S.PinError):
            S.load_pins(pins_file(tmp_path, bad))
    (tmp_path / S.PINS_FILE).write_text("[{")
    with pytest.raises(S.PinError):
        S.load_pins(tmp_path / S.PINS_FILE)
    assert S.load_pins(pins_file(tmp_path, [PIN])) == [S.Pin(**PIN)]


def test_invalid_pins_refuse_every_offer(tmp_path):
    p = pins_file(tmp_path, [PIN, {"name": "../x"}])
    svc = svc_of(S.Sideload(p, tmp_path / "models", free_bytes=lambda d: PLENTY))
    assert offer(svc) == {"ok": False, "error": S.PINS_INVALID}


def test_no_op_can_add_a_pin(sl, tmp_path):
    svc = svc_of(sl)
    for op in ("model_pin", "add_pin", "pin"):
        call(svc, json.dumps({"op": op, **PIN, "sha256": "c" * 64}))
    assert S.load_pins(tmp_path / S.PINS_FILE) == [S.Pin(**PIN)]


def test_symlinked_partial_file_is_refused(sl, tmp_path):
    victim = tmp_path / "victim"
    victim.write_bytes(b"keep me")
    d = tmp_path / "models" / S.PARTIAL_DIR
    d.mkdir(parents=True)
    os.symlink(victim, d / SHA)
    svc = svc_of(sl)
    assert offer(svc) == {"ok": False, "error": S.UNSAFE_PATH}
    (d / f"{SHA}.json").write_text(json.dumps({"v": 1, "size": PIN["size"], "chunk_size": CHUNK, "have": [[0, 1]]}))
    assert offer(svc) == {"ok": False, "error": S.UNSAFE_PATH}
    assert victim.read_bytes() == b"keep me"


def test_symlinked_directories_are_refused(sl, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "models").mkdir()
    os.symlink(elsewhere, tmp_path / "models" / S.PARTIAL_DIR)
    assert offer(svc_of(sl)) == {"ok": False, "error": S.UNSAFE_PATH}
    assert not any(elsewhere.iterdir())


def test_symlinked_slot_directory_is_refused_at_commit(sl, tmp_path):
    svc = svc_of(sl)
    offer(svc)
    send(svc, MODEL, range(6))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    os.symlink(elsewhere, tmp_path / "models" / "detector")
    assert call(svc, S.commit_request(SHA)) == {"ok": False, "error": S.UNSAFE_PATH}
    assert not any(elsewhere.iterdir())


def test_chunk_count_and_ranges():
    assert S.chunk_count(1, 512) == 1 and S.chunk_count(512, 512) == 1 and S.chunk_count(513, 512) == 2
    r: list = []
    for i in (5, 1, 2, 0, 9):
        r = S._add(r, i)
    assert r == [[0, 3], [5, 6], [9, 10]] and S._missing(r, 11) == [[3, 5], [6, 9], [10, 11]]
