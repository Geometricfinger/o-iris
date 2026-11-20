"""tests/vectors/link_ops.json (scripts/make_android_vectors.py) is current, covers every link op, and android/core
LinkOps.kt / Checks.kt are wired to it. The Kotlin side itself runs in `:core:check` (Checks.kt linkOpsChecks)."""
import importlib.util, json, re
from pathlib import Path

from oiris.link import ops as O, sideload as S
from oiris.link.pod_service import LINK_OPS, MODEL_OPS

ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "tests/vectors/link_ops.json"
KOTLIN = ROOT / "android/core/src/main/kotlin/org/oiris/core/LinkOps.kt"
CHECKS = ROOT / "android/core/src/jvmCheck/kotlin/org/oiris/core/Checks.kt"


def _gen():
    spec = importlib.util.spec_from_file_location("make_android_vectors", ROOT / "scripts/make_android_vectors.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_link_ops_vectors_are_current():
    mod = _gen()
    on_disk = json.loads(VECTORS.read_text())
    assert mod.differences(on_disk, json.loads(json.dumps(mod.link_ops()))) == []


def test_every_op_has_a_request_reply_pair_and_parses():
    v = json.loads(VECTORS.read_text())
    ops = {c["op"] for c in v["ops"]}
    assert ops == set(LINK_OPS) | set(MODEL_OPS)
    for op in ops:   # each op has at least one accepted reply and the refusals are there too
        assert any(c["op"] == op and c["parsed"]["ok"] for c in v["ops"]), op
    assert any(c["parsed"].get("error") == O.NOT_AVAILABLE for c in v["ops"])
    mod = _gen()
    for c in v["ops"]:
        assert mod._lo_build(c["op"], c["args"]) == c["request"]
        assert mod._lo_parser(c["op"])(c["reply"]) == c["parsed"]


def test_hostile_hud_vectors_cover_the_caps():
    v = json.loads(VECTORS.read_text())
    names = " ".join(h["name"] for h in v["hud"])
    for word in ("unknown op", "count capped", "text capped", "clamped", "NaN", "beyond a long", "code points"):
        assert word in names
    capped = next(h for h in v["hud"] if h["name"] == "op count capped")["parsed"]
    assert len(capped["ops"]) == O.MAX_HUD_OPS
    for h in v["hud"]:
        p = h["parsed"]
        if p["ok"] and p["ops"]:
            W, H = p["ops"][0]["w"], p["ops"][0]["h"]
            for op in p["ops"][1:]:
                assert 0 <= op.get("x", 0) <= W and 0 <= op.get("y", 0) <= H
                assert all(len(x) <= O.MAX_HUD_TEXT for x in op.values() if isinstance(x, str))


def test_kotlin_constants_match_python():
    kt = KOTLIN.read_text()

    def const(name: str) -> str:
        m = re.search(rf"\b{name}\s*=\s*([^\n]+)", kt)
        assert m, name
        return m.group(1).strip()

    assert const("MAX_HUD_OPS") == str(O.MAX_HUD_OPS) and const("MAX_HUD_TEXT") == str(O.MAX_HUD_TEXT)
    assert const("MAX_CANVAS_PX") == f"{O.MAX_CANVAS_PX}L"
    assert const("CHUNK_MIN") == str(S.CHUNK_MIN) and const("CHUNK_MAX") == str(S.CHUNK_MAX)
    assert const("MAX_MISSING_RANGES") == str(S.MAX_MISSING_RANGES)
    for rx in (S.NAME_RE, S.SLOT_RE, S.SHA256_RE):
        assert f'Regex("{rx.pattern}")' in kt


def test_checks_kt_runs_the_link_ops_vectors():
    checks = CHECKS.read_text()
    assert "linkOpsChecks(root)" in checks and "tests/vectors/link_ops.json" in checks
    for vector_file in ("noise_xx", "enrollment_protocol", "ble_framing", "thermal_render", "eye_android", "voice_link",
                        "glasses_hud", "time_sync"):
        assert f"tests/vectors/{vector_file}.json" in checks     # the existing vectors are still checked
