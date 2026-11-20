"""Spec E-27 Python-Kotlin replay parity for fusion: tests/vectors/fusion.json (scripts/make_android_vectors.py) is
current, covers the evidence-rule edges, and android/core FusionCore.kt / Checks.kt are wired to it. The Kotlin side
itself runs in `:core:check` (Checks.kt fusionChecks)."""
import importlib.util, json, re
from pathlib import Path

from oiris.fusion import stack as FS

ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "tests/vectors/fusion.json"
KOTLIN = ROOT / "android/core/src/main/kotlin/org/oiris/core/FusionCore.kt"
CHECKS = ROOT / "android/core/src/jvmCheck/kotlin/org/oiris/core/Checks.kt"


def _gen():
    spec = importlib.util.spec_from_file_location("make_android_vectors", ROOT / "scripts/make_android_vectors.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _kotlin_map(name: str) -> dict:
    block = re.search(rf"val {name} = mapOf\((.*?)\n    \)", KOTLIN.read_text(), re.S)
    assert block, name
    return dict(re.findall(r'"(\w+)" to "(\w+)"', block.group(1)))


def test_fusion_vectors_are_current():
    mod = _gen()
    on_disk = json.loads(VECTORS.read_text())
    assert mod.differences(on_disk, json.loads(json.dumps(mod.fusion()))) == []


def test_constants_match_python():
    c = json.loads(VECTORS.read_text())["constants"]
    assert c["families"] == FS.FAMILIES and c["score_family"] == FS.SCORE_FAMILY and c["score_signature"] == FS.SCORE_SIGNATURE
    assert c["score_family"]["S_wifi"] == "rf" and c["family_threshold"] == FS.FAMILY_THRESHOLD
    assert c["default_model"] == json.loads(FS.FusionModel().to_json())
    # the Kotlin source carries the same maps (the JVM checks compare them against the vectors too)
    assert _kotlin_map("SCORE_FAMILY") == FS.SCORE_FAMILY
    assert _kotlin_map("SCORE_SIGNATURE") == FS.SCORE_SIGNATURE


def test_vectors_cover_the_edges():
    v = json.loads(VECTORS.read_text())
    names = {c["name"] for c in v["cases"]}
    for required in ("exactly at the threshold: two families", "one family: fyi", "S_wifi plus rf: still one family",
                     "suppression: two unhealthy families", "NaN and out-of-range are clamped",
                     "a bad health name is rejected before fuse", "custom weights", "fallback node: optical + rf, advisory"):
        assert required in names, required
    by = {c["name"]: c for c in v["cases"]}
    assert by["exactly at the threshold: two families"]["result"]["level"] == "attention"
    assert by["S_wifi plus rf: still one family"]["result"]["families_active"] == ["rf"]
    assert by["S_wifi plus rf: still one family"]["result"]["level"] == "fyi"
    assert by["suppression: two unhealthy families"]["result"]["suppressed"] is True
    assert by["custom weights"]["result"]["model_version"] == "custom-1"
    assert by["a bad health name is rejected before fuse"]["accepted"] == [True, False, True]
    randoms = [c for c in v["cases"] if c["name"].startswith("random ")]
    assert len(randoms) == _gen().FU_RANDOM_CASES
    assert {c["result"]["level"] for c in randoms} == {"attention", "fyi", None}
    assert any(c["result"]["suppressed"] for c in randoms)
    assert {h for c in randoms for *_, h in c["scores"]} == {"ok", "degraded", "failed"}
    assert {m for c in randoms for m, *_ in c["scores"]} >= set(FS.SCORE_FAMILY)
    # NaN / out of range are clamped as schema.clamp01 does; only an unknown health is refused
    assert any(s["score"] is None for s in v["scores"]) and any(s["raw"][1] == "nan" for s in v["scores"])
    assert {w["accepted"] for w in v["weights"]} == {True, False}


def test_kotlin_checks_are_wired():
    checks = CHECKS.read_text()
    assert "fusionChecks(root)" in checks and "tests/vectors/fusion.json" in checks
    src = KOTLIN.read_text()
    assert "FAMILY_THRESHOLD = 0.5" in src and "round6" in src
