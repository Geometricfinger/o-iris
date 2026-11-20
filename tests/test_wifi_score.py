"""S_wifi: AP class in the adapter, the score, and its place in fusion. Synthetic BSSIDs only: the
ordinary one is in the IEEE documentation range (00-00-5E-00-53-xx, RFC 7042), the camera one is a
table OUI with an invented device part, the randomized one is locally administered."""
import json
from pathlib import Path

import pytest

from oiris.adapters import macos, sim
from oiris.config import Config
from oiris.features.rf import AP_CLASSES, CAMERA_OUI_HINTS, ap_class
from oiris.fusion import stack
from oiris.guard.privacy import hash_identifier
from oiris.schema import Health, Sample, Score
from oiris.scores import builtin
from oiris.scores.base import safe_update
from oiris.scores.builtin import WIFI_CAMERA_LIKE_WEIGHT, WIFI_CONFIDENCE, WIFI_NEAR_DBM, WifiApScore

KEY = b"k" * 32
CAM = sorted(CAMERA_OUI_HINTS)[0] + ":00:53:01"
RAND = "02:00:00:00:53:02"
PLAIN = "00:00:5e:00:53:03"
NEAR, FAR = WIFI_NEAR_DBM + 10, WIFI_NEAR_DBM - 20
ROOT = Path(__file__).resolve().parent.parent


def test_synthetic_bssids_are_what_they_claim():
    assert CAM[:8] in CAMERA_OUI_HINTS and PLAIN[:8] not in CAMERA_OUI_HINTS
    assert not int(CAM[:2], 16) & 2 and not int(PLAIN[:2], 16) & 2 and int(RAND[:2], 16) & 2


# ---- ap_class ----

def test_ap_class_three_classes():
    assert ap_class(CAM) == "camera_like" and ap_class(RAND) == "randomized" and ap_class(PLAIN) == "access_point"
    assert {ap_class(b) for b in (CAM, RAND, PLAIN)} == set(AP_CLASSES)
    assert ap_class(CAM.upper()) == "camera_like" and ap_class(CAM.replace(":", "-")) == "camera_like"
    assert ap_class(CAM.replace(":", "")) == "camera_like" and ap_class(f" {CAM} ") == "camera_like"


def test_ap_class_locally_administered_wins_over_the_table():
    # a camera OUI with the locally-administered bit set is not that vendor's address
    la = f"{int(CAM[:2], 16) | 2:02x}" + CAM[2:]
    assert ap_class(la) == "randomized"


@pytest.mark.parametrize("bad", [None, 7, b"00:00:5e:00:53:03", "", "zz:zz:zz:zz:zz:zz", CAM[:8], CAM + ":00",
                                 CAM.replace(":", "", 1), "00:02:d1:00:53", "0:2:d1:0:53:1", "x" + CAM,
                                 CAM[:8] + "-00:53:01", ["a"], {"bssid": CAM}])
def test_ap_class_malformed_is_ordinary_and_never_raises(bad):
    assert ap_class(bad) == "access_point"


# ---- adapter: only the class leaves ----

def test_adapter_sample_has_class_and_no_bssid_or_oui():
    a = macos.WifiMacosAdapter()
    a._key = KEY
    s = a._emit([{"bssid": b, "rssi": -50.0} for b in (CAM, RAND, PLAIN)], 0, "none")
    assert [ap["class"] for ap in s.data["aps"]] == ["camera_like", "randomized", "access_point"]
    assert all(set(ap) == {"key", "tsf_us", "rssi", "local_s", "class"} for ap in s.data["aps"])
    assert [ap["key"] for ap in s.data["aps"]] == [hash_identifier(b, KEY) for b in (CAM, RAND, PLAIN)]
    text = json.dumps(s.data).lower()
    for b in (CAM, RAND, PLAIN):
        h = b.replace(":", "")
        forms = {b, h, b.replace(":", "-")}
        for i in range(4):   # every 3-byte prefix window of the address
            forms |= {h[2 * i:2 * i + 6], ":".join(h[j:j + 2] for j in range(2 * i, 2 * i + 6, 2)),
                      "-".join(h[j:j + 2] for j in range(2 * i, 2 * i + 6, 2))}
        for f in forms:
            assert f not in text, f
    assert a.health()[0] == Health.OK


# ---- the score ----

def scan(*aps, key=None):
    return [Sample("wifi_macos", 0, 0.0, 1, "dBm", 1.0,
                   {"aps": [{"key": key or f"k{i}", "rssi": r, "class": c, "tsf_us": None, "local_s": 0.0}
                            for i, (c, r) in enumerate(aps)]})]


def run(w):
    return safe_update(WifiApScore(), w)


def test_constants_are_ordered():
    assert 0 < WIFI_CONFIDENCE and 0 < WIFI_CAMERA_LIKE_WEIGHT <= 1
    s = builtin.RfScore().update([Sample("x", 0, 0.0, 1, "", 1.0, {"emitters": [{"key": "a", "rssi": -50}]})])
    assert WIFI_CONFIDENCE < s.confidence


def test_near_camera_like_raises_it():
    s = run(scan(("camera_like", NEAR)))
    assert s.value == WIFI_CAMERA_LIKE_WEIGHT and s.confidence == WIFI_CONFIDENCE and s.health == Health.OK
    assert s.evidence == {"near_camera_like": 1, "near": 1, "randomized": 0}
    assert run(scan(("camera_like", WIFI_NEAR_DBM))).value == WIFI_CAMERA_LIKE_WEIGHT   # the edge counts


def test_far_or_ordinary_aps_do_not():
    assert run(scan(("camera_like", FAR))).value == 0
    many = [("access_point", NEAR)] * 10 + [("randomized", NEAR)] * 10
    s = run(scan(*many))
    assert s.value == 0 and s.evidence == {"near_camera_like": 0, "near": 20, "randomized": 10}
    # an adapter that does not classify (Linux iw): counts as ordinary; missing/None rssi is far
    s = run([Sample("wifi_beacons", 0, 0.0, 1, "dBm", 1.0,
                    {"aps": [{"key": "a", "rssi": NEAR}, {"key": "b", "rssi": None}, {"key": "c"}]})])
    assert s.value == 0 and s.evidence == {"near_camera_like": 0, "near": 1, "randomized": 0}


def test_cap_holds():
    s = run(scan(*[("camera_like", NEAR)] * 10))
    assert s.value == 1.0 and s.evidence["near_camera_like"] == 10


def test_key_only_dedupes_within_one_scan():
    s = run(scan(("camera_like", FAR), ("camera_like", NEAR), ("camera_like", NEAR), key="same"))
    assert s.value == WIFI_CAMERA_LIKE_WEIGHT and s.evidence == {"near_camera_like": 1, "near": 1, "randomized": 0}


@pytest.mark.parametrize("w", [[], scan(), [Sample("x", 0, 0.0, 1, "", 1.0, {})],
                               [Sample("x", 0, 0.0, 1, "", 1.0, {"aps": None})]])
def test_no_data(w):
    s = run(w)
    assert (s.value, s.confidence, s.health) == (0.0, 0.0, Health.DEGRADED)
    assert s.module == "S_wifi" and s.evidence == builtin._no_data("S_wifi").evidence


def test_evidence_is_counts_only():
    s = run(scan(("camera_like", NEAR), ("randomized", FAR), ("access_point", NEAR)))
    assert set(s.evidence) == {"near_camera_like", "near", "randomized"}
    assert all(type(v) is int for v in s.evidence.values())
    assert "k0" not in json.dumps(s.evidence)


def test_registered_under_m03():
    assert WifiApScore in builtin.ALL and (WifiApScore.name, WifiApScore.inputs, WifiApScore.module_flag) == \
        ("S_wifi", {"ap_scan"}, "M-03")
    from oiris.engine import Engine
    c = Config()
    assert "S_wifi" in {m.name for m in Engine(c).score_modules}
    c.flags["M-03"] = False
    assert "S_wifi" not in {m.name for m in Engine(c).score_modules}


# ---- fusion ----

def S(m, v, c=0.9, h=Health.OK):
    return Score(m, v, c, h)


def test_wifi_lands_in_rf_with_a4():
    assert stack.SCORE_FAMILY["S_wifi"] == "rf" and stack.SCORE_SIGNATURE["S_wifi"] == "A4"
    assert len(stack.FAMILIES) == 6 and len(set(stack.SCORE_FAMILY.values())) == 6
    r = stack.fuse([S("S_wifi", 1.0, WIFI_CONFIDENCE)])
    assert r["families_active"] == ["rf"] and r["signatures"] == ["A4"]


def test_rf_alone_is_never_an_alert():
    r = stack.fuse([S("S_wifi", 1.0, WIFI_CONFIDENCE), S("S_rf", 1.0), S("S_rf_drift", 1.0)])
    assert r["level"] == "fyi" and r["families_active"] == ["rf"]


def test_rf_plus_one_other_family_as_before():
    before = stack.fuse([S("S_glint", 0.9), S("S_rf", 0.9)])
    assert before["level"] == "attention"
    # a quiet or weaker S_wifi changes nothing: the family takes its strongest value x confidence
    for wifi in (S("S_wifi", 0.0, WIFI_CONFIDENCE), S("S_wifi", 1.0, WIFI_CONFIDENCE)):
        assert stack.fuse([S("S_glint", 0.9), S("S_rf", 0.9), wifi]) == before
    # S_wifi can stand in for S_rf as the rf family; still one rf family plus optical
    r = stack.fuse([S("S_glint", 0.9), S("S_wifi", 1.0, WIFI_CONFIDENCE)])
    assert r["level"] == "attention" and r["families_active"] == ["optical", "rf"]


# ---- golden ----

def test_golden_has_no_ap_scan():
    for s in sim.SCENARIOS:
        assert all("ap_scan" not in a.caps for a in sim.make(s)), s
    files = sorted((ROOT / "golden" / "expected").glob("*.json"))
    assert files and all("S_wifi" not in f.read_text(encoding="utf-8") for f in files)
