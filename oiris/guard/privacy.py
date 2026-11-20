"""Privacy guard: what may be persisted.

- No audio, ever (ADR-0039; the M-17 voice link included).
- No raw MACs, no full frames of people.
- No face templates except enrolled-person templates from the opted-in M-16 module (ADR-0049).
- No voice templates except enrolled-person templates from the opted-in M-17 module, sealed by its
  core (ADR-0057 decision 3).
- No face photos except of enrolled people with recorded photo consent, from M-16/M-17 while enabled
  (ADR-0057 decision 4).
- No stranger face data of any kind (crops, embeddings, IDs, hashes)."""
from __future__ import annotations
import hashlib, hmac


class PrivacyError(PermissionError):
    pass


def hash_identifier(identifier: str, site_key: bytes) -> str:
    """Raw MAC/BSSID never stored: HMAC-SHA256 with a per-site key (truncated)."""
    return hmac.new(site_key, identifier.lower().encode(), hashlib.sha256).hexdigest()[:32]


# Only the M-16 module core may persist a face template, and only while M-16 is enabled
# and opted in (ADR-0049). Every other source, and M-16 while off, is refused.
FACE_TEMPLATE_SOURCE = "M-16"
# Only the M-17 module core may persist a voice template (sealed with the person's own key), and only
# while M-17 is enabled and opted in (ADR-0057). Every other source, and M-17 while off, is refused.
VOICE_TEMPLATE_SOURCE = "M-17"
# A face photo of an enrolled, photo-consenting person may come from either module (ADR-0057 decision 4).
FACE_PHOTO_SOURCES = ("M-16", "M-17")


def check_persist(kind: str, contains_person: bool, payload=None, *, source: str | None = None, m16_enabled: bool = False,
                  m17_enabled: bool = False, photo_consent: bool = False) -> None:
    if kind in ("audio", "raw_audio", "voice_wav"):
        raise PrivacyError("audio is never written to disk")
    if kind in ("nir_frame", "eye_frame", "rgb_frame") and contains_person:
        if payload is None or not crop_ok(payload):
            raise PrivacyError("full frames containing a person may not be saved (64x64 crops only)")
    if kind == "face_template":
        if source != FACE_TEMPLATE_SOURCE or not m16_enabled:
            raise PrivacyError("no face templates (only the opted-in M-16 module may store enrolled templates)")
        if payload is not None:
            raise PrivacyError("M-16 stores templates sealed by its own core; no raw template payloads here")
    if kind == "voice_template":
        if source != VOICE_TEMPLATE_SOURCE or not m17_enabled:
            raise PrivacyError("no voice templates (only the opted-in M-17 module may store enrolled voice templates)")
        if payload is not None:
            raise PrivacyError("M-17 stores voice templates sealed by its own core; no raw template payloads here")
    if kind == "face_photo":
        enabled = {"M-16": m16_enabled, "M-17": m17_enabled}.get(source or "", False)
        if source not in FACE_PHOTO_SOURCES or not enabled:
            raise PrivacyError("no face photos (only the opted-in M-16/M-17 modules, for enrolled people)")
        if not photo_consent:
            raise PrivacyError("a face photo needs the person's recorded photo consent")
    if kind in ("face_embedding", "face_id", "face_hash", "face_crop"):
        raise PrivacyError("no stranger face data of any kind (research mode is counts only)")
    if kind == "mac" :
        raise PrivacyError("store hashed identifiers only")


def round_gps(lat: float | None, lon: float | None, decimals: int = 2):
    if lat is None or lon is None:
        return None, None
    return round(lat, decimals), round(lon, decimals)


CROP_MAX = 64


def crop_ok(arr) -> bool:
    """Each spatial dimension (and so the area) must be within CROP_MAX (Forge F4: no tuple compare)."""
    shape = getattr(arr, "shape", None)
    if shape is None or len(shape) < 2:
        return False
    h, w = int(shape[0]), int(shape[1])
    return 0 <= h <= CROP_MAX and 0 <= w <= CROP_MAX and h * w <= CROP_MAX * CROP_MAX
