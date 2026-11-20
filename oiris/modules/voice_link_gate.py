"""The only door to M-17 (voice link, ADR-0057). Importing this file does NOT import
`oiris.modules.voice_link`; the package is imported only by `load()` when the M-17 flag is on.
The speaker model is a `speaker_embed` slot plugin (registered by its own package), resolved through
SlotManager so a failing model only makes the voice link unavailable."""
from __future__ import annotations
import importlib
from typing import Any


def load(cfg, db, keys, registry, face_module, chain=None, speaker: Any = None):
    """Build the M-17 module, or return None while it is off. Refuses without the M-17 opt-in, and
    needs a loaded M-16 module in personal mode (voice templates exist only for face-enrolled people).
    `speaker`: a SlotManager to use instead of the one built from config (tests, CLI injection).
    `keys` is accepted for symmetry with the M-16 gate; templates use M-16's per-person keys."""
    if not cfg.enabled("M-17"):
        return None
    core = importlib.import_module("oiris.modules.voice_link.core")
    vcfg = core.VoiceLinkConfig.from_dict(cfg.voice_link)
    if face_module is None or not cfg.enabled("M-16"):
        raise core.VoiceLinkLocked("voice link needs face matching (M-16) on and loaded: oiris face-match enable")
    if "personal" not in getattr(face_module, "modes", set()):
        raise core.VoiceLinkLocked("voice link needs face matching in personal mode")
    if speaker is None:
        speaker = core.speaker_slots(cfg, vcfg, registry)
    return core.VoiceLinkModule(db, face_module, speaker, cfg=vcfg, chain=chain)
