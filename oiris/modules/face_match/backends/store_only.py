"""Store-only backend: for hosts that hold M-16 enrollments but do no face matching (yet).

O Iris.app runs the M-17 voice link, which needs M-16 loaded for each person's keys (their DEK),
but live mode does no face matching. This backend lets the module open without any model and
without OpenCV: `detect()` never finds a face, so `embed()` and `compare()` are never reached.
Its own `model_id` means templates made by a real model are never loaded with it (the core keeps a
template only for the model that made it). Selected with `face_match.adapter = "store_only"`.

It cannot enroll anyone: enrollment needs a real backend (`opencv_zoo`); the gate refuses
enrollment with ENROLL_REFUSAL when this backend is loaded (`face_match_gate.load`)."""
from __future__ import annotations
from typing import Any, Sequence

from ....schema import Health
from ..port import Embedding, FaceBox, MatchResult, Template

ENROLL_REFUSAL = ("face enrollment needs a real face backend; this config uses face_match.adapter "
                  "\"store_only\", which keeps enrollments but cannot see faces. Enroll with the default "
                  "adapter (opencv_zoo) and the same data_dir and passphrase")


class StoreOnlyBackend:
    name = "store_only"
    license = "Apache-2.0"
    model_id = "store-only"
    model_version = "1"
    test_only = False
    enrolls = False   # read by face_match_gate.load: enrollment is refused with ENROLL_REFUSAL

    def __init__(self) -> None:
        self._health = (Health.FAILED, "not opened")

    def open(self, cfg: dict) -> None:   # no models, no OpenCV
        self._health = (Health.OK, "store only: no face matching")

    def close(self) -> None:
        self._health = (Health.FAILED, "closed")

    def health(self) -> tuple[Health, str]:
        return self._health

    def detect(self, frame: Any, timeout_s: float) -> list[FaceBox]:
        return []

    def embed(self, frame: Any, box: FaceBox) -> Embedding:
        raise RuntimeError("store_only finds no faces, so it never embeds one")

    def compare(self, e: Embedding, templates: Sequence[Template]) -> MatchResult | None:
        return None
