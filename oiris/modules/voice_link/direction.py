"""Where a recognised voice came from (ADR-0065): one direction word for a matched utterance, or None.

Used only after a clear match of an enrolled person (listener.py); never for unknown voices, close calls,
self or a spoof-vetoed match. The word is spoken once with the name and shown on the live card; it is never
stored, logged or tracked over time (ADR-0060 decision 9, narrow exception in ADR-0065).

Pure and deterministic. The utterance is cut into WINDOW_S windows; only windows the listener's voice
activity gate marked as speech throughout are used. Each runs through oiris.features.doa.doa():
  * a result below doa.MIN_CONFIDENCE (or none) is no vote;
  * an ambiguous result (a temple pair cannot tell front from back) votes only when every candidate maps
    to the same word (60 deg: candidates 60 and 120, both "at left"); otherwise it is a vote for no word
    (30 deg: "ahead" or "behind");
  * elevation words ("above", "below") come only from a non-coplanar array, as doa decides.
The answer needs at least MIN_WINDOWS confident windows, and one word must hold at least MAJORITY_SHARE of
them. Otherwise None: never a guess.

Not modelled (ADR-0065 "Later"): real temple-mic geometry and head shadow, reverberation, several talkers,
head turns between speech and announcement, per-device mic phase mismatch."""
from __future__ import annotations
from collections import Counter

import numpy as np

from ...features.doa import MIN_CONFIDENCE, SILENCE_RMS, direction_of_bearing, doa

WINDOW_S = 0.1                # one direction estimate per window of speech
MIN_WINDOWS = 3               # fewer confident windows than this: no direction
MAJORITY_SHARE = 0.6          # the winning word needs this share of the confident windows


class VoiceDirection:
    """mic_positions_head_m: (M, 3) microphone positions in the head frame (x forward, y left, z up), in the
    order of the capture's channels. sample_rate: of the channels given to estimate()."""

    def __init__(self, mic_positions_head_m, sample_rate: int):
        pos = np.asarray(mic_positions_head_m, dtype=float).reshape(-1, 3)
        if pos.shape[0] < 2:
            raise ValueError("a voice direction needs at least two microphones")
        self.mic_positions_head_m = pos
        self.sample_rate = int(sample_rate)
        self.channels = pos.shape[0]
        self.window = max(1, int(round(WINDOW_S * self.sample_rate)))

    def window_word(self, channels: np.ndarray) -> tuple[bool, str | None]:
        """(confident, word) for one window. confident False: no vote. word None: a vote for no word."""
        r = doa(channels, self.mic_positions_head_m, self.sample_rate)
        if r is None or r.confidence < MIN_CONFIDENCE:
            return False, None
        cands = r.candidates if r.ambiguous else [r.az_deg]
        words = {direction_of_bearing(a, r.el_deg) for a in cands}
        return True, (words.pop() if len(words) == 1 else None)

    def estimate(self, channels: np.ndarray, speech_mask: np.ndarray) -> str | None:
        """channels (M, N); speech_mask (N,) True where the voice activity gate marked speech.
        Returns a word of oiris.features.doa.DIRECTION_WORDS, or None."""
        x = np.asarray(channels, dtype=float)
        mask = np.asarray(speech_mask, dtype=bool).reshape(-1)
        if x.ndim != 2 or x.shape[0] != self.channels or mask.size != x.shape[1]:
            raise ValueError(f"expected ({self.channels}, N) channels and an (N,) speech mask")
        votes: Counter = Counter()
        confident = 0
        w = self.window
        for start in range(0, x.shape[1] - w + 1, w):
            if not mask[start:start + w].all():
                continue
            seg = x[:, start:start + w]
            if float(np.std(seg[0])) < SILENCE_RMS:
                continue
            ok, word = self.window_word(seg)
            if not ok:
                continue
            confident += 1
            if word is not None:
                votes[word] += 1
        if confident < MIN_WINDOWS or not votes:
            return None
        word, n = votes.most_common(1)[0]
        return word if n >= MAJORITY_SHARE * confident else None


__all__ = ["VoiceDirection", "WINDOW_S", "MIN_WINDOWS", "MAJORITY_SHARE"]
