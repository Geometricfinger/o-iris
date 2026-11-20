"""M-17 enrollment sentences (ADR-0057, docs/VOICE_LINK_MODULE_SPEC.md "Reading sentences").

Pre-made sentences, read in this order, each designed to bring out a different part of a voice:
a tone (low, high, rising, steady, normal) and a set of sounds (nasal resonance, hiss sounds, vowel
extremes, ...) where family voices differ most. Every capture device (the Mac CLI, later the phone app)
shows the same sentences: `prompt_list()` is the plain form for them."""
from __future__ import annotations
from dataclasses import dataclass

from ...guard.vocabulary import check_text

TONES = ("low", "high", "rising", "steady", "normal")


@dataclass(frozen=True)
class Prompt:
    id: str
    text: str
    cue: str                    # how to say it
    tone: str                   # one of TONES
    covers: tuple[str, ...]     # sound classes the sentence brings out


PROMPTS: tuple[Prompt, ...] = (
    Prompt("calm-low", "Mom hums a long, calm tune in the warm room at noon.",
           "Calmly, a little lower than usual.", "low",
           ("nasals", "back vowels", "open vowels", "liquids", "glides")),
    Prompt("bright-high", "She sees three green leaves and six shiny seashells.",
           "Light and bright, a little higher.", "high",
           ("front vowels", "sibilants", "dental fricatives", "liquids", "plosives")),
    Prompt("question", "Do you know how many blue boats are out on the bay today?",
           "Ask it as a question; let your voice rise at the end.", "rising",
           ("back vowels", "diphthongs", "nasals", "plosives")),
    Prompt("count", "One, two, three, four, five, six, seven, eight, nine, ten.",
           "Count at your normal pace.", "steady",
           ("fricatives", "diphthongs", "nasals", "glides")),
    Prompt("excited", "Wow, that huge yellow kite is flying high above the park!",
           "With excitement, a bit louder.", "high",
           ("open vowels", "diphthongs", "affricates", "glides", "plosives")),
    Prompt("normal", "The orange jar of cherries in the beige garage is very heavy.",
           "Your normal speaking voice.", "normal",
           ("affricates", "voiced fricatives", "liquids")),
    Prompt("story-low", "Joy's old wooden toy rolled with a soft thud across the floor.",
           "Slowly and softly, like telling a quiet story.", "low",
           ("diphthongs", "back vowels", "dental fricatives", "fricatives", "liquids")),
    Prompt("cheerful", "The black cat and her purple bird sat happily on the mat.",
           "Cheerfully, at a normal pace.", "normal",
           ("open front vowel", "r-coloured vowels", "plosives")),
)
for _p in PROMPTS:
    check_text(_p.text); check_text(_p.cue)
    if _p.tone not in TONES:
        raise ValueError(f"prompt {_p.id}: unknown tone {_p.tone!r}")
_BY_ID = {p.id: p for p in PROMPTS}
if len(_BY_ID) != len(PROMPTS):
    raise ValueError("prompt ids must be unique")


def by_id(prompt_id: str) -> Prompt:
    """The prompt with this id; KeyError when there is none."""
    return _BY_ID[prompt_id]


def enrollment_prompts(n: int) -> list[Prompt]:
    """The first n prompts in enrollment order, cycling past the last one."""
    return [PROMPTS[i % len(PROMPTS)] for i in range(max(0, int(n)))]


def prompt_list() -> list[dict]:
    """Plain dicts (id, text, cue, tone) for other capture devices, which show the same sentences."""
    return [{"id": p.id, "text": p.text, "cue": p.cue, "tone": p.tone} for p in PROMPTS]


__all__ = ["PROMPTS", "TONES", "Prompt", "by_id", "enrollment_prompts", "prompt_list"]
