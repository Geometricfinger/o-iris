"""M-17 voice link (optional, off by default; ADR-0057, docs/VOICE_LINK_MODULE_SPEC.md).

An add-on to M-16 personal mode: people already enrolled in face matching can also get an encrypted
voice template, and live audio is matched against enrolled voices only. Unknown voices are dropped at
once, never stored or counted, and no audio is ever written.

Nothing in core imports this package. It is loaded only by `oiris.modules.voice_link_gate.load()` once
the `M-17` flag is on and the typed opt-in is recorded. The speaker model is a `speaker_embed` slot
plugin (oiris/slots.py); the module core here owns storage, consent, review and wipe."""
PORT = "speaker_embed"
