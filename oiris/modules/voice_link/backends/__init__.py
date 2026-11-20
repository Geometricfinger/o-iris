"""M-17 voice link speaker-model backends (ADR-0057, docs/VOICE_LINK_MODULE_SPEC.md).

Each backend fills the `speaker_embed` slot (oiris/slots.py). Backends are registered lazily
(oiris/registry.py BUILTIN_SLOT_PLUGINS), so nothing here, and no model runtime, is imported
until the slot is selected. No model weights live in this repo."""
