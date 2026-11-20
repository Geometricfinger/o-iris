# ADR-0065: Voice link says where a recognised voice came from

- Status: Accepted (owner decision, 2026-11-17)
- Date: 2026-11-17
- Amends: ADR-0058 decision 3 ("What is spoken").
- Narrow exception to: ADR-0060 decision 9 ("No tracking of people"), for the case below only.
- Source: the owner was asked, "Should a recognised voice also say its direction ('sounds like NAME, behind you')?" The owner answered, verbatim:

  > Yes add direction and name of it's available

  Read as: yes; say the name, and add the direction whenever it is available. ADR-0058 requires the owner's sign-off for anything said aloud about people; this is that sign-off.

## Context
- ADR-0058 says "sounds like NAME" after a clear match of an enrolled person.
- ADR-0060 added a shared head frame, microphone-array mounts in the rig config and sound direction of arrival (`oiris/features/doa.py`). Decision 6 ends: "Later a voice-link match can also say where it was heard."
- ADR-0060 decision 9 says spatial outputs describe signatures and their direction, never a person's identity or path. Saying where a named person is speaking from crosses that line, so it needs its own decision.
- The Mac test bench has one microphone. A direction needs a microphone array declared in the rig: a temple pair on glasses, or more microphones on a helmet.

## Decision
1. **When.** Only after a clear match of an enrolled, consenting person: the same moment ADR-0058 announces the name.
   - Never for unknown voices, close calls, the self person, a match the anti-spoof check drops, or while input is paused for an announcement.
   - The direction is computed at that moment, from that utterance's audio, and from nothing else.
2. **Only when the microphones can tell.** `oiris/modules/voice_link/direction.py` cuts the utterance into 100 ms windows and runs `doa()` on each window the listener's voice gate marked as speech.
   - A window counts only at or above doa's confidence floor (`MIN_CONFIDENCE`).
   - A front/back-ambiguous result (a temple pair) counts only when every candidate gives the same word: 60 degrees (candidates 60 and 120) is "at left"; 30 degrees ("ahead" or "behind") is no word.
   - "above" and "below" only come from a non-coplanar array, as doa decides.
   - The word needs at least 3 confident windows (`MIN_WINDOWS`) and at least 0.6 of them (`MAJORITY_SHARE`). Otherwise there is no direction. Never a guess.
3. **What is spoken** (amends ADR-0058 decision 3). "sounds like NAME, PHRASE", with PHRASE one of:

   | Direction word | Spoken |
   |---|---|
   | ahead | ahead of you |
   | at left | on your left |
   | at right | on your right |
   | behind | behind you |
   | above | above you |
   | below | below you |

   - The name always wins. When name and direction together would pass M-06's 8-word budget (a 4-word name with "on your left"), only "sounds like NAME" is spoken.
   - The vocabulary guard (`check_text`) checks the full text before speaking. A refusal falls back to the name-only phrase, which is checked too.
   - No direction means the announcement is exactly as before. A direction is never spoken without the name: a name that cannot be spoken gives the chime alone, as before.
   - `announce: chime` and `announce: off` are unchanged. The repeat limit stays per person: a new direction within `announce_repeat_s` does not re-announce.
4. **Shown, never kept.** The direction is on the live card in RAM ("Sounds like NAME, on your left" on the dashboard) for `card_seconds`, then gone with the card.
   - It is never stored, logged or written: not in `vl_heard`, the change log, the sweep record or any file.
   - It is not tracked over time: each announcement's direction comes from its own utterance only, and no history is kept.
5. **Narrow exception to ADR-0060 decision 9.** Only for the M-17 announcement of an enrolled, consenting person after a clear match, spoken once with the name and shown on the live card. For everyone else decision 9 stands unchanged, and unknown voices never get a direction.
6. **Capture.** In live mode, a `mic_array` mount in the rig config opens the voice microphone with that many channels, in the mount's `mic_positions_m` order. The positions go to the head frame through the mount's position and orientation (`oiris/spatial/rig.py`).
   - Matching (VAD, embedding, match) uses channel 0 only. Averaging delayed channels would smear the voice.
   - The multi-channel audio is a RAM-only buffer next to the mono one, zeroed after every utterance, matched or not.
   - Without a `mic_array` mount (the Mac today) the capture is mono and everything is exactly as before.
   - A device that cannot open the channels falls back to mono, with a plain log line and no direction. The listener never fails because of it.

## Built now vs later
- **Built now, tested on synthetic signals only:** the estimate, multi-channel capture, the listener wiring, the spoken phrase, the dashboard card, mono fallback, and tests that the direction is never computed for anyone but a clear match and never reaches the store.
- **Later, needing hardware:**
  - real temple-mic geometry, head shadow and diffraction around a real head, reverberant rooms, several talkers at once;
  - head turns between the speech and the announcement (IMU compensation);
  - per-device microphone gain and phase mismatch.

  Until those are measured, a spoken direction is only as good as the synthetic tests show.

## Where it is enforced
- `oiris/modules/voice_link/direction.py`: the estimate.
- `oiris/modules/voice_link/capture.py`: N-channel blocks, mono fallback.
- `oiris/modules/voice_link/listener.py`: reference channel, multi-channel buffer lifecycle, direction only after a clear match.
- `oiris/modules/voice_link/announce.py`: `DIRECTION_PHRASES`, the word budget, the guard fallback.
- `oiris/live.py`: the rig's `mic_array` mount and the capture factory.
- `oiris/dashboard/server.py`: the card.
- Tests: `tests/test_voice_link_direction.py`.

## Consequences
- With a microphone array, people near the speaker hear where an enrolled person is speaking from, as well as their name. The M-17 opt-in already says names are spoken aloud.
- On the Mac test bench nothing changes until a `mic_array` mount is configured.

## Changing it
Write a new ADR that supersedes this one, update CHANGELOG.md, and get the owner's sign-off. This decision sets what O Iris says aloud about people.
