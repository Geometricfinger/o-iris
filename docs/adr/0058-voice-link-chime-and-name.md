# ADR-0058: Voice link announces a recognised person: a chime and their name

- Status: Accepted (owner decision, 2026-11-16: "Chime + name"). Amended 2026-11-16, before it was built, after a review of the glasses-first plan:
  - it speaks "sounds like NAME", not the bare name, because a voice match is a hint, not proof;
  - the wearer or owner can be marked as "self" and is never announced, because on glasses the wearer's own voice is heard all the time;
  - a configured anti-spoof model can veto a match, because a recording or phone call of an enrolled person can match.
- Date: 2026-11-16
- Amends: ADR-0057 decision 5 ("no sound, no voice output, no push notification and nothing on the glasses or phone"), for a clear voice-link match only. Also makes one narrow exception to M-06's rule that no free text is ever spoken (`oiris/guard/phrases.py`).
- Source:
  - The owner asked whether the voice link gives "feedback so you can hear", then chose a chime plus the spoken name over screen-only.
  - The owner's stated target platform (2026-11-16): open-platform Android smart glasses first, then possibly a helmet with smart glasses, sensors and cameras. The Mac is the test bench for what can be built and tested now.

## Context
- ADR-0057 decision 5 shows a voice-link match only as a "heard: NAME" card on the local dashboard and a review-log entry. A match makes no sound.
- The owner wants to hear who was recognised.
- On the intended platform, smart glasses, the natural place for that is the glasses' own speaker, which mainly the wearer hears. On the Mac test bench it is the Mac's speaker, which everyone in the room hears.
- M-06 voice output speaks only whitelisted phrase templates with whitelisted slot values, at most 8 words. A person's enrolled name is a label the owner typed, so it is not on any whitelist.

## Decision
1. **What triggers it.** A clear match of an enrolled person plays a short chime, then speaks "sounds like" and that person's name:
   - the best enrolled voice is at or above the threshold;
   - it leads the next one by the margin.

   Unsure results (a close call between two enrolled people) and unknown voices make no sound at all, exactly as they leave no log entry. So do the "self" person (decision 8) and a match the anti-spoof check drops (decision 9).
2. **Where it plays.** Only through the audio output of the device O Iris runs on:
   - the Mac's speaker on the test bench today;
   - the glasses' own speaker on the intended platform, sent through M-06 voice output as an attention-tier item, with its queueing and barge-in rules.

   Nothing is sent over the network. There are no push notifications, and no other device is told.
3. **What is spoken.** "sounds like" followed by the person's enrolled display name, exactly as stored (for example "sounds like Alex"):
   - A voice match is a hint, not proof: a phone call, a video or a recording of an enrolled person can match. M-16 words its results as "possible match" for the same reason. The dashboard card says "Sounds like NAME".
   - The name was checked by the vocabulary guard at enrollment and is checked again just before speaking.
   - It is limited to at most 4 words and 40 characters of letters (any one script), spaces, hyphens and apostrophes. A name outside these limits is not spoken; the chime still plays.
   - The whole phrase is at most 6 words, within M-06's 8-word budget.

   This is the one exception to "no free text is ever spoken". Every other M-06 rule and template is unchanged.
4. **No repeats during a conversation.** The same person is announced at most once per `voice_link.announce_repeat_s`, default 60 s. Different people can be announced one after another.
5. **No feedback loop.** While the chime and name play, plus a short tail, the listener ignores microphone input. O Iris never hears and matches its own announcement.
6. **Setting.** `voice_link.announce` takes one of three values:
   - `chime_and_name`: the default, as the owner chose;
   - `chime`;
   - `off`.

   The M-17 opt-in notice says that "sounds like" and the name are spoken aloud, that anyone near the speaker can hear them, and that a recording or phone call of an enrolled person can trigger it.
7. **Engines.** No new dependency and no download:
   - On macOS: the built-in `say` command and a system sound played with `afplay`.
   - On other hosts and the glasses: the existing `tts` port (Flite, BSD-style) and a generated chime. With no speech engine available, the chime alone plays.
   - GPL speech engines (Piper, eSpeak NG) stay excluded.
8. **Self.** One enrolled person can be marked as "self": the wearer on glasses, the owner at the Mac (`oiris voice-link set-self --person ID`, `clear-self`). Self is never announced, never written as a heard entry, gets no card and is not counted. Their speech is simply ignored, like an unknown voice. Disabling voice link and deleting that person's voice remove the mark.
9. **Anti-spoof hook.** When a `voice_antispoof` slot is configured (ADR-0055) and its score for the utterance is at or above `voice_link.spoof_threshold` (default 0.5), the match is dropped: no announcement and no heard entry. No slot means the match goes ahead. No anti-spoof model ships yet (licence terms, STATUS.md), so today a recording of an enrolled voice can still be announced; the opt-in notice says so.

## Where it will be enforced (when built)
- `oiris/modules/voice_link/announce.py` (new): the chime, the name rules, the repeat limit and the setting.
- The live listener's `on_heard` hook calls it and pauses input while it plays.
- `oiris/config.py`: `announce`, `announce_repeat_s`, `spoof_threshold`.
- `oiris/modules/voice_link/core.py`: the self mark (in `vl_state`) and the anti-spoof check, before anything is logged.
- The M-17 opt-in notice: the spoken-name sentence.
- Tests:
  - the name rules (vocabulary, length, characters);
  - no sound for unsure or unknown voices;
  - the repeat limit;
  - the input pause;
  - each setting;
  - self and anti-spoof drops leave no row, card, count or sound.

## Consequences
- On the Mac test bench, anyone near the speaker hears who was recognised. That is the point of the feature, and the opt-in notice says so.
- On glasses, mainly the wearer hears it, which is more private than the dashboard card on a shared screen.
- ADR-0057's biometric-law notes are unchanged: voiceprints, consent per person and guardian consent for minors.

- Until an anti-spoof model ships, a loudspeaker playing an enrolled person's voice can be announced. The spec's live check (step 6) records whether it is.

**Later, on the glasses (needs the hardware; decided when the glasses are tested):**
- The wearer's own voice will be heard all the time. The wearer is marked as "self" (decision 8), and glasses microphones that favour the wearer's voice need their own measurement.
- An outward-facing indicator while the microphone is on, so people nearby can see that it listens.
- The legal review that ADR-0014 requires before any public use: ambient listening and recording laws, and all-party consent states (spec N-04, R-4).

## Changing it
Write a new ADR that supersedes this one, update CHANGELOG.md, and get the owner's sign-off. This decision sets what O Iris says aloud about people.
