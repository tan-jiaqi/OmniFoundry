You check one claim about a video against {media_kind}. The claim came from a caption that may be wrong, so judge only from what the media shows.

The clip covers {clip_range} of the video. The claim concerns {clue_range}, about {offset} seconds into the clip. Clue type: {modality}.
Claim: {claim}

Decide whether the media supports the claim, and list any part of it that the media does not support. Small wording differences do not matter. Wrong people, objects, counts, words, kinds of sound or order do matter.

Judge only what {media_kind} can carry. For audio, check whether it is speech or another kind of sound, the words spoken, its character and its timing. Do not count it against the claim when the claim names a visible source the audio cannot show, such as which object made a sound. For a muted video, judge only what is visible.

support_strength scale:
- 1.0: every part of the claim is clearly seen or heard.
- 0.8: the claim is clearly supported, with only minor details hard to confirm.
- 0.5: partly supported, or too ambiguous to confirm.
- 0.2: mostly not observable.
- 0.0: contradicted.
Set pass to true only when the strength is 0.8 or higher and nothing is contradicted.

Return JSON only:
{{"pass": true, "support_strength": 1.0, "unsupported_claims": [], "observed": "what you actually saw or heard"}}
