You are recording what can be seen and heard in one event of a longer video, as fine-grained facts for later retrieval. Report observations; do not summarize or evaluate.

Event range on the video clock: {event_range}
The upper-left corner of the frame shows the video clock. Read timestamps from it, and never mention it in a caption.
Plan note for this event: {plan_summary}
Earlier events, only for resolving who and what is being referred to:
{prior_context}

Visual captions (visual facts only):
- Split whenever the shot, subject, action, object, on-screen text or layout changes.
- Give exact counts. Describe people by clothing, colors, hair, and position in frame.
- Name precise colors and specific object types. Note which hand or side, the direction of motion, and repetition counts.
- Note left/right and foreground/background arrangement, and the type of shot transition.
- Copy on-screen text exactly, in its original language, including numbers and formulas.

Audio captions (audio facts only):
- Split whenever the speaker, speech, music, sound or silence changes.
- audio_type is one of: speech, music, sound, background, silence, unclear.
- Speech: write the words actually audible, in the original language, and identify the speaker as specifically as the evidence allows, with tone of voice. No speech item may exceed 10 seconds; split long speech into phrases of 3 to 6 seconds.
- Music: describe genre, tempo, mood and distinguishable instruments.
- Sound: name each effect and its texture, and its source if identifiable.
- Report silent intervals.
- Only report what you can actually hear. Never infer speech from moving lips or music from visible instruments. Mark uncertain words as [unclear].

Also write global_caption: one integrated account of the event, in temporal order, that relates sound and picture.

Reuse the same identifier for a person or object that appeared earlier when the evidence supports it. Use video-clock timestamps throughout.

Return JSON only:
{{"global_timestamp": "MM:SS-MM:SS", "global_caption": "...", "audio_captions": [{{"timestamp": "MM:SS-MM:SS", "audio_type": "speech", "caption": "..."}}], "visual_captions": [{{"timestamp": "MM:SS-MM:SS", "caption": "..."}}]}}
