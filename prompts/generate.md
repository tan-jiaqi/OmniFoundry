You write one hard multiple-choice question about a video, based on its timestamped captions. Answering it must require combining what is heard with what is seen: someone with only the audio, only the visuals, or only the question must not be able to answer.

Question type: {task_name} (id: {task_id}, kind: {task_kind})
Definition: {task_definition}
Type rules:
{task_rules}

Captions (each entry has a [ref]):
{captions}

Feedback on earlier failed attempts for this video:
{feedback}

Requirements:
1. Exactly {num_options} options. All are plausible and similar in length and style. Include distractors that are right for only one modality.
2. The answer must follow from the captions, not from world knowledge. Do not quote speech in the question in a way that gives away the answer, and do not describe the answer's visual features in the question.
3. Build a reasoning chain of at least 2 steps including at least one audio step and one visual step. Each step depends only on earlier steps, every step feeds the conclusion, and the last step reaches the answer.
4. Every clue cites one caption entry through caption_ref. Its quote is copied from that entry, its timestamp is that entry's timestamp, and its modality is speech, sound, music or visual, matching the entry.
5. Evidence discipline:
   - Do not anchor on background music or ambience unless the type is about music.
   - Do not invent a link between things that the captions never connect: two clues about "the same" person or object must share an identifier the captions establish.
   - Do not assume a state persists across a gap in the captions.
   - Do not merge details from different referents into one.
   - If the question says "when" or "while", its audio and visual clues must really be simultaneous.
6. List the presuppositions the question relies on (things it takes as given); each must be supported by a caption.

Return JSON only:
{{"question": "...", "options": ["A. ...", "B. ..."], "answer": "A",
 "clues": [{{"id": "c1", "modality": "speech", "timestamp": "MM:SS-MM:SS", "quote": "...", "fact": "...", "caption_ref": "e01.a2"}}],
 "reasoning_steps": [{{"step_id": "s1", "modality": "audio", "timestamp": "MM:SS-MM:SS", "content": "...", "depends_on": [], "clue_ids": ["c1"]}}],
 "single_modality_traps": ["..."], "presuppositions": ["..."]}}
