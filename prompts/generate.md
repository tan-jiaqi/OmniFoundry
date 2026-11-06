You write one hard multiple-choice question about a video, based on its timestamped captions. Answering it must require combining what is heard with what is seen: someone with only the audio, only the visuals, or only the question must not be able to answer.

Question type: {task_name} (id: {task_id}, kind: {task_kind})
Definition: {task_definition}
Type rules:
{task_rules}
{bridge}

Captions (each entry has a [ref]):
{captions}

Build the question around THIS anchor. The anchor is the moment the question points to; the answer must be something else, observed in the other modality:
{anchor}

Feedback on earlier failed attempts for this video:
{feedback}

If this anchor cannot carry a question of this type that satisfies every requirement below, reply with exactly `[Unavailable]` and a one-line reason, and write nothing else. Do not force a weak question: a skipped anchor costs nothing, a bad question costs a lot.

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
7. Make each modality indispensable. Before writing, test the question against two people:
   - A viewer with sound muted, who sees the whole video. If only one thing in the video could fit the question (one toothbrush, one person walking away, one card on top), they can answer without the audio. An anchor question therefore needs at least two visually similar candidates in the video (several players, several cards, several letters), and only the audio cue says which one is meant. Draw the distractors from the attributes of those other candidates.
   - A listener with no picture. If the speech or sound already states the answer (a spoken name, color, number or description), they can answer without the visuals. The answer must be a visual fact that the audio never states.
   Name in single_modality_traps what each of them would pick, and why it is wrong.
8. For ordering questions, the correct order must not be the order in which the events are listed in the question, and the listed order must not follow the order of the video.
9. The options must not give the answer away by their form. All options are about the same length (the correct one must not be the longest), equally specific and detailed, and written in the same style. Every distractor must be plausible to someone who has not watched the video, yet clearly ruled out by what the video shows and says; exactly one option can be correct.
10. When the question points to a moment through what is heard, describe that cue only by its sound: who speaks or sings, how it sounds, or what is said in words that are NOT shown anywhere on screen. Before finalizing, read the visual captions around that time. If a word, letter, name or number in your cue also appears on screen there (on-screen text, a displayed letter, a name caption, a title card), a muted viewer can find the moment by looking: rewrite the cue or choose another anchor. Example of a leak: "when the singer sings that a letter makes the sound 'nn'" while a large letter N is on screen.

Return JSON only:
{{"question": "...", "options": ["A. ...", "B. ..."], "answer": "A",
 "clues": [{{"id": "c1", "modality": "speech", "timestamp": "MM:SS-MM:SS", "quote": "...", "fact": "...", "caption_ref": "e01.a2"}}],
 "reasoning_steps": [{{"step_id": "s1", "modality": "audio", "timestamp": "MM:SS-MM:SS", "content": "...", "depends_on": [], "clue_ids": ["c1"]}}],
 "single_modality_traps": ["..."], "presuppositions": ["..."]}}
