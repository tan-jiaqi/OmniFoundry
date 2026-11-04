You choose which question types suit a video, given its captions. A good choice is one where the captions contain concrete evidence in both audio and visuals that can be linked.

Question types:
{catalog}

Prior preference weights (higher is preferred when the evidence is comparable; 0 means never): {weights}

Captions:
{captions}

Pick up to {max_choices} types, best first. Do not pick a music type unless there are specific music or sound-effect audio entries. Return an empty list if the video has no usable audio-visual link.

Return JSON only:
{{"task_ids": ["..."], "reason": "..."}}
