Answer a multiple-choice question about a video using only its captions. First check whether everything the question takes for granted is actually supported by the captions, then derive the answer.

Captions:
{captions}

{mcq}

Return JSON only:
{{"all_presuppositions_supported": true, "unsupported": ["anything the question assumes that the captions do not support"], "answer": "A", "reason": "short derivation"}}
