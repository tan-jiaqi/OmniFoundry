You are watching a whole video to split it into events before detailed captioning. The upper-left corner shows the current time; use it for timestamps and never mention it.

Return:
1. a short factual summary of the video;
2. a specific video type that names the domain or topic (for example "first-person cooking tutorial", not "tutorial");
3. event segments split at meaningful changes of content, not at fixed intervals.

Rules:
- Video length is {duration}. Timestamps are MM:SS on the video's own clock.
- Each segment is at most {max_event_s} seconds; split longer ones at natural boundaries.
- Use at most {max_events} segments and cover the whole video.
- Each summary mentions the main things seen and heard.

Return JSON only:
{{"video_summary": "...", "video_type": "...", "event_segments": [{{"event_id": "e01", "start": "MM:SS", "end": "MM:SS", "summary": "..."}}]}}
