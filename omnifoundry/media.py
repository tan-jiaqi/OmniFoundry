"""ffprobe / ffmpeg wrappers: probe, timestamp-overlay transcode, clip cutting, synthetic test media.

Only the bundled static ffmpeg is used. Timestamps are burned in through a generated ASS subtitle
track (the static build has libass but no drawtext), one cue per second, upper-left, MM:SS.
"""
import json
from pathlib import Path
import subprocess

from .config import resolve_path
from .schema import fmt_time

DEFAULTS = {"ffmpeg": "tools/bin/ffmpeg", "ffprobe": "tools/bin/ffprobe"}


class MediaError(RuntimeError):
    pass


class Media:
    def __init__(self, options=None):
        options = {**DEFAULTS, **(options or {})}
        self.ffmpeg = str(resolve_path(options["ffmpeg"]))
        self.ffprobe = str(resolve_path(options["ffprobe"]))

    def _run(self, args, what):
        proc = subprocess.run(args, capture_output=True, text=True)
        if proc.returncode != 0:
            raise MediaError(f"{what} failed: {proc.stderr.strip()[-800:]}")
        return proc.stdout

    def probe(self, path):
        out = self._run([self.ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                        "ffprobe")
        info = json.loads(out)
        streams = info.get("streams", [])
        video = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
        duration = float(info.get("format", {}).get("duration") or (video or {}).get("duration") or 0)
        return {
            "duration_s": duration,
            "has_video": video is not None,
            "has_audio": audio is not None,
            "width": int(video["width"]) if video else None,
            "height": int(video["height"]) if video else None,
            "audio_codec": audio.get("codec_name") if audio else None,
        }

    def overlay_transcode(self, src, dst, *, fps=2, width=640, crf=30, audio_kbps=64):
        """Re-encode at low fps / width with a burned-in MM:SS timestamp; keeps audio (mono AAC)."""
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        info = self.probe(src)
        ass = dst.with_suffix(".ass")
        ass.write_text(_timestamp_ass(info["duration_s"]), encoding="utf-8")
        vf = f"fps={fps},scale={width}:-2,subtitles={_filter_path(ass)}"
        args = [self.ffmpeg, "-v", "error", "-y", "-i", str(src), "-vf", vf,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p"]
        args += ["-c:a", "aac", "-ac", "1", "-b:a", f"{audio_kbps}k"] if info["has_audio"] else ["-an"]
        self._run(args + ["-movflags", "+faststart", str(dst)], "overlay transcode")
        ass.unlink(missing_ok=True)
        return dst

    def cut(self, src, dst, start_s, end_s, *, mode="av", width=640, fps=None):
        """Cut [start_s, end_s] from src. mode: 'av' (video+audio), 'video' (muted), 'audio' (mp3 only)."""
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        start_s = max(0.0, float(start_s))
        dur = max(0.1, float(end_s) - start_s)
        args = [self.ffmpeg, "-v", "error", "-y", "-ss", f"{start_s:.3f}", "-t", f"{dur:.3f}", "-i", str(src)]
        if mode == "audio":
            args += ["-vn", "-ac", "1", "-c:a", "libmp3lame", "-b:a", "64k"]
        else:
            vf = f"scale={width}:-2" + (f",fps={fps}" if fps else "")
            args += ["-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p"]
            args += ["-an"] if mode == "video" else ["-c:a", "aac", "-ac", "1", "-b:a", "64k"]
        if mode not in ("av", "video", "audio"):
            raise ValueError(f"unknown cut mode {mode!r}")
        self._run(args + [str(dst)], f"cut {mode}")
        return dst

    def frames(self, src, out_dir, start_s, end_s, *, n=4, width=512):
        """n evenly spaced JPEG frames from [start_s, end_s]; for image-only model routes."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(n):
            t = start_s + (end_s - start_s) * (i + 0.5) / n
            p = out_dir / f"frame_{i:02d}.jpg"
            self._run([self.ffmpeg, "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(src), "-frames:v", "1",
                       "-vf", f"scale={width}:-2", "-q:v", "4", str(p)], "frame")
            paths.append((t, p))
        return paths

    def synth(self, dst, *, duration=20, segments=None):
        """Synthetic test video: colored segments + tones. segments: [(start, end, color, tone_hz or None)]."""
        segments = segments or [(0, 10, "red", 440), (10, 20, "blue", 1200)]
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        inputs, vlabels, alabels = [], [], []
        for i, (s, e, color, hz) in enumerate(segments):
            d = e - s
            inputs += ["-f", "lavfi", "-i", f"color=c={color}:s=320x240:r=10:d={d}"]
            src = f"sine=frequency={hz}:duration={d}" if hz else f"anullsrc=r=16000:cl=mono:d={d}"
            inputs += ["-f", "lavfi", "-t", str(d), "-i", src]
            vlabels.append(f"[{2 * i}:v]")
            alabels.append(f"[{2 * i + 1}:a]")
        n = len(segments)
        fc = "".join(v + a for v, a in zip(vlabels, alabels)) + f"concat=n={n}:v=1:a=1[v][a]"
        self._run([self.ffmpeg, "-v", "error", "-y", *inputs, "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", "1", str(dst)], "synth")
        return dst


def _filter_path(path):
    # Escape for ffmpeg filter-graph syntax.
    return str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'").replace(",", "\\,")


def _ass_time(t):
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _timestamp_ass(duration):
    head = ("[Script Info]\nScriptType: v4.00+\nPlayResX: 640\nPlayResY: 360\n\n[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV\n"
            "Style: TS,DejaVu Sans Mono,20,&H00FFFFFF,&H00000000,&H99000000,3,2,0,7,6,6,6\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Text\n")
    lines = [f"Dialogue: 0,{_ass_time(t)},{_ass_time(t + 1)},TS,{fmt_time(t)}" for t in range(int(duration) + 1)]
    return head + "\n".join(lines) + "\n"
