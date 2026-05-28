# ── VisionRAG Phase 4: Ollama Cloud Video ─────────────────────────────────────
# File: backend/app/services/frame_extractor.py
# Changes: New file — ffmpeg frame extraction (CPU only, no GPU)
# Image pipeline: UNTOUCHED
# ─────────────────────────────────────────────────────────────────────────────

"""
Video frame extraction using ffmpeg scene detection. CPU only — no GPU used.
Independent of image pipeline.

Pipeline:
  1. Scene detection → timestamps where visuals change
  2. If fewer than MIN_FRAMES found → add uniform samples to reach minimum
  3. Cap at MAX_FRAMES → evenly subsample if exceeded
  4. Extract one JPEG per timestamp at FRAME_WIDTH resolution

Env vars:
  VIDEO_SCENE_THRESHOLD  float  default 0.25
  VIDEO_MAX_FRAMES       int    default 45
  VIDEO_MIN_FRAMES       int    default 15
  VIDEO_FRAME_WIDTH      int    default 640
"""

import os
import re
import json
import logging
import subprocess
from app.services.figure_extractor import _FIGURES_DIR, _FIGURES_SERVE_URL

logger = logging.getLogger(__name__)

_SCENE_THRESHOLD = float(os.getenv("VIDEO_SCENE_THRESHOLD", "0.25"))
_MAX_FRAMES      = int(os.getenv("VIDEO_MAX_FRAMES",       "45"))
_MIN_FRAMES      = int(os.getenv("VIDEO_MIN_FRAMES",       "15"))
_FRAME_WIDTH     = int(os.getenv("VIDEO_FRAME_WIDTH",      "640"))


def _ffmpeg_available() -> bool:
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


_FFMPEG_OK = _ffmpeg_available()


def get_video_duration(video_path: str) -> float:
    """
    Return video duration in seconds via ffprobe. Returns 0.0 on failure.
    PUBLIC — called by ingest.py upload route for the 3-minute check.
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", video_path],
            capture_output=True, text=True, timeout=30,
        )
        return float(json.loads(r.stdout).get("format", {}).get("duration", 0))
    except Exception as exc:
        logger.warning("ffprobe failed for %s: %s", video_path, exc)
        return 0.0


def _detect_scenes(video_path: str) -> list[float]:
    """Scene detection — find timestamps of visual changes."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-i", video_path,
             "-vf", f"select='gt(scene,{_SCENE_THRESHOLD})',showinfo",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=300,
        )
        ts = [0.0]
        for m in re.finditer(r"pts_time:([\d.]+)", r.stderr):
            t = float(m.group(1))
            if t > 0.0:
                ts.append(t)
        ts.sort()
        # Deduplicate timestamps within 0.5s of each other
        deduped = [ts[0]]
        for t in ts[1:]:
            if t - deduped[-1] >= 0.5:
                deduped.append(t)
        return deduped
    except Exception as exc:
        logger.warning("Scene detection failed: %s", exc)
        return [0.0]


def _fill_uniform(existing: list[float], duration: float, target: int) -> list[float]:
    """
    Add evenly-spaced timestamps between existing scene-detected ones
    until we reach 'target' total frames. Keeps all scene-detected frames
    and fills gaps with uniform samples.
    """
    if duration <= 0 or len(existing) >= target:
        return existing

    need = target - len(existing)
    interval = duration / (need + 1)
    candidates = []
    t = interval
    while t < duration and len(candidates) < need:
        # Only add if not too close to an existing timestamp
        if all(abs(t - ex) > 1.0 for ex in existing):
            candidates.append(t)
        t += interval

    merged = sorted(set(existing + candidates))
    return merged[:target]


def _extract_frame(video_path: str, ts: float, out_path: str) -> bool:
    """Extract one JPEG at timestamp, scaled to FRAME_WIDTH."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-ss", str(ts), "-i", video_path,
             "-vframes", "1",
             "-vf", f"scale={_FRAME_WIDTH}:-2",
             "-q:v", "2",
             out_path],
            capture_output=True, timeout=30,
        )
        return r.returncode == 0 and os.path.exists(out_path)
    except Exception as exc:
        logger.warning("Frame extract failed at %.2fs: %s", ts, exc)
        return False


class FrameExtractor:
    def extract_frames(self, video_path: str, doc_id: str) -> list[dict]:
        """
        Extract keyframes from video. Returns list of:
        {image_path, image_url, timestamp_ms, frame_index}

        Frame count logic:
        1. Scene detection finds natural scene changes
        2. If fewer than VIDEO_MIN_FRAMES (15): fill with uniform samples
        3. If more than VIDEO_MAX_FRAMES (45): evenly subsample
        Result: always between 15 and 45 frames for any video up to 3 min.

        Never raises. Returns [] on failure.
        """
        if not _FFMPEG_OK:
            logger.error("ffmpeg not found — install: brew/apt/winget install ffmpeg")
            return []

        os.makedirs(_FIGURES_DIR, exist_ok=True)
        duration = get_video_duration(video_path)

        # Step 1: scene detection
        timestamps = _detect_scenes(video_path)
        logger.info("Scene detection found %d timestamps for %s", len(timestamps), video_path)

        # Step 2: fill to minimum if too few
        if len(timestamps) < _MIN_FRAMES and duration > 0:
            logger.info(
                "Only %d scenes — filling to %d with uniform samples",
                len(timestamps), _MIN_FRAMES,
            )
            timestamps = _fill_uniform(timestamps, duration, _MIN_FRAMES)

        # Step 3: cap at maximum — evenly subsample
        if len(timestamps) > _MAX_FRAMES:
            logger.info("Capping %d timestamps to %d", len(timestamps), _MAX_FRAMES)
            step = len(timestamps) / _MAX_FRAMES
            timestamps = [timestamps[int(i * step)] for i in range(_MAX_FRAMES)]

        logger.info("Extracting %d frames from %s (duration=%.1fs)", len(timestamps), video_path, duration)

        # Step 4: extract JPEGs
        frames = []
        for idx, ts in enumerate(timestamps):
            ms   = int(ts * 1000)
            name = f"{doc_id}_f{idx:04d}_{ms}ms.jpg"
            path = os.path.join(_FIGURES_DIR, name)
            url  = f"{_FIGURES_SERVE_URL}/{name}"

            if not _extract_frame(video_path, ts, path):
                logger.warning("Skipping frame %d (%.2fs)", idx, ts)
                continue

            frames.append({
                "image_path": path, "image_url": url,
                "timestamp_ms": ms, "frame_index": idx,
            })

        logger.info("Extracted %d/%d frames from %s", len(frames), len(timestamps), video_path)
        return frames
