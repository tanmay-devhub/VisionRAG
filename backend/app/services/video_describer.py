# ── VisionRAG Phase 4: Ollama Cloud Video ─────────────────────────────────────
# File: backend/app/services/video_describer.py
# Changes: New file — video frame description via Ollama Cloud /api/chat
# Image pipeline: UNTOUCHED
# ─────────────────────────────────────────────────────────────────────────────

"""
Video frame description via Ollama Cloud.

Sends frames to qwen3-vl:235b-instruct-cloud for description.
Does NOT use local Ollama, VisionService, or describe_figure.
Completely independent of the image pipeline.

CRITICAL: Uses /api/chat endpoint, NOT /api/generate.
Ollama Cloud /api/generate silently ignores images (GitHub issue #12789).

Env vars:
  OLLAMA_BASE_URL         str   default http://localhost:11434
  VIDEO_CLOUD_MODEL       str   default qwen3-vl:235b-instruct-cloud
  VIDEO_REQUEST_TIMEOUT   int   default 300 (seconds per request)
"""

import os
import re
import json
import base64
import logging
import time

logger = logging.getLogger(__name__)

_OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_VIDEO_CLOUD_MODEL = os.getenv("VIDEO_CLOUD_MODEL", "qwen3-vl:235b-instruct-cloud")
_REQUEST_TIMEOUT   = int(os.getenv("VIDEO_REQUEST_TIMEOUT", "300"))

_EMPTY_RESULT = {
    "description": "", "entities": [], "relationships": [],
    "figure_type": "image", "caption": "",
}


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    """Parse model JSON output. Never raises."""
    if not raw or not raw.strip():
        return dict(_EMPTY_RESULT)
    try:
        cleaned = raw.strip()
        # Strip <think> tags (qwen3 thinking mode)
        cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL).strip()
        # Strip markdown fences
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
            else:
                return {"description": raw.strip()[:1500], "entities": [],
                        "relationships": [], "figure_type": "image", "caption": ""}

        if not isinstance(parsed, dict):
            return dict(_EMPTY_RESULT)

        result = {
            "description":   str(parsed.get("description") or "").strip(),
            "entities":      parsed.get("entities") or [],
            "relationships": parsed.get("relationships") or [],
            "figure_type":   str(parsed.get("figure_type") or "image").strip().lower(),
            "caption":       str(parsed.get("caption") or "").strip(),
        }
        if "key_events" in parsed:
            result["key_events"] = parsed["key_events"]

        if not isinstance(result["entities"], list):
            result["entities"] = []
        result["entities"] = [str(e).strip() for e in result["entities"] if e and str(e).strip()]
        if not isinstance(result["relationships"], list):
            result["relationships"] = []
        result["relationships"] = [
            r for r in result["relationships"]
            if isinstance(r, dict) and "from" in r and "to" in r
        ]
        return result
    except Exception as exc:
        logger.warning("_parse_json failed: %s", exc)
        return {"description": raw.strip()[:1500], "entities": [],
                "relationships": [], "figure_type": "image", "caption": ""}


# ── Prompts ───────────────────────────────────────────────────────────────────

_FRAME_PROMPT = (
    "You are analyzing a single frame from a video.\n"
    "Frame timestamp: {timestamp_ms}ms into the video.\n"
    "Video: {filename}\n\n"
    "This could be any video type: lecture, screen recording, documentary, "
    "action footage, presentation, animation, or short-form content.\n\n"
    "Instructions:\n"
    "- SLIDE/PRESENTATION: extract all visible text, describe charts/diagrams.\n"
    "- SCREEN RECORDING: describe UI, app, code, text on screen.\n"
    "- PEOPLE/ACTION: describe what is happening, who/what is visible, setting.\n"
    "- TEXT OVERLAYS: transcribe text exactly, then describe the visual.\n"
    "- ANIMATION/CARTOON: describe the characters, their actions, the scene.\n"
    "- BLURRY/LOW-LIGHT: describe what can be inferred — don't skip.\n\n"
    "Return ONLY valid JSON, no markdown:\n"
    '{{\n'
    '  "description": "2-3 sentence description of what this frame shows",\n'
    '  "entities": ["entity1", "entity2", ...],\n'
    '  "relationships": [{{"from": "A", "type": "RELATES_TO", "to": "B"}}],\n'
    '  "figure_type": "chart|diagram|flowchart|table|equation|image",\n'
    '  "caption": "visible text overlay or empty string"\n'
    '}}'
)

_SUMMARY_PROMPT = (
    "You are analyzing {num_frames} frames from a video in chronological order.\n"
    "Video: {filename}\n"
    "The frames are evenly sampled across the video — one every ~{interval_sec} seconds.\n\n"
    "Describe EVERYTHING that happens across all frames as a continuous narrative.\n"
    "Include: what changes between frames, actions occurring, all characters/objects,\n"
    "the setting, and the overall flow of events.\n\n"
    "Return ONLY valid JSON, no markdown:\n"
    '{{\n'
    '  "description": "detailed 5-10 sentence chronological narrative of the entire video",\n'
    '  "entities": ["every character, object, text, location visible across all frames"],\n'
    '  "relationships": [{{"from": "A", "type": "RELATES_TO", "to": "B"}}],\n'
    '  "key_events": [\n'
    '    {{"timestamp_approx": "M:SS", "event": "what happens"}}\n'
    '  ],\n'
    '  "figure_type": "image",\n'
    '  "caption": "any visible text/titles across frames — empty string if none"\n'
    '}}'
)


# ── Core API call ─────────────────────────────────────────────────────────────

def _call_cloud(content: str, image_b64_list: list[str]) -> dict:
    """
    Send one request to Ollama Cloud via /api/chat.
    Uses _VIDEO_CLOUD_MODEL. Retries 3x on failure.
    Returns parsed JSON dict. Never raises.

    CRITICAL: Uses /api/chat with messages format.
    /api/generate silently ignores images on cloud models.
    """
    import httpx

    payload = {
        "model": _VIDEO_CLOUD_MODEL,
        "messages": [{
            "role": "user",
            "content": content,
            "images": image_b64_list,
        }],
        "stream": False,
    }

    for attempt in range(3):
        try:
            resp = httpx.post(
                f"{_OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=float(_REQUEST_TIMEOUT),
            )
            resp.raise_for_status()
            raw = resp.json().get("message", {}).get("content", "").strip()
            if not raw:
                logger.warning("Cloud returned empty (attempt %d/3)", attempt + 1)
                if attempt < 2:
                    time.sleep(10)
                    continue
                return dict(_EMPTY_RESULT)
            return _parse_json(raw)

        except Exception as exc:
            if attempt < 2:
                wait = 15 * (attempt + 1)
                logger.warning("Cloud error (attempt %d/3): %s — retry in %ds", attempt + 1, exc, wait)
                time.sleep(wait)
                continue
            logger.error("Cloud failed after 3 attempts: %s", exc)
            return dict(_EMPTY_RESULT)

    return dict(_EMPTY_RESULT)


# ── Helper ────────────────────────────────────────────────────────────────────

def _read_b64(path: str) -> str | None:
    """Read file and return base64. Returns None on failure."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception as exc:
        logger.warning("Cannot read %s: %s", path, exc)
        return None


# ── Public functions ──────────────────────────────────────────────────────────

def describe_single_frame(
    image_path: str, timestamp_ms: int, filename: str
) -> dict:
    """
    Describe one video frame via Ollama Cloud /api/chat.
    Returns {description, entities, relationships, figure_type, caption}.
    """
    b64 = _read_b64(image_path)
    if not b64:
        return dict(_EMPTY_RESULT)
    prompt = _FRAME_PROMPT.format(timestamp_ms=timestamp_ms, filename=filename)
    return _call_cloud(prompt, [b64])


def describe_video_summary(
    frames: list[dict], filename: str, video_duration_sec: float = 0
) -> dict:
    """
    Send ALL frame images in ONE Ollama Cloud call for a narrative summary.
    The 235B model sees the entire sequence and generates a chronological narrative
    with key_events and timestamps.

    Args:
        frames:             list of {image_path, image_url, timestamp_ms, frame_index}
        filename:           original video filename
        video_duration_sec: total video duration for interval calculation

    Returns:
        {description, entities, relationships, key_events, figure_type, caption}
    """
    b64_list = []
    for frame in frames:
        b64 = _read_b64(frame["image_path"])
        if b64:
            b64_list.append(b64)

    if not b64_list:
        logger.error("No frames readable for summary of %s", filename)
        return dict(_EMPTY_RESULT)

    interval = round(video_duration_sec / max(len(b64_list), 1), 1) if video_duration_sec > 0 else 4
    prompt = _SUMMARY_PROMPT.format(
        num_frames=len(b64_list), filename=filename, interval_sec=interval,
    )
    logger.info("Sending %d frames to cloud (%s) for video summary", len(b64_list), _VIDEO_CLOUD_MODEL)
    return _call_cloud(prompt, b64_list)
