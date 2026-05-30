# ── VisionRAG Phase 4b: Gemini Video Pipeline ─────────────────────────────────
# File: backend/app/services/gemini_video_describer.py
# Changes: New file — native video description via Gemini 2.5 Flash
# Image pipeline: UNTOUCHED
# Ollama video pipeline: UNTOUCHED
# ─────────────────────────────────────────────────────────────────────────────

"""
Video description via Gemini 2.5 Flash API.
Sends entire video file to Google's API — native video understanding.
No frame extraction. No ffmpeg. No Ollama. No local GPU.

One API call → complete narrative with entities, key events, timestamps.

Env vars:
  GEMINI_API_KEY          str   required (get from https://aistudio.google.com/apikey)
  GEMINI_VIDEO_MODEL      str   default "gemini-2.5-flash"
  GEMINI_VIDEO_TIMEOUT    int   default 300 (seconds — includes upload + processing + generation)
"""

import os
import re
import json
import time
import logging

logger = logging.getLogger(__name__)

_GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
_GEMINI_MODEL      = os.getenv("GEMINI_VIDEO_MODEL", "gemini-2.5-flash")
_TIMEOUT           = int(os.getenv("GEMINI_VIDEO_TIMEOUT", "300"))
_POLL_INTERVAL     = 5  # seconds between file processing status checks

_EMPTY_RESULT = {
    "description": "", "entities": [], "relationships": [],
    "key_events": [], "figure_type": "image", "caption": "",
}


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    """Parse Gemini's JSON response. Never raises."""
    if not raw or not raw.strip():
        return dict(_EMPTY_RESULT)
    try:
        cleaned = raw.strip()
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
                return {"description": raw.strip()[:2000], "entities": [],
                        "relationships": [], "key_events": [],
                        "figure_type": "image", "caption": ""}

        if not isinstance(parsed, dict):
            return dict(_EMPTY_RESULT)

        result = {
            "description":   str(parsed.get("description") or "").strip(),
            "entities":      parsed.get("entities") or [],
            "relationships": parsed.get("relationships") or [],
            "key_events":    parsed.get("key_events") or [],
            "figure_type":   str(parsed.get("figure_type") or "image").strip().lower(),
            "caption":       str(parsed.get("caption") or "").strip(),
        }

        # Sanitise entities
        if not isinstance(result["entities"], list):
            result["entities"] = []
        result["entities"] = [str(e).strip() for e in result["entities"] if e and str(e).strip()]

        # Sanitise relationships
        if not isinstance(result["relationships"], list):
            result["relationships"] = []
        result["relationships"] = [
            r for r in result["relationships"]
            if isinstance(r, dict) and "from" in r and "to" in r
        ]

        # Sanitise key_events
        if not isinstance(result["key_events"], list):
            result["key_events"] = []
        result["key_events"] = [
            e for e in result["key_events"]
            if isinstance(e, dict) and "event" in e
        ]

        return result
    except Exception as exc:
        logger.warning("Gemini _parse_json failed: %s", exc)
        return {"description": raw.strip()[:2000], "entities": [],
                "relationships": [], "key_events": [],
                "figure_type": "image", "caption": ""}


# ── Prompt ────────────────────────────────────────────────────────────────────

_VIDEO_PROMPT = (
    "You are analyzing a video. Watch the ENTIRE video carefully from start to finish.\n\n"
    "Return ONLY valid JSON with no markdown, no backticks, no preamble:\n"
    '{\n'
    '  "description": "detailed 8-15 sentence chronological narrative covering: all scenes, '
    'every action and who performs it, all equipment and tools used, every cooking technique '
    'demonstrated, all visible ingredients, setting, and any text overlays",\n'
    '  "entities": ["every person, ingredient, cooking equipment, utensil, appliance, '
    'animal, location, and text visible or mentioned — be exhaustive"],\n'
    '  "relationships": [\n'
    '    {"from": "Entity A", "type": "ACTION_VERB", "to": "Entity B"}\n'
    '  ],\n'
    '  "key_events": [\n'
    '    {"timestamp_approx": "M:SS", "event": "2-3 sentence description: what action occurs, '
    'what equipment or ingredients are involved, what technique is demonstrated"}\n'
    '  ],\n'
    '  "figure_type": "image",\n'
    '  "caption": "any visible text overlays, titles, or captions — empty string if none"\n'
    '}\n\n'
    "For key_events density: one event every 3-5 seconds for videos under 60s, "
    "one every 5-10 seconds for longer videos. Aim for 10-20 events. "
    "Capture every scene change, technique change, ingredient addition, and notable visual moment."
)


# ── Public function ───────────────────────────────────────────────────────────

def describe_video_gemini(video_path: str, filename: str) -> dict:
    """
    Send entire video file to Gemini 2.5 Flash for native video understanding.

    Pipeline:
    1. Upload video to Gemini Files API
    2. Wait for processing (Gemini transcodes and indexes the video)
    3. Send generate_content request with the file reference + prompt
    4. Parse JSON response
    5. Delete uploaded file from Gemini

    Args:
        video_path: Absolute path to the video file on disk.
        filename:   Original filename (for logging).

    Returns:
        {description, entities, relationships, key_events, figure_type, caption}
        Returns _EMPTY_RESULT on any failure. Never raises.
    """
    if not _GEMINI_API_KEY:
        logger.error(
            "GEMINI_API_KEY not set — cannot process video with Gemini. "
            "Get a free key from https://aistudio.google.com/apikey"
        )
        return dict(_EMPTY_RESULT)

    from app.services.gemini_usage_tracker import is_quota_available, record_request
    if not is_quota_available():
        logger.error("Gemini daily quota exhausted — skipping video description for %s", filename)
        return dict(_EMPTY_RESULT)

    uploaded_file = None
    try:
        from google import genai

        client = genai.Client(api_key=_GEMINI_API_KEY)

        # Step 1 — Upload video to Gemini Files API
        logger.info("Gemini: uploading %s to Files API...", filename)
        uploaded_file = client.files.upload(file=video_path)
        logger.info("Gemini: file uploaded as %s, state=%s", uploaded_file.name, uploaded_file.state.name)

        # Step 2 — Wait for processing (Gemini transcodes the video)
        elapsed = 0
        while uploaded_file.state.name == "PROCESSING":
            if elapsed >= _TIMEOUT:
                logger.error("Gemini: file processing timed out after %ds for %s", _TIMEOUT, filename)
                return dict(_EMPTY_RESULT)
            time.sleep(_POLL_INTERVAL)
            elapsed += _POLL_INTERVAL
            uploaded_file = client.files.get(name=uploaded_file.name)
            logger.debug("Gemini: %s state=%s (%ds elapsed)", filename, uploaded_file.state.name, elapsed)

        if uploaded_file.state.name == "FAILED":
            logger.error("Gemini: file processing FAILED for %s", filename)
            return dict(_EMPTY_RESULT)

        # Step 3 — Generate description
        logger.info("Gemini: generating description for %s (model=%s)...", filename, _GEMINI_MODEL)
        response = client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=[uploaded_file, _VIDEO_PROMPT],
        )

        raw = response.text.strip() if response.text else ""
        if not raw:
            logger.warning("Gemini: empty response for %s", filename)
            return dict(_EMPTY_RESULT)

        usage = record_request()
        logger.info(
            "Gemini: got %d chars response for %s — usage %d/%d today (%d remaining)",
            len(raw), filename, usage["requests_today"], usage["daily_limit"], usage["remaining"],
        )
        return _parse_json(raw)

    except Exception as exc:
        logger.error("Gemini video description failed for %s: %s", filename, exc)
        return dict(_EMPTY_RESULT)

    finally:
        # Step 5 — Clean up uploaded file from Gemini
        if uploaded_file:
            try:
                from google import genai as _genai
                _client = _genai.Client(api_key=_GEMINI_API_KEY)
                _client.files.delete(name=uploaded_file.name)
                logger.debug("Gemini: deleted uploaded file %s", uploaded_file.name)
            except Exception as exc:
                logger.debug("Gemini: could not delete file %s: %s", uploaded_file.name, exc)


def is_gemini_configured() -> bool:
    """Check if GEMINI_API_KEY is set."""
    return bool(_GEMINI_API_KEY and _GEMINI_API_KEY.strip())
