# ── VisionRAG Video Quality + Gemini Tracking ────────────────────────────────
# File: backend/app/services/gemini_usage_tracker.py
# Fixes: Fix 3 — Gemini usage tracking
# ─────────────────────────────────────────────────────────────────────────────

"""
Local Gemini API usage tracker.
Counts every call to Gemini, persists to a JSON file, resets daily at midnight Pacific.

Google's free tier limits (as of May 2026):
  gemini-2.5-flash:      1,500 RPD, 15 RPM, 1,000,000 TPM
  gemini-2.5-flash-lite: 1,500 RPD, 30 RPM, 1,000,000 TPM
  gemini-2.5-pro:        50 RPD (paid-only for most accounts)

Quotas reset at midnight Pacific time.
"""

import os
import json
import logging
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

_TRACKER_PATH = os.getenv(
    "GEMINI_TRACKER_PATH",
    str(Path(__file__).resolve().parent.parent.parent / "gemini_usage.json"),
)
_DAILY_LIMIT    = int(os.getenv("GEMINI_DAILY_LIMIT", "1500"))
_PACIFIC_OFFSET = timedelta(hours=-7)  # PDT; PST = -8
_lock           = threading.Lock()


def _pacific_today() -> str:
    utc_now = datetime.now(timezone.utc)
    return (utc_now + _PACIFIC_OFFSET).strftime("%Y-%m-%d")


def _load() -> dict:
    try:
        with open(_TRACKER_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("date") != _pacific_today():
            return {"date": _pacific_today(), "requests": 0, "last_request": None}
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {"date": _pacific_today(), "requests": 0, "last_request": None}


def _save(data: dict) -> None:
    try:
        with open(_TRACKER_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        logger.warning("Could not save Gemini usage tracker: %s", exc)


def record_request() -> dict:
    """Record one Gemini API request. Call after every successful Gemini call. Thread-safe."""
    with _lock:
        data = _load()
        data["requests"] += 1
        data["last_request"] = datetime.now(timezone.utc).isoformat()
        _save(data)
        remaining = max(0, _DAILY_LIMIT - data["requests"])
        if remaining < 100:
            logger.warning(
                "Gemini usage: %d/%d requests today (%d remaining)",
                data["requests"], _DAILY_LIMIT, remaining,
            )
        return {
            "date":           data["date"],
            "requests_today": data["requests"],
            "daily_limit":    _DAILY_LIMIT,
            "remaining":      remaining,
            "last_request":   data["last_request"],
        }


def get_usage() -> dict:
    """Get current Gemini usage stats without recording a request."""
    with _lock:
        data = _load()
        remaining = max(0, _DAILY_LIMIT - data["requests"])
        return {
            "date":           data["date"],
            "requests_today": data["requests"],
            "daily_limit":    _DAILY_LIMIT,
            "remaining":      remaining,
            "last_request":   data.get("last_request"),
            "tracker_file":   _TRACKER_PATH,
        }


def is_quota_available() -> bool:
    """Check if there's remaining quota before making a Gemini call."""
    with _lock:
        data = _load()
        return data["requests"] < _DAILY_LIMIT
