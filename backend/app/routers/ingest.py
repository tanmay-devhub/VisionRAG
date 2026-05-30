# ── VisionRAG Phase 5: Bug Fixes + Security ───────────────────────────────────
# File: backend/app/routers/ingest.py
# Fixes: E5 (Windows temp cleanup), B1 (file size), B8 (magic bytes), B9 (path traversal)
# ─────────────────────────────────────────────────────────────────────────────

import os
import gc
import shutil
import logging
import tempfile
import time as _time
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query

from app.schemas import IngestJobResponse, JobStatusResponse
from app.services import job_store, graph_store
from app.services.chunker import MultimodalChunker
from app.services.vision import VisionService
from app.services.figure_extractor import _FIGURES_DIR, _FIGURES_SERVE_URL

logger = logging.getLogger(__name__)
router = APIRouter()

_PDF_EXTS           = {".pdf"}
_IMAGE_EXTS         = {".png", ".jpg", ".jpeg", ".webp"}
_VIDEO_EXTS         = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
_ALL_EXTS           = _PDF_EXTS | _IMAGE_EXTS | _VIDEO_EXTS
_VIDEO_MAX_DURATION = int(os.getenv("VIDEO_MAX_DURATION_SEC", "180"))  # 3 minutes


# ── Fix E5: Windows-safe temp file deletion ───────────────────────────────────

def _safe_delete(path: str, retries: int = 5, delay: float = 1.0) -> None:
    """
    Delete a temp file with retries — handles Windows file locking.
    Windows subprocess file handles can take 1-3 seconds to release.
    """
    import time
    import platform

    # Windows needs a longer initial wait for subprocess handles to release
    if platform.system() == "Windows":
        time.sleep(0.5)  # let ffprobe/ffmpeg release the handle

    for attempt in range(retries):
        try:
            gc.collect()  # release Python-held file handles
            if os.path.exists(path):
                os.unlink(path)
            return  # success or already deleted
        except PermissionError:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                logger.warning(
                    "Could not delete temp file %s after %d attempts (Windows file lock)",
                    path, retries,
                )
        except FileNotFoundError:
            return  # already cleaned up by another path
        except Exception as exc:
            logger.warning("Temp cleanup failed for %s: %s", path, exc)
            return


# ── PDF ingest ────────────────────────────────────────────────────────────────

def _run_ingest_pdf(job_id: str, filename: str, pdf_path: str) -> None:
    job_store.update_job(job_id, status="processing")
    try:
        chunker = MultimodalChunker()
        doc_id  = os.path.splitext(filename)[0] + "_" + job_id[:8]

        chunks = chunker.chunk_document(pdf_path, doc_id)
        if not chunks:
            raise ValueError("No content could be extracted from the file")

        figure_count = sum(1 for c in chunks if c.get("chunk_type") == "figure")
        table_count  = sum(1 for c in chunks if c.get("chunk_type") == "table")

        job_store.update_job(
            job_id,
            total_chunks=len(chunks),
            figure_count=figure_count,
            table_count=table_count,
        )

        graph_store.store_chunks(
            chunks,
            filename,
            progress_cb=lambda done: job_store.update_job(job_id, chunks_done=done),
        )

        job_store.update_job(job_id, status="done", chunks_done=len(chunks))
        logger.info(
            "Job %s done: %s — %d chunks (%d figures, %d tables)",
            job_id, filename, len(chunks), figure_count, table_count,
        )
    except Exception as exc:
        logger.error("Job %s failed: %s", job_id, exc)
        job_store.update_job(job_id, status="error", error=str(exc))
    finally:
        _safe_delete(pdf_path)


# ── Image ingest ──────────────────────────────────────────────────────────────

_MAX_IMAGE_DIM = 1280  # longest side — prevents Ollama VRAM exhaustion on large images


def _resize_image(src_path: str, dst_path: str, max_dim: int = _MAX_IMAGE_DIM) -> None:
    """Resize image so its longest side is at most max_dim, save as PNG."""
    from PIL import Image as PILImage
    with PILImage.open(src_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
            logger.info("Resized %s from %dx%d to %dx%d", src_path, w, h, img.size[0], img.size[1])
        img.save(dst_path, format="PNG")


def _run_ingest_image(job_id: str, filename: str, tmp_path: str) -> None:
    job_store.update_job(job_id, status="processing", total_chunks=1, figure_count=1)
    try:
        os.makedirs(_FIGURES_DIR, exist_ok=True)

        stem      = os.path.splitext(filename)[0]
        fig_name  = f"{stem}_{job_id[:8]}.png"   # always PNG after resize
        save_path = os.path.join(_FIGURES_DIR, fig_name)

        try:
            _resize_image(tmp_path, save_path)
        except Exception as exc:
            logger.warning("Resize failed for %s (%s) — copying original", filename, exc)
            shutil.copy2(tmp_path, save_path)

        vision   = VisionService()
        result   = vision.describe_figure(save_path, "")
        desc     = result.get("description", "")

        if not desc:
            raise ValueError("Vision model returned no description — check VISION_BACKEND and model setup")

        chunks = [{
            "text":               desc,
            "chunk_index":        0,
            "chunk_type":         "figure",
            "image_path":         save_path,
            "image_url":          f"{_FIGURES_SERVE_URL}/{fig_name}",
            "figure_type":        result.get("figure_type", "image"),
            "caption":            result.get("caption", ""),
            "page_number":        0,
            "extra_entities":     result.get("entities", []),
            "extra_relationships": result.get("relationships", []),
        }]

        graph_store.store_chunks(
            chunks,
            filename,
            progress_cb=lambda done: job_store.update_job(job_id, chunks_done=done),
        )

        job_store.update_job(job_id, status="done", chunks_done=1)
        logger.info("Job %s done: %s — image ingested as figure chunk", job_id, filename)

    except Exception as exc:
        logger.error("Job %s failed: %s", job_id, exc)
        job_store.update_job(job_id, status="error", error=str(exc))
    finally:
        _safe_delete(tmp_path)


# ── Video ingest via Ollama Cloud (frame-by-frame — no local GPU) ─────────────

def _run_ingest_video_ollama(job_id: str, filename: str, tmp_path: str) -> None:
    """
    Video ingest pipeline via Ollama Cloud:
    1. Extract 15-45 keyframes via ffmpeg (CPU only)
    2. Send ALL frames to Ollama Cloud for video summary (one /api/chat call)
    3. Send each frame individually for per-frame descriptions
    4. Store video_summary + frame chunks in Neo4j

    Uses qwen3-vl:235b-instruct-cloud via Ollama Cloud /api/chat.
    Does NOT use local Ollama, VisionService, or describe_figure.
    """
    from app.services.frame_extractor import FrameExtractor, get_video_duration
    from app.services.video_describer import describe_video_summary, describe_single_frame

    job_store.update_job(job_id, status="processing")
    try:
        os.makedirs(_FIGURES_DIR, exist_ok=True)
        stem     = os.path.splitext(filename)[0]
        doc_id   = stem + "_" + job_id[:8]
        duration = get_video_duration(tmp_path)

        # Step 1 — Extract keyframes (ffmpeg, CPU only)
        logger.info("Job %s: extracting frames from %s (%.1fs)", job_id, filename, duration)
        extractor = FrameExtractor()
        frames    = extractor.extract_frames(tmp_path, doc_id)

        if not frames:
            raise ValueError(
                "No frames extracted. Check: ffmpeg installed, video valid. "
                "Supported: MP4, MOV, AVI, MKV, WEBM."
            )

        job_store.update_job(
            job_id, total_chunks=len(frames) + 1,
            figure_count=len(frames), table_count=0,
        )

        chunks = []

        # Step 2 — Video summary (ALL frames in one Ollama Cloud call)
        logger.info("Job %s: generating video summary (%d frames via cloud)", job_id, len(frames))
        summary = describe_video_summary(frames, filename, duration)
        summary_desc = summary.get("description", "")

        if summary_desc:
            key_events = summary.get("key_events", [])
            desc_with_events = summary_desc
            if key_events:
                events_text = " | ".join(
                    f"[{e.get('timestamp_approx', '?')}] {e.get('event', '')}"
                    for e in key_events if e.get("event")
                )
                desc_with_events = f"{summary_desc}\n\nKey events: {events_text}"

            chunks.append({
                "text":                desc_with_events,
                "chunk_index":         0,
                "chunk_type":          "video_summary",
                "image_path":          "",
                "image_url":           "",
                "figure_type":         summary.get("figure_type", "image"),
                "caption":             summary.get("caption", ""),
                "page_number":         0,
                "timestamp_ms":        0,
                "extra_entities":      summary.get("entities", []),
                "extra_relationships": summary.get("relationships", []),
            })
            logger.info(
                "Job %s: video summary done (%d entities, %d events)",
                job_id, len(summary.get("entities", [])), len(summary.get("key_events", [])),
            )
        else:
            logger.warning("Job %s: summary returned empty — frames only", job_id)

        # Step 3 — Individual frame descriptions via Cloud (parallel)
        _PARALLEL = int(os.getenv("VIDEO_PARALLEL_FRAMES", "3"))
        logger.info(
            "Job %s: describing %d frames via cloud (%d workers)",
            job_id, len(frames), _PARALLEL,
        )

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _describe_frame(args):
            idx, frm = args
            return idx, frm, describe_single_frame(
                image_path=frm["image_path"],
                timestamp_ms=frm["timestamp_ms"],
                filename=filename,
            )

        frame_results: list[tuple[int, dict, dict]] = []
        completed_count = len(chunks)  # summary chunk already counted
        with ThreadPoolExecutor(max_workers=_PARALLEL) as executor:
            future_map = {executor.submit(_describe_frame, (i, f)): i for i, f in enumerate(frames)}
            for future in as_completed(future_map):
                try:
                    idx, frm, result = future.result()
                    frame_results.append((idx, frm, result))
                except Exception as exc:
                    logger.warning("Job %s: frame future failed: %s", job_id, exc)
                completed_count += 1
                job_store.update_job(job_id, chunks_done=completed_count)

        # Restore temporal order
        frame_results.sort(key=lambda x: x[0])

        for idx, frame, result in frame_results:
            desc = result.get("description", "")
            if not desc:
                logger.warning("Job %s: empty frame %d (t=%dms)", job_id, idx, frame["timestamp_ms"])
                continue
            chunks.append({
                "text":                desc,
                "chunk_index":         len(chunks),
                "chunk_type":          "frame",
                "image_path":          frame["image_path"],
                "image_url":           frame["image_url"],
                "figure_type":         result.get("figure_type", "image"),
                "caption":             result.get("caption", ""),
                "page_number":         frame["frame_index"],
                "timestamp_ms":        frame["timestamp_ms"],
                "extra_entities":      result.get("entities", []),
                "extra_relationships": result.get("relationships", []),
            })
            logger.info("Job %s: frame %d/%d processed (t=%dms)", job_id, idx + 1, len(frames), frame["timestamp_ms"])

        if not chunks:
            raise ValueError(
                "No content generated. Check Ollama Cloud model is accessible. "
                "Test: ollama run qwen3-vl:235b-instruct-cloud 'hello'"
            )

        # Step 4 — Store in Neo4j
        graph_store.store_chunks(
            chunks, filename,
            progress_cb=lambda done: job_store.update_job(job_id, chunks_done=done),
        )

        summary_count = 1 if summary_desc else 0
        frame_count   = len(chunks) - summary_count
        job_store.update_job(job_id, status="done", chunks_done=len(chunks))
        logger.info(
            "Job %s done: %s — %d summary + %d frames = %d chunks",
            job_id, filename, summary_count, frame_count, len(chunks),
        )

    except Exception as exc:
        logger.error("Job %s failed: %s", job_id, exc)
        job_store.update_job(job_id, status="error", error=str(exc))
    finally:
        _safe_delete(tmp_path)


# ── Video ingest via Gemini (parallel pipeline — no ffmpeg, no frames) ────────

def _run_ingest_video_gemini(job_id: str, filename: str, tmp_path: str) -> None:
    """
    Video ingest via Gemini 2.5 Flash:
    1. Upload entire MP4 to Gemini Files API
    2. ONE API call → complete video description with timestamps
    3. Store video_summary + key_event chunks in Neo4j

    No ffmpeg. No frame extraction. No Ollama.
    Gemini watches the entire video natively including audio.
    """
    from app.services.gemini_video_describer import describe_video_gemini

    job_store.update_job(job_id, status="processing")
    try:
        # ONE call — Gemini processes the entire video
        logger.info("Job %s: sending %s to Gemini for video description", job_id, filename)
        result = describe_video_gemini(tmp_path, filename)

        desc = result.get("description", "")
        if not desc:
            raise ValueError(
                "Gemini returned no description. Check: "
                "GEMINI_API_KEY is set, video file is valid, API quota not exceeded. "
                "Get a free key: https://aistudio.google.com/apikey"
            )

        chunks = []

        # Video summary chunk (primary — the full narrative)
        key_events = result.get("key_events", [])
        desc_with_events = desc
        if key_events:
            events_text = " | ".join(
                f"[{e.get('timestamp_approx', '?')}] {e.get('event', '')}"
                for e in key_events if e.get("event")
            )
            desc_with_events = f"{desc}\n\nKey events: {events_text}"

        chunks.append({
            "text":                desc_with_events,
            "chunk_index":         0,
            "chunk_type":          "video_summary",
            "image_path":          "",
            "image_url":           "",
            "figure_type":         result.get("figure_type", "image"),
            "caption":             result.get("caption", ""),
            "page_number":         0,
            "timestamp_ms":        0,
            "extra_entities":      result.get("entities", []),
            "extra_relationships": result.get("relationships", []),
        })

        # Key event chunks (one per timestamp — for fine-grained retrieval)
        _EVENT_STOPWORDS = {
            "the", "and", "with", "from", "into", "that", "this",
            "then", "also", "more", "very", "been", "have", "were",
            "what", "when", "where", "which", "while", "their",
            "there", "about", "after", "before", "between",
        }
        # Top entities from the summary — inherited by every event chunk so
        # all events share DEPICTS edges to the video's main subjects
        _summary_entities = result.get("entities", [])[:5]

        for i, event in enumerate(key_events):
            event_text = event.get("event", "")
            if not event_text:
                continue

            # Parse "M:SS" → milliseconds
            ts_str = event.get("timestamp_approx", "0:00")
            try:
                parts = ts_str.split(":")
                if len(parts) == 2:
                    ts_ms = (int(parts[0]) * 60 + int(parts[1])) * 1000
                elif len(parts) == 3:
                    ts_ms = (int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])) * 1000
                else:
                    ts_ms = 0
            except (ValueError, IndexError):
                ts_ms = 0

            # Extract meaningful words from event text as entities
            event_entities = [
                w for word in event_text.split()
                if len(w := word.strip(".,;:!?()[]{}\"'")) > 3
                and w.lower() not in _EVENT_STOPWORDS
            ]
            chunks.append({
                "text":                f"[{ts_str}] {event_text}",
                "chunk_index":         len(chunks),
                "chunk_type":          "frame",  # "frame" so entity storage triggers in graph_store
                "image_path":          "",
                "image_url":           "",
                "figure_type":         "image",
                "caption":             "",
                "page_number":         i + 1,
                "timestamp_ms":        ts_ms,
                "extra_entities":      list(set(event_entities + _summary_entities)),
                "extra_relationships": [],
            })

        job_store.update_job(
            job_id,
            total_chunks=len(chunks),
            figure_count=0,
            table_count=0,
        )

        # Store in Neo4j — same function as all other pipelines
        graph_store.store_chunks(
            chunks, filename,
            progress_cb=lambda done: job_store.update_job(job_id, chunks_done=done),
        )

        job_store.update_job(job_id, status="done", chunks_done=len(chunks))
        logger.info(
            "Job %s done (Gemini): %s — 1 summary + %d event chunks",
            job_id, filename, len(key_events),
        )

    except Exception as exc:
        logger.error("Job %s failed (Gemini): %s", job_id, exc)
        job_store.update_job(job_id, status="error", error=str(exc))
    finally:
        _safe_delete(tmp_path)


# ── dispatcher ────────────────────────────────────────────────────────────────

def _run_ingest(job_id: str, filename: str, file_path: str, video_backend: str = "ollama") -> None:
    ext = os.path.splitext(filename)[1].lower()
    if ext in _IMAGE_EXTS:
        _run_ingest_image(job_id, filename, file_path)
    elif ext in _VIDEO_EXTS:
        if video_backend == "gemini":
            _run_ingest_video_gemini(job_id, filename, file_path)
        else:
            _run_ingest_video_ollama(job_id, filename, file_path)
    else:
        _run_ingest_pdf(job_id, filename, file_path)


# ── routes ────────────────────────────────────────────────────────────────────

import re as _re

_MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "100")) * 1024 * 1024  # default 100MB

# ── Fix B8: Magic byte validation ─────────────────────────────────────────────
_MAGIC_BYTES = {
    ".png":  [b"\x89PNG"],
    ".jpg":  [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
    ".webp": [b"RIFF"],
    ".pdf":  [b"%PDF"],
    ".mp4":  [b"\x00\x00\x00", b"ftyp"],
    ".mov":  [b"\x00\x00\x00", b"ftyp", b"moov"],
    ".avi":  [b"RIFF"],
    ".mkv":  [b"\x1a\x45\xdf\xa3"],
    ".webm": [b"\x1a\x45\xdf\xa3"],
}


def _validate_file_content(data: bytes, ext: str) -> bool:
    """Check if file content matches its extension via magic bytes."""
    if ext not in _MAGIC_BYTES:
        return True
    magics = _MAGIC_BYTES[ext]
    header = data[:12]
    return any(magic in header for magic in magics)


# ── Fix B9: Filename sanitization ─────────────────────────────────────────────
def _sanitize_filename(filename: str) -> str:
    """Strip path components and dangerous characters from uploaded filename."""
    # Remove path separators
    filename = filename.replace("/", "_").replace("\\", "_")
    # Take only the basename
    filename = os.path.basename(filename)
    # Strip ".." sequences that survived path separator replacement
    while ".." in filename:
        filename = filename.replace("..", "")
    # Remove anything that's not alphanumeric, dot, dash, underscore
    filename = _re.sub(r"[^\w.\-]", "_", filename)
    # Collapse multiple underscores
    filename = _re.sub(r"_+", "_", filename).strip("_")
    # Prevent empty or dot-only names
    if not filename or filename.startswith("."):
        filename = f"upload_{filename}"
    # Truncate excessively long names
    return filename[:200]


@router.post("/ingest", response_model=IngestJobResponse, status_code=202)
async def ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    video_backend: str = Query(
        default="gemini",
        description="Video processing backend: 'ollama' (frame-by-frame via Ollama Cloud) or 'gemini' (native video via Gemini 2.5 Flash)",
        pattern="^(ollama|gemini)$",
    ),
) -> IngestJobResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # ── Fix B9: sanitize filename ─────────────────────────────────────────
    filename = _sanitize_filename(file.filename)
    ext      = os.path.splitext(filename)[1].lower()
    if ext not in _ALL_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: PDF, PNG, JPG, JPEG, WEBP, MP4, MOV, AVI, MKV, WEBM",
        )

    # ── Fix B1: file size limit (stream-check) ────────────────────────────
    content_length = file.size
    if content_length and content_length > _MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({content_length // (1024*1024)}MB). Maximum: {_MAX_FILE_SIZE // (1024*1024)}MB.",
        )

    # Read in chunks to enforce limit even without Content-Length header
    data = bytearray()
    chunk_size = 1024 * 1024  # 1MB chunks
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > _MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds {_MAX_FILE_SIZE // (1024*1024)}MB limit.",
            )
    data = bytes(data)

    # ── Fix B8: validate file content matches extension ───────────────────
    if not _validate_file_content(data, ext):
        raise HTTPException(
            status_code=400,
            detail=f"File content doesn't match extension '{ext}'. File may be corrupted or mislabeled.",
        )

    suffix = ext or ".bin"
    tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()

    # Video duration check — reject videos longer than 3 minutes
    if ext in _VIDEO_EXTS:
        from app.services.frame_extractor import get_video_duration
        # Brief pause for Windows to release ffprobe file handle
        _time.sleep(0.2)
        duration = get_video_duration(tmp.name)
        if duration > _VIDEO_MAX_DURATION:
            _safe_delete(tmp.name)
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video is {int(duration)}s long (max {_VIDEO_MAX_DURATION}s / "
                    f"{_VIDEO_MAX_DURATION // 60} minutes). "
                    "Please trim your video to 3 minutes or less."
                ),
            )
        if duration == 0:
            _safe_delete(tmp.name)
            raise HTTPException(
                status_code=400,
                detail="Could not determine video duration. Is ffmpeg installed? Is the file a valid video?",
            )

        # Gemini-specific validation
        if video_backend == "gemini":
            from app.services.gemini_video_describer import is_gemini_configured
            if not is_gemini_configured():
                _safe_delete(tmp.name)
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "GEMINI_API_KEY is not configured. "
                        "Get a free key from https://aistudio.google.com/apikey "
                        "and add it to your .env file."
                    ),
                )

    job_id = job_store.create_job(filename)
    background_tasks.add_task(_run_ingest, job_id, filename, tmp.name, video_backend)

    return IngestJobResponse(job_id=job_id, filename=filename, status="pending")


@router.get("/ingest/status/{job_id}", response_model=JobStatusResponse)
async def ingest_status(job_id: str) -> JobStatusResponse:
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(**job)


@router.get("/ingest/jobs")
async def list_ingest_jobs() -> dict:
    return {"jobs": job_store.list_jobs()}


@router.delete("/ingest/{filename}")
async def delete_file(filename: str) -> dict:
    """Delete all Neo4j data for a specific ingested file."""
    result = graph_store.delete_file(filename)
    return {"message": f"Deleted {filename}", **result}
