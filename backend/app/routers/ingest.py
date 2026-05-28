# ── VisionRAG Phase 4: Ollama Cloud Video ─────────────────────────────────────
# File: backend/app/routers/ingest.py
# Changes: Added _VIDEO_EXTS, duration check, _run_ingest_video (Ollama Cloud)
# Image pipeline: UNTOUCHED
# ─────────────────────────────────────────────────────────────────────────────

import os
import shutil
import logging
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks

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
        try:
            os.unlink(pdf_path)
        except Exception:
            pass


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
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── Video ingest (Ollama Cloud — no local GPU) ────────────────────────────────

def _run_ingest_video(job_id: str, filename: str, tmp_path: str) -> None:
    """
    Video ingest pipeline:
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

        # Step 3 — Individual frame descriptions via Cloud
        logger.info("Job %s: describing %d frames individually via cloud", job_id, len(frames))
        for i, frame in enumerate(frames):
            try:
                result = describe_single_frame(
                    image_path=frame["image_path"],
                    timestamp_ms=frame["timestamp_ms"],
                    filename=filename,
                )
                desc = result.get("description", "")
                if not desc:
                    logger.warning("Job %s: empty frame %d (t=%dms)", job_id, i, frame["timestamp_ms"])
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
                job_store.update_job(job_id, chunks_done=len(chunks))
                logger.info("Job %s: frame %d/%d (t=%dms)", job_id, i + 1, len(frames), frame["timestamp_ms"])

            except Exception as exc:
                logger.warning("Job %s: frame %d failed: %s", job_id, i, exc)
                continue

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
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── dispatcher ────────────────────────────────────────────────────────────────

def _run_ingest(job_id: str, filename: str, file_path: str) -> None:
    ext = os.path.splitext(filename)[1].lower()
    if ext in _IMAGE_EXTS:
        _run_ingest_image(job_id, filename, file_path)
    elif ext in _VIDEO_EXTS:
        _run_ingest_video(job_id, filename, file_path)
    else:
        _run_ingest_pdf(job_id, filename, file_path)


# ── routes ────────────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestJobResponse, status_code=202)
async def ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> IngestJobResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    filename = file.filename
    ext      = os.path.splitext(filename)[1].lower()
    if ext not in _ALL_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: PDF, PNG, JPG, JPEG, WEBP, MP4, MOV, AVI, MKV, WEBM",
        )

    data   = await file.read()
    suffix = ext or ".bin"
    tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()

    # Video duration check — reject videos longer than 3 minutes
    if ext in _VIDEO_EXTS:
        from app.services.frame_extractor import get_video_duration
        duration = get_video_duration(tmp.name)
        if duration > _VIDEO_MAX_DURATION:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Video is {int(duration)}s long (max {_VIDEO_MAX_DURATION}s / "
                    f"{_VIDEO_MAX_DURATION // 60} minutes). "
                    "Please trim your video to 3 minutes or less."
                ),
            )
        if duration == 0:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
            raise HTTPException(
                status_code=400,
                detail="Could not determine video duration. Is ffmpeg installed? Is the file a valid video?",
            )

    job_id = job_store.create_job(filename)
    background_tasks.add_task(_run_ingest, job_id, filename, tmp.name)

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
