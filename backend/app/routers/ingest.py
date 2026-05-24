# ── VisionRAG Neo4j Migration ─────────────────────────────────────────────────
# Replaces: nothing structural — adds DELETE /ingest/{filename} route only.
# Video-ready: dispatcher can be extended with _run_ingest_video() for .mp4/.mov
#              without touching any existing route.
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

_PDF_EXTS   = {".pdf"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
_ALL_EXTS   = _PDF_EXTS | _IMAGE_EXTS


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

def _run_ingest_image(job_id: str, filename: str, tmp_path: str) -> None:
    job_store.update_job(job_id, status="processing", total_chunks=1, figure_count=1)
    try:
        os.makedirs(_FIGURES_DIR, exist_ok=True)

        stem, ext = os.path.splitext(filename)
        fig_name  = f"{stem}_{job_id[:8]}{ext}"
        save_path = os.path.join(_FIGURES_DIR, fig_name)
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


# ── dispatcher ────────────────────────────────────────────────────────────────

def _run_ingest(job_id: str, filename: str, file_path: str) -> None:
    ext = os.path.splitext(filename)[1].lower()
    if ext in _IMAGE_EXTS:
        _run_ingest_image(job_id, filename, file_path)
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
            detail=f"Unsupported file type '{ext}'. Allowed: PDF, PNG, JPG, JPEG, WEBP",
        )

    data   = await file.read()
    suffix = ext or ".bin"
    tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()

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
