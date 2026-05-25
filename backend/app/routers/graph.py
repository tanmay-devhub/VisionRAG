# ── VisionRAG Phase 1 / 2 ──────────────────────────────────────────────────
# Step 4: Add /graph/entity-dedup and /graph/build-similarity endpoints
# RAG improvement: exposes dedup + visual similarity as on-demand API calls
# ──────────────────────────────────────────────────────────────────────────

import asyncio
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from app.services import graph_store

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/graph/files")
async def get_graph_files() -> dict:
    try:
        return graph_store.get_ingested_files()
    except Exception as exc:
        logger.error("Failed to fetch file list: %s", exc)
        return {"files": []}


@router.get("/graph/stats")
async def get_graph_stats() -> dict:
    try:
        return graph_store.get_graph_stats()
    except Exception as exc:
        logger.error("Failed to fetch graph stats: %s", exc)
        return {
            "total_chunks": 0, "figures": 0, "tables": 0,
            "text_chunks": 0, "entities": 0, "relationships": 0,
            "visually_similar_edges": 0,
        }


@router.get("/graph")
async def get_graph(filename: Optional[str] = Query(default=None)) -> dict:
    try:
        return graph_store.get_all_graph_data(filename=filename)
    except Exception as exc:
        logger.error("Failed to fetch graph data: %s", exc)
        return {"nodes": [], "links": []}


@router.delete("/graph")
async def delete_graph() -> dict:
    try:
        return graph_store.clear_graph()
    except Exception as exc:
        logger.error("Failed to clear graph: %s", exc)
        return {"deleted": 0, "error": str(exc)}


@router.post("/graph/entity-dedup")
async def run_entity_dedup(
    filename: Optional[str] = Query(
        default=None,
        description="Scope to one file. None = all files.",
    ),
) -> dict:
    """
    Merge near-duplicate VisualEntity nodes.
    Run after every new ingest, or on demand.
    Safe to run multiple times — idempotent.
    """
    try:
        from app.services.entity_dedup import run_entity_dedup as _dedup
        result = await asyncio.to_thread(_dedup, filename=filename)
        return {"status": "done", **result}
    except Exception as exc:
        logger.error("Entity dedup failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/graph/build-similarity")
async def build_similarity(
    filename: Optional[str] = Query(default=None),
    threshold: Optional[float] = Query(
        default=None,
        description="Cosine similarity cutoff. Overrides VISUAL_SIM_THRESHOLD env var.",
    ),
) -> dict:
    """
    Build VISUALLY_SIMILAR edges between figure/frame MediaChunk nodes.
    Run after entity-dedup completes.
    Safe to run multiple times — idempotent (updates scores on re-run).
    """
    try:
        from app.services.entity_dedup import build_visual_similarity
        result = await asyncio.to_thread(
            build_visual_similarity, filename=filename, threshold=threshold
        )
        return {"status": "done", **result}
    except Exception as exc:
        logger.error("Build similarity failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
