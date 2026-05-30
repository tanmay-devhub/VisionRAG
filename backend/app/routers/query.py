# ── VisionRAG Phase 5: Quality + Security ─────────────────────────────────────
# File: backend/app/routers/query.py
# Fixes: A2 (irrelevant source filtering)
# ─────────────────────────────────────────────────────────────────────────────

import os
import asyncio
import logging
from fastapi import APIRouter, HTTPException
from app.schemas import QueryRequest, QueryResponse, Source
from app.services import graph_store, llm, reranker

logger = logging.getLogger(__name__)
router = APIRouter()

_MIN_SOURCE_SCORE = float(os.getenv("MIN_SOURCE_SCORE", "0.01"))


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    try:
        candidates = await asyncio.to_thread(
            graph_store.query_chunks, request.question, request.top_k
        )
        ranked = await asyncio.to_thread(
            reranker.rerank, request.question, candidates, request.top_k
        )

        # ── Fix A2: filter out irrelevant sources ─────────────────────────
        ranked = [r for r in ranked if r["score"] >= _MIN_SOURCE_SCORE]
        ranked = ranked[:request.top_k]  # cap after filtering
        # ──────────────────────────────────────────────────────────────────

        sources = [
            Source(
                text=r["text"],
                source=r["source"],
                chunk_index=r["chunk_index"],
                score=r["score"],
                type=r.get("type", "vector"),
                chunk_type=r.get("chunk_type", "text"),
                media_type=r.get("media_type"),
                image_url=r.get("image_url"),
                figure_type=r.get("figure_type"),
                caption=r.get("caption"),
                page_number=r.get("page_number"),
                timestamp_ms=r.get("timestamp_ms"),
            )
            for r in ranked
        ]

        if sources:
            answer = await asyncio.to_thread(
                llm.generate_answer, request.question, [s.text for s in sources]
            )
        else:
            # No relevant sources found — use general knowledge
            answer = await asyncio.to_thread(
                llm.generate_general_answer, request.question
            )

        return QueryResponse(answer=answer, sources=sources)

    except Exception as exc:
        logger.error("Query failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
