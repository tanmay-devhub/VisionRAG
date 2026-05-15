import asyncio
import logging
from fastapi import APIRouter, HTTPException
from app.schemas import QueryRequest, QueryResponse, Source
from app.services import graph_store, llm, reranker

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    try:
        candidates = await asyncio.to_thread(
            graph_store.query_chunks, request.question, request.top_k * 4
        )
        ranked = await asyncio.to_thread(
            reranker.rerank, request.question, candidates, request.top_k
        )

        sources = [
            Source(
                text=r["text"],
                source=r["source"],
                chunk_index=r["chunk_index"],
                score=r["score"],
                type=r.get("type", "vector"),
                chunk_type=r.get("chunk_type", "text"),
                image_url=r.get("image_url"),
                figure_type=r.get("figure_type"),
                caption=r.get("caption"),
                page_number=r.get("page_number"),
            )
            for r in ranked
        ]

        if sources:
            answer = await asyncio.to_thread(
                llm.generate_answer, request.question, [s.text for s in sources]
            )
        else:
            answer = await asyncio.to_thread(
                llm.generate_general_answer, request.question
            )

        return QueryResponse(answer=answer, sources=sources)

    except Exception as exc:
        logger.error("Query failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
