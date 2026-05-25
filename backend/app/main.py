# ── VisionRAG Neo4j Migration ─────────────────────────────────────────────────
# Replaces: ChromaDB lifespan ping + health key "chroma"
# Video-ready: lifespan is generic; adding video routes requires no changes here.
# ─────────────────────────────────────────────────────────────────────────────

from dotenv import load_dotenv
load_dotenv()

import os
import logging
import httpx
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routers import ingest, query, graph
from app.services import graph_store, reranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_OLLAMA_BASE_URL     = os.getenv("OLLAMA_BASE_URL",     "http://localhost:11434")
_GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY",      "")
_VISION_BACKEND      = os.getenv("VISION_BACKEND",      "ollama")
_OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:7b")
_PALIGEMMA_MODEL     = os.getenv("PALIGEMMA_MODEL",     "google/paligemma2-3b-ft-docci-448")

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_raw_figures  = os.getenv("FIGURES_DIR", "../static/figures")
_FIGURES_DIR  = str((_BACKEND_ROOT / _raw_figures).resolve())


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    os.makedirs(_FIGURES_DIR, exist_ok=True)

    # 1. SQLite job store — stdlib only, instant
    try:
        from app.services import job_store
        job_store.mark_interrupted_jobs()
        logger.info("Job store ready (SQLite)")
    except Exception as exc:
        logger.warning("Job store init failed: %s", exc)

    # 2. Neo4j eager ping — fail fast rather than silently on first request
    try:
        graph_store.ping()
        graph_store.create_indexes()
        logger.info("Neo4j ready — indexes ensured")
    except Exception as exc:
        logger.warning("Neo4j startup ping failed: %s — will retry on first request", exc)

    # 3. Warm up embedding model in background (non-blocking)
    async def _warmup_embedder():
        try:
            await asyncio.to_thread(
                lambda: list(graph_store._get_embedder().embed(["warmup"]))
            )
            logger.info("Embedder warmed up")
        except Exception as exc:
            logger.warning("Embedder warmup failed: %s", exc)

    # 4. Warm up reranker in background (non-blocking)
    async def _warmup_reranker():
        try:
            await asyncio.to_thread(reranker._get_reranker)
            logger.info("Reranker warmed up")
        except Exception as exc:
            logger.warning("Reranker warmup failed: %s", exc)

    asyncio.create_task(_warmup_embedder())
    asyncio.create_task(_warmup_reranker())

    logger.info("VisionRAG startup complete")
    yield


app = FastAPI(title="VisionRAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/figures", StaticFiles(directory=_FIGURES_DIR), name="figures")

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(graph.router)


@app.get("/health")
async def health() -> dict:
    neo4j_status  = "ok"
    ollama_status = "ok"
    gemini_status = "configured" if _GEMINI_API_KEY else "missing"

    try:
        graph_store.ping()
    except Exception as exc:
        logger.warning("Neo4j health check failed: %s", exc)
        neo4j_status = f"error: {exc}"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{_OLLAMA_BASE_URL}/api/tags")
            if resp.status_code >= 500:
                ollama_status = f"error: HTTP {resp.status_code}"
    except Exception as exc:
        logger.warning("Ollama health check failed: %s", exc)
        ollama_status = f"error: {exc}"

    if _VISION_BACKEND == "paligemma":
        vision_label = f"PaliGemma · {_PALIGEMMA_MODEL.split('/')[-1]}"
    elif _VISION_BACKEND == "gemini":
        vision_label = "Gemini Vision"
    elif _VISION_BACKEND == "openai":
        vision_label = "OpenAI Vision"
    else:
        vision_label = f"Ollama · {_OLLAMA_VISION_MODEL}"

    return {
        "status":         "ok",
        "neo4j":          neo4j_status,
        "ollama":         ollama_status,
        "gemini":         gemini_status,
        "vision_backend": _VISION_BACKEND,
        "vision_label":   vision_label,
    }
