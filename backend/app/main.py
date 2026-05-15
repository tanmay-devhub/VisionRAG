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
from app.services import graph_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_OLLAMA_BASE_URL     = os.getenv("OLLAMA_BASE_URL",     "http://localhost:11434")
_GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY",      "")
_VISION_BACKEND      = os.getenv("VISION_BACKEND",      "ollama")
_OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:7b")
_PALIGEMMA_MODEL     = os.getenv("PALIGEMMA_MODEL",     "google/paligemma2-3b-ft-docci-448")

# Resolve FIGURES_DIR relative to the backend package root so that
# ../static/figures works whether uvicorn is launched from visionrag/backend/
# or from any other directory.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_raw_figures  = os.getenv("FIGURES_DIR", "../static/figures")
_FIGURES_DIR  = str((_BACKEND_ROOT / _raw_figures).resolve())


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(_FIGURES_DIR, exist_ok=True)
    try:
        graph_store.ping()
        logger.info("ChromaDB ready")
    except Exception as exc:
        logger.error("ChromaDB not ready: %s", exc)
        raise RuntimeError(f"ChromaDB init failed: {exc}") from exc
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
    chroma_status = "ok"
    ollama_status = "ok"
    gemini_status = "configured" if _GEMINI_API_KEY else "missing"

    try:
        graph_store.ping()
    except Exception as exc:
        logger.warning("ChromaDB health check failed: %s", exc)
        chroma_status = f"error: {exc}"

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
        "chroma":         chroma_status,
        "ollama":         ollama_status,
        "gemini":         gemini_status,
        "vision_backend": _VISION_BACKEND,
        "vision_label":   vision_label,
    }
