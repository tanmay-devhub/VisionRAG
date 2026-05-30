# ── VisionRAG Phase 5: Quality + Security ─────────────────────────────────────
# File: backend/app/main.py
# Fixes: B2 (auth), B3 (rate limit), B5 (key validation), B6 (weak password), B7 (dir listing)
# ─────────────────────────────────────────────────────────────────────────────

from dotenv import load_dotenv
load_dotenv()

import os
import time
import logging
import httpx
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from app.routers import ingest, query, graph
from app.services import graph_store, reranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress noisy Windows asyncio connection-reset spam (WinError 10054).
# This is harmless: browsers routinely close idle keep-alive connections,
# and Python's ProactorEventLoop logs the resulting TCP reset as ERROR.
class _SuppressWin10054(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "WinError 10054" not in record.getMessage()

logging.getLogger("asyncio").addFilter(_SuppressWin10054())

_OLLAMA_BASE_URL     = os.getenv("OLLAMA_BASE_URL",     "http://localhost:11434")
_GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY",      "")
_VISION_BACKEND      = os.getenv("VISION_BACKEND",      "ollama")
_OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:235b-instruct-cloud")
_PALIGEMMA_MODEL     = os.getenv("PALIGEMMA_MODEL",     "google/paligemma2-3b-ft-docci-448")
_API_KEY             = os.getenv("VISIONRAG_API_KEY",   "")  # empty = auth disabled (dev mode)

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_raw_figures  = os.getenv("FIGURES_DIR", "../static/figures")
_FIGURES_DIR  = str((_BACKEND_ROOT / _raw_figures).resolve())


# ── Fix B3: Rate Limiter ──────────────────────────────────────────────────────

class RateLimiter:
    """Simple in-memory rate limiter. Per-IP, sliding window."""
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests    = max_requests
        self.window_seconds  = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds
        self._requests[client_ip] = [
            t for t in self._requests[client_ip] if t > window_start
        ]
        if len(self._requests[client_ip]) >= self.max_requests:
            return False
        self._requests[client_ip].append(now)
        return True


_rate_limiter = RateLimiter(
    max_requests=int(os.getenv("RATE_LIMIT_RPM", "60")),
    window_seconds=60,
)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    os.makedirs(_FIGURES_DIR, exist_ok=True)

    # 1. SQLite job store
    try:
        from app.services import job_store
        job_store.mark_interrupted_jobs()
        logger.info("Job store ready (SQLite)")
    except Exception as exc:
        logger.warning("Job store init failed: %s", exc)

    # 2. Neo4j eager ping
    try:
        graph_store.ping()
        graph_store.create_indexes()
        logger.info("Neo4j ready — indexes ensured")
    except Exception as exc:
        logger.warning("Neo4j startup ping failed: %s — will retry on first request", exc)

    # ── Fix B6: Weak password warning ─────────────────────────────────────
    _neo4j_pass = os.getenv("NEO4J_PASSWORD", "")
    _WEAK_PASSWORDS = {"neo4j", "password", "admin", "test", "qwerty1234", "changeme", "123456"}
    if _neo4j_pass.lower() in _WEAK_PASSWORDS:
        logger.warning(
            "⚠️  Neo4j password is weak ('%s***'). Change it in .env for production.",
            _neo4j_pass[:3],
        )

    # 3. Warm up embedding model in background
    async def _warmup_embedder():
        try:
            await asyncio.to_thread(
                lambda: list(graph_store._get_embedder().embed(["warmup"]))
            )
            logger.info("Embedder warmed up")
        except Exception as exc:
            logger.warning("Embedder warmup failed: %s", exc)

    # 4. Warm up reranker in background
    async def _warmup_reranker():
        try:
            await asyncio.to_thread(reranker._get_reranker)
            logger.info("Reranker warmed up")
        except Exception as exc:
            logger.warning("Reranker warmup failed: %s", exc)

    asyncio.create_task(_warmup_embedder())
    asyncio.create_task(_warmup_reranker())

    # 5. ffmpeg check
    try:
        import subprocess as _sp
        _ffr = _sp.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        if _ffr.returncode == 0:
            logger.info("ffmpeg available — video ingest enabled")
        else:
            logger.warning("ffmpeg not working — video ingest will fail")
    except FileNotFoundError:
        logger.warning("ffmpeg not found — video ingest disabled")
    except Exception:
        pass

    # 6. Video cloud model info
    _cloud_model = os.getenv("VIDEO_CLOUD_MODEL", "qwen3-vl:235b-instruct-cloud")
    _max_dur = os.getenv("VIDEO_MAX_DURATION_SEC", "180")
    logger.info("Video: cloud model=%s, max duration=%ss", _cloud_model, _max_dur)

    # 7. Gemini video backend check
    try:
        from app.services.gemini_video_describer import is_gemini_configured
        if is_gemini_configured():
            logger.info("Gemini video backend configured (model=%s)", os.getenv("GEMINI_VIDEO_MODEL", "gemini-2.5-flash"))
        else:
            logger.info("Gemini video backend not configured (GEMINI_API_KEY not set)")
    except Exception as exc:
        logger.warning("Gemini check failed: %s", exc)

    # ── Fix B5: API key validation warnings ───────────────────────────────
    _keys_to_check = {
        "GEMINI_API_KEY":   os.getenv("GEMINI_API_KEY", ""),
        "HF_TOKEN":         os.getenv("HF_TOKEN", ""),
    }
    for key_name, key_val in _keys_to_check.items():
        if key_val:
            masked = key_val[:4] + "..." + key_val[-4:] if len(key_val) > 8 else "****"
            logger.info("API key %s is set (%s)", key_name, masked)
        else:
            logger.info("API key %s is not set", key_name)

    # Log security mode
    if _API_KEY:
        logger.info("Authentication ENABLED (API key required)")
    else:
        logger.warning("Authentication DISABLED — set VISIONRAG_API_KEY in .env for production")

    logger.info("VisionRAG startup complete")
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="VisionRAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Fix B7: Static files with directory listing disabled ──────────────────────
app.mount("/figures", StaticFiles(directory=_FIGURES_DIR, html=False), name="figures")


# ── Fix B2: API key authentication middleware ─────────────────────────────────

@app.middleware("http")
async def check_api_key(request: Request, call_next):
    # Skip auth if no key configured (development mode)
    if not _API_KEY:
        return await call_next(request)

    # Skip auth for health check and static files
    if request.url.path in ("/health", "/docs", "/openapi.json"):
        return await call_next(request)
    if request.url.path.startswith("/figures/"):
        return await call_next(request)

    # Check API key
    provided_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if provided_key != _API_KEY:
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing API key. Set X-API-Key header."},
        )

    return await call_next(request)


# ── Fix B3: Rate limiting middleware ──────────────────────────────────────────

@app.middleware("http")
async def rate_limit(request: Request, call_next):
    # Skip rate limiting for health and static
    if request.url.path in ("/health",) or request.url.path.startswith("/figures/"):
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    if not _rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Try again later."},
        )

    return await call_next(request)


# ── Fix B7: Block /figures/ directory listing ─────────────────────────────────

@app.middleware("http")
async def block_figures_listing(request: Request, call_next):
    path = request.url.path
    if path == "/figures/" or path == "/figures":
        return JSONResponse(status_code=403, content={"detail": "Directory listing not allowed"})
    return await call_next(request)


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(graph.router)


# ── Health ────────────────────────────────────────────────────────────────────

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

    gemini_usage: dict = {}
    try:
        from app.services.gemini_usage_tracker import get_usage
        gemini_usage = get_usage()
    except Exception:
        gemini_usage = {"error": "tracker not available"}

    return {
        "status":              "ok",
        "neo4j":               neo4j_status,
        "ollama":              ollama_status,
        "gemini":              gemini_status,
        "vision_backend":      _VISION_BACKEND,
        "vision_label":        vision_label,
        "video_cloud_model":   os.getenv("VIDEO_CLOUD_MODEL", "qwen3-vl:235b-instruct-cloud"),
        "video_max_duration":  int(os.getenv("VIDEO_MAX_DURATION_SEC", "180")),
        "gemini_video":        "configured" if os.getenv("GEMINI_API_KEY") else "not_configured",
        "gemini_video_model":  os.getenv("GEMINI_VIDEO_MODEL", "gemini-2.5-flash"),
        "auth_enabled":        bool(_API_KEY),
        "rate_limit_rpm":      int(os.getenv("RATE_LIMIT_RPM", "60")),
        "gemini_usage":        gemini_usage,
    }
