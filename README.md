# VisionRAG

A multimodal retrieval-augmented generation system that ingests images and videos, extracts visual understanding using configurable vision backends, stores everything in a Neo4j knowledge graph, and answers questions grounded in the uploaded content.

---

## How it works

```
INGEST
──────
Image  ──► Vision model (local or cloud) ──► text description + entities
                                                         │
Video ──────────┬─ Gemini: whole video → narrative + 10-20 timestamped events
                └─ Ollama Cloud: ffmpeg keyframes → summary + per-frame descriptions (parallel)
                                                         │
                                                         ▼
                                                   Neo4j graph
                                          (MediaChunk, VisualEntity nodes)
                                          (NEXT_CHUNK, DEPICTS, CO_OCCURS_WITH edges)

QUERY
─────
Question ──► vector search ─┐
         ──► fulltext search ├──► RRF fusion ──► cross-encoder rerank ──► LLM answer
         ──► entity traversal┘                        (top-k grounded sources)
```

---

## Supported files

| Type | Formats | Limit |
|------|---------|-------|
| Images | PNG · JPG · WEBP | 50 MB |
| Video | MP4 · MOV · AVI · MKV · WEBM | 500 MB · 3 min max |

---

## Vision backends

### Image / PDF figures `VISION_BACKEND`

| Value | Model | Notes |
|-------|-------|-------|
| `ollama` *(default)* | any `OLLAMA_VISION_MODEL` | Free, local, requires GPU |
| `paligemma` | `google/paligemma2-3b-ft-docci-448` | Free, ~3 GB download, `HF_TOKEN` required |
| `gemini` | `gemini-2.0-flash` | Free tier, `GEMINI_API_KEY` required |
| `openai` | any `OPENAI_VISION_MODEL` | Paid, `OPENAI_API_KEY` required |

### Video: chosen per upload in the UI

| Backend | How | Speed (45 s video) | Chunks |
|---------|----|-------------------|--------|
| **Gemini** | Uploads entire video; one API call returns narrative + 10–20 timestamped events | ~30–60 s | 1 summary + N events |
| **Ollama Cloud** | ffmpeg extracts 15–45 keyframes; frames described in parallel via cloud | ~1–2 min | 1 summary + 15–45 frames |

Gemini is faster. Ollama produces more sources (per-frame thumbnails, exact timestamps) and is private.

---

## Quick start

### Local

**Requirements**: Python 3.11+, Node.js 18+, Neo4j 5+, ffmpeg, Ollama

```bash
# 1. Pull models
ollama pull llama3.2                        # LLM
ollama pull qwen2.5vl:7b                    # local image vision

# ffmpeg (video support)
# Windows:  winget install ffmpeg
# macOS:    brew install ffmpeg
# Linux:    sudo apt install ffmpeg

# 2. Backend
cd backend
cp .env.example .env          # set NEO4J_PASSWORD at minimum
python -m venv .venv
.venv/Scripts/activate        # Windows: use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
uvicorn app.main:app --port 8081 --reload

# 3. Frontend
cd visionrag/frontend
npm install
npm run dev -- --port 3001
```

Open [http://localhost:3001](http://localhost:3001).

### Docker

```bash
cd visionrag
cp backend/.env.example backend/.env   # set keys as needed
docker compose up --build

# First-run model pull
docker exec visionrag-ollama ollama pull llama3.2
docker exec visionrag-ollama ollama pull qwen2.5vl:7b
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3001 |
| Backend | http://localhost:8081 |
| Neo4j Browser | http://localhost:7475 |
| Neo4j Bolt | bolt://localhost:7688 |

---

## Configuration

All settings are in `backend/.env` (copy from `backend/.env.example`).

### Required

| Variable | Description |
|----------|-------------|
| `NEO4J_PASSWORD` | Your Neo4j password |

### LLM and image vision

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `LLM_MODEL` | `llama3.2` | Model for answer generation |
| `VISION_BACKEND` | `ollama` | `ollama` · `paligemma` · `gemini` · `openai` |
| `OLLAMA_VISION_MODEL` | `qwen2.5vl:7b` | Local Ollama vision model |
| `GEMINI_API_KEY` | Required for Gemini vision or Gemini video |
| `OPENAI_API_KEY` | Required for OpenAI vision |
| `HF_TOKEN` | Required for PaliGemma |

### Video

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDEO_CLOUD_MODEL` | `qwen3-vl:235b-instruct-cloud` | Ollama Cloud model for frame descriptions |
| `VIDEO_PARALLEL_FRAMES` | `3` | Concurrent frame API calls. Increase for faster ingest; reduce if rate-limited. |
| `VIDEO_MAX_DURATION_SEC` | `180` | Max video length accepted (seconds) |
| `VIDEO_MIN_FRAMES` | `15` | Minimum keyframes extracted |
| `VIDEO_MAX_FRAMES` | `45` | Maximum keyframes extracted |
| `VIDEO_SCENE_THRESHOLD` | `0.25` | ffmpeg scene change sensitivity (lower = more frames) |
| `VIDEO_FRAME_WIDTH` | `640` | Frame JPEG width (pixels) |
| `GEMINI_VIDEO_MODEL` | `gemini-2.5-flash` | Gemini model for video analysis |
| `GEMINI_VIDEO_TIMEOUT` | `300` | Gemini API timeout (seconds) |
| `GEMINI_DAILY_LIMIT` | `1500` | Daily Gemini request cap for usage tracking |

### Retrieval

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence embedding model (384-dim) |
| `RERANK_CANDIDATE_MULTIPLIER` | `3` | `top_k × N` candidates sent to cross-encoder |
| `MIN_SOURCE_SCORE` | `0.01` | Minimum cross-encoder sigmoid score to include a source |

### Security

| Variable | Default | Description |
|----------|---------|-------------|
| `VISIONRAG_API_KEY` | *(empty)* | Require `X-API-Key` header on all API calls. Empty = dev mode (no auth). |
| `RATE_LIMIT_RPM` | `60` | Max requests per minute per IP |
| `MAX_FILE_SIZE_MB` | `100` | Max upload size (MB) |

---

## API reference

Base URL: `http://localhost:8081`

### Ingest

```
POST   /ingest?video_backend=gemini|ollama    Upload a file; returns {job_id}
GET    /jobs/{job_id}                          Poll ingest progress
GET    /files                                  List ingested files
DELETE /files/{filename}                       Delete file and all its graph data
```

### Query

```
POST /query
Body: {"question": "What is shown?", "top_k": 5}

Response:
{
  "answer": "The video shows...",
  "sources": [
    {
      "text": "...", "source": "cooking.mp4", "score": 0.87,
      "chunk_type": "frame",
      "image_url": "http://localhost:8081/figures/...",
      "timestamp_ms": 9000
    }
  ]
}
```

### Graph

```
GET  /graph                      Nodes and edges for the graph view
GET  /graph/files                Files with chunk counts
GET  /graph/stats                Aggregate stats
DELETE /graph                    Clear all data

POST /graph/entity-dedup         Merge near-duplicate entity nodes
POST /graph/build-similarity     Add VISUALLY_SIMILAR edges
```

### Health

```
GET /health

Response includes:
{
  "status": "ok",
  "neo4j": "ok",
  "ollama": "ok",
  "gemini": "configured",
  "gemini_usage": {
    "requests_today": 3,
    "daily_limit": 1500,
    "remaining": 1497,
    "date": "2026-05-29",
    "last_request": "2026-05-29T10:22:00Z"
  }
}
```

---

## Knowledge graph schema

```
(:Document)-[:CONTAINS]->(:MediaChunk)
(:MediaChunk)-[:NEXT_CHUNK]->(:MediaChunk)
(:MediaChunk)-[:DEPICTS]->(:VisualEntity)
(:VisualEntity)-[:CO_OCCURS_WITH]->(:VisualEntity)
(:MediaChunk)-[:VISUALLY_SIMILAR {score}]->(:MediaChunk)
```

| Node | Key properties |
|------|----------------|
| `Document` | `filename`, `source_type` (image/video/pdf), `chunk_count`, `ingested_at` |
| `MediaChunk` | `id`, `chunk_type` (figure/frame/video_summary/text/table), `text`, `embedding`, `image_url`, `timestamp_ms` |
| `VisualEntity` | `name`, `display_name`, `entity_type`, `mention_count` |

---

## Tech stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI + Uvicorn (Python 3.11+) |
| Graph store | Neo4j 5.x (Community Edition) |
| Embeddings | `all-MiniLM-L6-v2` via fastembed (ONNX, CPU only) |
| Reranker | `ms-marco-MiniLM-L-6-v2` via fastembed (ONNX, CPU only) |
| LLM | Ollama (local) via `/api/chat` |
| Image vision | Ollama · PaliGemma · Gemini Flash · GPT-4o |
| Video vision | Gemini 2.5 Flash (native) · Ollama Cloud 235B (frame-by-frame) |
| Frame extraction | ffmpeg (system binary, CPU only) |
| Image processing | Pillow |
| Entity matching | rapidfuzz |
| Frontend | Next.js 14 · TypeScript · Tailwind CSS · D3.js v7 |

---

## Project structure

```
visionrag/
├── backend/
│   ├── app/
│   │   ├── main.py                   FastAPI app, middleware, health endpoint
│   │   ├── schemas.py                Pydantic models
│   │   ├── routers/
│   │   │   ├── ingest.py             File upload + video pipelines (Gemini + Ollama)
│   │   │   ├── query.py              Hybrid retrieval + answer generation
│   │   │   └── graph.py             Graph admin + visualisation endpoints
│   │   └── services/
│   │       ├── graph_store.py        Neo4j store + RRF retrieval
│   │       ├── vision.py             Image description (Ollama/PaliGemma/Gemini/OpenAI)
│   │       ├── gemini_video_describer.py  Gemini native video pipeline
│   │       ├── video_describer.py    Ollama Cloud frame description
│   │       ├── frame_extractor.py    ffmpeg scene detection + keyframe extraction
│   │       ├── gemini_usage_tracker.py    Daily Gemini quota tracking
│   │       ├── llm.py                Answer generation with grounded prompts
│   │       ├── reranker.py           Cross-encoder reranking
│   │       ├── chunker.py            PDF multimodal chunking
│   │       ├── figure_extractor.py   PDF figure/table extraction
│   │       ├── entity_dedup.py       Entity deduplication + similarity edges
│   │       └── job_store.py          SQLite ingest job tracking
│   ├── eval/                         RAGAS evaluation harness
│   ├── .env.example
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app/
│   │   ├── page.tsx                  App shell, tab routing
│   │   ├── globals.css               Design tokens, layout, component styles
│   │   ├── components/
│   │   │   ├── ChatWindow.tsx        Chat with grounded source citations
│   │   │   ├── GraphView.tsx         D3 force-directed knowledge graph
│   │   │   ├── UploadPanel.tsx       Upload with progress tracking
│   │   │   ├── FilesPage.tsx         File management
│   │   │   ├── StatusPage.tsx        Service health + Gemini usage
│   │   │   └── SourcePanel.tsx       Source cards (text/figure/table/frame)
│   │   └── lib/api.ts                Typed API wrappers
│   ├── next.config.ts
│   └── Dockerfile
├── static/figures/                   Uploaded images + extracted video frames
├── docker-compose.yml
└── README.md
```

---

## Evaluation

```bash
cd visionrag/backend
python eval/ragas_eval.py --subset 3    # smoke test (3 questions)
python eval/ragas_eval.py               # full run
```

| Metric | Description |
|--------|-------------|
| `faithfulness` | Answer claims supported by retrieved context |
| `answer_relevancy` | Answer addresses the question |
| `context_precision` | Retrieved chunks are relevant |
| `context_recall` | Context covers the ground truth |
| `visual_grounding` | Visual claims grounded in frame descriptions |

---

## Post-ingest enrichment

These are optional one-time operations after ingesting new content:

```bash
# Merge near-duplicate entity names (e.g. "frying pan" vs "frying-pan")
curl -X POST http://localhost:8081/graph/entity-dedup

# Add VISUALLY_SIMILAR edges between chunks with similar embeddings
curl -X POST http://localhost:8081/graph/build-similarity
```

---

## Troubleshooting

**Neo4j won't connect**: Verify Neo4j is running and the password in `.env` is correct. Default bolt port: 7687.

**Ollama model not found**: Run `ollama pull <model>` before starting the backend.

**Gemini rate limit (429)**: Free tier allows 15 RPM. The backend retries with backoff. Check `GET /health` → `gemini_usage.remaining` for daily quota.

**ffmpeg not found**: Required for Ollama video ingest. Install via your package manager (`winget install ffmpeg` / `brew install ffmpeg` / `sudo apt install ffmpeg`).

**Video ingest slow with Ollama**: Default is 3 parallel frame requests. Increase `VIDEO_PARALLEL_FRAMES=5` in `.env` if your Ollama Cloud tier allows it. Alternatively switch to the Gemini backend (~30–60 s vs 1–2 min for short videos).

**Queries return no sources**: Check `GET /health` for Neo4j and Ollama status. Ensure the file shows "Ready" in the Files tab before querying. Lowering `MIN_SOURCE_SCORE` in `.env` can help for category-level queries.

**Empty frame descriptions**: Verify `VIDEO_CLOUD_MODEL` is accessible: `ollama run qwen3-vl:235b-instruct-cloud "describe this" --image /path/to/test.jpg`
