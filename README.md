# VisionRAG

**Multimodal RAG with a live knowledge graph: upload images and videos, ask questions, explore extracted entities.**

VisionRAG ingests images and short videos, routes them through vision models (local for images, cloud for video), stores descriptions and entities in a Neo4j graph database, and answers natural-language questions with hybrid retrieval (vector, full-text, entity graph traversal and visual similarity). A built-in graph viewer lets you explore the extracted knowledge graph interactively.

---

## Features

- **Image ingest**: PNG, JPG, and WEBP images; auto-resized to max 1280 px before processing
- **Video ingest**: MP4, MOV, AVI, MKV, WEBM videos up to 3 minutes; keyframes extracted via ffmpeg, described by Ollama Cloud (235B model)
- **Video summary**: all frames sent in one cloud call for a chronological narrative with key events
- **Four image vision backends**: Ollama (local GPU), PaliGemma (local HuggingFace), Gemini Flash, or OpenAI GPT-4o
- **Separate video pipeline**: Ollama Cloud `qwen3-vl:235b-instruct-cloud` — no local GPU used for video
- **Object detection awareness**: vision prompt describes annotated objects, not annotation colours
- **Knowledge graph**: entities and relationships stored in Neo4j, linked to source media
- **Entity deduplication**: on-demand fuzzy merge of near-duplicate `VisualEntity` nodes
- **Visual similarity edges**: `VISUALLY_SIMILAR` edges connect chunks with cosine-similar embeddings
- **Hybrid retrieval**: vector + full-text + entity graph traversal fused with Reciprocal Rank Fusion
- **CrossEncoder reranking**: `ms-marco-MiniLM-L-6-v2` reranker applied before the LLM
- **Grounded answer generation**: partial relevance handling — never fabricates story details from frames
- **Interactive graph viewer**: D3 force-directed graph with per-file filtering, node labels, and search
- **Per-file management**: delete individual files and all their graph nodes
- **Real-time ingest progress**: polling-based status updates during processing
- **Cancel upload**: cancel in-progress uploads and delete all associated data
- **RAGAS evaluation harness**: 5 LLM-as-judge metrics including visual grounding
- **Duration guard**: videos over 3 minutes rejected at upload with clear error message

---

## How it works

```
+----------------------------------------------------------------------+
|  IMAGE INGEST (local GPU)                                            |
|                                                                      |
|  Image (PNG / JPG / WEBP)                                            |
|      +-> Auto-resize (max 1280 px) -> saved to static/figures/      |
|      +-> Local Ollama (qwen3-vl:8b) -> description + entities       |
|              +-> Neo4j :MediaChunk (chunk_type="figure")             |
|                      +-> :DEPICTS, :CO_OCCURS_WITH                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
|  VIDEO INGEST (Ollama Cloud — no local GPU)                          |
|                                                                      |
|  Video (MP4 / MOV / AVI / MKV / WEBM, max 3 min)                    |
|      +-> Duration check (reject if > 180s)                           |
|      +-> ffmpeg scene detection -> 15-45 keyframe timestamps         |
|      +-> Extract JPEG frames at 640px (CPU only)                     |
|      +-> Ollama Cloud (qwen3-vl:235b) /api/chat:                     |
|          +-> ALL frames in one call -> video_summary chunk           |
|          +-> Each frame individually -> per-frame descriptions       |
|              +-> Neo4j :MediaChunk (chunk_type="video_summary"|"frame")
|                      +-> :DEPICTS, :CO_OCCURS_WITH                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
|  QUERY                                                               |
|                                                                      |
|  Question                                                            |
|      +-> Vector search          -+                                   |
|      +-> Full-text search        +-> RRF fusion -> CrossEncoder      |
|      +-> Graph traversal        -+       +-> top-k chunks           |
|              +-> DEPICTS                         +-> Ollama LLM      |
|              +-> CO_OCCURS_WITH                          +-> Answer  |
|              +-> VISUALLY_SIMILAR                                    |
+----------------------------------------------------------------------+
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn (Python 3.11+) |
| Graph store | Neo4j 5.x (Community Edition) |
| Embeddings | `all-MiniLM-L6-v2` via fastembed (ONNX, no PyTorch) |
| Reranker | `Xenova/ms-marco-MiniLM-L-6-v2` via fastembed (ONNX) |
| Text LLM | Ollama (`llama3.2`) via langchain-ollama |
| Image vision | Local Ollama (`qwen3-vl:8b`) · PaliGemma 2 · Gemini Flash · GPT-4o |
| Video vision | Ollama Cloud (`qwen3-vl:235b-instruct-cloud`) via `/api/chat` |
| Frame extraction | ffmpeg (system binary, CPU only) |
| Fuzzy matching | rapidfuzz (entity deduplication) |
| Image processing | Pillow (resize, format normalisation) |
| Evaluation | Custom RAGAS harness with Ollama as LLM judge |
| Frontend | Next.js 14 · TypeScript · Tailwind CSS · D3.js v7 |

---

## Quick start (local)

### Prerequisites

- Python 3.11+
- Node.js 18+
- Neo4j 5.x running locally ([Download Neo4j Desktop](https://neo4j.com/download/) or via Docker)
- [Ollama](https://ollama.com) installed and running
- ffmpeg installed (for video support)

```bash
# Models
ollama pull llama3.2                        # LLM for answers
ollama pull qwen3-vl:8b                     # local vision for images
ollama pull qwen3-vl:235b-instruct-cloud    # cloud vision for video

# ffmpeg (pick your OS)
# Windows: winget install ffmpeg
# Mac: brew install ffmpeg
# Linux: sudo apt install ffmpeg
```

### Backend

```bash
cd backend
cp .env.example .env       # fill in NEO4J_PASSWORD at minimum

pip install -r requirements.txt
uvicorn app.main:app --port 8081 --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev -- --port 3001
```

Open [http://localhost:3001](http://localhost:3001).

---

## Quick start (Docker)

```bash
cp backend/.env.example backend/.env   # set API keys if needed

docker compose up --build

# Pull models into the Ollama container (first run only)
docker exec visionrag-ollama ollama pull llama3.2
docker exec visionrag-ollama ollama pull qwen3-vl:8b
docker exec visionrag-ollama ollama pull qwen3-vl:235b-instruct-cloud
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3001 |
| Backend API | http://localhost:8081 |
| Neo4j Browser | http://localhost:7474 |

---

## Image vision backends

Configure `VISION_BACKEND` in `backend/.env` (images only — video always uses cloud):

| Value | Model | Requirement |
|---|---|---|
| `ollama` *(default)* | `qwen3-vl:8b` (or any `OLLAMA_VISION_MODEL`) | Ollama running locally + GPU |
| `paligemma` | `google/paligemma2-3b-ft-docci-448` | `HF_TOKEN` + license accepted |
| `gemini` | `gemini-2.0-flash` | `GEMINI_API_KEY` (free tier) |
| `openai` | `gpt-4o-mini` (or any `OPENAI_VISION_MODEL`) | `OPENAI_API_KEY` (paid) |

---

## Video support

### How it works

1. Upload a video (max 3 minutes, any common format)
2. ffmpeg detects scene changes and extracts 15–45 keyframes at 640px
3. All frames are sent to Ollama Cloud (`qwen3-vl:235b-instruct-cloud`) in one call for a narrative summary
4. Each frame is also described individually for fine-grained retrieval
5. Both the summary and per-frame descriptions are stored in Neo4j

### Why Ollama Cloud?

- The 235B model produces far better descriptions than any 8B local model
- No local GPU VRAM is consumed — video processing is entirely remote
- Your local GPU stays free for image ingest

### Video constraints

| Constraint | Value | Configurable |
|---|---|---|
| Max duration | 3 minutes (180s) | `VIDEO_MAX_DURATION_SEC` |
| Min frames | 15 | `VIDEO_MIN_FRAMES` |
| Max frames | 45 | `VIDEO_MAX_FRAMES` |
| Frame width | 640px | `VIDEO_FRAME_WIDTH` |
| Scene threshold | 0.25 | `VIDEO_SCENE_THRESHOLD` |

### Critical: /api/chat not /api/generate

Ollama Cloud's `/api/generate` endpoint silently ignores images. Video descriptions use `/api/chat` with the messages format. This is handled automatically by `video_describer.py`.

---

## Post-ingest graph enrichment

After uploading images or videos, two optional steps improve retrieval quality:

```bash
# 1. Merge near-duplicate entity nodes
curl -X POST "http://localhost:8081/graph/entity-dedup"

# 2. Build visual similarity edges
curl -X POST "http://localhost:8081/graph/build-similarity"
```

---

## Evaluation

```bash
cd backend
python eval/ragas_eval.py --subset 3    # smoke test
python eval/ragas_eval.py               # full run
```

| Metric | What it measures |
|---|---|
| `faithfulness` | Are answer claims supported by retrieved context? |
| `answer_relevancy` | Does the answer address the question? |
| `context_precision` | Are the retrieved chunks relevant? |
| `context_recall` | Does context cover the ground truth? |
| `visual_grounding` | Are visual claims grounded in descriptions? |

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `VISION_BACKEND` | `ollama` | Image vision backend |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_VISION_MODEL` | `qwen3-vl:8b` | Local vision model for images |
| `LLM_MODEL` | `llama3.2` | Ollama model for answer generation |
| `VIDEO_CLOUD_MODEL` | `qwen3-vl:235b-instruct-cloud` | Cloud model for video |
| `VIDEO_MAX_DURATION_SEC` | `180` | Max video duration (seconds) |
| `VIDEO_MAX_FRAMES` | `45` | Max frames extracted per video |
| `VIDEO_MIN_FRAMES` | `15` | Min frames (uniform fill if fewer) |
| `VIDEO_FRAME_WIDTH` | `640` | Frame extraction width (pixels) |
| `VIDEO_SCENE_THRESHOLD` | `0.25` | ffmpeg scene detection sensitivity |
| `VIDEO_REQUEST_TIMEOUT` | `300` | Cloud API timeout per request |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | *(required)* | Neo4j password |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `FIGURES_DIR` | `../static/figures` | Storage for images and frames |
| `FIGURES_SERVE_URL` | `http://localhost:8081/figures` | Public URL for stored files |
| `JOB_STORE_PATH` | `./jobs.db` | SQLite job tracking path |
| `RERANK_CANDIDATE_MULTIPLIER` | `3` | Candidates for CrossEncoder |

---

## Project structure

```
./
+-- backend/
|   +-- app/
|   |   +-- main.py                  # FastAPI app, CORS, lifespan, health
|   |   +-- schemas.py               # Pydantic request/response models
|   |   +-- routers/
|   |   |   +-- ingest.py            # POST /ingest (image + video), DELETE /ingest/{filename}
|   |   |   +-- query.py             # POST /query
|   |   |   +-- graph.py             # Graph admin endpoints
|   |   +-- services/
|   |       +-- vision.py            # IMAGE ONLY: local Ollama/PaliGemma/Gemini/OpenAI
|   |       +-- video_describer.py   # VIDEO ONLY: Ollama Cloud /api/chat
|   |       +-- frame_extractor.py   # VIDEO: ffmpeg scene detection + frame extraction
|   |       +-- graph_store.py       # Neo4j store + hybrid retrieval
|   |       +-- entity_dedup.py      # Entity dedup + visual similarity
|   |       +-- llm.py              # Answer generation (grounded prompt)
|   |       +-- reranker.py          # CrossEncoder reranking
|   |       +-- chunker.py           # PDF multimodal chunking
|   |       +-- figure_extractor.py  # PDF figure/table extraction
|   |       +-- job_store.py         # SQLite job tracking
|   +-- eval/
|   +-- .env.example
|   +-- requirements.txt
|   +-- Dockerfile
+-- frontend/
|   +-- app/
|   |   +-- page.tsx                 # Chat / Graph / Files tabs
|   |   +-- components/
|   |       +-- ChatWindow.tsx       # Chat with source citations
|   |       +-- GraphView.tsx        # D3 knowledge graph
|   |       +-- UploadPanel.tsx      # Upload with progress + cancel
|   |       +-- SourcePanel.tsx      # Source cards (figure/table/text/frame)
|   +-- lib/api.ts                   # Typed API wrappers
+-- static/figures/                  # Stored images and video frames
+-- docker-compose.yml
+-- start.bat
+-- README.md
```

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service status + video cloud model info |
| `POST` | `/ingest` | Upload image or video (max 3 min); returns `job_id` |
| `GET` | `/ingest/status/{job_id}` | Poll ingest progress |
| `GET` | `/ingest/jobs` | List all ingest jobs |
| `DELETE` | `/ingest/{filename}` | Delete file and all graph data |
| `POST` | `/query` | Ask a question; returns `answer` + `sources` |
| `GET` | `/graph` | Graph data for visualisation |
| `GET` | `/graph/files` | List ingested files with counts |
| `GET` | `/graph/stats` | Aggregate stats |
| `DELETE` | `/graph` | Clear all data |
| `POST` | `/graph/entity-dedup` | Merge duplicate entities |
| `POST` | `/graph/build-similarity` | Build similarity edges |
| `GET` | `/figures/{filename}` | Serve stored image/frame |

---

## Graph schema

```
(:Document)-[:CONTAINS]->(:MediaChunk)
(:MediaChunk)-[:NEXT_CHUNK]->(:MediaChunk)
(:MediaChunk)-[:DEPICTS]->(:VisualEntity)
(:VisualEntity)-[:CO_OCCURS_WITH]->(:VisualEntity)
(:MediaChunk)-[:VISUALLY_SIMILAR {score: float}]->(:MediaChunk)
```

| Node label | Key properties |
|---|---|
| `Document` | `filename`, `source_type` (`image`/`video`/`pdf`), `chunk_count`, `ingested_at` |
| `MediaChunk` | `id`, `filename`, `chunk_type` (`figure`/`frame`/`video_summary`/`text`/`table`), `text`, `embedding`, `image_url`, `timestamp_ms` |
| `VisualEntity` | `name`, `display_name`, `entity_type`, `mention_count` |

| Relationship | Properties | Description |
|---|---|---|
| `CONTAINS` | - | Document owns a chunk |
| `NEXT_CHUNK` | - | Sequential order within a document |
| `DEPICTS` | - | Chunk references a visual entity |
| `CO_OCCURS_WITH` | - | Two entities co-occur in same chunk |
| `VISUALLY_SIMILAR` | `score` (0-1) | Chunks with similar embeddings |
