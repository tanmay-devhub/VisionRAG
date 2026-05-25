# VisionRAG

**Image RAG with a live knowledge graph: upload images, ask questions, explore extracted entities.**

VisionRAG ingests images, routes each one through a configurable vision model, stores descriptions and entities in a Neo4j graph database, and answers natural-language questions with hybrid retrieval (vector + full-text + entity graph traversal + visual similarity). A built-in graph viewer lets you explore the extracted knowledge graph interactively.

---

## Features

- **Image ingest**: PNG, JPG, and WEBP images; all automatically resized to max 1280 px before processing
- **Four vision backends**: Ollama (local), PaliGemma (local HuggingFace), Gemini Flash, or OpenAI GPT-4o
- **Object detection awareness**: vision prompt instructs the model to describe annotated objects rather than annotation fill colours, improving entity extraction from labelled datasets
- **Knowledge graph**: entities and relationships extracted from visual content are stored in Neo4j and linked to their source images
- **Entity deduplication**: on-demand fuzzy merge of near-duplicate `VisualEntity` nodes using rapidfuzz token-sort ratio
- **Visual similarity edges**: `VISUALLY_SIMILAR` edges connect image chunks with cosine-similar embeddings, enabling cross-image retrieval
- **Hybrid retrieval**: vector search, full-text search, and entity graph traversal (including `VISUALLY_SIMILAR` paths) fused with Reciprocal Rank Fusion
- **CrossEncoder reranking**: `ms-marco-MiniLM-L-6-v2` reranker applied before the LLM
- **Context-aware answer generation**: uploaded images are the primary source; general knowledge fills gaps only, with explicit source attribution
- **Interactive graph viewer**: D3 force-directed graph with per-file filtering, node labels, and search
- **Per-file management**: delete individual images and all their graph nodes without clearing the whole store
- **Real-time ingest progress**: SSE-based status updates while images are processed
- **RAGAS evaluation harness**: 5 LLM-as-judge metrics (faithfulness, answer relevancy, context precision, context recall, visual grounding)
- **Zero cloud dependencies**: the default configuration runs entirely on your machine

---

## How it works

```
+----------------------------------------------------------------------+
|  INGEST                                                              |
|                                                                      |
|  Image (PNG / JPG / WEBP)                                            |
|      |                                                               |
|      +-> Auto-resize (max 1280 px) -> saved to static/figures/      |
|      |                                                               |
|      +-> Vision model -> description + entities                      |
|              +-> Neo4j :MediaChunk + :VisualEntity                   |
|                      +-> :DEPICTS, :CO_OCCURS_WITH                   |
|                                                                      |
|  POST-INGEST (on demand)                                             |
|      +-> /graph/entity-dedup    -> merge near-duplicate entities     |
|      +-> /graph/build-similarity -> :VISUALLY_SIMILAR edges         |
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
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` via fastembed |
| Text LLM | Ollama (`llama3.2`) via langchain-ollama |
| Vision | Ollama · PaliGemma 2 · Gemini Flash · GPT-4o |
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

```bash
ollama pull llama3.2
ollama pull qwen2.5vl:7b   # default vision model
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
docker exec visionrag-ollama ollama pull qwen2.5vl:7b
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3001 |
| Backend API | http://localhost:8081 |
| Neo4j Browser | http://localhost:7474 |

---

## Vision backends

Configure `VISION_BACKEND` in `backend/.env`:

| Value | Model | Requirement |
|---|---|---|
| `ollama` *(default)* | `qwen2.5vl:7b` (or any `OLLAMA_VISION_MODEL`) | Ollama running locally |
| `paligemma` | `google/paligemma2-3b-ft-docci-448` | `HF_TOKEN` + license accepted (free, ~3 GB download) |
| `gemini` | `gemini-2.0-flash` | `GEMINI_API_KEY` (free tier at aistudio.google.com) |
| `openai` | `gpt-4o-mini` (or any `OPENAI_VISION_MODEL`) | `OPENAI_API_KEY` (paid) |

### PaliGemma setup (one-time)

1. Accept the model license at [huggingface.co/google/paligemma2-3b-ft-docci-448](https://huggingface.co/google/paligemma2-3b-ft-docci-448)
2. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Set `HF_TOKEN=<your-token>` and `VISION_BACKEND=paligemma` in `.env`

The model downloads automatically on first ingest and is cached in `~/.cache/huggingface/`.

---

## Post-ingest graph enrichment

After uploading images, two optional steps improve retrieval quality:

```bash
# 1. Merge near-duplicate entity nodes (run after every new ingest)
curl -X POST "http://localhost:8081/graph/entity-dedup"

# 2. Build visual similarity edges between image chunks
curl -X POST "http://localhost:8081/graph/build-similarity"

# Scope either step to a single file
curl -X POST "http://localhost:8081/graph/entity-dedup?filename=photo.png"
curl -X POST "http://localhost:8081/graph/build-similarity?threshold=0.75"
```

**Entity dedup** uses `rapidfuzz.token_sort_ratio` to find names that are the same concept written differently (e.g. "bar chart" vs "a bar chart"), merges them into a single canonical node, and re-points all `DEPICTS` and `CO_OCCURS_WITH` edges.

**Visual similarity** computes cosine similarity across all image chunk embeddings and creates `VISUALLY_SIMILAR` edges for pairs above the threshold (default `0.82`). These edges add a fourth retrieval path in `query_chunks`, surfacing visually similar images that keyword and entity search would miss.

---

## Evaluation

A RAGAS-style evaluation harness is included at `backend/eval/ragas_eval.py`. It uses Ollama as an LLM judge and requires no external evaluation libraries.

```bash
cd backend

# Smoke test (3 questions)
python eval/ragas_eval.py --subset 3

# Full run (saves to eval/baseline_before_video.json)
python eval/ragas_eval.py

# Custom output path
python eval/ragas_eval.py --output eval/my_run.json
```

### Metrics

| Metric | What it measures |
|---|---|
| `faithfulness` | Are answer claims supported by retrieved context? |
| `answer_relevancy` | Does the answer address the question? |
| `context_precision` | Are the retrieved chunks relevant to the question? |
| `context_recall` | Does the retrieved context cover the ground truth? |
| `visual_grounding` | Are visual claims in the answer grounded in image descriptions? |

Scores range from `0.0` to `1.0`. Edit `backend/eval/golden_dataset.json` to add questions about your specific uploaded images before running.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `VISION_BACKEND` | `ollama` | Vision backend: `ollama`, `paligemma`, `gemini`, `openai` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_VISION_MODEL` | `qwen2.5vl:7b` | Vision model for Ollama backend |
| `LLM_MODEL` | `llama3.2` | Ollama model for answer generation |
| `PALIGEMMA_MODEL` | `google/paligemma2-3b-ft-docci-448` | HuggingFace model ID for PaliGemma |
| `HF_TOKEN` | *(required for PaliGemma)* | HuggingFace token |
| `GEMINI_API_KEY` | *(required for Gemini)* | Google AI Studio API key |
| `OPENAI_API_KEY` | *(required for OpenAI)* | OpenAI API key |
| `OPENAI_VISION_MODEL` | `gpt-4o-mini` | OpenAI model for vision |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | *(required)* | Neo4j password |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | fastembed model for chunk embeddings |
| `FIGURES_DIR` | `../static/figures` | Directory for stored image files |
| `FIGURES_SERVE_URL` | `http://localhost:8081/figures` | Public URL prefix for image files |
| `JOB_STORE_PATH` | `./jobs.db` | SQLite path for ingest job tracking |
| `ENTITY_DEDUP_THRESHOLD` | `88` | rapidfuzz score threshold for entity merging (0-100) |
| `VISUAL_SIM_THRESHOLD` | `0.82` | Cosine similarity cutoff for `VISUALLY_SIMILAR` edges |

---

## Project structure

```
./
+-- backend/
|   +-- app/
|   |   +-- main.py                  # FastAPI app, CORS, lifespan, health endpoint
|   |   +-- schemas.py               # Pydantic request/response models
|   |   +-- routers/
|   |   |   +-- ingest.py            # POST /ingest (with auto image resize), DELETE /ingest/{filename}
|   |   |   +-- query.py             # POST /query
|   |   |   +-- graph.py             # GET|DELETE /graph, /graph/files, /graph/stats,
|   |   |                            # POST /graph/entity-dedup, POST /graph/build-similarity
|   |   +-- services/
|   |       +-- vision.py            # VisionService: ollama/paligemma/gemini/openai,
|   |       |                        # serialised Ollama calls, retry with backoff,
|   |       |                        # object detection annotation prompt
|   |       +-- graph_store.py       # Neo4j store, hybrid retrieval (+ VISUALLY_SIMILAR path),
|   |       |                        # graph export, updated stats
|   |       +-- entity_dedup.py      # Entity dedup (rapidfuzz) + visual similarity builder
|   |       +-- llm.py               # Ollama LLM: context-first prompt, general knowledge fallback
|   |       +-- reranker.py          # CrossEncoder reranking via fastembed
|   |       +-- job_store.py         # SQLite-backed ingest job tracking
|   +-- eval/
|   |   +-- ragas_eval.py            # RAGAS evaluation harness (Ollama as judge)
|   |   +-- golden_dataset.json      # Ground-truth Q&A pairs for evaluation
|   |   +-- requirements.txt         # Eval-only dependencies
|   +-- .env.example
|   +-- requirements.txt
|   +-- Dockerfile
+-- frontend/
|   +-- app/
|   |   +-- page.tsx                 # Root page: Chat / Graph / Files tabs
|   |   +-- layout.tsx
|   |   +-- globals.css
|   |   +-- components/
|   |       +-- ChatWindow.tsx       # Chat interface with markdown + inline sources
|   |       +-- GraphView.tsx        # D3 force-directed knowledge graph (full-page tab)
|   |       +-- UploadPanel.tsx      # Image upload with real-time ingest progress
|   |       +-- SourcePanel.tsx      # Expandable source citations
|   +-- lib/
|   |   +-- api.ts                   # Typed fetch wrappers for all backend endpoints
|   +-- Dockerfile
+-- static/
|   +-- figures/                     # Stored and resized image files (served by backend)
+-- docker-compose.yml
+-- start.bat
+-- README.md
```

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service status: Neo4j, Ollama, vision backend |
| `POST` | `/ingest` | Upload an image (PNG, JPG, WEBP); auto-resizes to max 1280 px; returns `job_id` |
| `GET` | `/ingest/status/{job_id}` | Poll ingest progress and chunk counts |
| `GET` | `/ingest/jobs` | List all ingest jobs |
| `DELETE` | `/ingest/{filename}` | Delete one image and all its graph nodes |
| `POST` | `/query` | Ask a question; returns `answer` + `sources` |
| `GET` | `/graph` | Full graph data for visualisation (`?filename=` to filter) |
| `GET` | `/graph/files` | List ingested images with chunk and entity counts |
| `GET` | `/graph/stats` | Aggregate stats including `visually_similar_edges` count |
| `DELETE` | `/graph` | Clear all ingested data |
| `POST` | `/graph/entity-dedup` | Merge near-duplicate VisualEntity nodes (`?filename=` to scope) |
| `POST` | `/graph/build-similarity` | Build VISUALLY_SIMILAR edges (`?filename=`, `?threshold=`) |
| `GET` | `/figures/{filename}` | Serve a stored image file |

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
| `Document` | `filename`, `source_type`, `chunk_count`, `ingested_at` |
| `MediaChunk` | `id`, `filename`, `chunk_type` (`figure`/`frame`), `text`, `embedding`, `image_url` |
| `VisualEntity` | `name`, `display_name`, `entity_type`, `mention_count` |

| Relationship | Properties | Description |
|---|---|---|
| `CONTAINS` | - | Document owns a chunk |
| `NEXT_CHUNK` | - | Sequential order within a document |
| `DEPICTS` | - | Chunk references a visual entity |
| `CO_OCCURS_WITH` | - | Two entities appear together in the same chunk |
| `VISUALLY_SIMILAR` | `score` (float 0-1) | Two image chunks have similar embeddings |
