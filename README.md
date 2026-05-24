# VisionRAG

**Multimodal RAG over PDFs and images text, figures, tables, and a live knowledge graph.**

VisionRAG ingests PDF documents and images, routes each page element through a configurable vision model, stores everything in a Neo4j graph database, and answers natural-language questions with hybrid retrieval (vector + full-text + entity graph traversal). A built-in graph viewer lets you explore the extracted knowledge graph interactively.

---

## Features

- **Multimodal ingest** PDFs (text chunks, extracted figures, tables) and standalone images (PNG, JPG, WEBP)
- **Four vision backends** Ollama (local), PaliGemma (local HuggingFace), Gemini Flash, or OpenAI GPT-4o
- **Knowledge graph** entities and relationships extracted from visual content are stored in Neo4j and linked to their source chunks
- **Hybrid retrieval** vector search, full-text search, and entity graph traversal fused with Reciprocal Rank Fusion
- **CrossEncoder reranking** `ms-marco-MiniLM-L-6-v2` reranker applied before the LLM
- **Interactive graph viewer** D3 force-directed graph with per-file filtering, node labels, and search
- **Per-file management** delete individual documents and their graph nodes without clearing the whole store
- **Real-time ingest progress** SSE-based status updates while documents are being processed
- **Zero cloud dependencies** the default configuration runs entirely on your machine

---

## How it works

```
┌─────────────────────────────────────────────────────────────────┐
│  INGEST                                                         │
│                                                                 │
│  PDF / Image                                                    │
│      │                                                          │
│      ├─► Text chunks ──► fastembed (all-MiniLM-L6-v2)          │
│      │                        └─► Neo4j :MediaChunk nodes       │
│      │                                                          │
│      └─► Figures / Tables                                       │
│              └─► Vision model ──► description + entities        │
│                      └─► Neo4j :MediaChunk + :VisualEntity      │
│                              └─► :DEPICTS, :CO_OCCURS_WITH      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  QUERY                                                          │
│                                                                 │
│  Question                                                       │
│      ├─► Vector search    ─┐                                    │
│      ├─► Full-text search  ├─► RRF fusion ──► CrossEncoder      │
│      └─► Graph traversal  ─┘        └─► top-k chunks           │
│                                              └─► Ollama LLM     │
│                                                      └─► Answer │
└─────────────────────────────────────────────────────────────────┘
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
| PDF parsing | PyMuPDF + pdfplumber |
| Frontend | Next.js 14 · TypeScript · Tailwind CSS · D3.js v7 |

---

## Quick start  local

### Prerequisites

- Python 3.11+
- Node.js 18+
- Neo4j 5.x running locally [Download Neo4j Desktop](https://neo4j.com/download/) or via Docker
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

## Quick start Docker

```bash
cp backend/.env.example backend/.env   # set your API keys if needed

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

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `VISION_BACKEND` | `ollama` | Vision backend: `ollama`, `paligemma`, `gemini`, `openai` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_VISION_MODEL` | `qwen2.5vl:7b` | Vision model for Ollama backend |
| `LLM_MODEL` | `llama3.2` | Ollama model for answer generation |
| `PALIGEMMA_MODEL` | `google/paligemma2-3b-ft-docci-448` | HuggingFace model ID for PaliGemma |
| `HF_TOKEN`  | HuggingFace token (required for PaliGemma) |
| `GEMINI_API_KEY`  | Google AI Studio API key |
| `OPENAI_API_KEY`  | OpenAI API key |
| `OPENAI_VISION_MODEL` | `gpt-4o-mini` | OpenAI model for vision |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD`  | Neo4j password (required) |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | fastembed model for chunk embeddings |
| `FIGURES_DIR` | `../static/figures` | Directory for extracted figure images |
| `JOB_STORE_PATH` | `./jobs.db` | SQLite path for ingest job tracking |

---

## Project structure

```
./
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app, CORS, lifespan, health endpoint
│   │   ├── schemas.py               # Pydantic request/response models
│   │   ├── routers/
│   │   │   ├── ingest.py            # POST /ingest, GET /ingest/status/{id}, DELETE /ingest/{filename}
│   │   │   ├── query.py             # POST /query
│   │   │   └── graph.py             # GET|DELETE /graph, /graph/files, /graph/stats
│   │   └── services/
│   │       ├── vision.py            # VisionService dispatches to ollama/paligemma/gemini/openai
│   │       ├── graph_store.py       # Neo4j store, hybrid retrieval, graph export
│   │       ├── llm.py               # Ollama LLM answer generation
│   │       ├── reranker.py          # CrossEncoder reranking via fastembed
│   │       └── job_store.py         # SQLite-backed ingest job tracking
│   ├── .env.example
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app/
│   │   ├── page.tsx                 # Root page Chat / Graph / Files tabs
│   │   ├── layout.tsx
│   │   ├── globals.css
│   │   └── components/
│   │       ├── ChatWindow.tsx       # Chat interface with markdown + inline sources
│   │       ├── GraphView.tsx        # D3 force-directed knowledge graph
│   │       ├── UploadPanel.tsx      # File upload with real-time ingest progress
│   │       └── SourcePanel.tsx      # Expandable source citations
│   ├── lib/
│   │   └── api.ts                   # Typed fetch wrappers for all backend endpoints
│   └── Dockerfile
├── static/
│   └── figures/                     # Extracted figure images (served by backend)
├── docker-compose.yml
├── start.bat
└── README.md
```

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service status Neo4j, Ollama, vision backend |
| `POST` | `/ingest` | Upload a file (PDF or image); returns `job_id` |
| `GET` | `/ingest/status/{job_id}` | Poll ingest progress and chunk counts |
| `GET` | `/ingest/jobs` | List all ingest jobs |
| `DELETE` | `/ingest/{filename}` | Delete one file and all its graph nodes |
| `POST` | `/query` | Ask a question; returns `answer` + `sources` |
| `GET` | `/graph` | Full graph data for visualisation (`?filename=` to filter) |
| `GET` | `/graph/files` | List ingested files with chunk/figure/table counts |
| `GET` | `/graph/stats` | Aggregate stats across the entire graph |
| `DELETE` | `/graph` | Clear all ingested data |
| `GET` | `/figures/{filename}` | Serve an extracted figure image |

---

## Graph schema

```
(:Document)-[:CONTAINS]->(:MediaChunk)
(:MediaChunk)-[:NEXT_CHUNK]->(:MediaChunk)
(:MediaChunk)-[:DEPICTS]->(:VisualEntity)
(:VisualEntity)-[:CO_OCCURS_WITH]->(:VisualEntity)
```

| Node label | Key properties |
|---|---|
| `Document` | `filename`, `source_type`, `chunk_count`, `ingested_at` |
| `MediaChunk` | `id`, `filename`, `chunk_type` (`text`/`figure`/`table`), `text`, `embedding`, `image_url` |
| `VisualEntity` | `name`, `display_name`, `entity_type`, `mention_count` |
