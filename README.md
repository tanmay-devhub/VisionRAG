# VisionRAG

**Ask questions about your images powered by a local vision model.**

VisionRAG lets you upload images (PNG, JPG, WEBP), runs them through a configurable vision model to generate semantic descriptions, stores everything in a local ChromaDB vector store, and answers natural-language questions with the source images shown inline.

---

## Features

- **Image ingest** — upload PNG, JPG, or WEBP files; the vision model describes what it sees and the description becomes a searchable chunk
- **Two vision backends** — run fully local with Ollama or PaliGemma
- **Two-stage retrieval** — SentenceTransformer vector search followed by CrossEncoder reranking
- **Inline sources** — answers cite the source images, which render as thumbnails in the UI
- **Real-time progress** — ingest jobs report status as they complete
- **Zero external dependencies** — runs entirely on your machine

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn (Python 3.11+) |
| Vector store | ChromaDB (persistent, local) |
| Embeddings | `all-MiniLM-L6-v2` via SentenceTransformers |
| Text LLM | Ollama (`llama3.2`) via langchain-ollama |
| Vision | Ollama · PaliGemma 2 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Frontend | Next.js 14 App Router · TypeScript · Tailwind CSS |

---

## How it works

```
Upload image (PNG / JPG / WEBP)
        │
        └─► Vision model
                └─► description + figure_type + caption
                        │
                        └─► ChromaDB upsert  (all-MiniLM-L6-v2 embeddings)
                                             image saved to static/figures/

Query
        │
        ├─► ChromaDB vector search  (top_k × 4 candidates)
        ├─► CrossEncoder rerank     (top_k final results)
        └─► Ollama LLM              (answer grounded in image descriptions)
                └─► answer + sources  (with inline image thumbnails)
```

---

## Quick start: local

### Prerequisites

- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.com) installed and running

```bash
ollama pull llama3.2
```

### Backend

```bash
cd backend
cp .env.example .env

pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the URL printed in the terminal.

---

## Quick start: Docker

```bash
cp backend/.env.example backend/.env

docker compose up --build

docker exec visionrag-ollama ollama pull llama3.2
```

Open the frontend URL printed by Docker.

---

## Vision backends

Set `VISION_BACKEND` in `backend/.env`.

### `ollama` (default free, fully local)

Requires Ollama running locally or in Docker. Any vision-capable Ollama model works.

```env
VISION_BACKEND=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_VISION_MODEL=<your-vision-model>
```

### `paligemma` (free, local HuggingFace download ~3 GB)

Runs `google/paligemma2-3b-ft-docci-448` locally via `transformers`. Requires a one-time setup:

1. Accept the model license at [huggingface.co/google/paligemma2-3b-ft-docci-448](https://huggingface.co/google/paligemma2-3b-ft-docci-448)
2. Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

```env
VISION_BACKEND=paligemma
PALIGEMMA_MODEL=google/paligemma2-3b-ft-docci-448
HF_TOKEN=your_hf_token_here
```

The model downloads automatically on first ingest and is cached in `~/.cache/huggingface/`.

---

## Project structure

```
./
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI app, CORS, static mount, lifespan
│   │   ├── schemas.py            # Pydantic request/response models
│   │   ├── routers/
│   │   │   ├── ingest.py         # POST /ingest, GET /ingest/status/{id}
│   │   │   ├── query.py          # POST /query
│   │   │   └── graph.py          # GET|DELETE /graph, /graph/files, /graph/stats
│   │   └── services/
│   │       ├── vision.py         # VisionService Ollama / PaliGemma
│   │       ├── graph_store.py    # ChromaDB wrapper store, query, stats, clear
│   │       ├── llm.py            # Ollama LLM answer generation
│   │       ├── reranker.py       # CrossEncoder reranking
│   │       └── job_store.py      # In-memory ingest job tracking
│   ├── .env.example
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app/
│   │   ├── page.tsx              # Root layout Chat / Files tabs
│   │   └── components/
│   │       ├── ChatWindow.tsx    # Chat interface + source rendering
│   │       ├── UploadPanel.tsx   # File upload + ingest progress
│   │       └── SourcePanel.tsx   # Inline source display
│   ├── lib/api.ts                # Typed fetch wrappers for all backend endpoints
│   └── Dockerfile
├── static/figures/               # Uploaded images (served by backend)
├── docker-compose.yml
├── start.bat
└── README.md
```

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ingest` | Upload an image; returns `job_id` |
| `GET` | `/ingest/status/{job_id}` | Poll ingest progress |
| `GET` | `/ingest/jobs` | List all ingest jobs |
| `POST` | `/query` | Ask a question; returns answer + sources |
| `GET` | `/graph` | Chunk graph for visualization (optional `?filename=`) |
| `GET` | `/graph/files` | List ingested files with chunk counts |
| `GET` | `/graph/stats` | Aggregate stats across all ingested content |
| `DELETE` | `/graph` | Clear all ingested data from ChromaDB |
| `GET` | `/health` | Service health ChromaDB, Ollama, vision backend status |
| `GET` | `/figures/{filename}` | Serve an uploaded image |
