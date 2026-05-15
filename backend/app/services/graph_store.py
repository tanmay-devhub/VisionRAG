import os
import logging
from datetime import datetime, timezone
from typing import Callable

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)

_CHROMA_DIR  = os.getenv("CHROMA_DIR",  "./chroma_db")
_EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

_client: chromadb.PersistentClient | None = None
_ef: SentenceTransformerEmbeddingFunction | None = None


def _get_ef() -> SentenceTransformerEmbeddingFunction:
    global _ef
    if _ef is None:
        _ef = SentenceTransformerEmbeddingFunction(model_name=_EMBED_MODEL)
    return _ef


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=_CHROMA_DIR)
    return _client


def _get_col() -> chromadb.Collection:
    return _get_client().get_or_create_collection(
        name="chunks",
        embedding_function=_get_ef(),
        metadata={"hnsw:space": "cosine"},
    )


def ping() -> None:
    _get_client().heartbeat()


# ── ingest ────────────────────────────────────────────────────────────────────

def store_chunks(
    chunks: list[dict],
    filename: str,
    progress_cb: Callable[[int], None] | None = None,
) -> tuple[int, int]:
    col = _get_col()
    now = datetime.now(timezone.utc).isoformat()

    ids:       list[str]  = []
    texts:     list[str]  = []
    metadatas: list[dict] = []

    for i, chunk in enumerate(chunks):
        ids.append(f"{filename}_{chunk['chunk_index']}")
        texts.append(chunk["text"])
        metadatas.append({
            "filename":    filename,
            "chunk_index": chunk["chunk_index"],
            "chunk_type":  chunk.get("chunk_type", "text"),
            "image_path":  chunk.get("image_path")  or "",
            "image_url":   chunk.get("image_url")   or "",
            "figure_type": chunk.get("figure_type") or "",
            "caption":     chunk.get("caption")     or "",
            "page_number": chunk["page_number"] if chunk.get("page_number") is not None else -1,
            "ingested_at": now,
        })
        if progress_cb:
            progress_cb(i + 1)

    _BATCH = 500
    for start in range(0, len(ids), _BATCH):
        col.upsert(
            ids=ids[start : start + _BATCH],
            documents=texts[start : start + _BATCH],
            metadatas=metadatas[start : start + _BATCH],
        )

    return 0, 0


# ── retrieval ─────────────────────────────────────────────────────────────────

def query_chunks(question: str, top_k: int) -> list[dict]:
    col   = _get_col()
    count = col.count()
    if count == 0:
        return []

    try:
        results = col.query(
            query_texts=[question],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.error("ChromaDB query failed: %s", exc)
        return []

    chunks: list[dict] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        pn = meta.get("page_number")
        chunks.append({
            "text":        doc,
            "source":      meta["filename"],
            "chunk_index": int(meta["chunk_index"]),
            "score":       max(0.0, 1.0 - float(dist)),
            "type":        "vector",
            "chunk_type":  meta.get("chunk_type", "text"),
            "image_url":   meta.get("image_url")   or None,
            "figure_type": meta.get("figure_type") or None,
            "caption":     meta.get("caption")     or None,
            "page_number": None if pn == -1 else pn,
        })
    return chunks


# ── admin / visualisation ─────────────────────────────────────────────────────

def get_ingested_files() -> dict:
    col = _get_col()
    if col.count() == 0:
        return {"files": []}

    data = col.get(include=["metadatas"])
    file_stats: dict[str, dict] = {}

    for meta in data["metadatas"]:
        fn = meta["filename"]
        if fn not in file_stats:
            file_stats[fn] = {
                "filename":    fn,
                "chunks":      0,
                "figures":     0,
                "tables":      0,
                "ingested_at": meta.get("ingested_at"),
            }
        file_stats[fn]["chunks"] += 1
        ct = meta.get("chunk_type", "text")
        if ct == "figure":
            file_stats[fn]["figures"] += 1
        elif ct == "table":
            file_stats[fn]["tables"]  += 1

    return {"files": sorted(file_stats.values(), key=lambda x: x["filename"])}


def get_all_graph_data(filename: str | None = None) -> dict:
    col = _get_col()
    if col.count() == 0:
        return {"nodes": [], "links": []}

    try:
        if filename:
            data = col.get(where={"filename": filename}, include=["metadatas"])
        else:
            data = col.get(include=["metadatas"], limit=500)
    except Exception as exc:
        logger.error("get_all_graph_data failed: %s", exc)
        return {"nodes": [], "links": []}

    nodes: dict[str, dict] = {}
    file_chunks: dict[str, list[tuple[int, str]]] = {}

    for chunk_id, meta in zip(data["ids"], data["metadatas"]):
        nodes[chunk_id] = {
            "id":         chunk_id,
            "label":      f"Chunk {meta['chunk_index']}",
            "filename":   meta["filename"],
            "nodeType":   "chunk",
            "chunk_type": meta.get("chunk_type", "text"),
        }
        fn = meta["filename"]
        file_chunks.setdefault(fn, []).append((int(meta["chunk_index"]), chunk_id))

    links: list[dict] = []
    for fn, pairs in file_chunks.items():
        pairs.sort(key=lambda x: x[0])
        for i in range(len(pairs) - 1):
            links.append({
                "source": pairs[i][1],
                "target": pairs[i + 1][1],
                "type":   "NEXT_CHUNK",
            })

    return {"nodes": list(nodes.values()), "links": links}


def get_graph_stats() -> dict:
    col = _get_col()
    if col.count() == 0:
        return {"total_chunks": 0, "figures": 0, "tables": 0, "text_chunks": 0, "entities": 0}

    metas = col.get(include=["metadatas"])["metadatas"]
    return {
        "total_chunks": len(metas),
        "figures":      sum(1 for m in metas if m.get("chunk_type") == "figure"),
        "tables":       sum(1 for m in metas if m.get("chunk_type") == "table"),
        "text_chunks":  sum(1 for m in metas if m.get("chunk_type") == "text"),
        "entities":     0,
    }


def clear_graph() -> dict:
    client = _get_client()
    try:
        col   = _get_col()
        count = col.count()
        client.delete_collection("chunks")
    except Exception:
        count = 0
    return {"deleted": count}
