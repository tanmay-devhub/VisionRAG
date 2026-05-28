# ── VisionRAG Neo4j Migration ─────────────────────────────────────────────────
# Replaces: chromadb PersistentClient + SentenceTransformerEmbeddingFunction
# Video-ready: MediaChunk carries timestamp_ms + chunk_type "frame"/"transcript";
#              source_type "video" branch in store_chunks; all traversal queries
#              work identically on video frames — no schema change needed.
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import logging
import unicodedata
from datetime import datetime, timezone
from typing import Callable

logger = logging.getLogger(__name__)

_NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
_NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")
_EMBED_MODEL    = os.getenv("EMBED_MODEL",    "all-MiniLM-L6-v2")
_RERANK_CANDIDATE_MULTIPLIER = int(os.getenv("RERANK_CANDIDATE_MULTIPLIER", "3"))

_driver        = None   # neo4j.Driver, lazy
_embedder      = None   # SentenceTransformer, lazy
_indexes_ready = False  # created on first driver use

_STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could","should",
    "may","might","shall","can","to","of","in","on","at","by","for",
    "with","from","as","or","and","but","not","no","it","its",
    "this","that","these","those","what","which","who","how","when","where",
}


# ── singletons ────────────────────────────────────────────────────────────────

def _get_driver():
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase  # lazy — avoids any driver init at import time
        _driver = GraphDatabase.driver(
            _NEO4J_URI,
            auth=(_NEO4J_USER, _NEO4J_PASSWORD),
            connection_timeout=10,       # fail fast if Neo4j is unreachable
            max_connection_lifetime=3600,
        )
    return _driver


def _get_embedder():
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding  # ONNX-based — no torch, no Defender hang
        model_name = _EMBED_MODEL if "/" in _EMBED_MODEL else f"sentence-transformers/{_EMBED_MODEL}"
        _embedder = TextEmbedding(model_name)
    return _embedder


# ── startup ───────────────────────────────────────────────────────────────────

def _get_driver_ready():
    """Return driver, ensuring indexes exist on first call."""
    global _indexes_ready
    drv = _get_driver()
    if not _indexes_ready:
        create_indexes()
        _indexes_ready = True
    return drv


def ping() -> None:
    """Verify Neo4j connectivity. Raises on failure."""
    _get_driver().verify_connectivity()


def create_indexes() -> None:
    """Idempotent index/constraint setup. Run once at lifespan startup."""
    stmts = [
        # unique constraint doubles as a lookup index
        "CREATE CONSTRAINT mediaChunk_id IF NOT EXISTS FOR (c:MediaChunk) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT document_filename IF NOT EXISTS FOR (d:Document) REQUIRE d.filename IS UNIQUE",
        "CREATE INDEX visualEntity_name IF NOT EXISTS FOR (e:VisualEntity) ON (e.name)",
        "CREATE INDEX mediaChunk_filename IF NOT EXISTS FOR (c:MediaChunk) ON (c.filename)",
        # fulltext for keyword search
        """CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS
           FOR (c:MediaChunk) ON EACH [c.text]""",
        # vector index — 384-dim all-MiniLM-L6-v2, cosine
        """CREATE VECTOR INDEX chunk_vector IF NOT EXISTS
           FOR (c:MediaChunk) ON c.embedding
           OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}""",
    ]
    with _get_driver().session() as session:
        for stmt in stmts:
            try:
                session.run(stmt).consume()
            except Exception as exc:
                msg = str(exc).lower()
                if "already exists" not in msg and "equivalent" not in msg:
                    logger.warning("Index stmt warning: %s", exc)


# ── helpers ───────────────────────────────────────────────────────────────────

def _normalise_entity(name: str) -> str:
    n = unicodedata.normalize("NFKC", name)
    n = n.strip().lower()
    n = n.replace("_", " ")          # treat underscores as spaces for fuzzy matching
    n = re.sub(r"\s+", " ", n)
    return n


def _derive_source_type(chunks: list[dict]) -> str:
    for c in chunks:
        if c.get("chunk_type") in ("frame", "transcript", "video_summary"):
            return "video"
    if (
        len(chunks) == 1
        and chunks[0].get("chunk_type") == "figure"
        and chunks[0].get("page_number") == 0
    ):
        return "image"
    return "pdf"


# ── ingest ────────────────────────────────────────────────────────────────────

def store_chunks(
    chunks: list[dict],
    filename: str,
    progress_cb: Callable[[int], None] | None = None,
) -> tuple[int, int]:
    """
    Write all chunks for one file to Neo4j using UNWIND batch writes.
    Returns (entities_created, relationships_created).

    Existing ChromaDB data does not migrate — users re-ingest files.
    This is a clean-slate Neo4j store; no migration script is provided.
    """
    if not chunks:
        return 0, 0

    embedder    = _get_embedder()
    now         = datetime.now(timezone.utc).isoformat()
    source_type = _derive_source_type(chunks)
    _BATCH      = 100

    # ── 1. embed all texts in one batched encode call ─────────────────────────
    texts      = [c["text"] for c in chunks]
    embeddings = list(embedder.embed(texts, batch_size=64))

    # ── 2. build payload ──────────────────────────────────────────────────────
    chunk_rows: list[dict] = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        pn = chunk.get("page_number")
        chunk_rows.append({
            "id":           f"{filename}_{chunk['chunk_index']}",
            "filename":     filename,
            "chunk_index":  chunk["chunk_index"],
            "chunk_type":   chunk.get("chunk_type", "text"),
            "media_type":   source_type,
            "text":         chunk["text"],
            "image_url":    chunk.get("image_url")   or "",
            "figure_type":  chunk.get("figure_type") or "",
            "caption":      chunk.get("caption")     or "",
            "page_number":  pn if pn is not None else -1,
            "timestamp_ms": chunk.get("timestamp_ms"),   # None for images/pdf
            "embedding":    emb.tolist(),
            "ingested_at":  now,
        })

    # ── 3. upsert Document node ───────────────────────────────────────────────
    with _get_driver_ready().session() as session:
        session.run(
            """
            MERGE (d:Document {filename: $filename})
            SET d.source_type  = $source_type,
                d.chunk_count  = $chunk_count,
                d.ingested_at  = $ingested_at
            """,
            filename=filename,
            source_type=source_type,
            chunk_count=len(chunks),
            ingested_at=now,
        )

        # ── 4. upsert MediaChunk nodes + CONTAINS edges in batches ───────────
        for start in range(0, len(chunk_rows), _BATCH):
            batch = chunk_rows[start : start + _BATCH]
            session.run(
                """
                UNWIND $rows AS row
                MERGE (c:MediaChunk {id: row.id})
                SET c.filename    = row.filename,
                    c.chunk_index = row.chunk_index,
                    c.chunk_type  = row.chunk_type,
                    c.media_type  = row.media_type,
                    c.text        = row.text,
                    c.image_url   = row.image_url,
                    c.figure_type = row.figure_type,
                    c.caption     = row.caption,
                    c.page_number = row.page_number,
                    c.timestamp_ms= row.timestamp_ms,
                    c.embedding   = row.embedding,
                    c.ingested_at = row.ingested_at
                WITH c, row
                MATCH (d:Document {filename: row.filename})
                MERGE (d)-[:CONTAINS]->(c)
                """,
                rows=batch,
            )
            if progress_cb:
                progress_cb(min(start + _BATCH, len(chunks)))

        # ── 5. NEXT_CHUNK chain within this file ──────────────────────────────
        session.run(
            """
            MATCH (d:Document {filename: $filename})-[:CONTAINS]->(c:MediaChunk)
            WITH c ORDER BY c.chunk_index ASC
            WITH collect(c) AS ordered
            UNWIND range(0, size(ordered) - 2) AS i
            WITH ordered[i] AS a, ordered[i+1] AS b
            MERGE (a)-[:NEXT_CHUNK]->(b)
            """,
            filename=filename,
        )

        # ── 6. entity + relationship extraction (figure / frame chunks) ───────
        entities_created      = 0
        relationships_created = 0

        visual_chunks = [
            (chunk_rows[i], chunks[i])
            for i in range(len(chunks))
            if chunks[i].get("chunk_type") in ("figure", "frame", "video_summary")
        ]

        for crow, chunk in visual_chunks:
            raw_entities = chunk.get("extra_entities") or []
            raw_rels     = chunk.get("extra_relationships") or []

            if raw_entities:
                entity_rows = []
                for ename in raw_entities:
                    norm = _normalise_entity(str(ename))
                    if norm:
                        entity_rows.append({"norm": norm, "display": str(ename).strip()})

                if entity_rows:
                    result = session.run(
                        """
                        UNWIND $rows AS row
                        MERGE (e:VisualEntity {name: row.norm})
                        ON CREATE SET e.display_name  = row.display,
                                      e.entity_type   = 'object',
                                      e.mention_count = 1
                        ON MATCH  SET e.mention_count = e.mention_count + 1
                        WITH e, row
                        MATCH (c:MediaChunk {id: $chunk_id})
                        MERGE (c)-[:DEPICTS]->(e)
                        RETURN count(e) AS n
                        """,
                        rows=entity_rows,
                        chunk_id=crow["id"],
                    )
                    rec = result.single()
                    entities_created += rec["n"] if rec else 0

            if raw_rels:
                rel_rows = []
                for rel in raw_rels:
                    if not isinstance(rel, dict):
                        continue
                    f = _normalise_entity(str(rel.get("from", "")))
                    t = _normalise_entity(str(rel.get("to", "")))
                    if f and t and f != t:
                        rel_rows.append({
                            "from_norm":    f,
                            "from_display": str(rel.get("from", "")).strip(),
                            "to_norm":      t,
                            "to_display":   str(rel.get("to", "")).strip(),
                        })

                if rel_rows:
                    result = session.run(
                        """
                        UNWIND $rows AS row
                        MERGE (a:VisualEntity {name: row.from_norm})
                          ON CREATE SET a.display_name = row.from_display,
                                        a.entity_type  = 'object',
                                        a.mention_count = 1
                        MERGE (b:VisualEntity {name: row.to_norm})
                          ON CREATE SET b.display_name = row.to_display,
                                        b.entity_type  = 'object',
                                        b.mention_count = 1
                        MERGE (a)-[:CO_OCCURS_WITH]->(b)
                        MERGE (b)-[:CO_OCCURS_WITH]->(a)
                        RETURN count(*) AS n
                        """,
                        rows=rel_rows,
                    )
                    rec = result.single()
                    relationships_created += rec["n"] if rec else 0

    # ensure final progress tick
    if progress_cb:
        progress_cb(len(chunks))

    return entities_created, relationships_created


# ── retrieval ─────────────────────────────────────────────────────────────────

def _rrf(result_lists: list[list[tuple[str, float]]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion across multiple ranked lists."""
    scores: dict[str, float] = {}
    for ranked in result_lists:
        for rank, (chunk_id, _) in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


def query_chunks(question: str, top_k: int) -> list[dict]:
    """
    Hybrid retrieval: vector search + fulltext + entity graph traversal.
    Results fused with Reciprocal Rank Fusion (k=60).
    Returns top_k*4 candidates for the CrossEncoder reranker to reduce.
    """
    embedder  = _get_embedder()
    q_emb     = list(embedder.embed([question]))[0].tolist()
    candidate_n = top_k * _RERANK_CANDIDATE_MULTIPLIER

    # extract keywords (no stopwords, len > 2)
    keywords = [
        w.lower() for w in re.findall(r"\b\w+\b", question)
        if len(w) > 2 and w.lower() not in _STOPWORDS
    ]

    vec_ranked:    list[tuple[str, float]] = []
    ft_ranked:     list[tuple[str, float]] = []
    graph_ranked:  list[tuple[str, float]] = []
    chunk_map:     dict[str, dict]         = {}

    def _row_to_dict(node, score: float, rtype: str) -> dict:
        pn = node.get("page_number")
        return {
            "text":        node.get("text", ""),
            "source":      node.get("filename", ""),
            "chunk_index": int(node.get("chunk_index", 0)),
            "score":       float(score),
            "type":        rtype,
            "chunk_type":  node.get("chunk_type", "text"),
            "media_type":   node.get("media_type")   or None,
            "image_url":    node.get("image_url")    or None,
            "figure_type":  node.get("figure_type")  or None,
            "caption":      node.get("caption")      or None,
            "page_number":  None if pn in (-1, None) else int(pn),
            "timestamp_ms": node.get("timestamp_ms") or None,
        }

    with _get_driver_ready().session() as session:

        # ── strategy 1: vector search ─────────────────────────────────────────
        try:
            vres = session.run(
                """
                CALL db.index.vector.queryNodes("chunk_vector", $k, $embedding)
                YIELD node, score
                RETURN node, score
                """,
                k=candidate_n,
                embedding=q_emb,
            )
            for rec in vres:
                node  = rec["node"]
                score = float(rec["score"])
                cid   = node["id"]
                vec_ranked.append((cid, score))
                if cid not in chunk_map or score > chunk_map[cid]["score"]:
                    chunk_map[cid] = _row_to_dict(node, score, "vector")
        except Exception as exc:
            logger.warning("Vector search failed: %s", exc)

        # ── strategy 2: fulltext search ───────────────────────────────────────
        if keywords:
            ft_query = " OR ".join(keywords)
            try:
                ftres = session.run(
                    """
                    CALL db.index.fulltext.queryNodes("chunk_fulltext", $q)
                    YIELD node, score
                    RETURN node, score
                    LIMIT $lim
                    """,
                    q=ft_query,
                    lim=candidate_n,
                )
                rows = [(rec["node"], float(rec["score"])) for rec in ftres]
                max_score = max((s for _, s in rows), default=1.0) or 1.0
                for node, raw_score in rows:
                    cid   = node["id"]
                    score = raw_score / max_score
                    ft_ranked.append((cid, score))
                    if cid not in chunk_map or score > chunk_map[cid]["score"]:
                        chunk_map[cid] = _row_to_dict(node, score, "graph")
            except Exception as exc:
                logger.warning("Fulltext search failed: %s", exc)

        # ── strategy 3: entity graph traversal ───────────────────────────────
        if keywords:
            for kw in keywords[:6]:   # cap to avoid combinatorial explosion
                try:
                    gres = session.run(
                        """
                        MATCH (e:VisualEntity)
                        WHERE e.name CONTAINS $kw
                        CALL {
                            WITH e
                            MATCH (e)<-[:DEPICTS]-(c:MediaChunk)
                            RETURN c, 0.6 AS score
                            UNION
                            WITH e
                            MATCH (e)-[:CO_OCCURS_WITH]-(e2:VisualEntity)<-[:DEPICTS]-(c:MediaChunk)
                            RETURN c, 0.4 AS score
                            UNION
                            WITH e
                            MATCH (e)<-[:DEPICTS]-(c1:MediaChunk)-[:VISUALLY_SIMILAR]-(c2:MediaChunk)
                            RETURN c2 AS c, 0.35 AS score
                        }
                        RETURN c AS node, max(score) AS score
                        LIMIT $lim
                        """,
                        kw=kw,
                        lim=candidate_n,
                    )
                    for rec in gres:
                        node  = rec["node"]
                        score = float(rec["score"])
                        cid   = node["id"]
                        graph_ranked.append((cid, score))
                        if cid not in chunk_map or score > chunk_map[cid]["score"]:
                            chunk_map[cid] = _row_to_dict(node, score, "graph")
                except Exception as exc:
                    logger.warning("Graph traversal failed for kw=%s: %s", kw, exc)

    if not chunk_map:
        return []

    # ── RRF fusion ────────────────────────────────────────────────────────────
    rrf_scores = _rrf([vec_ranked, ft_ranked, graph_ranked])

    # update type to "hybrid" when a chunk appears in multiple strategies
    vec_ids   = {cid for cid, _ in vec_ranked}
    ft_ids    = {cid for cid, _ in ft_ranked}
    graph_ids = {cid for cid, _ in graph_ranked}
    for cid in chunk_map:
        found_in = sum([cid in vec_ids, cid in ft_ids, cid in graph_ids])
        if found_in > 1:
            chunk_map[cid]["type"] = "hybrid"

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    results    = []
    for cid in sorted_ids[: top_k * 4]:
        chunk = chunk_map[cid].copy()
        chunk["score"] = rrf_scores[cid]
        results.append(chunk)

    return results


# ── admin / visualisation ─────────────────────────────────────────────────────

def get_ingested_files() -> dict:
    with _get_driver_ready().session() as session:
        res = session.run(
            """
            MATCH (d:Document)-[:CONTAINS]->(c:MediaChunk)
            WITH d, c
            RETURN d.filename        AS filename,
                   d.ingested_at     AS ingested_at,
                   count(c)          AS chunks,
                   sum(CASE WHEN c.chunk_type = 'figure' THEN 1 ELSE 0 END) AS figures,
                   sum(CASE WHEN c.chunk_type = 'table'  THEN 1 ELSE 0 END) AS tables
            ORDER BY d.filename
            """
        )
        files = []
        for rec in res:
            files.append({
                "filename":    rec["filename"],
                "chunks":      int(rec["chunks"]),
                "figures":     int(rec["figures"]),
                "tables":      int(rec["tables"]),
                "ingested_at": rec["ingested_at"],
            })
    return {"files": files}


def get_all_graph_data(filename: str | None = None) -> dict:
    with _get_driver_ready().session() as session:
        # chunk nodes
        if filename:
            chunk_res = session.run(
                """
                MATCH (c:MediaChunk {filename: $fn})
                RETURN c LIMIT 500
                """,
                fn=filename,
            )
        else:
            chunk_res = session.run(
                "MATCH (c:MediaChunk) RETURN c LIMIT 500"
            )

        nodes: list[dict] = []
        chunk_ids: set[str] = set()
        for rec in chunk_res:
            n = rec["c"]
            cid = n["id"]
            chunk_ids.add(cid)
            nodes.append({
                "id":         cid,
                "label":      f"Chunk {n.get('chunk_index', '?')}",
                "filename":   n.get("filename", ""),
                "nodeType":   "chunk",
                "chunk_type": n.get("chunk_type", "text"),
            })

        # entity nodes (only those connected to visible chunks)
        if chunk_ids:
            ent_res = session.run(
                """
                MATCH (c:MediaChunk)-[:DEPICTS]->(e:VisualEntity)
                WHERE c.id IN $ids
                RETURN e LIMIT 1000
                """,
                ids=list(chunk_ids),
            )
            seen_entities: set[str] = set()
            for rec in ent_res:
                e    = rec["e"]
                eid  = f"entity_{e['name']}"
                if eid not in seen_entities:
                    seen_entities.add(eid)
                    nodes.append({
                        "id":       eid,
                        "label":    e.get("display_name", e["name"]),
                        "nodeType": "entity",
                    })

        # links
        all_ids = chunk_ids | {n["id"] for n in nodes if n["nodeType"] == "entity"}
        links: list[dict] = []

        if chunk_ids:
            nc_res = session.run(
                """
                MATCH (a:MediaChunk)-[:NEXT_CHUNK]->(b:MediaChunk)
                WHERE a.id IN $ids AND b.id IN $ids
                RETURN a.id AS src, b.id AS tgt
                LIMIT 3000
                """,
                ids=list(chunk_ids),
            )
            for rec in nc_res:
                links.append({"source": rec["src"], "target": rec["tgt"], "type": "NEXT_CHUNK"})

            dep_res = session.run(
                """
                MATCH (c:MediaChunk)-[:DEPICTS]->(e:VisualEntity)
                WHERE c.id IN $ids
                RETURN c.id AS src, 'entity_' + e.name AS tgt
                LIMIT 3000
                """,
                ids=list(chunk_ids),
            )
            for rec in dep_res:
                links.append({"source": rec["src"], "target": rec["tgt"], "type": "DEPICTS"})

            co_res = session.run(
                """
                MATCH (a:MediaChunk)-[:DEPICTS]->(e1:VisualEntity)-[:CO_OCCURS_WITH]->(e2:VisualEntity)
                WHERE a.id IN $ids
                RETURN 'entity_' + e1.name AS src, 'entity_' + e2.name AS tgt,
                       e2.name AS tgt_name, e2.display_name AS tgt_display
                LIMIT 3000
                """,
                ids=list(chunk_ids),
            )
            for rec in co_res:
                src, tgt = rec["src"], rec["tgt"]
                if src not in seen_entities:
                    continue
                # Add the neighbor entity as a node if not already present
                if tgt not in seen_entities:
                    seen_entities.add(tgt)
                    display = rec.get("tgt_display") or rec.get("tgt_name") or tgt[len("entity_"):]
                    nodes.append({
                        "id":       tgt,
                        "label":    display,
                        "nodeType": "entity",
                    })
                links.append({"source": src, "target": tgt, "type": "CO_OCCURS_WITH"})

            vs_res = session.run(
                """
                MATCH (a:MediaChunk)-[:VISUALLY_SIMILAR]->(b:MediaChunk)
                WHERE a.id IN $ids AND b.id IN $ids
                RETURN a.id AS src, b.id AS tgt
                LIMIT 1000
                """,
                ids=list(chunk_ids),
            )
            for rec in vs_res:
                links.append({"source": rec["src"], "target": rec["tgt"], "type": "VISUALLY_SIMILAR"})

    return {"nodes": nodes, "links": links}


def get_graph_stats() -> dict:
    with _get_driver_ready().session() as session:
        res = session.run(
            """
            MATCH (c:MediaChunk)
            WITH count(c)                                                           AS total,
                 sum(CASE WHEN c.chunk_type = 'figure' THEN 1 ELSE 0 END)          AS figures,
                 sum(CASE WHEN c.chunk_type = 'table'  THEN 1 ELSE 0 END)          AS tables,
                 sum(CASE WHEN c.chunk_type = 'text'   THEN 1 ELSE 0 END)          AS text_chunks
            OPTIONAL MATCH (e:VisualEntity)
            WITH total, figures, tables, text_chunks, count(e) AS entities
            OPTIONAL MATCH ()-[r:CO_OCCURS_WITH]->()
            WITH total, figures, tables, text_chunks, entities, count(r) AS co_occurs
            OPTIONAL MATCH ()-[s:VISUALLY_SIMILAR]->()
            RETURN total, figures, tables, text_chunks, entities,
                   co_occurs, count(s) AS visually_similar
            """
        )
        rec = res.single()
        if rec is None:
            return {
                "total_chunks": 0, "figures": 0, "tables": 0, "text_chunks": 0,
                "entities": 0, "relationships": 0, "visually_similar_edges": 0,
            }
        return {
            "total_chunks":           int(rec["total"]            or 0),
            "figures":                int(rec["figures"]           or 0),
            "tables":                 int(rec["tables"]            or 0),
            "text_chunks":            int(rec["text_chunks"]       or 0),
            "entities":               int(rec["entities"]          or 0),
            "relationships":          int(rec["co_occurs"]         or 0),
            "visually_similar_edges": int(rec["visually_similar"]  or 0),
        }


def clear_graph() -> dict:
    with _get_driver_ready().session() as session:
        res = session.run(
            """
            MATCH (n)
            WHERE n:Document OR n:MediaChunk OR n:VisualEntity
            WITH count(n) AS total
            CALL {
                MATCH (n)
                WHERE n:Document OR n:MediaChunk OR n:VisualEntity
                DETACH DELETE n
            }
            RETURN total
            """
        )
        rec = res.single()
        return {"deleted": int(rec["total"]) if rec else 0}


def delete_file(filename: str) -> dict:
    """
    Delete one file's Document + MediaChunk nodes, all their edges,
    and any VisualEntity nodes that become orphans (no remaining DEPICTS edges).
    """
    with _get_driver_ready().session() as session:
        # count chunks first for the return value
        count_res = session.run(
            "MATCH (c:MediaChunk {filename: $fn}) RETURN count(c) AS n",
            fn=filename,
        )
        chunk_count = int((count_res.single() or {}).get("n", 0))

        # delete chunks + document
        session.run(
            """
            MATCH (d:Document {filename: $fn})
            DETACH DELETE d
            """,
            fn=filename,
        )
        session.run(
            """
            MATCH (c:MediaChunk {filename: $fn})
            DETACH DELETE c
            """,
            fn=filename,
        )

        # orphan entity cleanup
        orphan_res = session.run(
            """
            MATCH (e:VisualEntity)
            WHERE NOT (e)<-[:DEPICTS]-()
            WITH count(e) AS n
            CALL {
                MATCH (e:VisualEntity)
                WHERE NOT (e)<-[:DEPICTS]-()
                DETACH DELETE e
            }
            RETURN n
            """
        )
        orphan_rec   = orphan_res.single()
        entity_count = int(orphan_rec["n"]) if orphan_rec else 0

    return {"deleted_chunks": chunk_count, "deleted_entities": entity_count}
