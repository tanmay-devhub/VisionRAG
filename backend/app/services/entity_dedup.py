# ── VisionRAG Phase 1 / 2 ──────────────────────────────────────────────────
# Step 2: Entity deduplication + visual similarity edge builder
# RAG improvement: merges near-duplicate VisualEntity nodes so entity graph
#                  traversal hits all relevant chunks; VISUALLY_SIMILAR edges
#                  surface cross-file visual matches that keyword search misses
# ──────────────────────────────────────────────────────────────────────────

import os
import logging

from app.services.graph_store import _get_driver_ready

logger = logging.getLogger(__name__)

_ENTITY_DEDUP_THRESHOLD = float(os.getenv("ENTITY_DEDUP_THRESHOLD", "88"))
_VISUAL_SIM_THRESHOLD   = float(os.getenv("VISUAL_SIM_THRESHOLD",   "0.82"))
_BATCH                  = 50


# ── minimal Union-Find ────────────────────────────────────────────────────────

class _UF:
    def __init__(self, elements):
        self._parent = {e: e for e in elements}

    def find(self, x):
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path compression
            x = self._parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


# ── public API ────────────────────────────────────────────────────────────────

def run_entity_dedup(filename: str | None = None) -> dict:
    """
    Merge near-duplicate VisualEntity nodes using fuzzy string matching.
    filename: scope to one file's entities. None = all entities.
    Returns: {merged, clusters_found, entities_before, entities_after}
    """
    from rapidfuzz.fuzz import token_sort_ratio

    drv = _get_driver_ready()

    with drv.session() as session:
        if filename:
            res = session.run(
                """
                MATCH (d:Document {filename: $fn})-[:CONTAINS]->(c:MediaChunk)-[:DEPICTS]->(e:VisualEntity)
                RETURN DISTINCT e.name AS name, e.display_name AS display_name,
                       e.mention_count AS mention_count
                """,
                fn=filename,
            )
        else:
            res = session.run(
                "MATCH (e:VisualEntity) RETURN e.name AS name, "
                "e.display_name AS display_name, e.mention_count AS mention_count"
            )
        entities = [
            {
                "name":          rec["name"],
                "display_name":  rec["display_name"],
                "mention_count": int(rec["mention_count"] or 1),
            }
            for rec in res
        ]

    entities_before = len(entities)
    if entities_before < 2:
        return {
            "merged":           0,
            "clusters_found":   0,
            "entities_before":  entities_before,
            "entities_after":   entities_before,
        }

    # pairwise fuzzy comparison
    names = [e["name"] for e in entities]
    uf    = _UF(names)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            score = token_sort_ratio(names[i], names[j])
            if score >= _ENTITY_DEDUP_THRESHOLD:
                logger.debug(
                    "Merging '%s' → '%s' (score=%.1f)", names[j], names[i], score
                )
                uf.union(names[i], names[j])

    # group into clusters
    clusters: dict[str, list[dict]] = {}
    for e in entities:
        root = uf.find(e["name"])
        clusters.setdefault(root, []).append(e)

    multi = {root: members for root, members in clusters.items() if len(members) > 1}
    clusters_found = len(multi)
    merged_total   = 0

    with drv.session() as session:
        for root, members in multi.items():
            # canonical = highest mention_count; tie-break: shortest name
            canonical = max(
                members,
                key=lambda e: (e["mention_count"], -len(e["name"])),
            )
            duplicates = [e for e in members if e["name"] != canonical["name"]]
            extra_mentions = sum(d["mention_count"] for d in duplicates)

            # accumulate mention_count on canonical
            session.run(
                "MATCH (e:VisualEntity {name: $name}) "
                "SET e.mention_count = e.mention_count + $extra",
                name=canonical["name"],
                extra=extra_mentions,
            )

            # process duplicates in batches
            for start in range(0, len(duplicates), _BATCH):
                batch = duplicates[start : start + _BATCH]
                for dup in batch:
                    dup_name = dup["name"]
                    can_name = canonical["name"]

                    # repoint DEPICTS edges
                    session.run(
                        """
                        MATCH (c:MediaChunk)-[r:DEPICTS]->(dup:VisualEntity {name: $dup})
                        MATCH (can:VisualEntity {name: $can})
                        MERGE (c)-[:DEPICTS]->(can)
                        DELETE r
                        """,
                        dup=dup_name,
                        can=can_name,
                    )

                    # repoint CO_OCCURS_WITH edges
                    session.run(
                        """
                        MATCH (dup:VisualEntity {name: $dup})-[r:CO_OCCURS_WITH]-(other:VisualEntity)
                        WHERE other.name <> $can
                        MATCH (can:VisualEntity {name: $can})
                        MERGE (can)-[:CO_OCCURS_WITH]->(other)
                        MERGE (other)-[:CO_OCCURS_WITH]->(can)
                        DELETE r
                        """,
                        dup=dup_name,
                        can=can_name,
                    )

                    # delete duplicate
                    session.run(
                        "MATCH (e:VisualEntity {name: $dup}) DETACH DELETE e",
                        dup=dup_name,
                    )
                    merged_total += 1

    with drv.session() as session:
        res = session.run("MATCH (e:VisualEntity) RETURN count(e) AS n")
        rec = res.single()
        entities_after = int(rec["n"]) if rec else 0

    return {
        "merged":          merged_total,
        "clusters_found":  clusters_found,
        "entities_before": entities_before,
        "entities_after":  entities_after,
    }


def build_visual_similarity(
    filename: str | None = None,
    threshold: float | None = None,
) -> dict:
    """
    Build VISUALLY_SIMILAR edges between figure/frame MediaChunk nodes.
    filename: scope which chunks to process (edges still created cross-file).
    threshold: cosine similarity cutoff, overrides VISUAL_SIM_THRESHOLD env var.
    Returns: {edges_created, chunks_processed}
    """
    import numpy as np

    cutoff = threshold if threshold is not None else _VISUAL_SIM_THRESHOLD
    drv    = _get_driver_ready()
    _MAX_CHUNKS = 2000

    with drv.session() as session:
        if filename:
            res = session.run(
                """
                MATCH (c:MediaChunk)
                WHERE c.chunk_type IN ['figure', 'frame'] AND c.filename = $fn
                RETURN c.id AS id, c.embedding AS embedding
                LIMIT $lim
                """,
                fn=filename,
                lim=_MAX_CHUNKS,
            )
        else:
            res = session.run(
                """
                MATCH (c:MediaChunk)
                WHERE c.chunk_type IN ['figure', 'frame']
                RETURN c.id AS id, c.embedding AS embedding
                LIMIT $lim
                """,
                lim=_MAX_CHUNKS,
            )
        rows = [(rec["id"], rec["embedding"]) for rec in res if rec["embedding"]]

    if not rows:
        return {"edges_created": 0, "chunks_processed": 0}

    chunks_processed = len(rows)
    if chunks_processed >= _MAX_CHUNKS:
        logger.warning(
            "build_visual_similarity: hit %d chunk cap — some chunks skipped", _MAX_CHUNKS
        )

    ids   = [r[0] for r in rows]
    embs  = np.array([r[1] for r in rows], dtype=np.float32)

    # fastembed outputs unit vectors — cosine = dot product
    sim_matrix = embs @ embs.T

    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            score = float(sim_matrix[i, j])
            if score >= cutoff:
                pairs.append({"id_a": ids[i], "id_b": ids[j], "score": score})

    if not pairs:
        return {"edges_created": 0, "chunks_processed": chunks_processed}

    _SIM_BATCH = 200
    edges_created = 0
    with drv.session() as session:
        for start in range(0, len(pairs), _SIM_BATCH):
            batch = pairs[start : start + _SIM_BATCH]
            result = session.run(
                """
                UNWIND $pairs AS p
                MATCH (a:MediaChunk {id: p.id_a})
                MATCH (b:MediaChunk {id: p.id_b})
                MERGE (a)-[r:VISUALLY_SIMILAR]->(b) SET r.score = p.score
                MERGE (b)-[s:VISUALLY_SIMILAR]->(a) SET s.score = p.score
                RETURN count(*) AS n
                """,
                pairs=batch,
            )
            rec = result.single()
            edges_created += int(rec["n"]) if rec else 0

    return {"edges_created": edges_created, "chunks_processed": chunks_processed}
