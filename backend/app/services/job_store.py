# ── VisionRAG Neo4j Migration ─────────────────────────────────────────────────
# Replaces: in-memory dict + threading.Lock (lost on every server restart)
# Video-ready: schema is generic; video ingest jobs use same columns —
#              no changes needed when video support is added.
# ─────────────────────────────────────────────────────────────────────────────

import os
import uuid
import sqlite3
import threading
from typing import Any

_DB_PATH = os.getenv("JOB_STORE_PATH", "./jobs.db")
_lock    = threading.Lock()
_inited  = False


# ── schema ────────────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id       TEXT    PRIMARY KEY,
    filename     TEXT    NOT NULL,
    status       TEXT    NOT NULL DEFAULT 'pending',
    chunks_done  INTEGER NOT NULL DEFAULT 0,
    total_chunks INTEGER NOT NULL DEFAULT 0,
    figure_count INTEGER NOT NULL DEFAULT 0,
    table_count  INTEGER NOT NULL DEFAULT 0,
    error        TEXT
)
"""

_COLUMNS = ("job_id", "filename", "status", "chunks_done",
            "total_chunks", "figure_count", "table_count", "error")


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(_DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def _ensure_table() -> None:
    global _inited
    if _inited:
        return
    with _lock:
        if _inited:
            return
        with _conn() as c:
            c.execute(_CREATE_TABLE)
        _inited = True


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["error"] = d["error"] or None
    return d


# ── startup helper ────────────────────────────────────────────────────────────

def mark_interrupted_jobs() -> None:
    """
    Called at FastAPI lifespan startup.
    Any job still in 'pending' or 'processing' state was mid-flight when the
    server died and will never complete — mark it 'interrupted'.
    """
    _ensure_table()
    with _lock:
        with _conn() as c:
            c.execute(
                "UPDATE jobs SET status = 'interrupted' "
                "WHERE status IN ('pending', 'processing')"
            )


# ── public API ────────────────────────────────────────────────────────────────

def create_job(filename: str) -> str:
    _ensure_table()
    job_id = str(uuid.uuid4())
    with _lock:
        with _conn() as c:
            c.execute(
                "INSERT INTO jobs (job_id, filename, status, chunks_done, "
                "total_chunks, figure_count, table_count, error) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (job_id, filename, "pending", 0, 0, 0, 0, None),
            )
    return job_id


def update_job(job_id: str, **kwargs: Any) -> None:
    _ensure_table()
    allowed = {k: v for k, v in kwargs.items() if k in _COLUMNS and k != "job_id"}
    if not allowed:
        return
    set_clause = ", ".join(f"{k} = ?" for k in allowed)
    values     = list(allowed.values()) + [job_id]
    with _lock:
        with _conn() as c:
            c.execute(f"UPDATE jobs SET {set_clause} WHERE job_id = ?", values)


def get_job(job_id: str) -> dict[str, Any] | None:
    _ensure_table()
    with _conn() as c:
        row = c.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    return _row_to_dict(row) if row else None


def list_jobs() -> list[dict[str, Any]]:
    _ensure_table()
    with _conn() as c:
        rows = c.execute("SELECT * FROM jobs ORDER BY rowid DESC").fetchall()
    return [_row_to_dict(r) for r in rows]
