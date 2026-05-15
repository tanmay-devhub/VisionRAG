import uuid
import threading
from typing import Any

_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def create_job(filename: str) -> str:
    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {
            "job_id":      job_id,
            "filename":    filename,
            "status":      "pending",
            "chunks_done": 0,
            "total_chunks": 0,
            "figure_count": 0,
            "table_count":  0,
            "error":        None,
        }
    return job_id


def update_job(job_id: str, **kwargs: Any) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def get_job(job_id: str) -> dict[str, Any] | None:
    return _jobs.get(job_id)


def list_jobs() -> list[dict[str, Any]]:
    with _lock:
        return list(_jobs.values())
