import math
import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info("Loading re-ranker model: %s", _MODEL_NAME)
        _reranker = CrossEncoder(_MODEL_NAME)
    return _reranker


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def rerank(question: str, candidates: list[dict], top_k: int) -> list[dict]:
    if not candidates:
        return []

    reranker = _get_reranker()
    pairs = [(question, c["text"]) for c in candidates]
    raw_scores = reranker.predict(pairs)

    for candidate, raw in zip(candidates, raw_scores):
        candidate["score"] = round(_sigmoid(float(raw)), 4)

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:top_k]
