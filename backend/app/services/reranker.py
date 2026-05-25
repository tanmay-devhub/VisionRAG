import math
import logging

logger = logging.getLogger(__name__)

_MODEL_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"
_reranker   = None  # fastembed TextCrossEncoder, lazy


def _get_reranker():
    global _reranker
    if _reranker is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # ONNX-based, no torch
        logger.info("Loading re-ranker model: %s", _MODEL_NAME)
        _reranker = TextCrossEncoder(_MODEL_NAME)
    return _reranker


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def rerank(question: str, candidates: list[dict], top_k: int) -> list[dict]:
    if not candidates:
        return []

    try:
        reranker = _get_reranker()
        texts    = [c["text"] for c in candidates]
        scores   = list(reranker.rerank(question, texts))

        for candidate, score in zip(candidates, scores):
            raw = float(score) if not hasattr(score, "score") else float(score.score)
            candidate["score"] = round(_sigmoid(raw), 4)

        candidates.sort(key=lambda c: c["score"], reverse=True)
    except Exception as exc:
        logger.warning("Reranker failed (%s) — using original order", exc)

    return candidates[:top_k]
