# ── VisionRAG Phase 1 / 2 ──────────────────────────────────────────────────
# Step 5: RAGAS-style evaluation harness
# RAG improvement: LLM-as-judge baseline that measures faithfulness,
#                  answer relevancy, context precision/recall, and a
#                  VisionRAG-specific visual grounding metric
# ──────────────────────────────────────────────────────────────────────────

"""
Usage:
  python eval/ragas_eval.py                           # full run → baseline_before_video.json
  python eval/ragas_eval.py --subset 3                # smoke test on 3 questions
  python eval/ragas_eval.py --output eval/my_run.json

Requirements: see eval/requirements.txt
Does NOT import any backend modules — hits the API over HTTP only.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

_API_BASE       = os.getenv("RAGAS_API_BASE",  "http://localhost:8081")
_OLLAMA_BASE    = os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434")
_LLM_MODEL      = os.getenv("LLM_MODEL",        "llama3.2")
_DEFAULT_OUTPUT = Path(__file__).resolve().parent / "baseline_before_video.json"
_DATASET_PATH   = Path(__file__).resolve().parent / "golden_dataset.json"


# ── LLM judge ────────────────────────────────────────────────────────────────

def _llm_score(prompt: str, timeout: float = 60.0) -> float:
    """
    Calls Ollama /api/generate, expects {"score": float, "reason": str} in the reply.
    Returns 0.0 on any parse failure.
    """
    try:
        resp = httpx.post(
            f"{_OLLAMA_BASE}/api/generate",
            json={"model": _LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        # extract the first JSON object from the response
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return 0.0
        parsed = json.loads(raw[start:end])
        return float(parsed.get("score", 0.0))
    except Exception as exc:
        print(f"  [WARN] _llm_score failed: {exc}", file=sys.stderr)
        return 0.0


# ── API caller ────────────────────────────────────────────────────────────────

def _query_api(question: str, top_k: int = 5) -> dict:
    resp = httpx.post(
        f"{_API_BASE}/query",
        json={"question": question, "top_k": top_k},
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()


# ── context extractor ─────────────────────────────────────────────────────────

def _extract_contexts(api_result: dict) -> list[str]:
    """
    Build context strings from query response sources.
    Prefixes figure and table chunks so the LLM judge knows the content type.
    """
    contexts = []
    for s in api_result.get("sources", []):
        chunk_type  = s.get("chunk_type", "text")
        text        = s.get("text", "")
        figure_type = s.get("figure_type") or ""
        if chunk_type == "figure":
            prefix = f"[Figure: {figure_type}] " if figure_type else "[Figure] "
        elif chunk_type == "table":
            prefix = "[Table] "
        else:
            prefix = ""
        contexts.append(prefix + text)
    return contexts


# ── metrics ───────────────────────────────────────────────────────────────────

def score_faithfulness(question: str, answer: str, contexts: list[str]) -> float:
    ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts)) or "No context."
    prompt = (
        "You are an evaluation judge for a RAG system that answers using uploaded "
        "documents as primary source and general knowledge as a supplement.\n"
        "Score how faithful the answer is to the retrieved context.\n"
        "Score 1.0: claims that come from the context are accurately represented; "
        "general knowledge supplements are reasonable and don't contradict context.\n"
        "Score 0.5: some context claims are accurate, but some are distorted or "
        "the answer contradicts parts of the context.\n"
        "Score 0.0: the answer contradicts or ignores the context entirely, or "
        "general knowledge claims directly conflict with what the context says.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Answer: {answer}\n\n"
        'Respond ONLY with valid JSON: {"score": <float 0.0-1.0>, "reason": "<str>"}'
    )
    return _llm_score(prompt)


def score_answer_relevancy(question: str, answer: str) -> float:
    prompt = (
        "You are an evaluation judge.\n"
        "Score how well the answer addresses the question asked.\n"
        "Score 1.0: the answer directly and completely addresses the question.\n"
        "Score 0.5: the answer is partially relevant but misses key aspects.\n"
        "Score 0.0: the answer is off-topic or does not address the question.\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        'Respond ONLY with valid JSON: {"score": <float 0.0-1.0>, "reason": "<str>"}'
    )
    return _llm_score(prompt)


def score_context_precision(question: str, contexts: list[str]) -> float:
    if not contexts:
        return 0.0
    ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    prompt = (
        "You are an evaluation judge.\n"
        "Score the precision of retrieved context — how many of the retrieved passages "
        "are actually relevant to answering the question.\n"
        "Score 1.0: all retrieved passages are relevant.\n"
        "Score 0.5: roughly half are relevant.\n"
        "Score 0.0: none are relevant.\n\n"
        f"Question: {question}\n\n"
        f"Retrieved context:\n{ctx}\n\n"
        'Respond ONLY with valid JSON: {"score": <float 0.0-1.0>, "reason": "<str>"}'
    )
    return _llm_score(prompt)


def score_context_recall(
    question: str,
    contexts: list[str],
    ground_truth: str,
) -> float:
    if not contexts:
        return 0.0
    ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    prompt = (
        "You are an evaluation judge.\n"
        "Score recall: how much of the information needed to answer the question "
        "(as indicated by the ground truth) is present in the retrieved context.\n"
        "Score 1.0: the context contains everything needed to produce the ground truth answer.\n"
        "Score 0.5: the context contains some of the required information.\n"
        "Score 0.0: the context is missing the key information in the ground truth.\n\n"
        f"Question: {question}\n\n"
        f"Ground truth answer: {ground_truth}\n\n"
        f"Retrieved context:\n{ctx}\n\n"
        'Respond ONLY with valid JSON: {"score": <float 0.0-1.0>, "reason": "<str>"}'
    )
    return _llm_score(prompt)


def score_visual_grounding(question: str, answer: str, contexts: list[str]) -> float:
    """
    Checks whether visual claims in the answer are grounded in retrieved figure descriptions.
    Returns 1.0 if no figure context was retrieved (metric not applicable).
    Returns 1.0 if the answer makes no visual claims.
    """
    figure_contexts = [c for c in contexts if c.startswith("[Figure")]
    if not figure_contexts:
        return 1.0

    ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(figure_contexts))
    prompt = (
        "You are an evaluation judge for a visual RAG system.\n"
        "Score whether the answer's claims about visual content (charts, diagrams, "
        "figures, images) are supported by the figure descriptions in the context.\n"
        "Score 1.0: every visual claim in the answer is grounded in a figure description.\n"
        "Score 0.5: some visual claims are grounded, some are not.\n"
        "Score 0.0: the answer makes visual claims not present in any figure description, "
        "or fabricates visual details.\n"
        "If the answer makes no visual claims, score 1.0.\n\n"
        f"Question: {question}\n\n"
        f"Figure descriptions from retrieved context:\n{ctx}\n\n"
        f"Answer: {answer}\n\n"
        'Respond ONLY with valid JSON: {"score": <float 0.0-1.0>, "reason": "<str>"}'
    )
    return _llm_score(prompt)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VisionRAG RAGAS evaluation harness")
    parser.add_argument("--subset", type=int, default=None, help="Run on first N questions only")
    parser.add_argument("--output", type=str, default=str(_DEFAULT_OUTPUT))
    args = parser.parse_args()

    if not _DATASET_PATH.exists():
        print(f"ERROR: dataset not found at {_DATASET_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(_DATASET_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    golden = raw["dataset"] if isinstance(raw, dict) and "dataset" in raw else raw
    if args.subset:
        golden = golden[: args.subset]

    print(f"Running eval on {len(golden)} questions -> {args.output}")

    totals = {
        "faithfulness":      0.0,
        "answer_relevancy":  0.0,
        "context_precision": 0.0,
        "context_recall":    0.0,
        "visual_grounding":  0.0,
    }
    per_question = []

    for i, entry in enumerate(golden, 1):
        question     = entry["question"]
        ground_truth = entry["ground_truth"]
        print(f"  [{i}/{len(golden)}] {question[:70]}...")

        try:
            api_result = _query_api(question)
        except Exception as exc:
            print(f"  [WARN] API call failed: {exc}", file=sys.stderr)
            api_result = {"answer": "", "sources": []}

        answer   = api_result.get("answer", "")
        contexts = _extract_contexts(api_result)

        faith  = score_faithfulness(question, answer, contexts)
        relev  = score_answer_relevancy(question, answer)
        prec   = score_context_precision(question, contexts)
        recall = score_context_recall(question, contexts, ground_truth)
        vg     = score_visual_grounding(question, answer, contexts)

        scores = {
            "faithfulness":      faith,
            "answer_relevancy":  relev,
            "context_precision": prec,
            "context_recall":    recall,
            "visual_grounding":  vg,
        }
        for k, v in scores.items():
            totals[k] += v

        per_question.append({
            "question":        question,
            "ground_truth":    ground_truth,
            "answer":          answer,
            "source_types":    [s.get("chunk_type", "text") for s in api_result.get("sources", [])],
            "retrieval_types": list({s.get("type", "vector") for s in api_result.get("sources", [])}),
            "scores":          {k: round(v, 4) for k, v in scores.items()},
        })

        print(
            f"      faith={faith:.2f} relev={relev:.2f} "
            f"prec={prec:.2f} recall={recall:.2f} vg={vg:.2f}"
        )

    n = len(golden)
    averages = {k: round(v / n, 4) for k, v in totals.items()}

    print("\n-- Averages ---------------------------------------------")
    for k, v in averages.items():
        print(f"  {k:<22} {v:.4f}")

    output = {
        "model":        _LLM_MODEL,
        "api_base":     _API_BASE,
        "n_questions":  n,
        "averages":     averages,
        "per_question": per_question,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
