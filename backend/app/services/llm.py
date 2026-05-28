# ── VisionRAG Phase 4: Ollama Cloud Video ─────────────────────────────────────
# File: backend/app/services/llm.py
# Changes: Fix 2 — LLM prompt grounding for partial relevance + video summaries
# Image pipeline: UNTOUCHED
# ─────────────────────────────────────────────────────────────────────────────

import os

_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_LLM_MODEL       = os.getenv("LLM_MODEL", "llama3.2")

_llm_instance = None  # OllamaLLM, lazy


def get_llm():
    global _llm_instance
    if _llm_instance is None:
        from langchain_ollama import OllamaLLM  # lazy — avoids langchain init at startup
        _llm_instance = OllamaLLM(base_url=_OLLAMA_BASE_URL, model=_LLM_MODEL)
    return _llm_instance


def generate_answer(question: str, context_chunks: list[str]) -> str:
    numbered_context = "\n".join(
        f"[{i + 1}] {chunk}" for i, chunk in enumerate(context_chunks)
    )
    prompt = (
        "You are a knowledgeable assistant with access to both uploaded content "
        "and broad general knowledge.\n\n"
        "The uploaded content below contains descriptions of images, video frames, "
        "video summaries, text passages, or table data from uploaded files.\n\n"
        "Priority rules — follow in order:\n\n"
        "1. ALWAYS lead with what the uploaded content explicitly shows or states. "
        "Cite directly: 'The video shows...', 'Frame [1] shows...', "
        "'According to the uploaded content...'\n\n"
        "2. PARTIAL RELEVANCE — if content shows the right scene but not the "
        "specific action asked about:\n"
        "   a. Describe what the content DOES show.\n"
        "   b. State the gap: 'The content doesn't explicitly show [X].'\n"
        "   c. Only then add general knowledge, prefixed: 'More generally...'\n"
        "   NEVER fabricate actions, events, or story details not in the content.\n\n"
        "3. FULL RELEVANCE — if content directly answers, use it only.\n\n"
        "4. NO RELEVANCE — say 'The uploaded content doesn't address this' then "
        "use general knowledge.\n\n"
        "5. VIDEO — describe the narrative arc and key events from video summaries. "
        "For individual frames, describe only what is visually present.\n\n"
        "6. FIGURES AND TABLES — describe what they show or summarise key data.\n\n"
        f"Uploaded content:\n{numbered_context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    llm = get_llm()
    return llm.invoke(prompt)


def generate_general_answer(question: str) -> str:
    """
    Called when no relevant content was retrieved from uploaded documents.
    Answers from general knowledge but tells the user no uploads matched.
    """
    prompt = (
        "You are a knowledgeable assistant. No relevant content was found in the "
        "user's uploaded documents for this question, so answer from general knowledge.\n\n"
        "Start your answer with a brief note that no matching uploaded content was found, "
        "then give a complete and accurate answer.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    llm = get_llm()
    return llm.invoke(prompt)
