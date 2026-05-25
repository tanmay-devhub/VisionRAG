# ── VisionRAG Phase 1 / 2 ──────────────────────────────────────────────────
# Step 1: LLM prompt fix
# LLM behaviour: uploaded content is primary source; general knowledge fills gaps only
# RAG improvement: prevents silent context-ignore; answers stay grounded in retrieved chunks
# ──────────────────────────────────────────────────────────────────────────

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
    """
    Answer using uploaded content as the primary source, supplemented by
    general knowledge where the context is incomplete or silent on a sub-point.
    Uploaded content always takes priority — general knowledge fills gaps only.
    """
    numbered_context = "\n".join(
        f"[{i + 1}] {chunk}" for i, chunk in enumerate(context_chunks)
    )
    prompt = (
        "You are a knowledgeable assistant with access to both uploaded documents "
        "and broad general knowledge.\n\n"
        "The uploaded content below may include text passages, figure descriptions, "
        "and table data extracted from documents or images.\n\n"
        "Priority rules:\n"
        "1. ALWAYS lead with information from the uploaded content when it is relevant. "
        "Cite it explicitly — e.g. 'According to the uploaded document...' or "
        "'The figure shows...' or 'The table indicates...'.\n"
        "2. After covering what the uploaded content says, you MAY add relevant "
        "general knowledge to fill gaps, provide background, or extend the answer — "
        "but clearly signal the shift: 'More generally...' or 'For additional context...'.\n"
        "3. If the uploaded content fully answers the question, do not pad with "
        "general knowledge — a focused answer is better.\n"
        "4. If the uploaded content is completely unrelated to the question, say so "
        "briefly ('The uploaded content doesn't directly address this') and then "
        "answer from general knowledge.\n"
        "5. When referencing a figure, mention what it shows. "
        "When referencing a table, summarise its key data.\n\n"
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
