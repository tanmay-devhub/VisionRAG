import os
from langchain_ollama import OllamaLLM

_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_LLM_MODEL       = os.getenv("LLM_MODEL", "llama3.2")

_llm_instance: OllamaLLM | None = None


def get_llm() -> OllamaLLM:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = OllamaLLM(base_url=_OLLAMA_BASE_URL, model=_LLM_MODEL)
    return _llm_instance


def generate_answer(question: str, context_chunks: list[str]) -> str:
    numbered_context = "\n".join(
        f"[{i + 1}] {chunk}" for i, chunk in enumerate(context_chunks)
    )
    prompt = (
        "You are a helpful, knowledgeable AI assistant.\n"
        "You have access to text passages, figure descriptions, and table data from the document.\n"
        "When referencing a figure, mention it explicitly.\n"
        "When referencing a table, summarize the key data from it.\n"
        "Follow these rules:\n"
        "1. If the context directly answers the question, use it and cite it.\n"
        "2. If the context is only loosely related or not relevant, IGNORE it and answer from your general knowledge.\n"
        "3. Never say 'I don't have enough context' for general knowledge questions — just answer them.\n"
        "4. Always give a complete, helpful answer.\n\n"
        f"Document context:\n{numbered_context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    llm = get_llm()
    return llm.invoke(prompt)


def generate_general_answer(question: str) -> str:
    prompt = (
        "You are a helpful, knowledgeable AI assistant. "
        "Answer the following question as accurately and helpfully as possible.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    llm = get_llm()
    return llm.invoke(prompt)
