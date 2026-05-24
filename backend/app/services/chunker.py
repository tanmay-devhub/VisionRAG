import logging

from app.services.vision import VisionService
from app.services.figure_extractor import FigureExtractor

logger = logging.getLogger(__name__)

_splitter = None  # RecursiveCharacterTextSplitter, lazy


def _get_splitter():
    global _splitter
    if _splitter is None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # lazy
        _splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return _splitter


class MultimodalChunker:
    def __init__(self) -> None:
        self._vision = VisionService()
        self._extractor = FigureExtractor()

    def chunk_document(self, pdf_path: str, doc_id: str) -> list[dict]:
        import fitz  # lazy — avoids PyMuPDF DLL scan at startup
        chunks: list[dict] = []

        # ── text chunks ───────────────────────────────────────────────────────
        try:
            doc = fitz.open(pdf_path)
            full_text = "\n".join(page.get_text() for page in doc)
            doc.close()
        except Exception as exc:
            logger.error("Text extraction failed for %s: %s", pdf_path, exc)
            full_text = ""

        splitter = _get_splitter()
        text_splits = splitter.split_text(full_text) if full_text.strip() else []
        for i, text in enumerate(text_splits):
            chunks.append({
                "text": text,
                "chunk_index": i,
                "chunk_type": "text",
                "image_path": None,
                "image_url": None,
                "figure_type": None,
                "caption": None,
                "page_number": None,
                "extra_entities": [],
                "extra_relationships": [],
            })

        next_index = len(chunks)

        # ── figure + table chunks ─────────────────────────────────────────────
        extracted = self._extractor.extract_all(pdf_path, doc_id)

        for item in extracted:
            if item["type"] == "figure":
                try:
                    vision_result = self._vision.describe_figure(
                        item["image_path"], item.get("context", "")
                    )
                except Exception as exc:
                    logger.warning("Vision failed for %s: %s", item["image_path"], exc)
                    vision_result = {
                        "description": "", "entities": [], "relationships": [],
                        "figure_type": "image", "caption": "",
                    }

                description = vision_result.get("description", "")
                if not description:
                    continue

                context_snippet = item.get("context", "")[:300].strip()
                chunk_text = f"{context_snippet}\n{description}" if context_snippet else description

                chunks.append({
                    "text": chunk_text,
                    "chunk_index": next_index,
                    "chunk_type": "figure",
                    "image_path": item["image_path"],
                    "image_url": item["image_url"],
                    "figure_type": vision_result.get("figure_type", "image"),
                    "caption": vision_result.get("caption", ""),
                    "page_number": item.get("page_number"),
                    "extra_entities": vision_result.get("entities", []),
                    "extra_relationships": vision_result.get("relationships", []),
                })
                next_index += 1

            elif item["type"] == "table":
                chunks.append({
                    "text": item["text"],
                    "chunk_index": next_index,
                    "chunk_type": "table",
                    "image_path": None,
                    "image_url": None,
                    "figure_type": "table",
                    "caption": None,
                    "page_number": item.get("page_number"),
                    "extra_entities": [],
                    "extra_relationships": [],
                })
                next_index += 1

        return chunks
