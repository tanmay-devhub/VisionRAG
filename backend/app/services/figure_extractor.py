import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_BACKEND_ROOT      = Path(__file__).resolve().parent.parent.parent
_raw_figures       = os.getenv("FIGURES_DIR", "../static/figures")
_FIGURES_DIR       = str((_BACKEND_ROOT / _raw_figures).resolve())
_FIGURES_SERVE_URL = os.getenv("FIGURES_SERVE_URL", "http://localhost:8081/figures")

# Pages with more drawing paths than this threshold contain charts/diagrams.
_DRAWING_THRESHOLD = int(os.getenv("DRAWING_THRESHOLD", "15"))


class FigureExtractor:
    def extract_all(self, pdf_path: str, doc_id: str) -> list[dict]:
        import fitz  # lazy — avoids PyMuPDF DLL scan at startup
        results: list[dict] = []
        os.makedirs(_FIGURES_DIR, exist_ok=True)

        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:
            logger.error("Cannot open PDF %s: %s", pdf_path, exc)
            return results

        try:
            import pdfplumber
            plumber_doc = pdfplumber.open(pdf_path)
            plumber_pages = plumber_doc.pages
        except Exception as exc:
            logger.warning("pdfplumber failed to open %s: %s", pdf_path, exc)
            plumber_doc = None
            plumber_pages = []

        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            raster_extracted = False

            # ── raster images (embedded PNG/JPEG) ─────────────────────────────
            try:
                images = page.get_images(full=True)
                for img_idx, img_info in enumerate(images):
                    xref = img_info[0]
                    try:
                        img_data = doc.extract_image(xref)
                        w = img_data.get("width", 0)
                        h = img_data.get("height", 0)
                        if w <= 100 or h <= 100:
                            continue
                        ext = img_data.get("ext", "png")
                        filename = f"{doc_id}_p{page_num}_{img_idx}.{ext}"
                        save_path = os.path.join(_FIGURES_DIR, filename)
                        with open(save_path, "wb") as fh:
                            fh.write(img_data["image"])
                        results.append({
                            "type": "figure",
                            "image_path": save_path,
                            "image_url": f"{_FIGURES_SERVE_URL}/{filename}",
                            "context": page_text[:500],
                            "page_number": page_num,
                        })
                        raster_extracted = True
                    except Exception as exc:
                        logger.warning("Skipping image xref=%d on page %d: %s", xref, page_num, exc)
            except Exception as exc:
                logger.warning("Image extraction failed on page %d: %s", page_num, exc)

            # ── vector charts (render page to PNG when no raster image found) ─
            # Most PDF charts are drawn as vector paths, not embedded images.
            # We detect them by counting drawing commands; if there are enough
            # we render the full page at 2x resolution for the vision model.
            if not raster_extracted:
                try:
                    drawings = page.get_drawings()
                    if len(drawings) >= _DRAWING_THRESHOLD:
                        mat = fitz.Matrix(2, 2)  # noqa: F821  # 144 DPI
                        pix = page.get_pixmap(matrix=mat, alpha=False)
                        filename = f"{doc_id}_p{page_num}_render.png"
                        save_path = os.path.join(_FIGURES_DIR, filename)
                        pix.save(save_path)
                        logger.info(
                            "Rendered page %d as chart image (%d drawings): %s",
                            page_num, len(drawings), filename,
                        )
                        results.append({
                            "type": "figure",
                            "image_path": save_path,
                            "image_url": f"{_FIGURES_SERVE_URL}/{filename}",
                            "context": page_text[:500],
                            "page_number": page_num,
                        })
                except Exception as exc:
                    logger.warning("Page render failed on page %d: %s", page_num, exc)

            # ── tables ────────────────────────────────────────────────────────
            if plumber_pages and page_num < len(plumber_pages):
                try:
                    plumber_page = plumber_pages[page_num]
                    tables = plumber_page.extract_tables()
                    for tbl in tables:
                        if not tbl or len(tbl) < 2:
                            continue
                        md = _table_to_markdown(tbl)
                        if md:
                            results.append({
                                "type": "table",
                                "text": md,
                                "image_path": None,
                                "image_url": None,
                                "page_number": page_num,
                            })
                except Exception as exc:
                    logger.warning("Table extraction failed on page %d: %s", page_num, exc)

        if plumber_doc:
            try:
                plumber_doc.close()
            except Exception:
                pass

        return results


def _table_to_markdown(table: list[list]) -> str:
    if not table:
        return ""
    rows = []
    headers = [str(cell or "").strip() for cell in table[0]]
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in table[1:]:
        cells = [str(cell or "").strip() for cell in row]
        while len(cells) < len(headers):
            cells.append("")
        rows.append("| " + " | ".join(cells[:len(headers)]) + " |")
    return "\n".join(rows)
