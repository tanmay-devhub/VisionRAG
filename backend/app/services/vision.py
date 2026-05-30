# ── VisionRAG Phase 4: Ollama Cloud Video ─────────────────────────────────────
# File: backend/app/services/vision.py
# Changes: NONE — this file is IMAGE-ONLY. Video uses video_describer.py.
# Image pipeline: UNTOUCHED
# ─────────────────────────────────────────────────────────────────────────────

import os
import io
import json
import base64
import logging
import re
import time
import threading

logger = logging.getLogger(__name__)

# Ollama processes one vision request at a time — serialize all calls
_ollama_lock = threading.Semaphore(1)

_VISION_BACKEND      = os.getenv("VISION_BACKEND",       "ollama")
_GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY",       "")
_GEMINI_MODEL        = os.getenv("GEMINI_MODEL",         "gemini-2.0-flash")
_OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY",       "")
_OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL",  "gpt-4o-mini")
_OLLAMA_BASE_URL     = os.getenv("OLLAMA_BASE_URL",      "http://localhost:11434")
_OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL",  "qwen3-vl:8b")
_PALIGEMMA_MODEL_ID  = os.getenv("PALIGEMMA_MODEL",      "google/paligemma2-3b-ft-docci-448")
_HF_TOKEN            = os.getenv("HF_TOKEN",             "")

_EMPTY_RESULT = {
    "description": "",
    "entities": [],
    "relationships": [],
    "figure_type": "image",
    "caption": "",
}

# Prompt for backends that understand instruction-following (Gemini, OpenAI, Ollama)
_PROMPT_TEMPLATE = (
    "You are analyzing an image or figure.\n"
    "The surrounding text context is: {context}\n\n"
    "IMPORTANT RULES:\n"
    "- If the image contains solid-colour silhouettes or bounding-box overlays, "
    "describe the underlying real objects, not the colour fills.\n"
    "- If the image contains ANY visible text (signs, labels, menus, code, titles, "
    "captions, watermarks, handwriting), transcribe ALL of it exactly as written.\n"
    "- If the image shows a menu, price list, or table: list EVERY item with its "
    "associated numbers/prices/values. Do not summarize — transcribe completely.\n"
    "- If text is in multiple languages, transcribe ALL languages present.\n"
    "- If the image shows a chart or graph: describe the type, axes, labels, "
    "data values, trends, and legend entries.\n"
    "- If the image shows code: identify the language, filename if visible, "
    "and describe the key functions/variables/logic.\n\n"
    "Analyze this image thoroughly. Return ONLY a valid JSON object "
    "with no markdown, no backticks, no preamble. Schema:\n"
    '{{\n'
    '  "description": "comprehensive description — as long as needed to capture ALL '
    'visible content. Include every readable text, every data point, every entity. '
    '5-20 sentences depending on image complexity.",\n'
    '  "entities": ["every distinct object, person, text element, brand, concept visible"],\n'
    '  "relationships": [\n'
    '    {{"from": "Entity A", "type": "RELATES_TO", "to": "Entity B"}}\n'
    '  ],\n'
    '  "figure_type": "chart|diagram|flowchart|table|equation|image|menu|code|sign",\n'
    '  "caption": "all visible caption/title/header text — empty string if none"\n'
    '}}'
)

# ── PaliGemma singleton ───────────────────────────────────────────────────────

_paligemma_model: object = None
_paligemma_proc:  object = None


def _get_paligemma():
    global _paligemma_model, _paligemma_proc
    if _paligemma_model is None:
        from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
        import torch

        logger.info("Loading PaliGemma model %s …", _PALIGEMMA_MODEL_ID)
        hf_kw = {"token": _HF_TOKEN} if _HF_TOKEN else {}

        _paligemma_proc = AutoProcessor.from_pretrained(_PALIGEMMA_MODEL_ID, **hf_kw)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype  = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        _paligemma_model = PaliGemmaForConditionalGeneration.from_pretrained(
            _PALIGEMMA_MODEL_ID,
            torch_dtype=dtype,
            **hf_kw,
        ).to(device).eval()

        logger.info("PaliGemma ready on %s (dtype=%s)", device, dtype)

    return _paligemma_model, _paligemma_proc


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    """
    Parse a JSON string from a vision model response.
    Always returns a complete dict with all five keys.
    Never raises — returns _EMPTY_RESULT on any failure.
    Applies figure_type fallback via _infer_figure_type (Fix 2).
    """
    if not raw or not raw.strip():
        return dict(_EMPTY_RESULT)

    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                from json_repair import repair_json
                parsed = json.loads(repair_json(cleaned))
            except Exception:
                return dict(_EMPTY_RESULT)

        if not isinstance(parsed, dict):
            return dict(_EMPTY_RESULT)

        result = {
            "description":   str(parsed.get("description") or "").strip(),
            "entities":      parsed.get("entities") or [],
            "relationships": parsed.get("relationships") or [],
            "figure_type":   str(parsed.get("figure_type") or "").strip().lower(),
            "caption":       str(parsed.get("caption") or "").strip(),
        }

        if not isinstance(result["entities"], list):
            result["entities"] = []
        result["entities"] = [
            str(e).strip() for e in result["entities"] if e and str(e).strip()
        ]

        if not isinstance(result["relationships"], list):
            result["relationships"] = []
        result["relationships"] = [
            r for r in result["relationships"]
            if isinstance(r, dict) and "from" in r and "to" in r
        ]

        # Fix 2: fall back to inferred type if model returned empty/unknown value
        _KNOWN_TYPES = {"chart", "diagram", "flowchart", "table", "equation", "image"}
        if result["figure_type"] not in _KNOWN_TYPES:
            result["figure_type"] = _infer_figure_type(result["description"])

        return result

    except Exception as exc:
        logger.warning("_parse_json failed: %s | raw[:200]=%s", exc, raw[:200])
        return dict(_EMPTY_RESULT)


def _infer_figure_type(text: str) -> str:
    tl = text.lower()
    if any(w in tl for w in ["chart", "bar chart", "line chart", "pie chart",
                               "histogram", "scatter", "plot", "axis", "axes", "graph"]):
        return "chart"
    if any(w in tl for w in ["flowchart", "flow chart", "flow diagram",
                               "diagram", "arrow", "process", "node", "network"]):
        return "diagram"
    if any(w in tl for w in ["equation", "formula", "math", "formula"]):
        return "equation"
    if any(w in tl for w in ["table", "row", "column", "cell", "header"]):
        return "table"
    return "image"


# ── main service ──────────────────────────────────────────────────────────────

class VisionService:
    def __init__(self) -> None:
        if _VISION_BACKEND == "gemini" and not _GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY is not set — figure analysis will be skipped")
        elif _VISION_BACKEND == "openai" and not _OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY is not set — figure analysis will be skipped")
        elif _VISION_BACKEND == "paligemma":
            logger.info("Vision backend: PaliGemma model=%s", _PALIGEMMA_MODEL_ID)
            if not _HF_TOKEN:
                logger.warning(
                    "HF_TOKEN not set — model download may fail for gated models; "
                    "accept the license at https://huggingface.co/%s first",
                    _PALIGEMMA_MODEL_ID,
                )
        else:
            logger.info("Vision backend: Ollama model=%s", _OLLAMA_VISION_MODEL)

    def describe_figure(self, image_path: str, context: str) -> dict:
        try:
            with open(image_path, "rb") as fh:
                image_bytes = fh.read()
        except Exception as exc:
            logger.error("Cannot read image %s: %s", image_path, exc)
            return dict(_EMPTY_RESULT)

        if _VISION_BACKEND == "paligemma":
            return self._describe_paligemma(image_bytes, image_path, context)

        prompt = _PROMPT_TEMPLATE.format(context=context[:500])
        if _VISION_BACKEND == "gemini":
            return self._describe_gemini(image_bytes, image_path, prompt)
        if _VISION_BACKEND == "openai":
            return self._describe_openai(image_bytes, image_path, prompt)
        return self._describe_ollama(image_bytes, image_path, prompt)

    # ── PaliGemma ─────────────────────────────────────────────────────────────

    def _describe_paligemma(self, image_bytes: bytes, image_path: str, context: str) -> dict:
        try:
            import torch
            from PIL import Image as PILImage

            model, processor = _get_paligemma()
            image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")

            prompt = (
                "Describe this technical figure in full detail. "
                "Include all data values, labels, axes, trends, legends, and key findings."
            )
            if context.strip():
                prompt += f" Document context: {context[:300]}"

            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                )

            input_len   = inputs["input_ids"].shape[-1]
            description = processor.decode(
                output_ids[0][input_len:],
                skip_special_tokens=True,
            ).strip()

            if not description:
                return dict(_EMPTY_RESULT)

            return {
                "description":   description,
                "entities":      [],
                "relationships": [],
                "figure_type":   _infer_figure_type(description),
                "caption":       "",
            }

        except Exception as exc:
            logger.error("PaliGemma vision error for %s: %s", image_path, exc)
            return dict(_EMPTY_RESULT)

    # ── Gemini ────────────────────────────────────────────────────────────────

    def _describe_gemini(self, image_bytes: bytes, image_path: str, prompt: str) -> dict:
        if not _GEMINI_API_KEY:
            return dict(_EMPTY_RESULT)

        ext  = os.path.splitext(image_path)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"

        for attempt in range(3):
            try:
                from google import genai
                from google.genai import types

                client   = genai.Client(api_key=_GEMINI_API_KEY)
                response = client.models.generate_content(
                    model=_GEMINI_MODEL,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime),
                        prompt,
                    ],
                )
                try:
                    from app.services.gemini_usage_tracker import record_request
                    record_request()
                except Exception:
                    pass
                return _parse_json(response.text)

            except Exception as exc:
                err_str = str(exc)
                if "429" in err_str and attempt < 2:
                    wait = 15 * (attempt + 1)
                    logger.warning("Gemini rate limit hit, retrying in %ds…", wait)
                    time.sleep(wait)
                    continue
                logger.error("Gemini vision error for %s: %s", image_path, exc)
                return dict(_EMPTY_RESULT)

        return dict(_EMPTY_RESULT)

    # ── OpenAI ────────────────────────────────────────────────────────────────

    def _describe_openai(self, image_bytes: bytes, image_path: str, prompt: str) -> dict:
        if not _OPENAI_API_KEY:
            return dict(_EMPTY_RESULT)

        ext      = os.path.splitext(image_path)[1].lower()
        mime     = "image/png" if ext == ".png" else "image/jpeg"
        b64      = base64.b64encode(image_bytes).decode()
        data_url = f"data:{mime};base64,{b64}"

        for attempt in range(3):
            try:
                from openai import OpenAI
                client = OpenAI(api_key=_OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model=_OPENAI_VISION_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                    max_tokens=1024,
                )
                raw = response.choices[0].message.content or ""
                return _parse_json(raw)

            except Exception as exc:
                err_str = str(exc)
                if "429" in err_str and attempt < 2:
                    wait = 15 * (attempt + 1)
                    logger.warning("OpenAI rate limit hit, retrying in %ds…", wait)
                    time.sleep(wait)
                    continue
                logger.error("OpenAI vision error for %s: %s", image_path, exc)
                return dict(_EMPTY_RESULT)

        return dict(_EMPTY_RESULT)

    # ── Ollama ────────────────────────────────────────────────────────────────

    @staticmethod
    def _resize_for_ollama(image_bytes: bytes, max_dim: int = 1280) -> bytes:
        """Resize image so its longest side is at most max_dim pixels.
        Large images cause VRAM exhaustion in Ollama and 500 errors."""
        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _describe_ollama(self, image_bytes: bytes, image_path: str, prompt: str) -> dict:
        import httpx

        # Resize before encoding — prevents large payloads
        try:
            image_bytes = self._resize_for_ollama(image_bytes)
        except Exception as exc:
            logger.warning("Could not resize %s: %s — sending original", image_path, exc)

        b64     = base64.b64encode(image_bytes).decode()

        # Use /api/chat (NOT /api/generate) — cloud models ignore images in /api/generate
        payload = {
            "model":  _OLLAMA_VISION_MODEL,
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": [b64],
            }],
            "stream": False,
        }

        with _ollama_lock:
            for attempt in range(3):
                try:
                    resp = httpx.post(
                        f"{_OLLAMA_BASE_URL}/api/chat",
                        json=payload,
                        timeout=300.0,
                    )
                    resp.raise_for_status()
                    raw = resp.json().get("message", {}).get("content", "").strip()
                    if not raw:
                        raise ValueError("Ollama returned empty response")

                    # Try structured JSON first; fall back to raw text as description
                    try:
                        return _parse_json(raw)
                    except Exception:
                        logger.warning(
                            "Could not parse JSON from Ollama for %s — using raw text as description",
                            image_path,
                        )
                        return {
                            "description":   raw,
                            "entities":      [],
                            "relationships": [],
                            "figure_type":   _infer_figure_type(raw),
                            "caption":       "",
                        }

                except httpx.HTTPStatusError as exc:
                    body = ""
                    try:
                        body = exc.response.text[:200]
                    except Exception:
                        pass
                    if exc.response.status_code in (500, 503) and attempt < 2:
                        wait = 15 * (attempt + 1)
                        logger.warning(
                            "Ollama %d for %s (attempt %d/3, body=%s), retrying in %ds...",
                            exc.response.status_code, image_path, attempt + 1, body, wait,
                        )
                        time.sleep(wait)
                        continue
                    logger.error("Ollama vision error for %s: %s | %s", image_path, exc, body)
                    return dict(_EMPTY_RESULT)
                except Exception as exc:
                    if attempt < 2:
                        wait = 15 * (attempt + 1)
                        logger.warning(
                            "Ollama error for %s (attempt %d/3): %s, retrying in %ds...",
                            image_path, attempt + 1, exc, wait,
                        )
                        time.sleep(wait)
                        continue
                    logger.error("Ollama vision error for %s: %s", image_path, exc)
                    return dict(_EMPTY_RESULT)

        return dict(_EMPTY_RESULT)
