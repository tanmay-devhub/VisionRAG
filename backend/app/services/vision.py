import os
import io
import json
import base64
import logging
import re
import time

logger = logging.getLogger(__name__)

_VISION_BACKEND      = os.getenv("VISION_BACKEND",       "ollama")
_GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY",       "")
_GEMINI_MODEL        = os.getenv("GEMINI_MODEL",         "gemini-2.0-flash")
_OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY",       "")
_OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL",  "gpt-4o-mini")
_OLLAMA_BASE_URL     = os.getenv("OLLAMA_BASE_URL",      "http://localhost:11434")
_OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL",  "qwen2.5vl:7b")
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
    "You are analyzing a figure extracted from a technical or scientific PDF.\n"
    "The surrounding text context is: {context}\n"
    "Analyze this figure carefully. Return ONLY a valid JSON object "
    "with no markdown, no backticks, no preamble. Schema:\n"
    '{{\n'
    '  "description": "detailed 3-5 sentence description of what this shows",\n'
    '  "entities": ["key concept 1", "key concept 2"],\n'
    '  "relationships": [\n'
    '    {{"from": "Entity A", "type": "RELATES_TO", "to": "Entity B"}}\n'
    '  ],\n'
    '  "figure_type": "chart|diagram|flowchart|table|equation|image",\n'
    '  "caption": "visible caption text or empty string"\n'
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
    cleaned = re.sub(r"^```(?:json)?", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"```$", "", cleaned.strip())
    return json.loads(cleaned.strip())


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

    def _describe_ollama(self, image_bytes: bytes, image_path: str, prompt: str) -> dict:
        try:
            import httpx
            b64     = base64.b64encode(image_bytes).decode()
            payload = {
                "model":  _OLLAMA_VISION_MODEL,
                "prompt": prompt,
                "images": [b64],
                "stream": False,
            }
            resp = httpx.post(
                f"{_OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=120.0,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
            return _parse_json(raw)
        except Exception as exc:
            logger.error("Ollama vision error for %s: %s", image_path, exc)
            return dict(_EMPTY_RESULT)
