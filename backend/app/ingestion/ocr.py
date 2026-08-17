"""OCR support.

Two independent paths, tried in order:

1. **Tesseract** via ``pytesseract``, when the binary is installed. Free, local,
   and purely a vision utility -- it is not a language model, so it does not
   conflict with RAGX's cloud-only LLM policy.
2. **Gemini vision**, used when Tesseract is unavailable or returns nothing
   useful for an image that clearly carries text.

If neither path is available the pipeline records a warning on the document and
continues with whatever text layer exists, rather than failing ingestion.
"""

from __future__ import annotations

import asyncio
import io
from functools import lru_cache
from typing import Any

from app.core.logging import get_logger

log = get_logger("ragx.ocr")


# Default install locations. The Windows installer does not add Tesseract to
# PATH, so pytesseract cannot find it even when it is installed -- checking
# these avoids making the user edit their PATH and restart.
_TESSERACT_CANDIDATES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/opt/homebrew/bin/tesseract",
)


def _locate_tesseract() -> str | None:
    """Find the tesseract binary on PATH, or at a known install location."""
    import shutil  # noqa: PLC0415

    found = shutil.which("tesseract")
    if found:
        return found
    from pathlib import Path  # noqa: PLC0415

    for candidate in _TESSERACT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


@lru_cache(maxsize=1)
def tesseract_available() -> bool:
    try:
        import pytesseract  # noqa: PLC0415

        binary = _locate_tesseract()
        if binary:
            pytesseract.pytesseract.tesseract_cmd = binary

        version = pytesseract.get_tesseract_version()
        log.info("ocr.tesseract_ready", version=str(version), binary=binary or "PATH")
        return True
    except Exception as exc:
        log.info("ocr.tesseract_unavailable", detail=str(exc)[:160])
        return False


async def ocr_image_bytes(data: bytes) -> str:
    """Extract text from an image with Tesseract. Returns '' when unavailable."""
    if not tesseract_available():
        return ""

    def _run() -> str:
        import pytesseract  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        with Image.open(io.BytesIO(data)) as image:
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            return pytesseract.image_to_string(image)

    try:
        text = await asyncio.to_thread(_run)
        return (text or "").strip()
    except Exception as exc:
        log.warning("ocr.failed", error=str(exc)[:160])
        return ""


async def describe_image_with_gemini(data: bytes, mime_type: str = "image/png") -> str:
    """Ask Gemini to describe a figure/chart so it becomes retrievable text.

    This is what makes Multimodal RAG work for charts that contain no text
    layer: the description is embedded and indexed alongside the image itself.
    """
    from app.llm.base import ImagePart, Message  # noqa: PLC0415
    from app.llm.gateway import Purpose, get_gateway  # noqa: PLC0415
    from app.llm.prompts import IMAGE_DESCRIPTION_SYSTEM  # noqa: PLC0415

    gateway = get_gateway()
    if not gateway.any_configured:
        return ""
    try:
        response = await gateway.complete(
            [
                Message.system(IMAGE_DESCRIPTION_SYSTEM),
                Message.user(
                    "Describe this visual for retrieval indexing.",
                    images=[ImagePart(data=data, mime_type=mime_type, label="figure")],
                ),
            ],
            Purpose.MULTIMODAL,
            temperature=0.1,
            max_output_tokens=320,
        )
        text = response.text.strip()
        return "" if text.lower().startswith("unreadable") else text
    except Exception as exc:
        log.warning("ocr.gemini_description_failed", error=str(exc)[:160])
        return ""


async def extract_text_from_image(
    data: bytes, mime_type: str = "image/png", use_vision_fallback: bool = True
) -> dict[str, Any]:
    """Combined extraction: OCR text plus an optional visual description."""
    ocr_text = await ocr_image_bytes(data)
    description = ""
    if use_vision_fallback and len(ocr_text) < 40:
        description = await describe_image_with_gemini(data, mime_type)
    parts = [p for p in (ocr_text, description) if p]
    return {
        "text": "\n\n".join(parts),
        "ocr_text": ocr_text,
        "description": description,
        "method": "tesseract+gemini" if ocr_text and description else ("tesseract" if ocr_text else ("gemini" if description else "none")),
    }


__all__ = [
    "tesseract_available",
    "ocr_image_bytes",
    "describe_image_with_gemini",
    "extract_text_from_image",
]
