"""Standalone image parser.

An uploaded image becomes one figure block carrying the raw bytes plus text
derived from OCR and/or a Gemini vision description. That text is what gets
embedded, so the image is retrievable by Naive/Hybrid RAG; the bytes are what
Multimodal RAG hands back to Gemini at answer time.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.core.logging import get_logger
from app.core.text import normalize_whitespace
from app.ingestion.ocr import extract_text_from_image
from app.ingestion.parsers.base import ContentBlock, DocumentParser, ParsedDocument

log = get_logger("ragx.parse.image")

MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".tiff": "image/tiff",
}


class ImageParser(DocumentParser):
    name = "image"
    extensions = (".png", ".jpg", ".jpeg", ".webp", ".tiff")

    async def parse(self, data: bytes, filename: str) -> ParsedDocument:
        extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ".png"
        mime = MIME_BY_EXT.get(extension, "image/png")

        dimensions = await asyncio.to_thread(self._dimensions, data)
        result = await extract_text_from_image(data, mime, use_vision_fallback=True)
        text = normalize_whitespace(result["text"])

        parsed = ParsedDocument(page_count=1, title=filename)
        parsed.metadata = {
            "mime_type": mime,
            "width": dimensions.get("width"),
            "height": dimensions.get("height"),
            "ocr_method": result["method"],
            "has_text_layer": bool(result["ocr_text"]),
        }
        if not text:
            text = f"Image file '{filename}' with no readable text content."
            parsed.warnings.append(
                "No text could be extracted from this image. Install Tesseract or configure "
                "GEMINI_API_KEY so images become searchable."
            )

        parsed.blocks.append(
            ContentBlock(
                text=text,
                modality="image",
                page_number=1,
                section=filename,
                section_path=[filename],
                figure_label="Figure 1",
                order=0,
                asset_bytes=data,
                asset_mime=mime,
                metadata={
                    "ocr_text": result["ocr_text"][:2000],
                    "description": result["description"][:2000],
                    "extraction_method": result["method"],
                },
            )
        )
        return parsed

    @staticmethod
    def _dimensions(data: bytes) -> dict[str, Any]:
        try:
            import io  # noqa: PLC0415

            from PIL import Image  # noqa: PLC0415

            with Image.open(io.BytesIO(data)) as image:
                return {"width": image.width, "height": image.height, "mode": image.mode}
        except Exception:
            return {}


__all__ = ["ImageParser"]
