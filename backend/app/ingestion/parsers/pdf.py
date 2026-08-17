"""PDF / research-paper parser.

Three extraction passes over each page:

1. **Layout-aware text** via PyMuPDF. Spans are grouped into blocks and font
   size + weight are used to detect headings, which builds the section path used
   for citations ("Methodology > Training setup").
2. **Tables** via pdfplumber, emitted as Markdown so they survive embedding and
   remain readable in an answer.
3. **Figures** -- embedded raster images above a size threshold are extracted,
   paired with the nearest ``Figure N`` caption, stored as artefacts and
   described so they are retrievable.

Pages whose text layer is thin (scanned papers) are rendered and sent through
OCR.
"""

from __future__ import annotations

import asyncio
import io
import re
import statistics
from typing import Any

from app.core.config import get_settings
from app.core.errors import IngestionError
from app.core.logging import get_logger
from app.core.text import normalize_whitespace
from app.ingestion.ocr import extract_text_from_image
from app.ingestion.parsers.base import ContentBlock, DocumentParser, ParsedDocument

log = get_logger("ragx.parse.pdf")

FIGURE_CAPTION = re.compile(r"^\s*(fig(?:ure)?\.?\s*(\d+[a-z]?))\b[.:]?\s*(.*)", re.IGNORECASE)
TABLE_CAPTION = re.compile(r"^\s*(table\s*(\d+[a-z]?))\b[.:]?\s*(.*)", re.IGNORECASE)

# Headings typical of research papers; used to normalise the section path.
CANONICAL_SECTIONS = (
    "abstract", "introduction", "related work", "background", "method", "methodology",
    "approach", "model", "architecture", "experiments", "experimental setup", "dataset",
    "datasets", "results", "evaluation", "discussion", "ablation", "limitations",
    "conclusion", "future work", "references", "appendix",
)

MIN_IMAGE_BYTES = 6_000
MIN_IMAGE_DIM = 90


class PDFParser(DocumentParser):
    name = "pdf"
    extensions = (".pdf",)

    async def parse(self, data: bytes, filename: str) -> ParsedDocument:
        settings = get_settings()
        parsed = await asyncio.to_thread(self._parse_sync, data, filename)

        # Async post-processing: OCR for thin pages, descriptions for figures.
        if settings.enable_ocr:
            await self._ocr_thin_pages(parsed, data, settings.ocr_min_chars_per_page)
        await self._describe_visuals(parsed)

        parsed.blocks.sort(key=lambda b: (b.page_number or 0, b.order))
        for index, block in enumerate(parsed.blocks):
            block.order = index
        return parsed

    # ------------------------------------------------------------------ sync
    def _parse_sync(self, data: bytes, filename: str) -> ParsedDocument:
        try:
            import fitz  # PyMuPDF  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise IngestionError("PyMuPDF is required to parse PDFs.") from exc

        try:
            document = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise IngestionError("This PDF could not be opened. It may be corrupt or encrypted.",
                                 detail=str(exc)[:200]) from exc

        parsed = ParsedDocument(page_count=document.page_count)
        raw_meta = document.metadata or {}
        parsed.metadata = {
            "producer": raw_meta.get("producer"),
            "creator": raw_meta.get("creator"),
            "author": raw_meta.get("author"),
            "subject": raw_meta.get("subject"),
            "keywords": raw_meta.get("keywords"),
            "creation_date": raw_meta.get("creationDate"),
            "pages": document.page_count,
        }
        parsed.title = (raw_meta.get("title") or "").strip() or None

        # Embedded outline, when the PDF ships one.
        try:
            for level, title, page in document.get_toc() or []:
                parsed.outline.append({"level": level, "title": title.strip(), "page": page})
        except Exception:  # pragma: no cover
            pass

        body_size = self._estimate_body_font_size(document)
        section_stack: list[tuple[int, str]] = []
        order = 0

        for page_index in range(document.page_count):
            page = document[page_index]
            page_number = page_index + 1
            try:
                page_dict = page.get_text("dict")
            except Exception:  # pragma: no cover
                continue

            page_blocks: list[ContentBlock] = []
            for raw_block in page_dict.get("blocks", []):
                if raw_block.get("type") != 0:  # 0 = text
                    continue
                text, max_size, bold_ratio = self._flatten_block(raw_block)
                text = normalize_whitespace(text)
                if not text:
                    continue

                heading_level = self._heading_level(text, max_size, body_size, bold_ratio)
                if heading_level:
                    while section_stack and section_stack[-1][0] >= heading_level:
                        section_stack.pop()
                    section_stack.append((heading_level, text[:160]))

                section_path = [title for _, title in section_stack]
                order += 1
                page_blocks.append(
                    ContentBlock(
                        text=text,
                        modality="text",
                        page_number=page_number,
                        section=section_path[-1] if section_path else None,
                        section_path=list(section_path),
                        heading_level=heading_level,
                        bbox=[round(v, 2) for v in raw_block.get("bbox", [])] or None,
                        order=order,
                        metadata={"font_size": round(max_size, 1)},
                    )
                )

            parsed.blocks.extend(page_blocks)

            # -- figures --------------------------------------------------
            order = self._extract_images(document, page, page_number, parsed, section_stack, order, page_blocks)

            if not parsed.title and page_number == 1:
                parsed.title = self._guess_title(page_blocks, body_size)

        # -- tables (pdfplumber) --------------------------------------------
        order = self._extract_tables(data, parsed, order)

        document.close()

        if parsed.text_length < 200 and parsed.page_count > 0:
            parsed.warnings.append(
                "This PDF has little or no extractable text layer; OCR was attempted."
            )
        return parsed

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _flatten_block(raw_block: dict) -> tuple[str, float, float]:
        lines: list[str] = []
        sizes: list[float] = []
        bold_chars = 0
        total_chars = 0
        for line in raw_block.get("lines", []):
            pieces: list[str] = []
            for span in line.get("spans", []):
                span_text = span.get("text", "")
                if not span_text:
                    continue
                pieces.append(span_text)
                sizes.append(span.get("size", 0.0))
                total_chars += len(span_text)
                font = (span.get("font") or "").lower()
                if "bold" in font or "black" in font or "semibold" in font:
                    bold_chars += len(span_text)
            if pieces:
                lines.append("".join(pieces))
        text = "\n".join(lines)
        max_size = max(sizes) if sizes else 0.0
        bold_ratio = bold_chars / total_chars if total_chars else 0.0
        return text, max_size, bold_ratio

    @staticmethod
    def _estimate_body_font_size(document: Any) -> float:
        """Median span size across the first pages -- the body-text baseline."""
        sizes: list[float] = []
        for page_index in range(min(4, document.page_count)):
            try:
                page_dict = document[page_index].get_text("dict")
            except Exception:  # pragma: no cover
                continue
            for raw_block in page_dict.get("blocks", []):
                if raw_block.get("type") != 0:
                    continue
                for line in raw_block.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("text", "").strip():
                            sizes.append(span.get("size", 0.0))
        return statistics.median(sizes) if sizes else 10.0

    @staticmethod
    def _heading_level(text: str, size: float, body_size: float, bold_ratio: float) -> int:
        stripped = text.strip()
        if len(stripped) > 140 or "\n" in stripped.strip():
            return 0
        lowered = re.sub(r"^\d+(\.\d+)*\.?\s*", "", stripped).lower().strip(" :.")
        numbered = bool(re.match(r"^\d+(\.\d+)*\.?\s+\S", stripped))

        if lowered in CANONICAL_SECTIONS:
            return 1
        if size >= body_size * 1.45:
            return 1
        if size >= body_size * 1.18 or (bold_ratio > 0.6 and size >= body_size):
            return 2 if numbered or len(stripped) < 80 else 0
        if numbered and bold_ratio > 0.4:
            return 3
        return 0

    @staticmethod
    def _guess_title(page_blocks: list[ContentBlock], body_size: float) -> str | None:
        candidates = [
            b for b in page_blocks
            if b.metadata.get("font_size", 0) >= body_size * 1.3 and 12 <= len(b.text) <= 220
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda b: b.metadata.get("font_size", 0))
        return normalize_whitespace(best.text.replace("\n", " "))

    def _extract_images(
        self,
        document: Any,
        page: Any,
        page_number: int,
        parsed: ParsedDocument,
        section_stack: list[tuple[int, str]],
        order: int,
        page_blocks: list[ContentBlock],
    ) -> int:
        try:
            image_list = page.get_images(full=True)
        except Exception:  # pragma: no cover
            return order

        captions = self._page_captions(page_blocks, FIGURE_CAPTION)
        for image_index, image_info in enumerate(image_list):
            xref = image_info[0]
            try:
                raw = document.extract_image(xref)
            except Exception:
                continue
            image_bytes = raw.get("image")
            if not image_bytes or len(image_bytes) < MIN_IMAGE_BYTES:
                continue
            if raw.get("width", 0) < MIN_IMAGE_DIM or raw.get("height", 0) < MIN_IMAGE_DIM:
                continue

            label, caption_text = (captions[image_index] if image_index < len(captions) else (None, ""))
            order += 1
            section_path = [title for _, title in section_stack]
            parsed.blocks.append(
                ContentBlock(
                    text=caption_text or (f"{label} on page {page_number}." if label else f"Figure on page {page_number}."),
                    modality="figure",
                    page_number=page_number,
                    section=section_path[-1] if section_path else None,
                    section_path=list(section_path),
                    figure_label=label,
                    order=order,
                    asset_bytes=image_bytes,
                    asset_mime=f"image/{raw.get('ext', 'png')}",
                    metadata={
                        "width": raw.get("width"),
                        "height": raw.get("height"),
                        "caption": caption_text,
                        "source": "embedded_image",
                    },
                )
            )
        return order

    @staticmethod
    def _page_captions(page_blocks: list[ContentBlock], pattern: re.Pattern) -> list[tuple[str | None, str]]:
        captions: list[tuple[str | None, str]] = []
        for block in page_blocks:
            match = pattern.match(block.text.strip())
            if match:
                label = f"Figure {match.group(2)}" if "fig" in match.group(1).lower() else f"Table {match.group(2)}"
                captions.append((label, normalize_whitespace(block.text)))
        return captions

    def _extract_tables(self, data: bytes, parsed: ParsedDocument, order: int) -> int:
        try:
            import pdfplumber  # noqa: PLC0415
        except ImportError:  # pragma: no cover
            parsed.warnings.append("pdfplumber is not installed; tables were not extracted.")
            return order

        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for page_index, page in enumerate(pdf.pages):
                    page_number = page_index + 1
                    try:
                        tables = page.extract_tables()
                    except Exception:
                        continue
                    page_text = page.extract_text() or ""
                    labels = [
                        f"Table {m.group(2)}"
                        for m in TABLE_CAPTION.finditer(page_text)
                    ]
                    for table_index, table in enumerate(tables):
                        markdown = self._table_to_markdown(table)
                        if not markdown:
                            continue
                        order += 1
                        label = labels[table_index] if table_index < len(labels) else None
                        section = self._section_for_page(parsed, page_number)
                        parsed.blocks.append(
                            ContentBlock(
                                text=markdown,
                                modality="table",
                                page_number=page_number,
                                section=section[-1] if section else None,
                                section_path=section,
                                table_label=label,
                                order=order,
                                metadata={
                                    "rows": len(table),
                                    "columns": max((len(r) for r in table), default=0),
                                    "source": "pdfplumber",
                                },
                            )
                        )
        except Exception as exc:
            log.warning("parse.table_extraction_failed", error=str(exc)[:200])
            parsed.warnings.append("Table extraction failed for part of this document.")
        return order

    @staticmethod
    def _section_for_page(parsed: ParsedDocument, page_number: int) -> list[str]:
        latest: list[str] = []
        for block in parsed.blocks:
            if (block.page_number or 0) > page_number:
                break
            if block.section_path:
                latest = list(block.section_path)
        return latest

    @staticmethod
    def _table_to_markdown(table: list[list[Any]]) -> str:
        rows = [
            [normalize_whitespace(str(cell)) if cell is not None else "" for cell in row]
            for row in table
            if any(cell not in (None, "") for cell in row)
        ]
        if len(rows) < 2:
            return ""
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        header, *body = rows
        if not any(header):
            header = [f"col{i + 1}" for i in range(width)]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * width) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in body[:60])
        return "\n".join(lines)

    # -------------------------------------------------------------- async ops
    async def _ocr_thin_pages(self, parsed: ParsedDocument, data: bytes, min_chars: int) -> None:
        chars_by_page: dict[int, int] = {}
        for block in parsed.blocks:
            if block.modality == "text" and block.page_number:
                chars_by_page[block.page_number] = chars_by_page.get(block.page_number, 0) + len(block.text)

        thin = [
            page for page in range(1, parsed.page_count + 1)
            if chars_by_page.get(page, 0) < min_chars
        ]
        if not thin:
            return
        # Cap OCR work so a large scanned document cannot stall ingestion.
        thin = thin[:25]

        def _render(page_number: int) -> bytes | None:
            import fitz  # noqa: PLC0415

            try:
                doc = fitz.open(stream=data, filetype="pdf")
                page = doc[page_number - 1]
                pixmap = page.get_pixmap(dpi=200)
                out = pixmap.tobytes("png")
                doc.close()
                return out
            except Exception:
                return None

        order = max((b.order for b in parsed.blocks), default=0)
        for page_number in thin:
            image_bytes = await asyncio.to_thread(_render, page_number)
            if not image_bytes:
                continue
            result = await extract_text_from_image(image_bytes, "image/png", use_vision_fallback=True)
            text = normalize_whitespace(result["text"])
            if len(text) < 30:
                continue
            order += 1
            parsed.blocks.append(
                ContentBlock(
                    text=text,
                    modality="ocr",
                    page_number=page_number,
                    section=None,
                    order=order,
                    metadata={"ocr_method": result["method"]},
                )
            )
            log.info("parse.ocr_recovered_page", page=page_number, chars=len(text), method=result["method"])

    async def _describe_visuals(self, parsed: ParsedDocument) -> None:
        """Attach a text description to each figure so it can be embedded."""
        figures = [b for b in parsed.blocks if b.modality == "figure" and b.asset_bytes][:20]
        if not figures:
            return
        semaphore = asyncio.Semaphore(3)

        async def _describe(block: ContentBlock) -> None:
            async with semaphore:
                result = await extract_text_from_image(
                    block.asset_bytes or b"", block.asset_mime or "image/png"
                )
                description = normalize_whitespace(result["text"])
                if description:
                    caption = block.metadata.get("caption") or ""
                    block.text = normalize_whitespace(f"{caption}\n\n{description}") if caption else description
                    block.metadata["described_by"] = result["method"]

        await asyncio.gather(*(_describe(b) for b in figures), return_exceptions=True)


__all__ = ["PDFParser"]
