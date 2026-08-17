"""Structure-aware chunking.

Rather than slicing the document into fixed-length strings, chunks are built
from parsed blocks under four rules:

1. **Never cross a section boundary.** A chunk belongs to exactly one section
   path, so its citation ("Methodology > Training setup") is truthful.
2. **Never split an atomic unit.** Tables, figures, images and code blocks
   become their own chunks; splitting a Markdown table mid-row destroys it.
3. **Split long prose at sentence boundaries**, with sentence-level overlap so a
   fact spanning a boundary is retrievable from either side.
4. **Prefix the heading breadcrumb** to each chunk's embedded text. This gives
   the embedding model the context a human reader gets from the page layout,
   and measurably helps when a section's body text omits its own topic.

Chunks below ``chunk_min_tokens`` are merged forward so the index is not
polluted with fragments.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.core.text import estimate_tokens, normalize_whitespace, split_sentences
from app.ingestion.parsers.base import ContentBlock, ParsedDocument

ATOMIC_MODALITIES = {"table", "figure", "image", "code"}


@dataclass(slots=True)
class ChunkCandidate:
    text: str
    modality: str = "text"
    page_number: int | None = None
    page_end: int | None = None
    section: str | None = None
    section_path: list[str] = field(default_factory=list)
    figure_label: str | None = None
    table_label: str | None = None
    bbox: list[float] | None = None
    token_count: int = 0
    asset_bytes: bytes | None = None
    asset_mime: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:32]

    @property
    def embedding_text(self) -> str:
        """Text handed to the embedder: breadcrumb + content."""
        breadcrumb = " > ".join(self.section_path[-3:]) if self.section_path else ""
        label = self.figure_label or self.table_label
        header_bits = [b for b in (breadcrumb, label) if b]
        if not header_bits:
            return self.text
        return f"[{' | '.join(header_bits)}]\n{self.text}"


class StructuralChunker:
    def __init__(
        self,
        target_tokens: int | None = None,
        overlap_tokens: int | None = None,
        min_tokens: int | None = None,
    ):
        settings = get_settings()
        self.target_tokens = target_tokens or settings.chunk_target_tokens
        self.overlap_tokens = overlap_tokens or settings.chunk_overlap_tokens
        self.min_tokens = min_tokens or settings.chunk_min_tokens
        # Hard ceiling: a single atomic block is kept whole up to this size.
        self.max_tokens = int(self.target_tokens * 2.2)

    # ------------------------------------------------------------------ main
    def chunk(self, parsed: ParsedDocument) -> list[ChunkCandidate]:
        chunks: list[ChunkCandidate] = []
        buffer: list[ContentBlock] = []
        buffer_tokens = 0
        current_path: list[str] = []

        def flush() -> None:
            nonlocal buffer, buffer_tokens
            if buffer:
                chunks.extend(self._emit_text_chunks(buffer))
            buffer = []
            buffer_tokens = 0

        for block in parsed.blocks:
            if block.modality in ATOMIC_MODALITIES:
                flush()
                chunks.extend(self._emit_atomic(block))
                current_path = block.section_path or current_path
                continue

            # A heading starts a new chunk; it is carried as context, not content.
            if block.is_heading:
                flush()
                current_path = list(block.section_path)
                continue

            if block.section_path != current_path:
                flush()
                current_path = list(block.section_path)

            block_tokens = estimate_tokens(block.text)
            if buffer_tokens + block_tokens > self.target_tokens and buffer:
                flush()
            buffer.append(block)
            buffer_tokens += block_tokens

        flush()
        chunks = self._merge_small(chunks)
        for chunk in chunks:
            chunk.token_count = estimate_tokens(chunk.text)
        return [c for c in chunks if c.text.strip()]

    # -------------------------------------------------------------- emitters
    def _emit_atomic(self, block: ContentBlock) -> list[ChunkCandidate]:
        """Tables/figures/images/code become standalone chunks.

        An oversized table is split on row boundaries with its header repeated,
        so each part is still a valid, readable table.
        """
        tokens = estimate_tokens(block.text)
        if tokens <= self.max_tokens or block.modality != "table":
            return [self._candidate(block, block.text)]

        lines = block.text.split("\n")
        header = lines[:2] if len(lines) > 2 and lines[1].strip().startswith("|") else []
        body = lines[len(header):]
        parts: list[ChunkCandidate] = []
        current: list[str] = []
        current_tokens = estimate_tokens("\n".join(header))

        for line in body:
            line_tokens = estimate_tokens(line)
            if current_tokens + line_tokens > self.max_tokens and current:
                parts.append(self._candidate(block, "\n".join(header + current), part=len(parts) + 1))
                current = []
                current_tokens = estimate_tokens("\n".join(header))
            current.append(line)
            current_tokens += line_tokens
        if current:
            parts.append(self._candidate(block, "\n".join(header + current), part=len(parts) + 1))
        return parts or [self._candidate(block, block.text)]

    def _emit_text_chunks(self, blocks: list[ContentBlock]) -> list[ChunkCandidate]:
        text = normalize_whitespace("\n\n".join(b.text for b in blocks if b.text.strip()))
        if not text:
            return []

        first, last = blocks[0], blocks[-1]
        base_kwargs = {
            "modality": "ocr" if first.modality == "ocr" else "text",
            "page_number": first.page_number,
            "page_end": last.page_number,
            "section": first.section,
            "section_path": list(first.section_path),
            "bbox": first.bbox,
            "metadata": {"block_count": len(blocks)},
        }

        if estimate_tokens(text) <= self.max_tokens:
            return [ChunkCandidate(text=text, **base_kwargs)]

        sentences = split_sentences(text) or [text]
        chunks: list[ChunkCandidate] = []
        window: list[str] = []
        window_tokens = 0

        for sentence in sentences:
            sentence_tokens = estimate_tokens(sentence)
            # A single sentence longer than the budget is emitted alone rather
            # than being cut mid-thought.
            if sentence_tokens > self.target_tokens and not window:
                chunks.append(ChunkCandidate(text=sentence, **{**base_kwargs, "metadata": dict(base_kwargs["metadata"])}))
                continue
            if window_tokens + sentence_tokens > self.target_tokens and window:
                chunks.append(
                    ChunkCandidate(text=" ".join(window), **{**base_kwargs, "metadata": dict(base_kwargs["metadata"])})
                )
                window, window_tokens = self._overlap_tail(window)
            window.append(sentence)
            window_tokens += sentence_tokens

        if window:
            chunks.append(ChunkCandidate(text=" ".join(window), **{**base_kwargs, "metadata": dict(base_kwargs["metadata"])}))

        for index, chunk in enumerate(chunks):
            chunk.metadata["split_index"] = index
            chunk.metadata["split_total"] = len(chunks)
        return chunks

    def _overlap_tail(self, window: list[str]) -> tuple[list[str], int]:
        """Carry the trailing sentences of a chunk into the next one."""
        tail: list[str] = []
        tokens = 0
        for sentence in reversed(window):
            sentence_tokens = estimate_tokens(sentence)
            if tokens + sentence_tokens > self.overlap_tokens:
                break
            tail.insert(0, sentence)
            tokens += sentence_tokens
        return tail, tokens

    def _candidate(self, block: ContentBlock, text: str, part: int | None = None) -> ChunkCandidate:
        metadata = dict(block.metadata)
        if part is not None:
            metadata["table_part"] = part
        return ChunkCandidate(
            text=text,
            modality=block.modality,
            page_number=block.page_number,
            page_end=block.page_number,
            section=block.section,
            section_path=list(block.section_path),
            figure_label=block.figure_label,
            table_label=block.table_label,
            bbox=block.bbox,
            asset_bytes=block.asset_bytes,
            asset_mime=block.asset_mime,
            metadata=metadata,
        )

    # ---------------------------------------------------------------- merging
    def _merge_small(self, chunks: list[ChunkCandidate]) -> list[ChunkCandidate]:
        """Fold undersized prose chunks into their neighbour in the same section."""
        merged: list[ChunkCandidate] = []
        for chunk in chunks:
            tokens = estimate_tokens(chunk.text)
            if (
                merged
                and tokens < self.min_tokens
                and chunk.modality not in ATOMIC_MODALITIES
                and merged[-1].modality not in ATOMIC_MODALITIES
                and merged[-1].section_path == chunk.section_path
                and estimate_tokens(merged[-1].text) + tokens <= self.max_tokens
            ):
                previous = merged[-1]
                previous.text = f"{previous.text}\n\n{chunk.text}".strip()
                previous.page_end = chunk.page_end or previous.page_end
                continue
            merged.append(chunk)
        return merged


def chunk_document(parsed: ParsedDocument) -> list[ChunkCandidate]:
    return StructuralChunker().chunk(parsed)


__all__ = ["StructuralChunker", "ChunkCandidate", "chunk_document", "ATOMIC_MODALITIES"]
