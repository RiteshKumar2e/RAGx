"""Parser contract and the intermediate representation shared by all formats.

Every parser produces a flat list of :class:`ContentBlock` objects that already
carry citation provenance (page, section path, figure/table label). Chunking
then operates on blocks rather than on a raw string, which is how document
hierarchy survives into the retrieval layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ContentBlock:
    """One extracted unit of document content."""

    text: str
    modality: str = "text"  # text | table | figure | image | ocr | code
    page_number: int | None = None
    section: str | None = None
    section_path: list[str] = field(default_factory=list)
    heading_level: int = 0
    figure_label: str | None = None
    table_label: str | None = None
    bbox: list[float] | None = None
    order: int = 0
    # Raw bytes for a figure/table image, persisted to object storage and used
    # as multimodal evidence.
    asset_bytes: bytes | None = None
    asset_mime: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_heading(self) -> bool:
        return self.heading_level > 0

    @property
    def is_visual(self) -> bool:
        return self.modality in {"figure", "image", "table"}


@dataclass(slots=True)
class ParsedDocument:
    blocks: list[ContentBlock] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    outline: list[dict[str, Any]] = field(default_factory=list)
    page_count: int = 0
    title: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def table_count(self) -> int:
        return sum(1 for b in self.blocks if b.modality == "table")

    @property
    def figure_count(self) -> int:
        return sum(1 for b in self.blocks if b.modality in {"figure", "image"})

    @property
    def text_length(self) -> int:
        return sum(len(b.text) for b in self.blocks)


class DocumentParser(ABC):
    """Format-specific extraction."""

    name: str = "base"
    extensions: tuple[str, ...] = ()

    def supports(self, extension: str) -> bool:
        return extension.lower() in self.extensions

    @abstractmethod
    async def parse(self, data: bytes, filename: str) -> ParsedDocument: ...


__all__ = ["ContentBlock", "ParsedDocument", "DocumentParser"]
