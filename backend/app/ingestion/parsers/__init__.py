"""Parser registry -- resolves a file extension to its parser."""

from __future__ import annotations

from app.core.errors import UnsupportedFileError
from app.ingestion.parsers.base import ContentBlock, DocumentParser, ParsedDocument
from app.ingestion.parsers.image import ImageParser
from app.ingestion.parsers.office import CSVParser, DocxParser, HTMLParser, TextParser
from app.ingestion.parsers.pdf import PDFParser

_PARSERS: list[DocumentParser] = [
    PDFParser(),
    DocxParser(),
    TextParser(),
    CSVParser(),
    HTMLParser(),
    ImageParser(),
]


def get_parser(extension: str) -> DocumentParser:
    extension = extension.lower()
    for parser in _PARSERS:
        if parser.supports(extension):
            return parser
    raise UnsupportedFileError(f"No parser is registered for '{extension}' files.")


def supported_extensions() -> list[str]:
    return sorted({ext for parser in _PARSERS for ext in parser.extensions})


__all__ = [
    "get_parser",
    "supported_extensions",
    "DocumentParser",
    "ParsedDocument",
    "ContentBlock",
    "PDFParser",
    "DocxParser",
    "TextParser",
    "CSVParser",
    "HTMLParser",
    "ImageParser",
]
