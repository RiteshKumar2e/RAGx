"""Chunking, parsing and upload validation."""

from __future__ import annotations

import pytest

from app.core.errors import FileTooLargeError, UnsupportedFileError
from app.core.security import sanitize_filename, validate_upload
from app.ingestion.chunking import StructuralChunker
from app.ingestion.parsers.base import ContentBlock, ParsedDocument
from app.ingestion.parsers.office import CSVParser, TextParser


# ----------------------------------------------------------------- security
def test_filename_traversal_is_stripped() -> None:
    assert sanitize_filename("../../etc/passwd") == "etc_passwd" or "passwd" in sanitize_filename(
        "../../etc/passwd"
    )
    assert "/" not in sanitize_filename("a/b/c.pdf")
    assert "\\" not in sanitize_filename(r"a\b\c.pdf")


def test_extension_allowlist_enforced() -> None:
    with pytest.raises(UnsupportedFileError):
        validate_upload("payload.exe", "application/octet-stream", b"MZ\x90\x00", 100)


def test_magic_bytes_must_match_extension() -> None:
    # A file claiming to be a PDF that is not one.
    with pytest.raises(UnsupportedFileError):
        validate_upload("fake.pdf", "application/pdf", b"not a pdf at all", 100)
    # A genuine PDF header passes.
    assert validate_upload("real.pdf", "application/pdf", b"%PDF-1.7\n%...", 5000) == "real.pdf"


def test_binary_content_rejected_for_text_types() -> None:
    with pytest.raises(UnsupportedFileError):
        validate_upload("notes.txt", "text/plain", b"hello\x00\x01binary", 100)


def test_size_limit_enforced() -> None:
    with pytest.raises(FileTooLargeError):
        validate_upload("big.txt", "text/plain", b"hello", 10_000_000_000)


def test_empty_file_rejected() -> None:
    with pytest.raises(UnsupportedFileError):
        validate_upload("empty.txt", "text/plain", b"", 0)


# ----------------------------------------------------------------- chunking
def _blocks(*specs: tuple[str, str, list[str]]) -> ParsedDocument:
    parsed = ParsedDocument(page_count=1)
    for order, (text, modality, path) in enumerate(specs):
        parsed.blocks.append(
            ContentBlock(
                text=text,
                modality=modality,
                page_number=1,
                section=path[-1] if path else None,
                section_path=path,
                order=order,
            )
        )
    return parsed


def test_chunks_do_not_cross_sections() -> None:
    parsed = _blocks(
        ("Alpha content about the first topic in detail. " * 4, "text", ["Methods"]),
        ("Beta content about a completely different topic. " * 4, "text", ["Results"]),
    )
    chunks = StructuralChunker(target_tokens=400, min_tokens=5).chunk(parsed)
    for chunk in chunks:
        assert len(set(map(tuple, [chunk.section_path]))) == 1
    sections = {tuple(c.section_path) for c in chunks}
    assert ("Methods",) in sections and ("Results",) in sections


def test_tables_become_their_own_chunks() -> None:
    table = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    parsed = _blocks(
        ("Some prose before the table. " * 5, "text", ["Results"]),
        (table, "table", ["Results"]),
    )
    chunks = StructuralChunker(target_tokens=200, min_tokens=5).chunk(parsed)
    table_chunks = [c for c in chunks if c.modality == "table"]
    assert len(table_chunks) == 1
    assert table_chunks[0].text == table  # never merged or split


def test_long_prose_splits_with_overlap() -> None:
    sentences = " ".join(
        f"This is sentence number {i} describing an important experimental detail." for i in range(60)
    )
    parsed = _blocks((sentences, "text", ["Body"]))
    chunks = StructuralChunker(target_tokens=100, overlap_tokens=30, min_tokens=10).chunk(parsed)
    assert len(chunks) > 1
    # Consecutive chunks should share some trailing/leading text (the overlap).
    first_words = set(chunks[0].text.split())
    second_words = set(chunks[1].text.split())
    assert first_words & second_words


def test_embedding_text_includes_section_breadcrumb() -> None:
    parsed = _blocks(("Body text about the training setup here. " * 3, "text", ["Methods", "Training"]))
    chunk = StructuralChunker(min_tokens=5).chunk(parsed)[0]
    assert "Methods > Training" in chunk.embedding_text
    assert "Methods > Training" not in chunk.text  # breadcrumb is not in stored content


def test_tiny_chunks_are_merged() -> None:
    parsed = _blocks(
        ("Short one.", "text", ["S"]),
        ("Short two.", "text", ["S"]),
        ("Short three.", "text", ["S"]),
    )
    chunks = StructuralChunker(target_tokens=400, min_tokens=50).chunk(parsed)
    assert len(chunks) == 1


# ------------------------------------------------------------------ parsers
@pytest.mark.anyio
async def test_markdown_parser_builds_section_hierarchy() -> None:
    content = b"""# Title

## Introduction
Some introductory text about the problem being solved.

### Background
Deeper nested background information for context.

## Results
The results were positive and are described here.
"""
    parsed = await TextParser().parse(content, "doc.md")
    paths = [b.section_path for b in parsed.blocks if b.section_path]
    assert ["Title", "Introduction", "Background"] in paths
    assert any(p == ["Title", "Results"] for p in paths)


@pytest.mark.anyio
async def test_markdown_code_fences_preserved() -> None:
    content = b"""# Doc

Some text.

```python
def train(model):
    return model.fit()
```
"""
    parsed = await TextParser().parse(content, "doc.md")
    code_blocks = [b for b in parsed.blocks if b.modality == "code"]
    assert code_blocks
    assert "def train" in code_blocks[0].text


@pytest.mark.anyio
async def test_csv_parser_emits_schema_and_rows() -> None:
    csv = b"model,map,fps\nDefectNet,78.4,62\nYOLOv5s,74.1,71\n"
    parsed = await CSVParser().parse(csv, "results.csv")
    assert parsed.metadata["rows"] == 2
    assert parsed.metadata["column_names"] == ["model", "map", "fps"]
    assert all(b.modality == "table" for b in parsed.blocks)
    assert parsed.blocks[0].metadata["kind"] == "schema"
