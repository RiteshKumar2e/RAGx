"""BM25 lexical index (Okapi BM25 via ``rank_bm25``).

This is the keyword half of Hybrid RAG. Dense retrieval is weak exactly where
BM25 is strong -- acronyms, model names, dataset identifiers, version strings --
so both are run and fused.

The index is held in memory and persisted to disk as a token list so a restart
does not require re-tokenising the corpus. It is rebuilt from PostgreSQL on
demand if the snapshot is missing.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import BACKEND_ROOT
from app.core.logging import get_logger
from app.core.text import tokenize

log = get_logger("ragx.bm25")

INDEX_PATH = BACKEND_ROOT / "data" / "bm25" / "index.json"


@dataclass(slots=True)
class BM25Document:
    chunk_id: str
    document_id: str
    text: str
    modality: str = "text"


@dataclass(slots=True)
class BM25Hit:
    chunk_id: str
    score: float
    document_id: str


class BM25Index:
    """Thread-safe in-memory BM25 index with disk persistence."""

    def __init__(self, path: Path | None = None):
        self.path = path or INDEX_PATH
        self._chunk_ids: list[str] = []
        self._document_ids: list[str] = []
        self._modalities: list[str] = []
        self._tokens: list[list[str]] = []
        self._bm25: Any = None
        self._lock = asyncio.Lock()
        self._loaded = False
        self._built_at: float = 0.0

    # ------------------------------------------------------------- internals
    def _rebuild_model(self) -> None:
        if not self._tokens:
            self._bm25 = None
            return
        from rank_bm25 import BM25Okapi  # noqa: PLC0415

        self._bm25 = BM25Okapi(self._tokens)
        self._built_at = time.time()

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "chunk_ids": self._chunk_ids,
            "document_ids": self._document_ids,
            "modalities": self._modalities,
            "tokens": self._tokens,
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self.path)

    def _load(self) -> None:
        if not self.path.exists():
            self._loaded = True
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._chunk_ids = payload.get("chunk_ids", [])
            self._document_ids = payload.get("document_ids", [])
            self._modalities = payload.get("modalities", ["text"] * len(self._chunk_ids))
            self._tokens = payload.get("tokens", [])
            self._rebuild_model()
            log.info("bm25.loaded", documents=len(self._chunk_ids))
        except Exception as exc:  # pragma: no cover - corrupt snapshot
            log.warning("bm25.load_failed_rebuilding", error=str(exc))
            self._chunk_ids, self._document_ids, self._modalities, self._tokens = [], [], [], []
        self._loaded = True

    async def ensure_loaded(self) -> None:
        if self._loaded:
            return
        async with self._lock:
            if not self._loaded:
                await asyncio.to_thread(self._load)

    # ---------------------------------------------------------------- writes
    async def add(self, documents: list[BM25Document]) -> int:
        if not documents:
            return 0
        await self.ensure_loaded()

        def _add() -> int:
            existing = {cid: i for i, cid in enumerate(self._chunk_ids)}
            for doc in documents:
                tokens = tokenize(doc.text, drop_stopwords=False)
                if doc.chunk_id in existing:
                    idx = existing[doc.chunk_id]
                    self._tokens[idx] = tokens
                    self._document_ids[idx] = doc.document_id
                    self._modalities[idx] = doc.modality
                    continue
                self._chunk_ids.append(doc.chunk_id)
                self._document_ids.append(doc.document_id)
                self._modalities.append(doc.modality)
                self._tokens.append(tokens)
            self._rebuild_model()
            self._persist()
            return len(documents)

        async with self._lock:
            return await asyncio.to_thread(_add)

    async def remove_document(self, document_id: str) -> int:
        await self.ensure_loaded()

        def _remove() -> int:
            keep = [i for i, did in enumerate(self._document_ids) if did != document_id]
            removed = len(self._document_ids) - len(keep)
            if removed:
                self._chunk_ids = [self._chunk_ids[i] for i in keep]
                self._document_ids = [self._document_ids[i] for i in keep]
                self._modalities = [self._modalities[i] for i in keep]
                self._tokens = [self._tokens[i] for i in keep]
                self._rebuild_model()
                self._persist()
            return removed

        async with self._lock:
            return await asyncio.to_thread(_remove)

    async def replace_all(self, documents: list[BM25Document]) -> int:
        def _replace() -> int:
            self._chunk_ids = [d.chunk_id for d in documents]
            self._document_ids = [d.document_id for d in documents]
            self._modalities = [d.modality for d in documents]
            self._tokens = [tokenize(d.text, drop_stopwords=False) for d in documents]
            self._rebuild_model()
            self._persist()
            self._loaded = True
            return len(documents)

        async with self._lock:
            return await asyncio.to_thread(_replace)

    async def clear(self) -> None:
        await self.replace_all([])

    # ---------------------------------------------------------------- search
    async def search(
        self,
        query: str,
        limit: int = 10,
        document_ids: list[str] | None = None,
        modalities: list[str] | None = None,
    ) -> list[BM25Hit]:
        await self.ensure_loaded()
        if self._bm25 is None or not self._chunk_ids:
            return []

        query_tokens = tokenize(query, drop_stopwords=False)
        if not query_tokens:
            return []

        def _search() -> list[BM25Hit]:
            raw_scores = self._bm25.get_scores(query_tokens)
            doc_filter = set(document_ids) if document_ids else None
            modality_filter = set(modalities) if modalities else None

            candidates: list[tuple[int, float]] = []
            for idx, score in enumerate(raw_scores):
                if score <= 0:
                    continue
                if doc_filter and self._document_ids[idx] not in doc_filter:
                    continue
                if modality_filter and self._modalities[idx] not in modality_filter:
                    continue
                candidates.append((idx, float(score)))

            candidates.sort(key=lambda x: x[1], reverse=True)
            top = candidates[:limit]
            if not top:
                return []
            # Normalise to 0..1 against the best score in this result set so
            # BM25 scores are comparable with cosine similarity during fusion.
            best = top[0][1] or 1.0
            return [
                BM25Hit(
                    chunk_id=self._chunk_ids[idx],
                    score=round(score / best, 6),
                    document_id=self._document_ids[idx],
                )
                for idx, score in top
            ]

        return await asyncio.to_thread(_search)

    # ---------------------------------------------------------------- status
    @property
    def size(self) -> int:
        return len(self._chunk_ids)

    async def health(self) -> dict[str, Any]:
        await self.ensure_loaded()
        return {
            "store": "bm25",
            "healthy": True,
            "status_text": "healthy",
            "documents": self.size,
            "persisted": self.path.exists(),
            "path": str(self.path),
        }


_index: BM25Index | None = None


def get_bm25_index() -> BM25Index:
    global _index
    if _index is None:
        _index = BM25Index()
    return _index


def reset_bm25_index() -> None:
    global _index
    _index = None


__all__ = ["BM25Index", "BM25Document", "BM25Hit", "get_bm25_index", "reset_bm25_index"]
