"""Embedding providers.

Production embeddings come from the Gemini embedding API. A deterministic
hashing embedder is included as an explicitly-labelled development/CI fallback:
it lets the ingestion pipeline, the vector store and the whole test suite run
without network access. It carries **no semantic knowledge** and must never be
used to produce reported benchmark numbers -- ``/api/v1/health`` and the
Settings page both surface a warning when it is active.

Note: an embedding model is not a generative LLM. RAGX's no-local-LLM rule
concerns text generation; no local generative model is used anywhere.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import struct
from abc import ABC, abstractmethod
from typing import Any

from app.core.cache import cache_key, embedding_cache
from app.core.config import get_settings
from app.core.errors import ProviderError
from app.core.logging import get_logger
from app.core.text import tokenize
from app.llm.gemini.provider import GeminiProvider

log = get_logger("ragx.embeddings")

# Gemini task types -- retrieval quality improves when queries and documents
# are embedded with matching, asymmetric task hints.
TASK_DOCUMENT = "retrieval_document"
TASK_QUERY = "retrieval_query"


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


class EmbeddingProvider(ABC):
    name: str = "base"
    dimension: int = 768
    production_ready: bool = True

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]: ...

    async def health(self) -> dict[str, Any]:
        return {"provider": self.name, "dimension": self.dimension, "production_ready": self.production_ready}


class GeminiEmbeddingProvider(EmbeddingProvider):
    name = "gemini"
    production_ready = True

    def __init__(self) -> None:
        settings = get_settings()
        self.dimension = settings.embedding_dimension
        self.batch_size = settings.embedding_batch_size
        self.model = settings.gemini_embedding_model
        self._provider = GeminiProvider()

    @property
    def is_configured(self) -> bool:
        return self._provider.is_configured

    async def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t if t.strip() else " " for t in texts[start : start + self.batch_size]]
            vectors = await self._provider.embed(
                batch, task_type=task_type, dimension=self.dimension
            )
            if len(vectors) != len(batch):
                raise ProviderError(
                    f"Gemini returned {len(vectors)} embeddings for {len(batch)} inputs."
                )
            out.extend(l2_normalize(v) for v in vectors)
        if out and len(out[0]) != self.dimension:
            # Keep configuration honest rather than silently truncating.
            log.warning(
                "embeddings.dimension_mismatch",
                configured=self.dimension,
                actual=len(out[0]),
                model=self.model,
            )
            self.dimension = len(out[0])
        return out

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await self._embed(texts, TASK_DOCUMENT)

    async def embed_query(self, text: str) -> list[float]:
        key = cache_key("gemini-query-embed", self.model, text)
        cached = await embedding_cache.get(key)
        if cached is not None:
            return cached
        vectors = await self._embed([text], TASK_QUERY)
        vector = vectors[0] if vectors else [0.0] * self.dimension
        await embedding_cache.set(key, vector)
        return vector

    async def health(self) -> dict[str, Any]:
        base = {
            "provider": self.name,
            "model": self.model,
            "dimension": self.dimension,
            "production_ready": True,
            "configured": self.is_configured,
        }
        if not self.is_configured:
            return {**base, "healthy": False, "status_text": "not_configured"}
        try:
            await self.embed_query("healthcheck")
            return {**base, "healthy": True, "status_text": "healthy"}
        except Exception as exc:
            return {**base, "healthy": False, "status_text": "unhealthy", "error": str(exc)[:200]}


class HashingEmbeddingProvider(EmbeddingProvider):
    """Deterministic, offline, DEVELOPMENT-ONLY embedder.

    Implements the hashing trick over word unigrams and bigrams with signed
    buckets and sublinear term weighting. Lexically similar texts land near each
    other, which is enough to exercise the vector-store code path end to end --
    but it has no semantic generalisation, so paraphrases will not match. It
    exists so the system runs and is testable without API keys.
    """

    name = "hashing"
    production_ready = False

    def __init__(self, dimension: int | None = None) -> None:
        self.dimension = dimension or get_settings().embedding_dimension

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = tokenize(text, drop_stopwords=False)
        if not tokens:
            return vector
        grams = list(tokens)
        grams.extend(f"{a}_{b}" for a, b in zip(tokens, tokens[1:]))
        counts: dict[str, int] = {}
        for gram in grams:
            counts[gram] = counts.get(gram, 0) + 1
        for gram, count in counts.items():
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            bucket = struct.unpack("<Q", digest)[0]
            index = bucket % self.dimension
            sign = 1.0 if (bucket >> 63) & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(count))
        return l2_normalize(vector)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(lambda: [self._vector(t) for t in texts])

    async def embed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._vector, text)

    async def health(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "dimension": self.dimension,
            "production_ready": False,
            "healthy": True,
            "status_text": "development_only",
            "warning": (
                "The hashing embedder is a lexical stand-in for offline development. "
                "Set EMBEDDING_PROVIDER=gemini with a GEMINI_API_KEY for semantic retrieval."
            ),
        }


_provider: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    """Resolve the configured embedder.

    If Gemini embeddings are requested but no key is present we degrade to the
    hashing embedder and log loudly rather than failing every ingestion.
    """
    global _provider
    if _provider is not None:
        return _provider

    settings = get_settings()
    if settings.embedding_provider == "gemini":
        gemini = GeminiEmbeddingProvider()
        if gemini.is_configured:
            _provider = gemini
            log.info("embeddings.provider_selected", provider="gemini", model=gemini.model)
            return _provider
        log.warning(
            "embeddings.gemini_unconfigured_using_dev_embedder",
            detail="GEMINI_API_KEY is missing; falling back to the development hashing embedder.",
        )
    _provider = HashingEmbeddingProvider()
    log.warning("embeddings.provider_selected", provider="hashing", production_ready=False)
    return _provider


def reset_embedding_provider() -> None:
    global _provider
    _provider = None


__all__ = [
    "EmbeddingProvider",
    "GeminiEmbeddingProvider",
    "HashingEmbeddingProvider",
    "get_embedding_provider",
    "reset_embedding_provider",
    "l2_normalize",
]
