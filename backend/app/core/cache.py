"""A small async-safe TTL + LRU cache.

Used for embedding lookups and query-analysis results, both of which are pure
functions of their input and are expensive because they cost an API call. This
is one of the mechanisms that keeps unnecessary LLM calls down.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Awaitable, Callable, TypeVar

from app.core.config import get_settings

T = TypeVar("T")


def cache_key(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class TTLCache:
    def __init__(self, max_entries: int | None = None, ttl_seconds: int | None = None):
        settings = get_settings()
        self.max_entries = max_entries or settings.cache_max_entries
        self.ttl = ttl_seconds or settings.cache_ttl_seconds
        self.enabled = settings.cache_enabled
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            expires_at, value = entry
            if expires_at < time.time():
                self._store.pop(key, None)
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return value

    async def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        async with self._lock:
            self._store[key] = (time.time() + self.ttl, value)
            self._store.move_to_end(key)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)

    async def get_or_set(self, key: str, factory: Callable[[], Awaitable[T]]) -> T:
        cached = await self.get(key)
        if cached is not None:
            return cached
        value = await factory()
        await self.set(key, value)
        return value

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "enabled": self.enabled,
            "entries": len(self._store),
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }


embedding_cache = TTLCache()
analysis_cache = TTLCache()
answer_cache = TTLCache()

ALL_CACHES = {
    "embeddings": embedding_cache,
    "query_analysis": analysis_cache,
    "answers": answer_cache,
}

__all__ = ["TTLCache", "cache_key", "embedding_cache", "analysis_cache", "answer_cache", "ALL_CACHES"]
