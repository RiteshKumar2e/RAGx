"""GeminiProvider.embed() must use the shared rate-limit-aware retry logic.

Ingesting a document fires many embedding batches back to back, which is the
path most likely to trip Gemini's quota in practice. It previously retried on
a blind fixed-doubling schedule instead of the shared ``app.llm.retry`` helpers
the chat gateway already used -- so a 429 that told the caller "retry in 10s"
got retried on whatever the fixed schedule happened to be at that attempt,
instead of waiting exactly as long as Gemini asked.
"""

from __future__ import annotations

import pytest

import app.llm.gemini.provider as gemini_provider_module
from app.llm.gemini.provider import GeminiProvider

# Real message captured from a live 429, reused from test_llm_retry.py's fixture.
GEMINI_429 = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your "
    "current quota. \\n* Quota exceeded for metric: "
    "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, "
    "model: gemini-3.7-flash\\nPlease retry in 10.969884611s.', 'status': "
    "'RESOURCE_EXHAUSTED', 'details': [{'@type': "
    "'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '10s'}]}}"
)


class _FakeTypes:
    @staticmethod
    def EmbedContentConfig(**kwargs):
        return kwargs


class _FakeEmbeddingItem:
    def __init__(self, values):
        self.values = values


class _FakeEmbedResponse:
    def __init__(self, vectors):
        self.embeddings = [_FakeEmbeddingItem(v) for v in vectors]


class _FakeModels:
    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    async def embed_content(self, model, contents, config):
        result = self._results[self.calls]
        self.calls += 1
        if isinstance(result, Exception):
            raise result
        return result


class _FakeClient:
    def __init__(self, models: _FakeModels) -> None:
        self.aio = type("_Aio", (), {"models": models})()


@pytest.mark.anyio
async def test_embed_retry_honours_the_providers_requested_delay(monkeypatch) -> None:
    provider = GeminiProvider()
    provider._api_key = "test-key"
    models = _FakeModels([Exception(GEMINI_429), _FakeEmbedResponse([[0.1, 0.2]])])
    provider._client = _FakeClient(models)
    provider._types = _FakeTypes()

    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(gemini_provider_module.asyncio, "sleep", fake_sleep)

    vectors = await provider.embed(["hello"], dimension=2)

    assert vectors == [[0.1, 0.2]]
    assert models.calls == 2
    # The error's own retryDelay ('10s'), not the fixed 2.0s starting backoff.
    assert waits == [pytest.approx(10.0)]
