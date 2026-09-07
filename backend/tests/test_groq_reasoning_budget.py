"""Empty completions from Groq's reasoning models must not pass as answers.

`openai/gpt-oss-*` are reasoning models: they spend output tokens on an internal
reasoning channel before emitting content. Verified against the live API, a
16-token budget produced `content=''` with `finish_reason='length'` and 16
completion tokens -- an HTTP 200 carrying no answer. Even a one-word reply needed
36 completion tokens.

Returning that as a success handed callers a blank string with no indication why,
so the failure surfaced far from its cause. Raising lets the gateway retry or
fall back to a provider that can answer.
"""

from __future__ import annotations

import pytest

from app.core.errors import ProviderError
from app.llm.base import LLMRequest, Message
from app.llm.groq.provider import GroqProvider


class _Message:
    def __init__(self, content, reasoning=""):
        self.content = content
        self.reasoning = reasoning


class _Choice:
    def __init__(self, content, finish_reason, reasoning=""):
        self.message = _Message(content, reasoning)
        self.finish_reason = finish_reason


class _Usage:
    prompt_tokens = 11
    completion_tokens = 16


class _Response:
    def __init__(self, content, finish_reason, reasoning=""):
        self.choices = [_Choice(content, finish_reason, reasoning)]
        self.usage = _Usage()


def _provider_returning(response) -> GroqProvider:
    provider = GroqProvider()

    class _Completions:
        async def create(self, **_kwargs):
            return response

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    provider._ensure_client = lambda: _Client()  # noqa: SLF001
    return provider


def _request(max_output_tokens: int = 16) -> LLMRequest:
    return LLMRequest(
        messages=[Message.user("Reply with the single word: OK")],
        purpose="synthesis",
        temperature=0.0,
        max_output_tokens=max_output_tokens,
        model="openai/gpt-oss-120b",
    )


@pytest.mark.anyio
async def test_budget_exhausted_by_reasoning_raises_instead_of_returning_blank() -> None:
    provider = _provider_returning(_Response("", "length", reasoning="We need to reply..."))

    with pytest.raises(ProviderError) as excinfo:
        await provider.generate(_request(16))

    message = str(excinfo.value)
    # The message has to name the cause and the fix, since the HTTP call
    # itself succeeded and nothing else points at the token budget.
    assert "reasoning" in message.lower()
    assert "max_output_tokens" in message
    assert "16" in message


@pytest.mark.anyio
async def test_empty_response_without_truncation_also_raises() -> None:
    provider = _provider_returning(_Response("", "stop"))
    with pytest.raises(ProviderError) as excinfo:
        await provider.generate(_request(2048))
    assert "empty" in str(excinfo.value).lower()


@pytest.mark.anyio
async def test_whitespace_only_content_is_treated_as_empty() -> None:
    provider = _provider_returning(_Response("   \n  ", "stop"))
    with pytest.raises(ProviderError):
        await provider.generate(_request(2048))


@pytest.mark.anyio
async def test_a_real_answer_is_returned_untouched() -> None:
    provider = _provider_returning(_Response("OK", "stop"))
    response = await provider.generate(_request(2048))
    assert response.text == "OK"
    assert response.finish_reason == "stop"


@pytest.mark.anyio
async def test_truncated_but_non_empty_content_is_kept() -> None:
    """A partial answer is still an answer; only an empty one is a failure."""
    provider = _provider_returning(_Response("The answer is partially wr", "length"))
    response = await provider.generate(_request(2048))
    assert response.text.startswith("The answer is")
    assert response.finish_reason == "length"
