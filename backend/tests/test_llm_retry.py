"""Rate-limit aware retry.

Both providers RAGX uses report *when* to retry, and both were being ignored in
favour of a fixed backoff. That turned a recoverable token-per-minute limit into
a hard failure: the retry fired while the window was still closed, the attempt
budget was spent, and the request failed over to a provider that was itself out
of daily quota -- surfacing as "every configured LLM provider failed".

The strings below are real, captured from live 429 responses.
"""

from __future__ import annotations

import pytest

from app.llm.retry import (
    MAX_RETRY_DELAY_SECONDS,
    is_rate_limited,
    is_transient,
    retry_delay_for,
)

GEMINI_429 = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your "
    "current quota. \\n* Quota exceeded for metric: "
    "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, "
    "model: gemini-3.7-flash\\nPlease retry in 10.969884611s.', 'status': "
    "'RESOURCE_EXHAUSTED', 'details': [{'@type': "
    "'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '10s'}]}}"
)

GROQ_TPD = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
    "`openai/gpt-oss-120b` in organization `org_x` service tier `on_demand` on tokens "
    "per day (TPD): Limit 200000, Used 199751, Requested 2126. Please try again in "
    "13m30.863999999s.', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
)


class _FakeResponse:
    def __init__(self, headers: dict) -> None:
        self.headers = headers


class _FakeSDKError(Exception):
    def __init__(self, message: str, headers: dict) -> None:
        super().__init__(message)
        self.response = _FakeResponse(headers)


def test_gemini_retry_delay_is_read_from_the_error_body() -> None:
    # 10s from retryDelay -- not the fixed 0.75s backoff, which would retry
    # while the quota window is still closed.
    assert retry_delay_for(Exception(GEMINI_429), fallback=0.75) == pytest.approx(10.0)


def test_groq_retry_after_header_wins_over_the_fallback() -> None:
    exc = _FakeSDKError("429 rate_limit_exceeded", {"retry-after": "2.5"})
    assert retry_delay_for(exc, fallback=0.75) == pytest.approx(2.5)


def test_groq_token_window_reset_is_parsed_from_its_duration_format() -> None:
    """A tokens-per-minute limit clears in well under a second; the header says so."""
    exc = _FakeSDKError("429 rate_limit_exceeded", {"x-ratelimit-reset-tokens": "810ms"})
    assert retry_delay_for(exc, fallback=5.0) == pytest.approx(0.81)


def test_groq_daily_limit_duration_is_not_read_as_its_leading_number() -> None:
    """"13m30.86s" must not be parsed as 13 seconds.

    Reading the leading number would retry ~13 minutes early, spend the attempt
    and fail over -- the exact behaviour this module exists to prevent.
    """
    # 13m30.86s = 810.86s, clamped: a daily limit is not worth blocking on.
    assert retry_delay_for(Exception(GROQ_TPD), fallback=0.75) == MAX_RETRY_DELAY_SECONDS
    assert is_rate_limited(Exception(GROQ_TPD))


def test_compound_durations_are_parsed() -> None:
    exc = _FakeSDKError("429", {"x-ratelimit-reset-requests": "2h8m9.6s"})
    assert retry_delay_for(exc, fallback=1.0) == MAX_RETRY_DELAY_SECONDS


def test_a_long_requested_delay_is_clamped() -> None:
    assert retry_delay_for(Exception("Please retry in 3600s"), fallback=1.0) == (
        MAX_RETRY_DELAY_SECONDS
    )


def test_falls_back_when_the_provider_says_nothing() -> None:
    assert retry_delay_for(Exception("connection reset"), fallback=1.25) == pytest.approx(1.25)


def test_rate_limits_are_recognised() -> None:
    assert is_rate_limited(Exception(GEMINI_429))
    assert is_rate_limited(Exception("429 rate_limit_exceeded"))
    assert is_rate_limited(Exception("You exceeded your current quota"))
    assert not is_rate_limited(Exception("400 invalid request"))


def test_transient_errors_are_recognised() -> None:
    assert is_transient(Exception("503 UNAVAILABLE"))
    assert is_transient(Exception(GEMINI_429))
    # A malformed request will fail identically on every retry.
    assert not is_transient(Exception("400 invalid argument"))
