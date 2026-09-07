"""Rate-limit aware retry helpers shared by the LLM gateway.

Both providers RAGX uses are commonly run on free tiers with tight limits, and
both *tell you when to come back*: Gemini returns ``retryDelay: '10s'`` inside a
429 body, and Groq sends a ``retry-after`` header. A fixed backoff ignores that
and picks the wrong delay in both directions -- retrying while the window is
still closed (wasting the retry budget, then failing over unnecessarily), or
waiting far longer than needed.

Reading the delay from the error makes a rate limit recoverable instead of
terminal, which matters most for the token-per-minute limits that a
multi-strategy query can hit part-way through a pipeline.
"""

from __future__ import annotations

import re

#: Never sleep longer than this on a single attempt, however long the provider
#: asks for. A daily quota reports a delay measured in minutes or hours;
#: blocking a request for that long is worse than failing over to the other
#: provider, or surfacing the limit to the caller.
MAX_RETRY_DELAY_SECONDS = 30.0

_RATE_LIMIT_MARKERS = ("429", "RESOURCE_EXHAUSTED", "rate_limit", "rate limit", "quota")
_TRANSIENT_MARKERS = ("429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED", "UNAVAILABLE")

#: Gemini reports `'retryDelay': '10s'`, prose form "Please retry in
#: 10.969884611s". Groq sends a `retry-after` header and, for a daily token
#: limit, prose in Go duration form: "Please try again in 13m30.863999999s".
#: The prose group is therefore captured loosely and handed to _parse_duration,
#: so a compound duration is not silently read as its leading number.
_DELAY_PATTERNS = (
    re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?[hms]+)", re.I),
    re.compile(r"retry[- ]after['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)", re.I),
    re.compile(r"(?:please )?(?:retry|try again) in\s+([\d.hms]+)", re.I),
)


def is_rate_limited(exc: BaseException) -> bool:
    text = str(exc)
    lowered = text.lower()
    return any(marker in lowered for marker in (m.lower() for m in _RATE_LIMIT_MARKERS))


def is_transient(exc: BaseException) -> bool:
    """Errors worth retrying: rate limits and upstream unavailability."""
    text = str(exc)
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def _from_headers(exc: BaseException) -> float | None:
    """Read ``retry-after`` off an SDK exception's HTTP response, if present."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    for key in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        try:
            raw = headers.get(key)
        except Exception:
            continue
        if not raw:
            continue
        seconds = _parse_duration(str(raw))
        if seconds is not None:
            return seconds
    return None


def _parse_duration(value: str) -> float | None:
    """Parse ``"2.5"``, ``"810ms"``, ``"13m30.86s"``, ``"2h8m9.6s"`` into seconds."""
    value = value.strip().rstrip(".")
    try:
        return float(value)
    except ValueError:
        pass

    match = re.fullmatch(
        r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?"
        r"(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?",
        value,
        re.I,
    )
    if not match or not any(match.groups()):
        return None
    hours, minutes, seconds, millis = (float(g) if g else 0.0 for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds + millis / 1000


def retry_delay_for(exc: BaseException, fallback: float) -> float:
    """Seconds to wait before retrying ``exc``.

    Prefers the delay the provider asked for -- from a response header, then from
    the error body -- and uses ``fallback`` only when neither is available. The
    result is clamped to :data:`MAX_RETRY_DELAY_SECONDS`.
    """
    delay = _from_headers(exc)
    if delay is None:
        text = str(exc)
        for pattern in _DELAY_PATTERNS:
            match = pattern.search(text)
            if match:
                delay = _parse_duration(match.group(1))
                if delay is not None:
                    break
    if delay is None or delay <= 0:
        delay = fallback
    return min(delay, MAX_RETRY_DELAY_SECONDS)


__all__ = [
    "MAX_RETRY_DELAY_SECONDS",
    "is_rate_limited",
    "is_transient",
    "retry_delay_for",
]
