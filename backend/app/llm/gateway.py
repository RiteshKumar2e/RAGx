"""LLM Gateway -- the single seam between RAGX and any cloud LLM.

Responsibilities:

* **Provider selection per purpose.** Latency-sensitive internal calls (query
  analysis, relevance grading, claim extraction) prefer Groq; long-form
  reasoning and synthesis prefer the configured primary; anything carrying
  images is forced to a multimodal-capable provider.
* **Retry and fallback.** Transient failures retry with backoff on the same
  provider, then fail over to the secondary provider.
* **Accounting.** Every call is recorded on the request's ``TraceRecorder``
  with provider, model, latency, token usage and estimated cost.

No module outside ``app/llm`` imports a provider SDK.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator

from app.core.config import get_settings
from app.core.errors import ProviderError, ProviderNotConfiguredError
from app.core.logging import TraceRecorder, get_logger
from app.llm.base import LLMProvider, LLMRequest, LLMResponse, Message
from app.llm.gemini.provider import GeminiProvider
from app.llm.groq.provider import GroqProvider

log = get_logger("ragx.llm.gateway")


class Purpose(str, Enum):
    """What a call is for. Drives model/provider selection."""

    QUERY_ANALYSIS = "query_analysis"
    QUERY_REWRITE = "query_rewrite"
    HYDE = "hyde"
    RELEVANCE_GRADING = "relevance_grading"
    RERANK = "rerank"
    ENTITY_EXTRACTION = "entity_extraction"
    CLAIM_EXTRACTION = "claim_extraction"
    EVIDENCE_MATCHING = "evidence_matching"
    AGENT_PLANNING = "agent_planning"
    SYNTHESIS = "synthesis"
    GENERATION = "generation"
    MULTIMODAL = "multimodal"
    JUDGE = "judge"
    HEALTHCHECK = "healthcheck"


# Purposes where latency matters more than depth -> prefer Groq's fast model.
_FAST_PURPOSES = {
    Purpose.QUERY_ANALYSIS,
    Purpose.QUERY_REWRITE,
    Purpose.RELEVANCE_GRADING,
    Purpose.RERANK,
    Purpose.CLAIM_EXTRACTION,
    Purpose.EVIDENCE_MATCHING,
}

# Purposes that must run on a multimodal-capable provider.
_MULTIMODAL_PURPOSES = {Purpose.MULTIMODAL}


@dataclass(slots=True)
class ProviderChoice:
    provider: LLMProvider
    model: str
    reason: str


class LLMGateway:
    """Facade over the configured cloud providers."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._providers: dict[str, LLMProvider] = {
            "gemini": GeminiProvider(),
            "groq": GroqProvider(),
        }

    # ------------------------------------------------------------ inventory
    @property
    def providers(self) -> dict[str, LLMProvider]:
        return self._providers

    def get(self, name: str) -> LLMProvider:
        provider = self._providers.get(name)
        if provider is None:
            raise ProviderNotConfiguredError(f"Unknown LLM provider '{name}'.")
        return provider

    @property
    def configured_providers(self) -> list[str]:
        return [name for name, p in self._providers.items() if p.is_configured]

    @property
    def any_configured(self) -> bool:
        return bool(self.configured_providers)

    # ------------------------------------------------------------- routing
    def select(self, purpose: Purpose, multimodal: bool = False) -> tuple[ProviderChoice, ProviderChoice | None]:
        """Return the (primary, fallback) provider choice for a purpose."""
        settings = self._settings
        gemini, groq = self._providers["gemini"], self._providers["groq"]
        available = [p for p in (gemini, groq) if p.is_configured]
        if not available:
            raise ProviderNotConfiguredError(
                "No cloud LLM provider is configured. Set GEMINI_API_KEY and/or GROQ_API_KEY."
            )

        # Images can only go to a multimodal-capable provider; there is no
        # meaningful text-only fallback for a visual question.
        if multimodal or purpose in _MULTIMODAL_PURPOSES:
            capable = [p for p in available if p.supports_multimodal]
            if not capable:
                raise ProviderNotConfiguredError(
                    "This query needs multimodal understanding, which requires Gemini. "
                    "Set GEMINI_API_KEY."
                )
            chosen = capable[0]
            return (
                ProviderChoice(chosen, chosen.default_model, "multimodal input requires a vision-capable provider"),
                None,
            )

        if purpose in _FAST_PURPOSES and groq.is_configured:
            primary = ProviderChoice(groq, groq.fast_model, "latency-sensitive internal reasoning step")
            fallback = (
                ProviderChoice(gemini, gemini.default_model, "primary provider failed")
                if gemini.is_configured
                else None
            )
            return primary, fallback

        preferred_name = settings.primary_llm_provider
        preferred = self._providers.get(preferred_name)
        if preferred is None or not preferred.is_configured:
            preferred = available[0]

        model = preferred.default_model
        if preferred.name == "gemini" and purpose in {Purpose.SYNTHESIS, Purpose.AGENT_PLANNING, Purpose.JUDGE}:
            model = settings.gemini_reasoning_model

        primary = ProviderChoice(preferred, model, f"configured primary provider for {purpose.value}")

        fallback_name = settings.fallback_llm_provider
        fallback: ProviderChoice | None = None
        if fallback_name != "none":
            candidate = self._providers.get(fallback_name)
            if candidate is not None and candidate.is_configured and candidate is not preferred:
                fallback = ProviderChoice(candidate, candidate.default_model, "primary provider failed")
        if fallback is None:
            for candidate in available:
                if candidate is not preferred:
                    fallback = ProviderChoice(candidate, candidate.default_model, "primary provider failed")
                    break
        return primary, fallback

    # ---------------------------------------------------------- invocation
    async def complete(
        self,
        messages: list[Message],
        purpose: Purpose = Purpose.GENERATION,
        *,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
        json_mode: bool = False,
        trace: TraceRecorder | None = None,
        provider_override: str | None = None,
    ) -> LLMResponse:
        """Run one completion with retry, then fallback."""
        multimodal = any(m.is_multimodal for m in messages)

        if provider_override:
            provider = self.get(provider_override)
            if not provider.is_configured:
                raise ProviderNotConfiguredError(f"Provider '{provider_override}' has no API key configured.")
            primary = ProviderChoice(provider, provider.default_model, "explicit provider override")
            fallback = None
        else:
            primary, fallback = self.select(purpose, multimodal=multimodal)

        request = LLMRequest(
            messages=messages,
            purpose=purpose.value,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            json_mode=json_mode,
            model=primary.model,
        )

        attempts: list[tuple[ProviderChoice, bool]] = [(primary, False)]
        if fallback is not None:
            attempts.append((fallback, True))

        last_error: Exception | None = None
        for choice, is_fallback in attempts:
            request.model = choice.model
            try:
                response = await self._with_retry(choice.provider, request)
            except Exception as exc:
                last_error = exc
                log.warning(
                    "llm.provider_failed",
                    provider=choice.provider.name,
                    purpose=purpose.value,
                    error=str(exc)[:200],
                    will_fallback=not is_fallback and fallback is not None,
                )
                if trace is not None:
                    trace.record_llm(
                        provider=choice.provider.name,
                        model=choice.model,
                        purpose=purpose.value,
                        latency_ms=0.0,
                        prompt_tokens=0,
                        completion_tokens=0,
                        cost_usd=0.0,
                        fallback_used=is_fallback,
                        error=str(exc)[:200],
                    )
                continue

            response.fallback_used = is_fallback
            if trace is not None:
                trace.record_llm(
                    provider=response.provider,
                    model=response.model,
                    purpose=purpose.value,
                    latency_ms=response.latency_ms,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    cost_usd=response.cost_usd,
                    fallback_used=is_fallback,
                )
            return response

        raise ProviderError(
            "Every configured LLM provider failed for this request.",
            detail=str(last_error)[:400] if last_error else None,
        )

    async def _with_retry(self, provider: LLMProvider, request: LLMRequest) -> LLMResponse:
        max_retries = max(0, self._settings.llm_max_retries)
        delay = 0.75
        last: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return await provider.generate(request)
            except ProviderNotConfiguredError:
                raise
            except Exception as exc:
                last = exc
                if attempt >= max_retries:
                    break
                await asyncio.sleep(delay)
                delay *= 2
        raise last if last else ProviderError("The LLM call failed.")

    async def complete_json(
        self,
        messages: list[Message],
        purpose: Purpose,
        *,
        default: Any = None,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        trace: TraceRecorder | None = None,
    ) -> tuple[Any, LLMResponse]:
        """Structured-output helper: returns ``(parsed_or_default, response)``."""
        response = await self.complete(
            messages,
            purpose,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            json_mode=True,
            trace=trace,
        )
        return response.json(default=default), response

    async def stream(
        self,
        messages: list[Message],
        purpose: Purpose = Purpose.GENERATION,
        *,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
        trace: TraceRecorder | None = None,
    ) -> AsyncIterator[str]:
        """Stream tokens from the selected provider.

        On a streaming failure we fall back to a single non-streaming call on
        the secondary provider and emit its text as one chunk, so the caller's
        contract (an async iterator of strings) always holds.
        """
        multimodal = any(m.is_multimodal for m in messages)
        primary, fallback = self.select(purpose, multimodal=multimodal)
        request = LLMRequest(
            messages=messages,
            purpose=purpose.value,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            model=primary.model,
        )

        emitted = False
        try:
            async for piece in primary.provider.stream(request):
                emitted = True
                yield piece
        except Exception as exc:
            if emitted or fallback is None:
                raise
            log.warning("llm.stream_failed_falling_back", provider=primary.provider.name, error=str(exc)[:200])
            request.model = fallback.model
            response = await self._with_retry(fallback.provider, request)
            if trace is not None:
                trace.record_llm(
                    provider=response.provider,
                    model=response.model,
                    purpose=purpose.value,
                    latency_ms=response.latency_ms,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    cost_usd=response.cost_usd,
                    fallback_used=True,
                )
            yield response.text

    # ------------------------------------------------------------- health
    async def health(self, probe: bool = False) -> dict[str, Any]:
        """Provider status. With ``probe=False`` this is configuration-only
        (free); with ``probe=True`` each provider gets a tiny live request."""
        providers: list[dict[str, Any]] = []
        for provider in self._providers.values():
            if probe and provider.is_configured:
                providers.append(await provider.health())
            else:
                providers.append(
                    {
                        "provider": provider.name,
                        "configured": provider.is_configured,
                        "model": provider.default_model,
                        "multimodal": provider.supports_multimodal,
                        "kind": "cloud",
                        "status_text": "configured" if provider.is_configured else "not_configured",
                        "healthy": None,
                    }
                )
        settings = self._settings
        return {
            "providers": providers,
            "primary": settings.primary_llm_provider,
            "fallback": settings.fallback_llm_provider,
            "any_configured": self.any_configured,
            "local_llms_supported": False,
        }


_gateway: LLMGateway | None = None


def get_gateway() -> LLMGateway:
    global _gateway
    if _gateway is None:
        _gateway = LLMGateway()
    return _gateway


def reset_gateway() -> None:
    """Rebuild the gateway (used after a settings change or in tests)."""
    global _gateway
    _gateway = None


__all__ = ["LLMGateway", "Purpose", "ProviderChoice", "get_gateway", "reset_gateway"]
