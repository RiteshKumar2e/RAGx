"""Groq provider.

Groq serves latency-sensitive work in RAGX -- query analysis, reranking
judgements, claim extraction -- and acts as the fallback generator when Gemini
is unavailable. Groq's OpenAI-compatible chat API is text-only here, so
multimodal requests are never routed to it (see ``app.llm.gateway``).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator

from app.core.config import get_settings
from app.core.errors import ProviderError, ProviderNotConfiguredError
from app.core.logging import get_logger
from app.core.text import estimate_tokens
from app.llm.base import LLMProvider, LLMRequest, LLMResponse, Role, TokenUsage

log = get_logger("ragx.llm.groq")


class GroqProvider(LLMProvider):
    name = "groq"
    supports_multimodal = False
    supports_streaming = True
    supports_embeddings = False

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.groq_api_key
        self._model = settings.groq_model
        self._fast_model = settings.groq_fast_model
        self.input_cost_per_mtok = settings.groq_input_cost_per_mtok
        self.output_cost_per_mtok = settings.groq_output_cost_per_mtok
        self._timeout = settings.llm_timeout_seconds
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise ProviderNotConfiguredError("GROQ_API_KEY is not set.")
        try:
            from groq import AsyncGroq  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ProviderNotConfiguredError(
                "The 'groq' package is not installed. Run: pip install groq"
            ) from exc
        self._client = AsyncGroq(api_key=self._api_key, timeout=self._timeout)
        return self._client

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    @property
    def default_model(self) -> str:
        return self._model

    @property
    def fast_model(self) -> str:
        return self._fast_model

    # -------------------------------------------------------------- payloads
    @staticmethod
    def _to_messages(request: LLMRequest) -> list[dict[str, str]]:
        role_map = {Role.SYSTEM: "system", Role.USER: "user", Role.ASSISTANT: "assistant"}
        payload: list[dict[str, str]] = []
        for message in request.messages:
            content = message.content
            if message.images:
                # Groq's text models cannot consume image bytes. Rather than
                # silently dropping the visual evidence we hand the model the
                # image's textual description so the caller still gets an
                # answer; the gateway prefers Gemini for multimodal work.
                labels = ", ".join(i.label or "image" for i in message.images)
                content = f"{content}\n\n[Non-textual evidence not renderable by this provider: {labels}]"
            payload.append({"role": role_map[message.role], "content": content})
        return payload

    def _kwargs(self, request: LLMRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": request.model or self._model,
            "messages": self._to_messages(request),
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }
        if request.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if request.stop:
            kwargs["stop"] = request.stop
        return kwargs

    # ------------------------------------------------------------- generate
    async def generate(self, request: LLMRequest) -> LLMResponse:
        client = self._ensure_client()
        kwargs = self._kwargs(request)
        started = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(**kwargs), timeout=self._timeout
            )
        except asyncio.TimeoutError as exc:
            raise ProviderError(f"Groq timed out after {self._timeout:.0f}s.") from exc
        except Exception as exc:
            raise ProviderError("The Groq API call failed.", detail=str(exc)) from exc

        latency_ms = (time.perf_counter() - started) * 1000
        choice = response.choices[0] if response.choices else None
        text = (choice.message.content if choice and choice.message else "") or ""
        finish_reason = getattr(choice, "finish_reason", None) if choice else None

        if not text.strip():
            # Groq's gpt-oss models are reasoning models: they spend output
            # tokens on an internal reasoning channel before emitting any
            # content. If max_output_tokens is small, the whole budget goes to
            # reasoning and `content` comes back empty with finish_reason
            # "length" -- a successful HTTP call carrying no answer.
            #
            # Returning that as success would hand the caller a blank answer and
            # hide the cause. Raising instead lets the gateway retry or fall back
            # to a provider that can actually answer.
            reasoning = getattr(choice.message, "reasoning", None) if choice and choice.message else None
            if finish_reason == "length":
                raise ProviderError(
                    f"Groq model '{kwargs['model']}' returned no content: the "
                    f"{request.max_output_tokens}-token output budget was consumed by "
                    "the model's internal reasoning. Raise max_output_tokens for this "
                    "call, or configure a non-reasoning GROQ_MODEL.",
                    detail=f"finish_reason=length, reasoning_chars={len(reasoning or '')}",
                )
            raise ProviderError(
                f"Groq model '{kwargs['model']}' returned an empty response.",
                detail=f"finish_reason={finish_reason}",
            )

        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            usage = TokenUsage(
                prompt_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(raw_usage, "completion_tokens", 0) or 0,
                reported=True,
            )
        else:  # pragma: no cover - Groq always reports usage today
            usage = TokenUsage(
                prompt_tokens=sum(estimate_tokens(m.content) for m in request.messages),
                completion_tokens=estimate_tokens(text),
            )

        return LLMResponse(
            text=text,
            provider=self.name,
            model=kwargs["model"],
            usage=usage,
            latency_ms=latency_ms,
            cost_usd=self.estimate_cost(usage),
            finish_reason=finish_reason,
            raw={"usage_reported": usage.reported},
        )

    # --------------------------------------------------------------- stream
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        client = self._ensure_client()
        kwargs = self._kwargs(request)
        kwargs["stream"] = True
        try:
            stream = await client.chat.completions.create(**kwargs)
            async for chunk in stream:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                piece = getattr(delta, "content", None) if delta else None
                if piece:
                    yield piece
        except Exception as exc:
            raise ProviderError("The Groq streaming call failed.", detail=str(exc)) from exc

    # ---------------------------------------------------------------- health
    async def health(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "provider": self.name,
            "configured": self.is_configured,
            "model": self._model,
            "fast_model": self._fast_model,
            "multimodal": False,
            "kind": "cloud",
        }
        if not self.is_configured:
            status.update(status_text="not_configured", healthy=False)
            return status
        try:
            from app.llm.base import Message  # noqa: PLC0415

            probe = LLMRequest(
                messages=[Message.user("ping")],
                purpose="healthcheck",
                # Enough headroom for a reasoning model to finish reasoning and
                # still emit a token. A budget of ~8 is spent entirely on the
                # reasoning channel, so the probe would report the provider
                # unhealthy while the API was in fact reachable.
                max_output_tokens=256,
                temperature=0.0,
                model=self._fast_model,
            )
            response = await self.generate(probe)
            status.update(status_text="healthy", healthy=True, latency_ms=round(response.latency_ms, 1))
        except Exception as exc:
            status.update(status_text="unhealthy", healthy=False, error=str(exc)[:200])
        return status


__all__ = ["GroqProvider"]
