"""Gemini provider (Google Gen AI SDK).

Gemini is the primary cloud LLM for RAGX: query analysis, reasoning, evidence
synthesis, generation and multimodal understanding. It also supplies the
production embedding model.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator

from app.core.config import get_settings
from app.core.errors import ProviderError, ProviderNotConfiguredError
from app.core.logging import get_logger
from app.core.text import estimate_tokens
from app.llm.base import LLMProvider, LLMRequest, LLMResponse, Message, Role, TokenUsage

log = get_logger("ragx.llm.gemini")


def _is_rate_limited(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "quota" in text.lower()


def _is_transient(exc: Exception) -> bool:
    """Errors worth retrying: rate limits and upstream unavailability."""
    text = str(exc)
    return any(code in text for code in ("429", "500", "502", "503", "504")) or (
        "RESOURCE_EXHAUSTED" in text or "UNAVAILABLE" in text
    )


class GeminiProvider(LLMProvider):
    name = "gemini"
    supports_multimodal = True
    supports_streaming = True
    supports_embeddings = True

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.gemini_api_key
        self._model = settings.gemini_model
        self._embedding_model = settings.gemini_embedding_model
        self.input_cost_per_mtok = settings.gemini_input_cost_per_mtok
        self.output_cost_per_mtok = settings.gemini_output_cost_per_mtok
        self._timeout = settings.llm_timeout_seconds
        # Embedding retries are more generous than chat retries: a rate-limit
        # window can be tens of seconds, and losing them fails a whole ingest.
        self._embed_max_retries = max(settings.llm_max_retries, 4)
        self._client: Any = None
        self._types: Any = None
        self._init_error: str | None = None

    # ------------------------------------------------------------------ init
    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise ProviderNotConfiguredError("GEMINI_API_KEY is not set.")
        try:
            from google import genai  # noqa: PLC0415
            from google.genai import types  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ProviderNotConfiguredError(
                "The 'google-genai' package is not installed. Run: pip install google-genai"
            ) from exc
        try:
            self._client = genai.Client(api_key=self._api_key)
            self._types = types
        except Exception as exc:
            self._init_error = str(exc)
            raise ProviderError("Failed to initialise the Gemini client.", detail=str(exc)) from exc
        return self._client

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    @property
    def default_model(self) -> str:
        return self._model

    # -------------------------------------------------------------- payloads
    def _build_contents(self, messages: list[Message]) -> tuple[list[Any], str | None]:
        """Translate the neutral message list into Gemini ``contents``.

        Gemini carries the system prompt out-of-band via ``system_instruction``.
        """
        types = self._types
        system_parts: list[str] = []
        contents: list[Any] = []

        for message in messages:
            if message.role is Role.SYSTEM:
                system_parts.append(message.content)
                continue
            parts: list[Any] = []
            if message.content:
                parts.append(types.Part.from_text(text=message.content))
            for image in message.images:
                parts.append(types.Part.from_bytes(data=image.data, mime_type=image.mime_type))
            if not parts:
                continue
            role = "user" if message.role is Role.USER else "model"
            contents.append(types.Content(role=role, parts=parts))

        return contents, "\n\n".join(system_parts) if system_parts else None

    def _build_config(self, request: LLMRequest, system_instruction: str | None) -> Any:
        types = self._types
        kwargs: dict[str, Any] = {
            "temperature": request.temperature,
            "max_output_tokens": request.max_output_tokens,
        }
        if system_instruction:
            kwargs["system_instruction"] = system_instruction
        if request.json_mode:
            kwargs["response_mime_type"] = "application/json"
        if request.stop:
            kwargs["stop_sequences"] = request.stop
        return types.GenerateContentConfig(**kwargs)

    @staticmethod
    def _usage_from(response: Any, request: LLMRequest, text: str) -> TokenUsage:
        meta = getattr(response, "usage_metadata", None)
        if meta is not None:
            prompt = getattr(meta, "prompt_token_count", None) or 0
            completion = getattr(meta, "candidates_token_count", None) or 0
            if prompt or completion:
                return TokenUsage(prompt_tokens=prompt, completion_tokens=completion, reported=True)
        prompt_estimate = sum(estimate_tokens(m.content) for m in request.messages)
        return TokenUsage(
            prompt_tokens=prompt_estimate, completion_tokens=estimate_tokens(text), reported=False
        )

    @staticmethod
    def _text_from(response: Any) -> str:
        text = getattr(response, "text", None)
        if text:
            return text
        # Fall back to walking the candidate parts when ``.text`` is empty
        # (happens when a response mixes modalities or is truncated).
        chunks: list[str] = []
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    chunks.append(part_text)
        return "".join(chunks)

    # ------------------------------------------------------------- generate
    async def generate(self, request: LLMRequest) -> LLMResponse:
        client = self._ensure_client()
        contents, system_instruction = self._build_contents(request.messages)
        config = self._build_config(request, system_instruction)
        model = request.model or self._model

        started = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(model=model, contents=contents, config=config),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError as exc:
            raise ProviderError(f"Gemini timed out after {self._timeout:.0f}s.") from exc
        except Exception as exc:
            raise ProviderError("The Gemini API call failed.", detail=str(exc)) from exc

        latency_ms = (time.perf_counter() - started) * 1000
        text = self._text_from(response)
        usage = self._usage_from(response, request, text)
        finish_reason = None
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish_reason = str(getattr(candidates[0], "finish_reason", "") or "") or None

        return LLMResponse(
            text=text,
            provider=self.name,
            model=model,
            usage=usage,
            latency_ms=latency_ms,
            cost_usd=self.estimate_cost(usage),
            finish_reason=finish_reason,
            raw={"usage_reported": usage.reported},
        )

    # --------------------------------------------------------------- stream
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        client = self._ensure_client()
        contents, system_instruction = self._build_contents(request.messages)
        config = self._build_config(request, system_instruction)
        model = request.model or self._model
        try:
            stream = await client.aio.models.generate_content_stream(
                model=model, contents=contents, config=config
            )
            async for chunk in stream:
                piece = self._text_from(chunk)
                if piece:
                    yield piece
        except Exception as exc:
            raise ProviderError("The Gemini streaming call failed.", detail=str(exc)) from exc

    # ------------------------------------------------------------ embeddings
    async def embed(
        self,
        texts: list[str],
        task_type: str = "retrieval_document",
        dimension: int | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        client = self._ensure_client()
        types = self._types

        config_kwargs: dict[str, Any] = {"task_type": task_type.upper()}
        if dimension:
            # gemini-embedding-001 emits 3072 dimensions by default. Requesting
            # the configured size keeps the Qdrant collection dimension stable
            # and avoids re-indexing when the model's native size changes.
            config_kwargs["output_dimensionality"] = dimension
        config = types.EmbedContentConfig(**config_kwargs)

        # Ingesting a large document issues many batches in quick succession,
        # which reliably trips the per-minute quota on Gemini's free tier. A 429
        # is recoverable by waiting, so retry it -- previously a single 429
        # aborted the whole document.
        attempts = max(1, self._embed_max_retries + 1)
        delay = 2.0
        last: Exception | None = None

        for attempt in range(attempts):
            try:
                response = await asyncio.wait_for(
                    client.aio.models.embed_content(
                        model=self._embedding_model, contents=texts, config=config
                    ),
                    timeout=self._timeout,
                )
                return [list(item.values) for item in (response.embeddings or [])]
            except asyncio.TimeoutError as exc:
                last = exc
            except Exception as exc:
                last = exc
                if not _is_transient(exc):
                    raise ProviderError(
                        "The Gemini embedding call failed.", detail=str(exc)[:400]
                    ) from exc

            if attempt < attempts - 1:
                log.warning(
                    "embeddings.retrying",
                    attempt=attempt + 1,
                    of=attempts,
                    wait_seconds=round(delay, 1),
                    rate_limited=_is_rate_limited(last) if last else False,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)

        if last is not None and _is_rate_limited(last):
            raise ProviderError(
                "Gemini embedding quota exceeded, so the document was not fully indexed. "
                "Wait for the quota window to reset, lower EMBEDDING_BATCH_SIZE, or upgrade "
                "the API plan — then reindex the document.",
                detail=str(last)[:400],
            ) from last
        raise ProviderError(
            "The Gemini embedding call failed after retries.", detail=str(last)[:400]
        ) from last

    # ---------------------------------------------------------------- health
    async def health(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "provider": self.name,
            "configured": self.is_configured,
            "model": self._model,
            "embedding_model": self._embedding_model,
            "multimodal": True,
            "kind": "cloud",
        }
        if not self.is_configured:
            status.update(status_text="not_configured", healthy=False)
            return status
        try:
            probe = LLMRequest(
                messages=[Message.user("ping")],
                purpose="healthcheck",
                max_output_tokens=8,
                temperature=0.0,
            )
            response = await self.generate(probe)
            status.update(status_text="healthy", healthy=True, latency_ms=round(response.latency_ms, 1))
        except Exception as exc:
            status.update(status_text="unhealthy", healthy=False, error=str(exc)[:200])
        return status


__all__ = ["GeminiProvider"]
