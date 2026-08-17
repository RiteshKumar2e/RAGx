"""Provider-agnostic LLM contract.

Business logic in RAGX only ever depends on these types and on
:class:`LLMProvider`. Swapping Gemini for Groq (or adding a third provider)
requires no change outside ``app/llm``.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Literal


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(slots=True)
class ImagePart:
    """An inline image passed to a multimodal-capable provider."""

    data: bytes
    mime_type: str = "image/png"
    label: str | None = None


@dataclass(slots=True)
class Message:
    role: Role
    content: str
    images: list[ImagePart] = field(default_factory=list)

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(Role.SYSTEM, content)

    @classmethod
    def user(cls, content: str, images: list[ImagePart] | None = None) -> "Message":
        return cls(Role.USER, content, images or [])

    @classmethod
    def assistant(cls, content: str) -> "Message":
        return cls(Role.ASSISTANT, content)

    @property
    def is_multimodal(self) -> bool:
        return bool(self.images)


@dataclass(slots=True)
class LLMRequest:
    messages: list[Message]
    purpose: str = "generation"
    temperature: float = 0.2
    max_output_tokens: int = 2048
    json_mode: bool = False
    model: str | None = None
    stop: list[str] | None = None

    @property
    def requires_multimodal(self) -> bool:
        return any(m.is_multimodal for m in self.messages)


@dataclass(slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reported: bool = False  # True when the provider returned real counts

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(slots=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    usage: TokenUsage
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    finish_reason: str | None = None
    fallback_used: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    # -- structured output helpers ----------------------------------------
    def json(self, default: Any = None) -> Any:
        """Parse the response as JSON, tolerating markdown code fences."""
        return parse_json_response(self.text, default=default)


_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def parse_json_response(text: str, default: Any = None) -> Any:
    """Best-effort JSON extraction from an LLM response.

    Models occasionally wrap JSON in prose or code fences even under explicit
    instruction; this recovers the payload instead of failing the request.
    """
    if not text:
        return default
    candidates: list[str] = []
    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())
    candidates.append(text.strip())
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    return default


class LLMProvider(ABC):
    """Interface every cloud LLM provider implements.

    RAGX only supports cloud providers. Local model runtimes (Ollama,
    llama.cpp, local Llama weights) are intentionally out of scope.
    """

    name: str = "base"
    supports_multimodal: bool = False
    supports_streaming: bool = True
    supports_embeddings: bool = False

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """True when an API key is present and a client could be constructed."""

    @property
    @abstractmethod
    def default_model(self) -> str: ...

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse: ...

    @abstractmethod
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]: ...

    @abstractmethod
    async def health(self) -> dict[str, Any]: ...

    async def embed(self, texts: list[str], task_type: str = "retrieval_document") -> list[list[float]]:
        raise NotImplementedError(f"{self.name} does not provide an embedding API.")

    # -- shared cost accounting -------------------------------------------
    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0

    def estimate_cost(self, usage: TokenUsage) -> float:
        return (
            usage.prompt_tokens / 1_000_000 * self.input_cost_per_mtok
            + usage.completion_tokens / 1_000_000 * self.output_cost_per_mtok
        )


ProviderName = Literal["gemini", "groq"]

__all__ = [
    "Role",
    "Message",
    "ImagePart",
    "LLMRequest",
    "LLMResponse",
    "TokenUsage",
    "LLMProvider",
    "ProviderName",
    "parse_json_response",
]
