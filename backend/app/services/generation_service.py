"""Answer generation.

Builds the grounded prompt from retrieved evidence and calls the LLM Gateway.
Three properties are enforced here rather than left to the model:

* **Numbered evidence blocks.** Each block is labelled ``[n]`` with its document,
  page, section and figure/table label, so a citation the model emits is
  mechanically checkable against the block it names.
* **Context budgeting.** Evidence is packed until a token budget is reached
  rather than truncated arbitrarily, so the highest-scoring evidence always
  survives.
* **Multimodal routing.** When the evidence contains figure or table images, the
  images are attached to the message and the request is routed to a
  vision-capable provider.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from app.core.config import get_settings
from app.core.logging import TraceRecorder, get_logger
from app.core.text import estimate_tokens, truncate_words
from app.llm.base import ImagePart, LLMResponse, Message
from app.llm.gateway import Purpose, get_gateway
from app.llm.prompts import ANSWER_SYSTEM, ANSWER_USER, MULTIMODAL_ANSWER_SYSTEM
from app.retrieval.base import RetrievedChunk

log = get_logger("ragx.generation")

MAX_CONTEXT_TOKENS = 7000
MAX_IMAGES = 4


@dataclass
class GenerationResult:
    answer: str = ""
    provider: str | None = None
    model: str | None = None
    fallback_used: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    multimodal: bool = False
    images_sent: int = 0
    evidence_used: list[RetrievedChunk] = field(default_factory=list)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "fallback_used": self.fallback_used,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.cost_usd, 6),
            "latency_ms": round(self.latency_ms, 2),
            "multimodal": self.multimodal,
            "images_sent": self.images_sent,
        }


class GenerationService:
    def __init__(self) -> None:
        self.gateway = get_gateway()

    # ------------------------------------------------------------ prompting
    @staticmethod
    def format_evidence(chunks: list[RetrievedChunk], budget: int = MAX_CONTEXT_TOKENS) -> tuple[str, list[RetrievedChunk]]:
        """Render evidence blocks within a token budget.

        Returns the rendered text and the chunks that actually fit -- the caller
        must cite against *that* list, not the full retrieval result, or the
        markers would point at evidence the model never saw.
        """
        blocks: list[str] = []
        used: list[RetrievedChunk] = []
        spent = 0

        for index, chunk in enumerate(chunks, start=1):
            header_bits = [chunk.document_name]
            if chunk.page_number:
                header_bits.append(
                    f"page {chunk.page_number}"
                    if not chunk.page_end or chunk.page_end == chunk.page_number
                    else f"pages {chunk.page_number}-{chunk.page_end}"
                )
            if chunk.section:
                header_bits.append(f"section: {chunk.section}")
            if chunk.figure_label:
                header_bits.append(chunk.figure_label)
            if chunk.table_label:
                header_bits.append(chunk.table_label)
            if chunk.modality not in ("text",):
                header_bits.append(f"type: {chunk.modality}")
            if chunk.graph_path:
                header_bits.append(f"graph: {chunk.graph_path}")

            body = chunk.content.strip()
            block = f"[{len(used) + 1}] ({' | '.join(header_bits)})\n{body}"
            block_tokens = estimate_tokens(block)

            if spent + block_tokens > budget:
                if not used:
                    # Always include at least one block, trimmed to fit.
                    block = f"[1] ({' | '.join(header_bits)})\n{truncate_words(body, budget // 2)}"
                    blocks.append(block)
                    used.append(chunk)
                break

            blocks.append(block)
            used.append(chunk)
            spent += block_tokens

        return ("\n\n".join(blocks) if blocks else "(no evidence retrieved)"), used

    @staticmethod
    def format_history(history: list[dict[str, str]], max_turns: int = 4) -> str:
        if not history:
            return ""
        lines = [
            f"{turn.get('role', 'user').capitalize()}: {truncate_words(turn.get('content', ''), 90)}"
            for turn in history[-max_turns:]
        ]
        return "Recent conversation (for pronoun resolution only -- do not treat as evidence):\n" + "\n".join(lines) + "\n\n"

    def build_messages(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        history: list[dict[str, str]] | None = None,
    ) -> tuple[list[Message], list[RetrievedChunk], int]:
        evidence_text, used = self.format_evidence(chunks)
        images: list[ImagePart] = []
        for chunk in used[:MAX_IMAGES]:
            data = chunk.metadata.get("image_bytes")
            if data:
                images.append(
                    ImagePart(
                        data=data,
                        mime_type=chunk.metadata.get("image_mime", "image/png"),
                        label=chunk.figure_label or chunk.table_label or chunk.document_name,
                    )
                )

        system = MULTIMODAL_ANSWER_SYSTEM if images else ANSWER_SYSTEM
        user = ANSWER_USER.format(
            question=question,
            history=self.format_history(history or []),
            evidence=evidence_text,
        )
        return [Message.system(system), Message.user(user, images=images)], used, len(images)

    # ----------------------------------------------------------- generation
    async def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        *,
        history: list[dict[str, str]] | None = None,
        trace: TraceRecorder | None = None,
        provider_override: str | None = None,
    ) -> GenerationResult:
        from app.verification.pipeline import INSUFFICIENT_EVIDENCE_MESSAGE  # noqa: PLC0415

        if not chunks:
            return GenerationResult(
                answer=(
                    f"{INSUFFICIENT_EVIDENCE_MESSAGE}\n\n"
                    "No passages in the indexed documents matched this question."
                ),
                evidence_used=[],
            )

        if not self.gateway.any_configured:
            return GenerationResult(
                answer=(
                    "No cloud LLM provider is configured, so an answer cannot be generated. "
                    "Set `GEMINI_API_KEY` and/or `GROQ_API_KEY` in the backend environment. "
                    "Retrieval still ran -- the evidence panel shows what was found."
                ),
                evidence_used=chunks[:8],
                error="provider_not_configured",
            )

        messages, used, image_count = self.build_messages(question, chunks, history)
        purpose = Purpose.MULTIMODAL if image_count else Purpose.SYNTHESIS

        started = time.perf_counter()
        try:
            response: LLMResponse = await self.gateway.complete(
                messages,
                purpose,
                temperature=0.15,
                max_output_tokens=2048,
                trace=trace,
                provider_override=provider_override,
            )
        except Exception as exc:
            log.error("generation.failed", error=str(exc)[:200])
            return GenerationResult(
                answer=(
                    "The answer could not be generated because every configured LLM provider "
                    "failed. The retrieved evidence is still shown below."
                ),
                evidence_used=used,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=str(exc)[:300],
            )

        return GenerationResult(
            answer=response.text.strip(),
            provider=response.provider,
            model=response.model,
            fallback_used=response.fallback_used,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            cost_usd=response.cost_usd,
            latency_ms=response.latency_ms,
            multimodal=bool(image_count),
            images_sent=image_count,
            evidence_used=used,
        )

    async def stream(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        *,
        history: list[dict[str, str]] | None = None,
        trace: TraceRecorder | None = None,
    ) -> AsyncIterator[str]:
        messages, _, image_count = self.build_messages(question, chunks, history)
        purpose = Purpose.MULTIMODAL if image_count else Purpose.SYNTHESIS
        async for piece in self.gateway.stream(
            messages, purpose, temperature=0.15, max_output_tokens=2048, trace=trace
        ):
            yield piece


_service: GenerationService | None = None


def get_generation_service() -> GenerationService:
    global _service
    if _service is None:
        _service = GenerationService()
    return _service


__all__ = ["GenerationService", "GenerationResult", "get_generation_service"]
