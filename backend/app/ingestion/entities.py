"""Entity and relation extraction for Graph RAG.

Extraction is LLM-driven (Gemini/Groq via the gateway) with a deterministic
rule-based pre-pass. The rules catch the high-precision cases -- acronyms, model
identifiers, CamelCase names -- and give the graph something real to work with
even when no API key is configured; the LLM supplies typed relations, which is
what makes multi-hop traversal possible.

Only the most informative chunks are sent to the LLM (bounded by
``entity_extraction_max_chunks``), because entity extraction is the single most
expensive part of ingestion.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.text import estimate_tokens, technical_terms, truncate_words
from app.indexing.graph_store import GraphEntity, GraphRelation, normalize_entity
from app.llm.base import Message
from app.llm.gateway import Purpose, get_gateway
from app.llm.prompts import ENTITY_EXTRACTION_SYSTEM, ENTITY_EXTRACTION_USER

log = get_logger("ragx.entities")

VALID_ENTITY_TYPES = {
    "METHOD", "MODEL", "DATASET", "METRIC", "TASK", "ORGANIZATION", "PERSON",
    "TOOL", "CONCEPT", "ARCHITECTURE", "FRAMEWORK",
}
VALID_RELATION_TYPES = {
    "USES", "PROPOSES", "EVALUATED_ON", "OUTPERFORMS", "EXTENDS", "PART_OF",
    "COMPARED_WITH", "TRAINED_ON", "ACHIEVES", "CITES", "AUTHORED_BY",
    "APPLIED_TO", "RELATED_TO",
}

# Generic phrases that add graph noise without adding retrievable meaning.
STOP_ENTITIES = {
    "the model", "our model", "the dataset", "the method", "our approach", "the approach",
    "this paper", "the paper", "the system", "the network", "the results", "the data",
    "figure", "table", "section", "appendix", "we", "it", "they",
}


@dataclass(slots=True)
class ExtractionResult:
    entities: list[GraphEntity] = field(default_factory=list)
    relations: list[GraphRelation] = field(default_factory=list)
    llm_calls: int = 0
    chunks_processed: int = 0

    def merge(self, other: "ExtractionResult") -> None:
        self.entities.extend(other.entities)
        self.relations.extend(other.relations)
        self.llm_calls += other.llm_calls
        self.chunks_processed += other.chunks_processed


class EntityExtractor:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._gateway = get_gateway()

    # ------------------------------------------------------------ selection
    def select_chunks(self, chunks: list[Any], limit: int | None = None) -> list[Any]:
        """Pick the chunks worth spending an LLM call on.

        Ranked by density of technical terms per token: an abstract or a results
        paragraph yields far more graph structure than a page of boilerplate.
        """
        limit = limit or self._settings.entity_extraction_max_chunks
        scored: list[tuple[float, Any]] = []
        for chunk in chunks:
            text = getattr(chunk, "content", None) or getattr(chunk, "text", "")
            tokens = estimate_tokens(text)
            if tokens < 30:
                continue
            terms = len(technical_terms(text))
            score = terms / max(1.0, tokens / 100)
            section = " ".join(getattr(chunk, "section_path", []) or []).lower()
            # Sections that describe methods and results carry the relations.
            if any(k in section for k in ("abstract", "method", "approach", "result", "experiment", "conclusion")):
                score *= 1.6
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scored[:limit] if score > 0]

    # ------------------------------------------------------------ rule pass
    @staticmethod
    def rule_based(text: str, document_id: str, chunk_id: str) -> list[GraphEntity]:
        counts = Counter(technical_terms(text))
        entities: list[GraphEntity] = []
        for term, count in counts.most_common(10):
            if normalize_entity(term) in STOP_ENTITIES or len(term) < 3:
                continue
            entities.append(
                GraphEntity(
                    name=term,
                    entity_type="CONCEPT",
                    description="",
                    salience=min(0.4 + count * 0.05, 0.75),
                    document_id=document_id,
                    chunk_ids=[chunk_id],
                )
            )
        return entities

    # ------------------------------------------------------------- LLM pass
    async def extract_from_chunk(
        self, text: str, document_id: str, chunk_id: str, document_name: str, section: str
    ) -> ExtractionResult:
        result = ExtractionResult(chunks_processed=1)
        result.entities.extend(self.rule_based(text, document_id, chunk_id))

        if not self._gateway.any_configured:
            return result

        try:
            payload, _ = await self._gateway.complete_json(
                [
                    Message.system(ENTITY_EXTRACTION_SYSTEM),
                    Message.user(
                        ENTITY_EXTRACTION_USER.format(
                            document=document_name,
                            section=section or "(none)",
                            passage=truncate_words(text, 700),
                        )
                    ),
                ],
                Purpose.ENTITY_EXTRACTION,
                default={},
                temperature=0.0,
                max_output_tokens=1400,
            )
            result.llm_calls = 1
        except Exception as exc:
            log.warning("entities.llm_extraction_failed", chunk_id=chunk_id, error=str(exc)[:160])
            return result

        if not isinstance(payload, dict):
            return result

        named: dict[str, str] = {}
        for raw in payload.get("entities", []) or []:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name", "")).strip()
            key = normalize_entity(name)
            if not name or not key or key in STOP_ENTITIES or len(name) > 120:
                continue
            entity_type = str(raw.get("type", "CONCEPT")).upper().strip()
            if entity_type not in VALID_ENTITY_TYPES:
                entity_type = "CONCEPT"
            named[key] = name
            result.entities.append(
                GraphEntity(
                    name=name,
                    entity_type=entity_type,
                    description=str(raw.get("description", ""))[:500],
                    salience=_clamp(raw.get("salience", 0.5)),
                    document_id=document_id,
                    chunk_ids=[chunk_id],
                )
            )

        for raw in payload.get("relations", []) or []:
            if not isinstance(raw, dict):
                continue
            source = str(raw.get("source", "")).strip()
            target = str(raw.get("target", "")).strip()
            src_key, tgt_key = normalize_entity(source), normalize_entity(target)
            # Both endpoints must have been declared as entities; this is the
            # rule that keeps hallucinated edges out of the graph.
            if not src_key or not tgt_key or src_key == tgt_key:
                continue
            if src_key not in named or tgt_key not in named:
                continue
            relation_type = str(raw.get("type", "RELATED_TO")).upper().strip().replace(" ", "_")
            if relation_type not in VALID_RELATION_TYPES:
                relation_type = "RELATED_TO"
            result.relations.append(
                GraphRelation(
                    source=named[src_key],
                    target=named[tgt_key],
                    relation_type=relation_type,
                    confidence=_clamp(raw.get("confidence", 0.5)),
                    document_id=document_id,
                    chunk_id=chunk_id,
                    context=str(raw.get("context", ""))[:400],
                )
            )
        return result

    async def extract_from_chunks(
        self, chunks: list[Any], document_id: str, document_name: str, concurrency: int = 4
    ) -> ExtractionResult:
        selected = self.select_chunks(chunks)
        if not selected:
            return ExtractionResult()

        semaphore = asyncio.Semaphore(concurrency)

        async def _one(chunk: Any) -> ExtractionResult:
            async with semaphore:
                text = getattr(chunk, "content", None) or getattr(chunk, "text", "")
                section = " > ".join(getattr(chunk, "section_path", []) or [])
                return await self.extract_from_chunk(
                    text=text,
                    document_id=document_id,
                    chunk_id=getattr(chunk, "id", ""),
                    document_name=document_name,
                    section=section,
                )

        results = await asyncio.gather(*(_one(c) for c in selected), return_exceptions=True)
        combined = ExtractionResult()
        for item in results:
            if isinstance(item, ExtractionResult):
                combined.merge(item)
            elif isinstance(item, Exception):
                log.warning("entities.chunk_failed", error=str(item)[:160])

        combined.entities = self._dedupe_entities(combined.entities)
        log.info(
            "entities.extracted",
            document_id=document_id,
            entities=len(combined.entities),
            relations=len(combined.relations),
            chunks=combined.chunks_processed,
            llm_calls=combined.llm_calls,
        )
        return combined

    @staticmethod
    def _dedupe_entities(entities: list[GraphEntity]) -> list[GraphEntity]:
        merged: dict[str, GraphEntity] = {}
        for entity in entities:
            key = entity.normalized
            if not key:
                continue
            existing = merged.get(key)
            if existing is None:
                merged[key] = entity
                continue
            existing.salience = max(existing.salience, entity.salience)
            if len(entity.description) > len(existing.description):
                existing.description = entity.description
            if existing.entity_type == "CONCEPT" and entity.entity_type != "CONCEPT":
                existing.entity_type = entity.entity_type
            for chunk_id in entity.chunk_ids:
                if chunk_id and chunk_id not in existing.chunk_ids:
                    existing.chunk_ids.append(chunk_id)
        return list(merged.values())


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return 0.5


__all__ = ["EntityExtractor", "ExtractionResult", "VALID_ENTITY_TYPES", "VALID_RELATION_TYPES"]
