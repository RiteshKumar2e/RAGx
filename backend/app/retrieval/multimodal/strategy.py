"""Multimodal RAG -- figures, charts, tables, images and OCR content.

Two things distinguish it from text retrieval:

1. **Modality-aware retrieval.** A visual query runs one probe restricted to
   visual/tabular chunks and one unrestricted probe, then fuses them with the
   visual side up-weighted. Restricting alone would lose the prose that explains
   the figure; not restricting at all buries the figure under prose.
2. **Image payload loading.** Retrieved figure/image chunks carry an object-store
   key. This strategy fetches those bytes and attaches them to the result so the
   generator can hand the actual pixels to Gemini rather than only the caption.
"""

from __future__ import annotations

import asyncio

from app.core.config import get_settings
from app.core.logging import get_logger
from app.indexing.bm25_index import get_bm25_index
from app.indexing.vector_store import get_vector_store
from app.llm.embeddings import get_embedding_provider
from app.retrieval.base import (
    RetrievalConfig,
    RetrievalContext,
    RetrievalResult,
    RetrievalStrategy,
    RetrievedChunk,
    StrategyName,
)
from app.retrieval.fusion import deduplicate, reciprocal_rank_fusion
from app.retrieval.loader import hydrate
from app.storage import get_object_store

log = get_logger("ragx.retrieval.multimodal")

VISUAL_MODALITIES = ["figure", "image", "table", "ocr"]
MAX_IMAGES = 4
MAX_IMAGE_BYTES = 4 * 1024 * 1024


class MultimodalRAG(RetrievalStrategy):
    name = StrategyName.MULTIMODAL
    description = (
        "Retrieves figures, charts, tables and OCR content alongside text, and loads the "
        "underlying images so a vision-capable model can read them directly."
    )
    uses_llm = False

    def __init__(self) -> None:
        self.vector_store = get_vector_store()
        self.bm25 = get_bm25_index()
        self.embedder = get_embedding_provider()
        self.object_store = get_object_store()

    async def retrieve(
        self, query: str, context: RetrievalContext, config: RetrievalConfig
    ) -> RetrievalResult:
        settings = get_settings()
        pool = max(config.top_k * 3, min(config.candidate_pool, settings.candidate_pool_size))

        requested = config.modalities or VISUAL_MODALITIES
        visual_modalities = [m for m in requested if m in VISUAL_MODALITIES] or VISUAL_MODALITIES

        vector = await self.embedder.embed_query(query)

        visual_hits, general_hits, sparse_hits = await asyncio.gather(
            self.vector_store.search(
                vector=vector, limit=pool, document_ids=config.document_ids, modalities=visual_modalities
            ),
            self.vector_store.search(
                vector=vector, limit=max(config.top_k, pool // 2), document_ids=config.document_ids
            ),
            self.bm25.search(
                query=query, limit=pool, document_ids=config.document_ids, modalities=visual_modalities
            ),
        )

        visual_chunks = await hydrate(
            context.session, [(h.chunk_id, h.score) for h in visual_hits], "visual_dense"
        )
        general_chunks = await hydrate(
            context.session, [(h.chunk_id, h.score) for h in general_hits], "context_dense"
        )
        sparse_chunks = await hydrate(
            context.session, [(h.chunk_id, h.score) for h in sparse_hits], "visual_sparse"
        )

        fused = reciprocal_rank_fusion(
            [
                ("visual_dense", visual_chunks),
                ("visual_sparse", sparse_chunks),
                ("context_dense", general_chunks),
            ],
            weights={"visual_dense": 1.0, "visual_sparse": 0.6, "context_dense": 0.45},
        )
        fused = deduplicate(fused)
        for chunk in fused:
            chunk.add_source(self.name.value, chunk.score)

        selected = fused[: config.top_k]
        images_loaded = await self._attach_images(selected)

        visual_count = sum(1 for c in selected if c.modality in VISUAL_MODALITIES)
        notes: list[str] = []
        if visual_count == 0:
            notes.append(
                "No figures, tables or images matched this query; only text evidence was found."
            )
        if images_loaded == 0 and visual_count:
            notes.append("Matching visuals were found but no image files were available to load.")

        result = RetrievalResult(
            chunks=selected,
            strategy=self.name.value,
            effective_query=query,
            retrieval_calls=3,
            notes=notes,
            diagnostics={
                "visual_chunks": visual_count,
                "images_loaded": images_loaded,
                "modalities_searched": visual_modalities,
                "modality_breakdown": _modality_counts(selected),
            },
        )
        result.rerank_positions()
        return result

    async def _attach_images(self, chunks: list[RetrievedChunk]) -> int:
        """Load image bytes for visual chunks so the generator can send them on."""
        targets = [c for c in chunks if c.asset_key and c.modality in {"figure", "image"}][:MAX_IMAGES]
        if not targets:
            return 0

        async def _load(chunk: RetrievedChunk) -> bool:
            try:
                data = await self.object_store.get(chunk.asset_key)
            except Exception as exc:
                log.warning("multimodal.asset_load_failed", key=chunk.asset_key, error=str(exc)[:140])
                return False
            if not data or len(data) > MAX_IMAGE_BYTES:
                return False
            suffix = (chunk.asset_key.rsplit(".", 1)[-1] or "png").lower()
            mime = "image/jpeg" if suffix in {"jpg", "jpeg"} else f"image/{suffix}"
            # Bytes ride on metadata; the generator pops them into ImageParts.
            chunk.metadata["image_bytes"] = data
            chunk.metadata["image_mime"] = mime
            return True

        results = await asyncio.gather(*(_load(c) for c in targets), return_exceptions=True)
        return sum(1 for r in results if r is True)


def _modality_counts(chunks: list[RetrievedChunk]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.modality] = counts.get(chunk.modality, 0) + 1
    return counts


__all__ = ["MultimodalRAG", "VISUAL_MODALITIES"]
