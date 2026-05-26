"""Cross-encoder reranking for improved retrieval precision."""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from app.models.responses import SourceDocument
from app.config.settings import settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranker using cross-encoder models for improved precision.

    Falls back to BM25-style scoring if cross-encoder model unavailable.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_k: int = None):
        self.model_name = model_name
        self.top_k = top_k or settings.rag_rerank_top_k
        self._model = None

    async def rerank(
        self,
        query: str,
        documents: list[SourceDocument],
        top_k: Optional[int] = None,
    ) -> list[SourceDocument]:
        """Rerank documents by cross-encoder relevance score."""
        if not documents:
            return []

        k = top_k or self.top_k

        try:
            model = await self._get_model()
            if model:
                return await self._cross_encoder_rerank(query, documents, k, model)
            else:
                return await self._keyword_rerank(query, documents, k)
        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")
            return documents[:k]

    async def _cross_encoder_rerank(
        self,
        query: str,
        documents: list[SourceDocument],
        k: int,
        model,
    ) -> list[SourceDocument]:
        """Rerank using cross-encoder model."""
        pairs = [(query, doc.excerpt or "") for doc in documents]

        def _score():
            return model.predict(pairs)

        scores = await asyncio.to_thread(_score)

        for doc, score in zip(documents, scores):
            doc.relevance_score = float(score)

        documents.sort(key=lambda d: d.relevance_score, reverse=True)
        return documents[:k]

    async def _keyword_rerank(
        self,
        query: str,
        documents: list[SourceDocument],
        k: int,
    ) -> list[SourceDocument]:
        """Simple keyword overlap reranking fallback."""
        query_terms = set(query.lower().split())

        def _overlap_score(doc: SourceDocument) -> float:
            text = (doc.excerpt or "").lower()
            doc_terms = set(text.split())
            overlap = len(query_terms & doc_terms)
            return (overlap / max(len(query_terms), 1)) * 0.5 + doc.relevance_score * 0.5

        scored = [(doc, _overlap_score(doc)) for doc in documents]
        scored.sort(key=lambda x: x[1], reverse=True)

        result = []
        for doc, score in scored[:k]:
            doc.relevance_score = score
            result.append(doc)
        return result

    async def _get_model(self):
        """Lazily load cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = await asyncio.to_thread(CrossEncoder, self.model_name)
                logger.info(f"Cross-encoder model loaded: {self.model_name}")
            except ImportError:
                logger.warning("sentence-transformers not installed; using keyword reranking fallback")
                self._model = False  # Mark as unavailable
        return self._model if self._model is not False else None
