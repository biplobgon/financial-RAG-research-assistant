"""End-to-end RAG pipeline orchestration."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Optional, Any

from app.config.settings import settings
from app.config.constants import SPAN_RAG_PIPELINE, SPAN_RETRIEVAL, SPAN_LLM_GENERATION
from app.models.responses import QueryResponse, RetrievalResponse, SourceDocument, ResponseMetadata
from app.utils.logging import get_logger
from app.utils.cache import cache_get, cache_set, compute_cache_key
from app.utils.retry import retry_async

logger = get_logger(__name__)


class RAGPipeline:
    """
    Production RAG pipeline with:
    - Hybrid retrieval
    - Contextual compression
    - LLM generation
    - Evaluation and grounding
    - Redis caching
    - OpenTelemetry tracing
    """

    SYSTEM_PROMPT = """You are an expert financial analyst AI assistant.
    You provide accurate, well-reasoned financial analysis based strictly on the provided context documents.
    Always cite specific sections, figures, and dates from the documents.
    If the context does not contain sufficient information to answer confidently, clearly state this.
    Do not speculate or hallucinate financial data. Maintain professional financial analysis standards."""

    def __init__(
        self,
        retriever=None,
        llm=None,
        embedder=None,
        evaluator=None,
        reranker=None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.embedder = embedder
        self.evaluator = evaluator
        self.reranker = reranker

    async def query(
        self,
        query: str,
        filters: Optional[dict] = None,
        top_k: Optional[int] = None,
        collection_name: Optional[str] = None,
        use_cache: bool = True,
        session_id: Optional[str] = None,
    ) -> QueryResponse:
        """
        Execute a full RAG query pipeline.

        Flow:
        1. Check Redis cache
        2. Retrieve relevant documents (hybrid)
        3. Rerank results
        4. Build context prompt
        5. Generate LLM response
        6. Evaluate grounding
        7. Cache result
        8. Return response with metadata
        """
        trace_id = str(uuid.uuid4())
        start_time = time.monotonic()

        # Cache check
        cache_key = None
        if use_cache:
            cache_key = compute_cache_key("query", query=query, filters=filters, top_k=top_k)
            cached = await cache_get(cache_key)
            if cached:
                logger.info("Cache hit for query", extra={"trace_id": trace_id, "cache_hit": True})
                cached["metadata"]["cache_hit"] = True
                cached["metadata"]["trace_id"] = trace_id
                return QueryResponse(**cached)

        # Retrieval
        k = top_k or settings.rag_top_k
        documents = []
        if self.retriever:
            documents = await self.retriever.retrieve(
                query=query,
                top_k=k,
                filters=filters,
                collection_name=collection_name,
                retrieval_mode="hybrid",
            )

        # Reranking
        if self.reranker and documents:
            documents = await self.reranker.rerank(query, documents, top_k=settings.rag_rerank_top_k)

        # Build context
        context = self._build_context(documents)

        # LLM generation
        answer = ""
        tokens_prompt = 0
        tokens_completion = 0
        model_name = settings.llm_model

        if self.llm:
            try:
                messages = [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
                ]
                llm_result = await self.llm.generate(messages)
                answer = llm_result.get("content", "")
                tokens_prompt = llm_result.get("tokens_prompt", 0)
                tokens_completion = llm_result.get("tokens_completion", 0)
                model_name = llm_result.get("model", model_name)
            except Exception as e:
                logger.error(f"LLM generation failed: {e}", exc_info=True)
                answer = f"Unable to generate response: {str(e)}"
        else:
            # Mock response for testing
            answer = f"Based on the retrieved documents, here is analysis for: {query}"

        # Evaluation
        grounding_score = None
        hallucination_risk = None
        if self.evaluator and settings.evaluation_enabled:
            try:
                eval_result = await self.evaluator.score_grounding(
                    query=query,
                    response=answer,
                    context_docs=[d.excerpt or "" for d in documents],
                )
                grounding_score = eval_result.get("grounding_score")
                hallucination_risk = eval_result.get("hallucination_risk")
            except Exception as e:
                logger.warning(f"Evaluation failed (non-blocking): {e}")

        latency_ms = (time.monotonic() - start_time) * 1000
        total_tokens = tokens_prompt + tokens_completion

        metadata = ResponseMetadata(
            agent="rag_pipeline",
            latency_ms=round(latency_ms, 2),
            tokens_used=total_tokens,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            grounding_score=grounding_score,
            hallucination_risk=hallucination_risk,
            trace_id=trace_id,
            model=model_name,
            retrieval_count=len(documents),
            cache_hit=False,
        )

        response = QueryResponse(
            status="success",
            query=query,
            answer=answer,
            sources=documents[:5],  # Return top 5 sources
            metadata=metadata,
        )

        # Cache successful responses
        if use_cache and cache_key and grounding_score and grounding_score >= settings.hallucination_threshold:
            await cache_set(cache_key, response.model_dump(), ttl=settings.redis_ttl)

        logger.info(
            "RAG query complete",
            extra={
                "trace_id": trace_id,
                "latency_ms": round(latency_ms, 2),
                "tokens": total_tokens,
                "sources": len(documents),
                "grounding_score": grounding_score,
            },
        )

        return response

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict] = None,
        collection_name: Optional[str] = None,
        retrieval_mode: str = "hybrid",
        rerank: bool = True,
    ) -> RetrievalResponse:
        """Standalone retrieval without LLM generation."""
        trace_id = str(uuid.uuid4())
        start_time = time.monotonic()

        documents = []
        if self.retriever:
            documents = await self.retriever.retrieve(
                query=query,
                top_k=top_k or settings.rag_top_k,
                filters=filters,
                collection_name=collection_name,
                retrieval_mode=retrieval_mode,
            )

        if rerank and self.reranker and documents:
            documents = await self.reranker.rerank(query, documents)

        latency_ms = (time.monotonic() - start_time) * 1000

        return RetrievalResponse(
            status="success",
            query=query,
            documents=documents,
            total_found=len(documents),
            metadata=ResponseMetadata(
                latency_ms=round(latency_ms, 2),
                retrieval_count=len(documents),
                trace_id=trace_id,
            ),
        )

    def _build_context(self, documents: list[SourceDocument]) -> str:
        """Build context string from retrieved documents."""
        if not documents:
            return "No relevant documents found."

        parts = []
        for i, doc in enumerate(documents, 1):
            parts.append(
                f"[Document {i}] Source: {doc.source or 'Unknown'} | "
                f"Category: {doc.category or 'N/A'} | "
                f"Ticker: {doc.ticker or 'N/A'} | "
                f"Relevance: {doc.relevance_score:.3f}\n"
                f"{doc.excerpt or ''}"
            )
        return "\n\n---\n\n".join(parts)
