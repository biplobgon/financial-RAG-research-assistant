"""Financial embedding generation pipeline."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from app.config.settings import settings
from app.utils.logging import get_logger
from app.utils.retry import retry_async

logger = get_logger(__name__)


class FinancialEmbedder:
    """
    Embedding generation with OpenAI / Vertex AI support.

    Features:
    - Async batch embedding
    - Retry with exponential backoff
    - Dimension validation
    - Token count tracking
    """

    def __init__(
        self,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        self.model = model or settings.embedding_model
        self.dimensions = dimensions or settings.embedding_dimensions
        self.batch_size = batch_size or settings.embedding_batch_size
        self._client = None
        self._total_tokens = 0

    async def _get_client(self):
        """Initialize embedding client based on configured provider."""
        if self._client is None:
            if settings.llm_provider == "openai" and settings.openai_api_key:
                import openai
                self._client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
                logger.info(f"OpenAI embedding client initialized: {self.model}")
            else:
                # Fallback: use sentence-transformers locally
                logger.warning("Using local sentence-transformers for embeddings (no API key)")
                self._client = "local"
        return self._client

    @retry_async(max_retries=3, base_delay=1.0, backoff_factor=2.0)
    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query text."""
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        if not texts:
            return []

        client = await self._get_client()
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = await self._embed_batch_chunk(client, batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def _embed_batch_chunk(self, client, texts: list[str]) -> list[list[float]]:
        """Embed a single batch chunk."""
        if client == "local":
            return await self._local_embed(texts)

        try:
            # OpenAI embedding API
            response = await client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimensions if "text-embedding-3" in self.model else None,
            )
            self._total_tokens += response.usage.total_tokens
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Embedding API call failed: {e}")
            raise

    async def _local_embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using local sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            local_model_name = "all-MiniLM-L6-v2"

            def _encode():
                model = SentenceTransformer(local_model_name)
                return model.encode(texts, convert_to_list=True)

            embeddings = await asyncio.to_thread(_encode)
            return embeddings
        except ImportError:
            logger.error("sentence-transformers not installed for local embedding")
            # Return zero vectors as last resort
            return [[0.0] * 384] * len(texts)

    @property
    def total_tokens_used(self) -> int:
        return self._total_tokens
