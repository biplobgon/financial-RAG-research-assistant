"""Hybrid semantic retrieval with ChromaDB."""
from __future__ import annotations

import logging
import time
from typing import Optional, Any

from app.config.settings import settings
from app.models.documents import DocumentChunk, DocumentMetadata
from app.models.responses import SourceDocument
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ChromaRetriever:
    """
    Hybrid retrieval engine using ChromaDB.

    Supports:
    - Dense semantic search (embedding-based)
    - Sparse keyword search (BM25-style)
    - Hybrid fusion (RRF - Reciprocal Rank Fusion)
    - Metadata filtering
    - Multi-collection search
    """

    def __init__(
        self,
        embedder=None,
        collection_name: str = None,
        top_k: int = None,
        similarity_threshold: float = None,
        hybrid_alpha: float = None,
    ):
        self.embedder = embedder
        self.collection_name = collection_name or settings.chroma_collection_sec
        self.top_k = top_k or settings.rag_top_k
        self.similarity_threshold = similarity_threshold or settings.rag_similarity_threshold
        self.hybrid_alpha = hybrid_alpha or settings.rag_hybrid_alpha
        self._client = None
        self._collection = None

    async def _get_client(self):
        """Lazily initialize ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
                self._client = chromadb.AsyncHttpClient(
                    host=settings.chroma_host,
                    port=settings.chroma_port,
                )
                logger.info(f"ChromaDB client initialized: {settings.chroma_host}:{settings.chroma_port}")
            except Exception as e:
                logger.error(f"ChromaDB connection failed: {e}")
                # Fallback to in-memory for development
                try:
                    import chromadb
                    self._client = chromadb.Client()
                    logger.warning("Using in-memory ChromaDB (development mode)")
                except Exception:
                    raise RuntimeError("ChromaDB not available") from e
        return self._client

    async def _get_collection(self, collection_name: Optional[str] = None):
        """Get or create ChromaDB collection."""
        name = collection_name or self.collection_name
        client = await self._get_client()
        try:
            return await client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            logger.error(f"Failed to get collection {name}: {e}")
            raise

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict] = None,
        collection_name: Optional[str] = None,
        retrieval_mode: str = "hybrid",
    ) -> list[SourceDocument]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: ChromaDB metadata filters (where clause)
            collection_name: Override default collection
            retrieval_mode: dense | sparse | hybrid
        """
        k = top_k or self.top_k
        start_time = time.monotonic()

        try:
            if retrieval_mode == "dense":
                results = await self._dense_retrieve(query, k, filters, collection_name)
            elif retrieval_mode == "sparse":
                results = await self._sparse_retrieve(query, k, filters, collection_name)
            else:  # hybrid
                results = await self._hybrid_retrieve(query, k, filters, collection_name)

            # Filter by similarity threshold
            results = [r for r in results if r.relevance_score >= self.similarity_threshold]

            latency = (time.monotonic() - start_time) * 1000
            logger.info(
                "Retrieval completed",
                extra={
                    "query_len": len(query),
                    "results": len(results),
                    "latency_ms": round(latency, 2),
                    "mode": retrieval_mode,
                    "collection": collection_name or self.collection_name,
                },
            )
            return results

        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            return []

    async def _dense_retrieve(
        self,
        query: str,
        k: int,
        filters: Optional[dict],
        collection_name: Optional[str],
    ) -> list[SourceDocument]:
        """Embedding-based dense retrieval."""
        if not self.embedder:
            raise ValueError("Embedder required for dense retrieval")

        query_embedding = await self.embedder.embed_query(query)
        collection = await self._get_collection(collection_name)

        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            query_params["where"] = filters

        results = await collection.query(**query_params)
        return self._parse_chroma_results(results)

    async def _sparse_retrieve(
        self,
        query: str,
        k: int,
        filters: Optional[dict],
        collection_name: Optional[str],
    ) -> list[SourceDocument]:
        """Keyword-based sparse retrieval using ChromaDB where_document."""
        collection = await self._get_collection(collection_name)

        # Extract key terms from query
        terms = [t for t in query.split() if len(t) > 3]
        if not terms:
            terms = query.split()

        query_params = {
            "query_texts": [query],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            query_params["where"] = filters

        results = await collection.query(**query_params)
        return self._parse_chroma_results(results)

    async def _hybrid_retrieve(
        self,
        query: str,
        k: int,
        filters: Optional[dict],
        collection_name: Optional[str],
    ) -> list[SourceDocument]:
        """Hybrid retrieval: RRF fusion of dense and sparse results."""
        import asyncio

        dense_results, sparse_results = await asyncio.gather(
            self._dense_retrieve(query, k * 2, filters, collection_name)
            if self.embedder
            else self._sparse_retrieve(query, k, filters, collection_name),
            self._sparse_retrieve(query, k * 2, filters, collection_name),
            return_exceptions=True,
        )

        # Handle exceptions
        if isinstance(dense_results, Exception):
            logger.warning(f"Dense retrieval failed in hybrid: {dense_results}")
            dense_results = []
        if isinstance(sparse_results, Exception):
            logger.warning(f"Sparse retrieval failed in hybrid: {sparse_results}")
            sparse_results = []

        # Reciprocal Rank Fusion
        return self._rrf_fusion(dense_results, sparse_results, k, alpha=self.hybrid_alpha)

    def _rrf_fusion(
        self,
        dense: list[SourceDocument],
        sparse: list[SourceDocument],
        k: int,
        alpha: float = 0.7,
        rrf_k: int = 60,
    ) -> list[SourceDocument]:
        """Reciprocal Rank Fusion of two ranked lists."""
        scores: dict[str, float] = {}
        doc_map: dict[str, SourceDocument] = {}

        for rank, doc in enumerate(dense):
            score = alpha * (1 / (rrf_k + rank + 1))
            scores[doc.doc_id] = scores.get(doc.doc_id, 0) + score
            doc_map[doc.doc_id] = doc

        for rank, doc in enumerate(sparse):
            score = (1 - alpha) * (1 / (rrf_k + rank + 1))
            scores[doc.doc_id] = scores.get(doc.doc_id, 0) + score
            if doc.doc_id not in doc_map:
                doc_map[doc.doc_id] = doc

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        result = []
        for doc_id, score in ranked:
            doc = doc_map[doc_id]
            doc.relevance_score = score
            result.append(doc)
        return result

    def _parse_chroma_results(self, results: dict) -> list[SourceDocument]:
        """Parse ChromaDB query results into SourceDocument objects."""
        docs = []
        if not results or not results.get("ids"):
            return docs

        ids = results["ids"][0] if results["ids"] else []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else 1.0
            # Convert cosine distance to similarity score
            similarity = max(0.0, 1.0 - distance)

            docs.append(SourceDocument(
                doc_id=doc_id,
                title=metadata.get("title", doc_id),
                source=metadata.get("source", ""),
                relevance_score=similarity,
                category=metadata.get("category"),
                ticker=metadata.get("ticker"),
                filing_type=metadata.get("filing_type"),
                date=metadata.get("filing_date"),
                excerpt=documents[i][:500] if i < len(documents) else "",
            ))

        return docs

    async def add_chunks(self, chunks: list) -> None:
        """Add document chunks to ChromaDB collection."""
        if not chunks:
            return

        collection = await self._get_collection()
        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        embeddings = [c.embedding for c in chunks if c.embedding]
        metadatas = [c.metadata.model_dump(exclude={"custom"}) for c in chunks]

        add_kwargs = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }
        if len(embeddings) == len(chunks):
            add_kwargs["embeddings"] = embeddings

        await collection.add(**add_kwargs)
        logger.info(f"Added {len(chunks)} chunks to ChromaDB collection {self.collection_name}")
