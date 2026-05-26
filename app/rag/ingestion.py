"""Async document ingestion pipeline for financial documents."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from pathlib import Path
from typing import Optional, AsyncIterator
from datetime import datetime

from app.config.settings import settings
from app.models.documents import Document, DocumentMetadata
from app.utils.text_processing import clean_financial_text, count_tokens_approx
from app.utils.logging import get_logger

logger = get_logger(__name__)


class DocumentIngestionPipeline:
    """
    Async pipeline for ingesting financial documents from multiple sources.

    Supports:
    - PDF files (SEC filings, research reports)
    - Text files (earnings transcripts)
    - JSON structured documents
    - SEC EDGAR API integration
    """

    def __init__(
        self,
        chunker=None,
        embedder=None,
        vector_store=None,
        batch_size: int = 50,
    ):
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.batch_size = batch_size
        self._ingested_count = 0
        self._failed_count = 0

    async def ingest_file(self, file_path: str | Path, metadata: Optional[dict] = None) -> Optional[Document]:
        """Ingest a single file and return a Document."""
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        try:
            content = await self._read_file(path)
            if not content:
                logger.warning(f"Empty file: {file_path}")
                return None

            content = clean_financial_text(content)
            doc_id = self._generate_doc_id(str(path), content)

            doc_metadata = DocumentMetadata(
                source=str(path),
                category=self._infer_category(path),
                **(metadata or {}),
            )

            doc = Document(
                doc_id=doc_id,
                title=path.stem.replace("_", " ").title(),
                content=content,
                metadata=doc_metadata,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

            logger.info(
                "Document ingested",
                extra={
                    "doc_id": doc_id,
                    "file": path.name,
                    "tokens": count_tokens_approx(content),
                    "category": doc_metadata.category,
                },
            )
            self._ingested_count += 1
            return doc

        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}", exc_info=True)
            self._failed_count += 1
            return None

    async def ingest_directory(
        self,
        directory: str | Path,
        glob_pattern: str = "**/*.txt",
        metadata: Optional[dict] = None,
    ) -> AsyncIterator[Document]:
        """Async generator that ingests all matching files in a directory."""
        directory = Path(directory)
        files = list(directory.glob(glob_pattern))
        logger.info(f"Found {len(files)} files in {directory} matching {glob_pattern}")

        semaphore = asyncio.Semaphore(10)  # Limit concurrent file reads

        async def _ingest_with_semaphore(f: Path) -> Optional[Document]:
            async with semaphore:
                return await self.ingest_file(f, metadata)

        tasks = [_ingest_with_semaphore(f) for f in files]
        for coro in asyncio.as_completed(tasks):
            doc = await coro
            if doc:
                yield doc

    async def ingest_sec_filing_url(
        self,
        cik: str,
        accession_number: str,
        filing_type: str = "10-K",
    ) -> Optional[Document]:
        """Fetch and ingest an SEC filing from EDGAR."""
        try:
            import httpx

            url = f"https://www.sec.gov/Archives/edgar/full-index/{accession_number.replace('-', '')}/filing-summary.json"
            headers = {"User-Agent": "Financial RAG Research Assistant contact@example.com"}

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()

            metadata = {
                "cik": cik,
                "accession_number": accession_number,
                "filing_type": filing_type,
                "url": url,
                "category": "sec_filing",
            }
            logger.info(f"Fetched SEC filing {accession_number} for CIK {cik}")

            # In production, parse the full filing text
            content = f"SEC Filing {filing_type} for CIK {cik}, Accession: {accession_number}"
            return await self._create_document_from_text(content, metadata)

        except Exception as e:
            logger.error(f"Failed to fetch SEC filing {accession_number}: {e}")
            return None

    async def ingest_batch(self, documents: list[Document]) -> dict[str, int]:
        """
        Chunk, embed, and index a batch of documents.
        Returns stats: {ingested, failed, chunks_created}.
        """
        if not documents:
            return {"ingested": 0, "failed": 0, "chunks_created": 0}

        stats = {"ingested": 0, "failed": 0, "chunks_created": 0}
        all_chunks = []

        for doc in documents:
            try:
                if self.chunker:
                    chunks = self.chunker.chunk_document(doc)
                    all_chunks.extend(chunks)
                    stats["chunks_created"] += len(chunks)
                stats["ingested"] += 1
            except Exception as e:
                logger.error(f"Failed to chunk document {doc.doc_id}: {e}")
                stats["failed"] += 1

        # Embed and index in batches
        if all_chunks and self.embedder and self.vector_store:
            try:
                for i in range(0, len(all_chunks), self.batch_size):
                    batch = all_chunks[i:i + self.batch_size]
                    texts = [c.content for c in batch]
                    embeddings = await self.embedder.embed_batch(texts)
                    for chunk, embedding in zip(batch, embeddings):
                        chunk.embedding = embedding
                    await self.vector_store.add_chunks(batch)
                    logger.info(f"Indexed batch {i//self.batch_size + 1}: {len(batch)} chunks")
            except Exception as e:
                logger.error(f"Failed to embed/index chunks: {e}")
                stats["failed"] += len(all_chunks)

        return stats

    async def _read_file(self, path: Path) -> str:
        """Read file content asynchronously."""
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return await self._read_pdf(path)
        elif suffix in (".txt", ".md", ".rst"):
            return await asyncio.to_thread(path.read_text, encoding="utf-8", errors="ignore")
        elif suffix == ".json":
            import json
            content = await asyncio.to_thread(path.read_text, encoding="utf-8")
            data = json.loads(content)
            # Extract text field from JSON documents
            if isinstance(data, dict):
                return data.get("text", data.get("content", str(data)))
            return str(data)
        else:
            return await asyncio.to_thread(path.read_text, encoding="utf-8", errors="ignore")

    async def _read_pdf(self, path: Path) -> str:
        """Extract text from PDF using PyMuPDF or pdfplumber."""
        try:
            import fitz  # PyMuPDF

            def _extract():
                doc = fitz.open(str(path))
                text = "\n\n".join(page.get_text() for page in doc)
                doc.close()
                return text

            return await asyncio.to_thread(_extract)
        except ImportError:
            pass

        try:
            import pdfplumber

            def _extract_plumber():
                with pdfplumber.open(str(path)) as pdf:
                    return "\n\n".join(
                        page.extract_text() or "" for page in pdf.pages
                    )

            return await asyncio.to_thread(_extract_plumber)
        except ImportError:
            logger.warning("No PDF library available. Install PyMuPDF or pdfplumber.")
            return ""

    def _generate_doc_id(self, source: str, content: str) -> str:
        """Generate deterministic document ID from source path and content hash."""
        content_hash = hashlib.sha256(content[:1000].encode()).hexdigest()[:8]
        source_hash = hashlib.md5(source.encode()).hexdigest()[:8]
        return f"doc_{source_hash}_{content_hash}"

    def _infer_category(self, path: Path) -> str:
        """Infer document category from file path."""
        path_str = str(path).lower()
        if "sec" in path_str or "filing" in path_str or "10-k" in path_str or "10-q" in path_str:
            return "sec_filing"
        elif "earnings" in path_str or "transcript" in path_str:
            return "earnings_transcript"
        elif "research" in path_str or "report" in path_str or "analyst" in path_str:
            return "research_report"
        elif "market" in path_str or "price" in path_str:
            return "market_data"
        return "general"

    async def _create_document_from_text(self, content: str, metadata: dict) -> Document:
        """Create a Document object from text content and metadata dict."""
        doc_id = self._generate_doc_id(metadata.get("url", ""), content)
        doc_metadata = DocumentMetadata(**{k: v for k, v in metadata.items()
                                           if k in DocumentMetadata.model_fields})
        return Document(
            doc_id=doc_id,
            title=f"{metadata.get('filing_type', 'Document')} — {metadata.get('cik', 'Unknown')}",
            content=content,
            metadata=doc_metadata,
        )

    @property
    def stats(self) -> dict:
        return {"ingested": self._ingested_count, "failed": self._failed_count}
