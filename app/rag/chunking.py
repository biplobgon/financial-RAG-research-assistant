"""Financial document chunking strategies."""
from __future__ import annotations

import logging
import re
import uuid
from typing import Optional

from app.models.documents import Document, DocumentChunk, DocumentMetadata
from app.config.settings import settings
from app.utils.text_processing import count_tokens_approx, split_into_sentences

logger = logging.getLogger(__name__)


class FinancialChunker:
    """
    Intelligent chunking for financial documents.

    Strategies:
    - recursive: Hierarchical splitting by paragraphs, sentences
    - semantic: Section-aware chunking (preserves SEC filing sections)
    - fixed: Fixed-size with overlap
    """

    # SEC filing section headers to use as split boundaries
    SEC_SECTION_HEADERS = [
        r"ITEM\s+\d+[A-Z]?\.",
        r"PART\s+[IVX]+",
        r"MANAGEMENT.S DISCUSSION",
        r"RISK FACTORS",
        r"FINANCIAL STATEMENTS",
        r"NOTES TO FINANCIAL STATEMENTS",
        r"QUANTITATIVE AND QUALITATIVE DISCLOSURES",
        r"CONTROLS AND PROCEDURES",
        r"LEGAL PROCEEDINGS",
    ]

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        strategy: str = "recursive",
    ):
        self.chunk_size = chunk_size or settings.rag_chunk_size
        self.chunk_overlap = chunk_overlap or settings.rag_chunk_overlap
        self.strategy = strategy

    def chunk_document(self, document: Document) -> list[DocumentChunk]:
        """Chunk a document using the configured strategy."""
        if self.strategy == "semantic":
            return self._semantic_chunk(document)
        elif self.strategy == "fixed":
            return self._fixed_chunk(document)
        else:  # recursive (default)
            return self._recursive_chunk(document)

    def _recursive_chunk(self, document: Document) -> list[DocumentChunk]:
        """Recursive character text splitting with overlap."""
        text = document.content
        separators = ["\n\n", "\n", ". ", " ", ""]
        chunks_text = self._recursive_split(text, separators, self.chunk_size)
        return self._create_chunks(document, chunks_text)

    def _semantic_chunk(self, document: Document) -> list[DocumentChunk]:
        """Section-aware chunking for SEC filings and financial documents."""
        text = document.content
        sections = self._split_by_sections(text)

        all_chunks_text = []
        for section_text in sections:
            if count_tokens_approx(section_text) <= self.chunk_size:
                all_chunks_text.append(section_text)
            else:
                # Recursively split large sections
                sub_chunks = self._recursive_split(section_text, ["\n\n", "\n", ". "], self.chunk_size)
                all_chunks_text.extend(sub_chunks)

        return self._create_chunks(document, all_chunks_text)

    def _fixed_chunk(self, document: Document) -> list[DocumentChunk]:
        """Fixed-size token chunking with overlap."""
        text = document.content
        words = text.split()
        chunk_words = self.chunk_size * 4  # Approx words per chunk
        overlap_words = self.chunk_overlap * 4

        chunks_text = []
        start = 0
        while start < len(words):
            end = min(start + chunk_words, len(words))
            chunk = " ".join(words[start:end])
            chunks_text.append(chunk)
            start += chunk_words - overlap_words
            if start >= len(words):
                break

        return self._create_chunks(document, chunks_text)

    def _recursive_split(self, text: str, separators: list[str], max_tokens: int) -> list[str]:
        """Recursively split text using hierarchical separators."""
        if count_tokens_approx(text) <= max_tokens:
            return [text]

        separator = separators[0] if separators else ""
        remaining_separators = separators[1:]

        if not separator:
            # Character-level split as last resort
            char_limit = max_tokens * 4
            return [text[i:i + char_limit] for i in range(0, len(text), char_limit)]

        parts = text.split(separator)
        chunks = []
        current = ""

        for part in parts:
            candidate = current + (separator if current else "") + part
            if count_tokens_approx(candidate) <= max_tokens:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if count_tokens_approx(part) > max_tokens and remaining_separators:
                    chunks.extend(self._recursive_split(part, remaining_separators, max_tokens))
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)

        return [c for c in chunks if c.strip()]

    def _split_by_sections(self, text: str) -> list[str]:
        """Split SEC filing text by section headers."""
        combined_pattern = "|".join(self.SEC_SECTION_HEADERS)
        parts = re.split(f"(?im)({combined_pattern})", text)

        sections = []
        current = ""
        for part in parts:
            if re.match(f"(?im){combined_pattern}", part):
                if current.strip():
                    sections.append(current.strip())
                current = part
            else:
                current += "\n" + part

        if current.strip():
            sections.append(current.strip())

        return sections if sections else [text]

    def _create_chunks(self, document: Document, chunks_text: list[str]) -> list[DocumentChunk]:
        """Create DocumentChunk objects from text fragments."""
        chunks = []
        char_offset = 0

        for i, chunk_text in enumerate(chunks_text):
            if not chunk_text.strip():
                continue

            chunk_id = f"{document.doc_id}_chunk_{i:04d}"
            start_char = document.content.find(chunk_text[:50], char_offset)
            if start_char == -1:
                start_char = char_offset
            end_char = start_char + len(chunk_text)

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                content=chunk_text,
                chunk_index=i,
                token_count=count_tokens_approx(chunk_text),
                metadata=document.metadata,
                start_char=start_char,
                end_char=end_char,
            )
            chunks.append(chunk)
            char_offset = end_char

        logger.debug(
            f"Created {len(chunks)} chunks from document {document.doc_id}",
            extra={"doc_id": document.doc_id, "chunks": len(chunks), "strategy": self.strategy},
        )
        return chunks
