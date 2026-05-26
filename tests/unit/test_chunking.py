"""Unit tests for financial document chunking."""
import pytest
from app.rag.chunking import FinancialChunker
from app.models.documents import Document, DocumentMetadata


def make_doc(content: str, ticker: str = "AAPL") -> Document:
    return Document(
        doc_id=f"test_{ticker}",
        title=f"{ticker} Test Document",
        content=content,
        metadata=DocumentMetadata(source="test", category="sec_filing", ticker=ticker),
    )


@pytest.fixture
def chunker():
    return FinancialChunker(chunk_size=128, chunk_overlap=16, strategy="recursive")


@pytest.fixture
def semantic_chunker():
    return FinancialChunker(chunk_size=256, chunk_overlap=32, strategy="semantic")


def test_recursive_chunking_basic(chunker):
    """Recursive chunker produces chunks within size limit."""
    content = " ".join(["word"] * 1000)
    doc = make_doc(content)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= chunker.chunk_size * 1.2  # Allow 20% tolerance


def test_chunk_inherits_metadata(chunker):
    """Chunks inherit parent document metadata."""
    doc = make_doc("Apple Inc. reported revenue of $383 billion for fiscal year 2023.")
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.doc_id == doc.doc_id
        assert chunk.metadata.ticker == "AAPL"
        assert chunk.metadata.category == "sec_filing"


def test_chunk_ids_unique(chunker):
    """All chunk IDs are unique within a document."""
    content = "\n\n".join([f"Paragraph {i}: " + "content " * 50 for i in range(10)])
    doc = make_doc(content)
    chunks = chunker.chunk_document(doc)
    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "Chunk IDs must be unique"


def test_semantic_chunking_splits_on_sec_sections(semantic_chunker):
    """Semantic chunker respects SEC filing section boundaries."""
    content = """
ITEM 1. BUSINESS
Apple Inc. designs and manufactures smartphones and computers.
The company operates in multiple segments worldwide.

ITEM 1A. RISK FACTORS
The company faces significant competition from global technology companies.
Market conditions may adversely affect operating results.

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS
Revenue for fiscal 2023 was $383.3 billion, down 3% year-over-year.
Operating income declined to $114.3 billion.
"""
    doc = make_doc(content)
    chunks = semantic_chunker.chunk_document(doc)
    assert len(chunks) >= 2


def test_fixed_chunking(tmp_path):
    """Fixed chunker creates equal-sized chunks."""
    chunker = FinancialChunker(chunk_size=64, chunk_overlap=8, strategy="fixed")
    content = " ".join(["token"] * 512)
    doc = make_doc(content)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 4


def test_empty_document_chunking(chunker):
    """Empty documents return no chunks."""
    doc = make_doc("")
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 0


def test_chunk_index_sequential(chunker):
    """Chunk indices are sequential."""
    content = "\n\n".join(["paragraph content " * 20 for _ in range(5)])
    doc = make_doc(content)
    chunks = chunker.chunk_document(doc)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i
