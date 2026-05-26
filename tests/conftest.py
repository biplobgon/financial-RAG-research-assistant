"""Pytest configuration and shared fixtures."""
from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from app.config.settings import settings


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings():
    """Override settings for testing."""
    with patch.dict("os.environ", {
        "ENVIRONMENT": "test",
        "DEBUG": "true",
        "LLM_PROVIDER": "mock",
        "CHROMA_HOST": "localhost",
        "REDIS_URL": "redis://localhost:6379/1",
        "EVALUATION_ENABLED": "false",
    }):
        yield settings


@pytest.fixture
def mock_llm():
    """Mock LLM client that returns predictable responses."""
    mock = AsyncMock()
    mock.generate = AsyncMock(return_value={
        "content": "This is a mock financial analysis response based on SEC filing data.",
        "tokens_prompt": 200,
        "tokens_completion": 100,
        "model": "mock-gpt-4",
    })
    return mock


@pytest.fixture
def mock_retriever():
    """Mock ChromaDB retriever."""
    from app.models.responses import SourceDocument
    mock = AsyncMock()
    mock.retrieve = AsyncMock(return_value=[
        SourceDocument(
            doc_id="doc_test_001",
            title="Apple Inc. 10-K Annual Report",
            source="data/raw/sec_filings/AAPL_10K_2023.txt",
            relevance_score=0.92,
            category="sec_filing",
            ticker="AAPL",
            filing_type="10-K",
            date="2023-10-27",
            excerpt="Apple Inc. reported total net sales of $383.3 billion for fiscal year 2023...",
        ),
        SourceDocument(
            doc_id="doc_test_002",
            title="Microsoft Corp. 10-K Annual Report",
            source="data/raw/sec_filings/MSFT_10K_2023.txt",
            relevance_score=0.88,
            category="sec_filing",
            ticker="MSFT",
            filing_type="10-K",
            date="2023-07-27",
            excerpt="Microsoft reported revenue of $211.9 billion for fiscal year 2023...",
        ),
    ])
    return mock


@pytest.fixture
def mock_evaluator():
    """Mock grounding evaluator."""
    mock = AsyncMock()
    mock.score_grounding = AsyncMock(return_value={
        "grounding_score": 0.92,
        "hallucination_risk": "low",
        "passed": True,
    })
    return mock


@pytest.fixture
def sample_sec_filing_text() -> str:
    """Sample SEC filing text for testing."""
    return """
UNITED STATES SECURITIES AND EXCHANGE COMMISSION
Washington, D.C. 20549

FORM 10-K
ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934

For the fiscal year ended September 30, 2023

Commission file number: 0-10030

APPLE INC.
(Exact name of registrant as specified in its charter)

California                                               94-2404110
(State or other jurisdiction of                       (I.R.S. Employer
incorporation or organization)                         Identification No.)

One Apple Park Way
Cupertino, California                                              95014
(Address of principal executive offices)                          (Zip Code)

ITEM 1. BUSINESS
Apple Inc. designs, manufactures and markets smartphones, personal computers, tablets, 
wearables and accessories, and sells a variety of related services.

ITEM 1A. RISK FACTORS
The Company's operations and financial results are subject to various risks and uncertainties,
including those described below, which could adversely affect its business, financial condition,
results of operations, cash flows, and the market price of the Company's stock.

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS
Fiscal Year 2023 Highlights
- Total net sales: $383.3 billion, down 3% year-over-year
- Net income: $97.0 billion  
- Earnings per share (diluted): $6.13
- Operating cash flow: $114.1 billion
- Services revenue: $85.2 billion, up 9% year-over-year
"""


@pytest.fixture
def sample_earnings_transcript() -> str:
    """Sample earnings call transcript for testing."""
    return """
APPLE INC. Q4 FY2023 EARNINGS CALL TRANSCRIPT
October 26, 2023

Participants:
- Tim Cook, CEO
- Luca Maestri, CFO
- Tejas Gala, VP Investor Relations

TIM COOK: Good afternoon, everyone. Thank you for joining us. 
We're pleased to report our fourth fiscal quarter results.

Revenue for the quarter came in at $89.5 billion, down 1% year-over-year.
We generated $21.4 billion in net income and diluted earnings per share of $1.46.

Services revenue reached an all-time high of $22.3 billion, up 16% year-over-year.
iPhone revenue was $43.8 billion for the quarter.

LUCA MAESTRI: Let me provide more details on our financial results.
For Q4, our gross margin was 45.2%, up from 42.3% in the year-ago quarter.
We returned over $25 billion to shareholders during the quarter.

Q&A SECTION:
ANALYST: Tim, can you comment on China demand trends?
TIM COOK: We're pleased with our performance in China and see continued strong demand...
"""


@pytest.fixture
def app_client():
    """FastAPI test client."""
    from main import app
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def async_app_client():
    """Async FastAPI test client."""
    from main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
