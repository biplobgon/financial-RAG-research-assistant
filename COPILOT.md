# COPILOT.md — Engineering Memory for GitHub Copilot

## Project: Financial RAG Research Assistant

### Codebase Purpose
Enterprise RAG platform for financial document intelligence, investment research, and SEC filing analysis.

### Key Modules to Reference

#### `app/rag/`
- `ingestion.py` — async document ingestion pipeline
- `chunking.py` — financial document chunking strategies
- `retriever.py` — hybrid semantic search with ChromaDB
- `pipeline.py` — end-to-end RAG pipeline

#### `app/agents/`
- `base_agent.py` — abstract base with retry, logging, evaluation hooks
- `sec_filing_agent.py`
- `earnings_agent.py`
- `portfolio_agent.py`
- `executive_summary_agent.py`
- `research_agent.py`
- `evaluation_agent.py`
- `orchestrator.py` — multi-agent router

#### `app/api/v1/`
- `routes/query.py` — /query endpoint
- `routes/retrieve.py` — /retrieve endpoint
- `routes/analyze.py` — /analyze endpoint
- `routes/summarize.py` — /summarize endpoint
- `routes/portfolio.py` — /portfolio endpoint
- `routes/evaluate.py` — /evaluate endpoint
- `routes/health.py` — /health endpoint

### Common Code Patterns

#### Dependency Injection
```python
async def get_rag_pipeline() -> RAGPipeline:
    return RAGPipeline(
        retriever=ChromaRetriever(),
        llm=VertexAILLM(),
        evaluator=GroundingEvaluator()
    )
```

#### Pydantic Schema
```python
class QueryRequest(BaseModel):
    query: str
    filters: dict = {}
    top_k: int = 10
    agent: str = "research_agent"
```

#### Logging
```python
logger.info("RAG query processed", extra={
    "query": query,
    "latency_ms": latency,
    "tokens": token_count,
    "grounding_score": score
})
```

### Test Fixtures Location
- `tests/fixtures/` — sample SEC filings, earnings transcripts
- `tests/unit/` — unit tests per module
- `tests/integration/` — full pipeline integration tests

### Do Not Generate
- Synchronous DB calls
- Hardcoded credentials
- Global mutable state
- Missing error handling
