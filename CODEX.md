# CODEX.md — Engineering Memory for GitHub Copilot and Codex

## Project: Financial RAG Research Assistant

### Project Type
Enterprise Financial AI Platform — Production RAG System

### Language and Framework
- Python 3.11+
- FastAPI 0.111+
- LangChain 0.2+
- LlamaIndex 0.10+
- ChromaDB 0.5+
- Pydantic v2

### File Naming
- Modules: `snake_case.py`
- Tests: `test_<module>.py`
- Config: `settings.py`, `constants.py`

### Import Ordering
1. Standard library
2. Third-party
3. Internal modules

### Async Patterns
```python
# Correct pattern for async RAG calls
async def retrieve_documents(query: str) -> list[Document]:
    async with AsyncChromaClient() as client:
        results = await client.query(query_texts=[query], n_results=10)
    return results
```

### Error Handling Pattern
```python
try:
    result = await llm_call(prompt)
except LLMProviderError as e:
    logger.error("LLM call failed", extra={"error": str(e)})
    result = await fallback_llm_call(prompt)
finally:
    span.end()
```

### Agent Pattern
```python
class BaseFinancialAgent:
    def __init__(self, llm, retriever, evaluator):
        self.llm = llm
        self.retriever = retriever
        self.evaluator = evaluator

    async def run(self, query: str) -> AgentResponse:
        context = await self.retriever.retrieve(query)
        response = await self.llm.generate(query, context)
        score = await self.evaluator.score(response, context)
        return AgentResponse(content=response, score=score)
```

### Evaluation Pipeline
1. Retrieve context documents
2. Generate LLM response
3. Score factual grounding (0.0 - 1.0)
4. Check hallucination threshold (>0.85 = pass)
5. Log to MLflow
6. Emit OTel trace

### ChromaDB Collection Names
- `sec_filings` — SEC 10-K, 10-Q documents
- `earnings_transcripts` — quarterly earnings calls
- `market_data` — financial indicators
- `research_reports` — analyst reports

### API Response Format
```json
{
  "status": "success",
  "data": {},
  "metadata": {
    "agent": "sec_filing_agent",
    "latency_ms": 1234,
    "tokens_used": 1500,
    "grounding_score": 0.92,
    "trace_id": "abc123"
  }
}
```

### Environment Config Pattern
Use `app/config/settings.py` with `pydantic-settings`.
Never hardcode secrets. Always load from environment.
