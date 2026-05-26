# GitHub Copilot Instructions — Financial RAG Research Assistant

## Project Context
Enterprise Financial AI Platform. Production RAG + Multi-Agent FastAPI application.
Python 3.11+, async-first, Pydantic v2, LangChain, ChromaDB, Vertex AI / OpenAI.

## Code Style
- All I/O must be async (use `async def`, `await`, `asyncio.gather`)
- Use Pydantic v2 models for all schemas
- Inject dependencies via FastAPI `Depends()`
- Use structured logging: `logger.info("message", extra={"key": "value"})`
- Every LLM call needs retry decorator: `@retry_async(max_retries=3)`
- Every agent response needs grounding evaluation

## File Conventions
- `app/agents/` → `BaseFinancialAgent` subclasses
- `app/rag/` → RAG pipeline components
- `app/api/v1/routes/` → FastAPI route handlers
- `app/models/requests.py` → Pydantic request schemas
- `app/models/responses.py` → Pydantic response schemas
- `tests/unit/test_<module>.py` → pytest unit tests

## Do NOT generate
- Synchronous database calls
- Hardcoded API keys or secrets
- Missing error handling on LLM calls
- Global mutable state
- Financial guarantees or price predictions in responses
