# OPENCODE.md — Engineering Memory for OpenCode AI Agent

## Project: Financial RAG Research Assistant

### Quick Context
Production enterprise Financial Intelligence platform. RAG + Multi-Agent + FastAPI + ChromaDB + Vertex AI.

### Stack Summary
| Layer | Technology |
|-------|-----------|
| LLM | Vertex AI / OpenAI-compatible |
| Orchestration | LangChain, LlamaIndex |
| Vector DB | ChromaDB |
| Backend | FastAPI (async) |
| Cache | Redis |
| Distributed | Ray |
| Inference | vLLM, Triton |
| Observability | Prometheus, Grafana, OTel, MLflow |
| Infra | Docker, K8s, Helm, Terraform |

### Active Modules
- `app/agents/` — 6 specialized financial AI agents
- `app/rag/` — full ingestion-to-retrieval pipeline
- `app/api/` — FastAPI async endpoints
- `app/evaluation/` — hallucination detection + scoring
- `app/observability/` — metrics, traces, logs
- `app/governance/` — prompt management, guardrails

### Key Implementation Rules
1. Always use async/await for I/O
2. Use Pydantic v2 models for all schemas
3. Inject dependencies via FastAPI Depends
4. Log all LLM calls with token counts
5. Score every response for hallucination risk
6. Emit OpenTelemetry spans for all agent calls

### Agent Names and Files
- `sec_filing_agent.py` — SEC document retrieval
- `earnings_agent.py` — earnings transcript analysis
- `portfolio_agent.py` — portfolio Q&A and insights
- `executive_summary_agent.py` — business summary generation
- `research_agent.py` — multi-doc financial research
- `evaluation_agent.py` — governance and quality scoring

### Common Utilities
- `app/utils/text_processing.py` — chunking, cleaning
- `app/utils/financial_utils.py` — ticker parsing, ratio computation
- `app/utils/retry.py` — exponential backoff, fallback
- `app/utils/logging.py` — structured JSON logging

### Do Not
- Add synchronous blocking I/O
- Hardcode API keys
- Skip error handling on LLM calls
- Skip hallucination checks on financial answers
