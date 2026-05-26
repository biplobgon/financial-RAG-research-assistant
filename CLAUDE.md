# CLAUDE.md — Engineering Memory for Claude AI Agent

## Project: Financial RAG Research Assistant

### Overview
Enterprise-grade Financial Intelligence Platform powered by RAG, LLM workflows, multi-agent orchestration, and AI-assisted investment research.

### Architecture Summary
- **Backend**: FastAPI (async), Python 3.11+
- **RAG Stack**: LangChain + LlamaIndex + ChromaDB
- **LLM**: Vertex AI (primary), OpenAI-compatible abstraction
- **Agents**: SEC Filing, Earnings Analysis, Portfolio Intelligence, Executive Summary, Financial Research, Evaluation & Governance
- **Infra**: Docker, Docker Compose, Kubernetes, Helm, Terraform
- **Observability**: Prometheus, Grafana, OpenTelemetry, MLflow, LangSmith-compatible tracing
- **Cache**: Redis
- **Distributed**: Ray
- **Inference**: vLLM, Triton simulation

### Folder Structure
```
app/
├── agents/          # Multi-agent orchestration (6 agents)
├── workflows/       # LangGraph/LangChain workflow orchestration
├── rag/             # RAG pipeline: ingestion, chunking, retrieval
├── embeddings/      # Embedding generation and management
├── evaluation/      # Hallucination detection, grounding, scoring
├── observability/   # Prometheus, OTel, MLflow integration
├── governance/      # AI governance, prompt management, guardrails
├── inference/       # vLLM/Triton inference abstraction
├── api/             # FastAPI routers and schemas
├── services/        # Business logic services
├── models/          # Pydantic models and domain schemas
├── utils/           # Shared utilities
└── config/          # Settings, environment, constants
```

### API Routes
- GET /health
- POST /query
- POST /retrieve
- POST /analyze
- POST /summarize
- POST /portfolio
- POST /evaluate
- GET /metrics
- GET /trace
- POST /governance

### Key Design Patterns
- Async-first: all I/O operations are async
- Dependency injection via FastAPI Depends
- Repository pattern for data access
- Factory pattern for agent creation
- Observer pattern for observability hooks
- Retry/fallback logic on LLM calls

### Environment Variables
- OPENAI_API_KEY / VERTEX_AI_KEY
- CHROMA_HOST, CHROMA_PORT
- REDIS_URL
- MLFLOW_TRACKING_URI
- OTEL_EXPORTER_OTLP_ENDPOINT
- RAY_ADDRESS

### Naming Conventions
- Files: snake_case
- Classes: PascalCase
- Functions: snake_case
- Constants: UPPER_SNAKE_CASE
- API routes: kebab-case paths

### Implementation Status
- [x] Repository structure created
- [x] Directory scaffold
- [x] Memory files (CLAUDE.md, OPENCODE.md, CODEX.md, COPILOT.md, CHANGELOG.md)
- [x] app/config module
- [x] app/models module
- [x] app/utils module
- [x] app/rag pipeline
- [x] app/embeddings
- [x] app/agents (6 agents)
- [x] app/api routes
- [x] app/evaluation
- [x] app/observability
- [x] app/governance
- [x] app/inference
- [x] app/services
- [x] app/workflows
- [x] infrastructure (Docker, K8s, Helm)
- [x] tests (unit + integration)
- [x] data/ structure + synthetic data
- [x] notebooks/
- [x] docs/ (architecture, reports, diagrams)
- [x] README.md
- [x] CI/CD workflows

### Agent Orchestration
Each agent follows: Input -> Context Retrieval -> LLM Generation -> Grounding Check -> Output

### Testing Strategy
- Unit tests: pytest + AsyncMock
- Integration tests: TestClient + fixture ChromaDB
- Evaluation tests: custom scoring assertions

### Future Roadmap
- Real-time streaming API endpoints
- GraphQL schema for complex financial queries
- Fine-tuned financial embeddings model
- Multi-modal document parsing (tables, charts)
- Agent memory with episodic storage
