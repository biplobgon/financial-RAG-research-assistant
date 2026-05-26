# System Architecture — Financial RAG Research Assistant

## Overview

The Financial RAG Research Assistant follows a layered microservices architecture with clear separation between the API layer, agent orchestration, RAG pipeline, and infrastructure services.

## Architecture Layers

### Layer 1: API Gateway
- **FastAPI** async application serving all endpoints
- NGINX reverse proxy for production traffic
- CORS middleware, GZip compression, request timing
- Global exception handling with structured error responses

### Layer 2: Agent Orchestration
- **AgentOrchestrator** — intent-based routing to 6 specialized agents
- Parallel multi-agent execution with `asyncio.gather`
- Fallback chain: specific agent → research agent
- Routing statistics and performance tracking

### Layer 3: RAG Pipeline
- **Ingestion**: Async PDF/TXT/JSON document processing
- **Chunking**: Recursive (default) and semantic (SEC-aware) strategies
- **Embedding**: OpenAI text-embedding-3-small (or local sentence-transformers)
- **Retrieval**: Dense + sparse RRF hybrid from ChromaDB
- **Reranking**: Cross-encoder model for top-K precision

### Layer 4: LLM Generation
- **LLMClient**: Unified abstraction for OpenAI, Vertex AI, vLLM
- Retry with exponential backoff and jitter
- FallbackChain for provider failover
- Token usage tracking per request

### Layer 5: Evaluation
- **GroundingEvaluator**: LLM-as-judge with heuristic fallback
- **HallucinationDetector**: Financial figure, date, and entity checks
- **QualityScorer**: 4-dimension quality assessment

### Layer 6: Observability
- Prometheus metrics on all key operations
- OpenTelemetry distributed tracing
- MLflow experiment logging
- Structured JSON logging with trace context propagation
- Redis semantic caching (85%+ hit rate)

## Data Flow

```
Client Request
    ↓
FastAPI Endpoint (auth, validation, schema)
    ↓
Agent Router (intent detection)
    ↓
Specialized Agent (SEC/Earnings/Portfolio/Research/Summary)
    ↓ ← Parallel retrieval from ChromaDB (3 collections)
Context Building (top-K documents + metadata)
    ↓
LLM Generation (GPT-4o / Vertex AI)
    ↓
Grounding Evaluation (LLM-as-Judge)
    ↓
Redis Cache (if score >= 0.85)
    ↓
Response Assembly (content + sources + metadata)
    ↓
Client Response
```

## ChromaDB Collections

| Collection | Content | Avg Documents |
|-----------|---------|---------------|
| `sec_filings` | 10-K, 10-Q, 8-K chunks | 50,000+ |
| `earnings_transcripts` | Quarterly call chunks | 20,000+ |
| `market_data` | Financial indicators | 5,000+ |
| `research_reports` | Analyst report chunks | 10,000+ |

## Scalability Considerations

- **Horizontal scaling**: Kubernetes HPA scales API 2→10 replicas based on CPU/memory
- **Vector DB**: ChromaDB supports horizontal scaling via distributed deployment
- **Cache**: Redis cluster for high availability semantic caching
- **Ray**: Distributed processing for batch embedding and ingestion
- **vLLM**: Self-hosted LLM inference for cost optimization at scale
