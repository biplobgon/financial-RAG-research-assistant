"""Application-wide constants."""

# Agent identifiers
AGENT_SEC_FILING = "sec_filing_agent"
AGENT_EARNINGS = "earnings_agent"
AGENT_PORTFOLIO = "portfolio_agent"
AGENT_EXECUTIVE_SUMMARY = "executive_summary_agent"
AGENT_RESEARCH = "research_agent"
AGENT_EVALUATION = "evaluation_agent"

AVAILABLE_AGENTS = [
    AGENT_SEC_FILING,
    AGENT_EARNINGS,
    AGENT_PORTFOLIO,
    AGENT_EXECUTIVE_SUMMARY,
    AGENT_RESEARCH,
    AGENT_EVALUATION,
]

# ChromaDB collection names
COLLECTION_SEC_FILINGS = "sec_filings"
COLLECTION_EARNINGS = "earnings_transcripts"
COLLECTION_MARKET_DATA = "market_data"
COLLECTION_RESEARCH = "research_reports"

# Filing types
SEC_FILING_TYPES = ["10-K", "10-Q", "8-K", "DEF 14A", "S-1", "20-F"]

# Financial document categories
DOC_CATEGORY_SEC = "sec_filing"
DOC_CATEGORY_EARNINGS = "earnings_transcript"
DOC_CATEGORY_RESEARCH = "research_report"
DOC_CATEGORY_MARKET = "market_data"

# Evaluation thresholds
GROUNDING_SCORE_EXCELLENT = 0.95
GROUNDING_SCORE_GOOD = 0.85
GROUNDING_SCORE_ACCEPTABLE = 0.70
GROUNDING_SCORE_POOR = 0.50

# Hallucination risk levels
HALLUCINATION_RISK_LOW = "low"
HALLUCINATION_RISK_MEDIUM = "medium"
HALLUCINATION_RISK_HIGH = "high"
HALLUCINATION_RISK_CRITICAL = "critical"

# API response status
STATUS_SUCCESS = "success"
STATUS_ERROR = "error"
STATUS_PARTIAL = "partial"

# Chunking strategies
CHUNKING_RECURSIVE = "recursive"
CHUNKING_SEMANTIC = "semantic"
CHUNKING_SENTENCE = "sentence"
CHUNKING_FIXED = "fixed"

# Retrieval modes
RETRIEVAL_DENSE = "dense"
RETRIEVAL_SPARSE = "sparse"
RETRIEVAL_HYBRID = "hybrid"

# Metric names for Prometheus
METRIC_REQUEST_DURATION = "financial_rag_request_duration_seconds"
METRIC_REQUEST_COUNT = "financial_rag_requests_total"
METRIC_LLM_TOKENS = "financial_rag_llm_tokens_total"
METRIC_RETRIEVAL_COUNT = "financial_rag_retrieval_documents_total"
METRIC_GROUNDING_SCORE = "financial_rag_grounding_score"
METRIC_CACHE_HITS = "financial_rag_cache_hits_total"
METRIC_CACHE_MISSES = "financial_rag_cache_misses_total"

# Trace span names
SPAN_RAG_PIPELINE = "rag.pipeline"
SPAN_RETRIEVAL = "rag.retrieval"
SPAN_LLM_GENERATION = "llm.generation"
SPAN_EMBEDDING = "embedding.encode"
SPAN_EVALUATION = "evaluation.score"
SPAN_AGENT_RUN = "agent.run"

# Default financial analysis sectors
FINANCIAL_SECTORS = [
    "Technology",
    "Financial Services",
    "Healthcare",
    "Energy",
    "Consumer Discretionary",
    "Consumer Staples",
    "Industrials",
    "Materials",
    "Real Estate",
    "Utilities",
    "Communication Services",
]

# SEC EDGAR base URL
SEC_EDGAR_BASE_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
SEC_EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions"
SEC_EDGAR_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts"

# Default timeouts (seconds)
DEFAULT_LLM_TIMEOUT = 60
DEFAULT_RETRIEVAL_TIMEOUT = 10
DEFAULT_EMBEDDING_TIMEOUT = 30

# Pagination defaults
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
