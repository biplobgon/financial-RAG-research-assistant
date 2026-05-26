"""Application settings using pydantic-settings."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Production application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="Financial RAG Research Assistant", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: str = Field(default="production", description="Deployment environment")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Log level")

    # API
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=4, description="Number of API workers")
    api_timeout: int = Field(default=120, description="Request timeout in seconds")

    # LLM Providers
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    vertex_ai_project: Optional[str] = Field(default=None, description="GCP project for Vertex AI")
    vertex_ai_location: str = Field(default="us-central1", description="Vertex AI region")
    vertex_ai_key: Optional[str] = Field(default=None, description="Vertex AI service account key path")
    llm_provider: str = Field(default="openai", description="Primary LLM provider: openai | vertex_ai")
    llm_model: str = Field(default="gpt-4o", description="Primary LLM model name")
    llm_temperature: float = Field(default=0.1, description="LLM temperature")
    llm_max_tokens: int = Field(default=4096, description="Max tokens for LLM responses")
    llm_timeout: int = Field(default=60, description="LLM call timeout in seconds")
    llm_max_retries: int = Field(default=3, description="Max LLM retry attempts")

    # Embeddings
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model")
    embedding_dimensions: int = Field(default=1536, description="Embedding vector dimensions")
    embedding_batch_size: int = Field(default=100, description="Embedding batch size")

    # ChromaDB
    chroma_host: str = Field(default="localhost", description="ChromaDB host")
    chroma_port: int = Field(default=8001, description="ChromaDB port")
    chroma_collection_sec: str = Field(default="sec_filings", description="SEC filings collection")
    chroma_collection_earnings: str = Field(default="earnings_transcripts", description="Earnings collection")
    chroma_collection_market: str = Field(default="market_data", description="Market data collection")
    chroma_collection_research: str = Field(default="research_reports", description="Research reports collection")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    redis_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    redis_max_connections: int = Field(default=20, description="Redis max connections")

    # RAG Pipeline
    rag_top_k: int = Field(default=10, description="Top-K documents to retrieve")
    rag_rerank_top_k: int = Field(default=3, description="Top-K after reranking")
    rag_chunk_size: int = Field(default=512, description="Document chunk size in tokens")
    rag_chunk_overlap: int = Field(default=64, description="Chunk overlap in tokens")
    rag_similarity_threshold: float = Field(default=0.7, description="Minimum similarity threshold")
    rag_hybrid_alpha: float = Field(default=0.7, description="Dense/sparse retrieval weight (1.0=dense)")

    # Evaluation
    hallucination_threshold: float = Field(default=0.85, description="Min grounding score to pass")
    evaluation_enabled: bool = Field(default=True, description="Enable response evaluation")
    grounding_model: str = Field(default="gpt-4o-mini", description="Model for grounding checks")

    # Observability
    otel_endpoint: str = Field(default="http://localhost:4317", description="OpenTelemetry collector endpoint")
    otel_service_name: str = Field(default="financial-rag-api", description="OTel service name")
    prometheus_port: int = Field(default=9090, description="Prometheus metrics port")
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", description="MLflow tracking URI")
    langsmith_api_key: Optional[str] = Field(default=None, description="LangSmith API key")
    langsmith_project: str = Field(default="financial-rag", description="LangSmith project name")

    # Ray
    ray_address: Optional[str] = Field(default=None, description="Ray cluster address")
    ray_num_cpus: int = Field(default=4, description="Ray CPUs to allocate")

    # Security
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    api_keys: str = Field(default="", description="Comma-separated valid API keys")
    cors_origins: str = Field(default="*", description="CORS allowed origins")

    # Data paths
    data_raw_path: str = Field(default="data/raw", description="Raw data directory")
    data_processed_path: str = Field(default="data/processed", description="Processed data directory")
    data_synthetic_path: str = Field(default="data/synthetic", description="Synthetic data directory")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return v.upper()

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        valid = {"development", "staging", "production", "test"}
        if v.lower() not in valid:
            raise ValueError(f"environment must be one of {valid}")
        return v.lower()

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_debug(self) -> bool:
        return self.debug or self.environment == "development"

    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()


settings = get_settings()
