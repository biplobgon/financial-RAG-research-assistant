"""Unified LLM client with OpenAI/Vertex AI/vLLM support."""
from __future__ import annotations

import logging
import time
from typing import Optional, Any

from app.config.settings import settings
from app.utils.retry import retry_async
from app.utils.logging import get_logger

logger = get_logger(__name__)


class LLMClient:
    """
    Unified async LLM client with provider abstraction.

    Supports:
    - OpenAI API
    - Vertex AI (Google)
    - vLLM (self-hosted OpenAI-compatible)
    - Mock (development/testing)
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = None,
        max_tokens: int = None,
    ):
        self.provider = provider or settings.llm_provider
        self.model = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self._client = None
        self._total_tokens = 0
        self._call_count = 0

    @retry_async(max_retries=3, base_delay=2.0, backoff_factor=2.0)
    async def generate(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> dict:
        """
        Generate LLM completion.

        Returns:
        - content: str
        - tokens_prompt: int
        - tokens_completion: int
        - model: str
        """
        start_time = time.monotonic()
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens or self.max_tokens

        client = await self._get_client()

        if self.provider == "mock":
            return self._mock_response(messages)

        try:
            if self.provider in ("openai", "vllm"):
                result = await self._openai_generate(client, messages, temp, tokens)
            elif self.provider == "vertex_ai":
                result = await self._vertex_generate(client, messages, temp, tokens)
            else:
                result = self._mock_response(messages)

            latency = (time.monotonic() - start_time) * 1000
            self._total_tokens += result.get("tokens_prompt", 0) + result.get("tokens_completion", 0)
            self._call_count += 1

            logger.info(
                "LLM generation complete",
                extra={
                    "provider": self.provider,
                    "model": self.model,
                    "tokens_prompt": result.get("tokens_prompt", 0),
                    "tokens_completion": result.get("tokens_completion", 0),
                    "latency_ms": round(latency, 2),
                },
            )
            return result

        except Exception as e:
            logger.error(f"LLM generation failed ({self.provider}): {e}")
            raise

    async def _get_client(self):
        """Lazily initialize LLM client."""
        if self._client is not None:
            return self._client

        if self.provider == "openai":
            import openai
            self._client = openai.AsyncOpenAI(
                api_key=settings.openai_api_key,
                timeout=settings.llm_timeout,
            )
        elif self.provider == "vllm":
            import openai
            # vLLM uses OpenAI-compatible API
            self._client = openai.AsyncOpenAI(
                base_url="http://vllm-service:8000/v1",
                api_key="not-needed",
                timeout=settings.llm_timeout,
            )
        elif self.provider == "vertex_ai":
            try:
                import vertexai
                from vertexai.generative_models import GenerativeModel
                vertexai.init(project=settings.vertex_ai_project, location=settings.vertex_ai_location)
                self._client = GenerativeModel(self.model)
            except Exception as e:
                logger.warning(f"Vertex AI initialization failed: {e}. Using mock.")
                self.provider = "mock"
                self._client = None
        else:
            self.provider = "mock"
            self._client = None

        return self._client

    async def _openai_generate(
        self,
        client,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> dict:
        """OpenAI API generation."""
        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {
            "content": response.choices[0].message.content,
            "tokens_prompt": response.usage.prompt_tokens,
            "tokens_completion": response.usage.completion_tokens,
            "model": response.model,
        }

    async def _vertex_generate(
        self,
        client,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> dict:
        """Vertex AI generation."""
        import asyncio
        prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

        def _generate():
            return client.generate_content(
                prompt,
                generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
            )

        response = await asyncio.to_thread(_generate)
        return {
            "content": response.text,
            "tokens_prompt": 0,  # Vertex AI doesn't always expose token counts
            "tokens_completion": 0,
            "model": self.model,
        }

    def _mock_response(self, messages: list[dict]) -> dict:
        """Mock LLM response for development/testing."""
        last_user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "No query",
        )
        return {
            "content": (
                f"[MOCK FINANCIAL ANALYSIS]\n\n"
                f"Based on the provided financial documents and context, here is a comprehensive analysis "
                f"addressing your query about: {last_user_msg[:100]}...\n\n"
                f"Key findings from available SEC filings and earnings transcripts indicate strong "
                f"fundamental performance with notable risk factors as disclosed in regulatory filings. "
                f"Revenue trends, balance sheet metrics, and management commentary have been synthesized "
                f"to provide this structured financial intelligence output.\n\n"
                f"Note: This is a mock response for development. Configure LLM_PROVIDER and API keys for live responses."
            ),
            "tokens_prompt": 500,
            "tokens_completion": 150,
            "model": "mock-gpt-4",
        }

    @property
    def usage_stats(self) -> dict:
        """Return cumulative usage statistics."""
        return {
            "provider": self.provider,
            "model": self.model,
            "total_tokens": self._total_tokens,
            "call_count": self._call_count,
        }
