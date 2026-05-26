"""Retry and fallback utilities for LLM and external API calls."""
from __future__ import annotations

import asyncio
import functools
import logging
import random
from typing import TypeVar, Callable, Any, Optional, Type

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryError(Exception):
    """Raised when all retry attempts are exhausted."""
    def __init__(self, message: str, last_exception: Optional[Exception] = None):
        super().__init__(message)
        self.last_exception = last_exception


async def async_retry(
    func: Callable,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple[Type[Exception], ...] = (Exception,),
    **kwargs,
) -> Any:
    """
    Async retry with exponential backoff and jitter.

    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        backoff_factor: Exponential backoff multiplier
        jitter: Add random jitter to prevent thundering herd
        retryable_exceptions: Exception types that trigger retry
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except retryable_exceptions as e:
            last_exception = e
            if attempt == max_retries:
                logger.error(
                    "All retry attempts exhausted",
                    extra={
                        "function": func.__name__,
                        "attempts": attempt + 1,
                        "error": str(e),
                    },
                )
                raise RetryError(
                    f"Function {func.__name__} failed after {max_retries + 1} attempts",
                    last_exception=e,
                ) from e

            delay = min(base_delay * (backoff_factor ** attempt), max_delay)
            if jitter:
                delay *= 0.5 + random.random() * 0.5

            logger.warning(
                "Retry attempt scheduled",
                extra={
                    "function": func.__name__,
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "delay_seconds": round(delay, 2),
                    "error": str(e),
                },
            )
            await asyncio.sleep(delay)

    raise RetryError(f"Unexpected retry loop exit for {func.__name__}")


def retry_async(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple[Type[Exception], ...] = (Exception,),
):
    """Decorator for async retry with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            return await async_retry(
                func, *args,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                backoff_factor=backoff_factor,
                jitter=jitter,
                retryable_exceptions=retryable_exceptions,
                **kwargs,
            )
        return wrapper
    return decorator


class FallbackChain:
    """Execute a chain of async callables, returning first successful result."""

    def __init__(self, *callables: Callable):
        self.callables = callables

    async def execute(self, *args, **kwargs) -> Any:
        """Try each callable in order, returning first success."""
        exceptions = []
        for i, func in enumerate(self.callables):
            try:
                logger.debug(f"FallbackChain: trying option {i+1}/{len(self.callables)}: {func.__name__}")
                result = await func(*args, **kwargs)
                if i > 0:
                    logger.info(f"FallbackChain: succeeded on fallback option {i+1}")
                return result
            except Exception as e:
                exceptions.append(e)
                logger.warning(
                    f"FallbackChain: option {i+1} failed",
                    extra={"function": func.__name__, "error": str(e)},
                )

        raise RetryError(
            f"All {len(self.callables)} fallback options failed",
            last_exception=exceptions[-1] if exceptions else None,
        )
