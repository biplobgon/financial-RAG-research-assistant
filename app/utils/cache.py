"""Redis caching utilities for RAG query results."""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_redis_client = None


async def get_redis():
    """Get or create Redis client."""
    global _redis_client
    if _redis_client is None:
        try:
            import redis.asyncio as aioredis
            from app.config.settings import settings
            _redis_client = aioredis.from_url(
                settings.redis_url,
                max_connections=settings.redis_max_connections,
                decode_responses=True,
            )
            await _redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis unavailable, caching disabled: {e}")
            _redis_client = None
    return _redis_client


def compute_cache_key(prefix: str, **kwargs) -> str:
    """Compute a deterministic cache key from parameters."""
    key_data = json.dumps(kwargs, sort_keys=True, default=str)
    hash_val = hashlib.sha256(key_data.encode()).hexdigest()[:16]
    return f"financial_rag:{prefix}:{hash_val}"


async def cache_get(key: str) -> Optional[Any]:
    """Get value from cache."""
    try:
        client = await get_redis()
        if client is None:
            return None
        value = await client.get(key)
        if value:
            return json.loads(value)
    except Exception as e:
        logger.debug(f"Cache get failed for key {key}: {e}")
    return None


async def cache_set(key: str, value: Any, ttl: int = 3600) -> bool:
    """Set value in cache with TTL."""
    try:
        client = await get_redis()
        if client is None:
            return False
        await client.setex(key, ttl, json.dumps(value, default=str))
        return True
    except Exception as e:
        logger.debug(f"Cache set failed for key {key}: {e}")
        return False


async def cache_delete(key: str) -> bool:
    """Delete key from cache."""
    try:
        client = await get_redis()
        if client is None:
            return False
        await client.delete(key)
        return True
    except Exception as e:
        logger.debug(f"Cache delete failed for key {key}: {e}")
        return False


async def cache_invalidate_pattern(pattern: str) -> int:
    """Invalidate all cache keys matching pattern."""
    try:
        client = await get_redis()
        if client is None:
            return 0
        keys = await client.keys(f"financial_rag:{pattern}:*")
        if keys:
            return await client.delete(*keys)
    except Exception as e:
        logger.debug(f"Cache invalidation failed: {e}")
    return 0
