"""Redis client for session management and caching."""

import json
import uuid
from typing import Any, Optional

import redis

from src.core.logging import logger

_redis_client: Optional[redis.Redis] = None


def get_redis_client() -> Optional[redis.Redis]:
    """Get the Redis client singleton."""
    global _redis_client
    return _redis_client


def init_redis(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    password: Optional[str] = None,
    decode_responses: bool = True,
) -> redis.Redis:
    """Initialize Redis client."""
    global _redis_client
    _redis_client = redis.Redis(
        host=host,
        port=port,
        db=db,
        password=password,
        decode_responses=decode_responses,
    )
    logger.info(f"Redis client initialized: {host}:{port}/{db}")
    return _redis_client


def close_redis() -> None:
    """Close Redis connection."""
    global _redis_client
    if _redis_client:
        _redis_client.close()
        _redis_client = None


# ─── Session Management ────────────────────────────────────────────────────────


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return str(uuid.uuid4())


def save_chat_message(
    session_id: str,
    role: str,
    content: str,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Save a chat message to session history."""
    client = get_redis_client()
    if not client:
        return

    key = f"session:{session_id}:messages"
    message = {
        "role": role,
        "content": content,
        "metadata": metadata or {},
    }
    client.rpush(key, json.dumps(message, ensure_ascii=False))
    client.expire(key, 86400 * 30)  # 30 days TTL


def get_chat_history(session_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """Get chat history for a session."""
    client = get_redis_client()
    if not client:
        return []

    key = f"session:{session_id}:messages"
    messages = client.lrange(key, -limit, -1)
    return [json.loads(m) for m in messages]


def clear_chat_history(session_id: str) -> None:
    """Clear chat history for a session."""
    client = get_redis_client()
    if not client:
        return

    key = f"session:{session_id}:messages"
    client.delete(key)


def delete_session(session_id: str) -> None:
    """Delete entire session."""
    client = get_redis_client()
    if not client:
        return

    client.delete(f"session:{session_id}:messages")


# ─── Retrieval Caching ────────────────────────────────────────────────────────


def cache_retrieval_result(
    query_hash: str,
    result: list[dict[str, Any]],
    ttl: int = 3600,
) -> None:
    """Cache retrieval results for similar queries."""
    client = get_redis_client()
    if not client:
        return

    key = f"retrieval_cache:{query_hash}"
    client.setex(key, ttl, json.dumps(result, ensure_ascii=False))


def get_cached_retrieval(query_hash: str) -> Optional[list[dict[str, Any]]]:
    """Get cached retrieval result."""
    client = get_redis_client()
    if not client:
        return None

    key = f"retrieval_cache:{query_hash}"
    cached = client.get(key)
    if cached:
        return json.loads(cached)
    return None


def hash_query(query: str) -> str:
    """Create a hash key for a query."""
    import hashlib
    return hashlib.sha256(query.encode()).hexdigest()[:32]
