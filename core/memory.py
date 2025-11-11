# core/memory.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal

import redis.asyncio as redis

from core.logging import get_logger

MessageRole = Literal["system", "user", "assistant"]
ChatMessage = Dict[str, Any]

logger = get_logger(__name__)


class BaseMemory(ABC):
    """Abstract memory API."""

    @abstractmethod
    async def append(self, dialog_id: str, role: str, content: str) -> None: ...

    @abstractmethod
    async def get_history(self, dialog_id: str, limit: int = 50) -> List[Dict[str, Any]]: ...


class InMemoryMemory(BaseMemory):
    """Simple in-memory history store for development/testing."""

    def __init__(self) -> None:
        self.store: Dict[str, List[Dict[str, str]]] = {}

    async def append(self, dialog_id: str, role: MessageRole, content: str) -> None:
        self.store.setdefault(dialog_id, []).append({"role": role, "content": content})

    async def get_history(self, dialog_id: str, limit: int = 50):
        return self.store.get(dialog_id, [])[-limit:]


class RedisMemory(BaseMemory):
    """
    Persistent memory implementation using Redis Lists.
    Conforms to the BaseMemory interface.
    """

    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.key_prefix = "history:"
        logger.info("🧠 RedisMemory initialized. Connecting to Redis...")

    def _get_key(self, dialog_id: str) -> str:
        return f"{self.key_prefix}{dialog_id}"

    async def get_history(self, dialog_id: str, limit: int = 50) -> List[ChatMessage]:
        key = self._get_key(dialog_id)
        raw_messages = await self.redis.lrange(key, 0, -1)  # type: ignore

        history: List[ChatMessage] = []
        for msg in reversed(raw_messages):
            try:
                role, content = msg.split(":", 1)
                history.append({"role": role, "content": content})
            except ValueError:
                logger.warning("Skipping malformed message in Redis history for key %s: %s", key, msg)
                continue

        return history[-limit:]

    async def append(self, dialog_id: str, role: MessageRole, content: str) -> None:
        key = self._get_key(dialog_id)
        message_to_store = f"{role}:{content}"
        await self.redis.lpush(key, message_to_store)  # type: ignore
