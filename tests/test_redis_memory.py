# tests/test_redis_memory.py

import fakeredis.aioredis
import pytest

from core.memory import RedisMemory

pytestmark = pytest.mark.asyncio


@pytest.fixture
def redis_memory() -> RedisMemory:
    """
    Fixture that creates a RedisMemory instance, but "under the hood"
    uses a fake (in-memory) asynchronous Redis client.
    """
    # 1. Create a fake client. It behaves like a real one but works in memory.
    fake_redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)

    # 2. Create our RedisMemory class. URL doesn't matter here since we override the client.
    memory_instance = RedisMemory(redis_url="redis://localhost:6379/fake")

    # 3. "Monkey patch": replace the real client with our fake one.
    # This key trick avoids the need to change RedisMemory code.
    memory_instance.redis = fake_redis_client

    return memory_instance


async def test_append_and_get_history(redis_memory: RedisMemory):
    """Check that messages are appended and retrieved in the correct order."""
    dialog_id = "test-redis-1"

    await redis_memory.append(dialog_id, "user", "Hello Redis")
    await redis_memory.append(dialog_id, "assistant", "Hi there!")

    history = await redis_memory.get_history(dialog_id)

    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "Hello Redis"}
    assert history[1] == {"role": "assistant", "content": "Hi there!"}


async def test_get_history_limit(redis_memory: RedisMemory):
    """Check that the 'limit' parameter works correctly."""
    dialog_id = "test-redis-2"
    for i in range(10):
        await redis_memory.append(dialog_id, "user", f"Message {i}")

    # Request only the last 4 messages
    limited_history = await redis_memory.get_history(dialog_id, limit=4)

    assert len(limited_history) == 4
    assert limited_history[0]["content"] == "Message 6"
    assert limited_history[-1]["content"] == "Message 9"


async def test_redis_memory_isolates_dialogs(redis_memory: RedisMemory):
    """Check that memory correctly isolates different dialogs using key prefixes."""
    await redis_memory.append("convo-A", "user", "This is for A")
    await redis_memory.append("convo-B", "user", "This is for B")

    history_A = await redis_memory.get_history("convo-A")
    history_B = await redis_memory.get_history("convo-B")

    assert len(history_A) == 1
    assert history_A[0]["content"] == "This is for A"

    assert len(history_B) == 1
    assert history_B[0]["content"] == "This is for B"


async def test_get_history_for_nonexistent_dialog(redis_memory: RedisMemory):
    """Check that a nonexistent dialog returns an empty list."""
    history = await redis_memory.get_history("i-do-not-exist")
    assert history == []
