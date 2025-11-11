# tests/test_memory.py

import pytest

from core.memory import InMemoryMemory

# Mark all tests in this file as asynchronous
pytestmark = pytest.mark.asyncio


@pytest.fixture
def memory() -> InMemoryMemory:
    """Fixture that creates a fresh memory instance for each test."""
    return InMemoryMemory()


async def test_append_and_get_history(memory: InMemoryMemory):
    """Check basic logic for appending and retrieving history."""
    dialog_id = "test-1"

    await memory.append(dialog_id, "user", "Hello")
    await memory.append(dialog_id, "assistant", "Hi there!")

    history = await memory.get_history(dialog_id)

    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "Hello"}
    assert history[1] == {"role": "assistant", "content": "Hi there!"}


async def test_get_history_limit(memory: InMemoryMemory):
    """Check that the 'limit' parameter works correctly."""
    dialog_id = "test-2"
    for i in range(5):
        await memory.append(dialog_id, "user", f"Message {i}")

    # Request only the last 3 messages
    limited_history = await memory.get_history(dialog_id, limit=3)

    assert len(limited_history) == 3
    assert limited_history[0]["content"] == "Message 2"
    assert limited_history[1]["content"] == "Message 3"
    assert limited_history[2]["content"] == "Message 4"


async def test_memory_isolates_dialogs(memory: InMemoryMemory):
    """Check that memory correctly isolates different dialogs."""
    await memory.append("dialog-A", "user", "Info for A")
    await memory.append("dialog-B", "user", "Info for B")

    history_A = await memory.get_history("dialog-A")
    history_B = await memory.get_history("dialog-B")

    assert len(history_A) == 1
    assert history_A[0]["content"] == "Info for A"

    assert len(history_B) == 1
    assert history_B[0]["content"] == "Info for B"


async def test_get_history_for_nonexistent_dialog(memory: InMemoryMemory):
    """Check that a nonexistent dialog returns an empty list."""
    history = await memory.get_history("nonexistent-id")
    assert history == []
