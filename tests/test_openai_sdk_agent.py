from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.openai_sdk.agent import OpenAISDKAgent
from core.memory import InMemoryMemory

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_openai_client(mocker):
    """Fixture that mocks the OpenAI client."""
    mock_create = AsyncMock()
    # Patch the class in the module where it is used
    mock_client_class = mocker.patch("agents.openai_sdk.agent.AsyncOpenAI")
    mock_client_class.return_value.chat.completions.create = mock_create
    return mock_create


async def test_agent_uses_memory_to_build_context(mock_openai_client: AsyncMock):
    """
    Meaningful test: checks that the agent uses memory history
    to build the context on the second call.
    """
    # --- 1. Arrange ---

    # Create a REAL memory instance
    memory = InMemoryMemory()
    # Create agent and inject our memory
    agent = OpenAISDKAgent(memory=memory)

    dialog_id = "test-convo-context"

    # Prepare two different responses from the "smart" mock
    response1 = MagicMock()
    response1.choices[0].message.content = "Nice to meet you, Vitaliy!"
    response2 = MagicMock()
    response2.choices[0].message.content = "Your name is Vitaliy."
    mock_openai_client.side_effect = [response1, response2]

    # --- 2. Act (First call) ---
    await agent.chat("My name is Vitaliy", dialog_id=dialog_id)

    # Intermediate check: ensure memory was updated
    history_after_first_call = await memory.get_history(dialog_id)
    assert len(history_after_first_call) == 2
    assert history_after_first_call[0]["content"] == "My name is Vitaliy"

    # --- 3. Act (Second call) ---
    await agent.chat("What is my name?", dialog_id=dialog_id)

    # --- 4. Assert (Main behavior check) ---

    # Ensure LLM was called twice
    assert mock_openai_client.call_count == 2

    # Get arguments of the SECOND LLM call
    second_call_args = mock_openai_client.call_args_list[1]
    messages_sent_to_llm = second_call_args.kwargs["messages"]

    # Check that the first dialog's history was included in the prompt for the second!
    expected_history_in_prompt = [
        {"role": "user", "content": "My name is Vitaliy"},
        {"role": "assistant", "content": "Nice to meet you, Vitaliy!"},
    ]

    # Check that messages at index 1-2 are our history
    assert messages_sent_to_llm[1:3] == expected_history_in_prompt
    # Check that the last message is the new question
    assert messages_sent_to_llm[3]["content"] == "What is my name?"

    # Final check: memory now contains 4 messages
    history_after_second_call = await memory.get_history(dialog_id)
    assert len(history_after_second_call) == 4
