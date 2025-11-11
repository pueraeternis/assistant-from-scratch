# tests/test_echo_agent.py

import asyncio

from agents.echo.agent import EchoAgent


def test_echo_agent_basic():
    agent = EchoAgent(prefix="> ")
    response = asyncio.run(agent.chat("hello", dialog_id="d1"))
    assert response == "> hello"
