# agents/__init__.py

from typing import List, Optional

from config.settings import settings
from core.base_tool import BaseTool
from core.memory import RedisMemory
from core.registry import register_agent

from .echo.agent import EchoAgent
from .openai.agent import OpenAIAgent


def openai_factory(tools: Optional[List[BaseTool]] = None) -> OpenAIAgent:
    """Factory for creating OpenAIAgent with tools."""
    memory = RedisMemory(redis_url=settings.REDIS_URL)
    return OpenAIAgent(memory=memory, tools=tools)


register_agent("echo", lambda: EchoAgent(prefix="Echo: "))
register_agent("openai", openai_factory)
