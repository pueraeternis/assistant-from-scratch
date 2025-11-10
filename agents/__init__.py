from typing import List, Optional

from config.settings import settings
from core.base_tool import BaseTool
from core.memory import RedisMemory
from core.registry import register_agent

from .echo.agent import EchoAgent
from .openai_sdk.agent import OpenAISDKAgent


def openai_sdk_factory(tools: Optional[List[BaseTool]] = None) -> OpenAISDKAgent:
    """Factory for creating OpenAISDKAgent with tools."""
    memory = RedisMemory(redis_url=settings.REDIS_URL)
    return OpenAISDKAgent(memory=memory, tools=tools)


register_agent("echo", lambda: EchoAgent(prefix="Echo: "))
register_agent("openai_sdk", openai_sdk_factory)
