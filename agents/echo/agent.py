import asyncio
from typing import Optional

from core.base_agent import BaseAgent
from core.logging import get_logger

logger = get_logger(__name__)


class EchoAgent(BaseAgent):
    """Simple echo agent for development and smoke tests."""

    def __init__(self, prefix: str = "Echo: ") -> None:
        self.prefix = prefix

    async def chat(self, message: str, dialog_id: Optional[str] = None) -> str:
        """Return a deterministic echoed response."""
        logger.info("EchoAgent.chat called with dialog_id=%s", dialog_id)
        # simulate async work
        await asyncio.sleep(0)
        return f"{self.prefix}{message}"
