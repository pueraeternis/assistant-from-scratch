# core/base_agent.py

from abc import ABC, abstractmethod
from typing import Optional


class BaseAgent(ABC):
    """Abstract interface that all agents must implement."""

    @abstractmethod
    async def chat(self, message: str, dialog_id: Optional[str] = None) -> str:
        """Accept a message and return agent response."""
