# core/base_tool.py

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Abstract base class for all tools."""

    name: str = "base_tool"
    description: str = "A brief description of what this tool does."

    @abstractmethod
    def _run(self, **kwargs: Any) -> str:
        """The synchronous core logic of the tool."""
        raise NotImplementedError("This tool does not support synchronous execution.")

    def run(self, **kwargs: Any) -> str:
        """Public method for synchronous execution."""
        try:
            return self._run(**kwargs)
        except Exception as e:
            return f"Error while running tool {self.name} with args {kwargs}: {e}"

    async def _arun(self, **kwargs: Any) -> str:
        """The asynchronous core logic of the tool."""
        return self.run(**kwargs)

    async def arun(self, **kwargs: Any) -> str:
        """Public method for asynchronous execution with error handling."""
        try:
            return await self._arun(**kwargs)
        except Exception as e:
            return f"Error while running tool {self.name} with args {kwargs}: {e}"
