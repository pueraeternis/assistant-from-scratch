# core/base_tool.py

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Abstract base class for all tools."""

    name: str = "base_tool"
    description: str = "A brief description of what this tool does."

    @abstractmethod
    def _run(self, **kwargs: Any) -> str:
        """
        The core logic of the tool that must be implemented by subclasses.
        It accepts keyword arguments to perform its action.
        """
        raise NotImplementedError

    def run(self, **kwargs: Any) -> str:
        """
        Public method to execute the tool with error handling.
        This method wraps the core logic and passes all keyword arguments.
        """
        try:
            return self._run(**kwargs)
        except Exception as e:
            return f"Error while running tool {self.name} with args {kwargs}: {e}"
