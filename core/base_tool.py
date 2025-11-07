from abc import ABC, abstractmethod


class BaseTool(ABC):
    """Abstract base class for all tools."""

    name: str = "base_tool"
    description: str = "A brief description of what this tool does."

    @abstractmethod
    def _run(self, query: str) -> str:
        """
        The core logic of the tool that must be implemented by subclasses.
        It should process the input query and return a string result.
        """
        raise NotImplementedError

    def run(self, query: str) -> str:
        """
        Public method to execute the tool with error handling.
        This method wraps the core logic to catch and report exceptions.
        """
        try:
            return self._run(query)
        except Exception as e:
            return f"Error while running tool {self.name}: {e}"
