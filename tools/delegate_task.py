# tools/delegate_task.py

import logging
from typing import Any, Callable, List

from core.base_agent import BaseAgent
from core.base_tool import BaseTool

logger = logging.getLogger(__name__)


class DelegateTaskTool(BaseTool):
    """
    Tool for delegating a specific, well-defined sub-task to a specialized agent.
    """

    name = "DelegateTask"
    description = (
        "Use this tool to delegate a specific, well-defined sub-task to a specialist agent. "
        "Specify which specialist to use via the 'specialist_role' parameter "
        "(e.g., 'researcher', 'database_analyst') and provide a clear 'task_description' "
        "for them to execute."
    )

    def __init__(self, get_agent_func: Callable[..., BaseAgent], all_tools: List[BaseTool]):
        super().__init__()
        self.get_agent = get_agent_func
        self.all_tools = all_tools

    def _run(self, **kwargs: Any) -> str:
        """Synchronous version is not supported for this tool."""
        raise NotImplementedError("DelegateTaskTool must be run asynchronously.")

    async def _arun(self, **kwargs: Any) -> str:
        """Asynchronously creates a specialist agent and executes the delegated task."""
        role = kwargs.get("specialist_role")
        task = kwargs.get("task_description")

        if not role or not task:
            return "Error: Both 'specialist_role' and 'task_description' parameters are required."

        logger.info("Delegating task to '%s': %s", role, task)

        try:
            specialist_agent = self.get_agent(role, tools=self.all_tools)
            response = await specialist_agent.chat(message=task)

            logger.info("Received response from '%s': %s", role, response)
            return response

        except KeyError:
            logger.error("Specialist role '%s' not found.", role)
            return f"Error: Specialist role '{role}' not found."
        except Exception as e:
            logger.error("Error while delegating task to '%s': %s", role, e)
            return f"An error occurred while delegating task to '{role}': {e}"
