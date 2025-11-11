# agents/openai/agent.py

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

from openai import APIError, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from config.settings import settings
from core.base_agent import BaseAgent
from core.base_tool import BaseTool
from core.logging import get_logger
from core.memory import BaseMemory

logger = get_logger(__name__)


class OpenAIAgent(BaseAgent):
    """
    Agent implemented using the standard asynchronous OpenAI SDK.
    It supports a Reason-Act (ReAct) loop to use tools.
    """

    def __init__(self, memory: BaseMemory, tools: Optional[List[BaseTool]] = None):
        self.client = AsyncOpenAI(
            base_url=settings.OPENAI_API_URL,
            api_key=settings.OPENAI_API_KEY,
        )
        self.memory = memory
        self.model: str = settings.LLM_MODEL_NAME
        self.temperature = settings.LLM_TEMPERATURE
        self.max_tokens = settings.LLM_MAX_TOKENS

        # Store tools in a dictionary for fast access by name
        self.tools: Dict[str, BaseTool] = {tool.name: tool for tool in tools} if tools else {}

        # Create a system prompt that includes tool descriptions
        self.system_prompt = self._construct_system_prompt(settings.SYSTEM_PROMPT)

        logger.info("🤖 OpenAIAgent initialized for model: %s", self.model)
        if self.tools:
            logger.info("   ... with tools: %s", list(self.tools.keys()))

    def _construct_system_prompt(self, base_prompt: str) -> str:
        """Dynamically builds the system prompt with a clear, structured order of instructions."""

        # Step 1: Define the base role (who are you?)
        prompt_parts = [base_prompt]

        # Step 2: Add context (what is the current situation?)
        current_date = datetime.now().strftime("%A, %d %B %Y")
        prompt_parts.append(f"Current date is {current_date}.")

        # Step 3: If there are no tools, we finish here.
        if not self.tools:
            return "\n\n".join(prompt_parts)

        # Step 4: Describe capabilities (what resources do you have?)
        tool_descriptions = "\n\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools.values()])
        prompt_parts.append(f"YOU HAVE ACCESS TO THE FOLLOWING TOOLS:\n{tool_descriptions}")

        # Step 5: Give a strict instruction (what are you required to do?)
        # Combine the rule and the format into one final block of instructions.
        final_instructions = (
            "Your internal knowledge is cut off in early 2024. "
            "You MUST use your tools for any questions about events, news, or specific facts from mid-2024 onwards. "
            "Do not answer from memory for recent topics.\n\n"
            "To use a tool, respond in the following JSON format inside <tool_call> tags. "
            "The tool's description specifies the exact argument names it expects.\n\n"
            "Example format for a tool call:\n"
            "<tool_call>\n"
            "{\n"
            '  "tool_name": "NameOfTheTool",\n'
            '  "arg_name": "value"\n'
            "}\n"
            "</tool_call>"
        )
        prompt_parts.append(final_instructions)

        # Assemble the final prompt
        return "\n\n".join(prompt_parts)

    async def chat(self, message: str, dialog_id: Optional[str] = None) -> str:
        """
        Sends a request to the LLM and returns the response.
        Implements a loop to handle tool calls.
        """
        if not dialog_id:
            dialog_id = "default"

        logger.info("Agent.chat called for dialog_id=%s with message: '%s'", dialog_id, message)
        history = await self.memory.get_history(dialog_id)

        # Messages for the LLM will be accumulated in this list during the loop
        messages_for_llm: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            *history,
            {"role": "user", "content": message},
        ]

        # "Thought -> Action -> Observation" cycle
        max_loops = 10  # Limit to avoid infinite loops
        for i in range(max_loops):
            logger.debug("🧠 Calling LLM (loop %d)...", i + 1)
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=cast(List[ChatCompletionMessageParam], messages_for_llm),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content or ""

                # Add the assistant's response to the history (even if it's a tool call)
                messages_for_llm.append({"role": "assistant", "content": content})

                # Search for a tool invocation in the LLM response
                tool_call_match = re.search(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)

                if tool_call_match:
                    tool_call_json_str = tool_call_match.group(1).strip()
                    logger.info("Tool call detected: %s", tool_call_json_str)

                    try:
                        tool_call_data = json.loads(tool_call_json_str)
                        tool_name = tool_call_data.pop("tool_name", None)

                        if tool_name in self.tools:
                            # Execute the tool
                            tool = self.tools[tool_name]
                            logger.info("Executing tool '%s' with args: %s", tool_name, tool_call_data)
                            tool_result = tool.run(**tool_call_data)
                            logger.info("Tool '%s' raw result:\n%s", tool_name, tool_result)

                            # Add the tool's result to the history for the next LLM step
                            tool_output_message = f"<tool_output>\n{tool_result}\n</tool_output>"
                            messages_for_llm.append({"role": "user", "content": tool_output_message})
                            continue  # Continue the loop with new information
                        else:
                            error_message = f"Error: Tool '{tool_name}' not found."
                            messages_for_llm.append({"role": "user", "content": error_message})

                    except json.JSONDecodeError:
                        error_message = "Error: Invalid JSON in tool_call."
                        messages_for_llm.append({"role": "user", "content": error_message})
                else:
                    # If there is no tool invocation, this is the final answer
                    logger.info("Final answer received from LLM.")
                    final_answer = content.strip()

                    # Only save the original user message and the final answer
                    await self.memory.append(dialog_id, "user", message)
                    await self.memory.append(dialog_id, "assistant", final_answer)

                    return final_answer

            except APIError as e:
                logger.error("OpenAI API error during LLM call for dialog_id=%s: %s", dialog_id, e)
                return f"An error occurred: {e}"

        final_answer = "Error: Agent reached maximum loop limit."
        await self.memory.append(dialog_id, "user", message)
        await self.memory.append(dialog_id, "assistant", final_answer)
        return final_answer
