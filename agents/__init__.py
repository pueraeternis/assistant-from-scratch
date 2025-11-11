# agents/__init__.py

from datetime import datetime
from typing import List, Optional

from config.settings import settings
from core.base_agent import BaseAgent
from core.base_tool import BaseTool
from core.memory import RedisMemory
from core.registry import get_agent, register_agent
from tools.delegate_task import DelegateTaskTool

from .echo.agent import EchoAgent
from .openai.agent import OpenAIAgent

# --- COMMON INSTRUCTION BLOCKS FOR SPECIALISTS ---

TOOL_CALL_INSTRUCTION = """
To use a tool, you MUST respond in the following JSON format inside <tool_call> tags:
<tool_call>
{
  "tool_name": "NameOfTheTool",
  "arg_name": "value"
}
</tool_call>
"""

DB_SCHEMA_DESCRIPTION = """
You have access to a SQLite database for company data with the following schema:

CREATE TABLE employees (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  position TEXT,
  department_id INTEGER,
  salary INTEGER,
  FOREIGN KEY (department_id) REFERENCES departments (id)
);

CREATE TABLE departments (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE
);
"""

SQL_FEW_SHOT_EXAMPLES = """
Here are examples of how to respond to user queries about the database:

**Example 1:**
User query: "Who are the employees in the Sales department?"
Your response:
<tool_call>
{
  "tool_name": "SQLQueryTool",
  "query": "SELECT e.name FROM employees e JOIN departments d ON e.department_id = d.id WHERE d.name = 'Sales';"
}
</tool_call>

**Example 2 (Crucial for calculations):**
User query: "What is the average salary?"
Your response:
<tool_call>
{
  "tool_name": "SQLQueryTool",
  "query": "SELECT AVG(salary) FROM employees;"
}
</tool_call>
"""
# ----------------------------------------------------


def openai_factory(
    tools: Optional[List[BaseTool]] = None,
    system_prompt: Optional[str] = None,
) -> OpenAIAgent:
    """Factory for creating OpenAIAgent. It constructs a default prompt if none is provided."""
    memory = RedisMemory(redis_url=settings.REDIS_URL)

    final_prompt = system_prompt

    if final_prompt is None:
        prompt_parts = [
            settings.SYSTEM_PROMPT,
            f"Current date is {datetime.now().strftime('%A, %d %B %Y')}.",
        ]
        if tools:
            tool_descriptions = "\n\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
            prompt_parts.append(f"YOU HAVE ACCESS TO THE FOLLOWING TOOLS:\n{tool_descriptions}")
            prompt_parts.append(TOOL_CALL_INSTRUCTION)

        final_prompt = "\n\n".join(prompt_parts)

    return OpenAIAgent(memory=memory, tools=tools, system_prompt=final_prompt)


# ==============================================================================
# AGENT ROLE REGISTRATION
# ==============================================================================

# 1. Simple Echo agent for testing
register_agent("echo", lambda: EchoAgent(prefix="Echo: "))


# 2. Main, general-purpose assistant
def assistant_factory(**kwargs) -> BaseAgent:
    """
    Factory for a general-purpose Assistant.
    Constructs a detailed prompt with all available tools and context.
    """
    tools = kwargs.get("tools", [])
    tool_names = [t.name for t in tools]

    prompt_parts = [
        settings.SYSTEM_PROMPT,
        f"Current date is {datetime.now().strftime('%A, %d %B %Y')}.",
    ]
    if tools:
        tool_descriptions = "\n\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
        prompt_parts.append(f"YOU HAVE ACCESS TO THE FOLLOWING TOOLS:\n{tool_descriptions}")

        if "SQLQueryTool" in tool_names:
            prompt_parts.append(DB_SCHEMA_DESCRIPTION)
            prompt_parts.append(SQL_FEW_SHOT_EXAMPLES)

        final_instructions = (
            "You MUST use your tools for any questions about recent events (from mid-2024 onwards).\n"
            "For ANY query about company data, you are FORBIDDEN from answering from memory and MUST use the SQLQueryTool.\n"
            "Follow the examples provided."
        )
        prompt_parts.append(final_instructions)
        prompt_parts.append(TOOL_CALL_INSTRUCTION)

    final_prompt = "\n\n".join(prompt_parts)

    return openai_factory(tools=tools, system_prompt=final_prompt)


register_agent("assistant", assistant_factory)


# 3. Specialist: Researcher (Internet)
def researcher_factory(**kwargs) -> BaseAgent:
    """Factory for the Researcher role."""
    researcher_prompt = (
        """You are a specialized researcher agent. Your ONLY function is to find information on the internet.

    Follow this algorithm strictly:
    1. When you receive a user query, your FIRST and ONLY initial action MUST BE to use the "InternetSearch" tool.
    2. Analyze the search results. If snippets are insufficient, your NEXT action MUST BE to use "BrowseWebpage".
    3. Formulate the final answer based ONLY on the information you have gathered.
    4. You are FORBIDDEN from answering from memory. If you cannot find an answer, state that.
    """
        + TOOL_CALL_INSTRUCTION
    )

    available_tools = kwargs.get("tools", [])
    researcher_tools = [t for t in available_tools if t.name in ("InternetSearch", "BrowseWebpage")]
    return openai_factory(tools=researcher_tools, system_prompt=researcher_prompt)


register_agent("researcher", researcher_factory)


# 4. Specialist: Knowledge Base Expert
def knowledge_expert_factory(**kwargs) -> BaseAgent:
    """Factory for the Knowledge Base Expert role."""
    expert_prompt = (
        "You are a world-class expert on AI research. "
        "Your ONLY function is to answer questions using the internal knowledge base of scientific papers. "
        "You MUST use the 'KnowledgeBaseSearch' tool for every query. "
        "Do not use any other tools or your internal memory."
    ) + TOOL_CALL_INSTRUCTION

    available_tools = kwargs.get("tools", [])
    expert_tools = [t for t in available_tools if t.name == "KnowledgeBaseSearch"]
    return openai_factory(tools=expert_tools, system_prompt=expert_prompt)


register_agent("knowledge_expert", knowledge_expert_factory)


# 5. Specialist: Database Analyst
def database_analyst_factory(**kwargs) -> BaseAgent:
    """Factory for the Database Analyst role."""
    analyst_prompt = (
        (
            "You are a senior database analyst. "
            "Your ONLY function is to answer questions about company data by writing and executing SQLite queries. "
            "You MUST use the 'SQLQueryTool' for every query. Do not use any other tools or your internal memory.\n\n"
        )
        + DB_SCHEMA_DESCRIPTION
        + TOOL_CALL_INSTRUCTION
    )

    available_tools = kwargs.get("tools", [])
    analyst_tools = [t for t in available_tools if t.name == "SQLQueryTool"]
    return openai_factory(tools=analyst_tools, system_prompt=analyst_prompt)


register_agent("database_analyst", database_analyst_factory)


# 6. Specialist: Writer/Editor
def writer_factory(**kwargs) -> BaseAgent:  # pylint: disable=unused-argument
    """Factory for the Writer role."""
    writer_prompt = (
        "You are a professional technical writer and editor. "
        "Your ONLY function is to take the provided text, facts, and data, and synthesize them into a single, "
        "well-structured, and easy-to-read final answer for the user. "
        "You are FORBIDDEN from seeking new information or using any tools. "
        "Your answer must be based solely on the context you are given."
    )
    return openai_factory(tools=[], system_prompt=writer_prompt)


register_agent("writer", writer_factory)


# 7. Orchestrator
def orchestrator_factory(**kwargs) -> BaseAgent:
    """Factory for the Orchestrator agent."""

    # Get the complete pool of tools provided by the CLI
    available_tools = kwargs.get("tools", [])

    # Create a description of ALL tools for context
    all_tools_description = "\n\n".join([f"- {tool.name}: {tool.description}" for tool in available_tools])

    orchestrator_prompt = f"""You are a master orchestrator agent. Your primary function is to decompose complex user queries into a sequence of smaller, manageable sub-tasks and delegate them to the appropriate specialist agent.

    First, here is a list of ALL tools available in the entire system, to give you context on what is possible:
    {all_tools_description}

    You have a team of specialists available. You must choose the right specialist for the right sub-task:
    - 'researcher': Uses 'InternetSearch' and 'BrowseWebpage'. Ideal for finding current, external information.
    - 'knowledge_expert': Uses 'KnowledgeBaseSearch'. Ideal for querying the internal knowledge base of scientific AI papers.
    - 'database_analyst': Uses 'SQLQueryTool'. Ideal for answering questions about internal company data.
    - 'writer': Has NO tools. Use this specialist LAST to formulate the final, polished answer for the user.

    Your workflow MUST be as follows:
    1.  Analyze the user's request and create a step-by-step plan.
    2.  For each step, use your ONLY tool, "DelegateTask", to assign the task to the most suitable specialist. You must specify their role and give them a clear task description.
    3.  Review the specialist's response. If more information is needed, delegate another task.
    4.  Once you have gathered all necessary information, your FINAL action is to delegate the complete set of information to the 'writer' agent to formulate the final response.
    5.  You are FORBIDDEN from answering the user directly. Your only output to the user should be the final, synthesized response from the 'writer' agent.

    {TOOL_CALL_INSTRUCTION}
    """

    delegation_tool = DelegateTaskTool(get_agent_func=get_agent, all_tools=available_tools)

    return openai_factory(tools=[delegation_tool], system_prompt=orchestrator_prompt)


register_agent("orchestrator", orchestrator_factory)
