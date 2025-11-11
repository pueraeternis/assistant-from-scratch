# agents/__init__.py

from typing import List, Optional

from config.settings import settings
from core.base_agent import BaseAgent
from core.base_tool import BaseTool
from core.memory import RedisMemory
from core.registry import register_agent

from .echo.agent import EchoAgent
from .openai.agent import OpenAIAgent


def openai_factory(
    tools: Optional[List[BaseTool]] = None,
    system_prompt: Optional[str] = None,
) -> OpenAIAgent:
    """Factory for creating OpenAIAgent with custom tools and system prompt."""
    memory = RedisMemory(redis_url=settings.REDIS_URL)
    agent_instance = OpenAIAgent(memory=memory, tools=tools)
    if system_prompt:
        agent_instance.system_prompt = system_prompt

    return agent_instance


# ==============================================================================
# AGENT ROLE REGISTRATION
# ==============================================================================

# 1. Simple Echo agent for testing
register_agent("echo", lambda: EchoAgent(prefix="Echo: "))


# 2. Main, general-purpose assistant
def assistant_factory(**kwargs) -> BaseAgent:
    """Factory for creating a general-purpose Assistant agent with all available tools."""
    return openai_factory(tools=kwargs.get("tools"))


register_agent("assistant", assistant_factory)


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
# ----------------------------------------------------


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
