import asyncio
from typing import List

import click

import agents  # noqa: F401 # pylint: disable=unused-import
from core.base_tool import BaseTool
from core.logging import get_logger, setup_logging
from core.registry import AGENT_REGISTRY, get_agent
from tools.browse import BrowseTool
from tools.internet_search import InternetSearchTool

# configure logging
setup_logging()
logger = get_logger(__name__)

AVAILABLE_AGENTS = list(AGENT_REGISTRY.keys())
AVAILABLE_TOOLS: List[BaseTool] = [InternetSearchTool(), BrowseTool()]


@click.group()
def cli() -> None:
    """Assistant CLI entrypoint."""


@cli.command()
@click.option("--agent", default="openai_sdk", type=click.Choice(AVAILABLE_AGENTS), help="Agent to use")
@click.option("--dialog-id", default="default", help="Dialog/session id")
def chat(agent: str, dialog_id: str) -> None:
    """Start interactive chat loop with the selected agent."""
    asyncio.run(_chat_loop(agent_name=agent, dialog_id=dialog_id))


async def _chat_loop(agent_name: str, dialog_id: str) -> None:
    """Interactive loop. Type exit/quit to finish."""
    try:
        agent_obj = get_agent(agent_name, tools=AVAILABLE_TOOLS)
    except KeyError as exc:
        logger.error("Agent not found: %s", exc)
        raise SystemExit(1) from exc

    while True:
        user_prompt = click.style("User> ", fg="blue", bold=True)
        user_text = input(user_prompt).strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        response = await agent_obj.chat(message=user_text, dialog_id=dialog_id)

        assistant_prompt = click.style("Assistant> ", fg="green", bold=True)
        print(f"{assistant_prompt}{response}")


if __name__ == "__main__":
    cli()
