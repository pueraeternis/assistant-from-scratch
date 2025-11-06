import asyncio
import logging

import click

from core.logging import get_logger, setup_logging
from core.memory import InMemoryMemory
from core.registry import get_agent, register_agent
from frameworks.echo.agent import EchoAgent

# configure logging
setup_logging()
logger = get_logger(__name__)

# register the simple echo agent
register_agent("echo", lambda prefix="Echo: ": EchoAgent(prefix=prefix))


@click.group()
def cli() -> None:
    """Assistant CLI entrypoint."""


@cli.command()
@click.option("--agent", default="echo", help="Agent to use (registered name)")
@click.option("--dialog-id", default="default", help="Dialog/session id")
@click.option("--prefix", default="Echo: ", help="EchoAgent prefix (for the echo agent)")
def chat(agent: str, dialog_id: str, prefix: str) -> None:
    """Start interactive chat loop with the selected agent."""
    # Suppress INFO logs during chat for cleaner output
    logging.getLogger().setLevel(logging.WARNING)
    asyncio.run(_chat_loop(agent_name=agent, dialog_id=dialog_id, prefix=prefix))


async def _chat_loop(agent_name: str, dialog_id: str, prefix: str) -> None:
    """Interactive loop. Type exit/quit to finish."""
    # instantiate agent from registry
    try:
        agent_obj = get_agent(agent_name, prefix=prefix)
    except KeyError as exc:
        logger.error("Agent not found: %s", exc)
        raise SystemExit(1) from exc

    memory = InMemoryMemory()
    print(f"Interactive chat with agent='{agent_name}', dialog_id='{dialog_id}'. Type 'exit' to quit.")

    while True:
        # use builtin input to keep it simple and synchronous for CLI
        user_prompt = click.style("User> ", fg="blue", bold=True)
        user_text = input(user_prompt).strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        # append user message to memory
        await memory.append(dialog_id, role="user", text=user_text)

        # get agent response
        response = await agent_obj.chat(user_text, dialog_id=dialog_id)

        # append agent response to memory
        await memory.append(dialog_id, role="assistant", text=response)

        assistant_prompt = click.style("Assistant> ", fg="green", bold=True)
        print(f"{assistant_prompt}{response}")


if __name__ == "__main__":
    cli()
