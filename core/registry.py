# core/registry.py

from typing import Callable, Dict

from core.base_agent import BaseAgent

AGENT_REGISTRY: Dict[str, Callable[..., BaseAgent]] = {}


def register_agent(name: str, factory: Callable[..., BaseAgent]) -> None:
    """Register an agent factory by name."""
    AGENT_REGISTRY[name] = factory


def get_agent(name: str, **kwargs) -> BaseAgent:
    """Instantiate an agent by name."""
    factory = AGENT_REGISTRY.get(name)
    if factory is None:
        raise KeyError(f"Agent '{name}' is not registered")
    return factory(**kwargs)
