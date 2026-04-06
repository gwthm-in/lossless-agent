"""Adapter factory for lossless-agent."""
from __future__ import annotations

from typing import Callable, Awaitable

from lossless_agent.adapters.base import AgentAdapter, LCMConfig

SummarizeFn = Callable[[str], Awaitable[str]]


def create_adapter(
    agent_type: str, config: LCMConfig, summarize_fn: SummarizeFn
) -> AgentAdapter:
    """Create an adapter instance by agent type name.

    Args:
        agent_type: One of 'hermes' or 'openclaw'.
        config: LCM configuration.
        summarize_fn: Async callable that summarizes text.

    Returns:
        An AgentAdapter instance.

    Raises:
        ValueError: If *agent_type* is not recognised.
    """
    if agent_type == "hermes":
        from lossless_agent.adapters.hermes import HermesAdapter
        return HermesAdapter(config, summarize_fn)
    elif agent_type == "openclaw":
        from lossless_agent.adapters.openclaw import OpenClawAdapter
        return OpenClawAdapter(config, summarize_fn)
    else:
        raise ValueError(
            f"Unknown agent type: {agent_type!r}. "
            f"Supported types: 'hermes', 'openclaw'."
        )
