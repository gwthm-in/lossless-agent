"""Adapter factory for lossless-agent."""
from __future__ import annotations

from typing import Callable, Awaitable, Optional

from lossless_agent.adapters.base import AgentAdapter, LCMConfig

SummarizeFn = Callable[[str], Awaitable[str]]


def create_adapter(
    agent_type: str,
    config: Optional[LCMConfig] = None,
    summarize_fn: Optional[SummarizeFn] = None,
    *,
    db_path: Optional[str] = None,
) -> AgentAdapter:
    """Create an adapter instance by agent type name.

    Args:
        agent_type: One of 'hermes', 'openclaw', 'generic', or 'simple'.
        config: LCM configuration (required for hermes/openclaw/generic).
        summarize_fn: Async callable that summarizes text.
        db_path: Database path (only used for 'simple' when config is None).

    Returns:
        An AgentAdapter instance (or SimpleAdapter for 'simple').

    Raises:
        ValueError: If *agent_type* is not recognised or required args missing.
    """
    if agent_type == "hermes":
        from lossless_agent.adapters.hermes import HermesAdapter
        if config is None or summarize_fn is None:
            raise ValueError("hermes adapter requires config and summarize_fn")
        return HermesAdapter(config, summarize_fn)
    elif agent_type == "openclaw":
        from lossless_agent.adapters.openclaw import OpenClawAdapter
        if config is None or summarize_fn is None:
            raise ValueError("openclaw adapter requires config and summarize_fn")
        return OpenClawAdapter(config, summarize_fn)
    elif agent_type == "generic":
        from lossless_agent.adapters.generic import GenericAdapter
        if config is None or summarize_fn is None:
            raise ValueError("generic adapter requires config and summarize_fn")
        return GenericAdapter(config, summarize_fn)
    elif agent_type == "simple":
        from lossless_agent.adapters.simple import SimpleAdapter
        if summarize_fn is None:
            raise ValueError("simple adapter requires summarize_fn")
        resolved_db_path = db_path or (config.db_path if config else None)
        if resolved_db_path is None:
            raise ValueError(
                "simple adapter requires db_path or config with db_path"
            )
        return SimpleAdapter(resolved_db_path, summarize_fn, config)  # type: ignore[return-value]
    else:
        raise ValueError(
            f"Unknown agent type: {agent_type!r}. "
            f"Supported types: 'hermes', 'openclaw', 'generic', 'simple'."
        )
