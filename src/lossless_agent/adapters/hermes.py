"""Hermes Memory Provider adapter for lossless-agent."""
from __future__ import annotations

from typing import Callable, Awaitable, List

from lossless_agent.adapters.base_impl import BaseAdapter

SummarizeFn = Callable[[str], Awaitable[str]]

_TOOL_SCHEMAS: List[dict] = [
    {
        "type": "function",
        "function": {
            "name": "lcm_grep",
            "description": (
                "Search messages and summaries by keyword. "
                "Returns matching snippets with IDs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "FTS5 search query.",
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["all", "messages", "summaries"],
                        "description": "What to search. Default: all.",
                    },
                    "conversation_id": {
                        "type": "integer",
                        "description": "Restrict to a specific conversation.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results. Default: 20.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lcm_describe",
            "description": (
                "Look up a summary node by ID and return its full metadata."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary_id": {
                        "type": "string",
                        "description": "The summary ID to describe.",
                    },
                },
                "required": ["summary_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lcm_expand",
            "description": (
                "Expand a summary: return source messages (leaf) or "
                "child summaries (condensed)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary_id": {
                        "type": "string",
                        "description": "The summary ID to expand.",
                    },
                },
                "required": ["summary_id"],
            },
        },
    },
]

_SYSTEM_PROMPT_BLOCK = """\
## Lossless Recall Policy

You have access to a long-term memory system. Old messages are compacted into \
summaries organised in a DAG (directed acyclic graph). Use the following tools \
to navigate it:

- **lcm_grep(query)** — full-text search across messages and summaries.
- **lcm_describe(summary_id)** — get metadata for a summary node.
- **lcm_expand(summary_id)** — drill into a summary to see its children.

When the user refers to past conversations or facts you don't see in the \
current context, call lcm_grep first. If a summary snippet looks relevant, \
use lcm_describe then lcm_expand to retrieve detail. Always cite the \
summary_id when referencing recalled information."""


class HermesAdapter(BaseAdapter):
    """Hermes Memory Provider: wraps the LCM system for the Hermes agent."""

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def get_tools(self) -> List[dict]:
        """Return OpenAI function-calling tool schemas."""
        return list(_TOOL_SCHEMAS)

    def get_system_prompt_block(self) -> str:
        """Return the Lossless Recall Policy text."""
        return _SYSTEM_PROMPT_BLOCK
