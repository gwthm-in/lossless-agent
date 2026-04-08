"""OpenClaw adapter stub for lossless-agent."""
from __future__ import annotations

import copy
from typing import Callable, Awaitable, List

from lossless_agent.adapters.base_impl import BaseAdapter

SummarizeFn = Callable[[str], Awaitable[str]]

_OPENCLAW_METADATA = {"plugin_name": "lossless-claw"}

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
        "openclaw_metadata": _OPENCLAW_METADATA,
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
        "openclaw_metadata": _OPENCLAW_METADATA,
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
        "openclaw_metadata": _OPENCLAW_METADATA,
    },
]

_SYSTEM_PROMPT_BLOCK = """\
## OpenClaw Lossless Recall Policy

You are running inside an OpenClaw agent. Long-term memory is provided by the \
lossless-claw plugin. Old messages are compacted into summaries organised in a \
DAG (directed acyclic graph). Use the following tools to navigate it:

- **lcm_grep(query)** — full-text search across messages and summaries.
- **lcm_describe(summary_id)** — get metadata for a summary node.
- **lcm_expand(summary_id)** — drill into a summary to see its children.

OpenClaw conventions: always prefer lcm_grep before generating answers that \
reference past sessions. Cite summary_id when referencing recalled information. \
Tool results are returned as JSON and should be parsed before presenting to \
the user."""


class OpenClawAdapter(BaseAdapter):
    """OpenClaw adapter: wraps the LCM system for OpenClaw agents."""

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def get_tools(self) -> List[dict]:
        """Return OpenAI function-calling tool schemas with openclaw_metadata."""
        return copy.deepcopy(_TOOL_SCHEMAS)

    def get_system_prompt_block(self) -> str:
        """Return the OpenClaw Lossless Recall Policy text."""
        return _SYSTEM_PROMPT_BLOCK
