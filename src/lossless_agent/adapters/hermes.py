"""Hermes Memory Provider adapter for lossless-agent."""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Callable, Awaitable, List, Optional

from lossless_agent.adapters.base import AgentAdapter, LCMConfig
from lossless_agent.store import Database, ConversationStore, MessageStore, SummaryStore
from lossless_agent.engine import CompactionEngine, ContextAssembler
from lossless_agent.tools import lcm_grep, lcm_describe, lcm_expand

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


class HermesAdapter(AgentAdapter):
    """Hermes Memory Provider: wraps the LCM system for the Hermes agent."""

    def __init__(self, config: LCMConfig, summarize_fn: SummarizeFn) -> None:
        self._config = config
        self._summarize_fn = summarize_fn
        self._db = Database(config.db_path)
        self._conv_store = ConversationStore(self._db)
        self._msg_store = MessageStore(self._db)
        self._sum_store = SummaryStore(self._db)
        self._engine = CompactionEngine(
            self._msg_store, self._sum_store, summarize_fn, config.compaction
        )
        self._assembler = ContextAssembler(
            self._msg_store, self._sum_store, config.assembler
        )

    # ------------------------------------------------------------------
    # Turn lifecycle
    # ------------------------------------------------------------------

    async def on_turn_start(
        self, session_key: str, user_message: str
    ) -> Optional[str]:
        """Get or create conversation, assemble context, return formatted string."""
        conv = self._conv_store.get_or_create(session_key)
        assembled = self._assembler.assemble(conv.id)
        if not assembled.summaries and not assembled.messages:
            return None
        context = self._assembler.format_context(assembled)
        return context if context.strip() else None

    async def on_turn_end(
        self, session_key: str, messages: List[dict]
    ) -> None:
        """Persist new messages and run incremental compaction."""
        conv = self._conv_store.get_or_create(session_key)
        for msg in messages:
            self._msg_store.append(
                conversation_id=conv.id,
                role=msg["role"],
                content=msg.get("content", ""),
                token_count=msg.get("token_count", len(msg.get("content", "").split())),
            )
        await self._engine.run_incremental(
            conv.id, self._config.assembler.max_context_tokens
        )

    async def on_session_end(self, session_key: str) -> None:
        """Run final compaction passes until nothing remains."""
        conv = self._conv_store.get_or_create(session_key)
        # Run leaf compaction until exhausted
        while True:
            result = await self._engine.compact_leaf(conv.id)
            if result is None:
                break
        # Run condensed compaction at each depth
        depth = 0
        while True:
            result = await self._engine.compact_condensed(conv.id, depth)
            if result is None:
                break
            depth += 1

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def get_tools(self) -> List[dict]:
        """Return OpenAI function-calling tool schemas."""
        return list(_TOOL_SCHEMAS)

    async def handle_tool_call(self, name: str, arguments: dict) -> str:
        """Dispatch a tool call and return JSON result."""
        if name == "lcm_grep":
            results = lcm_grep(
                db=self._db,
                query=arguments["query"],
                scope=arguments.get("scope", "all"),
                conversation_id=arguments.get("conversation_id"),
                limit=arguments.get("limit", 20),
            )
            return json.dumps(
                [asdict(r) for r in results],
                default=str,
            )
        elif name == "lcm_describe":
            result = lcm_describe(self._db, arguments["summary_id"])
            if result is None:
                return json.dumps({"error": "summary not found"})
            return json.dumps(asdict(result), default=str)
        elif name == "lcm_expand":
            result = lcm_expand(self._db, arguments["summary_id"])
            if result is None:
                return json.dumps({"error": "summary not found"})
            return json.dumps(asdict(result), default=str)
        else:
            return json.dumps({"error": f"unknown tool: {name}"})

    def get_system_prompt_block(self) -> str:
        """Return the Lossless Recall Policy text."""
        return _SYSTEM_PROMPT_BLOCK
