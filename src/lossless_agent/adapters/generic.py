"""Generic agent-agnostic adapter for lossless-agent.

This adapter makes no assumptions about the host agent framework.
Any agent can use it directly via the AgentAdapter lifecycle, or
through the convenience methods (store_message, get_context, etc.).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Callable, Awaitable, Dict, List, Optional

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


class GenericAdapter(AgentAdapter):
    """Agent-agnostic adapter: wraps the LCM system for any agent framework.

    This adapter implements the full AgentAdapter lifecycle with no
    agent-specific assumptions, plus convenience methods for direct use.
    """

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
    # Turn lifecycle (AgentAdapter ABC)
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
    # Tools (AgentAdapter ABC)
    # ------------------------------------------------------------------

    def get_tools(self) -> List[dict]:
        """Return standard OpenAI function-calling tool schemas.

        No agent-specific metadata is included.
        """
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
            expand_result = lcm_expand(self._db, arguments["summary_id"])
            if expand_result is None:
                return json.dumps({"error": "summary not found"})
            return json.dumps(asdict(expand_result), default=str)
        else:
            return json.dumps({"error": f"unknown tool: {name}"})

    def get_system_prompt_block(self) -> str:
        """Return the Lossless Recall Policy text (agent-agnostic)."""
        return _SYSTEM_PROMPT_BLOCK

    # ------------------------------------------------------------------
    # Convenience methods (NOT in the ABC)
    # ------------------------------------------------------------------

    async def store_message(
        self,
        session_key: str,
        role: str,
        content: str,
        token_count: int = 0,
        tool_call_id: Optional[str] = None,
        tool_name: Optional[str] = None,
    ) -> None:
        """Manually persist a single message."""
        conv = self._conv_store.get_or_create(session_key)
        self._msg_store.append(
            conversation_id=conv.id,
            role=role,
            content=content,
            token_count=token_count,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )

    async def get_context(
        self, session_key: str, max_tokens: int
    ) -> Optional[str]:
        """One-call context retrieval: assemble and format.

        Returns formatted context string, or None if the session is empty.
        """
        conv = self._conv_store.get_or_create(session_key)
        assembled = self._assembler.assemble(conv.id)
        if not assembled.summaries and not assembled.messages:
            return None
        context = self._assembler.format_context(assembled)
        return context if context.strip() else None

    async def force_compact(self, session_key: str) -> None:
        """Manually trigger compaction for a session."""
        conv = self._conv_store.get_or_create(session_key)
        # Run leaf compaction until exhausted
        while True:
            result = await self._engine.compact_leaf(conv.id)
            if result is None:
                break
        # Run one pass of condensed compaction
        depth = 0
        while True:
            result = await self._engine.compact_condensed(conv.id, depth)
            if result is None:
                break
            depth += 1

    async def get_stats(self, session_key: str) -> dict:
        """Return stats for a session.

        Returns a dict with:
        - message_count: number of messages
        - summary_counts_by_depth: dict of depth -> count
        - total_tokens: sum of all message token counts
        - db_size: database file size in bytes (0 for :memory:)
        """
        conv = self._conv_store.get_or_create(session_key)
        message_count = self._msg_store.count(conv.id)
        total_tokens = self._msg_store.total_tokens(conv.id)

        # Build summary counts by depth
        summary_counts: Dict[int, int] = {}
        all_summaries = self._sum_store.get_by_conversation(conv.id)
        for s in all_summaries:
            depth = s.depth
            summary_counts[depth] = summary_counts.get(depth, 0) + 1

        # Database file size
        db_size = 0
        if self._config.db_path != ":memory:":
            try:
                db_size = os.path.getsize(self._config.db_path)
            except OSError:
                db_size = 0

        return {
            "message_count": message_count,
            "summary_counts_by_depth": summary_counts,
            "total_tokens": total_tokens,
            "db_size": db_size,
        }
