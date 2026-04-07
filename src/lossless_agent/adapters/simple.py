"""Minimal adapter for lossless-agent — no lifecycle, no ABC.

SimpleAdapter is the 'I just want LCM in 5 lines of code' option.
It does not inherit from AgentAdapter and exposes only the essential
operations: ingest, retrieve, compact, search, expand, and close.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Awaitable, List, Optional

from lossless_agent.config import LCMConfig
from lossless_agent.store import Database, ConversationStore, MessageStore, SummaryStore
from lossless_agent.engine import CompactionEngine, ContextAssembler
from lossless_agent.tools import lcm_grep, lcm_expand

SummarizeFn = Callable[[str], Awaitable[str]]


class SimpleAdapter:
    """Minimal LCM adapter — no lifecycle hooks, just the essentials.

    Usage::

        adapter = SimpleAdapter("my.db", my_summarize_fn)
        await adapter.ingest("session-1", messages)
        await adapter.compact("session-1")
        context = await adapter.retrieve("session-1", budget_tokens=4000)
        results = await adapter.search("deployment")
        adapter.close()
    """

    def __init__(
        self,
        db_path: str,
        summarize_fn: SummarizeFn,
        config: Optional[LCMConfig] = None,
    ) -> None:
        if config is None:
            config = LCMConfig(db_path=db_path)
        else:
            config = LCMConfig.merge(config, {"db_path": db_path})

        self._config = config
        self._db_path = db_path
        self._summarize_fn = summarize_fn
        self._db = Database(db_path)
        self._conv_store = ConversationStore(self._db)
        self._msg_store = MessageStore(self._db)
        self._sum_store = SummaryStore(self._db)
        self._engine = CompactionEngine(
            self._msg_store, self._sum_store, summarize_fn, config.compaction
        )
        self._assembler = ContextAssembler(
            self._msg_store, self._sum_store, config.assembler
        )
        self._closed = False

    async def ingest(
        self, session_key: str, messages: List[dict]
    ) -> None:
        """Bulk store messages.

        Each dict should have 'role', 'content', and optionally 'token_count'.
        """
        conv = self._conv_store.get_or_create(session_key)
        for msg in messages:
            self._msg_store.append(
                conversation_id=conv.id,
                role=msg["role"],
                content=msg.get("content", ""),
                token_count=msg.get("token_count", len(msg.get("content", "").split())),
            )

    async def retrieve(
        self, session_key: str, budget_tokens: int
    ) -> Optional[str]:
        """Get assembled context within token budget.

        Returns formatted context string, or None if the session is empty.
        """
        conv = self._conv_store.get_or_create(session_key)
        # Create a temporary assembler config with the requested budget
        from lossless_agent.engine.assembler import AssemblerConfig
        assembler_cfg = AssemblerConfig(
            max_context_tokens=budget_tokens,
            summary_budget_ratio=self._config.summary_budget_ratio,
            fresh_tail_count=self._config.fresh_tail_count,
        )
        assembler = ContextAssembler(
            self._msg_store, self._sum_store, assembler_cfg
        )
        assembled = assembler.assemble(conv.id)
        if not assembled.summaries and not assembled.messages:
            return None
        context = assembler.format_context(assembled)
        return context if context.strip() else None

    async def compact(self, session_key: str) -> int:
        """Run compaction and return number of summaries created."""
        conv = self._conv_store.get_or_create(session_key)

        # Count summaries before
        before = len(self._sum_store.get_by_conversation(conv.id))

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

        # Count summaries after
        after = len(self._sum_store.get_by_conversation(conv.id))
        return after - before

    async def search(
        self, query: str, session_key: Optional[str] = None
    ) -> List[dict]:
        """Search across history, optionally scoped to a session.

        Returns a list of dicts with search results.
        """
        conversation_id = None
        if session_key is not None:
            conv = self._conv_store.get_or_create(session_key)
            conversation_id = conv.id

        results = lcm_grep(
            db=self._db,
            query=query,
            scope="all",
            conversation_id=conversation_id,
            limit=20,
        )
        return [asdict(r) for r in results]

    async def expand(self, summary_id: str) -> dict:
        """Expand a summary to its sources.

        Returns a dict with the expansion result, or an error dict.
        """
        result = lcm_expand(self._db, summary_id)
        if result is None:
            return {"error": "summary not found"}
        return asdict(result)

    def close(self) -> None:
        """Close the database connection."""
        if not self._closed:
            self._db.close()
            self._closed = True
