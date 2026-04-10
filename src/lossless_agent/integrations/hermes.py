"""Hermes Memory Provider integration for lossless-agent.

Drop-in replacement for Hermes's lossy ContextCompressor. One config line:

    memory_provider: lossless

Or programmatic setup:

    from lossless_agent.integrations.hermes import LosslessMemoryProvider

    provider = LosslessMemoryProvider(
        db_path="~/.hermes/lossless.db",
        summarize_fn=my_summarize,
    )

Lifecycle hooks map to Hermes MemoryProvider ABC:
    initialize()       -> create DB, stores, engine
    prefetch(query)    -> assemble DAG context for injection into user message
    sync_turn(msgs)    -> persist messages + incremental compaction
    on_pre_compress()  -> intercept lossy compression, run LCM compaction instead
    on_session_end()   -> final compaction sweep
    get_tools()        -> return lcm_grep/describe/expand tool schemas
    handle_tool_call() -> dispatch tool calls

IMPORTANT: All context is injected via user message (not system prompt)
to preserve Anthropic prompt caching (~75% cost savings).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from typing import Any, Callable, Awaitable, Dict, List, Optional

from lossless_agent.config import LCMConfig
from lossless_agent.store import (
    Database, ConversationStore, ContextItemStore, MessageStore, SummaryStore,
    create_database,
)
from lossless_agent.engine import (
    CircuitBreaker,
    CompactionAwarePrompt,
    CompactionEngine,
    ContextAssembler,
    HeartbeatPruner,
    SessionPatternMatcher,
    StartupBanner,
)
from lossless_agent.engine.embedder import EmbedFn, make_embedder
from lossless_agent.tools import lcm_grep, lcm_describe, lcm_expand

logger = logging.getLogger(__name__)

SummarizeFn = Callable[[str], Awaitable[str]]


# -- Tool schemas (OpenAI function-calling format) --

_TOOL_SCHEMAS: List[dict] = [
    {
        "type": "function",
        "function": {
            "name": "lcm_grep",
            "description": (
                "Search the full conversation history by keyword. "
                "Finds messages and summaries even from compacted context. "
                "Returns matching snippets with IDs for follow-up."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (FTS5 full-text or regex).",
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["all", "messages", "summaries"],
                        "description": "What to search. Default: all.",
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
                "Look up a summary node by ID. Returns metadata: "
                "kind, depth, token count, time range, child IDs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary_id": {
                        "type": "string",
                        "description": "The summary ID (e.g. sum_abc123def456).",
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
                "Drill into a summary to recover original detail. "
                "Leaf summaries expand to source messages. "
                "Condensed summaries expand to child summaries."
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

_RECALL_INSTRUCTIONS = """\
You have lossless memory. Old messages are compacted into summaries but \
NEVER lost. When you need past context:
1. lcm_grep(query) — search all history
2. lcm_describe(summary_id) — inspect a summary
3. lcm_expand(summary_id) — recover original messages
Cite summary_id when referencing recalled information."""


class LosslessMemoryProvider:
    """Hermes Memory Provider that replaces lossy compression with LCM.

    Designed to implement the Hermes MemoryProvider ABC. If Hermes isn't
    installed, this class still works standalone — it just provides the
    same lifecycle interface.

    Usage with Hermes::

        # In your Hermes config (config.yaml):
        memory_provider: lossless
        memory_provider_config:
          db_path: ~/.hermes/lossless.db

    Usage standalone::

        provider = LosslessMemoryProvider(
            db_path="memory.db",
            summarize_fn=my_summarize,
        )
        await provider.initialize()
        context = await provider.prefetch("session-1", "user query")
        await provider.sync_turn("session-1", messages)
    """

    def __init__(
        self,
        summarize_fn: SummarizeFn,
        db_path: str = "~/.hermes/lossless.db",
        config: Optional[LCMConfig] = None,
    ) -> None:
        self._summarize_fn = summarize_fn
        self._db_path = os.path.expanduser(db_path)

        if config is None:
            config = LCMConfig.from_env()
            config = LCMConfig.merge(config, {"db_path": self._db_path})
        self._config = config

        # Initialized in initialize()
        self._db: Optional[Database] = None
        self._conv_store: Optional[ConversationStore] = None
        self._msg_store: Optional[MessageStore] = None
        self._sum_store: Optional[SummaryStore] = None
        self._ctx_store: Optional[ContextItemStore] = None
        self._engine: Optional[CompactionEngine] = None
        self._assembler: Optional[ContextAssembler] = None
        self._session_matcher: Optional[SessionPatternMatcher] = None
        self._heartbeat_pruner: Optional[HeartbeatPruner] = None
        self._compaction_prompt = CompactionAwarePrompt()
        self._embed_fn: Optional[EmbedFn] = None
        self._vector_store = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Hermes MemoryProvider lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Set up database, stores, and engines.

        Called once when Hermes starts. Safe to call multiple times.
        """
        if self._initialized:
            return

        if not self._config.database_dsn:
            db_dir = os.path.dirname(self._db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
        self._db = create_database(self._config)
        self._conv_store = ConversationStore(self._db)
        self._msg_store = MessageStore(self._db)
        self._sum_store = SummaryStore(self._db)
        self._ctx_store = ContextItemStore(self._db)

        cb = CircuitBreaker(
            threshold=self._config.circuit_breaker_threshold,
            cooldown_ms=self._config.circuit_breaker_cooldown_ms,
        )

        # Semantic layer (optional)
        self._embed_fn = make_embedder(self._config)
        self._vector_store = None
        if self._embed_fn is not None and self._config.database_dsn:
            try:
                from lossless_agent.store.vector_store import VectorStore
                self._vector_store = VectorStore(
                    self._config.database_dsn, dim=self._config.embedding_dim
                )
            except Exception:
                logger.warning(
                    "Failed to initialise VectorStore — cross-session disabled",
                    exc_info=True,
                )
                self._embed_fn = None

        self._engine = CompactionEngine(
            self._msg_store,
            self._sum_store,
            self._summarize_fn,
            self._config.compaction,
            circuit_breaker=cb,
            context_item_store=self._ctx_store,
            embed_fn=self._embed_fn,
            vector_store=self._vector_store,
        )

        self._assembler = ContextAssembler(
            self._msg_store, self._sum_store, self._config.assembler,
        )

        self._session_matcher = SessionPatternMatcher(
            ignore_patterns=self._config.ignore_session_patterns,
            stateless_patterns=self._config.stateless_session_patterns,
        )

        if self._config.prune_heartbeat_ok:
            self._heartbeat_pruner = HeartbeatPruner()

        StartupBanner.log_plugin_loaded(self._config)
        StartupBanner.log_compaction_model(self._config)

        self._initialized = True
        logger.info("LosslessMemoryProvider initialized (db=%s)", self._db_path)

    def system_prompt_block(self) -> str:
        """Return recall instructions for the system prompt.

        NOTE: Keep this MINIMAL and STATIC. For Anthropic prompt caching,
        the system prompt must not change between turns. Dynamic context
        goes through prefetch() -> user message injection instead.
        """
        return _RECALL_INSTRUCTIONS

    async def prefetch(self, session_key: str, query: str = "") -> Optional[str]:
        """Assemble DAG context for injection into the user message.

        Called before each LLM call. Returns a context string to be
        injected into the user message (NOT system prompt, for caching).

        Returns None if no context is available.
        """
        if not self._initialized:
            await self.initialize()

        if self._session_matcher and self._session_matcher.is_ignored(session_key):
            return None

        conv = self._conv_store.get_or_create(session_key)
        assembled = self._assembler.assemble(conv.id, prompt=query)

        in_session = ""
        if assembled.summaries or assembled.messages:
            in_session = self._assembler.format_context(assembled)

        # Cross-session semantic search
        cross_session = ""
        if self._embed_fn is not None and self._vector_store is not None and query:
            try:
                embedding = await self._embed_fn(query)
                cross_session = await self._assembler.cross_session_context(
                    embedding,
                    conv.id,
                    self._vector_store,
                    top_k=self._config.cross_session_top_k,
                    token_budget=self._config.cross_session_token_budget,
                    min_score=self._config.cross_session_min_score,
                )
            except Exception:
                logger.warning("Cross-session search failed", exc_info=True)

        # Add dynamic compaction warning if heavily compacted
        summaries = self._sum_store.get_by_conversation(conv.id)
        compaction_note = self._compaction_prompt.generate(summaries)

        parts = [p for p in (compaction_note or "", cross_session, in_session) if p.strip()]
        if not parts:
            return None
        return "\n\n".join(parts)

    async def sync_turn(
        self,
        session_key: str,
        messages: List[Dict[str, Any]],
    ) -> None:
        """Persist messages and run incremental compaction.

        Called after each LLM turn with the new messages from that turn.
        """
        if not self._initialized:
            await self.initialize()

        conv = self._conv_store.get_or_create(session_key)

        for msg in messages:
            self._msg_store.append(
                conversation_id=conv.id,
                role=msg["role"],
                content=msg.get("content", ""),
                token_count=msg.get("token_count", len(msg.get("content", "")) // 4),
                tool_call_id=msg.get("tool_call_id"),
                tool_name=msg.get("tool_name"),
            )

        # Run incremental compaction
        await self._engine.run_incremental(
            conv.id, self._config.max_context_tokens,
        )

    async def on_pre_compress(self, session_key: str) -> bool:
        """Intercept Hermes's lossy ContextCompressor.

        Called when Hermes would normally run its lossy compression.
        Returns True to SKIP the default compression (LCM handles it).

        This is the key integration point — it replaces lossy compression
        with DAG-based lossless compaction.
        """
        if not self._initialized:
            await self.initialize()

        conv = self._conv_store.get_or_create(session_key)

        # Run aggressive compaction (compact_until_under)
        await self._engine.compact_until_under(
            conv.id, self._config.max_context_tokens,
        )

        logger.info(
            "LCM compaction completed for session %s (skipping lossy compression)",
            session_key,
        )

        return True  # Tell Hermes to SKIP its default lossy compression

    async def on_session_end(self, session_key: str) -> None:
        """Final compaction sweep when session ends."""
        if not self._initialized:
            return

        conv = self._conv_store.get_or_create(session_key)
        await self._engine.compact_full_sweep(conv.id)

    # ------------------------------------------------------------------
    # Tool support
    # ------------------------------------------------------------------

    def get_tools(self) -> List[dict]:
        """Return LCM recall tool schemas (OpenAI function-calling format)."""
        return list(_TOOL_SCHEMAS)

    async def handle_tool_call(self, name: str, arguments: dict) -> str:
        """Dispatch an LCM tool call and return JSON result."""
        if not self._initialized:
            await self.initialize()

        if name == "lcm_grep":
            results = lcm_grep(
                db=self._db,
                query=arguments["query"],
                scope=arguments.get("scope", "all"),
                limit=arguments.get("limit", 20),
            )
            return json.dumps([asdict(r) for r in results], default=str)

        elif name == "lcm_describe":
            result = lcm_describe(self._db, arguments["summary_id"])
            if result is None:
                return json.dumps({"error": "summary not found"})
            return json.dumps(asdict(result), default=str)

        elif name == "lcm_expand":
            result = lcm_expand(
                self._db, arguments["summary_id"], is_sub_agent=True,
            )
            if result is None:
                return json.dumps({"error": "summary not found"})
            return json.dumps(asdict(result), default=str)

        return json.dumps({"error": f"unknown tool: {name}"})

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Close database connections."""
        if self._vector_store is not None:
            self._vector_store.close()
            self._vector_store = None
        if self._db is not None:
            self._db.close()
            self._db = None
        self._initialized = False
