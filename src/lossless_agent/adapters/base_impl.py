"""Concrete base adapter that wires all LCM features.

Subclasses only need to override ``get_tools()`` and
``get_system_prompt_block()`` for adapter-specific tool schemas
and prompt text.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Callable, Awaitable, List, Optional

from lossless_agent.adapters.base import AgentAdapter, LCMConfig
from lossless_agent.store import (
    Database, ConversationStore, ContextItemStore, MessageStore, SummaryStore,
)
from lossless_agent.engine import (
    CircuitBreaker,
    CompactionAwarePrompt,
    CompactionEngine,
    ContextAssembler,
    HeartbeatPruner,
    LargeFileInterceptor,
    SessionPatternMatcher,
    StartupBanner,
)
from lossless_agent.tools import lcm_grep, lcm_describe, lcm_expand

logger = logging.getLogger(__name__)

SummarizeFn = Callable[[str], Awaitable[str]]

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


class BaseAdapter(AgentAdapter):
    """Concrete base adapter that wires all standalone LCM modules.

    Subclasses typically only override ``get_tools()`` and
    ``get_system_prompt_block()``.
    """

    def __init__(self, config: LCMConfig, summarize_fn: SummarizeFn) -> None:
        self._config = config
        self._summarize_fn = summarize_fn

        # --- Core stores ---
        self._db = Database(config.db_path)
        self._conv_store = ConversationStore(self._db)
        self._msg_store = MessageStore(self._db)
        self._sum_store = SummaryStore(self._db)
        self._ctx_store = ContextItemStore(self._db)

        # --- Circuit breaker ---
        self._circuit_breaker = CircuitBreaker(
            threshold=config.circuit_breaker_threshold,
            cooldown_ms=config.circuit_breaker_cooldown_ms,
        )

        # --- Compaction engine (with circuit breaker + context item store) ---
        self._engine = CompactionEngine(
            self._msg_store,
            self._sum_store,
            summarize_fn,
            config.compaction,
            circuit_breaker=self._circuit_breaker,
            context_item_store=self._ctx_store,
        )

        # --- Assembler ---
        self._assembler = ContextAssembler(
            self._msg_store, self._sum_store, config.assembler
        )

        # --- Session pattern matcher ---
        self._session_matcher = SessionPatternMatcher(
            ignore_patterns=config.ignore_session_patterns,
            stateless_patterns=config.stateless_session_patterns,
        )

        # --- Heartbeat pruner ---
        self._heartbeat_pruner: Optional[HeartbeatPruner] = (
            HeartbeatPruner() if config.prune_heartbeat_ok else None
        )

        # --- Compaction-aware prompt ---
        self._compaction_prompt = CompactionAwarePrompt()

        # --- Track last session key for dynamic prompt generation ---
        self._last_session_key: Optional[str] = None

        # --- Track first-turn for startup banner ---
        self._banner_emitted = False

    # ------------------------------------------------------------------
    # Turn lifecycle (AgentAdapter ABC)
    # ------------------------------------------------------------------

    async def on_turn_start(
        self, session_key: str, user_message: str
    ) -> Optional[str]:
        """Get or create conversation, assemble context, return formatted string."""
        self._last_session_key = session_key

        # a. Check ignored sessions
        if self._session_matcher.is_ignored(session_key):
            return None

        # b. Check stateless sessions
        if (
            self._config.skip_stateless_sessions
            and self._session_matcher.is_stateless(session_key)
        ):
            # Read-only mode: assemble context but don't persist
            conv = self._conv_store.get_or_create(session_key)
            assembled = self._assembler.assemble(conv.id)
            if not assembled.summaries and not assembled.messages:
                return None
            context = self._assembler.format_context(assembled)
            return context if context.strip() else None

        conv = self._conv_store.get_or_create(session_key)

        # c. Run heartbeat pruner if configured
        if self._heartbeat_pruner is not None:
            messages = self._msg_store.get_messages(conv.id)
            if messages:
                HeartbeatPruner.prune(messages, self._msg_store)

        # d. Startup banner on first turn
        if not self._banner_emitted:
            StartupBanner.log_plugin_loaded(self._config)
            StartupBanner.log_compaction_model(self._config)
            StartupBanner.log_session_patterns(self._config)
            self._banner_emitted = True

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
        while True:
            result = await self._engine.compact_leaf(conv.id)
            if result is None:
                break
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

        Subclasses may override to add agent-specific metadata.
        """
        raise NotImplementedError("Subclasses must implement get_tools()")

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
            expand_result = lcm_expand(self._db, arguments["summary_id"], is_sub_agent=True)
            if expand_result is None:
                return json.dumps({"error": "summary not found"})
            return json.dumps(asdict(expand_result), default=str)
        else:
            return json.dumps({"error": f"unknown tool: {name}"})

    def get_system_prompt_block(self) -> str:
        """Return the Lossless Recall Policy text.

        Includes dynamic compaction-aware additions when heavy
        compaction is detected for the current session.
        """
        base = _SYSTEM_PROMPT_BLOCK

        if self._last_session_key is not None:
            try:
                conv = self._conv_store.get_or_create(self._last_session_key)
                summaries = self._sum_store.get_by_conversation(conv.id)
                addition = self._compaction_prompt.generate(summaries)
                if addition is not None:
                    base = base + "\n\n" + addition
            except Exception:
                logger.debug("Could not generate compaction-aware prompt", exc_info=True)

        return base
