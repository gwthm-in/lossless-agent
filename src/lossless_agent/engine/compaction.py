"""Compaction engine: summarises old messages into a DAG of summaries."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Awaitable, List, Optional

from lossless_agent.store.abc import AbstractMessageStore, AbstractSummaryStore
from lossless_agent.store.models import Message, Summary


@dataclass
class CompactionConfig:
    """Tunables for the compaction engine."""

    fresh_tail_count: int = 8
    leaf_chunk_tokens: int = 20_000
    leaf_min_fanout: int = 4
    condensed_min_fanout: int = 3
    context_threshold: float = 0.75
    leaf_target_tokens: int = 1200
    condensed_target_tokens: int = 2000


SummarizeFn = Callable[[str], Awaitable[str]]


def _format_messages(messages: List[Message]) -> str:
    """Format messages into a single text block for the summariser."""
    parts = []
    for m in messages:
        parts.append(f"[{m.role}] {m.content}")
    return "\n".join(parts)


def _format_summaries(summaries: List[Summary]) -> str:
    """Format summaries into a single text block for the summariser."""
    parts = []
    for s in summaries:
        parts.append(f"[summary depth={s.depth}] {s.content}")
    return "\n".join(parts)


class CompactionEngine:
    """Drives leaf and condensed compaction passes over a conversation."""

    def __init__(
        self,
        msg_store: AbstractMessageStore,
        sum_store: AbstractSummaryStore,
        summarize_fn: SummarizeFn,
        config: CompactionConfig | None = None,
    ) -> None:
        self._msg = msg_store
        self._sum = sum_store
        self._summarize = summarize_fn
        self.cfg = config or CompactionConfig()

    # ------------------------------------------------------------------
    # Chunk selection
    # ------------------------------------------------------------------

    def select_chunk(self, conv_id: int) -> List[Message]:
        """Pick the oldest uncompacted messages (excluding the fresh tail).

        Returns up to ``leaf_chunk_tokens`` worth of messages, ensuring at
        least ``leaf_min_fanout`` are available.  Returns [] when nothing
        qualifies.
        """
        all_msgs = self._msg.get_messages(conv_id)
        # Protect the most recent messages
        if len(all_msgs) <= self.cfg.fresh_tail_count:
            return []
        eligible = all_msgs[: -self.cfg.fresh_tail_count]

        # Exclude already-compacted messages
        compacted_ids = set(self._sum.get_compacted_message_ids(conv_id))
        uncompacted = [m for m in eligible if m.id not in compacted_ids]

        if len(uncompacted) < self.cfg.leaf_min_fanout:
            return []

        # Take oldest chunk up to token limit
        chunk: List[Message] = []
        total_tokens = 0
        for m in uncompacted:
            if total_tokens + m.token_count > self.cfg.leaf_chunk_tokens and chunk:
                break
            chunk.append(m)
            total_tokens += m.token_count

        if len(chunk) < self.cfg.leaf_min_fanout:
            return []

        return chunk

    # ------------------------------------------------------------------
    # Needs-compaction check
    # ------------------------------------------------------------------

    def needs_compaction(self, conv_id: int, context_limit: int) -> bool:
        """Return True when total conversation tokens exceed the threshold."""
        total = self._msg.total_tokens(conv_id)
        return total > self.cfg.context_threshold * context_limit

    # ------------------------------------------------------------------
    # Leaf compaction
    # ------------------------------------------------------------------

    async def compact_leaf(self, conv_id: int) -> Optional[Summary]:
        """Run one leaf compaction pass: summarise a chunk of messages."""
        chunk = self.select_chunk(conv_id)
        if not chunk:
            return None

        text = _format_messages(chunk)
        summary_text = await self._summarize(text)

        source_tokens = sum(m.token_count for m in chunk)
        msg_ids = [m.id for m in chunk]
        token_count = len(summary_text.split())  # rough estimate

        summary = self._sum.create_leaf(
            conversation_id=conv_id,
            content=summary_text,
            token_count=token_count,
            source_token_count=source_tokens,
            message_ids=msg_ids,
            earliest_at=chunk[0].created_at,
            latest_at=chunk[-1].created_at,
            model="compaction",
        )
        return summary

    # ------------------------------------------------------------------
    # Condensed compaction
    # ------------------------------------------------------------------

    async def compact_condensed(
        self, conv_id: int, depth: int = 0
    ) -> Optional[Summary]:
        """Merge orphan summaries at *depth* into one condensed node."""
        orphan_ids = self._sum.get_orphan_ids(conv_id, depth)
        if len(orphan_ids) < self.cfg.condensed_min_fanout:
            return None

        # Fetch full summary objects for formatting
        all_at_depth = self._sum.get_by_depth(conv_id, depth)
        orphan_set = set(orphan_ids)
        orphans = [s for s in all_at_depth if s.summary_id in orphan_set]

        text = _format_summaries(orphans)
        summary_text = await self._summarize(text)
        token_count = len(summary_text.split())

        summary = self._sum.create_condensed(
            conversation_id=conv_id,
            content=summary_text,
            token_count=token_count,
            child_ids=orphan_ids,
            earliest_at=orphans[0].earliest_at,
            latest_at=orphans[-1].latest_at,
            model="compaction",
        )
        return summary

    # ------------------------------------------------------------------
    # Incremental compaction
    # ------------------------------------------------------------------

    async def run_incremental(
        self, conv_id: int, context_limit: int
    ) -> List[Summary]:
        """Run one incremental compaction cycle (leaf then condensed)."""
        created: List[Summary] = []

        if not self.needs_compaction(conv_id, context_limit):
            return created

        leaf = await self.compact_leaf(conv_id)
        if leaf is not None:
            created.append(leaf)

        condensed = await self.compact_condensed(conv_id, depth=0)
        if condensed is not None:
            created.append(condensed)

        return created
