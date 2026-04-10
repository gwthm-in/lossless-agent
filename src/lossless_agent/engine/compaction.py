"""Compaction engine: summarises old messages into a DAG of summaries."""
from __future__ import annotations

import asyncio
import enum
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Awaitable, List, Optional

if TYPE_CHECKING:
    from lossless_agent.store.vector_store import VectorStore
    from lossless_agent.engine.embedder import EmbedFn

from lossless_agent.store.abc import (
    AbstractContextItemStore,
    AbstractMessageStore,
    AbstractSummaryStore,
)
from lossless_agent.store.models import Message, Summary
from lossless_agent.engine.media import MediaAnnotator
from lossless_agent.engine.summarize_prompt import build_leaf_prompt, build_condensed_prompt

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Custom exceptions
# ------------------------------------------------------------------

class LcmProviderAuthError(Exception):
    """Raised when the summarisation provider rejects credentials."""


class SummarizerTimeoutError(Exception):
    """Raised when the summarisation call exceeds the configured timeout."""


class CompactionUrgency(enum.Enum):
    """Urgency level for compaction based on dual-threshold system."""
    NONE = "none"
    ASYNC = "async"
    BLOCKING = "blocking"



# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

@dataclass
class CompactionConfig:
    """Tunables for the compaction engine."""

    fresh_tail_count: int = 8
    leaf_chunk_tokens: int = 20_000
    leaf_min_fanout: int = 4
    condensed_min_fanout: int = 3
    condensed_min_fanout_hard: int = 2
    context_threshold: float = 0.75
    soft_threshold: Optional[float] = None  # tau_soft; defaults to context_threshold
    hard_threshold: float = 0.85  # tau_hard; blocking compaction above this
    leaf_target_tokens: int = 1200
    condensed_target_tokens: int = 2000
    condensed_min_input_ratio: float = 0.1
    summary_max_overage_factor: float = 3.0
    summary_timeout_ms: int = 60_000
    custom_instructions: str = ""

    @property
    def effective_soft_threshold(self) -> float:
        """Return soft threshold, falling back to context_threshold."""
        return self.soft_threshold if self.soft_threshold is not None else self.context_threshold


SummarizeFn = Callable[[str], Awaitable[str]]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token count: 1 token ≈ 4 chars."""
    return len(text) // 4


_media_annotator = MediaAnnotator()


def _format_messages(messages: List[Message]) -> str:
    """Format messages into a single text block for the summariser."""
    parts = []
    for m in messages:
        content = _media_annotator.annotate(m.content)
        parts.append(f"[{m.role}] {content}")
    return "\n".join(parts)


def _format_summaries(summaries: List[Summary]) -> str:
    """Format summaries into a single text block for the summariser."""
    parts = []
    for s in summaries:
        parts.append(f"[summary depth={s.depth}] {s.content}")
    return "\n".join(parts)


# ------------------------------------------------------------------
# Summarization escalation
# ------------------------------------------------------------------

async def summarize_with_escalation(
    text: str,
    summarize_fn: SummarizeFn,
    target_tokens: int,
    *,
    max_overage_factor: float = 3.0,
    timeout_ms: int = 60_000,
) -> Optional[str]:
    """Wrap *summarize_fn* with 4-level escalation to guarantee bounded output.

    Returns ``None`` only when the provider raises :class:`LcmProviderAuthError`
    (caller should skip / not persist in that case).

    Levels:
      1. *normal*  – call summarize_fn as-is
      2. *aggressive* – if output tokens >= input tokens, re-call with
         "AGGRESSIVE: Compress harder, be more concise" prepended
      3. *fallback* – deterministic truncation to target_tokens * 4 chars
      4. *capped* – hard cap at max_overage_factor * target_tokens
    """
    input_tokens = _estimate_tokens(text)
    timeout_s = timeout_ms / 1000.0

    # --- Level 1: normal ---
    result: Optional[str] = None
    try:
        result = await asyncio.wait_for(summarize_fn(text), timeout=timeout_s)
    except LcmProviderAuthError:
        logger.warning("Provider auth error during summarisation – skipping")
        return None
    except asyncio.TimeoutError:
        logger.warning(
            "Summarisation timed out after %dms – falling through to fallback",
            timeout_ms,
        )
    except Exception:
        logger.exception("Summarisation failed – falling through to fallback")

    if result is not None:
        result_tokens = _estimate_tokens(result)
        if result_tokens < input_tokens:
            # --- Level 4 cap check even on "good" results ---
            cap_tokens = int(max_overage_factor * target_tokens)
            if result_tokens > cap_tokens:
                cap_chars = cap_tokens * 4
                result = result[:cap_chars]
                logger.info("Capped summary from %d to %d tokens", result_tokens, cap_tokens)
            return result

    # --- Level 2: aggressive ---
    if result is not None and _estimate_tokens(result) >= input_tokens:
        aggressive_prompt = (
            "AGGRESSIVE: Compress harder, be more concise\n" + text
        )
        try:
            result = await asyncio.wait_for(
                summarize_fn(aggressive_prompt), timeout=timeout_s
            )
        except LcmProviderAuthError:
            return None
        except asyncio.TimeoutError:
            logger.warning("Aggressive summarisation timed out – falling to fallback")
            result = None
        except Exception:
            logger.exception("Aggressive summarisation failed – falling to fallback")
            result = None

        if result is not None:
            result_tokens = _estimate_tokens(result)
            if result_tokens < input_tokens:
                cap_tokens = int(max_overage_factor * target_tokens)
                if result_tokens > cap_tokens:
                    result = result[: cap_tokens * 4]
                return result

    # --- Level 3: fallback (deterministic truncation) ---
    target_chars = target_tokens * 4
    source = result if result is not None else text
    truncated = source[:target_chars]
    suffix = f" [Truncated from {input_tokens} tokens]"
    result = truncated + suffix
    logger.info("Fell through to deterministic truncation (level 3)")

    # --- Level 4: hard cap ---
    cap_tokens = int(max_overage_factor * target_tokens)
    cap_chars = cap_tokens * 4
    if len(result) > cap_chars:
        result = result[:cap_chars]
        logger.info("Hard-capped summary to %d tokens", cap_tokens)

    return result


# ------------------------------------------------------------------
# Prior summary context resolution
# ------------------------------------------------------------------

def resolve_prior_summary_context(
    conversation_id: int,
    chunk_start_seq: int,
    summary_store: AbstractSummaryStore,
    context_item_store: AbstractContextItemStore | None = None,
    limit: int = 2,
) -> str:
    """Retrieve previous summary content to provide context for leaf compaction.

    Strategy 1 (context_item_store available):
      Get context items with ordinal < chunk's first message ordinal,
      filter to summaries, take last *limit*, fetch content.

    Strategy 2 (fallback):
      Get all summaries for conversation ordered by created_at,
      filter to those covering time before chunk start, take last *limit*.

    Returns joined summary content (double newline separated), or ''.
    """
    if context_item_store is not None:
        items = context_item_store.get_items(str(conversation_id))
        # Filter to items before the chunk start ordinal that are summaries
        summary_items = [
            it for it in items
            if it.item_type == "summary" and it.ordinal < chunk_start_seq
        ]
        # Take last `limit` items
        recent = summary_items[-limit:]
        contents = []
        for it in recent:
            if it.summary_id:
                s = summary_store.get_by_id(it.summary_id)
                if s:
                    contents.append(s.content)
        return "\n\n".join(contents)

    # Fallback: use summary store directly
    all_summaries = summary_store.get_by_conversation(conversation_id)
    # Sort by created_at
    all_summaries.sort(key=lambda s: s.created_at)
    # We need summaries whose latest_at is before our chunk
    # Since we don't have exact timestamps from seq, we use all summaries
    # that were created before our chunk's messages
    # Use depth 0 (leaf) summaries for context
    prior = [s for s in all_summaries if s.depth == 0]
    recent = prior[-limit:]
    return "\n\n".join(s.content for s in recent)


# ------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------

class CompactionEngine:
    """Drives leaf and condensed compaction passes over a conversation."""

    def __init__(
        self,
        msg_store: AbstractMessageStore,
        sum_store: AbstractSummaryStore,
        summarize_fn: SummarizeFn,
        config: CompactionConfig | None = None,
        circuit_breaker: "CircuitBreaker | None" = None,
        context_item_store: AbstractContextItemStore | None = None,
        embed_fn: "EmbedFn | None" = None,
        vector_store: "VectorStore | None" = None,
    ) -> None:
        self._msg = msg_store
        self._sum = sum_store
        self._summarize = summarize_fn
        self.cfg = config or CompactionConfig()
        self._cb = circuit_breaker
        self._ctx = context_item_store
        self._embed_fn = embed_fn
        self._vector_store = vector_store

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------

    async def _maybe_embed(self, summary: Summary) -> None:
        """Embed a summary node and store in vector_store if configured."""
        if self._embed_fn is None or self._vector_store is None:
            return
        try:
            embedding = await self._embed_fn(summary.content)
            self._vector_store.store(
                summary.summary_id, summary.conversation_id, embedding
            )
        except Exception:
            logger.warning(
                "Failed to embed summary %s — cross-session search may be incomplete",
                summary.summary_id,
                exc_info=True,
            )

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
        """Return True when total conversation tokens exceed the soft threshold."""
        return self.compaction_urgency(conv_id, context_limit) is not CompactionUrgency.NONE

    def compaction_urgency(self, conv_id: int, context_limit: int) -> CompactionUrgency:
        """Return how urgently compaction is needed (dual-threshold)."""
        if context_limit <= 0:
            return CompactionUrgency.NONE
        total = self._msg.total_tokens(conv_id)
        ratio = total / context_limit
        soft = self.cfg.effective_soft_threshold
        hard = self.cfg.hard_threshold
        if ratio >= hard:
            return CompactionUrgency.BLOCKING
        if ratio >= soft:
            return CompactionUrgency.ASYNC
        return CompactionUrgency.NONE

    # ------------------------------------------------------------------
    # Leaf compaction
    # ------------------------------------------------------------------

    async def compact_leaf(
        self, conv_id: int, previous_summary_content: str = "",
    ) -> Optional[Summary]:
        """Run one leaf compaction pass: summarise a chunk of messages."""
        chunk = self.select_chunk(conv_id)
        if not chunk:
            return None

        messages_text = _format_messages(chunk)

        # Resolve prior summary context if not provided
        if not previous_summary_content:
            previous_summary_content = resolve_prior_summary_context(
                conversation_id=conv_id,
                chunk_start_seq=chunk[0].seq,
                summary_store=self._sum,
                context_item_store=self._ctx,
            )

        # Build structured prompt with custom instructions + prior context
        prompt = build_leaf_prompt(
            messages_text=messages_text,
            target_tokens=self.cfg.leaf_target_tokens,
            custom_instructions=self.cfg.custom_instructions,
            previous_summary=previous_summary_content,
        )

        summary_text = await summarize_with_escalation(
            prompt,
            self._summarize,
            target_tokens=self.cfg.leaf_target_tokens,
            max_overage_factor=self.cfg.summary_max_overage_factor,
            timeout_ms=self.cfg.summary_timeout_ms,
        )
        if summary_text is None:
            return None

        source_tokens = sum(m.token_count for m in chunk)
        msg_ids = [m.id for m in chunk]
        token_count = _estimate_tokens(summary_text)

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

        # Maintain context items if store is configured
        if self._ctx is not None:
            conv_id_str = str(conv_id)
            msg_id_strs = [str(mid) for mid in msg_ids]
            new_ordinal = self._ctx.get_max_ordinal(conv_id_str) + 1
            self._ctx.replace_messages_with_summary(
                conv_id_str, msg_id_strs, summary.summary_id, new_ordinal
            )

        await self._maybe_embed(summary)
        return summary

    # ------------------------------------------------------------------
    # Condensed compaction
    # ------------------------------------------------------------------

    async def compact_condensed(
        self, conv_id: int, depth: int = 0, *, hard_trigger: bool = False,
    ) -> Optional[Summary]:
        """Merge orphan summaries at *depth* into one condensed node."""
        orphan_ids = self._sum.get_orphan_ids(conv_id, depth)
        min_fanout = (
            self.cfg.condensed_min_fanout_hard
            if hard_trigger
            else self.cfg.condensed_min_fanout
        )
        if len(orphan_ids) < min_fanout:
            return None

        # Fetch full summary objects for formatting
        all_at_depth = self._sum.get_by_depth(conv_id, depth)
        orphan_set = set(orphan_ids)
        orphans = [s for s in all_at_depth if s.summary_id in orphan_set]

        summaries_text = _format_summaries(orphans)

        # Build structured prompt with depth-aware guidance + custom instructions
        prompt = build_condensed_prompt(
            summaries_text=summaries_text,
            target_tokens=self.cfg.condensed_target_tokens,
            depth=depth + 1,  # target depth of the new condensed summary
            custom_instructions=self.cfg.custom_instructions,
        )

        summary_text = await summarize_with_escalation(
            prompt,
            self._summarize,
            target_tokens=self.cfg.condensed_target_tokens,
            max_overage_factor=self.cfg.summary_max_overage_factor,
            timeout_ms=self.cfg.summary_timeout_ms,
        )
        if summary_text is None:
            return None

        token_count = _estimate_tokens(summary_text)

        summary = self._sum.create_condensed(
            conversation_id=conv_id,
            content=summary_text,
            token_count=token_count,
            child_ids=orphan_ids,
            earliest_at=orphans[0].earliest_at,
            latest_at=orphans[-1].latest_at,
            model="compaction",
        )
        await self._maybe_embed(summary)
        return summary

    # ------------------------------------------------------------------
    # Incremental compaction
    # ------------------------------------------------------------------

    async def run_incremental(
        self, conv_id: int, context_limit: int
    ) -> List[Summary]:
        """Run one incremental compaction cycle (leaf then condensed).

        Respects dual-threshold urgency:
        - NONE  -> skip
        - ASYNC -> single pass (leaf + condensed)
        - BLOCKING -> compact_until_under for aggressive compaction
        """
        urgency = self.compaction_urgency(conv_id, context_limit)
        if urgency is CompactionUrgency.NONE:
            return []

        if urgency is CompactionUrgency.BLOCKING:
            return await self.compact_until_under(conv_id, context_limit)

        created: List[Summary] = []

        # Circuit breaker guard
        if self._cb is not None:
            cb_key = str(conv_id)
            if self._cb.is_open(cb_key):
                logger.warning(
                    "Circuit breaker open for conv %d – skipping compaction",
                    conv_id,
                )
                return created

        try:
            leaf = await self.compact_leaf(conv_id)
            if leaf is not None:
                created.append(leaf)

            condensed = await self.compact_condensed(conv_id, depth=0)
            if condensed is not None:
                created.append(condensed)

            if self._cb is not None:
                self._cb.record_success(str(conv_id))
        except Exception:
            if self._cb is not None:
                self._cb.record_failure(str(conv_id))
            raise

        return created

    # ------------------------------------------------------------------
    # Full sweep
    # ------------------------------------------------------------------

    async def compact_full_sweep(
        self, conv_id: int, *, hard_trigger: bool = False,
    ) -> List[Summary]:
        """Phase 1: repeated leaf passes, Phase 2: condensed at increasing depths."""
        created: List[Summary] = []

        # Phase 1: exhaust leaf compaction
        # Chain previous_summary_content: each leaf pass's output feeds the next
        previous_summary = ""
        while True:
            leaf = await self.compact_leaf(conv_id, previous_summary_content=previous_summary)
            if leaf is None:
                break
            created.append(leaf)
            previous_summary = leaf.content  # chain for continuity

        # Phase 2: condensed at increasing depths
        depth = 0
        while True:
            condensed = await self.compact_condensed(
                conv_id, depth=depth, hard_trigger=hard_trigger
            )
            if condensed is None:
                break
            created.append(condensed)
            depth += 1

        return created

    # ------------------------------------------------------------------
    # Compact until under budget
    # ------------------------------------------------------------------

    async def compact_until_under(
        self,
        conv_id: int,
        context_limit: int,
        *,
        max_rounds: int = 10,
    ) -> List[Summary]:
        """Repeatedly sweep until tokens are under budget or no progress."""
        all_created: List[Summary] = []

        for _ in range(max_rounds):
            tokens_before = self._msg.total_tokens(conv_id)
            if not self.needs_compaction(conv_id, context_limit):
                break

            # Circuit breaker guard
            if self._cb is not None and self._cb.is_open(str(conv_id)):
                logger.warning(
                    "Circuit breaker open for conv %d – aborting compact_until_under",
                    conv_id,
                )
                break

            try:
                created = await self.compact_full_sweep(
                    conv_id, hard_trigger=True
                )
            except Exception:
                if self._cb is not None:
                    self._cb.record_failure(str(conv_id))
                raise

            all_created.extend(created)

            if not created:
                break

            tokens_after = self._msg.total_tokens(conv_id)
            if tokens_after >= tokens_before:
                # No progress
                break

            if self._cb is not None:
                self._cb.record_success(str(conv_id))

        return all_created


# Lazy import to avoid circular deps
from lossless_agent.engine.circuit_breaker import CircuitBreaker  # noqa: E402
