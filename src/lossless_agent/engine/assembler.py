"""Context assembler: builds a budget-aware context from summaries and messages."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from lossless_agent.store.vector_store import VectorStore

from lossless_agent.store.abc import AbstractMessageStore, AbstractSummaryStore
from lossless_agent.store.models import Message, Summary
from lossless_agent.engine.transcript_repair import TranscriptRepairer


@dataclass
class AssemblerConfig:
    """Tunables for context assembly."""

    max_context_tokens: int
    summary_budget_ratio: float = 0.4
    fresh_tail_count: int = 8
    repair_transcripts: bool = True
    max_assembly_token_budget: Optional[int] = None  # Hard cap from LCM_MAX_ASSEMBLY_TOKEN_BUDGET

    @property
    def effective_max_tokens(self) -> int:
        """Return the effective token budget (respects hard cap if set)."""
        if self.max_assembly_token_budget is not None:
            return min(self.max_context_tokens, self.max_assembly_token_budget)
        return self.max_context_tokens


@dataclass
class AssembledContext:
    """Result of context assembly."""

    summaries: List[Summary]
    messages: List[Message]
    total_tokens: int


class ContextAssembler:
    """Assemble an LLM context window from summaries and recent messages."""

    def __init__(
        self,
        msg_store: AbstractMessageStore,
        sum_store: AbstractSummaryStore,
        config: AssemblerConfig,
    ) -> None:
        self._msg = msg_store
        self._sum = sum_store
        self.cfg = config

    @staticmethod
    def _tokenize_query(prompt: str) -> List[str]:
        """Simple whitespace tokenization with punctuation stripping."""
        return [
            re.sub(r"[^\w]", "", w.lower())
            for w in prompt.split()
            if re.sub(r"[^\w]", "", w.lower())
        ]

    def _bm25_score(
        self,
        query_terms: List[str],
        document: str,
        avg_doc_len: float,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> float:
        """BM25-lite relevance score for a document against query terms."""
        if not query_terms:
            return 0.0

        doc_terms = document.lower().split()
        doc_len = len(doc_terms)
        tf = Counter(doc_terms)

        score = 0.0
        for term in query_terms:
            term_freq = tf.get(term, 0)
            if term_freq == 0:
                continue
            numerator = term_freq * (k1 + 1)
            denominator = term_freq + k1 * (1 - b + b * doc_len / avg_doc_len)
            score += numerator / denominator

        return score

    @staticmethod
    def _collect_tool_call_ids_from_tail(tail: List[Message]) -> set:
        """Collect tool_call_ids from assistant messages in the tail."""
        return {
            m.tool_call_id
            for m in tail
            if m.role == "assistant" and m.tool_call_id
        }

    @staticmethod
    def _collect_tool_result_ids(messages: List[Message]) -> set:
        """Collect tool_call_ids from tool result messages."""
        return {
            m.tool_call_id
            for m in messages
            if m.role == "tool" and m.tool_call_id
        }

    @staticmethod
    def _generate_fallback_tool_call_id(message_id: int, seq: int) -> str:
        """Generate a fallback tool_call_id for messages missing one."""
        return f"toolu_lcm_{message_id}_{seq}"

    @staticmethod
    def _ensure_tool_call_ids(messages: List[Message]) -> List[Message]:
        """Ensure all assistant tool-call messages have a tool_call_id.

        If an assistant message has a tool_name but no tool_call_id,
        generate a fallback ID to prevent API crashes.
        """
        for m in messages:
            if m.role == "assistant" and m.tool_name and not m.tool_call_id:
                m.tool_call_id = ContextAssembler._generate_fallback_tool_call_id(
                    m.id, m.seq
                )
        return messages

    @staticmethod
    def _filter_orphaned_tool_calls(messages: List[Message]) -> List[Message]:
        """Filter out non-fresh assistant tool-call messages with no matching result.

        For assistant messages that have tool_call_id references, check if
        matching tool results exist in the message list. Drop assistant
        tool-call messages that have no matching tool result.
        """
        result_ids = {
            m.tool_call_id
            for m in messages
            if m.role == "tool" and m.tool_call_id
        }

        filtered: List[Message] = []
        for m in messages:
            if (
                m.role == "assistant"
                and m.tool_call_id
                and m.tool_call_id not in result_ids
            ):
                continue
            filtered.append(m)
        return filtered

    def assemble(
        self, conv_id: int, prompt: Optional[str] = None
    ) -> AssembledContext:
        """Build a budget-aware context for a conversation.

        Strategy:
        1. Take the last ``fresh_tail_count`` messages as the tail.
        2. Compute summary budget from remaining tokens.
        3. Fill with summaries ordered by depth DESC, earliest_at ASC.
        4. Skip any summary whose children are already included.

        If ``prompt`` is provided, score summaries by BM25 relevance
        and prefer higher-scoring ones when filling the budget.
        """
        # 1. Fresh tail
        tail = self._msg.tail(conv_id, self.cfg.fresh_tail_count)

        # 1a. Ensure all tool calls have IDs (fallback generation)
        tail = self._ensure_tool_call_ids(tail)

        # 1b. Repair tool-call/result pairing if enabled
        if self.cfg.repair_transcripts:
            repairer = TranscriptRepairer()
            tail = repairer.repair(tail)

        # 1c. Fresh tail tool call protection:
        # Collect tool_call_ids from assistant messages in the tail
        tail_tool_call_ids = self._collect_tool_call_ids_from_tail(tail)

        # Determine which tool_call_ids already have results in the tail
        tail_result_ids = self._collect_tool_result_ids(tail)
        # IDs that need results pulled from prefix
        missing_result_ids = tail_tool_call_ids - tail_result_ids

        # If there are missing results, scan the evictable prefix
        protected_from_prefix: List[Message] = []
        if missing_result_ids:
            all_messages = self._msg.get_messages(conv_id)
            tail_ids = {m.id for m in tail}
            prefix = [m for m in all_messages if m.id not in tail_ids]
            for m in prefix:
                if m.role == "tool" and m.tool_call_id in missing_result_ids:
                    protected_from_prefix.append(m)

        tail_tokens = sum(m.token_count for m in tail)
        protected_tokens = sum(m.token_count for m in protected_from_prefix)

        # 2. Summary budget
        remaining = self.cfg.effective_max_tokens - tail_tokens - protected_tokens
        summary_budget = remaining * self.cfg.summary_budget_ratio

        if summary_budget <= 0:
            final_messages = protected_from_prefix + tail
            return AssembledContext(
                summaries=[],
                messages=final_messages,
                total_tokens=tail_tokens + protected_tokens,
            )

        # 3. Fetch all summaries
        all_summaries = self._sum.get_by_conversation(conv_id)

        # Determine child IDs to skip (from parent-child hierarchy)
        all_summaries.sort(key=lambda s: (-s.depth, s.earliest_at))
        child_ids_of_included: set = set()
        eligible: List[Summary] = []

        for s in all_summaries:
            if s.summary_id in child_ids_of_included:
                continue
            eligible.append(s)
            children = self._sum.get_child_ids(s.summary_id)
            child_ids_of_included.update(children)

        # Sort eligible summaries
        if prompt:
            query_terms = self._tokenize_query(prompt)
            if query_terms and eligible:
                avg_doc_len = (
                    sum(len(s.content.split()) for s in eligible) / len(eligible)
                )
                eligible.sort(
                    key=lambda s: self._bm25_score(
                        query_terms, s.content, avg_doc_len
                    ),
                    reverse=True,
                )
        else:
            pass

        # 4. Fill budget
        included: List[Summary] = []
        summary_tokens = 0

        for s in eligible:
            if summary_tokens + s.token_count > summary_budget:
                continue
            included.append(s)
            summary_tokens += s.token_count

        final_messages = protected_from_prefix + tail
        total = tail_tokens + protected_tokens + summary_tokens
        return AssembledContext(
            summaries=included, messages=final_messages, total_tokens=total
        )

    async def cross_session_context(
        self,
        query_embedding: List[float],
        current_conv_id: int,
        vector_store: "VectorStore",
        top_k: int = 5,
        token_budget: int = 2000,
        min_score: float = 0.70,
    ) -> str:
        """Search other conversations for semantically similar summaries.

        Uses cosine similarity via pgvector to find summaries from past
        sessions that are relevant to the current query. Results are
        formatted as ``<cross_session_memory>`` blocks for injection into
        the user message (separate from the current-session context).

        Returns an empty string when no relevant cross-session memories
        are found or when vector_store returns no hits.
        """
        hits = vector_store.search(
            query_embedding, top_k=top_k, exclude_conversation_id=current_conv_id
        )
        if not hits:
            return ""

        parts: List[str] = []
        tokens_used = 0
        for summary_id, score in hits:
            if score < min_score:
                continue
            s = self._sum.get_by_id(summary_id)
            if s is None:
                continue
            if tokens_used + s.token_count > token_budget:
                continue
            parts.append(
                f"<cross_session_memory id='{s.summary_id}' score={score:.3f}>"
                f"\n{s.content}\n</cross_session_memory>"
            )
            tokens_used += s.token_count

        return "\n".join(parts)

    def format_context(self, assembled: AssembledContext) -> str:
        """Format an AssembledContext into a string for the LLM.

        Summaries become XML-like blocks, messages are ``[role] content``.
        """
        parts: List[str] = []

        for s in assembled.summaries:
            parts.append(
                f"<summary id='{s.summary_id}' depth={s.depth}>"
                f"{s.content}</summary>"
            )

        for m in assembled.messages:
            parts.append(f"[{m.role}] {m.content}")

        return "\n".join(parts)
