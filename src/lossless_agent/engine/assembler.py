"""Context assembler: builds a budget-aware context from summaries and messages."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

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

        # 1b. Repair tool-call/result pairing if enabled
        if self.cfg.repair_transcripts:
            repairer = TranscriptRepairer()
            tail = repairer.repair(tail)

        tail_tokens = sum(m.token_count for m in tail)

        # 2. Summary budget
        remaining = self.cfg.max_context_tokens - tail_tokens
        summary_budget = remaining * self.cfg.summary_budget_ratio

        if summary_budget <= 0:
            return AssembledContext(
                summaries=[], messages=tail, total_tokens=tail_tokens
            )

        # 3. Fetch all summaries
        all_summaries = self._sum.get_by_conversation(conv_id)

        # Determine child IDs to skip (from parent-child hierarchy)
        # First pass: sort by depth DESC to find parents first
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
            # Default: depth DESC, earliest_at ASC (already sorted)
            pass

        # 4. Fill budget
        included: List[Summary] = []
        summary_tokens = 0

        for s in eligible:
            if summary_tokens + s.token_count > summary_budget:
                continue
            included.append(s)
            summary_tokens += s.token_count

        total = tail_tokens + summary_tokens
        return AssembledContext(
            summaries=included, messages=tail, total_tokens=total
        )

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
