"""Context assembler: builds a budget-aware context from summaries and messages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from lossless_agent.store.message_store import MessageStore
from lossless_agent.store.summary_store import SummaryStore
from lossless_agent.store.models import Message, Summary


@dataclass
class AssemblerConfig:
    """Tunables for context assembly."""

    max_context_tokens: int
    summary_budget_ratio: float = 0.4
    fresh_tail_count: int = 8


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
        msg_store: MessageStore,
        sum_store: SummaryStore,
        config: AssemblerConfig,
    ) -> None:
        self._msg = msg_store
        self._sum = sum_store
        self.cfg = config

    def assemble(self, conv_id: int) -> AssembledContext:
        """Build a budget-aware context for a conversation.

        Strategy:
        1. Take the last ``fresh_tail_count`` messages as the tail.
        2. Compute summary budget from remaining tokens.
        3. Fill with summaries ordered by depth DESC, earliest_at ASC.
        4. Skip any summary whose children are already included.
        """
        # 1. Fresh tail
        tail = self._msg.tail(conv_id, self.cfg.fresh_tail_count)
        tail_tokens = sum(m.token_count for m in tail)

        # 2. Summary budget
        remaining = self.cfg.max_context_tokens - tail_tokens
        summary_budget = remaining * self.cfg.summary_budget_ratio

        if summary_budget <= 0:
            return AssembledContext(
                summaries=[], messages=tail, total_tokens=tail_tokens
            )

        # 3. Fetch all summaries, sort by depth DESC then earliest_at ASC
        all_summaries = self._sum.get_by_conversation(conv_id)
        all_summaries.sort(key=lambda s: (-s.depth, s.earliest_at))

        # 4. Fill budget, skipping children of already-included parents
        included: List[Summary] = []
        included_ids: set = set()
        child_ids_of_included: set = set()
        summary_tokens = 0

        for s in all_summaries:
            # Skip if this summary is a child of an already-included parent
            if s.summary_id in child_ids_of_included:
                continue
            if summary_tokens + s.token_count > summary_budget:
                continue
            included.append(s)
            included_ids.add(s.summary_id)
            summary_tokens += s.token_count
            # Mark children as redundant
            children = self._sum.get_child_ids(s.summary_id)
            child_ids_of_included.update(children)

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
