"""Sub-agent expansion tool for navigating the DAG and answering questions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Coroutine, List, Optional

from lossless_agent.store.database import Database
from lossless_agent.store.message_store import MessageStore
from lossless_agent.store.summary_store import SummaryStore
from lossless_agent.tools.recall import (
    lcm_grep,
    lcm_describe,
    lcm_expand,
    DescribeResult,
)


@dataclass
class ExpandQueryConfig:
    """Configuration for sub-agent expansion."""

    max_tokens: int = 4000
    timeout_ms: int = 120000
    max_steps: int = 10


@dataclass
class ExpandQueryResult:
    """Result from a sub-agent expansion query."""

    answer: str
    cited_summaries: List[str]
    tokens_used: int
    steps_taken: int


class ExpansionOrchestrator:
    """Orchestrates sub-agent recall by navigating the summary DAG."""

    def __init__(
        self,
        db: Database,
        msg_store: MessageStore,
        sum_store: SummaryStore,
        expand_fn: Callable[..., Coroutine[Any, Any, str]],
        config: Optional[ExpandQueryConfig] = None,
    ) -> None:
        self.db = db
        self.msg_store = msg_store
        self.sum_store = sum_store
        self.expand_fn = expand_fn
        self.config = config or ExpandQueryConfig()

    async def expand_query(
        self, conversation_id: int, query: str
    ) -> ExpandQueryResult:
        """Run a simplified sub-agent loop to answer a query about past context.

        Steps:
        1. lcm_grep to find relevant summaries/messages
        2. lcm_describe on each summary hit (up to 5)
        3. lcm_expand on leaf summaries to get source messages
        4. Call expand_fn with compiled context + query
        5. Return ExpandQueryResult
        """
        steps = 0
        max_steps = self.config.max_steps
        cited_summaries: List[str] = []
        compiled_context: List[str] = []

        # Step 1: grep for relevant content
        if steps >= max_steps:
            return self._empty_result(steps)
        steps += 1

        grep_results = lcm_grep(
            self.db,
            query,
            scope="all",
            conversation_id=conversation_id,
            limit=5,
        )

        if not grep_results:
            # No matches - call expand_fn with empty context
            if steps >= max_steps:
                return self._empty_result(steps)
            steps += 1
            answer = await self.expand_fn(
                self._build_prompt(query, []),
            )
            return ExpandQueryResult(
                answer=answer,
                cited_summaries=[],
                tokens_used=len(answer),
                steps_taken=steps,
            )

        # Step 2: describe each summary hit (up to 5)
        summary_hits = [r for r in grep_results if r.type == "summary"]
        message_hits = [r for r in grep_results if r.type == "message"]

        # Add message snippets to context
        for mhit in message_hits:
            compiled_context.append(
                f"[Message {mhit.id}] ({mhit.metadata.get('role', 'unknown')}): "
                f"{mhit.content_snippet}"
            )

        described: List[DescribeResult] = []
        for shit in summary_hits[:5]:
            if steps >= max_steps:
                break
            steps += 1
            desc = lcm_describe(self.db, str(shit.id))
            if desc is not None:
                described.append(desc)
                cited_summaries.append(desc.summary_id)
                compiled_context.append(
                    f"[Summary {desc.summary_id}] (kind={desc.kind}, "
                    f"depth={desc.depth}): {desc.content}"
                )

        # Step 3: expand leaf summaries to get source messages
        for desc in described:
            if steps >= max_steps:
                break
            if desc.kind == "leaf":
                steps += 1
                expanded = lcm_expand(self.db, desc.summary_id)
                if expanded is not None:
                    for child in expanded.children:
                        compiled_context.append(
                            f"[Source from {desc.summary_id}] "
                            f"({child.role}): {child.content}"  # type: ignore[union-attr]
                        )

        # Step 4: synthesize answer via expand_fn
        if steps >= max_steps:
            # Hit the limit - return what we have without calling expand_fn
            return ExpandQueryResult(
                answer="Max steps reached. Partial context retrieved.",
                cited_summaries=cited_summaries,
                tokens_used=0,
                steps_taken=steps,
            )
        steps += 1

        prompt = self._build_prompt(query, compiled_context)
        answer = await self.expand_fn(prompt)

        return ExpandQueryResult(
            answer=answer,
            cited_summaries=cited_summaries,
            tokens_used=len(answer),
            steps_taken=steps,
        )

    def _build_prompt(self, query: str, context_parts: List[str]) -> str:
        """Build the synthesis prompt for expand_fn."""
        strategy = (
            "You are a recall agent. Your job is to find information about: "
            f"{query}. You have access to lcm_grep, lcm_describe, and "
            "lcm_expand tools. Search the conversation history, drill into "
            "relevant summaries, and synthesize an answer. Cite summary IDs."
        )
        if not context_parts:
            return (
                f"{strategy}\n\nNo relevant context found. "
                "Provide a minimal answer indicating no information was found."
            )
        context_block = "\n".join(context_parts)
        return (
            f"{strategy}\n\nRetrieved context:\n{context_block}\n\n"
            f"Based on the above context, answer: {query}"
        )

    def _empty_result(self, steps: int) -> ExpandQueryResult:
        """Return an empty result when no processing could be done."""
        return ExpandQueryResult(
            answer="",
            cited_summaries=[],
            tokens_used=0,
            steps_taken=steps,
        )
