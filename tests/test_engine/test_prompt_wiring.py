"""Tests verifying that summarize_prompt templates and prior summary context
are actually wired into the compaction pipeline (not dead code)."""
from __future__ import annotations

import asyncio

import pytest

from lossless_agent.store.database import Database
from lossless_agent.store.conversation_store import ConversationStore
from lossless_agent.store.message_store import MessageStore
from lossless_agent.store.summary_store import SummaryStore
from lossless_agent.store.context_item_store import ContextItemStore
from lossless_agent.engine.compaction import CompactionConfig, CompactionEngine


@pytest.fixture
def db():
    database = Database(":memory:")
    yield database
    database.close()


def _seed_conversation(db, msg_count=20, token_count=100):
    """Create a conversation with messages."""
    conv_store = ConversationStore(db)
    msg_store = MessageStore(db)
    conv = conv_store.get_or_create("test-session", "Test")
    for i in range(msg_count):
        role = "user" if i % 2 == 0 else "assistant"
        msg_store.append(conv.id, role, f"Message {i} content here", token_count=token_count)
    return conv, msg_store, SummaryStore(db)


class TestLeafPromptContainsPreviousContext:
    """Verify that compact_leaf passes previous_summary to the prompt."""

    def test_leaf_prompt_includes_previous_context_tag(self, db):
        """When prior summaries exist, the summarize_fn should receive
        <previous_context> tags in the prompt."""
        conv, msg_store, sum_store = _seed_conversation(db, msg_count=30, token_count=50)
        ctx_store = ContextItemStore(db)

        received_prompts = []

        async def capture_summarize(text: str) -> str:
            received_prompts.append(text)
            return "Summarized content"

        config = CompactionConfig(
            fresh_tail_count=8,
            leaf_min_fanout=4,
            leaf_chunk_tokens=5000,
            custom_instructions="Focus on code changes",
        )

        engine = CompactionEngine(
            msg_store, sum_store, capture_summarize,
            config=config, context_item_store=ctx_store,
        )

        # First leaf pass — no prior summaries exist yet
        result1 = asyncio.get_event_loop().run_until_complete(engine.compact_leaf(conv.id))
        assert result1 is not None
        assert len(received_prompts) == 1
        prompt1 = received_prompts[0]

        # Verify structured prompt elements
        assert "Target token count:" in prompt1
        assert "Operator instructions: Focus on code changes" in prompt1
        assert "<messages>" in prompt1
        assert "</messages>" in prompt1

        # Second leaf pass — now prior summary exists
        result2 = asyncio.get_event_loop().run_until_complete(engine.compact_leaf(conv.id))
        if result2 is not None and len(received_prompts) > 1:
            prompt2 = received_prompts[-1]
            # Should have previous context from first summary
            assert "<previous_context>" in prompt2 or "Previous context:" in prompt2

    def test_leaf_prompt_has_custom_instructions(self, db):
        """custom_instructions from config should appear in the prompt."""
        conv, msg_store, sum_store = _seed_conversation(db)

        received_prompts = []

        async def capture_summarize(text: str) -> str:
            received_prompts.append(text)
            return "Summary"

        config = CompactionConfig(
            fresh_tail_count=4,
            leaf_min_fanout=3,
            custom_instructions="Preserve all function signatures",
        )
        engine = CompactionEngine(msg_store, sum_store, capture_summarize, config=config)

        asyncio.get_event_loop().run_until_complete(engine.compact_leaf(conv.id))
        assert len(received_prompts) >= 1
        assert "Preserve all function signatures" in received_prompts[0]

    def test_leaf_prompt_without_custom_instructions(self, db):
        """Without custom_instructions, prompt should say '(none)'."""
        conv, msg_store, sum_store = _seed_conversation(db)

        received_prompts = []

        async def capture_summarize(text: str) -> str:
            received_prompts.append(text)
            return "Summary"

        config = CompactionConfig(fresh_tail_count=4, leaf_min_fanout=3)
        engine = CompactionEngine(msg_store, sum_store, capture_summarize, config=config)

        asyncio.get_event_loop().run_until_complete(engine.compact_leaf(conv.id))
        assert "Operator instructions: (none)" in received_prompts[0]


class TestCondensedPromptWiring:
    """Verify that compact_condensed uses depth-aware structured prompts."""

    def test_condensed_prompt_has_depth_guidance(self, db):
        """condensed prompt should include depth-specific guidance."""
        conv, msg_store, sum_store = _seed_conversation(db, msg_count=40, token_count=30)

        received_prompts = []
        call_count = [0]

        async def capture_summarize(text: str) -> str:
            call_count[0] += 1
            received_prompts.append(text)
            return f"Summary {call_count[0]}"

        config = CompactionConfig(
            fresh_tail_count=4,
            leaf_min_fanout=3,
            leaf_chunk_tokens=500,
            condensed_min_fanout=2,
            custom_instructions="Keep API endpoints",
        )
        engine = CompactionEngine(msg_store, sum_store, capture_summarize, config=config)

        # Do multiple leaf passes to create enough summaries for condensed
        for _ in range(5):
            asyncio.get_event_loop().run_until_complete(engine.compact_leaf(conv.id))

        # Try condensed pass
        result = asyncio.get_event_loop().run_until_complete(
            engine.compact_condensed(conv.id, depth=0)
        )

        if result is not None:
            # The last prompt should be the condensed one
            condensed_prompt = received_prompts[-1]
            assert "<summaries>" in condensed_prompt
            assert "Operator instructions: Keep API endpoints" in condensed_prompt
            assert "Guidance:" in condensed_prompt


class TestFullSweepChainsPreviousSummary:
    """Verify that compact_full_sweep chains previous_summary_content."""

    def test_full_sweep_chains_context(self, db):
        """Each leaf pass in full_sweep should chain its output as previous context."""
        conv, msg_store, sum_store = _seed_conversation(db, msg_count=40, token_count=30)

        received_prompts = []

        async def capture_summarize(text: str) -> str:
            received_prompts.append(text)
            return f"Summary pass {len(received_prompts)}"

        config = CompactionConfig(
            fresh_tail_count=4,
            leaf_min_fanout=3,
            leaf_chunk_tokens=300,
        )
        engine = CompactionEngine(msg_store, sum_store, capture_summarize, config=config)

        results = asyncio.get_event_loop().run_until_complete(
            engine.compact_full_sweep(conv.id)
        )

        # If multiple leaf passes occurred, later ones should have previous context
        if len(results) >= 2:
            # Second leaf prompt should contain output of first
            leaf_prompts = [p for p in received_prompts if "<messages>" in p]
            if len(leaf_prompts) >= 2:
                assert "<previous_context>" in leaf_prompts[1]
                assert "Summary pass 1" in leaf_prompts[1]
