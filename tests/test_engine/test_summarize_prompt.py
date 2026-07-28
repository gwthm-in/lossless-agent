"""Tests for summarize_prompt module."""
from __future__ import annotations

from lossless_agent.engine.summarize_prompt import (
    SYSTEM_PROMPT,
    build_leaf_prompt,
    build_condensed_prompt,
)


class TestSystemPrompt:
    def test_system_prompt_content(self):
        assert "context-compaction summarization engine" in SYSTEM_PROMPT
        assert "plain text summary content only" in SYSTEM_PROMPT


class TestBuildLeafPrompt:
    def test_instructs_summary_not_continuation(self):
        # The core guarantee: the prompt must tell the model to SUMMARIZE, never continue the chat.
        result = build_leaf_prompt("msgs", 1200).lower()
        assert "summary" in result
        assert "not a reply" in result
        assert "do not continue" in result

    def test_includes_target_tokens(self):
        assert "at most 1200 tokens" in build_leaf_prompt("some messages", 1200)

    def test_includes_messages(self):
        result = build_leaf_prompt("hello world msgs", 1200)
        assert "<messages>" in result
        assert "hello world msgs" in result
        assert "</messages>" in result

    def test_no_custom_instructions(self):
        assert "operator instructions" not in build_leaf_prompt("msgs", 1200).lower()

    def test_with_custom_instructions(self):
        result = build_leaf_prompt("msgs", 1200, custom_instructions="Be concise")
        assert "operator instructions: Be concise".lower() in result.lower()

    def test_no_previous_summary(self):
        assert "<previous_context>" not in build_leaf_prompt("msgs", 1200)

    def test_with_previous_summary(self):
        result = build_leaf_prompt("msgs", 1200, previous_summary="prior context here")
        assert "<previous_context>" in result
        assert "prior context here" in result
        assert "</previous_context>" in result

    def test_not_aggressive_by_default(self):
        assert "compress harder" not in build_leaf_prompt("msgs", 1200).lower()

    def test_aggressive_mode(self):
        assert "compress harder" in build_leaf_prompt("msgs", 1200, aggressive=True).lower()

    def test_all_options(self):
        result = build_leaf_prompt(
            "msgs", 2400,
            custom_instructions="Focus on code changes",
            previous_summary="earlier summary",
            aggressive=True,
        )
        assert "at most 2400 tokens" in result
        assert "Focus on code changes" in result
        assert "<previous_context>" in result
        assert "earlier summary" in result
        assert "compress harder" in result.lower()


class TestBuildCondensedPrompt:
    def test_instructs_summary_not_continuation(self):
        result = build_condensed_prompt("summaries", 2000, depth=1).lower()
        assert "summary" in result
        assert "not a reply" in result

    def test_includes_target_tokens(self):
        assert "at most 2000 tokens" in build_condensed_prompt("summaries", 2000, depth=1)

    def test_includes_summaries(self):
        result = build_condensed_prompt("summary content", 2000, depth=1)
        assert "<summaries>" in result
        assert "summary content" in result
        assert "</summaries>" in result

    def test_depth_1_guidance(self):
        assert "done, decided, changed, or resolved" in build_condensed_prompt("s", 2000, depth=1)

    def test_depth_2_guidance(self):
        assert "key decisions and outcomes" in build_condensed_prompt("s", 2000, depth=2)

    def test_depth_3_guidance(self):
        assert "most critical durable facts" in build_condensed_prompt("s", 2000, depth=3)

    def test_depth_5_uses_depth3_guidance(self):
        assert "most critical durable facts" in build_condensed_prompt("s", 2000, depth=5)

    def test_no_custom_instructions(self):
        assert "operator instructions" not in build_condensed_prompt("s", 2000, depth=1).lower()

    def test_with_custom_instructions(self):
        result = build_condensed_prompt("s", 2000, depth=1, custom_instructions="Keep names")
        assert "Keep names" in result

    def test_not_aggressive_by_default(self):
        assert "compress harder" not in build_condensed_prompt("s", 2000, depth=1).lower()

    def test_aggressive_mode(self):
        assert "compress harder" in build_condensed_prompt("s", 2000, depth=1, aggressive=True).lower()
