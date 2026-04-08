"""Tests for summarize_prompt module."""
from __future__ import annotations

from lossless_agent.engine.summarize_prompt import (
    SYSTEM_PROMPT,
    build_leaf_prompt,
    build_condensed_prompt,
)


# ------------------------------------------------------------------
# SYSTEM_PROMPT
# ------------------------------------------------------------------

class TestSystemPrompt:
    def test_system_prompt_content(self):
        assert "context-compaction summarization engine" in SYSTEM_PROMPT
        assert "plain text summary content only" in SYSTEM_PROMPT


# ------------------------------------------------------------------
# build_leaf_prompt
# ------------------------------------------------------------------

class TestBuildLeafPrompt:
    def test_includes_target_tokens(self):
        result = build_leaf_prompt("some messages", 1200)
        assert "Target token count: 1200" in result

    def test_includes_messages(self):
        result = build_leaf_prompt("hello world msgs", 1200)
        assert "<messages>" in result
        assert "hello world msgs" in result
        assert "</messages>" in result

    def test_no_custom_instructions(self):
        result = build_leaf_prompt("msgs", 1200)
        assert "Operator instructions: (none)" in result

    def test_with_custom_instructions(self):
        result = build_leaf_prompt("msgs", 1200, custom_instructions="Be concise")
        assert "Operator instructions: Be concise" in result

    def test_no_previous_summary(self):
        result = build_leaf_prompt("msgs", 1200)
        assert "Previous context: (none)" in result

    def test_with_previous_summary(self):
        result = build_leaf_prompt("msgs", 1200, previous_summary="prior context here")
        assert "<previous_context>" in result
        assert "prior context here" in result
        assert "</previous_context>" in result

    def test_not_aggressive_by_default(self):
        result = build_leaf_prompt("msgs", 1200)
        assert "AGGRESSIVE" not in result

    def test_aggressive_mode(self):
        result = build_leaf_prompt("msgs", 1200, aggressive=True)
        assert "AGGRESSIVE: Compress much harder." in result

    def test_all_options(self):
        result = build_leaf_prompt(
            "msgs", 2400,
            custom_instructions="Focus on code changes",
            previous_summary="earlier summary",
            aggressive=True,
        )
        assert "Target token count: 2400" in result
        assert "Focus on code changes" in result
        assert "<previous_context>" in result
        assert "earlier summary" in result
        assert "AGGRESSIVE" in result


# ------------------------------------------------------------------
# build_condensed_prompt
# ------------------------------------------------------------------

class TestBuildCondensedPrompt:
    def test_includes_target_tokens(self):
        result = build_condensed_prompt("summaries", 2000, depth=1)
        assert "Target token count: 2000" in result

    def test_includes_summaries(self):
        result = build_condensed_prompt("summary content", 2000, depth=1)
        assert "<summaries>" in result
        assert "summary content" in result
        assert "</summaries>" in result

    def test_depth_1_guidance(self):
        result = build_condensed_prompt("s", 2000, depth=1)
        assert "focus on what is new, changed, or resolved" in result

    def test_depth_2_guidance(self):
        result = build_condensed_prompt("s", 2000, depth=2)
        assert "preserve key decisions and outcomes" in result

    def test_depth_3_guidance(self):
        result = build_condensed_prompt("s", 2000, depth=3)
        assert "retain only the most critical facts" in result

    def test_depth_5_uses_depth3_guidance(self):
        result = build_condensed_prompt("s", 2000, depth=5)
        assert "retain only the most critical facts" in result

    def test_no_custom_instructions(self):
        result = build_condensed_prompt("s", 2000, depth=1)
        assert "Operator instructions: (none)" in result

    def test_with_custom_instructions(self):
        result = build_condensed_prompt("s", 2000, depth=1, custom_instructions="Keep names")
        assert "Operator instructions: Keep names" in result

    def test_not_aggressive_by_default(self):
        result = build_condensed_prompt("s", 2000, depth=1)
        assert "AGGRESSIVE" not in result

    def test_aggressive_mode(self):
        result = build_condensed_prompt("s", 2000, depth=1, aggressive=True)
        assert "AGGRESSIVE: Compress much harder." in result
