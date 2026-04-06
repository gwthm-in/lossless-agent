"""Tests for the AgentAdapter ABC and LCMConfig."""
from __future__ import annotations

import pytest
from dataclasses import fields

from lossless_agent.adapters.base import AgentAdapter, LCMConfig
from lossless_agent.engine.compaction import CompactionConfig
from lossless_agent.engine.assembler import AssemblerConfig


class TestLCMConfig:
    """LCMConfig dataclass tests."""

    def test_default_values(self):
        cfg = LCMConfig()
        assert cfg.db_path == ":memory:"
        assert cfg.summary_model == "default"
        assert isinstance(cfg.compaction, CompactionConfig)
        assert isinstance(cfg.assembler, AssemblerConfig)

    def test_custom_db_path(self):
        cfg = LCMConfig(db_path="/tmp/test.db")
        assert cfg.db_path == "/tmp/test.db"

    def test_custom_compaction(self):
        cc = CompactionConfig(fresh_tail_count=4)
        cfg = LCMConfig(compaction=cc)
        assert cfg.compaction.fresh_tail_count == 4

    def test_custom_assembler(self):
        ac = AssemblerConfig(max_context_tokens=64_000)
        cfg = LCMConfig(assembler=ac)
        assert cfg.assembler.max_context_tokens == 64_000


class TestAgentAdapterABC:
    """AgentAdapter cannot be instantiated directly."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            AgentAdapter()

    def test_has_required_methods(self):
        methods = [
            "on_turn_start",
            "on_turn_end",
            "on_session_end",
            "get_tools",
            "handle_tool_call",
            "get_system_prompt_block",
        ]
        for m in methods:
            assert hasattr(AgentAdapter, m), f"Missing method: {m}"

    def test_concrete_subclass_must_implement_all(self):
        """A subclass that misses a method should fail to instantiate."""

        class Incomplete(AgentAdapter):
            pass

        with pytest.raises(TypeError):
            Incomplete()
