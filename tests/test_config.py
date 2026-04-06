"""Tests for LCMConfig — the canonical configuration system."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from lossless_agent.config import LCMConfig
from lossless_agent.engine.compaction import CompactionConfig
from lossless_agent.engine.assembler import AssemblerConfig


# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------

class TestDefaults:
    """LCMConfig() with no arguments gives sane defaults."""

    def test_from_env_returns_all_defaults(self, monkeypatch):
        """from_env with no LCM_* vars set returns field defaults."""
        # Clear any LCM_ env vars that might be set
        for key in list(os.environ):
            if key.startswith("LCM_"):
                monkeypatch.delenv(key)

        cfg = LCMConfig.from_env()
        assert cfg.enabled is True
        assert cfg.db_path == "~/.lossless-agent/lcm.db"
        assert cfg.fresh_tail_count == 8
        assert cfg.leaf_chunk_tokens == 20_000
        assert cfg.leaf_min_fanout == 4
        assert cfg.condensed_min_fanout == 3
        assert cfg.context_threshold == 0.75
        assert cfg.leaf_target_tokens == 1200
        assert cfg.condensed_target_tokens == 2000
        assert cfg.max_context_tokens == 128_000
        assert cfg.summary_budget_ratio == 0.4
        assert cfg.summary_model == ""
        assert cfg.summary_provider == ""
        assert cfg.expansion_model == ""
        assert cfg.ignore_session_patterns == []
        assert cfg.incremental_max_depth == 1
        assert cfg.summary_timeout_ms == 60_000


# ------------------------------------------------------------------
# from_env – env var reading
# ------------------------------------------------------------------

class TestFromEnv:
    """from_env reads each LCM_* environment variable."""

    def test_reads_string_vars(self, monkeypatch):
        monkeypatch.setenv("LCM_DATABASE_PATH", "/tmp/my.db")
        monkeypatch.setenv("LCM_SUMMARY_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("LCM_SUMMARY_PROVIDER", "openai")
        monkeypatch.setenv("LCM_EXPANSION_MODEL", "claude-3-haiku")
        cfg = LCMConfig.from_env()
        assert cfg.db_path == "/tmp/my.db"
        assert cfg.summary_model == "gpt-4o-mini"
        assert cfg.summary_provider == "openai"
        assert cfg.expansion_model == "claude-3-haiku"

    def test_reads_int_vars(self, monkeypatch):
        monkeypatch.setenv("LCM_FRESH_TAIL_COUNT", "12")
        monkeypatch.setenv("LCM_LEAF_CHUNK_TOKENS", "30000")
        monkeypatch.setenv("LCM_LEAF_MIN_FANOUT", "6")
        monkeypatch.setenv("LCM_CONDENSED_MIN_FANOUT", "5")
        monkeypatch.setenv("LCM_LEAF_TARGET_TOKENS", "1500")
        monkeypatch.setenv("LCM_CONDENSED_TARGET_TOKENS", "3000")
        monkeypatch.setenv("LCM_MAX_CONTEXT_TOKENS", "200000")
        monkeypatch.setenv("LCM_INCREMENTAL_MAX_DEPTH", "3")
        monkeypatch.setenv("LCM_SUMMARY_TIMEOUT_MS", "90000")
        cfg = LCMConfig.from_env()
        assert cfg.fresh_tail_count == 12
        assert cfg.leaf_chunk_tokens == 30_000
        assert cfg.leaf_min_fanout == 6
        assert cfg.condensed_min_fanout == 5
        assert cfg.leaf_target_tokens == 1500
        assert cfg.condensed_target_tokens == 3000
        assert cfg.max_context_tokens == 200_000
        assert cfg.incremental_max_depth == 3
        assert cfg.summary_timeout_ms == 90_000

    def test_reads_float_vars(self, monkeypatch):
        monkeypatch.setenv("LCM_CONTEXT_THRESHOLD", "0.9")
        monkeypatch.setenv("LCM_SUMMARY_BUDGET_RATIO", "0.6")
        cfg = LCMConfig.from_env()
        assert cfg.context_threshold == 0.9
        assert cfg.summary_budget_ratio == 0.6

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("Yes", True),
            ("false", False),
            ("False", False),
            ("0", False),
            ("no", False),
        ],
    )
    def test_bool_parsing(self, monkeypatch, raw, expected):
        monkeypatch.setenv("LCM_ENABLED", raw)
        cfg = LCMConfig.from_env()
        assert cfg.enabled is expected

    def test_comma_separated_list(self, monkeypatch):
        monkeypatch.setenv(
            "LCM_IGNORE_SESSION_PATTERNS", "test-*,debug-*,tmp-session"
        )
        cfg = LCMConfig.from_env()
        assert cfg.ignore_session_patterns == [
            "test-*",
            "debug-*",
            "tmp-session",
        ]

    def test_comma_separated_list_with_whitespace(self, monkeypatch):
        monkeypatch.setenv(
            "LCM_IGNORE_SESSION_PATTERNS", " test-* , debug-* , "
        )
        cfg = LCMConfig.from_env()
        assert cfg.ignore_session_patterns == ["test-*", "debug-*"]

    def test_comma_separated_empty(self, monkeypatch):
        monkeypatch.setenv("LCM_IGNORE_SESSION_PATTERNS", "")
        cfg = LCMConfig.from_env()
        assert cfg.ignore_session_patterns == []


# ------------------------------------------------------------------
# from_dict
# ------------------------------------------------------------------

class TestFromDict:
    """from_dict creates a config from a plain dictionary."""

    def test_creates_config_from_dict(self):
        d = {
            "enabled": False,
            "db_path": "/data/lcm.db",
            "fresh_tail_count": 4,
            "max_context_tokens": 64_000,
            "summary_model": "gpt-4o-mini",
        }
        cfg = LCMConfig.from_dict(d)
        assert cfg.enabled is False
        assert cfg.db_path == "/data/lcm.db"
        assert cfg.fresh_tail_count == 4
        assert cfg.max_context_tokens == 64_000
        assert cfg.summary_model == "gpt-4o-mini"
        # Unset fields keep defaults
        assert cfg.leaf_chunk_tokens == 20_000

    def test_ignores_unknown_keys(self):
        d = {"db_path": "/tmp/test.db", "unknown_key": "ignored"}
        cfg = LCMConfig.from_dict(d)
        assert cfg.db_path == "/tmp/test.db"
        assert not hasattr(cfg, "unknown_key")


# ------------------------------------------------------------------
# merge
# ------------------------------------------------------------------

class TestMerge:
    """merge applies overrides onto a base config."""

    def test_overrides_specific_fields(self):
        base = LCMConfig()
        merged = LCMConfig.merge(base, {"fresh_tail_count": 16, "enabled": False})
        assert merged.fresh_tail_count == 16
        assert merged.enabled is False
        # Other fields unchanged
        assert merged.db_path == base.db_path
        assert merged.max_context_tokens == base.max_context_tokens

    def test_ignores_unknown_keys(self):
        base = LCMConfig()
        merged = LCMConfig.merge(base, {"bogus": 123})
        assert merged == base

    def test_does_not_mutate_base(self):
        base = LCMConfig()
        LCMConfig.merge(base, {"fresh_tail_count": 99})
        assert base.fresh_tail_count == 8


# ------------------------------------------------------------------
# to_compaction_config / to_assembler_config
# ------------------------------------------------------------------

class TestConversion:
    """Conversion to engine config dataclasses."""

    def test_to_compaction_config(self):
        cfg = LCMConfig(
            fresh_tail_count=10,
            leaf_chunk_tokens=15_000,
            leaf_min_fanout=5,
            condensed_min_fanout=4,
            context_threshold=0.8,
            leaf_target_tokens=1000,
            condensed_target_tokens=1800,
        )
        cc = cfg.to_compaction_config()
        assert isinstance(cc, CompactionConfig)
        assert cc.fresh_tail_count == 10
        assert cc.leaf_chunk_tokens == 15_000
        assert cc.leaf_min_fanout == 5
        assert cc.condensed_min_fanout == 4
        assert cc.context_threshold == 0.8
        assert cc.leaf_target_tokens == 1000
        assert cc.condensed_target_tokens == 1800

    def test_to_assembler_config(self):
        cfg = LCMConfig(
            max_context_tokens=64_000,
            summary_budget_ratio=0.5,
            fresh_tail_count=6,
        )
        ac = cfg.to_assembler_config()
        assert isinstance(ac, AssemblerConfig)
        assert ac.max_context_tokens == 64_000
        assert ac.summary_budget_ratio == 0.5
        assert ac.fresh_tail_count == 6

    def test_compaction_property_backward_compat(self):
        cfg = LCMConfig(leaf_min_fanout=7)
        assert isinstance(cfg.compaction, CompactionConfig)
        assert cfg.compaction.leaf_min_fanout == 7

    def test_assembler_property_backward_compat(self):
        cfg = LCMConfig(max_context_tokens=50_000)
        assert isinstance(cfg.assembler, AssemblerConfig)
        assert cfg.assembler.max_context_tokens == 50_000


# ------------------------------------------------------------------
# validate
# ------------------------------------------------------------------

class TestValidate:
    """validate() returns a list of errors."""

    def test_valid_config_returns_empty_list(self):
        cfg = LCMConfig()
        assert cfg.validate() == []

    def test_catches_bad_context_threshold_high(self):
        cfg = LCMConfig(context_threshold=1.5)
        errors = cfg.validate()
        assert any("context_threshold" in e for e in errors)

    def test_catches_bad_context_threshold_low(self):
        cfg = LCMConfig(context_threshold=-0.1)
        errors = cfg.validate()
        assert any("context_threshold" in e for e in errors)

    def test_catches_bad_summary_budget_ratio_high(self):
        cfg = LCMConfig(summary_budget_ratio=1.1)
        errors = cfg.validate()
        assert any("summary_budget_ratio" in e for e in errors)

    def test_catches_bad_summary_budget_ratio_low(self):
        cfg = LCMConfig(summary_budget_ratio=-0.5)
        errors = cfg.validate()
        assert any("summary_budget_ratio" in e for e in errors)

    def test_catches_bad_fresh_tail_count(self):
        cfg = LCMConfig(fresh_tail_count=0)
        errors = cfg.validate()
        assert any("fresh_tail_count" in e for e in errors)

    def test_catches_bad_leaf_min_fanout(self):
        cfg = LCMConfig(leaf_min_fanout=1)
        errors = cfg.validate()
        assert any("leaf_min_fanout" in e for e in errors)

    def test_catches_bad_condensed_min_fanout(self):
        cfg = LCMConfig(condensed_min_fanout=1)
        errors = cfg.validate()
        assert any("condensed_min_fanout" in e for e in errors)

    def test_catches_bad_leaf_chunk_tokens(self):
        cfg = LCMConfig(leaf_chunk_tokens=0)
        errors = cfg.validate()
        assert any("leaf_chunk_tokens" in e for e in errors)

    def test_catches_bad_max_context_tokens(self):
        cfg = LCMConfig(max_context_tokens=-1)
        errors = cfg.validate()
        assert any("max_context_tokens" in e for e in errors)

    def test_multiple_errors(self):
        cfg = LCMConfig(
            context_threshold=2.0,
            summary_budget_ratio=-1.0,
            fresh_tail_count=0,
        )
        errors = cfg.validate()
        assert len(errors) == 3


# ------------------------------------------------------------------
# Path expansion
# ------------------------------------------------------------------

class TestPathExpansion:
    """db_path ~ expansion."""

    def test_expands_tilde(self):
        cfg = LCMConfig(db_path="~/.lossless-agent/lcm.db")
        resolved = cfg.resolved_db_path
        assert "~" not in resolved
        assert resolved == str(Path.home() / ".lossless-agent" / "lcm.db")

    def test_absolute_path_unchanged(self):
        cfg = LCMConfig(db_path="/tmp/test.db")
        assert cfg.resolved_db_path == "/tmp/test.db"

    def test_memory_path_unchanged(self):
        cfg = LCMConfig(db_path=":memory:")
        assert cfg.resolved_db_path == ":memory:"
