"""Tests for new LCMConfig fields (Feature 4)."""
from __future__ import annotations

import os

import pytest

from lossless_agent.config import LCMConfig


# ------------------------------------------------------------------
# New defaults
# ------------------------------------------------------------------

class TestNewDefaults:
    def test_updated_defaults(self):
        cfg = LCMConfig()
        assert cfg.fresh_tail_count == 64
        assert cfg.leaf_min_fanout == 8
        assert cfg.condensed_min_fanout == 4
        assert cfg.leaf_target_tokens == 2400
        assert cfg.circuit_breaker_cooldown_ms == 1_800_000

    def test_new_field_defaults(self):
        cfg = LCMConfig()
        assert cfg.stateless_session_patterns == []
        assert cfg.skip_stateless_sessions is True
        assert cfg.new_session_retain_depth == 2
        assert cfg.bootstrap_max_tokens == 6000
        assert cfg.large_file_summary_provider == ""
        assert cfg.large_file_summary_model == ""
        assert cfg.delegation_timeout_ms == 120_000
        assert cfg.prune_heartbeat_ok is False
        assert cfg.max_assembly_token_budget is None
        assert cfg.custom_instructions == ""
        assert cfg.timezone == ""


# ------------------------------------------------------------------
# Env var reading for new fields
# ------------------------------------------------------------------

class TestNewEnvVars:
    def test_stateless_session_patterns(self, monkeypatch):
        monkeypatch.setenv("LCM_STATELESS_SESSION_PATTERNS", "temp-*,ephemeral-*")
        cfg = LCMConfig.from_env()
        assert cfg.stateless_session_patterns == ["temp-*", "ephemeral-*"]

    def test_skip_stateless_sessions(self, monkeypatch):
        monkeypatch.setenv("LCM_SKIP_STATELESS_SESSIONS", "false")
        cfg = LCMConfig.from_env()
        assert cfg.skip_stateless_sessions is False

    def test_new_session_retain_depth(self, monkeypatch):
        monkeypatch.setenv("LCM_NEW_SESSION_RETAIN_DEPTH", "3")
        cfg = LCMConfig.from_env()
        assert cfg.new_session_retain_depth == 3

    def test_bootstrap_max_tokens(self, monkeypatch):
        monkeypatch.setenv("LCM_BOOTSTRAP_MAX_TOKENS", "8000")
        cfg = LCMConfig.from_env()
        assert cfg.bootstrap_max_tokens == 8000

    def test_large_file_summary_provider(self, monkeypatch):
        monkeypatch.setenv("LCM_LARGE_FILE_SUMMARY_PROVIDER", "openai")
        cfg = LCMConfig.from_env()
        assert cfg.large_file_summary_provider == "openai"

    def test_large_file_summary_model(self, monkeypatch):
        monkeypatch.setenv("LCM_LARGE_FILE_SUMMARY_MODEL", "gpt-4o")
        cfg = LCMConfig.from_env()
        assert cfg.large_file_summary_model == "gpt-4o"

    def test_delegation_timeout_ms(self, monkeypatch):
        monkeypatch.setenv("LCM_DELEGATION_TIMEOUT_MS", "60000")
        cfg = LCMConfig.from_env()
        assert cfg.delegation_timeout_ms == 60_000

    def test_prune_heartbeat_ok(self, monkeypatch):
        monkeypatch.setenv("LCM_PRUNE_HEARTBEAT_OK", "true")
        cfg = LCMConfig.from_env()
        assert cfg.prune_heartbeat_ok is True

    def test_max_assembly_token_budget(self, monkeypatch):
        monkeypatch.setenv("LCM_MAX_ASSEMBLY_TOKEN_BUDGET", "50000")
        cfg = LCMConfig.from_env()
        assert cfg.max_assembly_token_budget == 50_000

    def test_max_assembly_token_budget_empty(self, monkeypatch):
        monkeypatch.setenv("LCM_MAX_ASSEMBLY_TOKEN_BUDGET", "")
        cfg = LCMConfig.from_env()
        assert cfg.max_assembly_token_budget is None

    def test_custom_instructions(self, monkeypatch):
        monkeypatch.setenv("LCM_CUSTOM_INSTRUCTIONS", "Focus on code changes")
        cfg = LCMConfig.from_env()
        assert cfg.custom_instructions == "Focus on code changes"

    def test_timezone(self, monkeypatch):
        monkeypatch.setenv("LCM_TIMEZONE", "America/New_York")
        cfg = LCMConfig.from_env()
        assert cfg.timezone == "America/New_York"


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

class TestNewValidation:
    def test_valid_new_config(self):
        cfg = LCMConfig()
        assert cfg.validate() == []

    def test_negative_delegation_timeout(self):
        cfg = LCMConfig(delegation_timeout_ms=-1)
        errors = cfg.validate()
        assert any("delegation_timeout_ms" in e for e in errors)

    def test_negative_new_session_retain_depth(self):
        cfg = LCMConfig(new_session_retain_depth=-1)
        errors = cfg.validate()
        assert any("new_session_retain_depth" in e for e in errors)


# ------------------------------------------------------------------
# effective_bootstrap_max_tokens
# ------------------------------------------------------------------

class TestEffectiveBootstrapMaxTokens:
    def test_default_value(self):
        cfg = LCMConfig()
        # max(6000, 20000 * 0.3) = max(6000, 6000) = 6000
        assert cfg.effective_bootstrap_max_tokens == 6000

    def test_with_large_leaf_chunk(self):
        cfg = LCMConfig(leaf_chunk_tokens=40_000, bootstrap_max_tokens=6000)
        # max(6000, 40000 * 0.3) = max(6000, 12000) = 12000
        assert cfg.effective_bootstrap_max_tokens == 12000

    def test_explicit_higher_value(self):
        cfg = LCMConfig(bootstrap_max_tokens=20000)
        # max(20000, 20000 * 0.3) = max(20000, 6000) = 20000
        assert cfg.effective_bootstrap_max_tokens == 20000


# ------------------------------------------------------------------
# from_dict with new fields
# ------------------------------------------------------------------

class TestFromDictNewFields:
    def test_new_fields_from_dict(self):
        d = {
            "prune_heartbeat_ok": True,
            "custom_instructions": "Be brief",
            "timezone": "UTC",
            "stateless_session_patterns": ["temp-*"],
        }
        cfg = LCMConfig.from_dict(d)
        assert cfg.prune_heartbeat_ok is True
        assert cfg.custom_instructions == "Be brief"
        assert cfg.timezone == "UTC"
        assert cfg.stateless_session_patterns == ["temp-*"]


# ------------------------------------------------------------------
# merge with new fields
# ------------------------------------------------------------------

class TestMergeNewFields:
    def test_merge_new_fields(self):
        base = LCMConfig()
        merged = LCMConfig.merge(base, {
            "prune_heartbeat_ok": True,
            "timezone": "Europe/London",
        })
        assert merged.prune_heartbeat_ok is True
        assert merged.timezone == "Europe/London"
        assert base.prune_heartbeat_ok is False
