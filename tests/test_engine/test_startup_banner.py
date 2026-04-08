"""Tests for the startup banner module."""
from __future__ import annotations

import logging

import pytest

from lossless_agent.engine.startup_banner import StartupBanner
from lossless_agent.config import LCMConfig


@pytest.fixture(autouse=True)
def _reset_banner():
    """Reset the banner state before each test."""
    StartupBanner.reset()
    yield
    StartupBanner.reset()


# ------------------------------------------------------------------
# emit_once
# ------------------------------------------------------------------

class TestEmitOnce:
    def test_emits_first_time(self):
        assert StartupBanner.emit_once("key1", "hello") is True

    def test_does_not_emit_second_time(self):
        StartupBanner.emit_once("key1", "hello")
        assert StartupBanner.emit_once("key1", "hello again") is False

    def test_different_keys_both_emit(self):
        assert StartupBanner.emit_once("key1", "msg1") is True
        assert StartupBanner.emit_once("key2", "msg2") is True

    def test_logs_message(self, caplog):
        with caplog.at_level(logging.INFO, logger="lossless_agent.engine.startup_banner"):
            StartupBanner.emit_once("log_test", "Test message here")
        assert "Test message here" in caplog.text


# ------------------------------------------------------------------
# reset
# ------------------------------------------------------------------

class TestReset:
    def test_reset_allows_reemit(self):
        StartupBanner.emit_once("key1", "first")
        StartupBanner.reset()
        assert StartupBanner.emit_once("key1", "second") is True


# ------------------------------------------------------------------
# log_plugin_loaded
# ------------------------------------------------------------------

class TestLogPluginLoaded:
    def test_logs_plugin_info(self, caplog):
        cfg = LCMConfig(enabled=True, db_path="/tmp/test.db", context_threshold=0.8)
        with caplog.at_level(logging.INFO, logger="lossless_agent.engine.startup_banner"):
            StartupBanner.log_plugin_loaded(cfg)
        assert "[lcm] Plugin loaded" in caplog.text
        assert "enabled=True" in caplog.text
        assert "db=/tmp/test.db" in caplog.text
        assert "threshold=0.8" in caplog.text

    def test_only_emits_once(self):
        cfg = LCMConfig()
        StartupBanner.log_plugin_loaded(cfg)
        # Second call should not re-emit
        assert StartupBanner.emit_once("plugin_loaded", "different") is False


# ------------------------------------------------------------------
# log_compaction_model
# ------------------------------------------------------------------

class TestLogCompactionModel:
    def test_logs_default_model(self, caplog):
        cfg = LCMConfig()
        with caplog.at_level(logging.INFO, logger="lossless_agent.engine.startup_banner"):
            StartupBanner.log_compaction_model(cfg)
        assert "[lcm] Compaction model: (default) via (default)" in caplog.text

    def test_logs_custom_model(self, caplog):
        cfg = LCMConfig(summary_model="gpt-4o-mini", summary_provider="openai")
        with caplog.at_level(logging.INFO, logger="lossless_agent.engine.startup_banner"):
            StartupBanner.log_compaction_model(cfg)
        assert "gpt-4o-mini" in caplog.text
        assert "openai" in caplog.text


# ------------------------------------------------------------------
# log_session_patterns
# ------------------------------------------------------------------

class TestLogSessionPatterns:
    def test_no_patterns_no_log(self, caplog):
        cfg = LCMConfig()
        with caplog.at_level(logging.INFO, logger="lossless_agent.engine.startup_banner"):
            StartupBanner.log_session_patterns(cfg)
        assert "[lcm] Session patterns" not in caplog.text

    def test_ignore_patterns_logged(self, caplog):
        cfg = LCMConfig(ignore_session_patterns=["test-*", "debug-*"])
        with caplog.at_level(logging.INFO, logger="lossless_agent.engine.startup_banner"):
            StartupBanner.log_session_patterns(cfg)
        assert "[lcm] Session patterns" in caplog.text
        assert "ignore=" in caplog.text

    def test_stateless_patterns_logged(self, caplog):
        cfg = LCMConfig(stateless_session_patterns=["ephemeral-*"])
        with caplog.at_level(logging.INFO, logger="lossless_agent.engine.startup_banner"):
            StartupBanner.log_session_patterns(cfg)
        assert "stateless=" in caplog.text

    def test_both_patterns_logged(self, caplog):
        cfg = LCMConfig(
            ignore_session_patterns=["test-*"],
            stateless_session_patterns=["ephemeral-*"],
        )
        with caplog.at_level(logging.INFO, logger="lossless_agent.engine.startup_banner"):
            StartupBanner.log_session_patterns(cfg)
        assert "ignore=" in caplog.text
        assert "stateless=" in caplog.text
