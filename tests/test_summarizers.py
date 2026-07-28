"""Tests for the summarizer factory, including Anthropic bearer/OAuth-token auth."""
import sys
import types

from lossless_agent import summarizers


def _fake_anthropic(recorder):
    """A stub `anthropic` module whose AsyncAnthropic records its constructor kwargs."""
    mod = types.ModuleType("anthropic")

    class AsyncAnthropic:
        def __init__(self, **kwargs):
            recorder.update(kwargs)

    mod.AsyncAnthropic = AsyncAnthropic
    return mod


def test_anthropic_uses_bearer_when_oauth_token_set(monkeypatch):
    rec = {}
    monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic(rec))
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-example")
    summarizers.make_anthropic_summarizer("claude-haiku-4-5-20251001")
    assert rec.get("auth_token") == "sk-ant-oat01-example"     # Bearer auth, not x-api-key
    assert rec.get("default_headers", {}).get("anthropic-beta") == "oauth-2025-04-20"
    assert "api_key" not in rec


def test_anthropic_oauth_beta_is_overridable(monkeypatch):
    rec = {}
    monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic(rec))
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-example")
    monkeypatch.setenv("LCM_ANTHROPIC_OAUTH_BETA", "oauth-2099-01-01")
    summarizers.make_anthropic_summarizer("")
    assert rec.get("default_headers", {}).get("anthropic-beta") == "oauth-2099-01-01"


def test_anthropic_uses_api_key_when_no_oauth(monkeypatch):
    rec = {}
    monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic(rec))
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    summarizers.make_anthropic_summarizer("")
    assert "auth_token" not in rec        # standard SDK x-api-key path
