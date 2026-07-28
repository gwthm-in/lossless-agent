"""Tests for the summarizer factory, including Anthropic bearer/OAuth-token auth."""
import sys
import types

import pytest

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


def test_anthropic_oauth_sends_bearer_not_x_api_key(monkeypatch):
    """Real-SDK guard (skips without the anthropic extra): the OAuth construction must send
    Authorization: Bearer even when ANTHROPIC_API_KEY is in the env — never x-api-key. Passing
    api_key="" here would emit BOTH headers, so the factory deliberately omits api_key."""
    anthropic = pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-garbage-should-be-ignored")
    client = anthropic.AsyncAnthropic(  # same construction make_anthropic_summarizer uses
        auth_token="sk-ant-oat01-example",
        default_headers={"anthropic-beta": "oauth-2025-04-20"},
    )
    headers = {k.lower() for k in client.auth_headers}
    assert "authorization" in headers and "x-api-key" not in headers
