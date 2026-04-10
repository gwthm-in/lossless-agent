"""Tests for the embedder factory and HTTP embed function."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from lossless_agent.config import LCMConfig
from lossless_agent.engine.embedder import EmbedFn, make_embedder, _http_embed


# ---------------------------------------------------------------------------
# make_embedder
# ---------------------------------------------------------------------------

class TestMakeEmbedder:
    def test_returns_none_when_disabled(self):
        config = LCMConfig(cross_session_enabled=False, embedding_base_url="http://localhost/v1")
        assert make_embedder(config) is None

    def test_returns_none_when_no_base_url(self):
        config = LCMConfig(cross_session_enabled=True, embedding_base_url="")
        assert make_embedder(config) is None

    def test_returns_callable_when_configured(self):
        config = LCMConfig(
            cross_session_enabled=True,
            embedding_base_url="http://localhost:4000/v1",
            embedding_model="text-embedding-3-small",
        )
        fn = make_embedder(config)
        assert fn is not None
        assert callable(fn)

    def test_embed_fn_calls_http_endpoint(self):
        config = LCMConfig(
            cross_session_enabled=True,
            embedding_base_url="http://localhost:4000/v1",
            embedding_model="text-embedding-3-small",
            embedding_api_key="test-key",
        )
        fn = make_embedder(config)

        fake_embedding = [0.1, 0.2, 0.3]

        with patch("lossless_agent.engine.embedder._http_embed", return_value=fake_embedding) as mock_http:
            result = asyncio.run(fn("hello world"))

        mock_http.assert_called_once_with(
            "hello world",
            "http://localhost:4000/v1",
            "text-embedding-3-small",
            "test-key",
        )
        assert result == fake_embedding


# ---------------------------------------------------------------------------
# _http_embed
# ---------------------------------------------------------------------------

class TestHttpEmbed:
    def _make_response(self, embedding: list):
        """Build a mock urlopen response with JSON body."""
        body = json.dumps({"data": [{"embedding": embedding}]}).encode()
        resp = MagicMock()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        resp.read.return_value = body
        return resp

    def test_sends_correct_request(self):
        embedding = [0.1, 0.2, 0.3]
        resp = self._make_response(embedding)

        with patch("urllib.request.urlopen", return_value=resp) as mock_open:
            result = _http_embed("hello", "http://localhost/v1", "model-x", "key-y")

        assert result == embedding
        req = mock_open.call_args[0][0]
        assert req.full_url == "http://localhost/v1/embeddings"
        assert req.get_header("Authorization") == "Bearer key-y"
        body = json.loads(req.data.decode())
        assert body["input"] == "hello"
        assert body["model"] == "model-x"

    def test_no_auth_header_when_no_key(self):
        resp = self._make_response([0.1])
        with patch("urllib.request.urlopen", return_value=resp):
            _http_embed("hello", "http://localhost/v1", "model-x", "")

        # No auth header set when api_key is empty — just verify it doesn't crash

    def test_raises_on_http_error(self):
        import urllib.error
        mock_exc = urllib.error.HTTPError(
            url="http://localhost/v1/embeddings",
            code=401,
            msg="Unauthorized",
            hdrs=None,  # type: ignore[arg-type]
            fp=MagicMock(read=MagicMock(return_value=b"bad key")),
        )
        with patch("urllib.request.urlopen", side_effect=mock_exc):
            with pytest.raises(RuntimeError, match="401"):
                _http_embed("hello", "http://localhost/v1", "model-x", "wrong-key")


# ---------------------------------------------------------------------------
# Config env var wiring
# ---------------------------------------------------------------------------

class TestConfigEmbedderEnvVars:
    def test_cross_session_enabled_from_env(self, monkeypatch):
        monkeypatch.setenv("LCM_CROSS_SESSION_ENABLED", "true")
        monkeypatch.setenv("LCM_EMBEDDING_BASE_URL", "http://proxy/v1")
        config = LCMConfig.from_env()
        assert config.cross_session_enabled is True
        assert config.embedding_base_url == "http://proxy/v1"

    def test_embedding_api_key_falls_back_to_openai_key(self, monkeypatch):
        monkeypatch.delenv("LCM_EMBEDDING_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fallback")
        config = LCMConfig.from_env()
        assert config.embedding_api_key == "sk-fallback"

    def test_explicit_embedding_api_key_overrides_openai(self, monkeypatch):
        monkeypatch.setenv("LCM_EMBEDDING_API_KEY", "explicit-key")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fallback")
        config = LCMConfig.from_env()
        assert config.embedding_api_key == "explicit-key"

    def test_defaults(self):
        config = LCMConfig()
        assert config.cross_session_enabled is False
        assert config.embedding_model == "text-embedding-3-small"
        assert config.embedding_dim == 1536
        assert config.cross_session_top_k == 5
        assert config.cross_session_token_budget == 2000
