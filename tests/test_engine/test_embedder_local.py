"""Tests for local embedding functions (fastembed)."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch
import pytest

from lossless_agent.engine.embedder import (
    make_raw_vector_embedder,
    make_raw_vector_batch_embedder,
    _get_local_model,
)
from lossless_agent.config import LCMConfig
import lossless_agent.engine.embedder as embedder_module


class TestMakeRawVectorEmbedder:
    def test_returns_none_when_disabled(self):
        cfg = LCMConfig(raw_vector_enabled=False)
        assert make_raw_vector_embedder(cfg) is None

    def test_returns_none_batch_when_disabled(self):
        cfg = LCMConfig(raw_vector_enabled=False)
        assert make_raw_vector_batch_embedder(cfg) is None

    def test_returns_function_when_enabled_local(self):
        cfg = LCMConfig(raw_vector_enabled=True, raw_vector_use_local=True)
        # This will fail if fastembed is not installed, but should return a callable
        try:
            fn = make_raw_vector_embedder(cfg)
            assert callable(fn)
        except ImportError:
            pytest.skip("fastembed not installed")

    def test_returns_none_when_api_mode_no_url(self):
        cfg = LCMConfig(
            raw_vector_enabled=True,
            raw_vector_use_local=False,
            cross_session_enabled=False,
            embedding_base_url="",
        )
        assert make_raw_vector_embedder(cfg) is None


class TestMakeRawVectorBatchEmbedder:
    def test_returns_function_when_enabled_local(self):
        cfg = LCMConfig(raw_vector_enabled=True, raw_vector_use_local=True)
        try:
            fn = make_raw_vector_batch_embedder(cfg)
            assert callable(fn)
        except ImportError:
            pytest.skip("fastembed not installed")

    def test_api_mode_falls_back_to_sequential_single_embed(self):
        """API mode must return a callable, not None — no silent skip."""
        cfg = LCMConfig(
            raw_vector_enabled=True,
            raw_vector_use_local=False,
            cross_session_enabled=True,  # required for make_embedder to return non-None
            embedding_base_url="http://localhost/v1",
            embedding_model="text-embedding-3-small",
            embedding_api_key="key",
        )
        fn = make_raw_vector_batch_embedder(cfg)
        assert fn is not None
        assert callable(fn)

    def test_api_mode_batch_calls_single_embed_for_each_text(self):
        """Batch wrapper must embed every text in the list."""
        cfg = LCMConfig(
            raw_vector_enabled=True,
            raw_vector_use_local=False,
            cross_session_enabled=True,
            embedding_base_url="http://localhost/v1",
            embedding_model="text-embedding-3-small",
            embedding_api_key="key",
        )
        fn = make_raw_vector_batch_embedder(cfg)
        fake_emb = [0.1, 0.2, 0.3]
        with patch(
            "lossless_agent.engine.embedder._http_embed", return_value=fake_emb
        ):
            results = asyncio.run(fn(["hello", "world"]))
        assert len(results) == 2
        assert results[0] == fake_emb
        assert results[1] == fake_emb


class TestLocalModelSingleton:
    def setup_method(self):
        """Reset the singleton before each test."""
        embedder_module._local_model = None
        embedder_module._local_model_name = ""

    def teardown_method(self):
        embedder_module._local_model = None
        embedder_module._local_model_name = ""

    def test_singleton_reloads_on_model_name_change(self):
        """Calling _get_local_model with a different name must load a new model."""
        mock_model_a = MagicMock()
        mock_model_b = MagicMock()

        models = {"model-a": mock_model_a, "model-b": mock_model_b}
        call_counts = {"model-a": 0, "model-b": 0}

        def fake_text_embedding(name):
            call_counts[name] = call_counts.get(name, 0) + 1
            return models[name]

        fake_fastembed = MagicMock()
        fake_fastembed.TextEmbedding = fake_text_embedding

        with patch.dict("sys.modules", {"fastembed": fake_fastembed}):
            # First call: load model-a
            result_a1 = _get_local_model("model-a")
            assert result_a1 is mock_model_a
            assert call_counts["model-a"] == 1

            # Second call: same name — must return cached, NOT reload
            result_a2 = _get_local_model("model-a")
            assert result_a2 is mock_model_a
            assert call_counts["model-a"] == 1  # still 1, no reload

            # Third call: different name — must reload
            result_b = _get_local_model("model-b")
            assert result_b is mock_model_b
            assert call_counts["model-b"] == 1

            # Singleton state updated to model-b
            assert embedder_module._local_model is mock_model_b
            assert embedder_module._local_model_name == "model-b"
