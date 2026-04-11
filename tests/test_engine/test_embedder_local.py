"""Tests for local embedding functions (fastembed)."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch
import pytest

from lossless_agent.engine.embedder import (
    make_local_embedder,
    make_local_batch_embedder,
    make_raw_vector_embedder,
    make_raw_vector_batch_embedder,
    _local_model,
)
from lossless_agent.config import LCMConfig


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

    def test_returns_none_when_api_mode(self):
        cfg = LCMConfig(
            raw_vector_enabled=True,
            raw_vector_use_local=False,
        )
        assert make_raw_vector_batch_embedder(cfg) is None
