"""Shared fixtures for store tests."""

import pytest

from lossless_agent.store.database import Database


@pytest.fixture(autouse=True)
def _isolate_metrics(monkeypatch, tmp_path):
    """Point the usage-metrics emitter at a per-test tmp dir so the test suite never writes JSONL
    files into the real ``~/.lossless-agent/metrics`` (metrics defaults to ON). Tests that want to
    assert on emitted events set ``LCM_METRICS_DIR``/``LCM_METRICS_ENABLED`` themselves, which
    simply overrides this default."""
    monkeypatch.setenv("LCM_METRICS_DIR", str(tmp_path / "metrics"))


@pytest.fixture
def db():
    """Provide a fresh in-memory Database for each test."""
    database = Database(":memory:")
    yield database
    database.close()


@pytest.fixture
def db_file(tmp_path):
    """Provide a fresh file-backed Database for each test."""
    path = str(tmp_path / "test.db")
    database = Database(path)
    yield database
    database.close()
