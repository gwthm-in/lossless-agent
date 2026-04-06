"""Shared fixtures for store tests."""
import os
import tempfile

import pytest

from lossless_agent.store.database import Database


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
