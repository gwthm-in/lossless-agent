"""Tests for LargeFileInterceptor using large_files_v2 (Feature 8)."""
from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock

import pytest

from lossless_agent.engine.large_files import LargeFileConfig, LargeFileInterceptor


@pytest.fixture
def config(tmp_path):
    return LargeFileConfig(
        token_threshold=100,
        summary_target_tokens=20,
        file_storage_dir=str(tmp_path / "files"),
    )


@pytest.fixture
def summarize_fn():
    fn = AsyncMock(return_value="This is a summary of the large content.")
    return fn


@pytest.fixture
def interceptor(db, summarize_fn, config):
    return LargeFileInterceptor(db=db, summarize_fn=summarize_fn, config=config)


@pytest.fixture
def conversation_id(db):
    db.conn.execute(
        "INSERT INTO conversations (session_key, title) VALUES (?, ?)",
        ("sess_lf2", "Large File V2 Test"),
    )
    db.conn.commit()
    return db.conn.execute(
        "SELECT id FROM conversations WHERE session_key='sess_lf2'"
    ).fetchone()[0]


class TestFileIdFormat:
    def test_file_id_starts_with_file_prefix(self, interceptor, conversation_id):
        big = "x" * 50000
        _, file_id = asyncio.get_event_loop().run_until_complete(
            interceptor.intercept(conversation_id, big, token_count=30000)
        )
        assert file_id is not None
        assert file_id.startswith("file_")
        # 'file_' + 16 hex chars = 21 chars total
        assert len(file_id) == 21

    def test_file_ids_are_unique(self, interceptor, conversation_id):
        ids = set()
        for _ in range(5):
            _, file_id = asyncio.get_event_loop().run_until_complete(
                interceptor.intercept(conversation_id, "x" * 50000, token_count=30000)
            )
            ids.add(file_id)
        assert len(ids) == 5


class TestSmallContentPassthroughV2:
    def test_small_content_passes_through(self, interceptor, conversation_id):
        content = "Small content"
        result_content, file_id = asyncio.get_event_loop().run_until_complete(
            interceptor.intercept(conversation_id, content, token_count=10)
        )
        assert result_content == content
        assert file_id is None


class TestLargeContentInterceptionV2:
    def test_large_content_gets_intercepted(self, interceptor, conversation_id):
        big = "x" * 50000
        result_content, file_id = asyncio.get_event_loop().run_until_complete(
            interceptor.intercept(conversation_id, big, token_count=30000)
        )
        assert file_id is not None
        assert result_content != big
        assert f"file_id={file_id}" in result_content

    def test_content_saved_to_filesystem(self, interceptor, conversation_id, config):
        big = "saved content " * 5000
        _, file_id = asyncio.get_event_loop().run_until_complete(
            interceptor.intercept(conversation_id, big, token_count=30000)
        )
        storage_path = os.path.join(
            os.path.expanduser(config.file_storage_dir), file_id
        )
        assert os.path.exists(storage_path)
        with open(storage_path, "r") as f:
            assert f.read() == big

    def test_stored_in_large_files_v2_table(self, interceptor, conversation_id, db):
        big = "x" * 50000
        _, file_id = asyncio.get_event_loop().run_until_complete(
            interceptor.intercept(conversation_id, big, token_count=30000)
        )
        row = db.conn.execute(
            "SELECT file_id, conversation_id, file_name, mime_type, byte_size, "
            "storage_uri, exploration_summary FROM large_files_v2 WHERE file_id = ?",
            (file_id,),
        ).fetchone()
        assert row is not None
        assert row[0] == file_id
        assert row[1] == conversation_id
        assert row[4] > 0  # byte_size


class TestGetFileV2:
    def test_get_file_retrieves_stored_content(self, interceptor, conversation_id):
        big = "stored content " * 5000
        _, file_id = asyncio.get_event_loop().run_until_complete(
            interceptor.intercept(conversation_id, big, token_count=30000)
        )
        result = interceptor.get_file(file_id)
        assert result is not None
        assert result["content"] == big
        assert result["file_id"] == file_id
        assert result["conversation_id"] == conversation_id
        assert result["exploration_summary"] == "This is a summary of the large content."

    def test_get_file_returns_none_for_missing_id(self, interceptor):
        assert interceptor.get_file("file_nonexistent0000") is None

    def test_get_files_for_conversation(self, interceptor, conversation_id):
        for i in range(3):
            asyncio.get_event_loop().run_until_complete(
                interceptor.intercept(
                    conversation_id, f"big content {i}" * 5000, token_count=30000
                )
            )
        files = interceptor.get_files_for_conversation(conversation_id)
        assert len(files) == 3
        assert all(f["conversation_id"] == conversation_id for f in files)
        assert all(f["file_id"].startswith("file_") for f in files)


class TestLargeFileConfigStorage:
    def test_config_has_file_storage_dir(self):
        config = LargeFileConfig()
        assert config.file_storage_dir == "~/.lossless-agent/files/"

    def test_config_custom_storage_dir(self, tmp_path):
        custom_dir = str(tmp_path / "custom")
        config = LargeFileConfig(file_storage_dir=custom_dir)
        assert config.file_storage_dir == custom_dir


class TestSchemaV2:
    def test_large_files_v2_table_exists(self, db):
        cur = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='large_files_v2'"
        )
        assert cur.fetchone() is not None
