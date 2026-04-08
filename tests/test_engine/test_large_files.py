"""Tests for large file interception and storage."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from lossless_agent.engine.large_files import LargeFileConfig, LargeFileInterceptor


@pytest.fixture
def config():
    return LargeFileConfig(token_threshold=100, summary_target_tokens=20)


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
        ("sess_lf", "Large File Test"),
    )
    db.conn.commit()
    return db.conn.execute(
        "SELECT id FROM conversations WHERE session_key='sess_lf'"
    ).fetchone()[0]


class TestSmallContentPassthrough:
    def test_small_content_passes_through_unchanged(self, interceptor, conversation_id):
        content = "Small content"
        result_content, file_id = asyncio.get_event_loop().run_until_complete(
            interceptor.intercept(conversation_id, content, token_count=10)
        )
        assert result_content == content
        assert file_id is None

    def test_summarize_fn_not_called_for_small_content(
        self, interceptor, summarize_fn, conversation_id
    ):
        asyncio.get_event_loop().run_until_complete(
            interceptor.intercept(conversation_id, "tiny", token_count=5)
        )
        summarize_fn.assert_not_called()


class TestLargeContentInterception:
    def test_large_content_gets_intercepted(self, interceptor, conversation_id):
        big = "x" * 50000
        result_content, file_id = asyncio.get_event_loop().run_until_complete(
            interceptor.intercept(conversation_id, big, token_count=30000)
        )
        assert file_id is not None
        assert result_content != big

    def test_replacement_format(self, interceptor, conversation_id, summarize_fn):
        big = "x" * 50000
        result_content, file_id = asyncio.get_event_loop().run_until_complete(
            interceptor.intercept(conversation_id, big, token_count=30000)
        )
        assert f"file_id={file_id}" in result_content
        assert "Summary: This is a summary of the large content." in result_content
        assert "lcm_expand" in result_content

    def test_summarize_fn_called_for_large_content(
        self, interceptor, summarize_fn, conversation_id, config
    ):
        big = "x" * 50000
        asyncio.get_event_loop().run_until_complete(
            interceptor.intercept(conversation_id, big, token_count=30000)
        )
        summarize_fn.assert_called_once_with(big, config.summary_target_tokens)


class TestGetFile:
    def test_get_file_retrieves_stored_content(self, interceptor, conversation_id):
        big = "stored content " * 5000
        _, file_id = asyncio.get_event_loop().run_until_complete(
            interceptor.intercept(conversation_id, big, token_count=30000)
        )
        result = interceptor.get_file(file_id)
        assert result is not None
        assert result["content"] == big
        assert result["token_count"] == 30000
        assert result["summary"] == "This is a summary of the large content."
        assert result["conversation_id"] == conversation_id

    def test_get_file_returns_none_for_missing_id(self, interceptor):
        assert interceptor.get_file(99999) is None

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


class TestMultipleConversations:
    def test_files_scoped_to_conversation(self, db, summarize_fn, config):
        interceptor = LargeFileInterceptor(db=db, summarize_fn=summarize_fn, config=config)
        # Create two conversations
        db.conn.execute(
            "INSERT INTO conversations (session_key, title) VALUES (?, ?)",
            ("sess_a", "Conv A"),
        )
        db.conn.execute(
            "INSERT INTO conversations (session_key, title) VALUES (?, ?)",
            ("sess_b", "Conv B"),
        )
        db.conn.commit()
        conv_a = db.conn.execute(
            "SELECT id FROM conversations WHERE session_key='sess_a'"
        ).fetchone()[0]
        conv_b = db.conn.execute(
            "SELECT id FROM conversations WHERE session_key='sess_b'"
        ).fetchone()[0]

        asyncio.get_event_loop().run_until_complete(
            interceptor.intercept(conv_a, "big a" * 10000, token_count=30000)
        )
        asyncio.get_event_loop().run_until_complete(
            interceptor.intercept(conv_b, "big b" * 10000, token_count=30000)
        )

        assert len(interceptor.get_files_for_conversation(conv_a)) == 1
        assert len(interceptor.get_files_for_conversation(conv_b)) == 1


class TestSchema:
    def test_large_files_table_exists(self, db):
        cur = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='large_files'"
        )
        assert cur.fetchone() is not None

    def test_schema_version_is_four(self, db):
        row = db.conn.execute("SELECT version FROM schema_version").fetchone()
        assert row[0] == 4
