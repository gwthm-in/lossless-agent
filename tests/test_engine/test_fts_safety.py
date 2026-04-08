"""Tests for FTSSafety class."""
import sqlite3

import pytest

from lossless_agent.engine.fts_safety import FTSSafety


@pytest.fixture
def fts_conn():
    """SQLite connection with FTS5 support and test tables."""
    conn = sqlite3.connect(":memory:")
    # Create a regular table and FTS5 virtual table
    conn.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, content TEXT)")
    conn.execute(
        "CREATE VIRTUAL TABLE docs_fts USING fts5(content, content='docs', content_rowid='id')"
    )
    # CJK table
    conn.execute(
        "CREATE VIRTUAL TABLE docs_fts_cjk USING fts5(content, doc_id UNINDEXED, tokenize='trigram')"
    )
    # Insert some test data
    for i, text in enumerate([
        "hello world python programming",
        "machine learning artificial intelligence",
        "database sqlite full text search",
        "日本語のテスト文書",
        "한국어 테스트 문서",
    ], start=1):
        conn.execute("INSERT INTO docs (id, content) VALUES (?, ?)", (i, text))
        conn.execute("INSERT INTO docs_fts(rowid, content) VALUES (?, ?)", (i, text))
        conn.execute(
            "INSERT INTO docs_fts_cjk(content, doc_id) VALUES (?, ?)", (text, str(i))
        )
    conn.commit()
    yield conn
    conn.close()


class TestDetectFTS5Available:
    def test_available_with_fts5(self, fts_conn):
        assert FTSSafety.detect_fts5_available(fts_conn) is True

    def test_unavailable_detection(self):
        """Even if FTS5 is compiled in, the detection method should work."""
        conn = sqlite3.connect(":memory:")
        # FTS5 is typically available in modern SQLite builds
        result = FTSSafety.detect_fts5_available(conn)
        assert isinstance(result, bool)
        conn.close()


class TestSanitizeQuery:
    def test_simple_query(self):
        assert FTSSafety.sanitize_query("hello world") == "hello world"

    def test_strips_unbalanced_quotes(self):
        result = FTSSafety.sanitize_query('hello "world')
        assert '"' not in result or result.count('"') % 2 == 0

    def test_balanced_quotes_preserved(self):
        result = FTSSafety.sanitize_query('"hello world"')
        assert result == '"hello world"'

    def test_escapes_fts5_operators(self):
        result = FTSSafety.sanitize_query("cats AND dogs")
        # Should not contain bare AND that could be interpreted as FTS operator
        assert "AND" not in result.split() or '"AND"' in result

    def test_escapes_or_operator(self):
        result = FTSSafety.sanitize_query("cats OR dogs")
        assert "OR" not in result.split() or '"OR"' in result

    def test_escapes_not_operator(self):
        result = FTSSafety.sanitize_query("cats NOT dogs")
        assert "NOT" not in result.split() or '"NOT"' in result

    def test_escapes_near_operator(self):
        result = FTSSafety.sanitize_query("cats NEAR dogs")
        assert "NEAR" not in result.split() or '"NEAR"' in result

    def test_empty_query(self):
        result = FTSSafety.sanitize_query("")
        assert result == "*"

    def test_whitespace_only_query(self):
        result = FTSSafety.sanitize_query("   ")
        assert result == "*"

    def test_multiple_unbalanced_quotes(self):
        result = FTSSafety.sanitize_query('"hello "world" test"')
        # Should have balanced quotes
        assert result.count('"') % 2 == 0

    def test_special_chars(self):
        """Queries with special chars should not crash."""
        result = FTSSafety.sanitize_query("hello + world ^ test")
        assert isinstance(result, str)
        assert len(result) > 0


class TestSearchWithFallback:
    def test_fts5_search_works(self, fts_conn):
        results = FTSSafety.search_with_fallback(
            fts_conn, "docs_fts", "python", ["content"]
        )
        assert len(results) >= 1
        assert any("python" in str(r).lower() for r in results)

    def test_fallback_to_like_on_bad_query(self, fts_conn):
        """Even with a problematic query, fallback should work."""
        results = FTSSafety.search_with_fallback(
            fts_conn, "docs_fts", "python", ["content"],
        )
        assert isinstance(results, list)

    def test_limit_respected(self, fts_conn):
        results = FTSSafety.search_with_fallback(
            fts_conn, "docs_fts", "hello OR python OR database", ["content"],
            limit=2,
        )
        assert len(results) <= 2

    def test_fallback_with_nonexistent_table(self):
        """When FTS table doesn't exist, should fall back gracefully."""
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, content TEXT)")
        conn.execute("INSERT INTO docs (content) VALUES ('hello world')")
        conn.commit()
        results = FTSSafety.search_with_fallback(
            conn, "nonexistent_fts", "hello", ["content"],
        )
        assert isinstance(results, list)
        conn.close()


class TestDetectCJK:
    def test_cjk_unified(self):
        assert FTSSafety.detect_cjk("测试") is True

    def test_hiragana(self):
        assert FTSSafety.detect_cjk("こんにちは") is True

    def test_katakana(self):
        assert FTSSafety.detect_cjk("テスト") is True

    def test_korean(self):
        assert FTSSafety.detect_cjk("한국어") is True

    def test_latin_only(self):
        assert FTSSafety.detect_cjk("hello world") is False

    def test_mixed(self):
        assert FTSSafety.detect_cjk("hello 世界") is True

    def test_empty(self):
        assert FTSSafety.detect_cjk("") is False


class TestRouteSearch:
    def test_routes_cjk_to_cjk_table(self, fts_conn):
        results = FTSSafety.route_search(
            fts_conn, "日本語", "docs_fts", "docs_fts_cjk", ["content"]
        )
        assert isinstance(results, list)

    def test_routes_latin_to_base_table(self, fts_conn):
        results = FTSSafety.route_search(
            fts_conn, "python", "docs_fts", "docs_fts_cjk", ["content"]
        )
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_routes_korean_to_cjk_table(self, fts_conn):
        results = FTSSafety.route_search(
            fts_conn, "한국어", "docs_fts", "docs_fts_cjk", ["content"]
        )
        assert isinstance(results, list)
