"""FTS5 safety: detection, query sanitization, and fallback search."""
from __future__ import annotations

import re
import sqlite3
from typing import List


class FTSSafety:
    """Static methods for safe FTS5 usage with fallback."""

    # FTS5 operators that can cause syntax errors if used raw
    _FTS_OPERATORS = re.compile(r'\b(AND|OR|NOT|NEAR)\b')

    @staticmethod
    def detect_fts5_available(conn: sqlite3.Connection) -> bool:
        """Try creating a temp FTS5 table to detect availability."""
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_probe USING fts5(x)"
            )
            conn.execute("DROP TABLE IF EXISTS _fts5_probe")
            return True
        except Exception:
            return False

    @staticmethod
    def sanitize_query(query: str) -> str:
        """Strip FTS5 operators and fix unbalanced quotes.

        Returns '*' for empty/whitespace-only queries.
        """
        q = query.strip()
        if not q:
            return "*"

        # Escape FTS5 operators by quoting them
        q = FTSSafety._FTS_OPERATORS.sub(lambda m: f'"{m.group(0)}"', q)

        # Fix unbalanced quotes: count quotes, strip trailing unbalanced one
        quote_count = q.count('"')
        if quote_count % 2 != 0:
            # Remove the last quote to balance
            idx = q.rfind('"')
            q = q[:idx] + q[idx + 1:]

        q = q.strip()
        if not q:
            return "*"

        return q

    @staticmethod
    def search_with_fallback(
        conn: sqlite3.Connection,
        table: str,
        query: str,
        columns: List[str],
        limit: int = 50,
    ) -> list:
        """Try FTS5 search, fall back to LIKE-based search on failure."""
        sanitized = FTSSafety.sanitize_query(query)

        # Try FTS5 first
        try:
            col_list = ", ".join(columns)
            sql = f"SELECT {col_list} FROM {table} WHERE {table} MATCH ? ORDER BY rank LIMIT ?"
            rows = conn.execute(sql, (sanitized, limit)).fetchall()
            return rows
        except Exception:
            pass

        # Fallback: LIKE search on the base table (strip _fts suffix)
        base_table = table
        if base_table.endswith("_fts"):
            base_table = base_table[:-4]
        elif base_table.endswith("_fts_cjk"):
            base_table = base_table[:-8]

        try:
            col_list = ", ".join(columns)
            # Build OR conditions for LIKE across columns
            like_clauses = " OR ".join(f"{col} LIKE ?" for col in columns)
            like_param = f"%{query.strip()}%"
            params = [like_param] * len(columns) + [limit]
            sql = f"SELECT {col_list} FROM {base_table} WHERE {like_clauses} LIMIT ?"
            return conn.execute(sql, params).fetchall()
        except Exception:
            return []

    @staticmethod
    def detect_cjk(text: str) -> bool:
        """Return True if text contains CJK unified ideographs, hiragana, katakana, or Korean."""
        # CJK Unified: \u4e00-\u9fff
        # Hiragana: \u3040-\u309f
        # Katakana: \u30a0-\u30ff
        # Korean: \uac00-\ud7af
        return bool(re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', text))

    @staticmethod
    def route_search(
        conn: sqlite3.Connection,
        query: str,
        base_table: str,
        cjk_table: str,
        columns: List[str],
        limit: int = 50,
    ) -> list:
        """Route search to CJK table if query contains CJK characters."""
        table = cjk_table if FTSSafety.detect_cjk(query) else base_table
        return FTSSafety.search_with_fallback(conn, table, query, columns, limit)
