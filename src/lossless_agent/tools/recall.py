"""Recall tools for searching and navigating the memory DAG."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from lossless_agent.engine.fts_safety import FTSSafety
from lossless_agent.store.database import Database
from lossless_agent.store.models import Message, Summary


# ── CJK detection & FTS5 safety (delegated to FTSSafety) ──────────


def _contains_cjk(text: str) -> bool:
    """Return True if *text* contains any CJK characters."""
    return FTSSafety.detect_cjk(text)


def _sanitize_fts5_query(query: str) -> str:
    """Strip FTS5 special syntax characters to prevent query errors.

    Delegates to FTSSafety.sanitize_query, then strips remaining FTS5
    metacharacters for extra safety in the recall context.
    """
    if not query:
        return query
    sanitized = FTSSafety.sanitize_query(query)
    # Extra: strip remaining FTS5 metacharacters
    sanitized = re.sub(r'["\*\(\)\-\+\^:]', " ", sanitized)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    # Remove FTS5 boolean keywords at word boundaries
    sanitized = re.sub(r"\bNOT\b", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\bOR\b", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\bAND\b", "", sanitized, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", sanitized).strip()


@dataclass
class GrepResult:
    type: str  # 'message' or 'summary'
    id: Union[str, int]
    content_snippet: str
    conversation_id: int
    metadata: Dict[str, Any]
    created_at: Optional[str] = None


@dataclass
class DescribeResult:
    summary_id: str
    kind: str
    depth: int
    content: str
    token_count: int
    source_token_count: int
    earliest_at: str
    latest_at: str
    child_ids: List[str]
    source_message_count: int


@dataclass
class ExpandResult:
    summary_id: str
    kind: str
    children: List[Union[Message, Summary]]


def _truncate(text: str, max_len: int = 200) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _like_search(
    db: Database,
    table: str,
    query: str,
    conversation_id: Optional[int],
    limit: int,
    result_type: str,
    since: Optional[str] = None,
    before: Optional[str] = None,
) -> List[GrepResult]:
    """Generic LIKE fallback search for messages or summaries."""
    results: List[GrepResult] = []
    if table == "messages":
        sql = (
            "SELECT id, conversation_id, content, role, seq, created_at "
            "FROM messages WHERE content LIKE ?"
        )
        params: list = [f"%{query}%"]
        if conversation_id is not None:
            sql += " AND conversation_id = ?"
            params.append(conversation_id)
        if since is not None:
            sql += " AND created_at >= ?"
            params.append(since)
        if before is not None:
            sql += " AND created_at < ?"
            params.append(before)
        sql += " LIMIT ?"
        params.append(limit)
        for row in db.conn.execute(sql, params).fetchall():
            results.append(
                GrepResult(
                    type="message",
                    id=row[0],
                    content_snippet=_truncate(row[2]),
                    conversation_id=row[1],
                    metadata={"role": row[3], "seq": row[4]},
                    created_at=row[5],
                )
            )
    else:
        sql = (
            "SELECT summary_id, conversation_id, content, kind, depth, created_at "
            "FROM summaries WHERE content LIKE ?"
        )
        params = [f"%{query}%"]
        if conversation_id is not None:
            sql += " AND conversation_id = ?"
            params.append(conversation_id)
        if since is not None:
            sql += " AND created_at >= ?"
            params.append(since)
        if before is not None:
            sql += " AND created_at < ?"
            params.append(before)
        sql += " LIMIT ?"
        params.append(limit)
        for row in db.conn.execute(sql, params).fetchall():
            results.append(
                GrepResult(
                    type="summary",
                    id=row[0],
                    content_snippet=_truncate(row[2]),
                    conversation_id=row[1],
                    metadata={"kind": row[3], "depth": row[4]},
                    created_at=row[5],
                )
            )
    return results


def _fts_search_messages(
    db: Database,
    query: str,
    conversation_id: Optional[int],
    limit: int,
    since: Optional[str] = None,
    before: Optional[str] = None,
) -> List[GrepResult]:
    """Search messages via FTS5 with LIKE fallback."""
    # CJK: default FTS5 tokenizer can't handle CJK, use LIKE directly
    if _contains_cjk(query):
        return _like_search(db, "messages", query, conversation_id, limit, "message", since=since, before=before)

    safe_query = _sanitize_fts5_query(query)
    if not safe_query:
        return _like_search(db, "messages", query, conversation_id, limit, "message", since=since, before=before)

    try:
        sql = (
            "SELECT m.id, m.conversation_id, m.content, m.role, m.seq, m.created_at "
            "FROM messages m "
            "JOIN messages_fts f ON m.id = f.rowid "
            "WHERE messages_fts MATCH ?"
        )
        params: list = [safe_query]
        if conversation_id is not None:
            sql += " AND m.conversation_id = ?"
            params.append(conversation_id)
        if since is not None:
            sql += " AND m.created_at >= ?"
            params.append(since)
        if before is not None:
            sql += " AND m.created_at < ?"
            params.append(before)
        sql += " ORDER BY f.rank LIMIT ?"
        params.append(limit)

        results: List[GrepResult] = []
        for row in db.conn.execute(sql, params).fetchall():
            results.append(
                GrepResult(
                    type="message",
                    id=row[0],
                    content_snippet=_truncate(row[2]),
                    conversation_id=row[1],
                    metadata={"role": row[3], "seq": row[4]},
                    created_at=row[5],
                )
            )
        return results
    except Exception:
        return _like_search(db, "messages", query, conversation_id, limit, "message", since=since, before=before)


def _fts_search_summaries(
    db: Database,
    query: str,
    conversation_id: Optional[int],
    limit: int,
    since: Optional[str] = None,
    before: Optional[str] = None,
) -> List[GrepResult]:
    """Search summaries via FTS5, routing CJK to trigram table with LIKE fallback."""
    is_cjk = _contains_cjk(query)
    results: List[GrepResult] = []

    try:
        if is_cjk:
            # Use trigram CJK FTS table
            sql = (
                "SELECT s.summary_id, s.conversation_id, s.content, s.kind, s.depth, s.created_at "
                "FROM summaries s "
                "JOIN summaries_fts_cjk fc ON s.summary_id = fc.summary_id "
                "WHERE summaries_fts_cjk MATCH ?"
            )
            params: list = [query]
        else:
            safe_query = _sanitize_fts5_query(query)
            if not safe_query:
                return results
            sql = (
                "SELECT s.summary_id, s.conversation_id, s.content, s.kind, s.depth, s.created_at "
                "FROM summaries s "
                "JOIN summaries_fts f ON s.rowid = f.rowid "
                "WHERE summaries_fts MATCH ?"
            )
            params = [safe_query]

        if conversation_id is not None:
            sql += " AND s.conversation_id = ?"
            params.append(conversation_id)
        if since is not None:
            sql += " AND s.created_at >= ?"
            params.append(since)
        if before is not None:
            sql += " AND s.created_at < ?"
            params.append(before)
        sql += " LIMIT ?"
        params.append(limit)

        for row in db.conn.execute(sql, params).fetchall():
            results.append(
                GrepResult(
                    type="summary",
                    id=row[0],
                    content_snippet=_truncate(row[2]),
                    conversation_id=row[1],
                    metadata={"kind": row[3], "depth": row[4]},
                    created_at=row[5],
                )
            )
        # If CJK FTS returned nothing (e.g. query too short for trigrams),
        # fall through to LIKE
        if results or not is_cjk:
            return results
    except Exception:
        pass  # fall through to LIKE

    # LIKE fallback
    return _like_search(db, "summaries", query, conversation_id, limit, "summary", since=since, before=before)


def _regex_search(
    db: Database,
    query: str,
    scope: str,
    conversation_id: Optional[int],
    limit: int,
    since: Optional[str] = None,
    before: Optional[str] = None,
) -> List[GrepResult]:
    """Search messages and/or summaries using LIKE-based pattern matching with Python re post-filter."""
    results: List[GrepResult] = []
    pattern = re.compile(query)

    if scope in ("all", "messages"):
        sql = (
            "SELECT id, conversation_id, content, role, seq, created_at "
            "FROM messages WHERE 1=1"
        )
        params: list = []
        if conversation_id is not None:
            sql += " AND conversation_id = ?"
            params.append(conversation_id)
        if since is not None:
            sql += " AND created_at >= ?"
            params.append(since)
        if before is not None:
            sql += " AND created_at < ?"
            params.append(before)
        for row in db.conn.execute(sql, params).fetchall():
            if pattern.search(row[2] or ""):
                results.append(
                    GrepResult(
                        type="message",
                        id=row[0],
                        content_snippet=_truncate(row[2]),
                        conversation_id=row[1],
                        metadata={"role": row[3], "seq": row[4]},
                        created_at=row[5],
                    )
                )
                if len(results) >= limit:
                    return results

    if scope in ("all", "summaries"):
        sql = (
            "SELECT summary_id, conversation_id, content, kind, depth, created_at "
            "FROM summaries WHERE 1=1"
        )
        params = []
        if conversation_id is not None:
            sql += " AND conversation_id = ?"
            params.append(conversation_id)
        if since is not None:
            sql += " AND created_at >= ?"
            params.append(since)
        if before is not None:
            sql += " AND created_at < ?"
            params.append(before)
        for row in db.conn.execute(sql, params).fetchall():
            if pattern.search(row[2] or ""):
                results.append(
                    GrepResult(
                        type="summary",
                        id=row[0],
                        content_snippet=_truncate(row[2]),
                        conversation_id=row[1],
                        metadata={"kind": row[3], "depth": row[4]},
                        created_at=row[5],
                    )
                )
                if len(results) >= limit:
                    return results

    return results[:limit]


def lcm_grep(
    db: Database,
    query: str,
    scope: str = "all",
    conversation_id: Optional[int] = None,
    limit: int = 20,
    mode: str = "full_text",
    since: Optional[str] = None,
    before: Optional[str] = None,
) -> List[GrepResult]:
    """Search messages and/or summaries via FTS5.

    CJK queries are automatically routed to the trigram FTS table for
    summaries. All FTS5 queries are sanitized. If FTS5 fails, falls
    back to LIKE search.

    Args:
        mode: 'full_text' (default) for FTS5 search, 'regex' for regex matching
        since: Optional ISO timestamp to filter results created at or after
        before: Optional ISO timestamp to filter results created before
    """
    if mode == "regex":
        return _regex_search(db, query, scope, conversation_id, limit, since=since, before=before)

    results: List[GrepResult] = []

    is_cjk = _contains_cjk(query)

    if scope in ("all", "messages"):
        if is_cjk:
            # CJK: use LIKE directly for messages (default tokenizer can't handle CJK)
            results.extend(_fts_search_messages(db, query, conversation_id, limit, since=since, before=before))
        else:
            results.extend(_fts_search_messages(db, query, conversation_id, limit, since=since, before=before))

    if scope in ("all", "summaries"):
        results.extend(_fts_search_summaries(db, query, conversation_id, limit, since=since, before=before))

    # Global limit when scope='all'
    return results[:limit]


def lcm_describe(db: Database, summary_id: str) -> Optional[DescribeResult]:
    """Look up a summary node by ID and return its full metadata."""
    row = db.conn.execute(
        "SELECT summary_id, conversation_id, kind, depth, content, "
        "token_count, source_token_count, earliest_at, latest_at "
        "FROM summaries WHERE summary_id = ?",
        (summary_id,),
    ).fetchone()
    if row is None:
        return None

    kind = row[2]

    # Get child_ids for condensed summaries
    child_ids: List[str] = []
    if kind == "condensed":
        child_rows = db.conn.execute(
            "SELECT child_id FROM summary_parents WHERE parent_id = ?",
            (summary_id,),
        ).fetchall()
        child_ids = [r[0] for r in child_rows]

    # Get source message count for leaf summaries
    source_message_count = 0
    if kind == "leaf":
        count_row = db.conn.execute(
            "SELECT COUNT(*) FROM summary_messages WHERE summary_id = ?",
            (summary_id,),
        ).fetchone()
        source_message_count = count_row[0]

    return DescribeResult(
        summary_id=row[0],
        kind=kind,
        depth=row[3],
        content=row[4],
        token_count=row[5],
        source_token_count=row[6],
        earliest_at=row[7],
        latest_at=row[8],
        child_ids=child_ids,
        source_message_count=source_message_count,
    )


def lcm_expand(db: Database, summary_id: str) -> Optional[ExpandResult]:
    """Expand a summary: return source messages (leaf) or child summaries (condensed)."""
    row = db.conn.execute(
        "SELECT summary_id, kind FROM summaries WHERE summary_id = ?",
        (summary_id,),
    ).fetchone()
    if row is None:
        return None

    kind = row[1]
    children: List[Union[Message, Summary]] = []

    if kind == "leaf":
        msg_rows = db.conn.execute(
            "SELECT m.id, m.conversation_id, m.seq, m.role, m.content, "
            "m.token_count, m.tool_call_id, m.tool_name, m.created_at "
            "FROM messages m "
            "JOIN summary_messages sm ON m.id = sm.message_id "
            "WHERE sm.summary_id = ? "
            "ORDER BY m.seq ASC",
            (summary_id,),
        ).fetchall()
        for r in msg_rows:
            children.append(
                Message(
                    id=r[0], conversation_id=r[1], seq=r[2], role=r[3],
                    content=r[4], token_count=r[5], tool_call_id=r[6],
                    tool_name=r[7], created_at=r[8],
                )
            )
    elif kind == "condensed":
        child_rows = db.conn.execute(
            "SELECT s.summary_id, s.conversation_id, s.kind, s.depth, s.content, "
            "s.token_count, s.source_token_count, s.earliest_at, s.latest_at, "
            "s.model, s.created_at "
            "FROM summaries s "
            "JOIN summary_parents sp ON s.summary_id = sp.child_id "
            "WHERE sp.parent_id = ? "
            "ORDER BY s.created_at ASC",
            (summary_id,),
        ).fetchall()
        for r in child_rows:
            children.append(
                Summary(
                    summary_id=r[0], conversation_id=r[1], kind=r[2],
                    depth=r[3], content=r[4], token_count=r[5],
                    source_token_count=r[6], earliest_at=r[7],
                    latest_at=r[8], model=r[9], created_at=r[10],
                )
            )

    return ExpandResult(summary_id=summary_id, kind=kind, children=children)
