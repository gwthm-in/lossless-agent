"""Recall tools for searching and navigating the memory DAG."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from lossless_agent.store.database import Database
from lossless_agent.store.models import Message, Summary


@dataclass
class GrepResult:
    type: str  # 'message' or 'summary'
    id: Union[str, int]
    content_snippet: str
    conversation_id: int
    metadata: Dict[str, Any]


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


def lcm_grep(
    db: Database,
    query: str,
    scope: str = "all",
    conversation_id: Optional[int] = None,
    limit: int = 20,
) -> List[GrepResult]:
    """Search messages and/or summaries via FTS5."""
    results: List[GrepResult] = []

    if scope in ("all", "messages"):
        sql = (
            "SELECT m.id, m.conversation_id, m.content, m.role, m.seq "
            "FROM messages m "
            "JOIN messages_fts f ON m.id = f.rowid "
            "WHERE messages_fts MATCH ?"
        )
        params: list = [query]
        if conversation_id is not None:
            sql += " AND m.conversation_id = ?"
            params.append(conversation_id)
        sql += " ORDER BY f.rank LIMIT ?"
        params.append(limit)

        for row in db.conn.execute(sql, params).fetchall():
            results.append(
                GrepResult(
                    type="message",
                    id=row[0],
                    content_snippet=_truncate(row[2]),
                    conversation_id=row[1],
                    metadata={"role": row[3], "seq": row[4]},
                )
            )

    if scope in ("all", "summaries"):
        sql = (
            "SELECT s.summary_id, s.conversation_id, s.content, s.kind, s.depth "
            "FROM summaries s "
            "JOIN summaries_fts f ON s.rowid = f.rowid "
            "WHERE summaries_fts MATCH ?"
        )
        params = [query]
        if conversation_id is not None:
            sql += " AND s.conversation_id = ?"
            params.append(conversation_id)
        sql += " ORDER BY f.rank LIMIT ?"
        params.append(limit)

        for row in db.conn.execute(sql, params).fetchall():
            results.append(
                GrepResult(
                    type="summary",
                    id=row[0],
                    content_snippet=_truncate(row[2]),
                    conversation_id=row[1],
                    metadata={"kind": row[3], "depth": row[4]},
                )
            )

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
