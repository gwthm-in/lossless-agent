#!/usr/bin/env python3
"""Import data from an OpenClaw lossless-claw SQLite database into lossless-agent format.

Usage:
    python scripts/ingest_openclaw.py --source /path/to/openclaw.db --target /path/to/lossless-agent.db
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import os

# Add project src to path so we can import lossless_agent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lossless_agent.store.database import Database


def connect_source(path: str) -> sqlite3.Connection:
    """Open the OpenClaw source database read-only."""
    if not os.path.exists(path):
        print(f"ERROR: Source database not found: {path}")
        sys.exit(1)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_source_tables(src: sqlite3.Connection) -> set[str]:
    """Get available table names in the source database."""
    rows = src.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {r[0] for r in rows}


def ingest_conversations(src: sqlite3.Connection, tgt: Database) -> dict[int, int]:
    """Import conversations. Returns mapping from old_id -> new_id."""
    rows = src.execute(
        "SELECT conversation_id, session_key, title, created_at, updated_at "
        "FROM conversations ORDER BY conversation_id"
    ).fetchall()

    id_map: dict[int, int] = {}
    conn = tgt.conn
    for r in rows:
        old_id = r["conversation_id"]
        session_key = r["session_key"]
        title = r["title"] or ""
        created_at = r["created_at"]
        updated_at = r["updated_at"]

        cur = conn.execute(
            "INSERT INTO conversations (session_key, title, active, created_at, updated_at) "
            "VALUES (?, ?, 1, ?, ?)",
            (session_key, title, created_at, updated_at),
        )
        id_map[old_id] = cur.lastrowid
    conn.commit()
    print(f"  Conversations: {len(id_map)} imported")
    return id_map


def ingest_messages(
    src: sqlite3.Connection,
    tgt: Database,
    conv_map: dict[int, int],
    source_tables: set[str],
) -> dict[int, int]:
    """Import messages. Returns mapping from old_message_id -> new_message_id."""
    rows = src.execute(
        "SELECT message_id, conversation_id, seq, role, content, token_count, created_at "
        "FROM messages ORDER BY conversation_id, seq"
    ).fetchall()

    # Try to get tool_call_id / tool_name from message_parts if available
    tool_info: dict[int, tuple[str | None, str | None]] = {}
    if "message_parts" in source_tables:
        try:
            parts = src.execute(
                "SELECT message_id, tool_call_id, tool_name FROM message_parts "
                "WHERE tool_call_id IS NOT NULL OR tool_name IS NOT NULL"
            ).fetchall()
            for p in parts:
                tool_info[p["message_id"]] = (p["tool_call_id"], p["tool_name"])
        except Exception:
            pass  # Table structure might differ, that's OK

    id_map: dict[int, int] = {}
    conn = tgt.conn

    # Disable FTS triggers temporarily for bulk insert (we rebuild later)
    batch = []
    for r in rows:
        old_id = r["message_id"]
        new_conv_id = conv_map.get(r["conversation_id"])
        if new_conv_id is None:
            print(f"  WARNING: Skipping message {old_id} — unknown conversation {r['conversation_id']}")
            continue

        tool_call_id, tool_name = tool_info.get(old_id, (None, None))

        batch.append((
            new_conv_id,
            r["seq"],
            r["role"],
            r["content"] or "",
            r["token_count"] or 0,
            tool_call_id,
            tool_name,
            r["created_at"],
            old_id,
        ))

    for item in batch:
        new_conv_id, seq, role, content, token_count, tool_call_id, tool_name, created_at, old_id = item
        cur = conn.execute(
            "INSERT INTO messages (conversation_id, seq, role, content, token_count, "
            "tool_call_id, tool_name, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (new_conv_id, seq, role, content, token_count, tool_call_id, tool_name, created_at),
        )
        id_map[old_id] = cur.lastrowid

    conn.commit()
    print(f"  Messages: {len(id_map)} imported")
    return id_map


def ingest_summaries(
    src: sqlite3.Connection,
    tgt: Database,
    conv_map: dict[int, int],
) -> None:
    """Import summaries, summary_messages, and summary_parents."""
    conn = tgt.conn

    # -- summaries --
    rows = src.execute(
        "SELECT summary_id, conversation_id, kind, depth, content, token_count, "
        "earliest_at, latest_at, source_message_token_count, model, created_at "
        "FROM summaries ORDER BY depth, created_at"
    ).fetchall()

    imported = 0
    skipped = 0
    for r in rows:
        new_conv_id = conv_map.get(r["conversation_id"])
        if new_conv_id is None:
            skipped += 1
            continue

        # OpenClaw uses source_message_token_count; lossless-agent uses source_token_count
        source_token_count = r["source_message_token_count"] or 0

        conn.execute(
            "INSERT INTO summaries (summary_id, conversation_id, kind, depth, content, "
            "token_count, source_token_count, earliest_at, latest_at, model, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                r["summary_id"],
                new_conv_id,
                r["kind"],
                r["depth"] or 0,
                r["content"] or "",
                r["token_count"] or 0,
                source_token_count,
                r["earliest_at"],
                r["latest_at"],
                r["model"] or "",
                r["created_at"],
            ),
        )
        imported += 1

    conn.commit()
    print(f"  Summaries: {imported} imported, {skipped} skipped")


def ingest_summary_messages(
    src: sqlite3.Connection,
    tgt: Database,
    msg_map: dict[int, int],
) -> None:
    """Import summary_messages links (leaf summary -> source messages)."""
    conn = tgt.conn

    rows = src.execute(
        "SELECT summary_id, message_id FROM summary_messages"
    ).fetchall()

    imported = 0
    skipped = 0
    for r in rows:
        new_msg_id = msg_map.get(r["message_id"])
        if new_msg_id is None:
            skipped += 1
            continue

        try:
            conn.execute(
                "INSERT OR IGNORE INTO summary_messages (summary_id, message_id) VALUES (?, ?)",
                (r["summary_id"], new_msg_id),
            )
            imported += 1
        except sqlite3.IntegrityError:
            skipped += 1

    conn.commit()
    print(f"  Summary-messages links: {imported} imported, {skipped} skipped")


def ingest_summary_parents(src: sqlite3.Connection, tgt: Database) -> None:
    """Import summary_parents (DAG edges).

    OpenClaw schema: summary_id (the condensed/parent), parent_summary_id (the leaf/child source)
    lossless-agent schema: parent_id (condensed), child_id (leaf/child source)

    So: openclaw.summary_id -> lossless-agent.parent_id
        openclaw.parent_summary_id -> lossless-agent.child_id
    """
    conn = tgt.conn

    rows = src.execute(
        "SELECT summary_id, parent_summary_id FROM summary_parents"
    ).fetchall()

    imported = 0
    skipped = 0
    for r in rows:
        # In OpenClaw: summary_id is the condensed summary (parent in DAG)
        # parent_summary_id is the source/leaf summary (child in DAG)
        parent_id = r["summary_id"]          # condensed = parent
        child_id = r["parent_summary_id"]    # source leaf = child

        try:
            conn.execute(
                "INSERT OR IGNORE INTO summary_parents (parent_id, child_id) VALUES (?, ?)",
                (parent_id, child_id),
            )
            imported += 1
        except sqlite3.IntegrityError:
            skipped += 1

    conn.commit()
    print(f"  Summary-parents edges: {imported} imported, {skipped} skipped")


def rebuild_fts(tgt: Database) -> None:
    """Rebuild FTS5 indexes from scratch."""
    conn = tgt.conn

    # Rebuild messages_fts
    conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")

    # Rebuild summaries_fts
    conn.execute("INSERT INTO summaries_fts(summaries_fts) VALUES('rebuild')")

    conn.commit()
    print("  FTS5 indexes rebuilt")


def print_stats(tgt: Database) -> None:
    """Print summary statistics for the target database."""
    conn = tgt.conn

    row = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()
    print(f"\n  Conversations: {row[0]}")

    row = conn.execute("SELECT COUNT(*), COALESCE(SUM(token_count), 0) FROM messages").fetchone()
    print(f"  Messages: {row[0]} ({row[1]:,} tokens)")

    row = conn.execute("SELECT COUNT(*), COALESCE(SUM(token_count), 0) FROM summaries").fetchone()
    print(f"  Summaries: {row[0]} ({row[1]:,} tokens)")

    rows = conn.execute(
        "SELECT depth, kind, COUNT(*) FROM summaries GROUP BY depth, kind ORDER BY depth, kind"
    ).fetchall()
    if rows:
        print("  Summary breakdown:")
        for r in rows:
            print(f"    depth={r[0]} kind={r[1]}: {r[2]}")

    row = conn.execute("SELECT COUNT(*) FROM summary_messages").fetchone()
    print(f"  Summary-message links: {row[0]}")

    row = conn.execute("SELECT COUNT(*) FROM summary_parents").fetchone()
    print(f"  Summary-parent edges: {row[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import OpenClaw lossless-claw data into lossless-agent format"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to OpenClaw lossless-claw SQLite database",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Path to lossless-agent SQLite database (will be created if needed)",
    )
    args = parser.parse_args()

    print(f"Source: {args.source}")
    print(f"Target: {args.target}")

    # Open source read-only
    src = connect_source(args.source)
    source_tables = get_source_tables(src)
    print(f"Source tables: {sorted(source_tables)}")

    # Verify required tables exist
    required = {"conversations", "messages", "summaries", "summary_messages", "summary_parents"}
    missing = required - source_tables
    if missing:
        print(f"ERROR: Source database missing required tables: {missing}")
        sys.exit(1)

    # Open/create target with lossless-agent schema
    tgt = Database(args.target)

    # Check target is empty
    row = tgt.conn.execute("SELECT COUNT(*) FROM conversations").fetchone()
    if row[0] > 0:
        print(f"WARNING: Target database already has {row[0]} conversations.")
        resp = input("Continue and add data? [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted.")
            sys.exit(0)

    print("\nImporting...")

    # 1. Conversations
    conv_map = ingest_conversations(src, tgt)

    # 2. Messages
    msg_map = ingest_messages(src, tgt, conv_map, source_tables)

    # 3. Summaries
    ingest_summaries(src, tgt, conv_map)

    # 4. Summary-message links
    ingest_summary_messages(src, tgt, msg_map)

    # 5. Summary-parent edges
    ingest_summary_parents(src, tgt)

    # 6. Rebuild FTS
    print("\nRebuilding FTS indexes...")
    rebuild_fts(tgt)

    # 7. Stats
    print("\n=== Target Database Stats ===")
    print_stats(tgt)

    tgt.close()
    src.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
