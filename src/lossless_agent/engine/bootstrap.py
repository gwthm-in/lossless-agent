"""Session bootstrap engine: seed a new conversation from a parent session."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from lossless_agent.store.database import Database
from lossless_agent.store.models import Message, Summary


@dataclass
class BootstrapResult:
    """Result of a bootstrap operation."""
    summaries_imported: int = 0
    messages_imported: int = 0
    tokens_used: int = 0


class SessionBootstrap:
    """Bootstrap a new conversation from a parent session's content."""

    def __init__(
        self,
        db: Database,
        summarize_fn: Callable[[str], str],
        bootstrap_max_tokens: int = 6000,
    ) -> None:
        self.db = db
        self.summarize_fn = summarize_fn
        self.bootstrap_max_tokens = bootstrap_max_tokens

    async def bootstrap(
        self, new_conversation_id: int, parent_session_key: str
    ) -> BootstrapResult:
        """Bootstrap new_conversation_id from the parent session.

        1. Look up parent conversation by session_key
        2. Get parent's summaries (highest depth first) and recent messages
        3. Fill bootstrap budget with summaries, then messages
        4. Create initial summaries in new conversation
        5. Update conversation.bootstrapped_at timestamp
        6. Track progress in conversation_bootstrap_state table
        """
        conn = self.db.conn

        # 1. Find parent conversation
        row = conn.execute(
            "SELECT id FROM conversations WHERE session_key = ? AND active = 1",
            (parent_session_key,),
        ).fetchone()
        if row is None:
            return BootstrapResult()

        parent_id = row[0]

        # 2. Get parent summaries (highest depth first) and messages
        summaries = conn.execute(
            "SELECT summary_id, content, token_count, depth, earliest_at, latest_at, model "
            "FROM summaries WHERE conversation_id = ? ORDER BY depth DESC, created_at DESC",
            (parent_id,),
        ).fetchall()

        messages = conn.execute(
            "SELECT id, content, token_count, role, created_at "
            "FROM messages WHERE conversation_id = ? ORDER BY seq DESC",
            (parent_id,),
        ).fetchall()

        # 3. Fill budget with summaries first, then messages
        budget = self.bootstrap_max_tokens
        summaries_imported = 0
        messages_imported = 0
        tokens_used = 0
        imported_summary_ids = []

        for s in summaries:
            s_id, s_content, s_tokens, s_depth, s_earliest, s_latest, s_model = s
            if s_tokens <= 0:
                s_tokens = len(s_content.split())
            if tokens_used + s_tokens > budget:
                continue
            # Create a summary in the new conversation linking to parent content
            import secrets
            new_sid = "sum_" + secrets.token_hex(6)
            conn.execute(
                "INSERT INTO summaries "
                "(summary_id, conversation_id, kind, depth, content, token_count, "
                "source_token_count, earliest_at, latest_at, model) "
                "VALUES (?, ?, 'leaf', 0, ?, ?, 0, ?, ?, ?)",
                (new_sid, new_conversation_id, s_content, s_tokens,
                 s_earliest, s_latest, s_model),
            )
            imported_summary_ids.append(new_sid)
            tokens_used += s_tokens
            summaries_imported += 1

        # Fill remaining budget with messages
        for m in messages:
            m_id, m_content, m_tokens, m_role, m_created = m
            if m_tokens <= 0:
                m_tokens = len(m_content.split())
            if tokens_used + m_tokens > budget:
                continue
            # Store message content as a leaf summary in new conversation
            import secrets
            new_sid = "sum_" + secrets.token_hex(6)
            conn.execute(
                "INSERT INTO summaries "
                "(summary_id, conversation_id, kind, depth, content, token_count, "
                "source_token_count, earliest_at, latest_at, model) "
                "VALUES (?, ?, 'leaf', 0, ?, ?, 0, ?, ?, 'bootstrap')",
                (new_sid, new_conversation_id,
                 f"[{m_role}] {m_content}", m_tokens,
                 m_created, m_created),
            )
            tokens_used += m_tokens
            messages_imported += 1

        if summaries_imported > 0 or messages_imported > 0:
            # 5. Update bootstrapped_at
            conn.execute(
                "UPDATE conversations SET bootstrapped_at = strftime('%Y-%m-%dT%H:%M:%f', 'now'), "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now') WHERE id = ?",
                (new_conversation_id,),
            )

            # 6. Track in bootstrap state table
            conn.execute(
                "INSERT OR REPLACE INTO conversation_bootstrap_state "
                "(conversation_id, session_file_path, last_seen_size, last_seen_mtime_ms, "
                "last_processed_offset) VALUES (?, ?, ?, ?, ?)",
                (new_conversation_id, parent_session_key,
                 summaries_imported + messages_imported, 0,
                 tokens_used),
            )

            conn.commit()

        return BootstrapResult(
            summaries_imported=summaries_imported,
            messages_imported=messages_imported,
            tokens_used=tokens_used,
        )
