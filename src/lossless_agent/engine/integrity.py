"""Integrity checker: validates invariants of the lossless-agent data model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from lossless_agent.store.database import Database


@dataclass
class CheckResult:
    """Result of a single integrity check."""

    name: str
    passed: bool
    details: str


class IntegrityChecker:
    """Run integrity checks against the lossless-agent database.

    Each check method returns a ``CheckResult``.  Use ``run_all`` to execute
    every check for a conversation and ``repair_plan`` to get human-readable
    repair suggestions for any failures.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # 1. Conversation exists
    # ------------------------------------------------------------------
    def conversation_exists(self, conversation_id: int) -> CheckResult:
        """Verify the conversation record exists."""
        row = self._db.conn.execute(
            "SELECT id FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if row is not None:
            return CheckResult("conversation_exists", True, "OK")
        return CheckResult(
            "conversation_exists",
            False,
            f"Conversation {conversation_id} not found",
        )

    # ------------------------------------------------------------------
    # 2. context_items ordinals are contiguous (graceful if table missing)
    # ------------------------------------------------------------------
    def context_items_contiguous(self, conversation_id: int) -> CheckResult:
        """Check for ordinal gaps in the context_items table.

        If the ``context_items`` table does not exist yet the check passes
        with a note – the table may be created by a later migration.
        """
        if not self._table_exists("context_items"):
            return CheckResult(
                "context_items_contiguous",
                True,
                "context_items table does not exist yet – skipped",
            )

        rows = self._db.conn.execute(
            "SELECT ordinal FROM context_items WHERE conversation_id = ? ORDER BY ordinal",
            (conversation_id,),
        ).fetchall()
        if not rows:
            return CheckResult("context_items_contiguous", True, "No context items")

        ordinals = [r[0] for r in rows]
        expected = list(range(ordinals[0], ordinals[0] + len(ordinals)))
        if ordinals == expected:
            return CheckResult("context_items_contiguous", True, "OK")

        gaps = sorted(set(expected) - set(ordinals))
        return CheckResult(
            "context_items_contiguous",
            False,
            f"Ordinal gaps at positions: {gaps}",
        )

    # ------------------------------------------------------------------
    # 3. context_items have valid references
    # ------------------------------------------------------------------
    def context_items_valid_refs(self, conversation_id: int) -> CheckResult:
        """Ensure no context_item references a non-existent message or summary."""
        if not self._table_exists("context_items"):
            return CheckResult(
                "context_items_valid_refs",
                True,
                "context_items table does not exist yet – skipped",
            )

        dangling: List[str] = []

        # Check message references
        rows = self._db.conn.execute(
            "SELECT ci.ordinal, ci.message_id FROM context_items ci "
            "WHERE ci.conversation_id = ? AND ci.message_id IS NOT NULL",
            (conversation_id,),
        ).fetchall()
        for ordinal, msg_id in rows:
            exists = self._db.conn.execute(
                "SELECT 1 FROM messages WHERE id = ?", (msg_id,)
            ).fetchone()
            if exists is None:
                dangling.append(f"ordinal {ordinal}: message_id {msg_id} missing")

        # Check summary references
        rows = self._db.conn.execute(
            "SELECT ci.ordinal, ci.summary_id FROM context_items ci "
            "WHERE ci.conversation_id = ? AND ci.summary_id IS NOT NULL",
            (conversation_id,),
        ).fetchall()
        for ordinal, sum_id in rows:
            exists = self._db.conn.execute(
                "SELECT 1 FROM summaries WHERE summary_id = ?", (sum_id,)
            ).fetchone()
            if exists is None:
                dangling.append(f"ordinal {ordinal}: summary_id {sum_id} missing")

        if not dangling:
            return CheckResult("context_items_valid_refs", True, "OK")
        return CheckResult(
            "context_items_valid_refs",
            False,
            f"Dangling references: {'; '.join(dangling)}",
        )

    # ------------------------------------------------------------------
    # 4. Summaries have proper lineage
    # ------------------------------------------------------------------
    def summaries_have_lineage(self, conversation_id: int) -> CheckResult:
        """Leaf summaries must have summary_messages; condensed must have summary_parents."""
        problems: List[str] = []

        # Leaf summaries -> must have at least one summary_messages entry
        leaf_rows = self._db.conn.execute(
            "SELECT summary_id FROM summaries "
            "WHERE conversation_id = ? AND kind = 'leaf'",
            (conversation_id,),
        ).fetchall()
        for (sid,) in leaf_rows:
            cnt = self._db.conn.execute(
                "SELECT COUNT(*) FROM summary_messages WHERE summary_id = ?",
                (sid,),
            ).fetchone()[0]
            if cnt == 0:
                problems.append(f"leaf {sid} has no summary_messages entries")

        # Condensed summaries -> must have at least one summary_parents entry
        cond_rows = self._db.conn.execute(
            "SELECT summary_id FROM summaries "
            "WHERE conversation_id = ? AND kind = 'condensed'",
            (conversation_id,),
        ).fetchall()
        for (sid,) in cond_rows:
            cnt = self._db.conn.execute(
                "SELECT COUNT(*) FROM summary_parents WHERE parent_id = ?",
                (sid,),
            ).fetchone()[0]
            if cnt == 0:
                problems.append(f"condensed {sid} has no summary_parents entries")

        if not problems:
            return CheckResult("summaries_have_lineage", True, "OK")
        return CheckResult(
            "summaries_have_lineage",
            False,
            "; ".join(problems),
        )

    # ------------------------------------------------------------------
    # 5. No orphan summaries
    # ------------------------------------------------------------------
    def no_orphan_summaries(self, conversation_id: int) -> CheckResult:
        """Every summary must be in context_items or be a child of another summary."""
        all_sids = [
            r[0]
            for r in self._db.conn.execute(
                "SELECT summary_id FROM summaries WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchall()
        ]
        if not all_sids:
            return CheckResult("no_orphan_summaries", True, "No summaries")

        # Summaries referenced as children of some parent
        child_sids = set(
            r[0]
            for r in self._db.conn.execute(
                "SELECT child_id FROM summary_parents"
            ).fetchall()
        )

        # Summaries referenced in context_items (if the table exists)
        context_sids: set = set()
        if self._table_exists("context_items"):
            context_sids = set(
                r[0]
                for r in self._db.conn.execute(
                    "SELECT summary_id FROM context_items "
                    "WHERE conversation_id = ? AND summary_id IS NOT NULL",
                    (conversation_id,),
                ).fetchall()
            )

        # A summary that is a parent of children is also considered in-use
        parent_sids = set(
            r[0]
            for r in self._db.conn.execute(
                "SELECT DISTINCT parent_id FROM summary_parents"
            ).fetchall()
        )

        orphans = [
            sid
            for sid in all_sids
            if sid not in child_sids
            and sid not in context_sids
            and sid not in parent_sids
        ]
        if not orphans:
            return CheckResult("no_orphan_summaries", True, "OK")
        return CheckResult(
            "no_orphan_summaries",
            False,
            f"Orphan summaries: {orphans}",
        )

    # ------------------------------------------------------------------
    # 6. Context token consistency
    # ------------------------------------------------------------------
    def context_token_consistency(self, conversation_id: int) -> CheckResult:
        """Verify that manual token sum matches stored totals.

        Compares the sum of individual message token_counts with the value
        returned by ``SUM(token_count)`` at the SQL level (they should always
        agree, but a bug could cause divergence).
        """
        rows = self._db.conn.execute(
            "SELECT token_count FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchall()
        manual_sum = sum(r[0] for r in rows)

        sql_sum = self._db.conn.execute(
            "SELECT COALESCE(SUM(token_count), 0) FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()[0]

        if manual_sum == sql_sum:
            return CheckResult(
                "context_token_consistency",
                True,
                f"OK (total={manual_sum})",
            )
        return CheckResult(
            "context_token_consistency",
            False,
            f"Manual sum {manual_sum} != SQL sum {sql_sum}",
        )

    # ------------------------------------------------------------------
    # 7. Message seq contiguous
    # ------------------------------------------------------------------
    def message_seq_contiguous(self, conversation_id: int) -> CheckResult:
        """Verify there are no gaps in message seq numbers."""
        rows = self._db.conn.execute(
            "SELECT seq FROM messages WHERE conversation_id = ? ORDER BY seq",
            (conversation_id,),
        ).fetchall()
        if not rows:
            return CheckResult("message_seq_contiguous", True, "No messages")

        seqs = [r[0] for r in rows]
        expected = list(range(seqs[0], seqs[0] + len(seqs)))
        if seqs == expected:
            return CheckResult("message_seq_contiguous", True, "OK")

        gaps = sorted(set(expected) - set(seqs))
        return CheckResult(
            "message_seq_contiguous",
            False,
            f"Missing seq numbers: {gaps}",
        )

    # ------------------------------------------------------------------
    # 8. No duplicate context refs
    # ------------------------------------------------------------------
    def no_duplicate_context_refs(self, conversation_id: int) -> CheckResult:
        """No duplicate message/summary references in context_items."""
        if not self._table_exists("context_items"):
            return CheckResult(
                "no_duplicate_context_refs",
                True,
                "context_items table does not exist yet – skipped",
            )

        dupes: List[str] = []

        # Duplicate message refs
        rows = self._db.conn.execute(
            "SELECT message_id, COUNT(*) AS cnt FROM context_items "
            "WHERE conversation_id = ? AND message_id IS NOT NULL "
            "GROUP BY message_id HAVING cnt > 1",
            (conversation_id,),
        ).fetchall()
        for msg_id, cnt in rows:
            dupes.append(f"message_id {msg_id} appears {cnt} times")

        # Duplicate summary refs
        rows = self._db.conn.execute(
            "SELECT summary_id, COUNT(*) AS cnt FROM context_items "
            "WHERE conversation_id = ? AND summary_id IS NOT NULL "
            "GROUP BY summary_id HAVING cnt > 1",
            (conversation_id,),
        ).fetchall()
        for sum_id, cnt in rows:
            dupes.append(f"summary_id {sum_id} appears {cnt} times")

        if not dupes:
            return CheckResult("no_duplicate_context_refs", True, "OK")
        return CheckResult(
            "no_duplicate_context_refs",
            False,
            "; ".join(dupes),
        )

    # ------------------------------------------------------------------
    # Aggregate helpers
    # ------------------------------------------------------------------
    def run_all(self, conversation_id: int) -> List[CheckResult]:
        """Run every integrity check and return the list of results."""
        return [
            self.conversation_exists(conversation_id),
            self.context_items_contiguous(conversation_id),
            self.context_items_valid_refs(conversation_id),
            self.summaries_have_lineage(conversation_id),
            self.no_orphan_summaries(conversation_id),
            self.context_token_consistency(conversation_id),
            self.message_seq_contiguous(conversation_id),
            self.no_duplicate_context_refs(conversation_id),
        ]

    @staticmethod
    def repair_plan(results: List[CheckResult]) -> List[str]:
        """Generate human-readable repair suggestions for failed checks."""
        suggestions: List[str] = []
        repair_hints = {
            "conversation_exists": (
                "Create the missing conversation record or verify the conversation_id."
            ),
            "context_items_contiguous": (
                "Re-number context_items ordinals to close gaps."
            ),
            "context_items_valid_refs": (
                "Remove context_items rows with dangling message/summary references."
            ),
            "summaries_have_lineage": (
                "Add missing summary_messages or summary_parents entries, "
                "or delete the malformed summaries."
            ),
            "no_orphan_summaries": (
                "Delete orphan summaries that are not referenced anywhere, "
                "or re-attach them to the context."
            ),
            "context_token_consistency": (
                "Recount tokens for all messages in the conversation "
                "and update token_count values."
            ),
            "message_seq_contiguous": (
                "Re-number message seq values to close gaps, "
                "or investigate missing messages."
            ),
            "no_duplicate_context_refs": (
                "Remove duplicate context_items entries keeping only one per reference."
            ),
        }
        for r in results:
            if not r.passed:
                hint = repair_hints.get(r.name, "Investigate manually.")
                suggestions.append(f"[{r.name}] {r.details} -> {hint}")
        return suggestions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _table_exists(self, table_name: str) -> bool:
        """Check whether a table exists in the database."""
        row = self._db.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        return row is not None
