"""Tests for engine.integrity – IntegrityChecker."""
from __future__ import annotations

import pytest

from lossless_agent.store.database import Database
from lossless_agent.engine.integrity import IntegrityChecker


@pytest.fixture
def db():
    database = Database(":memory:")
    yield database
    database.close()


@pytest.fixture
def checker(db):
    return IntegrityChecker(db)


def _insert_conversation(db, conv_id=1):
    db.conn.execute(
        "INSERT INTO conversations (id, session_key) VALUES (?, ?)",
        (conv_id, f"sess_{conv_id}"),
    )
    db.conn.commit()
    return conv_id


def _insert_message(db, conv_id, seq, role="user", content="hi", token_count=5,
                     tool_call_id=None, tool_name=None):
    cur = db.conn.execute(
        "INSERT INTO messages (conversation_id, seq, role, content, token_count, "
        "tool_call_id, tool_name) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (conv_id, seq, role, content, token_count, tool_call_id, tool_name),
    )
    db.conn.commit()
    return cur.lastrowid


def _insert_leaf_summary(db, summary_id, conv_id, message_ids, token_count=10):
    db.conn.execute(
        "INSERT INTO summaries (summary_id, conversation_id, kind, depth, content, "
        "token_count, source_token_count, earliest_at, latest_at, model) "
        "VALUES (?, ?, 'leaf', 0, 'summary text', ?, 50, '2025-01-01', '2025-01-02', 'gpt')",
        (summary_id, conv_id, token_count),
    )
    for mid in message_ids:
        db.conn.execute(
            "INSERT INTO summary_messages (summary_id, message_id) VALUES (?, ?)",
            (summary_id, mid),
        )
    db.conn.commit()


def _insert_condensed_summary(db, summary_id, conv_id, child_ids, depth=1, token_count=10):
    db.conn.execute(
        "INSERT INTO summaries (summary_id, conversation_id, kind, depth, content, "
        "token_count, source_token_count, earliest_at, latest_at, model) "
        "VALUES (?, ?, 'condensed', ?, 'condensed text', ?, 50, '2025-01-01', '2025-01-02', 'gpt')",
        (summary_id, conv_id, depth, token_count),
    )
    for cid in child_ids:
        db.conn.execute(
            "INSERT INTO summary_parents (parent_id, child_id) VALUES (?, ?)",
            (summary_id, cid),
        )
    db.conn.commit()


# ── 1. conversation_exists ──────────────────────────────────────────

class TestConversationExists:
    def test_exists(self, db, checker):
        _insert_conversation(db)
        r = checker.conversation_exists(1)
        assert r.passed is True

    def test_not_exists(self, checker):
        r = checker.conversation_exists(999)
        assert r.passed is False
        assert "999" in r.details


# ── 2. context_items_contiguous ─────────────────────────────────────

class TestContextItemsContiguous:
    def test_no_items(self, db, checker):
        _insert_conversation(db)
        r = checker.context_items_contiguous(1)
        assert r.passed is True

    def test_contiguous(self, db, checker):
        _insert_conversation(db)
        for i in range(1, 4):
            mid = _insert_message(db, 1, i)
            db.conn.execute(
                "INSERT INTO context_items (conversation_id, ordinal, item_type, message_id) "
                "VALUES (?, ?, 'message', ?)", (1, i, mid),
            )
        db.conn.commit()
        r = checker.context_items_contiguous(1)
        assert r.passed is True

    def test_gap(self, db, checker):
        _insert_conversation(db)
        for i in [1, 2, 5]:
            mid = _insert_message(db, 1, i)
            db.conn.execute(
                "INSERT INTO context_items (conversation_id, ordinal, item_type, message_id) "
                "VALUES (?, ?, 'message', ?)", (1, i, mid),
            )
        db.conn.commit()
        r = checker.context_items_contiguous(1)
        assert r.passed is False
        assert "gaps" in r.details.lower() or "3" in r.details


# ── 3. context_items_valid_refs ─────────────────────────────────────

class TestContextItemsValidRefs:
    def test_no_items(self, db, checker):
        _insert_conversation(db)
        r = checker.context_items_valid_refs(1)
        assert r.passed is True

    def test_valid(self, db, checker):
        _insert_conversation(db)
        mid = _insert_message(db, 1, 1)
        db.conn.execute(
            "INSERT INTO context_items (conversation_id, ordinal, item_type, message_id) "
            "VALUES (1, 1, 'message', ?)", (mid,),
        )
        db.conn.commit()
        r = checker.context_items_valid_refs(1)
        assert r.passed is True

    def test_dangling_message(self, db, checker):
        _insert_conversation(db)
        db.conn.execute(
            "INSERT INTO context_items (conversation_id, ordinal, item_type, message_id) "
            "VALUES (1, 1, 'message', 9999)",
        )
        db.conn.commit()
        r = checker.context_items_valid_refs(1)
        assert r.passed is False
        assert "9999" in r.details

    def test_dangling_summary(self, db, checker):
        _insert_conversation(db)
        db.conn.execute(
            "INSERT INTO context_items (conversation_id, ordinal, item_type, summary_id) "
            "VALUES (1, 1, 'summary', 'sum_nonexistent')",
        )
        db.conn.commit()
        r = checker.context_items_valid_refs(1)
        assert r.passed is False
        assert "sum_nonexistent" in r.details


# ── 4. summaries_have_lineage ───────────────────────────────────────

class TestSummariesHaveLineage:
    def test_leaf_with_messages(self, db, checker):
        _insert_conversation(db)
        mid = _insert_message(db, 1, 1)
        _insert_leaf_summary(db, "sum_a", 1, [mid])
        r = checker.summaries_have_lineage(1)
        assert r.passed is True

    def test_leaf_without_messages(self, db, checker):
        _insert_conversation(db)
        db.conn.execute(
            "INSERT INTO summaries (summary_id, conversation_id, kind, depth, content, "
            "token_count, source_token_count, earliest_at, latest_at, model) "
            "VALUES ('sum_orphan', 1, 'leaf', 0, 'x', 10, 50, '2025-01-01', '2025-01-02', 'gpt')",
        )
        db.conn.commit()
        r = checker.summaries_have_lineage(1)
        assert r.passed is False
        assert "sum_orphan" in r.details

    def test_condensed_with_parents(self, db, checker):
        _insert_conversation(db)
        mid = _insert_message(db, 1, 1)
        _insert_leaf_summary(db, "sum_a", 1, [mid])
        _insert_condensed_summary(db, "sum_c", 1, ["sum_a"])
        r = checker.summaries_have_lineage(1)
        assert r.passed is True

    def test_condensed_without_parents(self, db, checker):
        _insert_conversation(db)
        db.conn.execute(
            "INSERT INTO summaries (summary_id, conversation_id, kind, depth, content, "
            "token_count, source_token_count, earliest_at, latest_at, model) "
            "VALUES ('sum_bad', 1, 'condensed', 1, 'x', 10, 50, '2025-01-01', '2025-01-02', 'gpt')",
        )
        db.conn.commit()
        r = checker.summaries_have_lineage(1)
        assert r.passed is False
        assert "sum_bad" in r.details


# ── 5. no_orphan_summaries ──────────────────────────────────────────

class TestNoOrphanSummaries:
    def test_no_summaries(self, db, checker):
        _insert_conversation(db)
        r = checker.no_orphan_summaries(1)
        assert r.passed is True

    def test_summary_is_parent(self, db, checker):
        _insert_conversation(db)
        mid = _insert_message(db, 1, 1)
        _insert_leaf_summary(db, "sum_a", 1, [mid])
        _insert_condensed_summary(db, "sum_c", 1, ["sum_a"])
        r = checker.no_orphan_summaries(1)
        # sum_c is a parent, sum_a is a child -> neither orphan
        assert r.passed is True

    def test_orphan_detected(self, db, checker):
        _insert_conversation(db)
        db.conn.execute(
            "INSERT INTO summaries (summary_id, conversation_id, kind, depth, content, "
            "token_count, source_token_count, earliest_at, latest_at, model) "
            "VALUES ('sum_lonely', 1, 'leaf', 0, 'x', 10, 50, '2025-01-01', '2025-01-02', 'gpt')",
        )
        db.conn.commit()
        r = checker.no_orphan_summaries(1)
        assert r.passed is False
        assert "sum_lonely" in r.details


# ── 6. context_token_consistency ────────────────────────────────────

class TestContextTokenConsistency:
    def test_consistent(self, db, checker):
        _insert_conversation(db)
        _insert_message(db, 1, 1, token_count=10)
        _insert_message(db, 1, 2, token_count=20)
        r = checker.context_token_consistency(1)
        assert r.passed is True
        assert "30" in r.details

    def test_no_messages(self, db, checker):
        _insert_conversation(db)
        r = checker.context_token_consistency(1)
        assert r.passed is True


# ── 7. message_seq_contiguous ───────────────────────────────────────

class TestMessageSeqContiguous:
    def test_contiguous(self, db, checker):
        _insert_conversation(db)
        for i in range(1, 5):
            _insert_message(db, 1, i)
        r = checker.message_seq_contiguous(1)
        assert r.passed is True

    def test_gap(self, db, checker):
        _insert_conversation(db)
        for i in [1, 2, 5]:
            _insert_message(db, 1, i)
        r = checker.message_seq_contiguous(1)
        assert r.passed is False
        assert "3" in r.details

    def test_no_messages(self, db, checker):
        _insert_conversation(db)
        r = checker.message_seq_contiguous(1)
        assert r.passed is True


# ── 8. no_duplicate_context_refs ────────────────────────────────────

class TestNoDuplicateContextRefs:
    def test_no_items(self, db, checker):
        _insert_conversation(db)
        r = checker.no_duplicate_context_refs(1)
        assert r.passed is True

    def test_no_dupes(self, db, checker):
        _insert_conversation(db)
        mid = _insert_message(db, 1, 1)
        db.conn.execute(
            "INSERT INTO context_items (conversation_id, ordinal, item_type, message_id) "
            "VALUES (1, 1, 'message', ?)", (mid,),
        )
        db.conn.commit()
        r = checker.no_duplicate_context_refs(1)
        assert r.passed is True

    def test_duplicate_message(self, db, checker):
        _insert_conversation(db)
        mid = _insert_message(db, 1, 1)
        db.conn.execute(
            "INSERT INTO context_items (conversation_id, ordinal, item_type, message_id) "
            "VALUES (1, 1, 'message', ?)", (mid,),
        )
        db.conn.execute(
            "INSERT INTO context_items (conversation_id, ordinal, item_type, message_id) "
            "VALUES (1, 2, 'message', ?)", (mid,),
        )
        db.conn.commit()
        r = checker.no_duplicate_context_refs(1)
        assert r.passed is False
        assert "2 times" in r.details


# ── run_all & repair_plan ───────────────────────────────────────────

class TestRunAll:
    def test_all_pass(self, db, checker):
        _insert_conversation(db)
        _insert_message(db, 1, 1)
        results = checker.run_all(1)
        assert len(results) == 8
        assert all(r.passed for r in results)

    def test_repair_plan_empty_on_pass(self, db, checker):
        _insert_conversation(db)
        results = checker.run_all(1)
        plan = IntegrityChecker.repair_plan(results)
        assert plan == []

    def test_repair_plan_on_failure(self, checker):
        results = checker.run_all(999)
        plan = IntegrityChecker.repair_plan(results)
        assert len(plan) >= 1
        assert "conversation_exists" in plan[0]



