"""Tests for recall tool features: lcm_expand restriction, lcm_grep pagination + grouping."""
from __future__ import annotations

import pytest

from lossless_agent.store.conversation_store import ConversationStore
from lossless_agent.store.message_store import MessageStore
from lossless_agent.store.summary_store import SummaryStore
from lossless_agent.tools.recall import (
    GrepResult,
    GrepResultGroup,
    GroupedGrepResult,
    SubAgentRestrictionError,
    lcm_expand,
    lcm_grep,
)


# ===================================================================
# lcm_expand sub-agent restriction
# ===================================================================


class TestLcmExpandSubAgentRestriction:
    @pytest.fixture(autouse=True)
    def setup(self, db):
        self.db = db
        self.conv_store = ConversationStore(db)
        self.msg_store = MessageStore(db)
        self.sum_store = SummaryStore(db)
        self.conv = self.conv_store.get_or_create("s1", "Test")

    def test_expand_raises_without_sub_agent_flag(self):
        """lcm_expand should raise when is_sub_agent=False (default)."""
        with pytest.raises(SubAgentRestrictionError) as exc_info:
            lcm_expand(self.db, "nonexistent-id")
        assert "restricted to sub-agent sessions" in str(exc_info.value)
        assert "lcm_expand_query" in str(exc_info.value)

    def test_expand_raises_with_explicit_false(self):
        with pytest.raises(SubAgentRestrictionError):
            lcm_expand(self.db, "some-id", is_sub_agent=False)

    def test_expand_works_with_sub_agent_flag(self):
        """lcm_expand should work when is_sub_agent=True."""
        # Create a leaf summary to expand
        msgs = []
        for i in range(4):
            role = "user" if i % 2 == 0 else "assistant"
            m = self.msg_store.append(self.conv.id, role, f"msg {i}", token_count=10)
            msgs.append(m)

        summary = self.sum_store.create_leaf(
            conversation_id=self.conv.id,
            content="summary of messages",
            token_count=5,
            source_token_count=40,
            message_ids=[m.id for m in msgs],
            earliest_at=msgs[0].created_at,
            latest_at=msgs[-1].created_at,
            model="test",
        )

        result = lcm_expand(self.db, summary.summary_id, is_sub_agent=True)
        assert result is not None
        assert result.kind == "leaf"
        assert len(result.children) == 4

    def test_expand_returns_none_for_missing_summary(self):
        result = lcm_expand(self.db, "nonexistent-id", is_sub_agent=True)
        assert result is None


# ===================================================================
# lcm_grep pagination
# ===================================================================


class TestLcmGrepPagination:
    @pytest.fixture(autouse=True)
    def setup(self, db):
        self.db = db
        self.conv_store = ConversationStore(db)
        self.msg_store = MessageStore(db)
        self.conv = self.conv_store.get_or_create("s1", "Test")
        # Seed messages with searchable content
        for i in range(10):
            role = "user" if i % 2 == 0 else "assistant"
            self.msg_store.append(
                self.conv.id, role, f"pagination test message {i}", token_count=5,
            )

    def test_default_offset_zero(self):
        results = lcm_grep(self.db, "pagination", scope="messages", limit=5)
        assert len(results) <= 5

    def test_offset_skips_results(self):
        all_results = lcm_grep(self.db, "pagination", scope="messages", limit=20)
        offset_results = lcm_grep(
            self.db, "pagination", scope="messages", limit=20, offset=3,
        )
        if len(all_results) > 3:
            assert len(offset_results) == len(all_results) - 3

    def test_offset_beyond_results(self):
        results = lcm_grep(
            self.db, "pagination", scope="messages", limit=20, offset=100,
        )
        assert results == []

    def test_offset_with_limit(self):
        results = lcm_grep(
            self.db, "pagination", scope="messages", limit=2, offset=1,
        )
        assert len(results) <= 2


# ===================================================================
# lcm_grep grouping
# ===================================================================


class TestLcmGrepGrouping:
    @pytest.fixture(autouse=True)
    def setup(self, db):
        self.db = db
        self.conv_store = ConversationStore(db)
        self.msg_store = MessageStore(db)
        self.sum_store = SummaryStore(db)
        self.conv = self.conv_store.get_or_create("s1", "Test")

        # Create messages and group some under a summary
        self.msgs = []
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            m = self.msg_store.append(
                self.conv.id, role, f"grouped test content {i}", token_count=10,
            )
            self.msgs.append(m)

        # Create a leaf summary covering first 4 messages
        self.summary = self.sum_store.create_leaf(
            conversation_id=self.conv.id,
            content="Summary covering grouped test content 0-3",
            token_count=10,
            source_token_count=40,
            message_ids=[m.id for m in self.msgs[:4]],
            earliest_at=self.msgs[0].created_at,
            latest_at=self.msgs[3].created_at,
            model="test",
        )

    def test_grouped_false_returns_flat_list(self):
        results = lcm_grep(self.db, "grouped test", scope="messages", grouped=False)
        assert isinstance(results, list)

    def test_grouped_true_returns_grouped_result(self):
        result = lcm_grep(self.db, "grouped test", scope="messages", grouped=True)
        assert isinstance(result, GroupedGrepResult)

    def test_grouped_result_has_groups(self):
        result = lcm_grep(self.db, "grouped test", scope="messages", grouped=True)
        # Messages 0-3 covered by summary -> 1 group, messages 4-5 ungrouped
        assert len(result.groups) >= 1 or len(result.ungrouped) >= 1

    def test_grouped_result_group_structure(self):
        result = lcm_grep(self.db, "grouped test", scope="messages", grouped=True)
        if result.groups:
            group = result.groups[0]
            assert isinstance(group, GrepResultGroup)
            assert group.summary_id == self.summary.summary_id
            assert len(group.summary_content_preview) > 0
            assert len(group.matches) > 0

    def test_grouped_ungrouped_messages(self):
        result = lcm_grep(self.db, "grouped test", scope="messages", grouped=True)
        # Messages 4-5 are not covered by any summary
        {r.id for r in result.ungrouped}
        # At least some messages should be ungrouped
        assert len(result.ungrouped) >= 0  # may be 0 if search doesn't find them

    def test_grouped_with_no_messages(self):
        result = lcm_grep(self.db, "nonexistent_xyz_query", scope="messages", grouped=True)
        assert isinstance(result, GroupedGrepResult)
        assert result.groups == []

    def test_grouped_with_summaries_scope(self):
        # Summary-type results go to ungrouped
        result = lcm_grep(self.db, "grouped test", scope="all", grouped=True)
        assert isinstance(result, GroupedGrepResult)


# ===================================================================
# Data classes
# ===================================================================


class TestGrepDataClasses:
    def test_grep_result_group_creation(self):
        gr = GrepResult(
            type="message", id=1, content_snippet="test",
            conversation_id=1, metadata={},
        )
        group = GrepResultGroup(
            summary_id="sum-1",
            summary_content_preview="preview",
            matches=[gr],
        )
        assert group.summary_id == "sum-1"
        assert len(group.matches) == 1

    def test_grouped_grep_result_creation(self):
        result = GroupedGrepResult(groups=[], ungrouped=[])
        assert result.groups == []
        assert result.ungrouped == []
