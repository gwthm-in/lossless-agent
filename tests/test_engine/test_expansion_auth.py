"""Tests for expansion auth system."""
import time

import pytest

from lossless_agent.engine.expansion_auth import (
    ExpansionAuthManager,
    Grant,
    InvalidGrantError,
    ExhaustedBudgetError,
)


@pytest.fixture
def auth():
    return ExpansionAuthManager()


class TestCreateGrant:
    def test_creates_grant_with_defaults(self, auth):
        grant = auth.create_grant("session-1", ["conv-a", "conv-b"])
        assert isinstance(grant, Grant)
        assert grant.issuer_session_id == "session-1"
        assert grant.allowed_conversation_ids == ["conv-a", "conv-b"]
        assert grant.allowed_summary_ids is None
        assert grant.max_depth == 1
        assert grant.token_cap == 4000
        assert grant.tokens_consumed == 0
        assert grant.revoked is False
        assert grant.expires_at > time.time()

    def test_creates_grant_with_custom_params(self, auth):
        grant = auth.create_grant(
            "session-2", ["conv-x"], allowed_summary_ids=["s1", "s2"],
            max_depth=3, token_cap=8000, ttl_seconds=600,
        )
        assert grant.allowed_summary_ids == ["s1", "s2"]
        assert grant.max_depth == 3
        assert grant.token_cap == 8000

    def test_grant_id_is_unique(self, auth):
        g1 = auth.create_grant("s1", ["c1"])
        g2 = auth.create_grant("s1", ["c1"])
        assert g1.grant_id != g2.grant_id


class TestValidateGrant:
    def test_valid_grant_returns_grant(self, auth):
        grant = auth.create_grant("s1", ["c1"])
        result = auth.validate_grant(grant.grant_id)
        assert result.grant_id == grant.grant_id

    def test_nonexistent_grant_raises(self, auth):
        with pytest.raises(InvalidGrantError):
            auth.validate_grant("bogus-id")

    def test_revoked_grant_raises(self, auth):
        grant = auth.create_grant("s1", ["c1"])
        auth.revoke_grant(grant.grant_id)
        with pytest.raises(InvalidGrantError):
            auth.validate_grant(grant.grant_id)

    def test_expired_grant_raises(self, auth):
        grant = auth.create_grant("s1", ["c1"], ttl_seconds=0)
        time.sleep(0.01)
        with pytest.raises(InvalidGrantError):
            auth.validate_grant(grant.grant_id)


class TestValidateScope:
    def test_allowed_conversation(self, auth):
        grant = auth.create_grant("s1", ["c1", "c2"])
        assert auth.validate_scope(grant.grant_id, "c1") is True

    def test_disallowed_conversation(self, auth):
        grant = auth.create_grant("s1", ["c1"])
        assert auth.validate_scope(grant.grant_id, "c99") is False

    def test_summary_id_allowed_when_none(self, auth):
        grant = auth.create_grant("s1", ["c1"], allowed_summary_ids=None)
        assert auth.validate_scope(grant.grant_id, "c1", summary_id="any-sum") is True

    def test_summary_id_allowed_when_in_list(self, auth):
        grant = auth.create_grant("s1", ["c1"], allowed_summary_ids=["s1", "s2"])
        assert auth.validate_scope(grant.grant_id, "c1", summary_id="s1") is True

    def test_summary_id_denied_when_not_in_list(self, auth):
        grant = auth.create_grant("s1", ["c1"], allowed_summary_ids=["s1"])
        assert auth.validate_scope(grant.grant_id, "c1", summary_id="s99") is False


class TestTokenBudget:
    def test_consume_returns_remaining(self, auth):
        grant = auth.create_grant("s1", ["c1"], token_cap=1000)
        remaining = auth.consume_token_budget(grant.grant_id, 300)
        assert remaining == 700

    def test_consume_multiple_times(self, auth):
        grant = auth.create_grant("s1", ["c1"], token_cap=1000)
        auth.consume_token_budget(grant.grant_id, 300)
        remaining = auth.consume_token_budget(grant.grant_id, 200)
        assert remaining == 500

    def test_consume_exceeding_budget_raises(self, auth):
        grant = auth.create_grant("s1", ["c1"], token_cap=100)
        with pytest.raises(ExhaustedBudgetError):
            auth.consume_token_budget(grant.grant_id, 200)

    def test_consume_exact_budget(self, auth):
        grant = auth.create_grant("s1", ["c1"], token_cap=100)
        remaining = auth.consume_token_budget(grant.grant_id, 100)
        assert remaining == 0

    def test_get_remaining_budget(self, auth):
        grant = auth.create_grant("s1", ["c1"], token_cap=500)
        auth.consume_token_budget(grant.grant_id, 150)
        assert auth.get_remaining_budget(grant.grant_id) == 350


class TestRevokeGrant:
    def test_revoke_makes_grant_invalid(self, auth):
        grant = auth.create_grant("s1", ["c1"])
        auth.revoke_grant(grant.grant_id)
        with pytest.raises(InvalidGrantError):
            auth.validate_grant(grant.grant_id)


class TestCleanupExpired:
    def test_removes_expired_grants(self, auth):
        auth.create_grant("s1", ["c1"], ttl_seconds=0)
        auth.create_grant("s1", ["c2"], ttl_seconds=0)
        auth.create_grant("s1", ["c3"], ttl_seconds=3600)
        time.sleep(0.01)
        removed = auth.cleanup_expired()
        assert removed == 2

    def test_returns_zero_when_none_expired(self, auth):
        auth.create_grant("s1", ["c1"], ttl_seconds=3600)
        assert auth.cleanup_expired() == 0
