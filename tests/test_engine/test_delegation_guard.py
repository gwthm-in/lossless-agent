"""Tests for the DelegationGuard scope-reduction invariant."""
from __future__ import annotations


from lossless_agent.engine.delegation_guard import DelegationGuard


class TestDelegationGuard:
    def setup_method(self):
        self.guard = DelegationGuard()

    # --- Exempt categories always allow ---

    def test_root_always_allowed(self):
        assert self.guard.validate_delegation("", "", is_root=True) is True

    def test_read_only_always_allowed(self):
        assert self.guard.validate_delegation("", "", is_read_only=True) is True

    def test_parallel_always_allowed(self):
        assert self.guard.validate_delegation("", "", is_parallel=True) is True

    def test_root_with_empty_scopes(self):
        assert self.guard.validate_delegation("", "", is_root=True) is True

    # --- Rejection cases ---

    def test_empty_delegated_scope_rejected(self):
        assert self.guard.validate_delegation("", "my work") is False

    def test_whitespace_delegated_scope_rejected(self):
        assert self.guard.validate_delegation("   ", "my work") is False

    def test_empty_kept_work_rejected(self):
        assert self.guard.validate_delegation("delegate this", "") is False

    def test_whitespace_kept_work_rejected(self):
        assert self.guard.validate_delegation("delegate this", "  ") is False

    def test_identical_scopes_rejected(self):
        assert self.guard.validate_delegation("do X", "do X") is False

    def test_identical_with_whitespace_rejected(self):
        assert self.guard.validate_delegation(" do X ", " do X ") is False

    # --- Valid cases ---

    def test_valid_delegation(self):
        assert self.guard.validate_delegation(
            "search the database", "analyze the results"
        ) is True

    def test_valid_different_scopes(self):
        assert self.guard.validate_delegation(
            "fetch data from API", "process and render results"
        ) is True

    # --- Static method ---

    def test_can_call_statically(self):
        assert DelegationGuard.validate_delegation(
            "task A", "task B"
        ) is True

    # --- Edge cases ---

    def test_both_empty_rejected_without_exemption(self):
        assert self.guard.validate_delegation("", "") is False

    def test_multiple_exemptions(self):
        assert self.guard.validate_delegation(
            "", "", is_root=True, is_read_only=True, is_parallel=True
        ) is True
