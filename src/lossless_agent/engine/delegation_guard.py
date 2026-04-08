"""Scope-reduction invariant for sub-agent delegation."""
from __future__ import annotations


class DelegationGuard:
    """Validates that delegated work reduces scope (whitepaper §3.4).

    The scope-reduction invariant prevents infinite delegation loops by
    ensuring that:
    - The delegated work is non-empty
    - The caller retains some work (not delegating 100%)
    - The delegated scope is not identical to the kept work
    - Root, read-only, and parallel delegations are always allowed
    """

    @staticmethod
    def validate_delegation(
        delegated_scope: str,
        kept_work: str,
        *,
        is_root: bool = False,
        is_read_only: bool = False,
        is_parallel: bool = False,
    ) -> bool:
        """Check whether the delegation satisfies the scope-reduction invariant.

        Args:
            delegated_scope: Description of the work being delegated.
            kept_work: Description of the work the caller retains.
            is_root: True if this is the top-level (root) agent – always allowed.
            is_read_only: True if the delegation is read-only – always allowed.
            is_parallel: True if running in parallel – always allowed.

        Returns:
            True if the delegation is valid, False otherwise.
        """
        # Exempt categories: always allow
        if is_root or is_read_only or is_parallel:
            return True

        # Delegated scope must be non-empty
        if not delegated_scope or not delegated_scope.strip():
            return False

        # Caller must retain some work (not delegating 100%)
        if not kept_work or not kept_work.strip():
            return False

        # Suspiciously identical scopes suggest no real reduction
        if delegated_scope.strip() == kept_work.strip():
            return False

        return True
