"""Expansion auth system: grant-based authorization for expansion queries."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional


class InvalidGrantError(Exception):
    """Raised when a grant is invalid, revoked, or expired."""


class ExhaustedBudgetError(Exception):
    """Raised when token budget is exhausted."""


@dataclass
class Grant:
    grant_id: str
    issuer_session_id: str
    allowed_conversation_ids: List[str]
    allowed_summary_ids: Optional[List[str]]
    max_depth: int = 1
    token_cap: int = 4000
    tokens_consumed: int = 0
    expires_at: float = 0.0
    revoked: bool = False


class ExpansionAuthManager:
    """Manages grants for expansion queries."""

    def __init__(self) -> None:
        self._grants: Dict[str, Grant] = {}

    def create_grant(
        self,
        issuer_session_id: str,
        allowed_conversation_ids: List[str],
        allowed_summary_ids: Optional[List[str]] = None,
        max_depth: int = 1,
        token_cap: int = 4000,
        ttl_seconds: int = 300,
    ) -> Grant:
        grant = Grant(
            grant_id=str(uuid.uuid4()),
            issuer_session_id=issuer_session_id,
            allowed_conversation_ids=allowed_conversation_ids,
            allowed_summary_ids=allowed_summary_ids,
            max_depth=max_depth,
            token_cap=token_cap,
            expires_at=time.time() + ttl_seconds,
        )
        self._grants[grant.grant_id] = grant
        return grant

    def validate_grant(self, grant_id: str) -> Grant:
        grant = self._grants.get(grant_id)
        if grant is None:
            raise InvalidGrantError(f"Grant {grant_id} not found")
        if grant.revoked:
            raise InvalidGrantError(f"Grant {grant_id} has been revoked")
        if time.time() > grant.expires_at:
            raise InvalidGrantError(f"Grant {grant_id} has expired")
        return grant

    def validate_scope(
        self, grant_id: str, conversation_id: str, summary_id: Optional[str] = None
    ) -> bool:
        grant = self.validate_grant(grant_id)
        if conversation_id not in grant.allowed_conversation_ids:
            return False
        if summary_id is not None and grant.allowed_summary_ids is not None:
            if summary_id not in grant.allowed_summary_ids:
                return False
        return True

    def consume_token_budget(self, grant_id: str, tokens: int) -> int:
        grant = self.validate_grant(grant_id)
        remaining = grant.token_cap - grant.tokens_consumed
        if tokens > remaining:
            raise ExhaustedBudgetError(
                f"Requested {tokens} tokens but only {remaining} remaining"
            )
        grant.tokens_consumed += tokens
        return grant.token_cap - grant.tokens_consumed

    def get_remaining_budget(self, grant_id: str) -> int:
        grant = self.validate_grant(grant_id)
        return grant.token_cap - grant.tokens_consumed

    def revoke_grant(self, grant_id: str) -> None:
        grant = self._grants.get(grant_id)
        if grant is not None:
            grant.revoked = True

    def cleanup_expired(self) -> int:
        now = time.time()
        expired = [
            gid for gid, g in self._grants.items() if now > g.expires_at
        ]
        for gid in expired:
            del self._grants[gid]
        return len(expired)
