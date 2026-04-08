"""Expansion policy routing: decides how to handle expansion queries."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List

from lossless_agent.store.models import Summary


class PolicyAction(Enum):
    ANSWER_DIRECTLY = "answer_directly"
    EXPAND_SHALLOW = "expand_shallow"
    DELEGATE_TRAVERSAL = "delegate_traversal"


@dataclass
class PolicyDecision:
    action: PolicyAction
    reasons: List[str]


class ExpansionPolicy:
    """Decides expansion strategy based on query, candidates, and budget."""

    BASE_TOKENS_PER_SUMMARY = 500
    MAX_DEPTH = 1

    _BROAD_TIME_RE = re.compile(
        r"(?:last\s+month|past\s+year|all\s+time|history|everything|since\s|over\s+the\s+past)",
        re.IGNORECASE,
    )

    _MULTI_HOP_RE = re.compile(
        r"(?:compare|relationship\s+between|how\s+did\s+\w+\s+affect)",
        re.IGNORECASE,
    )

    def decide(
        self,
        query: str,
        candidates: List[Summary],
        token_budget: int,
        current_depth: int = 0,
    ) -> PolicyDecision:
        reasons: List[str] = []

        if not candidates:
            reasons.append("No candidates to expand")
            return PolicyDecision(action=PolicyAction.ANSWER_DIRECTLY, reasons=reasons)

        if current_depth >= self.MAX_DEPTH:
            reasons.append(f"Recursion depth limit reached ({current_depth} >= {self.MAX_DEPTH})")
            return PolicyDecision(action=PolicyAction.ANSWER_DIRECTLY, reasons=reasons)

        # Estimate tokens
        multiplier = 1.0
        broad_time = self._detect_broad_time_range(query)
        multi_hop = self._detect_multi_hop(query, candidates, current_depth)

        if broad_time:
            multiplier *= 1.5
            reasons.append("Broad time range detected")
        if multi_hop:
            multiplier *= 1.3
            reasons.append("Multi-hop query detected")

        estimated = self.BASE_TOKENS_PER_SUMMARY * len(candidates) * multiplier
        ratio = estimated / token_budget if token_budget > 0 else float("inf")

        reasons.append(f"Token ratio: {ratio:.2f} ({len(candidates)} candidates)")

        if ratio < 0.35:
            reasons.append("Low token risk")
            return PolicyDecision(action=PolicyAction.EXPAND_SHALLOW, reasons=reasons)
        elif ratio <= 0.7:
            if len(candidates) <= 3:
                reasons.append("Moderate risk but few candidates")
                return PolicyDecision(action=PolicyAction.EXPAND_SHALLOW, reasons=reasons)
            else:
                reasons.append("Moderate risk with many candidates")
                return PolicyDecision(action=PolicyAction.DELEGATE_TRAVERSAL, reasons=reasons)
        else:
            reasons.append("High token risk")
            return PolicyDecision(action=PolicyAction.DELEGATE_TRAVERSAL, reasons=reasons)

    def _detect_broad_time_range(self, query: str) -> bool:
        return bool(self._BROAD_TIME_RE.search(query))

    def _detect_multi_hop(
        self, query: str, candidates: List[Summary], current_depth: int
    ) -> bool:
        if current_depth >= 3:
            return True
        if len(candidates) >= 5:
            return True
        if self._MULTI_HOP_RE.search(query):
            return True
        return False
