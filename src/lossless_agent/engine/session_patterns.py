"""Stateless session pattern matching with glob-style patterns."""
from __future__ import annotations

import re
from typing import List, Optional


class SessionPatternMatcher:
    """Match session keys against glob patterns for ignore/stateless classification."""

    def __init__(
        self,
        ignore_patterns: Optional[List[str]] = None,
        stateless_patterns: Optional[List[str]] = None,
    ) -> None:
        self._ignore_re = [self._glob_to_regex(p) for p in (ignore_patterns or [])]
        self._stateless_re = [self._glob_to_regex(p) for p in (stateless_patterns or [])]

    @staticmethod
    def _glob_to_regex(pattern: str) -> re.Pattern:
        """Convert a glob pattern to a compiled regex.

        * matches non-colon chars ([^:]*)
        ** matches anything (.*)
        All other characters are escaped.
        """
        result = []
        i = 0
        while i < len(pattern):
            if i + 1 < len(pattern) and pattern[i] == "*" and pattern[i + 1] == "*":
                result.append(".*")
                i += 2
            elif pattern[i] == "*":
                result.append("[^:]*")
                i += 1
            else:
                result.append(re.escape(pattern[i]))
                i += 1
        return re.compile("^" + "".join(result) + "$")

    def is_ignored(self, session_key: str) -> bool:
        """Return True if the session key matches any ignore pattern."""
        return any(r.match(session_key) for r in self._ignore_re)

    def is_stateless(self, session_key: str) -> bool:
        """Return True if the session key matches any stateless pattern."""
        return any(r.match(session_key) for r in self._stateless_re)

    def should_persist(self, session_key: str) -> bool:
        """Return True if the session should be persisted (not ignored and not stateless)."""
        return not self.is_ignored(session_key) and not self.is_stateless(session_key)
