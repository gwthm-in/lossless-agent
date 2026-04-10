"""Circuit breaker for compaction summarisation calls.

Tracks consecutive failures per key (e.g. conversation ID) and opens the
circuit once a threshold is reached, preventing further calls until a
cooldown period has elapsed.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass
class _BreakerState:
    """Internal state for a single circuit-breaker key."""
    failures: int = 0
    last_failure_ts: float = 0.0


class CircuitBreaker:
    """Per-key circuit breaker with configurable threshold and cooldown."""

    def __init__(
        self,
        threshold: int = 5,
        cooldown_ms: int = 30_000,
    ) -> None:
        self.threshold = threshold
        self.cooldown_ms = cooldown_ms
        self._states: Dict[str, _BreakerState] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_failure(self, key: str) -> None:
        """Increment consecutive failure count for *key*."""
        state = self._states.setdefault(key, _BreakerState())
        state.failures += 1
        state.last_failure_ts = time.monotonic()
        logger.debug(
            "CircuitBreaker: failure #%d for key=%s", state.failures, key
        )

    def record_success(self, key: str) -> None:
        """Reset failure count for *key*."""
        if key in self._states:
            self._states[key] = _BreakerState()

    def is_open(self, key: str) -> bool:
        """Return True if circuit is open (calls should be skipped).

        The circuit opens when failures >= threshold **and** the cooldown
        period has not yet elapsed since the last failure.
        """
        state = self._states.get(key)
        if state is None:
            return False
        if state.failures < self.threshold:
            return False
        elapsed_ms = (time.monotonic() - state.last_failure_ts) * 1000
        if elapsed_ms >= self.cooldown_ms:
            # Cooldown elapsed – half-open: allow next attempt
            return False
        return True

    def reset(self, key: str) -> None:
        """Fully reset state for *key*."""
        self._states.pop(key, None)
