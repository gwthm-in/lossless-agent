"""Tests for the circuit breaker."""
from __future__ import annotations

import time

from lossless_agent.engine.circuit_breaker import CircuitBreaker


class TestCircuitBreaker:
    def test_initially_closed(self):
        cb = CircuitBreaker(threshold=3, cooldown_ms=1000)
        assert cb.is_open("k1") is False

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker(threshold=3, cooldown_ms=1000)
        cb.record_failure("k1")
        cb.record_failure("k1")
        assert cb.is_open("k1") is False

    def test_opens_at_threshold(self):
        cb = CircuitBreaker(threshold=3, cooldown_ms=60_000)
        for _ in range(3):
            cb.record_failure("k1")
        assert cb.is_open("k1") is True

    def test_keys_are_independent(self):
        cb = CircuitBreaker(threshold=2, cooldown_ms=60_000)
        cb.record_failure("k1")
        cb.record_failure("k1")
        assert cb.is_open("k1") is True
        assert cb.is_open("k2") is False

    def test_success_resets(self):
        cb = CircuitBreaker(threshold=2, cooldown_ms=60_000)
        cb.record_failure("k1")
        cb.record_failure("k1")
        assert cb.is_open("k1") is True
        cb.record_success("k1")
        assert cb.is_open("k1") is False

    def test_reset_clears_state(self):
        cb = CircuitBreaker(threshold=2, cooldown_ms=60_000)
        cb.record_failure("k1")
        cb.record_failure("k1")
        cb.reset("k1")
        assert cb.is_open("k1") is False

    def test_cooldown_closes_circuit(self):
        cb = CircuitBreaker(threshold=2, cooldown_ms=100)
        cb.record_failure("k1")
        cb.record_failure("k1")
        assert cb.is_open("k1") is True
        # Simulate time passing beyond cooldown
        time.sleep(0.15)
        assert cb.is_open("k1") is False

    def test_success_on_unknown_key_is_noop(self):
        cb = CircuitBreaker()
        cb.record_success("unknown")  # should not raise

    def test_reset_on_unknown_key_is_noop(self):
        cb = CircuitBreaker()
        cb.reset("unknown")  # should not raise
