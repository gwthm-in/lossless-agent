"""Tests for the best-effort usage-metrics emitter (lossless_agent.metrics)."""
from __future__ import annotations

import json

import pytest

from lossless_agent import metrics


# ------------------------------------------------------------------
# metrics_dir()
# ------------------------------------------------------------------

class TestMetricsDir:
    def test_default_dir(self, monkeypatch):
        monkeypatch.delenv("LCM_METRICS_DIR", raising=False)
        monkeypatch.delenv("LCM_CAPTURE_STATE_DIR", raising=False)
        from pathlib import Path
        assert metrics.metrics_dir() == Path.home() / ".lossless-agent" / "metrics"

    def test_respects_capture_state_dir(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LCM_METRICS_DIR", raising=False)
        monkeypatch.setenv("LCM_CAPTURE_STATE_DIR", str(tmp_path))
        assert metrics.metrics_dir() == tmp_path / "metrics"

    def test_explicit_override_wins(self, monkeypatch, tmp_path):
        override = tmp_path / "custom-metrics"
        monkeypatch.setenv("LCM_METRICS_DIR", str(override))
        monkeypatch.setenv("LCM_CAPTURE_STATE_DIR", str(tmp_path / "state"))
        assert metrics.metrics_dir() == override


# ------------------------------------------------------------------
# emit()
# ------------------------------------------------------------------

class TestEmit:
    def test_writes_parseable_jsonl_with_required_keys(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LCM_METRICS_DIR", str(tmp_path))
        monkeypatch.delenv("LCM_METRICS_ENABLED", raising=False)

        event = metrics.build_event(
            kind="retrieval", tool="lcm_grep", session_id="sess-1",
            latency_ms=12.6, result_count=3,
            hits={"fts": 2, "vector": 1, "summary": 0},
            returned_ids=[1, 2, 3],
            query="hello world",
        )
        metrics.emit(event)

        files = list(tmp_path.glob("*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text().strip().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])

        required = {
            "ts", "system", "kind", "tool", "session_id", "call_id",
            "latency_ms", "result_count", "zero_result", "hits",
            "returned_ids", "target_id", "query_len", "query_hash", "extra",
        }
        assert required.issubset(parsed.keys())
        assert parsed["system"] == "lcm"
        assert parsed["kind"] == "retrieval"
        assert parsed["tool"] == "lcm_grep"
        assert parsed["zero_result"] is False
        assert parsed["hits"] == {"fts": 2, "vector": 1, "summary": 0}
        assert parsed["returned_ids"] == ["1", "2", "3"]
        assert parsed["latency_ms"] == 13  # rounded int

    def test_fills_ts_and_call_id_if_absent(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LCM_METRICS_DIR", str(tmp_path))
        monkeypatch.delenv("LCM_METRICS_ENABLED", raising=False)

        metrics.emit({"system": "lcm", "kind": "capture", "tool": "capture"})

        files = list(tmp_path.glob("*.jsonl"))
        parsed = json.loads(files[0].read_text().strip().splitlines()[0])
        assert parsed["ts"]
        assert len(parsed["call_id"]) == 12

    def test_noop_when_disabled(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LCM_METRICS_DIR", str(tmp_path))
        monkeypatch.setenv("LCM_METRICS_ENABLED", "0")

        metrics.emit({"system": "lcm", "kind": "capture", "tool": "capture"})

        assert list(tmp_path.glob("*.jsonl")) == []

    @pytest.mark.parametrize("value", ["0", "false", "False", "no", "NO"])
    def test_various_falsy_values_disable(self, monkeypatch, tmp_path, value):
        monkeypatch.setenv("LCM_METRICS_DIR", str(tmp_path))
        monkeypatch.setenv("LCM_METRICS_ENABLED", value)

        metrics.emit({"system": "lcm", "kind": "capture", "tool": "capture"})

        assert list(tmp_path.glob("*.jsonl")) == []

    def test_unset_env_is_enabled(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LCM_METRICS_DIR", str(tmp_path))
        monkeypatch.delenv("LCM_METRICS_ENABLED", raising=False)

        metrics.emit({"system": "lcm", "kind": "capture", "tool": "capture"})

        assert len(list(tmp_path.glob("*.jsonl"))) == 1

    def test_never_raises_on_unwritable_dir(self, monkeypatch):
        # A path nested under a file (not a directory) can never be mkdir'd into.
        monkeypatch.setenv("LCM_METRICS_DIR", "/dev/null/impossible/metrics-dir")
        monkeypatch.delenv("LCM_METRICS_ENABLED", raising=False)

        # Must return cleanly — no exception escapes.
        metrics.emit({"system": "lcm", "kind": "capture", "tool": "capture"})

    def test_never_raises_on_bad_event_type(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LCM_METRICS_DIR", str(tmp_path))
        monkeypatch.delenv("LCM_METRICS_ENABLED", raising=False)

        # Not even a dict — emit() must still swallow the failure.
        metrics.emit(None)  # type: ignore[arg-type]

    def test_appends_multiple_events_same_month_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LCM_METRICS_DIR", str(tmp_path))
        monkeypatch.delenv("LCM_METRICS_ENABLED", raising=False)

        metrics.emit({"system": "lcm", "kind": "capture", "tool": "capture"})
        metrics.emit({"system": "lcm", "kind": "capture", "tool": "capture"})

        files = list(tmp_path.glob("*.jsonl"))
        assert len(files) == 1
        assert len(files[0].read_text().strip().splitlines()) == 2


# ------------------------------------------------------------------
# query_fingerprint()
# ------------------------------------------------------------------

class TestQueryFingerprint:
    def test_length_and_stable_hash(self):
        q = "what did we decide about the schema?"
        length, digest = metrics.query_fingerprint(q)
        assert length == len(q)
        assert len(digest) == 12
        assert all(c in "0123456789abcdef" for c in digest)

        # Deterministic across calls.
        length2, digest2 = metrics.query_fingerprint(q)
        assert (length2, digest2) == (length, digest)

    def test_never_echoes_raw_text(self):
        q = "super secret raw query text with PII"
        _length, digest = metrics.query_fingerprint(q)
        assert q not in digest

    def test_empty_or_none(self):
        assert metrics.query_fingerprint(None) == (0, "")
        assert metrics.query_fingerprint("") == (0, "")

    def test_different_queries_differ(self):
        _, d1 = metrics.query_fingerprint("alpha")
        _, d2 = metrics.query_fingerprint("beta")
        assert d1 != d2


# ------------------------------------------------------------------
# new_call_id()
# ------------------------------------------------------------------

class TestNewCallId:
    def test_format(self):
        cid = metrics.new_call_id()
        assert len(cid) == 12
        assert all(c in "0123456789abcdef" for c in cid)

    def test_unique(self):
        ids = {metrics.new_call_id() for _ in range(50)}
        assert len(ids) == 50


# ------------------------------------------------------------------
# build_event()
# ------------------------------------------------------------------

class TestBuildEvent:
    def test_retrieval_sets_zero_result_true(self):
        event = metrics.build_event(kind="retrieval", tool="lcm_grep", result_count=0)
        assert event["zero_result"] is True
        assert event["hits"] == {"fts": 0, "vector": 0, "summary": 0}
        assert event["returned_ids"] == []

    def test_retrieval_caps_returned_ids_at_ten(self):
        event = metrics.build_event(
            kind="retrieval", tool="lcm_grep", result_count=15,
            returned_ids=list(range(15)),
        )
        assert len(event["returned_ids"]) == 10
        assert all(isinstance(i, str) for i in event["returned_ids"])

    def test_non_retrieval_has_no_zero_result_field(self):
        event = metrics.build_event(kind="capture", tool="capture", extra={"ingested": 3, "deduped": 0})
        assert "zero_result" not in event
        assert "hits" not in event
        assert event["extra"] == {"ingested": 3, "deduped": 0}

    def test_query_fields_derived_from_query(self):
        event = metrics.build_event(kind="retrieval", tool="lcm_grep", query="hello")
        assert event["query_len"] == 5
        assert len(event["query_hash"]) == 12
        assert "hello" not in json.dumps(event)


# ------------------------------------------------------------------
# Dispatch-level: exercised via the mcp_server module using a real in-memory DB
# ------------------------------------------------------------------

class TestDispatchLevelMetrics:
    """Confirms lcm_grep's call_tool dispatch actually emits a retrieval event with the right
    tool name and zero_result flag, end to end (real SQLite, no mocked store)."""

    @pytest.fixture(autouse=True)
    def _metrics_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LCM_METRICS_DIR", str(tmp_path))
        monkeypatch.delenv("LCM_METRICS_ENABLED", raising=False)
        self._dir = tmp_path

    def _events(self):
        files = list(self._dir.glob("*.jsonl"))
        if not files:
            return []
        return [json.loads(line) for line in files[0].read_text().strip().splitlines()]

    @pytest.mark.asyncio
    async def test_lcm_grep_zero_results_emits_event(self):
        import lossless_agent.mcp_server as mcp_mod
        from lossless_agent.store.database import Database

        database = Database(":memory:")
        try:
            mcp_mod._db = database
            await mcp_mod.call_tool("lcm_grep", {"query": "nothing matches anything here"})
        finally:
            database.close()
            mcp_mod._db = None

        events = self._events()
        assert len(events) == 1
        assert events[0]["tool"] == "lcm_grep"
        assert events[0]["kind"] == "retrieval"
        assert events[0]["zero_result"] is True
        assert events[0]["result_count"] == 0

    @pytest.mark.asyncio
    async def test_unknown_tool_emits_error_capture_event(self):
        import lossless_agent.mcp_server as mcp_mod
        from lossless_agent.store.database import Database

        database = Database(":memory:")
        try:
            mcp_mod._db = database
            await mcp_mod.call_tool("totally_unknown_tool", {})
        finally:
            database.close()
            mcp_mod._db = None

        events = self._events()
        assert len(events) == 1
        assert events[0]["tool"] == "totally_unknown_tool"
        assert events[0]["extra"] == {"error": True}
