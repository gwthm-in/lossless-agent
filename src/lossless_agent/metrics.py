"""Best-effort, non-blocking usage-metrics emitter.

One JSON object per line, appended to ``metrics_dir()/YYYY-MM.jsonl``. Consumed by a shared
dashboard to compute call counts, zero-result rates, latency, and FTS-vs-vector contribution.

Everything in this module is fail-safe: a metrics write must never raise or slow down a tool
call. Every public function swallows its own exceptions.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Optional, Tuple


def _truthy_enabled(v: Optional[str]) -> bool:
    """Unset -> enabled. Only an explicit 0/false/no disables metrics."""
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no")


def metrics_enabled() -> bool:
    return _truthy_enabled(os.environ.get("LCM_METRICS_ENABLED"))


def metrics_dir() -> Path:
    """Directory metrics files are written to.

    Precedence: ``LCM_METRICS_DIR`` explicit override > ``LCM_CAPTURE_STATE_DIR``/metrics >
    ``~/.lossless-agent/metrics``.
    """
    override = os.environ.get("LCM_METRICS_DIR")
    if override:
        return Path(override)
    state_dir = os.environ.get("LCM_CAPTURE_STATE_DIR")
    if state_dir:
        return Path(state_dir) / "metrics"
    return Path.home() / ".lossless-agent" / "metrics"


def new_call_id() -> str:
    return uuid.uuid4().hex[:12]


def query_fingerprint(q: Optional[str]) -> Tuple[int, str]:
    """``(query_len, query_hash)`` for a query string. Never returns the raw text — queries can
    carry PII. ``query_hash`` is the first 12 hex chars of ``sha256(q)``; empty string if ``q``
    is falsy."""
    if not q:
        return 0, ""
    try:
        digest = hashlib.sha256(q.encode("utf-8")).hexdigest()[:12]
    except Exception:
        return len(q), ""
    return len(q), digest


def emit(event: dict) -> None:
    """Append one JSON line describing ``event`` to this month's metrics file.

    Best-effort: gated on ``LCM_METRICS_ENABLED`` (unset = on), and wrapped end-to-end in
    try/except so a metrics failure can never propagate into (or slow down) a tool call.
    """
    try:
        if not metrics_enabled():
            return

        evt = dict(event)
        if "ts" not in evt:
            evt["ts"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        if "call_id" not in evt:
            evt["call_id"] = new_call_id()

        out_dir = metrics_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        month = datetime.datetime.now().strftime("%Y-%m")
        path = out_dir / f"{month}.jsonl"
        line = json.dumps(evt, default=str)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # Metrics must never raise or block the caller.
        pass


def build_event(
    *,
    kind: str,
    tool: str,
    session_id: str = "",
    latency_ms: Any = 0,
    result_count: int = 0,
    hits: Optional[dict] = None,
    returned_ids: Optional[list] = None,
    target_id: str = "",
    query: Optional[str] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Build a metrics event dict matching the canonical schema. Kept separate from ``emit`` so
    it can be unit-tested without touching the filesystem, and so a failure while *building* the
    dict (e.g. a bad field) can be caught by the caller without also skipping the emit gate."""
    query_len, query_hash = query_fingerprint(query)
    event: dict = {
        "system": "lcm",
        "kind": kind,
        "tool": tool,
        "session_id": session_id or "",
        "call_id": new_call_id(),
        "latency_ms": int(round(latency_ms or 0)),
        "result_count": int(result_count or 0),
        "target_id": target_id or "",
        "query_len": query_len,
        "query_hash": query_hash,
        "extra": extra or {},
    }
    if kind == "retrieval":
        # A failed query is neither a hit nor a zero-result — don't let errors inflate the
        # zero-result rate (the headline metric). Only a *successful* empty query is zero-result.
        event["zero_result"] = (event["result_count"] == 0) and not (extra or {}).get("error")
        event["hits"] = hits or {"fts": 0, "vector": 0, "summary": 0}
        event["returned_ids"] = [str(i) for i in (returned_ids or [])][:10]
    return event
