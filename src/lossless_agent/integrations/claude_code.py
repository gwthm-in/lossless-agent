"""Claude Code capture integration — a zero-code Stop / SessionEnd hook.

Claude Code has two extension points: MCP *tools* (recall) and *hooks* (events). lossless
ships the tools half (``lossless-agent-mcp``); this is the hook half. Register the
``lossless-agent-capture`` console script and lossless captures every turn into a per-project
store and builds the summary DAG — no agent tool-calls, no custom glue.

``~/.claude/settings.json``::

    "Stop":       [{ "hooks": [{ "type": "command", "command": "lossless-agent-capture" }] }]
    "SessionEnd": [{ "hooks": [{ "type": "command", "command": "lossless-agent-capture --final" }] }]

Configuration is entirely via environment (see ``docs/configuration.md``)::

    LCM_DATABASE_DSN            explicit store DSN; else derived per-project from CLAUDE_PROJECT_DIR
    LCM_SUMMARY_PROVIDER        anthropic (recommended: fast, no cold-start) | openai | (unset -> truncation)
    ANTHROPIC_API_KEY           required when LCM_SUMMARY_PROVIDER=anthropic
    LCM_SUMMARY_MODEL           e.g. claude-haiku-4-5-20251001
    LCM_SUMMARIZE_COMMAND       alternative: external stdin->stdout summarizer command
    LCM_LEAF_CHUNK_TOKENS / LCM_SUMMARY_TIMEOUT_MS / ...   compaction tuning (honoured)

Capture runs through the *generic adapter's own* lifecycle (``on_turn_end`` per turn,
``on_session_end`` on ``--final``), so compaction, the semantic layer, and config are all the
library's — nothing is reimplemented here.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import json
import os
import re
import sys
from pathlib import Path

_STATE_DIR = Path(os.environ.get("LCM_CAPTURE_STATE_DIR", str(Path.home() / ".lossless-agent")))
_CURSOR_DIR = _STATE_DIR / "capture-cursors"
_LOG_FILE = _STATE_DIR / "capture.log"
# Set by our own summarizer subprocesses (command provider) so a summarizer's own headless
# agent session — which would inherit this same hook — never re-captures itself.
_GUARD_ENV = "LCM_SUMMARIZING"


def _log(msg: str) -> None:
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{ts} [{os.getpid()}] {msg}\n")
    except Exception:
        pass


def project_db_name(root: str) -> str:
    """Deterministic per-project database name: ``lcm_<basename>_<8 hex of sha256(path)>``.
    The basename stays human-readable; the path hash prevents same-name collisions."""
    base = re.sub(r"[^a-z0-9]", "_", os.path.basename(root.rstrip("/")).lower()) or "root"
    h = hashlib.sha256(root.encode("utf-8")).hexdigest()[:8]
    return f"lcm_{base}_{h}"


def resolve_dsn(payload: dict) -> str:
    """Explicit ``LCM_DATABASE_DSN`` wins; otherwise derive a per-project Postgres DSN from
    the project root (``CLAUDE_PROJECT_DIR``, the reliable root Claude Code injects)."""
    dsn = os.environ.get("LCM_DATABASE_DSN")
    if dsn:
        return dsn
    root = os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or os.getcwd()
    host = os.environ.get("LCM_PGHOST", "localhost")
    port = os.environ.get("LCM_PGPORT", "5432")
    return f"postgresql://{host}:{port}/{project_db_name(root)}"


def ensure_database(dsn: str) -> bool:
    """Create the (empty) database if missing — lossless builds its own tables + pgvector on
    first connect. No-op for non-Postgres DSNs. Returns True if the store is usable."""
    if not dsn.startswith("postgres"):
        return True
    try:
        import psycopg2
        from psycopg2 import sql
        dbname = dsn.rsplit("/", 1)[-1]
        admin = dsn.rsplit("/", 1)[0] + "/postgres"
        conn = psycopg2.connect(admin, connect_timeout=4)
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (dbname,))
                if cur.fetchone() is None:
                    cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
                    _log(f"created database {dbname}")
        finally:
            conn.close()
        return True
    except Exception as e:
        _log(f"ensure_database failed: {e}")
        return False


def parse_transcript(path: str) -> list[dict]:
    """Parse a Claude Code transcript JSONL into ``[{role, content}]`` (user/assistant text)."""
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("type") not in ("user", "assistant"):
                continue
            msg = obj.get("message")
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts = [
                    b.get("text", "") if isinstance(b, dict) and b.get("type") == "text"
                    else (b if isinstance(b, str) else "")
                    for b in content
                ]
                text = "\n".join(p for p in parts if p)
            else:
                text = ""
            text = (text or "").strip()
            if text:
                out.append({"role": role, "content": text})
    return out


def _cursor_path(session_key: str, dbname: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{dbname}.{session_key}")
    return _CURSOR_DIR / f"{safe}.txt"


async def _capture(dsn: str, session_key: str, messages: list[dict], final: bool) -> dict:
    """Drive the generic adapter's own lifecycle — this is the single, canonical capture path."""
    os.environ["LCM_DATABASE_DSN"] = dsn
    from lossless_agent.config import LCMConfig
    from lossless_agent.adapters.factory import create_adapter
    from lossless_agent.summarizers import build_summarize_fn

    config = LCMConfig.from_env()
    summarize_fn = build_summarize_fn(config, os.environ.get("LCM_SUMMARIZE_COMMAND"))
    adapter = create_adapter("generic", config, summarize_fn)
    try:
        if messages:
            await adapter.on_turn_end(session_key, messages)
        if final:
            await adapter.on_session_end(session_key)
    finally:
        closer = getattr(adapter, "aclose", None)
        if closer is not None:
            try:
                await closer()
            except Exception:
                pass
        else:
            closer = getattr(adapter, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass
    return {"ingested": len(messages), "final": final}


def main(argv=None) -> int:
    # A summarizer's own headless session inherits this hook; bail so it isn't captured.
    if os.environ.get(_GUARD_ENV):
        return 0

    ap = argparse.ArgumentParser(prog="lossless-agent-capture", description=__doc__)
    ap.add_argument("--final", action="store_true",
                    help="run the end-of-session compaction sweep (register on SessionEnd)")
    args = ap.parse_args(argv)

    try:
        payload = json.load(sys.stdin) if not sys.stdin.isatty() else {}
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    session_key = payload.get("session_id") or payload.get("session_key") or "default"
    dsn = resolve_dsn(payload)
    dbname = dsn.rsplit("/", 1)[-1]

    # Cursor dedup: the transcript grows every turn, so ingest only messages past the cursor.
    messages: list[dict] = []
    cursor = 0
    transcript_path = payload.get("transcript_path")
    if transcript_path and os.path.exists(transcript_path):
        all_msgs = parse_transcript(transcript_path)
        cursor_file = _cursor_path(session_key, dbname)
        try:
            cursor = int(cursor_file.read_text().strip() or "0")
        except Exception:
            cursor = 0
        messages = all_msgs[cursor:]

    if not messages and not args.final:
        return 0
    if not ensure_database(dsn):
        return 0

    try:
        asyncio.run(_capture(dsn, session_key, messages, args.final))
    except Exception as e:
        _log(f"capture failed (non-fatal, cursor not advanced): {e}")
        return 0

    if messages:
        try:
            _CURSOR_DIR.mkdir(parents=True, exist_ok=True)
            _cursor_path(session_key, dbname).write_text(str(cursor + len(messages)))
        except Exception:
            pass
        _log(f"ingested {len(messages)} message(s) -> {dbname} (session {session_key[:8]})")
    if args.final:
        _log(f"final compaction sweep -> {dbname} (session {session_key[:8]})")
    return 0


def cli() -> None:
    sys.exit(main())


if __name__ == "__main__":
    cli()
