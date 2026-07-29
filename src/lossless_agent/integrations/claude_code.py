"""Claude Code capture integration — a zero-code Stop / SessionEnd hook.

Claude Code has two extension points: MCP *tools* (recall) and *hooks* (events). lossless
ships the tools half (``lossless-agent-mcp``); this is the hook half. Register the
``lossless-agent-capture`` console script and lossless captures every turn into a per-project
store and builds the summary DAG — no agent tool-calls, no custom glue.

``~/.claude/settings.json``::

    "Stop":       [{ "hooks": [{ "type": "command", "command": "lossless-agent-capture" }] }]
    "SessionEnd": [{ "hooks": [{ "type": "command", "command": "lossless-agent-capture --final" }] }]

Configuration is entirely via environment (see ``docs/configuration.md``)::

    (default, no config)        shared SQLite store at ~/.lossless-agent/lcm.db — the SAME store
                                the MCP server defaults to, so recall sees captures. Zero deps.
    LCM_DATABASE_DSN            explicit Postgres DSN (unlocks the pgvector semantic layer)
    LCM_DATABASE_PATH           explicit SQLite path
    LCM_CAPTURE_POSTGRES=1      per-project Postgres (lcm_<base>_<hash>); pair with a matching MCP
    LCM_CAPTURE_ASYNC=1         non-blocking capture: fork a detached worker and return in ms so
                                summarization never blocks the turn / SessionEnd (workers serialize
                                per-conversation via a flock, so per-turn firing can't double-ingest)
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
import contextlib
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

try:
    import fcntl  # POSIX only; used to serialize concurrent async-capture workers
except ImportError:  # pragma: no cover - Windows
    fcntl = None

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
    """Deterministic per-project store name: ``lcm_<basename>_<8 hex of sha256(path)>``.
    The basename stays human-readable; the path hash prevents same-name collisions."""
    base = re.sub(r"[^a-z0-9]", "_", os.path.basename(root.rstrip("/")).lower()) or "root"
    base = base[:48]   # keep "lcm_" + slug + "_" + 8-hex under Postgres's 63-byte identifier limit
    h = hashlib.sha256(root.encode("utf-8")).hexdigest()[:8]
    return f"lcm_{base}_{h}"


def _truthy(v) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def _project_root(payload: dict) -> str:
    return os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or os.getcwd()


def build_config(payload: dict):
    """Build the ``LCMConfig`` for this project's store. Precedence is chosen so the hook and the
    MCP recall server land on the *same* store out of the box:

      * ``LCM_DATABASE_DSN`` / ``LCM_DATABASE_PATH`` set -> that explicit store (the MCP server
        honours the same env now that its ``--db-path`` defaults to the env/config value).
      * ``LCM_CAPTURE_POSTGRES`` truthy -> a per-project Postgres DSN ``lcm_<base>_<hash>``
        (opt-in; pair it with an MCP server pointed at the same per-project DB).
      * otherwise -> the library default SQLite store (``~/.lossless-agent/lcm.db``), which the
        MCP server also defaults to — so recall sees captures with zero configuration.
    """
    from lossless_agent.config import LCMConfig

    config = LCMConfig.from_env()
    if config.database_dsn or os.environ.get("LCM_DATABASE_PATH"):
        return config
    if _truthy(os.environ.get("LCM_CAPTURE_POSTGRES")):
        host = os.environ.get("LCM_PGHOST", "localhost")
        port = os.environ.get("LCM_PGPORT", "5432")
        config.database_dsn = f"postgresql://{host}:{port}/{project_db_name(_project_root(payload))}"
    return config


def _store_label(config) -> str:
    """Short, stable identifier for this store (for cursor filenames + logs)."""
    if config.database_dsn:
        return urlsplit(config.database_dsn).path.lstrip("/") or "lcm"
    return os.path.splitext(os.path.basename(config.resolved_db_path))[0]


def _pg_target(dsn: str):
    """Return ``(dbname, admin_dsn)`` for a Postgres DSN, parsing the URL so query options
    (e.g. ``?sslmode=require``) land on neither the database name nor the admin path.
    ``postgresql://u@h/lcm?sslmode=require`` -> ``("lcm", "postgresql://u@h/postgres?sslmode=require")``."""
    parts = urlsplit(dsn)
    dbname = parts.path.lstrip("/")
    admin = urlunsplit((parts.scheme, parts.netloc, "/postgres", parts.query, parts.fragment))
    return dbname, admin


def ensure_database(dsn: str) -> bool:
    """True if the target Postgres database is reachable. If it doesn't exist yet, try to create
    it via the server's admin (``postgres``) database — but never *require* admin access when the
    target already exists: managed / least-privilege roles frequently cannot connect to the
    ``postgres`` database at all, and doing so unconditionally would block capture."""
    try:
        import psycopg2
    except Exception as e:
        _log(f"ensure_database: psycopg2 not installed ({e}); install lossless-agent[postgres]")
        return False

    # 1) Target reachable? Then we're done — no admin connection needed.
    target_err = None
    try:
        psycopg2.connect(dsn, connect_timeout=4).close()
        return True
    except Exception as e:
        target_err = e

    # 2) Target unreachable (typically: does not exist yet) — try to create it via admin.
    try:
        from psycopg2 import sql
        dbname, admin = _pg_target(dsn)          # 'lcm', not 'lcm?sslmode=…'
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
    except Exception as admin_err:
        _log(f"ensure_database: target unreachable ({target_err}); admin create failed ({admin_err})")
        return False


def prepare_store(config) -> bool:
    """Ensure the store exists. Postgres: create the DB if missing. SQLite: create the parent
    directory (the library creates the file + schema on first connect)."""
    if config.database_dsn:
        return ensure_database(config.database_dsn)
    try:
        parent = os.path.dirname(config.resolved_db_path)
        if parent:                               # a bare filename has no parent dir
            os.makedirs(parent, exist_ok=True)
        return True
    except Exception as e:
        _log(f"prepare_store (sqlite) failed: {e}")
        return False


# Text blocks that are injected context / harness noise, not real conversation — filtered out of
# capture so they don't pollute memory (they otherwise outrank real content in recall).
_INJECTED_PREFIXES = (
    "<system-reminder>", "<task-notification>", "<command-name>", "<command-message>",
    "<command-args>", "<local-command-stdout>", "[Request interrupted",
)
_INJECTED_SUBSTRINGS = (
    "Summarize the session transcript below",   # summarizer-prompt echo
    "You are compacting a long AI-agent working session",
    "You are merging several lower-level summaries",
)


def _is_injected(text: str) -> bool:
    """True if a text block is harness-injected context rather than conversation."""
    t = text.lstrip()
    return t.startswith(_INJECTED_PREFIXES) or any(s in text for s in _INJECTED_SUBSTRINGS)


def parse_transcript(path: str) -> list[dict]:
    """Parse a Claude Code transcript JSONL into ``[{role, content}]`` (user/assistant text),
    dropping harness-injected blocks (system-reminders, task-notifications, slash-command noise,
    summarizer-prompt echoes) so they don't pollute memory."""
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
                text = "" if _is_injected(content) else content
            elif isinstance(content, list):
                parts = []
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text":
                        bt = b.get("text", "")
                    elif isinstance(b, str):
                        bt = b
                    else:
                        bt = ""
                    if bt and not _is_injected(bt):
                        parts.append(bt)
                text = "\n".join(parts)
            else:
                text = ""
            text = (text or "").strip()
            if text:
                out.append({"role": role, "content": text})
    return out


def _cursor_path(session_key: str, store_label: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{store_label}.{session_key}")
    return _CURSOR_DIR / f"{safe}.txt"


def _lock_path(session_key: str, store_label: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{store_label}.{session_key}")
    return _CURSOR_DIR / f"{safe}.lock"


@contextlib.contextmanager
def _capture_lock(session_key: str, store_label: str):
    """Serialize capture for one conversation. Async mode fires a detached worker per turn, so two
    workers can race the read-cursor → ingest → advance-cursor section on the same store and
    double-ingest or corrupt the summary DAG. An exclusive per-(store, session) flock makes the
    later worker wait, then read the advanced cursor and process only genuinely-new messages.
    A no-op where fcntl is unavailable (single-invocation sync capture doesn't need it)."""
    if fcntl is None:
        yield
        return
    _CURSOR_DIR.mkdir(parents=True, exist_ok=True)
    lock = _lock_path(session_key, store_label)
    f = open(lock, "w")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        f.close()


def _async_enabled() -> bool:
    return _truthy(os.environ.get("LCM_CAPTURE_ASYNC"))


def _spawn_detached(payload: dict, final: bool) -> None:
    """Fork a detached worker that does the slow capture out of band, then return at once — so the
    hook never blocks session teardown (the cause of 'Hook cancelled' on the synchronous path).
    Re-invokes this module in --worker mode (self-contained; no dependency on the console script
    being on PATH), matching the SessionEnd-digest hook's pattern."""
    fd, tmp = tempfile.mkstemp(prefix="lcm_capture_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    argv = [sys.executable, os.path.abspath(__file__), "--worker", tmp]
    if final:
        argv.append("--final")
    devnull = open(os.devnull, "wb")
    subprocess.Popen(
        argv, stdin=subprocess.DEVNULL, stdout=devnull, stderr=devnull,
        start_new_session=True, close_fds=True,
    )


async def _capture(config, session_key: str, messages: list[dict], final: bool) -> None:
    """Drive the generic adapter's own lifecycle — the single, canonical capture path."""
    from lossless_agent.adapters.factory import create_adapter
    from lossless_agent.summarizers import build_summarize_fn

    summarize_fn = build_summarize_fn(config, os.environ.get("LCM_SUMMARIZE_COMMAND"))
    adapter = create_adapter("generic", config, summarize_fn)
    try:
        if messages:
            await adapter.on_turn_end(session_key, messages)
        if final:
            await adapter.on_session_end(session_key)
    finally:
        aclose = getattr(adapter, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:
                pass
        else:
            close = getattr(adapter, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass


def _do_capture(payload: dict, final: bool) -> int:
    """Read new messages past the cursor, ingest + compact them, then advance the cursor — all
    under an exclusive per-conversation lock so concurrent async workers can't race. Runs inline
    (sync mode) or inside the detached worker (async mode)."""
    session_key = payload.get("session_id") or payload.get("session_key") or "default"
    config = build_config(payload)
    label = _store_label(config)

    with _capture_lock(session_key, label):
        # Cursor dedup: the transcript grows every turn, so ingest only messages past the cursor.
        # Reading the cursor *inside* the lock is what makes async safe — a queued worker sees the
        # previous worker's advanced cursor and never reprocesses the same messages.
        messages: list[dict] = []
        cursor = 0
        transcript_path = payload.get("transcript_path")
        if transcript_path and os.path.exists(transcript_path):
            all_msgs = parse_transcript(transcript_path)
            cursor_file = _cursor_path(session_key, label)
            try:
                cursor = int(cursor_file.read_text().strip() or "0")
            except Exception:
                cursor = 0
            messages = all_msgs[cursor:]

        if not messages and not final:
            return 0
        if not prepare_store(config):
            return 0

        try:
            asyncio.run(_capture(config, session_key, messages, final))
        except Exception as e:
            _log(f"capture failed (non-fatal, cursor not advanced): {e}")
            return 0

        if messages:
            try:
                _CURSOR_DIR.mkdir(parents=True, exist_ok=True)
                _cursor_path(session_key, label).write_text(str(cursor + len(messages)))
            except Exception:
                pass
            _log(f"ingested {len(messages)} message(s) -> {label} (session {session_key[:8]})")
        if final:
            _log(f"final compaction sweep -> {label} (session {session_key[:8]})")
    return 0


def main(argv=None) -> int:
    # A summarizer's own headless session inherits this hook; bail so it isn't captured.
    if os.environ.get(_GUARD_ENV):
        return 0

    ap = argparse.ArgumentParser(prog="lossless-agent-capture", description=__doc__)
    ap.add_argument("--final", action="store_true",
                    help="run the end-of-session compaction sweep (register on SessionEnd)")
    ap.add_argument("--worker", metavar="PAYLOAD_JSON",
                    help=argparse.SUPPRESS)  # internal: run the detached async-capture worker
    args = ap.parse_args(argv)

    # Worker mode (async): payload was handed off via a temp file by _spawn_detached.
    if args.worker:
        try:
            with open(args.worker, encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return 0
        try:
            return _do_capture(payload if isinstance(payload, dict) else {}, args.final)
        finally:
            try:
                os.unlink(args.worker)
            except OSError:
                pass

    try:
        payload = json.load(sys.stdin) if not sys.stdin.isatty() else {}
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    # Async mode: fork a detached worker and return in milliseconds so the hook never blocks
    # (summarization can take up to LCM_SUMMARY_TIMEOUT_MS). Fall back to inline if the fork fails.
    if _async_enabled():
        try:
            _spawn_detached(payload, args.final)
            return 0
        except Exception as e:
            _log(f"async spawn failed ({e}); running capture inline")

    return _do_capture(payload, args.final)


def cli() -> None:
    sys.exit(main())


if __name__ == "__main__":
    cli()
