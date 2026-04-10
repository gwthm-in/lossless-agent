"""MCP server exposing LCM recall tools over stdio transport.

Provides both read-only recall tools (grep, describe, expand) and
write tools for the full conversation lifecycle (ingest, compact,
get_context, session_end).

The write tools use a deterministic truncation summarizer by default.
For higher-quality summaries, pass --summarize-command to pipe text
through an external LLM-backed command.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from typing import Any, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

from lossless_agent.store.database import Database
from lossless_agent.store.conversation_store import ConversationStore
from lossless_agent.store.message_store import MessageStore
from lossless_agent.store.summary_store import SummaryStore
from lossless_agent.engine.compaction import CompactionEngine, SummarizeFn
from lossless_agent.engine.assembler import ContextAssembler, AssemblerConfig
from lossless_agent.tools.recall import (
    lcm_grep,
    lcm_describe,
    lcm_expand,
)
from lossless_agent.tools.expand_query import (
    ExpansionOrchestrator,
)

server = Server("lossless-agent")
_db: Database | None = None
_summarize_command: Optional[str] = None


def _serialize(obj: Any) -> Any:
    """Recursively convert dataclass instances to dicts for JSON serialization."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [_serialize(item) for item in obj]
    return obj


# ------------------------------------------------------------------
# Summarizer factory
# ------------------------------------------------------------------

def _make_truncation_summarizer() -> SummarizeFn:
    """Built-in deterministic summarizer: truncates to target length.

    This is a fallback — it preserves the DAG structure so original
    messages are still recoverable via lcm_expand, even though the
    summary text itself is truncated rather than intelligently compressed.

    For production use, pass --summarize-command to use an external
    LLM-backed summarizer.
    """
    async def summarize(prompt: str) -> str:
        # Extract target token count from the prompt if present,
        # otherwise use a reasonable default
        target_tokens = 1200  # default leaf target
        # The prompt from build_leaf_prompt / build_condensed_prompt
        # contains the text to summarize. We just truncate it.
        char_limit = target_tokens * 4
        if len(prompt) <= char_limit:
            return prompt
        return prompt[:char_limit] + "\n[Summary truncated — configure --summarize-command for LLM-quality summaries]"
    return summarize


def _make_command_summarizer(command: str) -> SummarizeFn:
    """Summarizer that pipes the prompt to an external command via stdin.

    The command should read the prompt from stdin and write the summary
    to stdout. Example: --summarize-command 'python my_summarizer.py'
    """
    async def summarize(prompt: str) -> str:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(prompt.encode("utf-8"))
        if proc.returncode != 0:
            raise RuntimeError(
                f"Summarize command failed (exit {proc.returncode}): {stderr.decode()}"
            )
        return stdout.decode("utf-8").strip()
    return summarize


def _get_summarize_fn() -> SummarizeFn:
    """Return the configured summarize function."""
    if _summarize_command:
        return _make_command_summarizer(_summarize_command)
    return _make_truncation_summarizer()


# ------------------------------------------------------------------
# Tool listing
# ------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="lcm_grep",
            description=(
                "Full-text search across messages and summaries in the LCM database. "
                "Returns matching snippets with metadata."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "FTS5 search query string",
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["all", "messages", "summaries"],
                        "default": "all",
                        "description": "Search scope: all, messages, or summaries",
                    },
                    "conversation_id": {
                        "type": ["integer", "null"],
                        "default": None,
                        "description": "Optional conversation ID to restrict search",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum results to return",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="lcm_describe",
            description=(
                "Get full metadata for a summary node by its ID. "
                "Returns kind, depth, content, token counts, time range, child IDs."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "summary_id": {
                        "type": "string",
                        "description": "The summary ID (e.g. sum_xxxxxxxxxxxx)",
                    },
                },
                "required": ["summary_id"],
            },
        ),
        types.Tool(
            name="lcm_expand",
            description=(
                "Expand a summary node: returns source messages for leaf summaries, "
                "or child summaries for condensed summaries."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "summary_id": {
                        "type": "string",
                        "description": "The summary ID to expand",
                    },
                },
                "required": ["summary_id"],
            },
        ),
        types.Tool(
            name="lcm_stats",
            description=(
                "Get database statistics: message counts, summary counts by depth, "
                "total token counts, conversation count."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        types.Tool(
            name="lcm_expand_query",
            description=(
                "Run a sub-agent expansion query: searches the DAG, describes and "
                "expands relevant summaries, then synthesizes an answer."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "conversation_id": {
                        "type": "integer",
                        "description": "Conversation ID to search within",
                    },
                    "query": {
                        "type": "string",
                        "description": "The question to answer from past context",
                    },
                },
                "required": ["conversation_id", "query"],
            },
        ),
        # ── Write tools for full conversation lifecycle ──
        types.Tool(
            name="lcm_ingest",
            description=(
                "Store messages into the LCM database. Persists messages and runs "
                "incremental compaction automatically when the conversation exceeds "
                "the soft threshold. Call this after each agent turn."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_key": {
                        "type": "string",
                        "description": "Session identifier (creates conversation if new)",
                    },
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {
                                    "type": "string",
                                    "description": "Message role: user, assistant, system, or tool",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "Message content text",
                                },
                                "token_count": {
                                    "type": "integer",
                                    "description": "Optional token count (estimated from chars if omitted)",
                                },
                            },
                            "required": ["role", "content"],
                        },
                        "description": "Array of messages to ingest",
                    },
                },
                "required": ["session_key", "messages"],
            },
        ),
        types.Tool(
            name="lcm_compact",
            description=(
                "Force compaction for a session. Runs a full compaction sweep "
                "(leaf + condensed passes). Use when you want to explicitly "
                "compress history rather than waiting for automatic compaction."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_key": {
                        "type": "string",
                        "description": "Session identifier",
                    },
                },
                "required": ["session_key"],
            },
        ),
        types.Tool(
            name="lcm_get_context",
            description=(
                "Assemble optimized context within a token budget. Returns "
                "summaries of older conversation + recent tail messages, "
                "formatted for LLM consumption. Call this before each agent turn "
                "to get the best context window."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_key": {
                        "type": "string",
                        "description": "Session identifier",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "default": 100000,
                        "description": "Maximum token budget for assembled context",
                    },
                },
                "required": ["session_key"],
            },
        ),
        types.Tool(
            name="lcm_session_end",
            description=(
                "Signal session end for final compaction. Runs a full compaction "
                "sweep to compress the entire session. Call when the conversation "
                "is ending."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_key": {
                        "type": "string",
                        "description": "Session identifier",
                    },
                },
                "required": ["session_key"],
            },
        ),
    ]


# ------------------------------------------------------------------
# Tool dispatch
# ------------------------------------------------------------------

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    assert _db is not None, "Database not initialized"

    if name == "lcm_grep":
        results = lcm_grep(
            _db,
            query=arguments["query"],
            scope=arguments.get("scope", "all"),
            conversation_id=arguments.get("conversation_id"),
            limit=arguments.get("limit", 20),
        )
        payload = [_serialize(r) for r in results]
        return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]

    elif name == "lcm_describe":
        result = lcm_describe(_db, summary_id=arguments["summary_id"])
        if result is None:
            return [types.TextContent(type="text", text=json.dumps({"error": "summary not found"}))]
        return [types.TextContent(type="text", text=json.dumps(_serialize(result), indent=2))]

    elif name == "lcm_expand":
        result = lcm_expand(_db, summary_id=arguments["summary_id"], is_sub_agent=True)
        if result is None:
            return [types.TextContent(type="text", text=json.dumps({"error": "summary not found"}))]
        return [types.TextContent(type="text", text=json.dumps(_serialize(result), indent=2))]

    elif name == "lcm_expand_query":
        async def _passthrough_summarize(prompt: str) -> str:
            return prompt

        msg_store = MessageStore(_db)
        sum_store = SummaryStore(_db)
        orch = ExpansionOrchestrator(
            db=_db,
            msg_store=msg_store,
            sum_store=sum_store,
            expand_fn=_passthrough_summarize,
        )
        result = await orch.expand_query(
            conversation_id=arguments["conversation_id"],
            query=arguments["query"],
        )
        payload = _serialize(result)
        return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]

    elif name == "lcm_stats":
        conn = _db.conn
        stats: dict[str, Any] = {}

        # Conversation count
        row = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()
        stats["conversations"] = row[0]

        # Message counts and tokens
        row = conn.execute("SELECT COUNT(*), COALESCE(SUM(token_count), 0) FROM messages").fetchone()
        stats["messages"] = {"count": row[0], "total_tokens": row[1]}

        # Summary counts by depth
        rows = conn.execute(
            "SELECT depth, kind, COUNT(*), COALESCE(SUM(token_count), 0) "
            "FROM summaries GROUP BY depth, kind ORDER BY depth, kind"
        ).fetchall()
        summary_breakdown = []
        for r in rows:
            summary_breakdown.append({
                "depth": r[0],
                "kind": r[1],
                "count": r[2],
                "total_tokens": r[3],
            })
        row = conn.execute("SELECT COUNT(*), COALESCE(SUM(token_count), 0) FROM summaries").fetchone()
        stats["summaries"] = {
            "count": row[0],
            "total_tokens": row[1],
            "by_depth_kind": summary_breakdown,
        }

        return [types.TextContent(type="text", text=json.dumps(stats, indent=2))]

    # ── Write tools ──

    elif name == "lcm_ingest":
        return await _handle_ingest(arguments)

    elif name == "lcm_compact":
        return await _handle_compact(arguments)

    elif name == "lcm_get_context":
        return await _handle_get_context(arguments)

    elif name == "lcm_session_end":
        return await _handle_session_end(arguments)

    else:
        return [types.TextContent(type="text", text=json.dumps({"error": f"unknown tool: {name}"}))]


# ------------------------------------------------------------------
# Write tool handlers
# ------------------------------------------------------------------

async def _handle_ingest(arguments: dict) -> list[types.TextContent]:
    """Store messages and run incremental compaction if needed."""
    assert _db is not None
    session_key = arguments["session_key"]
    raw_messages = arguments["messages"]

    conv_store = ConversationStore(_db)
    msg_store = MessageStore(_db)

    conv = conv_store.get_or_create(session_key)

    ingested = []
    for msg_data in raw_messages:
        role = msg_data["role"]
        content = msg_data["content"]
        # Estimate tokens as ~chars/4 if not provided
        token_count = msg_data.get("token_count") or max(1, len(content) // 4)

        m = msg_store.append(
            conversation_id=conv.id,
            role=role,
            content=content,
            token_count=token_count,
        )
        ingested.append({"id": m.id, "seq": m.seq, "role": m.role})

    # Run incremental compaction if needed
    sum_store = SummaryStore(_db)
    summarize_fn = _get_summarize_fn()
    engine = CompactionEngine(msg_store, sum_store, summarize_fn)

    compacted_count = 0
    # Use a reasonable context limit for auto-compaction check
    context_limit = 100_000
    if engine.needs_compaction(conv.id, context_limit):
        created = await engine.compact_full_sweep(conv.id)
        compacted_count = len(created)

    result = {
        "status": "ok",
        "conversation_id": conv.id,
        "messages_ingested": len(ingested),
        "messages": ingested,
        "summaries_created": compacted_count,
    }
    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


async def _handle_compact(arguments: dict) -> list[types.TextContent]:
    """Force a full compaction sweep."""
    assert _db is not None
    session_key = arguments["session_key"]

    conv_store = ConversationStore(_db)
    msg_store = MessageStore(_db)
    sum_store = SummaryStore(_db)

    conv = conv_store.get_or_create(session_key)
    summarize_fn = _get_summarize_fn()
    engine = CompactionEngine(msg_store, sum_store, summarize_fn)

    created = await engine.compact_full_sweep(conv.id)

    result = {
        "status": "ok",
        "conversation_id": conv.id,
        "summaries_created": len(created),
        "summary_ids": [s.summary_id for s in created],
    }
    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


async def _handle_get_context(arguments: dict) -> list[types.TextContent]:
    """Assemble optimized context within token budget."""
    assert _db is not None
    session_key = arguments["session_key"]
    max_tokens = arguments.get("max_tokens", 100_000)

    conv_store = ConversationStore(_db)
    msg_store = MessageStore(_db)
    sum_store = SummaryStore(_db)

    conv = conv_store.get_or_create(session_key)

    assembler = ContextAssembler(
        msg_store=msg_store,
        sum_store=sum_store,
        config=AssemblerConfig(max_context_tokens=max_tokens),
    )

    assembled = assembler.assemble(conv.id)
    context_text = assembler.format_context(assembled)

    result = {
        "status": "ok",
        "conversation_id": conv.id,
        "total_tokens": assembled.total_tokens,
        "summary_count": len(assembled.summaries),
        "message_count": len(assembled.messages),
        "context": context_text,
    }
    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


async def _handle_session_end(arguments: dict) -> list[types.TextContent]:
    """Signal session end: run final compaction sweep."""
    assert _db is not None
    session_key = arguments["session_key"]

    conv_store = ConversationStore(_db)
    msg_store = MessageStore(_db)
    sum_store = SummaryStore(_db)

    conv = conv_store.get_or_create(session_key)
    summarize_fn = _get_summarize_fn()
    engine = CompactionEngine(msg_store, sum_store, summarize_fn)

    created = await engine.compact_full_sweep(conv.id)

    # Mark conversation as inactive
    _db.conn.execute(
        "UPDATE conversations SET active = 0 WHERE id = ?",
        (conv.id,),
    )
    _db.conn.commit()

    result = {
        "status": "ok",
        "conversation_id": conv.id,
        "summaries_created": len(created),
        "session_closed": True,
    }
    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

async def main(db_path: str) -> None:
    global _db
    _db = Database(db_path)
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        _db.close()


def cli() -> None:
    parser = argparse.ArgumentParser(description="LCM MCP Server (stdio transport)")
    parser.add_argument(
        "--db-path",
        required=True,
        help="Path to the lossless-agent SQLite database",
    )
    parser.add_argument(
        "--summarize-command",
        default=None,
        help=(
            "External command for summarization. The prompt is piped via stdin, "
            "summary is read from stdout. Example: 'python my_summarizer.py'. "
            "If not set, uses deterministic truncation (lower quality but functional)."
        ),
    )
    args = parser.parse_args()
    global _summarize_command
    _summarize_command = args.summarize_command
    asyncio.run(main(args.db_path))


if __name__ == "__main__":
    cli()
