"""MCP server exposing LCM recall tools over stdio transport.

Provides both read-only recall tools (grep, describe, expand) and
write tools for the full conversation lifecycle (ingest, compact,
get_context, session_end).

Summarizer selection (first match wins):
  1. --summarize-command CLI flag  — pipe prompt to external command
  2. LCM_SUMMARY_PROVIDER=anthropic — call Claude via Anthropic SDK
       requires: ANTHROPIC_API_KEY, LCM_SUMMARY_MODEL (default: claude-haiku-4-5-20251001)
  3. deterministic truncation fallback (no LLM, lower quality)

Expansion model for lcm_expand_query:
  LCM_EXPANSION_MODEL (default: same as LCM_SUMMARY_MODEL)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import asdict
from typing import Any, List, Optional, cast

from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

from lossless_agent.store.factory import create_database
from lossless_agent.config import LCMConfig
from lossless_agent.store.conversation_store import ConversationStore
from lossless_agent.store.message_store import MessageStore
from lossless_agent.store.summary_store import SummaryStore
from lossless_agent.engine.compaction import CompactionEngine, SummarizeFn
from lossless_agent.engine.assembler import ContextAssembler, AssemblerConfig
from lossless_agent.tools.recall import (
    lcm_grep,
    lcm_describe,
    lcm_expand,
    GrepResult,
    _truncate,
)
from lossless_agent.tools.expand_query import (
    ExpansionOrchestrator,
)

from lossless_agent.store.vector_store import VectorStore
from lossless_agent.engine.embedder import (
    BatchEmbedFn,
    EmbedFn,
    make_raw_vector_batch_embedder,
    make_raw_vector_embedder,
)
from lossless_agent.engine.fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

server = Server("lossless-agent")
_db: Any = None  # Database | PostgresDatabase | None
_summarize_command: Optional[str] = None
_config: Optional[LCMConfig] = None
_vector_store: Optional[VectorStore] = None
_raw_embed_fn: Optional[EmbedFn] = None
_raw_batch_embed_fn: Optional[BatchEmbedFn] = None


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

def _make_anthropic_summarizer(model: str) -> SummarizeFn:
    """Summarizer that calls the Anthropic API directly.

    Requires ANTHROPIC_API_KEY in the environment and the ``anthropic``
    package (``pip install anthropic``).
    """
    _model = model or "claude-haiku-4-5-20251001"

    async def summarize(prompt: str) -> str:
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError(
                "LCM_SUMMARY_PROVIDER=anthropic requires the anthropic package. "
                "Install with: pip install anthropic"
            )
        client = _anthropic.AsyncAnthropic()
        message = await client.messages.create(
            model=_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text  # type: ignore[union-attr]

    return summarize


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
    """Return the configured summarize function (first match wins):
    1. --summarize-command CLI flag
    2. LCM_SUMMARY_PROVIDER=anthropic
    3. deterministic truncation fallback
    """
    if _summarize_command:
        return _make_command_summarizer(_summarize_command)
    if _config and _config.summary_provider == "anthropic":
        return _make_anthropic_summarizer(_config.summary_model)
    return _make_truncation_summarizer()


def _get_expansion_fn() -> SummarizeFn:
    """Return the summarize function to use for lcm_expand_query synthesis.

    Uses LCM_EXPANSION_MODEL if set, falls back to the summary model/provider.
    """
    if _summarize_command:
        return _make_command_summarizer(_summarize_command)
    if _config and _config.summary_provider == "anthropic":
        model = _config.expansion_model or _config.summary_model
        return _make_anthropic_summarizer(model)
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
        types.Tool(
            name="lcm_backfill",
            description=(
                "Embed existing messages that were stored before raw vector retrieval "
                "was enabled. Iterates all unembedded messages in the database and "
                "stores their vectors. Safe to re-run — skips already-embedded messages. "
                "Only available when raw_vector_enabled=True."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "batch_size": {
                        "type": "integer",
                        "default": 256,
                        "description": "Number of messages to embed per batch",
                    },
                    "conversation_id": {
                        "type": ["integer", "null"],
                        "default": None,
                        "description": "Limit backfill to a single conversation (optional)",
                    },
                },
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
        query = arguments["query"]
        scope = arguments.get("scope", "all")
        conversation_id = arguments.get("conversation_id")
        limit = arguments.get("limit", 20)

        # FTS5 keyword search (existing)
        fts_results = lcm_grep(
            _db,
            query=query,
            scope=scope,
            conversation_id=conversation_id,
            limit=limit,
        )

        # Vector search (if raw vector retrieval is enabled)
        if _vector_store and _raw_embed_fn and scope in ("all", "messages"):
            try:
                query_embedding = await _raw_embed_fn(query)
                conv_ids = [conversation_id] if conversation_id else None
                vec_hits = _vector_store.search_messages(
                    query_embedding,
                    top_k=_config.raw_vector_top_k if _config else 20,
                    conversation_ids=conv_ids,
                    min_score=_config.raw_vector_min_score if _config else 0.35,
                )

                if vec_hits:
                    flat_fts = cast(List[GrepResult], fts_results)
                    # Build FTS ranked list: (id, rank_score)
                    fts_ranked = [(str(r.id), 1.0 / (i + 1)) for i, r in enumerate(flat_fts)]
                    # Vec ranked list: (id, similarity)
                    vec_ranked = [(mid, sim) for mid, sim in vec_hits]

                    # Merge with RRF
                    merged = reciprocal_rank_fusion(fts_ranked, vec_ranked, k=60)
                    merged_ids = [doc_id for doc_id, _ in merged[:limit]]

                    # Build a lookup of existing FTS results by ID
                    fts_by_id = {str(r.id): r for r in flat_fts}

                    # Fetch message content for vector-only hits
                    vec_only_ids = [mid for mid in merged_ids if mid not in fts_by_id]
                    vec_messages = {}
                    if vec_only_ids:
                        for mid in vec_only_ids:
                            # message_id stored as str in pgvector; SQLite id is int
                            row = _db.conn.execute(
                                "SELECT id, conversation_id, content, role, seq, created_at "
                                "FROM messages WHERE id = ?",
                                (int(mid),),
                            ).fetchone()
                            if row:
                                vec_messages[mid] = GrepResult(
                                    type="message",
                                    id=row[0],
                                    content_snippet=_truncate(row[2]),
                                    conversation_id=row[1],
                                    metadata={"role": row[3], "seq": row[4], "source": "vector"},
                                    created_at=row[5],
                                )

                    # Reorder results by RRF ranking
                    reordered = []
                    for doc_id in merged_ids:
                        if doc_id in fts_by_id:
                            reordered.append(fts_by_id[doc_id])
                        elif doc_id in vec_messages:
                            reordered.append(vec_messages[doc_id])
                    fts_results = reordered
            except Exception as exc:
                logger.warning("Vector search failed, using FTS only: %s", exc)

        payload = [_serialize(r) for r in fts_results]  # type: ignore[union-attr]
        return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]

    elif name == "lcm_describe":
        result = lcm_describe(_db, summary_id=arguments["summary_id"])
        if result is None:
            return [types.TextContent(type="text", text=json.dumps({"error": "summary not found"}))]
        return [types.TextContent(type="text", text=json.dumps(_serialize(result), indent=2))]

    elif name == "lcm_expand":
        result = lcm_expand(_db, summary_id=arguments["summary_id"], is_sub_agent=True) # type: ignore[assignment]
        if result is None:
            return [types.TextContent(type="text", text=json.dumps({"error": "summary not found"}))]
        return [types.TextContent(type="text", text=json.dumps(_serialize(result), indent=2))]

    elif name == "lcm_expand_query":
        msg_store = MessageStore(_db)
        sum_store = SummaryStore(_db)
        orch = ExpansionOrchestrator(
            db=_db,
            msg_store=msg_store,
            sum_store=sum_store,
            expand_fn=_get_expansion_fn(),
        )
        result = await orch.expand_query( # type: ignore[assignment]
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

    elif name == "lcm_backfill":
        return await _handle_backfill(arguments)

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

    # Embed messages at ingestion time (raw vector retrieval)
    embedded_count = 0
    if _vector_store and _raw_batch_embed_fn:
        try:
            texts = []
            msg_ids = []
            for msg_data, ing in zip(raw_messages, ingested):
                content = msg_data.get("content", "")
                if content and content.strip():
                    texts.append(content)
                    msg_ids.append(ing["id"])

            if texts:
                embeddings = await _raw_batch_embed_fn(texts)
                items = [
                    (str(mid), conv.id, emb)
                    for mid, emb in zip(msg_ids, embeddings)
                ]
                _vector_store.store_messages_batch(items)
                embedded_count = len(items)
        except Exception as exc:
            logger.warning("Failed to embed messages at ingestion: %s", exc)

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
        "messages_embedded": embedded_count,
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


async def _handle_backfill(arguments: dict) -> list[types.TextContent]:
    """Embed all messages that don't yet have a vector — idempotent."""
    assert _db is not None

    if not _vector_store or not _raw_batch_embed_fn:
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": "raw vector retrieval not enabled or no vector store configured"}),
        )]

    batch_size = int(arguments.get("batch_size") or 256)
    filter_conv = arguments.get("conversation_id")

    conn = _db.conn
    if filter_conv is not None:
        rows = conn.execute(
            "SELECT id, content FROM messages WHERE conversation_id = ? ORDER BY id",
            (filter_conv,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, content FROM messages ORDER BY id"
        ).fetchall()

    # Fetch already-embedded message IDs from pgvector
    existing_ids: set[str] = set()
    try:
        pg_conn = _vector_store._get_conn()
        cur = pg_conn.cursor()
        cur.execute("SELECT message_id FROM message_embeddings")
        existing_ids = {row[0] for row in cur.fetchall()}
        cur.close()
    except Exception as exc:
        logger.warning("Could not fetch existing embeddings for backfill check: %s", exc)

    # Filter to unembedded messages with non-empty content
    todo = [
        (row[0], row[1]) for row in rows
        if row[1] and row[1].strip() and str(row[0]) not in existing_ids
    ]

    if not todo:
        return [types.TextContent(
            type="text",
            text=json.dumps({"status": "ok", "embedded": 0, "skipped": len(rows)}),
        )]

    # Get conversation_id for each message
    msg_conv = {}
    if filter_conv is not None:
        msg_conv = {row[0]: filter_conv for row in rows}
    else:
        conv_rows = conn.execute("SELECT id, conversation_id FROM messages").fetchall()
        msg_conv = {r[0]: r[1] for r in conv_rows}

    embedded = 0
    failed = 0
    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        ids = [r[0] for r in batch]
        texts = [r[1] for r in batch]
        try:
            embeddings = await _raw_batch_embed_fn(texts)
            items = [
                (str(mid), msg_conv.get(mid, 0), emb)
                for mid, emb in zip(ids, embeddings)
            ]
            _vector_store.store_messages_batch(items)
            embedded += len(items)
        except Exception as exc:
            logger.warning("Backfill batch %d failed: %s", i // batch_size, exc)
            failed += len(batch)

    return [types.TextContent(
        type="text",
        text=json.dumps({
            "status": "ok",
            "embedded": embedded,
            "failed": failed,
            "skipped": len(rows) - len(todo),
        }),
    )]


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

async def main(db_path: str, db_dsn: str = "") -> None:
    global _db, _config, _vector_store, _raw_embed_fn, _raw_batch_embed_fn
    # Start from env vars, then apply CLI overrides on top
    cfg = LCMConfig.from_env()
    overrides: dict = {}
    if db_path:
        overrides["db_path"] = db_path
    if db_dsn:
        overrides["database_dsn"] = db_dsn
    if overrides:
        cfg = LCMConfig.merge(cfg, overrides)
    _config = cfg
    _db = create_database(cfg)

    # Initialize raw vector retrieval if configured
    if cfg.raw_vector_enabled and cfg.database_dsn:
        try:
            _vector_store = VectorStore(
                dsn=cfg.database_dsn,
                dim=cfg.embedding_dim,
                msg_dim=cfg.raw_vector_dim,
            )
            _raw_embed_fn = make_raw_vector_embedder(cfg)
            _raw_batch_embed_fn = make_raw_vector_batch_embedder(cfg)
            import logging
            logging.getLogger(__name__).info(
                "Raw vector retrieval enabled (model=%s, dim=%d)",
                cfg.raw_vector_model,
                cfg.raw_vector_dim,
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Failed to initialize raw vector retrieval: %s", exc
            )
            _vector_store = None
            _raw_embed_fn = None
            _raw_batch_embed_fn = None

    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        if _vector_store:
            _vector_store.close()
        _db.close()


def cli() -> None:
    parser = argparse.ArgumentParser(description="LCM MCP Server (stdio transport)")
    parser.add_argument(
        "--db-path",
        default="~/.lossless-agent/lcm.db",
        help="Path to the lossless-agent SQLite database (ignored when --db-dsn is set)",
    )
    parser.add_argument(
        "--db-dsn",
        default="",
        help=(
            "PostgreSQL DSN, e.g. postgresql://user:pass@host/dbname. "
            "When set, Postgres is used instead of SQLite. "
            "Requires: pip install 'lossless-agent[postgres]'"
        ),
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
    asyncio.run(main(args.db_path, db_dsn=args.db_dsn))


if __name__ == "__main__":
    cli()
