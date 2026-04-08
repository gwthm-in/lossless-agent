"""MCP server exposing LCM recall tools over stdio transport."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

from lossless_agent.store.database import Database
from lossless_agent.store.message_store import MessageStore
from lossless_agent.store.summary_store import SummaryStore
from lossless_agent.tools.recall import (
    lcm_grep,
    lcm_describe,
    lcm_expand,
    GrepResult,
    DescribeResult,
    ExpandResult,
)
from lossless_agent.tools.expand_query import (
    ExpansionOrchestrator,
    ExpandQueryConfig,
    ExpandQueryResult,
)
from lossless_agent.store.models import Message, Summary

server = Server("lossless-agent")
_db: Database | None = None


def _serialize(obj: Any) -> Any:
    """Recursively convert dataclass instances to dicts for JSON serialization."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [_serialize(item) for item in obj]
    return obj


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
    ]


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

    else:
        return [types.TextContent(type="text", text=json.dumps({"error": f"unknown tool: {name}"}))]


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
    args = parser.parse_args()
    asyncio.run(main(args.db_path))


if __name__ == "__main__":
    cli()
