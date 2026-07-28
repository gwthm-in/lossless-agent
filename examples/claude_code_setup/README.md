# Claude Code Setup (2 Steps)

Give Claude Code lossless context management in under 2 minutes.

## Step 1: Install

```bash
pip install lossless-agent
```

## Step 2: Copy `.mcp.json` to your project root

```bash
cp .mcp.json /path/to/your/project/.mcp.json
```

Or create `.mcp.json` in your project root with:

```json
{
  "mcpServers": {
    "lossless-agent": {
      "command": "lossless-agent-mcp",
      "args": [
        "--db-path",
        "./data/lcm.db"
      ]
    }
  }
}
```

That's it. Claude Code automatically discovers `.mcp.json` and connects to
the MCP server.

## Available Tools

### Recall (read-only)
- `lcm_grep` — Full-text search across messages and summaries
- `lcm_describe` — Get metadata for a summary node
- `lcm_expand` — Drill into a summary's source content
- `lcm_expand_query` — AI-powered contextual search
- `lcm_stats` — Database statistics

### Lifecycle (read-write)
- `lcm_ingest` — Store messages into the database (auto-compacts)
- `lcm_compact` — Force compaction sweep
- `lcm_get_context` — Assemble optimized context within token budget
- `lcm_session_end` — Signal session end for final compaction

## Automatic capture (hooks) — zero-code, recommended

Claude Code has two extension points: **MCP tools** (recall, above) and **hooks** (events).
The `lossless-agent-capture` console script is the hook half — register it and lossless
captures **every turn automatically** into a per-project store and builds the summary DAG.
No `CLAUDE.md` instructions, no reliance on the model remembering to call `lcm_ingest`.

Add to `~/.claude/settings.json` (or a project `.claude/settings.json`):

```json
{
  "hooks": {
    "Stop":       [{ "hooks": [{ "type": "command", "command": "lossless-agent-capture" }] }],
    "SessionEnd": [{ "hooks": [{ "type": "command", "command": "lossless-agent-capture --final" }] }]
  }
}
```

- **Stop** (after each assistant turn) → ingest new messages + incremental compaction.
- **SessionEnd** `--final` → final leaf/condensed compaction sweep.

Everything is configured via **environment variables** — no code:

| Env var | Purpose |
|---|---|
| *(none)* | **Default:** a per-project **SQLite** store at `~/.lossless-agent/stores/lcm_<basename>_<hash>.db` — zero dependencies, no server. Works right after `pip install`. |
| `LCM_DATABASE_DSN` | Opt into **Postgres** (unlocks the pgvector semantic layer): a full DSN, DB auto-created if missing. Requires `pip install 'lossless-agent[postgres]'` + a running Postgres. |
| `LCM_DATABASE_PATH` | Explicit SQLite file path (shared store, not per-project). |
| `LCM_SUMMARY_PROVIDER=anthropic` + `ANTHROPIC_API_KEY` | **Recommended** summarizer — a direct API call (fast, no CLI cold-start). Needs `pip install 'lossless-agent[anthropic]'`. Also `openai` (OpenAI/LiteLLM/Azure/Groq) via `lossless-agent[openai]`. Unset → deterministic truncation fallback. |
| `LCM_SUMMARY_MODEL` | e.g. `claude-haiku-4-5-20251001`. |
| `LCM_SUMMARIZE_COMMAND` | Alternative: an external `stdin → stdout` summarizer command. |
| `LCM_LEAF_CHUNK_TOKENS`, `LCM_SUMMARY_TIMEOUT_MS`, … | Compaction tuning (honoured on the capture + ingest paths). |

Point the MCP server (recall) at the **same** store so reads and writes share it — set the same
`LCM_DATABASE_PATH` / `LCM_DATABASE_DSN` for both, or give `lossless-agent-mcp` the matching
`--db-path` (SQLite) or `--db-dsn` (Postgres).

This is the seamless path: `pip install lossless-agent` → register the hook → set env. The
capture runs through the generic adapter's own `on_turn_end` / `on_session_end` lifecycle, so
compaction, the semantic layer, and configuration are all the library's.

## Full Lifecycle Setup (prompt-driven alternative)

If you prefer the model to drive capture via tool calls instead of hooks, add instructions to
your `CLAUDE.md`:

```bash
python -m examples.claude_code_middleware my-project > CLAUDE.md
```

This tells Claude Code to:
1. Call `lcm_get_context` at session start to load prior context
2. Call `lcm_ingest` after each turn to persist messages
3. Call `lcm_session_end` when the session ends

See [`../claude_code_middleware.py`](../claude_code_middleware.py) for details.

## Customization

**Change the database path** — edit `--db-path` to any location:

```json
"args": ["--db-path", "~/.lcm/my-project.db"]
```

**Use LLM-quality summaries** — add `--summarize-command`:

```json
"args": ["--db-path", "./data/lcm.db", "--summarize-command", "python my_summarizer.py"]
```

**Per-project databases** — each project gets its own `.mcp.json` with a
separate `--db-path`, so context stays isolated.

## Verify

Start Claude Code in your project directory. Ask:

> "What LCM tools do you have?"

You should see all 9 tools listed in the response.
