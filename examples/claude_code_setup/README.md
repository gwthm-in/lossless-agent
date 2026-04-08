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

## Full Lifecycle Setup

For automatic context management, add instructions to your `CLAUDE.md`:

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
