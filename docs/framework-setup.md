# Framework Integration Guide

Set up Lossless Agent with your AI framework in under 2 minutes.

Lossless Agent supports **two integration modes**:

1. **MCP Server** — framework connects to `lossless-agent-mcp` over stdio.
   Best for: Claude Code, Google ADK, Anthropic SDK, Cursor, Windsurf, or
   any MCP-compatible client.
2. **Python Adapter** — import and use directly in your Python code.
   Best for: OpenClaw, custom agents, or when you need programmatic control.

---

## Conversation Lifecycle

The LCM MCP server provides **9 tools** covering the full conversation lifecycle:

### Read-Only Tools (Recall)
| Tool | Description |
|------|-------------|
| `lcm_grep` | Full-text search across messages and summaries |
| `lcm_describe` | Get metadata for a summary node by ID |
| `lcm_expand` | Expand a summary to see source messages/children |
| `lcm_expand_query` | AI-powered search with automatic expansion |
| `lcm_stats` | Database statistics (message/summary counts) |

### Read-Write Tools (Lifecycle)
| Tool | Description |
|------|-------------|
| `lcm_ingest` | Store messages + auto-compact when threshold exceeded |
| `lcm_compact` | Force a full compaction sweep |
| `lcm_get_context` | Assemble optimized context within token budget |
| `lcm_session_end` | Signal session end for final compaction |

### Full Loop Pattern

Every framework integration should follow this pattern:

```
┌─────────────────────────────────────────────┐
│                SESSION START                │
│  lcm_get_context(session_key, max_tokens)   │
│  → Returns: summaries + recent messages     │
└──────────────────┬──────────────────────────┘
                   │
          ┌────────▼────────┐
          │   AGENT TURN    │◄──────────────┐
          │  (user + reply) │               │
          └────────┬────────┘               │
                   │                        │
     ┌─────────────▼─────────────┐          │
     │  lcm_ingest(session_key,  │          │
     │    messages=[...])        │          │
     │  → Auto-compacts if needed│          │
     └─────────────┬─────────────┘          │
                   │                        │
          ┌────────▼────────┐    yes        │
          │  More turns?    │───────────────┘
          └────────┬────────┘
                   │ no
     ┌─────────────▼─────────────┐
     │  lcm_session_end(         │
     │    session_key)           │
     │  → Final compaction       │
     └───────────────────────────┘
```

### Summarization

The MCP server uses deterministic truncation by default for compaction
summaries. This preserves the DAG structure — original messages are always
recoverable via `lcm_expand`, even if summary text is truncated.

For LLM-quality summaries, use `--summarize-command`:

```bash
lossless-agent-mcp --db-path ./data/lcm.db --summarize-command 'python my_summarizer.py'
```

The command receives the summarization prompt on stdin and should write
the summary to stdout.

---

## 1. Claude Code

**Mode:** MCP Server (zero code)

### Step 1: Install

```bash
pip install lossless-agent
```

### Step 2: Add `.mcp.json` to your project root

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

Claude Code auto-discovers `.mcp.json` and connects. Your agent now has
all 9 LCM tools: `lcm_grep`, `lcm_describe`, `lcm_expand`, `lcm_stats`,
`lcm_expand_query`, `lcm_ingest`, `lcm_compact`, `lcm_get_context`, and
`lcm_session_end`.

### Step 3 (optional): Add lifecycle instructions to CLAUDE.md

```bash
python examples/claude_code_middleware.py my-project > CLAUDE.md
```

This generates instructions that tell Claude Code to automatically call
`lcm_get_context` at session start, `lcm_ingest` after each turn, and
`lcm_session_end` when done.

**Verify:** Ask Claude Code _"What LCM tools do you have?"_

See: [`examples/claude_code_setup/`](../examples/claude_code_setup/)
See: [`examples/claude_code_middleware.py`](../examples/claude_code_middleware.py)

---

## 2. Cursor / Windsurf / Any MCP Client

**Mode:** MCP Server

Same as Claude Code — these editors support MCP via `.mcp.json` or their
settings UI. Add this server config:

- **Command:** `lossless-agent-mcp`
- **Args:** `["--db-path", "./data/lcm.db"]`

For Cursor, add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "lossless-agent": {
      "command": "lossless-agent-mcp",
      "args": ["--db-path", "./data/lcm.db"]
    }
  }
}
```

---

## 3. Google ADK (Agent Development Kit)

**Mode:** MCP Server via `McpToolset` + lifecycle callbacks

### Install

```bash
pip install lossless-agent google-adk
```

### Setup

```python
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
from examples.google_adk_middleware import LCMLifecycleCallbacks

# Connect to LCM MCP server
mcp_tools, cleanup = await McpToolset.from_server(
    connection_params=StdioConnectionParams(
        command="lossless-agent-mcp",
        args=["--db-path", "./data/lcm.db"],
    )
)

# Set up lifecycle callbacks
lcm = LCMLifecycleCallbacks(session_key="my-adk-project")

agent = Agent(
    model="gemini-2.0-flash",
    name="my_agent",
    instruction=lcm.get_instruction(),
    tools=mcp_tools,
    before_agent_callback=lcm.before_agent,
    after_agent_callback=lcm.after_agent,
)

# ... run your agent ...

await lcm.on_session_end(runner)
await cleanup()
```

See: [`examples/google_adk_middleware.py`](../examples/google_adk_middleware.py)
See: [`examples/google_adk_setup.py`](../examples/google_adk_setup.py)

---

## 4. Anthropic SDK (Client-Side)

**Mode:** Python middleware wrapping Anthropic API calls

### Install

```bash
pip install lossless-agent anthropic
```

### Setup

```python
import anthropic
from examples.anthropic_sdk_middleware import LCMMiddleware

client = anthropic.Anthropic()
lcm = LCMMiddleware(
    session_key="my-project",
    db_path="./data/lcm.db",
)

# Chat with automatic context loading + ingestion
response = lcm.chat(client, "What were we discussing last time?")
print(response)

# End session
lcm.end_session()
```

The middleware automatically:
1. Calls `lcm_get_context` before each API call
2. Calls `lcm_ingest` after each API call
3. Calls `lcm_session_end` when you're done

See: [`examples/anthropic_sdk_middleware.py`](../examples/anthropic_sdk_middleware.py)
See: [`examples/anthropic_agents_setup.py`](../examples/anthropic_agents_setup.py)

---

## 5. OpenClaw (Python Adapter)

**Mode:** Direct Python integration

### Install

```bash
pip install lossless-agent
```

### Setup

```python
from lossless_agent import OpenClawAdapter, LCMConfig

config = LCMConfig(db_path="./data/lcm.db", max_context_tokens=8000)
adapter = OpenClawAdapter(config, your_summarize_fn)

# Get tool schemas (include openclaw_metadata)
tools = adapter.get_tools()

# Get system prompt block
system_prompt = adapter.get_system_prompt_block()

# Lifecycle
context = await adapter.on_turn_start(session_id, user_message)
# ... agent processes turn ...
await adapter.on_turn_end(session_id, messages)

# Handle tool calls
result = await adapter.handle_tool_call("lcm_grep", {"query": "deployment"})

# End session
await adapter.on_session_end(session_id)
```

See: [`examples/openclaw_setup.py`](../examples/openclaw_setup.py)

---

## 6. Custom / Generic Agent

**Mode:** Direct Python integration

For any Python agent framework, use `GenericAdapter`:

```python
from lossless_agent import GenericAdapter, LCMConfig

config = LCMConfig(db_path="./data/lcm.db", max_context_tokens=8000)
adapter = GenericAdapter(config, your_summarize_fn)

# Same lifecycle as OpenClaw:
tools = adapter.get_tools()
system_prompt = adapter.get_system_prompt_block()
context = await adapter.on_turn_start(session_id, user_message)
await adapter.on_turn_end(session_id, messages)
result = await adapter.handle_tool_call("lcm_grep", {"query": "search term"})
await adapter.on_session_end(session_id)
```

See: [`examples/framework_integration.py`](../examples/framework_integration.py)

---

## Database Path

All integrations use `--db-path` (MCP) or `db_path` (Python) to set the
SQLite database location:

- **Per-project:** `./data/lcm.db` (relative to project root)
- **Global:** `~/.lcm/global.db`
- **In-memory (testing):** `:memory:`

The directory is created automatically. Each project should use its own
database to keep context isolated.

## Environment Variables

All `LCM_*` environment variables work with both MCP and Python modes.
See [Configuration](configuration.md) for the full list.
