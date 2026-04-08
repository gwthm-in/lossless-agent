# Framework Integration Guide

Set up Lossless Agent with your AI framework in under 2 minutes.

Lossless Agent supports **two integration modes**:

1. **MCP Server** — framework connects to `lossless-agent-mcp` over stdio.
   Best for: Claude Code, Google ADK, Anthropic SDK, Cursor, Windsurf, or
   any MCP-compatible client.
2. **Python Adapter** — import and use directly in your Python code.
   Best for: OpenClaw, custom agents, or when you need programmatic control.

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
`lcm_grep`, `lcm_describe`, and `lcm_expand` tools.

**Verify:** Ask Claude Code _"What LCM tools do you have?"_

See: [`examples/claude_code_setup/`](../examples/claude_code_setup/)

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

**Mode:** MCP Server via `McpToolset`

### Install

```bash
pip install lossless-agent google-adk
```

### Setup

```python
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams

mcp_tools, cleanup = await McpToolset.from_server(
    connection_params=StdioConnectionParams(
        command="lossless-agent-mcp",
        args=["--db-path", "./data/lcm.db"],
    )
)

agent = Agent(
    model="gemini-2.0-flash",
    name="my_agent",
    instruction="You have lossless context management tools.",
    tools=mcp_tools,
)

# ... run your agent ...

await cleanup()  # close MCP connection when done
```

See: [`examples/google_adk_setup.py`](../examples/google_adk_setup.py)

---

## 4. Anthropic SDK (Client-Side MCP)

**Mode:** MCP Server via `anthropic[mcp]`

### Install

```bash
pip install lossless-agent 'anthropic[mcp]'
```

### Setup

```python
from anthropic.types.mcp import MCPServerStdio

mcp_server = MCPServerStdio(
    command="lossless-agent-mcp",
    args=["--db-path", "./data/lcm.db"],
)

client = anthropic.Anthropic()

async with client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    mcp_servers=[mcp_server],
    messages=[{"role": "user", "content": "Search history for deployment"}],
) as stream:
    async for text in stream.text_stream:
        print(text, end="")
```

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
