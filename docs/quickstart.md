# Quick Start

## Installation

```bash
pip install lossless-agent
```

## SimpleAdapter — five lines to lossless memory

SimpleAdapter is the fastest way to get started. No lifecycle hooks,
no abstract methods — just ingest, compact, and retrieve.

```python
from lossless_agent.adapters import SimpleAdapter

async def my_summarize(text: str) -> str:
    """Call your LLM to summarise text."""
    return await call_my_llm(f"Summarise:\n{text}")

adapter = SimpleAdapter("memory.db", my_summarize)

# Store messages
await adapter.ingest("session-1", [
    {"role": "user", "content": "How do I deploy to prod?"},
    {"role": "assistant", "content": "Run `make deploy`. It uses..."},
])

# Compact old messages into summaries
created = await adapter.compact("session-1")
print(f"Created {created} new summaries")

# Retrieve context within a token budget
context = await adapter.retrieve("session-1", budget_tokens=4000)
print(context)

# Search across all history
results = await adapter.search("deploy")
for r in results:
    print(r["content_snippet"])

# Clean up
adapter.close()
```

## GenericAdapter — full lifecycle with tool support

GenericAdapter implements the AgentAdapter lifecycle (on_turn_start,
on_turn_end, on_session_end) and exposes recall tools for the LLM.

```python
from lossless_agent.config import LCMConfig
from lossless_agent.adapters import GenericAdapter

config = LCMConfig(db_path="memory.db", max_context_tokens=128_000)

adapter = GenericAdapter(config, my_summarize)

# Before the LLM call — get context to inject
context = await adapter.on_turn_start("session-1", user_message)
# Inject `context` into the system prompt or as a preamble

# After the LLM responds — persist and compact
await adapter.on_turn_end("session-1", [
    {"role": "user", "content": user_message},
    {"role": "assistant", "content": assistant_reply},
])

# Get tool schemas to pass to the LLM
tools = adapter.get_tools()

# When the LLM calls a recall tool
result = await adapter.handle_tool_call("lcm_grep", {"query": "deploy"})

# Get the system prompt block for recall instructions
prompt_block = adapter.get_system_prompt_block()
```

## HermesAdapter — for Hermes agents

HermesAdapter has the same interface as GenericAdapter but is tailored
for the Hermes agent framework. Usage is identical:

```python
from lossless_agent.config import LCMConfig
from lossless_agent.adapters import HermesAdapter

config = LCMConfig(db_path="memory.db")
adapter = HermesAdapter(config, my_summarize)

context = await adapter.on_turn_start("session-1", user_message)
await adapter.on_turn_end("session-1", messages)
await adapter.on_session_end("session-1")
```

## Environment variables

You can configure lossless-agent entirely through environment variables
instead of passing a config object. See [Configuration](configuration.md)
for the full list.

```bash
export LCM_DATABASE_PATH=~/.my-agent/memory.db
export LCM_MAX_CONTEXT_TOKENS=64000
```

```python
from lossless_agent.config import LCMConfig

config = LCMConfig.from_env()
```
