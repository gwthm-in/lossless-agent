# Adapters

Adapters are the main entry point for integrating lossless-agent into
your application. Each adapter wires together the stores, engine, and
tools behind a clean API.

## Choosing an adapter

| Adapter | Best for | Lifecycle hooks | Tool support |
|---------|----------|-----------------|--------------|
| **SimpleAdapter** | Scripts, notebooks, quick experiments | No | Manual via `search()` / `expand()` |
| **GenericAdapter** | Any agent framework | Yes | Yes (OpenAI function-calling format) |
| **HermesAdapter** | Hermes agents | Yes | Yes |

## SimpleAdapter

The simplest option. No abstract base class, no lifecycle — just methods
you call when you need them.

```python
from lossless_agent.adapters import SimpleAdapter

async def summarize(text: str) -> str:
    return await my_llm(f"Summarise:\n{text}")

adapter = SimpleAdapter("memory.db", summarize)

await adapter.ingest("session-1", messages)
await adapter.compact("session-1")
context = await adapter.retrieve("session-1", budget_tokens=4000)
results = await adapter.search("deploy")
expansion = await adapter.expand(summary_id)
adapter.close()
```

### Methods

- `ingest(session_key, messages)` — store a list of message dicts
- `compact(session_key)` — run compaction, return number of summaries created
- `retrieve(session_key, budget_tokens)` — assemble context within budget
- `search(query, session_key=None)` — full-text search
- `expand(summary_id)` — drill into a summary
- `close()` — close the database connection

## GenericAdapter

Implements the `AgentAdapter` abstract base class. Use this when you
want the full lifecycle (turn start/end, session end) and tool support.

```python
from lossless_agent.config import LCMConfig
from lossless_agent.adapters import GenericAdapter

config = LCMConfig(db_path="memory.db")
adapter = GenericAdapter(config, summarize)

# Lifecycle
context = await adapter.on_turn_start("session-1", user_message)
await adapter.on_turn_end("session-1", new_messages)
await adapter.on_session_end("session-1")

# Tools
tools = adapter.get_tools()           # OpenAI function-calling schemas
result = await adapter.handle_tool_call("lcm_grep", {"query": "deploy"})
prompt = adapter.get_system_prompt_block()

# Convenience methods (not in the ABC)
await adapter.store_message("session-1", "user", "Hello", token_count=1)
context = await adapter.get_context("session-1", max_tokens=4000)
await adapter.force_compact("session-1")
stats = await adapter.get_stats("session-1")
```

## HermesAdapter

Same interface as GenericAdapter, designed for the Hermes agent. Usage
is identical to GenericAdapter — just swap the class name:

```python
from lossless_agent.adapters import HermesAdapter

adapter = HermesAdapter(config, summarize)
```

## Writing a custom adapter

To create your own adapter, subclass `AgentAdapter` and implement
the six abstract methods:

```python
from lossless_agent.adapters.base import AgentAdapter

class MyAdapter(AgentAdapter):
    async def on_turn_start(self, session_key, user_message):
        """Return context string or None."""
        ...

    async def on_turn_end(self, session_key, messages):
        """Persist messages and trigger compaction."""
        ...

    async def on_session_end(self, session_key):
        """Final compaction pass."""
        ...

    def get_tools(self):
        """Return tool schemas for your framework."""
        ...

    async def handle_tool_call(self, name, arguments):
        """Dispatch tool calls and return JSON."""
        ...

    def get_system_prompt_block(self):
        """Return recall instructions for the system prompt."""
        ...
```

You can reuse the store and engine classes directly:

```python
from lossless_agent.store import Database, ConversationStore, MessageStore, SummaryStore
from lossless_agent.engine import CompactionEngine, ContextAssembler
from lossless_agent.tools import lcm_grep, lcm_describe, lcm_expand
```

## Factory function

Use `create_adapter` to instantiate an adapter by name:

```python
from lossless_agent.adapters import create_adapter

adapter = create_adapter("generic", config, summarize_fn)
```
