# Architecture

lossless-agent is organised into four layers: **stores**, **engine**,
**tools**, and **adapters**. Each layer depends only on the ones below it.

```
┌─────────────────────────────────────────────┐
│               Adapters                      │
│  SimpleAdapter · GenericAdapter · Hermes    │
├─────────────────────────────────────────────┤
│           Tools (recall)                    │
│  lcm_grep · lcm_describe · lcm_expand      │
├─────────────────────────────────────────────┤
│              Engine                         │
│  CompactionEngine · ContextAssembler        │
├─────────────────────────────────────────────┤
│              Stores                         │
│  MessageStore · SummaryStore · ConvStore    │
│              Database (SQLite)              │
└─────────────────────────────────────────────┘
```

## The summary DAG

Messages are never deleted. Instead, old messages are compacted into
**leaf summaries**, and groups of leaf summaries are merged into
**condensed summaries** at increasing depth. The result is a directed
acyclic graph (DAG):

```
depth 2    ┌──────────────┐
           │  condensed   │
           └──┬───────┬───┘
              │       │
depth 1    ┌──▼──┐ ┌──▼──┐
           │leaf │ │leaf │   ← condensed from depth-0
           └──┬──┘ └──┬──┘
              │       │
depth 0    ┌──▼──┐ ┌──▼──┐ ┌─────┐ ┌─────┐
           │leaf │ │leaf │ │leaf │ │leaf │   ← direct message summaries
           └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
              │       │       │       │
           ┌──▼──────▼───────▼───────▼──┐
           │        Messages            │
           │  (always preserved)        │
           └────────────────────────────┘
```

Every summary records which messages (leaf) or child summaries
(condensed) it was derived from. The `lcm_expand` tool lets the
agent drill down from any summary to its sources.

## Data flow

Here is what happens during a typical agent turn:

```
User message
    │
    ▼
on_turn_start(session_key, message)
    │
    ├── ConversationStore.get_or_create(session_key)
    ├── ContextAssembler.assemble(conv_id)
    │       ├── tail = MessageStore.tail(conv_id, N)
    │       ├── summaries = SummaryStore.get_by_conversation(conv_id)
    │       └── budget-aware selection
    └── return formatted context string
    │
    ▼
LLM generates response (with recall tools available)
    │
    ▼
on_turn_end(session_key, messages)
    │
    ├── MessageStore.append(each message)
    └── CompactionEngine.run_incremental(conv_id)
            ├── needs_compaction? (threshold check)
            ├── compact_leaf  → SummaryStore.create_leaf
            └── compact_condensed → SummaryStore.create_condensed
```

## Store layer

The store layer wraps a single SQLite database with FTS5 full-text
search. Three stores share the same connection:

- **MessageStore** — append-only log of messages with sequence numbers
  and token counts.
- **SummaryStore** — leaf and condensed summaries linked to their
  sources via junction tables (`summary_messages`, `summary_parents`).
- **ConversationStore** — maps external session keys to internal
  integer conversation IDs.

All stores implement abstract base classes (`AbstractMessageStore`,
`AbstractSummaryStore`) so you can swap in a different backend if needed.

## Engine layer

- **CompactionEngine** — selects chunks of old messages, calls a
  summarize function, and writes leaf summaries. Also merges orphan
  summaries at each depth into condensed nodes.
- **ContextAssembler** — builds a token-budget-aware context by
  picking the highest-depth summaries first, then filling with recent
  messages.

## Tools layer

Three recall tools let the agent navigate its own memory:

| Tool | Purpose |
|------|---------|
| `lcm_grep(query)` | FTS5 search across messages and summaries |
| `lcm_describe(summary_id)` | Metadata for a specific summary node |
| `lcm_expand(summary_id)` | Drill into a summary to see its children |

## Adapters layer

Adapters wire everything together behind a turn-oriented API. See
[Adapters](adapters.md) for details on each one.
