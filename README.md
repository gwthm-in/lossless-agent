# Lossless Agent

**Lossless Context Management (LCM) for AI agents.**

An agent-agnostic plugin that replaces lossy context compression with a
DAG-based summarization system. Every message is preserved. Nothing is
forgotten. The agent can always drill back into any summary to recover
original detail.

Based on the [LCM paper](https://papers.voltropy.com/LCM) from
[Voltropy](https://x.com/Voltropy) and inspired by
[lossless-claw](https://github.com/Martian-Engineering/lossless-claw) for
OpenClaw.

## The Problem

When a conversation grows beyond the model's context window, most AI agents
truncate or lossily compress older messages:

1. Old tool outputs, intermediate reasoning, file contents, and nuanced
   decisions are **permanently lost** from in-session context.
2. The agent forgets file paths it worked on, decisions it made, errors it
   resolved.
3. The user has to repeat themselves.

Cross-session tools (persistent memory, session search) help between sessions
but do nothing for **in-session context loss**.

## The Solution

Lossless Agent intercepts context management and replaces lossy compression
with a **directed acyclic graph (DAG)** of hierarchical summaries:

```
Raw messages ──► Leaf summaries (depth 0)
                      │
                      ▼
              Condensed summaries (depth 1)
                      │
                      ▼
              Condensed summaries (depth 2)
                      │
                      ▼
                    ...
```

### How it works

1. **Every message is persisted** in a SQLite database, indexed for full-text
   search.
2. **Leaf summarization**: When context pressure builds, the oldest chunk of
   uncompacted messages is summarized by a cheap LLM. The raw messages remain
   in the database — only the summary enters the active context.
3. **Condensed summarization**: When enough leaf summaries accumulate, they are
   condensed into higher-level summaries, forming a DAG. Each condensed node
   links back to its children.
4. **Context assembly**: Each turn, the plugin assembles context from condensed
   summaries + leaf summaries + recent raw messages, staying within the model's
   token budget.
5. **Recall tools**: The agent gets tools to search and drill into compacted
   history:
   - `lcm_grep` — full-text and regex search across all messages and summaries
   - `lcm_describe` — inspect a summary node's metadata, content, and lineage
   - `lcm_expand` — traverse the DAG to recover original messages from any
     summary

**Nothing is lost. The agent never truly forgets.**

## Why This Matters

| Without LCM | With LCM |
|---|---|
| Context compressed into a single lossy summary | Hierarchical DAG preserves detail at every level |
| Old tool results permanently discarded | Raw messages always recoverable via `lcm_expand` |
| Agent can't recall file paths from 50 turns ago | `lcm_grep` finds anything from any point in the conversation |
| User must repeat themselves after long sessions | Agent drills into summaries to recover what it needs |
| Compression is all-or-nothing | Incremental compaction — only the oldest chunks are summarized |

## Architecture

Lossless Agent is **agent-agnostic** with a clean adapter interface. The core
engine knows nothing about any specific agent framework. Adapters bridge to
specific agents:

```
┌─────────────────────────────────┐
│         Agent Framework         │
│   (Hermes, OpenClaw, custom)    │
└──────────────┬──────────────────┘
               │ Adapter
┌──────────────▼──────────────────┐
│        Lossless Agent Core      │
│  ┌───────────┐ ┌──────────────┐ │
│  │ Compaction │ │   Context    │ │
│  │  Engine    │ │  Assembler   │ │
│  └─────┬─────┘ └──────┬───────┘ │
│        │              │         │
│  ┌─────▼──────────────▼───────┐ │
│  │      SQLite Store          │ │
│  │  Messages │ Summaries │DAG │ │
│  └────────────────────────────┘ │
└─────────────────────────────────┘
```

### Lifecycle hooks (adapter interface)

- **`on_messages(messages)`** — Persist new messages to the store
- **`needs_compaction(context_limit)`** — Check if context pressure is high
- **`compact()`** — Run incremental leaf + condensed compaction
- **`assemble(context_limit)`** — Build context: summaries + raw tail
- **`get_tools()`** — Return recall tool definitions for the agent
- **`on_session_end()`** — Final compaction pass

### Summarization

Compaction uses a **separate, cheaper model** for summarization (e.g.
claude-haiku, gpt-4o-mini) to keep costs low. The summarization model is
configurable independently from the agent's main model.

### Design constraint: prompt caching

For agents that use prompt caching (e.g. Anthropic ~75% cost reduction), all
dynamic context injection goes through **message injection** or **tool
results**, never mid-session system prompt modification.

### Storage

SQLite with WAL mode, FTS5 full-text search:

- **Messages**: role, content, token count, timestamps, tool metadata
- **Summaries**: DAG nodes with depth, kind (leaf/condensed), content, token
  counts, temporal range
- **DAG edges**: parent-child relationships between summary nodes

## Language

**Python** — because:

- Most AI agent frameworks are Python (Hermes, LangChain, CrewAI, AutoGen)
- `sqlite3` is in the standard library
- The core algorithm is IO-bound (LLM calls + SQLite), not CPU-bound
- Lowest friction for the target ecosystem

## Development

This project follows **strict TDD** — tests are written before implementation.

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=lossless_agent --cov-report=term-missing
```

## Commit Convention

All commits follow [Chris Beams' rules](https://cbea.ms/git-commit/) with
[Conventional Commits](https://www.conventionalcommits.org/) prefixes.

## Features

- **SQLite store** with FTS5 full-text search (messages, summaries, DAG edges)
- **Leaf compaction engine** — chunk selection + LLM summarization
- **Condensed compaction** — hierarchical DAG construction at arbitrary depth
- **Context assembler** — budget-aware summary + raw message assembly
- **Recall tools**: `lcm_grep`, `lcm_describe`, `lcm_expand`
- **Adapter system**: Hermes, OpenClaw, Generic, and SimpleAdapter
- **Incremental compaction** — per-turn, automatic
- **Sub-agent expansion** (`lcm_expand_query`) for deep retrieval
- **Large file interception** and separate storage
- **Configuration system** — env vars (`LCM_*`) + programmatic config
- **Compaction-aware prompts** — dynamic uncertainty checklist when heavily compacted
- **Circuit breaker** — automatic backoff on repeated summarization failures
- **Heartbeat pruning** — removes noisy heartbeat messages
- **Session pattern matching** — ignore or mark sessions as stateless
- **MCP server** — Model Context Protocol server for tool-based integrations
- **py.typed** — PEP 561 compatible, full type annotations

## Roadmap

- [x] SQLite store with FTS5 (messages, summaries, DAG edges)
- [x] Leaf compaction engine (chunk selection + LLM summarization)
- [x] Condensed compaction (hierarchical DAG construction)
- [x] Context assembler (budget-aware summary + raw message assembly)
- [x] Recall tools: `lcm_grep`, `lcm_describe`, `lcm_expand`
- [x] Hermes Memory Provider adapter
- [x] Incremental compaction (per-turn, background)
- [x] Sub-agent expansion (`lcm_expand_query`)
- [x] Large file interception and separate storage
- [x] Configuration system (env vars + plugin config)

## Contributing

This project uses a Contributor License Agreement (CLA). By contributing, you
agree to the terms in [CLA.md](CLA.md). This allows us to offer commercial
licenses while keeping the project open source.

## License

Copyright (c) 2026 Gowtham Sai

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

See [LICENSE](LICENSE) for the full text.

### Commercial Licensing

For commercial use that is incompatible with the AGPL v3 (e.g. offering as a
managed service, embedding in proprietary software), contact us for a
commercial license.
